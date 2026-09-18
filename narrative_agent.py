"""
narrative_agent.py — PydanticAI synthesis of the executive deliverable.

WHY PYDANTICAI HERE, WHEN LCEL RUNS EVERYTHING ELSE
---------------------------------------------------
The planning stage is a fixed sequence: retrieve, template, bind, validate.
LCEL is the right tool for that, because a reader can trace the pipe in one
pass and every step is inspectable.

This stage is different. It is constrained generation: the model must turn
two verified result sets and two salience rankings into prose that (1) fits
a typed output model with separate fields per lens, (2) cites figures only in
a typed list, and (3) survives the zero-token-math gate. When the output
violates any of those, the correct response is not to fail the request but to
send the validation error back to the model and let it try again. That loop —
typed output, validation, automatic retry with the error as feedback — is
PydanticAI's native strength, and it is the genuine reason to keep PydanticAI
for this one stage rather than forcing it through LCEL or bolting a retry
loop onto a chain by hand.

The gate itself lives in governed_duckdb_tool.py. Here it is wired in as an
output validator: a deliverable whose cited_metrics do not trace to executed
results raises ModelRetry, so the model sees exactly which figures failed.
If retries are exhausted the pipeline fails closed: the narrative is
suppressed and the deterministic result sets are surfaced instead.

Plain Pydantic models are what cross this boundary in both directions. The
orchestration framework is an implementation choice; the contract layer is not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Keep PydanticAI's start-up banner out of notebook and demo output. Set before the import.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic import BaseModel, Field  # noqa: E402
from pydantic_ai import Agent, ModelRetry, RunContext  # noqa: E402
from pydantic_ai.exceptions import UnexpectedModelBehavior  # noqa: E402
from pydantic_ai.models import Model  # noqa: E402

from governed_duckdb_tool import (  # noqa: E402
    CitedMetric,
    ExecutionResult,
    FactBase,
    GateOutcome,
    verify_cited_metrics,
)
from salience import SalienceRanking  # noqa: E402
from semantic_contracts import ResolvedContext  # noqa: E402

# Default Gemini model for synthesis, addressed through PydanticAI's provider prefix.
DEFAULT_NARRATIVE_MODEL = "google:gemini-3.6-flash"
# Retries are how a gate failure becomes feedback rather than an error.
DEFAULT_OUTPUT_RETRIES = 2


# ---------------------------------------------------------------------------
# Output model: the typed surface the gate checks
# ---------------------------------------------------------------------------


class LensSummary(BaseModel):
    """The executive read of one lens."""

    headline: str = Field(description="One sentence naming the segment and its single most distinctive trait.")
    narrative: str = Field(
        description=(
            "Two to four sentences describing who this segment is, using the top salience attributes. "
            "Write figures with thousands separators and one decimal place. Do not use the dollar sign."
        )
    )
    top_attributes: list[str] = Field(
        description="The two or three attribute names (or attribute = level) that most distinguish this segment."
    )


class ExecutiveDeliverable(BaseModel):
    """The final deliverable. Narrative lives in prose fields; every cited figure lives in cited_metrics."""

    department_summary: LensSummary = Field(description="Summary of the license-grain department lens.")
    enterprise_summary: LensSummary = Field(description="Summary of the organization-grain enterprise lens.")
    reconciliation_memo: str = Field(
        description=(
            "Three to five sentences explaining why the two lenses surface different populations, "
            "referring to the grain, the metric, and the threshold of each contract. "
            "State that neither is wrong; they answer different questions."
        )
    )
    cited_metrics: list[CitedMetric] = Field(
        description=(
            "Every numeric figure mentioned in the summaries or the memo, as typed values copied exactly "
            "from the verified results. This list is checked by the zero-token-math gate."
        )
    )

    def prose(self) -> str:
        """All narrative text in one string, so the gate can confirm every formatted figure in it is cited."""
        return "\n".join(
            [
                self.department_summary.headline,
                self.department_summary.narrative,
                self.enterprise_summary.headline,
                self.enterprise_summary.narrative,
                self.reconciliation_memo,
            ]
        )


class NarrativeOutcome(BaseModel):
    """What the pipeline gets back: a verified deliverable, or a fail-closed record of why there is none."""

    deliverable: ExecutiveDeliverable | None = Field(description="The verified deliverable; None when suppressed.")
    gate: GateOutcome = Field(description="The final zero-token-math gate outcome.")
    suppressed: bool = Field(description="True when the narrative failed the gate after all retries.")
    attempts: int = Field(description="How many model outputs were validated.")


# ---------------------------------------------------------------------------
# Input model: what the agent is told
# ---------------------------------------------------------------------------


class NarrativeBrief(BaseModel):
    """Everything the synthesis model may draw on. Nothing outside this brief exists, as far as the model knows."""

    context: ResolvedContext = Field(description="Both contracts and segment definitions.")
    department_result: ExecutionResult = Field(description="Executed department-lens result set.")
    enterprise_result: ExecutionResult = Field(description="Executed enterprise-lens result set.")
    department_salience: SalienceRanking = Field(description="Salience ranking for the department lens.")
    enterprise_salience: SalienceRanking = Field(description="Salience ranking for the enterprise lens.")

    def render(self, top_n: int = 6) -> str:
        """Plain-text brief. Figures are pre-formatted so the model copies rather than computes."""
        sections = [f"Business question: {self.context.query}", ""]
        for lens in ("department", "enterprise"):
            contract = self.context.contract_for(lens)
            result = self.department_result if lens == "department" else self.enterprise_result
            ranking = self.department_salience if lens == "department" else self.enterprise_salience
            sections.append(f"=== {lens.upper()} LENS — {contract.contract_id} v{contract.version} ===")
            sections.append(f"Grain: {contract.entity_grain}. Metric: {contract.metric_name} ({contract.metric_expression}).")
            sections.append(f"Threshold: {contract.thresholds[0].as_sql()}. {contract.description}")
            sections.append(
                f"Segment size: {ranking.segment_size} of {ranking.baseline_size} {contract.entity_grain}s "
                f"(result rows: {result.row_count})."
            )
            sections.append("Top differentiating attributes (segment vs baseline, standardised effect):")
            for score in ranking.top(top_n):
                unit = "%" if score.method == "percentage_point_delta" else ""
                sections.append(
                    f"  - {score.display_name}: {score.segment_value:,.1f}{unit} vs {score.baseline_value:,.1f}{unit} "
                    f"({score.direction}; effect {score.effect_size:.2f}; {score.method})"
                )
            sections.append("")
        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class NarrativeDeps:
    """Dependencies injected into the run: the fact base the gate checks against, and a place to keep the last outcome."""

    fact_base: FactBase
    attempts: int = 0
    last_gate: GateOutcome | None = None
    gate_log: list[GateOutcome] = field(default_factory=list)


_INSTRUCTIONS = (
    "You are writing an executive comparison for a product and revenue leadership audience. "
    "You will receive a brief containing two verified result sets and two salience rankings. "
    "You write; you never compute. Every number you mention must be copied from the brief "
    "exactly as written, and every number you mention must also appear in cited_metrics under the "
    "lens whose section of the brief it came from. Do not derive new figures (no differences, "
    "ratios, or totals of your own). "
    "Write measured figures (revenue, means, percentages) with thousands separators and one decimal "
    "place; write counts of licenses or organizations as whole numbers. Never use the dollar sign or "
    "underscores in prose; write attribute names as plain words. "
    "Explain the divergence between lenses in terms of grain, metric, and threshold."
)


def build_narrative_agent(model: Model | str | None = None, retries: int = DEFAULT_OUTPUT_RETRIES) -> Agent[NarrativeDeps, ExecutiveDeliverable]:
    """Construct the agent with the zero-token-math gate wired in as an output validator."""
    agent: Agent[NarrativeDeps, ExecutiveDeliverable] = Agent(
        model or DEFAULT_NARRATIVE_MODEL,
        deps_type=NarrativeDeps,
        output_type=ExecutiveDeliverable,
        instructions=_INSTRUCTIONS,
        retries=retries,
    )

    @agent.output_validator
    def zero_token_math_gate(ctx: RunContext[NarrativeDeps], output: ExecutiveDeliverable) -> ExecutiveDeliverable:
        """ZERO-TOKEN-MATH GATE as a validator: a failing deliverable is sent back to the model with the failing figures named."""
        ctx.deps.attempts += 1
        outcome = verify_cited_metrics(output.cited_metrics, ctx.deps.fact_base, prose=output.prose())
        ctx.deps.last_gate = outcome
        ctx.deps.gate_log.append(outcome)
        if not outcome.passed:
            raise ModelRetry(
                outcome.detail
                + " Every figure must be copied from the brief for the lens it belongs to, and every "
                "formatted figure in the prose must also appear in cited_metrics."
            )
        return output

    return agent


def synthesize_narrative(
    brief: NarrativeBrief,
    fact_base: FactBase,
    model: Model | str | None = None,
    retries: int = DEFAULT_OUTPUT_RETRIES,
) -> NarrativeOutcome:
    """Run the synthesis agent and fail closed if the gate cannot be satisfied.

    `model` accepts a PydanticAI Model instance so tests can pass TestModel or
    FunctionModel; production passes nothing and uses Gemini.
    """
    if model is None and not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set; cannot run the narrative agent.")

    agent = build_narrative_agent(model=model, retries=retries)
    deps = NarrativeDeps(fact_base=fact_base)
    try:
        run = agent.run_sync(brief.render(), deps=deps)
    except UnexpectedModelBehavior:
        # Retries exhausted. Fail closed: no narrative, but a full record of why.
        gate = deps.last_gate or GateOutcome(
            passed=False, checked=0, unverifiable=[], detail="Model produced no validatable output."
        )
        return NarrativeOutcome(deliverable=None, gate=gate, suppressed=True, attempts=deps.attempts)

    assert deps.last_gate is not None  # the validator always runs before a successful return
    return NarrativeOutcome(deliverable=run.output, gate=deps.last_gate, suppressed=False, attempts=deps.attempts)
