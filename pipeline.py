"""
pipeline.py — the one place the stages are joined end to end.

WHY THIS DESIGN
---------------
Each module owns one boundary: contracts, planning, execution, salience,
synthesis. This module owns the order they run in and nothing else. The
alternative — letting run_demo.py and the notebook each wire the stages
themselves — means two copies of the orchestration that drift apart. One
composition function, typed at both ends, is what both entry points call.

The return value, PipelineRun, carries every intermediate artifact as a typed
field. Nothing is discarded between stages, so the notebook can show the
plans, the SQL, the result sets, the salience rankings, the audit trail and
the gate outcome without re-running anything.
"""

from __future__ import annotations

import time

import duckdb
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from pydantic_ai.models import Model

from decomposition import LensDecomposition, decompose
from governed_duckdb_tool import (
    AuditRecord,
    ExecutionResult,
    GateOutcome,
    GovernedDuckDBTool,
    build_fact_base,
)
from langchain_context_chain import build_planning_chain, default_chat_model
from narrative_agent import ExecutiveDeliverable, NarrativeBrief, synthesize_narrative
from salience import SalienceRanking, compute_salience
from semantic_contracts import GovernedPlan
from synthetic_data import TierSummary, build_database


class RunTelemetry(BaseModel):
    """Observed behaviour of one run: what was called, and how long each stage took.

    These are counts and timings the run actually recorded. They are not a
    comparison against any alternative design, so they support no savings claim.
    """

    llm_calls: int = Field(
        description="Model requests made: one planner call plus one narrative request per validated model output."
    )
    governed_queries: int = Field(description="Queries executed through the governed tool (audit records written).")
    salience_rankings: int = Field(description="Deterministic salience rankings computed.")
    decompositions: int = Field(description="Deterministic revenue decompositions computed and reconciled.")
    gate_evaluations: int = Field(description="Zero-token-math gate evaluations, one per validated narrative output.")
    stage_ms: dict[str, float] = Field(description="Wall-clock milliseconds per stage: plan, execute, salience, decomposition, narrative.")

    @property
    def deterministic_operations(self) -> int:
        return self.governed_queries + self.salience_rankings + self.decompositions + self.gate_evaluations

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


class PipelineRun(BaseModel):
    """Every artifact of one governed run, in the order it was produced."""

    query: str = Field(description="The business question asked.")
    dataset: TierSummary = Field(description="Shape of the synthetic dataset the run executed against.")
    governed_plan: GovernedPlan = Field(description="Resolved contracts and the plans generated against them.")
    department_result: ExecutionResult = Field(description="Executed department-lens result set.")
    enterprise_result: ExecutionResult = Field(description="Executed enterprise-lens result set.")
    department_salience: SalienceRanking = Field(description="Salience ranking for the department lens.")
    enterprise_salience: SalienceRanking = Field(description="Salience ranking for the enterprise lens.")
    department_decomposition: LensDecomposition = Field(description="Reconciled revenue movement, department lens.")
    enterprise_decomposition: LensDecomposition = Field(description="Reconciled revenue movement, enterprise lens.")
    audit_trail: list[AuditRecord] = Field(description="One record per tool call.")
    gate: GateOutcome = Field(description="Final zero-token-math gate outcome.")
    deliverable: ExecutiveDeliverable | None = Field(description="Verified narrative, or None if suppressed.")
    narrative_suppressed: bool = Field(description="True when the pipeline failed closed on the narrative.")
    narrative_attempts: int = Field(description="Model outputs validated before success or suppression.")
    telemetry: RunTelemetry = Field(description="Observed call counts and per-stage timings for this run.")

    def result_for(self, lens: str) -> ExecutionResult:
        return self.department_result if lens == "department" else self.enterprise_result

    def salience_for(self, lens: str) -> SalienceRanking:
        return self.department_salience if lens == "department" else self.enterprise_salience


def run_dual_lens_pipeline(
    query: str,
    *,
    chat_model: BaseChatModel | None = None,
    narrative_model: Model | str | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    dataset_summary: TierSummary | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> PipelineRun:
    """Question in, PipelineRun out. Raises ContextResolutionError or GovernanceViolation rather than guessing.

    Stage order: plan (LCEL) -> execute both lenses (governed tool) -> salience
    (deterministic) -> synthesise (PydanticAI, gated). Models are injectable so
    the whole path runs under test with no network.
    """
    if conn is None:
        conn, dataset_summary = build_database(seed=seed, verbose=verbose)
    if dataset_summary is None:
        raise ValueError("dataset_summary is required when an existing connection is supplied")

    stage_ms: dict[str, float] = {}
    mark = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal mark
        now = time.perf_counter()
        stage_ms[stage] = (now - mark) * 1000
        mark = now

    # 1. Plan both lenses in one structured call.
    chain = build_planning_chain(chat_model or default_chat_model())
    governed_plan: GovernedPlan = chain.invoke(query)
    lap("plan")

    # 2. Validate and execute each plan through the governed tool.
    tool = GovernedDuckDBTool(conn, governed_plan.context)
    department_result = tool.tool.invoke({"plan": governed_plan.plan.department.model_dump()})
    enterprise_result = tool.tool.invoke({"plan": governed_plan.plan.enterprise.model_dump()})

    lap("execute")

    # 3. Salience per lens against its own baseline. No model involvement.
    department_salience = compute_salience(
        conn, governed_plan.context.department_contract, governed_plan.plan.department.sql
    )
    enterprise_salience = compute_salience(
        conn, governed_plan.context.enterprise_contract, governed_plan.plan.enterprise.sql
    )

    lap("salience")

    # 3b. Revenue movement per lens. Deterministic; raises ReconciliationError (a
    # GovernanceViolation) before any narration if components do not add up.
    department_decomposition = decompose(conn, "department")
    enterprise_decomposition = decompose(conn, "enterprise")

    lap("decomposition")

    # 4. Synthesise, with the zero-token-math gate as the output validator.
    # The fact base is exactly what the brief shows the model, per lens: row
    # counts, salience statistics, and the governed thresholds of each contract.
    fact_base = build_fact_base(
        governed_plan.context,
        department_result,
        enterprise_result,
        department_salience,
        enterprise_salience,
        department_decomposition,
        enterprise_decomposition,
    )
    brief = NarrativeBrief(
        context=governed_plan.context,
        department_result=department_result,
        enterprise_result=enterprise_result,
        department_salience=department_salience,
        enterprise_salience=enterprise_salience,
        department_decomposition=department_decomposition,
        enterprise_decomposition=enterprise_decomposition,
    )
    narrative = synthesize_narrative(brief, fact_base, model=narrative_model)
    lap("narrative")

    return PipelineRun(
        query=query,
        dataset=dataset_summary,
        governed_plan=governed_plan,
        department_result=department_result,
        enterprise_result=enterprise_result,
        department_salience=department_salience,
        enterprise_salience=enterprise_salience,
        department_decomposition=department_decomposition,
        enterprise_decomposition=enterprise_decomposition,
        audit_trail=tool.audit_trail,
        gate=narrative.gate,
        deliverable=narrative.deliverable,
        narrative_suppressed=narrative.suppressed,
        narrative_attempts=narrative.attempts,
        telemetry=RunTelemetry(
            llm_calls=1 + narrative.attempts,
            governed_queries=len(tool.audit_trail),
            salience_rankings=2,
            decompositions=2,
            gate_evaluations=narrative.attempts,
            stage_ms=stage_ms,
        ),
    )
