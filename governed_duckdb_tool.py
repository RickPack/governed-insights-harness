"""
governed_duckdb_tool.py — deterministic execution behind a typed tool boundary.

WHY THIS DESIGN
---------------
The language model plans queries and writes prose. It never computes. Every
number in the final deliverable is produced here, by DuckDB, from SQL that
passed a pre-execution validator. The obvious alternative — let the model call
a generic "run SQL" tool — puts the whole governance burden on the prompt.
This module puts it on code that can be unit-tested:

  * Grain discipline. Any ratio must divide by COUNT(DISTINCT entity_id).
    A raw-row-count denominator is the classic way an organization metric
    gets silently computed at license grain; the validator rejects it
    before execution.
  * Governed thresholds. Every numeric comparison in the SQL must match a
    ThresholdDefinition on the contract the plan cites. A cutoff the model
    invented is rejected even if it looks reasonable.
  * Cross-grain joins. A query that reaches across to the other grain's table
    without an intermediate aggregation is rejected; it would double-count.

THE ZERO-TOKEN-MATH GATE
------------------------
The final narrative cites figures in a typed list field (cited_metrics), not
in prose. This module checks every one of those values against a FactBase:
the exact set of figures the model was shown for each lens, all of which came
from executed queries, salience arithmetic, or versioned contracts. A figure
that cannot be traced fails the gate; so does a real figure cited under the
wrong lens. Because the cited figures are a typed field rather than prose
parsed with a regular expression, the primary check is exact: a year, a rank,
or a version number in the prose can never be mistaken for a metric. A second,
narrower pass confirms that every executive-formatted figure in the prose is
present in the typed list, so the model cannot route a number around the gate
by leaving it out of cited_metrics.

Every tool call writes an AuditRecord: lens, contract version, SQL, row
count, validator outcomes, timestamp. The audit trail is a typed list, so it
can be rendered, stored, or diffed without parsing log lines.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Union

import duckdb
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from semantic_contracts import CONTRACT_REGISTRY, Lens, MetricContract, PlannedQuery, ResolvedContext

if TYPE_CHECKING:  # salience does not import this module; the name is only needed for hints
    from salience import SalienceRanking

# A DuckDB cell as it crosses the tool boundary. Kept to scalars so ResultTable
# is JSON-serialisable and comparable without pandas.
Scalar = Union[str, int, float, bool, None]

# Every figure in the brief is presented to one decimal place, so a cited value
# must equal a fact to that precision. There is deliberately no relative
# tolerance: one part in a thousand of a five-figure revenue is a spread of
# tens, which is room enough for a fabricated number to pass.
_GATE_ABS_TOLERANCE = 0.051


class GovernanceViolation(Exception):
    """Raised when a plan fails the pre-execution validator. Nothing is executed."""


# ---------------------------------------------------------------------------
# Typed inputs and outputs
# ---------------------------------------------------------------------------


class GateCheck(BaseModel):
    """Outcome of one validator rule."""

    name: str = Field(description="Rule identifier, e.g. 'grain_discipline'.")
    passed: bool = Field(description="True when the rule was satisfied.")
    detail: str = Field(description="Human-readable explanation of the outcome.")


class ResultTable(BaseModel):
    """A materialised query result. Columns plus rows of scalars, in query order."""

    columns: list[str] = Field(description="Column names in select order.")
    rows: list[list[Scalar]] = Field(description="Row values, aligned with columns.")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


class ExecutionResult(BaseModel):
    """What the tool returns: the result set tagged with the lens and contract that produced it."""

    lens: Lens = Field(description="Lens the query served.")
    contract_id: str = Field(description="Contract the plan cited.")
    contract_version: str = Field(description="Contract version the plan cited.")
    sql: str = Field(description="The exact SQL executed.")
    table: ResultTable = Field(description="The materialised result set.")
    row_count: int = Field(description="Number of rows returned. One row per entity at the contract's grain.")


class AuditRecord(BaseModel):
    """One record per tool call, written whether the call succeeded or was refused."""

    model_config = ConfigDict(frozen=True)

    lens: Lens = Field(description="Lens the call served.")
    contract_id: str = Field(description="Contract cited by the plan.")
    contract_version: str = Field(description="Contract version cited by the plan.")
    sql: str = Field(description="SQL submitted for validation.")
    row_count: int = Field(description="Rows returned; 0 when the call was refused.")
    executed: bool = Field(description="False when the validator refused the plan.")
    checks: list[GateCheck] = Field(description="Every validator rule and its outcome.")
    timestamp: datetime = Field(description="UTC time the record was written.")


class CitedMetric(BaseModel):
    """One figure the narrative cites. This typed field is the surface the zero-token-math gate checks."""

    label: str = Field(description="What the figure is, e.g. 'department lens mean tenure (years)'.")
    lens: Lens = Field(description="Which lens the figure belongs to.")
    value: float = Field(description="The numeric value exactly as it appears in the verified results.")


class GateOutcome(BaseModel):
    """Result of the zero-token-math gate over a deliverable's cited metrics."""

    passed: bool = Field(description="True when every cited metric traced to an executed result.")
    checked: int = Field(description="Number of cited metrics examined.")
    unverifiable: list[CitedMetric] = Field(description="Cited metrics with no matching computed value for their lens.")
    uncited_prose: list[float] = Field(
        default_factory=list, description="Formatted figures found in the prose but absent from cited_metrics."
    )
    detail: str = Field(description="Explanation suitable for a reviewer or an error message.")


# ---------------------------------------------------------------------------
# Pre-execution validator
# ---------------------------------------------------------------------------

_KNOWN_TABLES = {contract.source_table: contract.entity_grain for contract in CONTRACT_REGISTRY}
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_COMPARISON = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)\b")
# A '/' followed (optionally through a CAST or a scalar subquery) by COUNT( that is not COUNT(DISTINCT.
_RAW_COUNT_DENOMINATOR = re.compile(
    r"/\s*(?:\(\s*select\s+)?(?:cast\s*\(\s*)?count\s*\(\s*(?!distinct\b)", re.IGNORECASE
)
_SUBQUERY = re.compile(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", re.DOTALL)


def _check_grain_discipline(sql: str) -> GateCheck:
    """Every ratio divides by COUNT(DISTINCT ...). Raw row counts are how grains get mixed."""
    if _RAW_COUNT_DENOMINATOR.search(sql):
        return GateCheck(
            name="grain_discipline",
            passed=False,
            detail="A ratio divides by a raw row count. Denominators must be COUNT(DISTINCT entity_id).",
        )
    return GateCheck(name="grain_discipline", passed=True, detail="No raw-row-count denominators found.")


def _check_governed_thresholds(sql: str, contract: MetricContract) -> GateCheck:
    """Every numeric comparison must match a ThresholdDefinition on the cited contract, verbatim."""
    governed = {(t.column, t.operator, float(t.value)) for t in contract.thresholds}
    found = [(col, op, float(val)) for col, op, val in _COMPARISON.findall(sql)]
    if not found:
        return GateCheck(
            name="governed_thresholds",
            passed=False,
            detail="The plan applies no threshold; a segment query must filter by a governed cutoff.",
        )
    ungoverned = [f"{col} {op} {val:g}" for col, op, val in found if (col, op, val) not in governed]
    if ungoverned:
        return GateCheck(
            name="governed_thresholds",
            passed=False,
            detail=f"Ungoverned threshold(s) in SQL: {', '.join(ungoverned)}. "
            f"Contract {contract.contract_id} v{contract.version} allows only: "
            + ", ".join(t.as_sql() for t in contract.thresholds),
        )
    return GateCheck(name="governed_thresholds", passed=True, detail="All numeric cutoffs trace to the contract.")


def _check_cross_grain_join(sql: str, contract: MetricContract) -> GateCheck:
    """A query touching the other grain's table must aggregate it in a subquery before joining."""
    referenced = {name.lower() for name in _TABLE_REF.findall(sql)}
    foreign = {t for t in referenced if t in _KNOWN_TABLES and _KNOWN_TABLES[t] != contract.entity_grain}
    if not foreign:
        return GateCheck(name="cross_grain_join", passed=True, detail="Query reads a single grain.")
    # The foreign table is allowed only inside a subquery that GROUP BYs it back to this grain.
    for body in _SUBQUERY.findall(sql):
        lowered = body.lower()
        if any(t in lowered for t in foreign) and "group by" in lowered:
            return GateCheck(
                name="cross_grain_join",
                passed=True,
                detail=f"Foreign-grain table(s) {sorted(foreign)} are aggregated before the join.",
            )
    return GateCheck(
        name="cross_grain_join",
        passed=False,
        detail=f"Query joins foreign-grain table(s) {sorted(foreign)} without intermediate aggregation; "
        "this would double-count entities.",
    )


def _check_source_table(sql: str, contract: MetricContract) -> GateCheck:
    """The plan must at least read its own contract's table."""
    referenced = {name.lower() for name in _TABLE_REF.findall(sql)}
    if contract.source_table.lower() not in referenced:
        return GateCheck(
            name="source_table",
            passed=False,
            detail=f"Plan does not read the contract's source table {contract.source_table!r}.",
        )
    return GateCheck(name="source_table", passed=True, detail=f"Reads {contract.source_table}.")


def validate_plan(plan: PlannedQuery, contract: MetricContract) -> list[GateCheck]:
    """Run every pre-execution rule. Returns all outcomes so the audit record is complete even on failure."""
    return [
        _check_source_table(plan.sql, contract),
        _check_grain_discipline(plan.sql),
        _check_governed_thresholds(plan.sql, contract),
        _check_cross_grain_join(plan.sql, contract),
    ]


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


class GovernedDuckDBTool:
    """Holds a DuckDB connection, the resolved context, and the audit trail behind a LangChain tool.

    A closure over the connection is used rather than passing it as a tool
    argument: connections are not serialisable, and a tool argument the model
    can set is a tool argument the model can get wrong.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, context: ResolvedContext):
        self._conn = conn
        self._context = context
        self.audit_trail: list[AuditRecord] = []
        self.tool: StructuredTool = StructuredTool.from_function(
            func=self.execute,
            name="execute_governed_query",
            description=(
                "Validate a PlannedQuery against its contract and execute it in DuckDB. "
                "Returns an ExecutionResult. Refuses plans that break grain discipline, "
                "use ungoverned thresholds, or join across grains without aggregation."
            ),
        )

    def execute(self, plan: PlannedQuery) -> ExecutionResult:
        """Validate, execute, and audit one plan. Raises GovernanceViolation if any rule fails."""
        contract = self._context.contract_for(plan.lens)
        checks = validate_plan(plan, contract)
        failed = [c for c in checks if not c.passed]

        if failed:
            self._record(plan, row_count=0, executed=False, checks=checks)
            raise GovernanceViolation(
                f"{plan.lens} plan refused by pre-execution validator: "
                + " | ".join(f"{c.name}: {c.detail}" for c in failed)
            )

        relation = self._conn.execute(plan.sql)
        columns = [col[0] for col in relation.description]
        rows = [[_to_scalar(v) for v in row] for row in relation.fetchall()]
        table = ResultTable(columns=columns, rows=rows)

        self._record(plan, row_count=len(rows), executed=True, checks=checks)
        return ExecutionResult(
            lens=plan.lens,
            contract_id=plan.contract_id,
            contract_version=plan.contract_version,
            sql=plan.sql,
            table=table,
            row_count=len(rows),
        )

    def _record(self, plan: PlannedQuery, *, row_count: int, executed: bool, checks: list[GateCheck]) -> None:
        self.audit_trail.append(
            AuditRecord(
                lens=plan.lens,
                contract_id=plan.contract_id,
                contract_version=plan.contract_version,
                sql=plan.sql,
                row_count=row_count,
                executed=executed,
                checks=checks,
                timestamp=datetime.now(timezone.utc),
            )
        )


def _to_scalar(value: object) -> Scalar:
    """Coerce DuckDB cell values (Decimal, numpy types) to plain Python scalars."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# ZERO-TOKEN-MATH GATE
# ---------------------------------------------------------------------------

# A figure as it is written for an executive: thousands separators, one or
# more decimals, or both. Plain integers, version strings (1.2.0) and ranges
# (6-10) are deliberately not matched, so they can never cause a false positive.
_FORMATTED_FIGURE = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)(?![\d.])")


class FactBase(BaseModel):
    """Every figure the narrative is allowed to cite, partitioned by lens.

    The base is exactly the set of numbers the model was shown in its brief:
    segment and baseline sizes, salience statistics, and the governed
    thresholds of each contract. Raw result cells are excluded on purpose.
    The model never sees individual rows, so a cited value that happens to
    equal some row's tenure is a fabrication that coincides with data, and
    the gate must fail it. Partitioning by lens closes the other gap: a real
    enterprise figure cited under the department lens is a wrong claim, not a
    verified one.
    """

    department: list[float] = Field(description="Figures the department-lens narrative may cite.")
    enterprise: list[float] = Field(description="Figures the enterprise-lens narrative may cite.")

    def for_lens(self, lens: Lens) -> list[float]:
        return self.department if lens == "department" else self.enterprise

    def all_values(self) -> list[float]:
        return self.department + self.enterprise


def build_fact_base(
    context: ResolvedContext,
    department_result: ExecutionResult,
    enterprise_result: ExecutionResult,
    department_salience: "SalienceRanking",
    enterprise_salience: "SalienceRanking",
) -> FactBase:
    """Assemble the per-lens fact base from executed results, salience statistics and contract thresholds."""

    def lens_facts(lens: Lens, result: ExecutionResult, ranking: "SalienceRanking") -> list[float]:
        thresholds = [float(t.value) for t in context.contract_for(lens).thresholds]
        return [float(result.row_count)] + ranking.numeric_facts() + thresholds

    return FactBase(
        department=lens_facts("department", department_result, department_salience),
        enterprise=lens_facts("enterprise", enterprise_result, enterprise_salience),
    )


def _matches(cited: float, fact: float) -> bool:
    """A cited value matches a fact when they agree to one decimal place."""
    return abs(cited - fact) <= _GATE_ABS_TOLERANCE


def uncited_prose_figures(prose: str, cited_metrics: list[CitedMetric]) -> list[float]:
    """Formatted figures that appear in the prose but not in the typed cited_metrics list.

    This is a completeness check on the typed field, not a substitute for it.
    The typed list is what gets verified against executed results; this sweep
    only guarantees the model cannot route a figure around that list by
    leaving it out. It looks solely for executive-formatted numbers, so plain
    integers, version strings and category ranges are never flagged.
    """
    cited_values = [m.value for m in cited_metrics]
    found = [float(token.replace(",", "")) for token in _FORMATTED_FIGURE.findall(prose)]
    return sorted({v for v in found if not any(_matches(v, c) for c in cited_values)})


def verify_cited_metrics(
    cited_metrics: list[CitedMetric],
    fact_base: FactBase | Iterable[float],
    prose: str = "",
) -> GateOutcome:
    """ZERO-TOKEN-MATH GATE.

    Every value in the deliverable's typed cited_metrics field must trace to a
    figure in the fact base for the lens it claims. This is an exact check over
    typed numbers, so a year, a rank or a version number in the prose can never
    produce a false positive. When prose is supplied, a second pass confirms
    that every executive-formatted figure in the prose is present in the typed
    list, so a figure cannot bypass the check by being omitted from it.

    A plain iterable of floats is accepted as a lens-agnostic fact base for
    callers that have only one lens in hand.
    """
    if isinstance(fact_base, FactBase):
        facts_for = fact_base.for_lens
    else:
        shared = list(fact_base)

        def facts_for(_: Lens) -> list[float]:
            return shared

    unverifiable = [m for m in cited_metrics if not any(_matches(m.value, f) for f in facts_for(m.lens))]
    uncited = uncited_prose_figures(prose, cited_metrics) if prose else []

    if unverifiable or uncited:
        problems: list[str] = []
        if unverifiable:
            listing = "; ".join(f"{m.label} = {m.value:,.1f} ({m.lens})" for m in unverifiable)
            problems.append(
                f"{len(unverifiable)} cited figure(s) do not trace to any executed result for their lens: {listing}"
            )
        if uncited:
            listing = ", ".join(f"{v:,.1f}" for v in uncited)
            problems.append(f"{len(uncited)} figure(s) appear in the prose but not in cited_metrics: {listing}")
        return GateOutcome(
            passed=False,
            checked=len(cited_metrics),
            unverifiable=unverifiable,
            uncited_prose=uncited,
            detail="ZERO-TOKEN-MATH GATE FAILED: "
            + " | ".join(problems)
            + ". Narrative suppressed; deterministic result sets are surfaced instead.",
        )
    return GateOutcome(
        passed=True,
        checked=len(cited_metrics),
        unverifiable=[],
        uncited_prose=[],
        detail=f"ZERO-TOKEN-MATH GATE PASSED: all {len(cited_metrics)} cited figure(s) trace to executed results "
        "for their lens, and every formatted figure in the prose is cited.",
    )
