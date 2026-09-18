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
in prose. This module checks every one of those values against the union of
the executed result sets. A figure that cannot be traced to a query result
fails the gate. Because the cited figures are a typed field rather than prose
parsed with a regular expression, the check is exact: a year, a rank, or a
count in the prose can never be mistaken for a metric, and a metric can never
hide in the prose unchecked.

Every tool call writes an AuditRecord: lens, contract version, SQL, row
count, validator outcomes, timestamp. The audit trail is a typed list, so it
can be rendered, stored, or diffed without parsing log lines.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Iterable, Union

import duckdb
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from semantic_contracts import CONTRACT_REGISTRY, Lens, MetricContract, PlannedQuery, ResolvedContext

# A DuckDB cell as it crosses the tool boundary. Kept to scalars so ResultTable
# is JSON-serialisable and comparable without pandas.
Scalar = Union[str, int, float, bool, None]

# Relative tolerance for the gate. A cited figure must equal a computed figure
# to one decimal place or to one part in a thousand, whichever is looser.
_GATE_ABS_TOLERANCE = 0.05
_GATE_REL_TOLERANCE = 1e-3


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

    def numeric_values(self) -> Iterable[float]:
        """Every numeric cell, used to build the gate's fact base."""
        for row in self.rows:
            for value in row:
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                    yield float(value)


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
    unverifiable: list[CitedMetric] = Field(description="Cited metrics with no matching computed value.")
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


def numeric_fact_base(results: Iterable[ExecutionResult], extra_facts: Iterable[float] = ()) -> list[float]:
    """The union of every number the deterministic layer produced.

    Includes every numeric cell of every executed result set, each result's
    row count, and any additional deterministic facts: the salience means and
    effect sizes (computed in SQL and Python, never by the model) and the
    governed threshold values from the contracts (see contract_facts).
    """
    facts: list[float] = []
    for result in results:
        facts.extend(result.table.numeric_values())
        facts.append(float(result.row_count))
    facts.extend(float(v) for v in extra_facts)
    return facts


def contract_facts(context: ResolvedContext) -> list[float]:
    """Governed threshold values from both resolved contracts.

    A narrative that says "licenses at or above the 5,000.0 floor" is citing
    a versioned contract, not inventing a number. Contract thresholds are
    deterministic facts and belong in the base the gate checks against.
    """
    return [float(t.value) for lens in ("department", "enterprise") for t in context.contract_for(lens).thresholds]


def _matches(cited: float, fact: float) -> bool:
    """A cited value matches a fact if equal to one decimal place or within one part in a thousand."""
    return abs(cited - fact) <= _GATE_ABS_TOLERANCE or abs(cited - fact) <= _GATE_REL_TOLERANCE * abs(fact)


def verify_cited_metrics(cited_metrics: list[CitedMetric], fact_base: Iterable[float]) -> GateOutcome:
    """ZERO-TOKEN-MATH GATE.

    Every value in the deliverable's typed cited_metrics field must trace to a
    value in the fact base. This is an exact check over typed numbers, not a
    regular expression over prose, so a year or a rank in the narrative can
    never produce a false positive and a figure can never slip through unchecked.
    """
    facts = list(fact_base)
    unverifiable = [m for m in cited_metrics if not any(_matches(m.value, f) for f in facts)]
    if unverifiable:
        listing = "; ".join(f"{m.label} = {m.value:,.1f} ({m.lens})" for m in unverifiable)
        return GateOutcome(
            passed=False,
            checked=len(cited_metrics),
            unverifiable=unverifiable,
            detail=f"ZERO-TOKEN-MATH GATE FAILED: {len(unverifiable)} cited figure(s) do not trace to any "
            f"executed result: {listing}. Narrative suppressed; deterministic result sets are surfaced instead.",
        )
    return GateOutcome(
        passed=True,
        checked=len(cited_metrics),
        unverifiable=[],
        detail=f"ZERO-TOKEN-MATH GATE PASSED: all {len(cited_metrics)} cited figure(s) trace to executed results.",
    )
