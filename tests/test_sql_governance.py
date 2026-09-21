"""
tests/test_sql_governance.py — the parse-based WHERE-clause check.

The regex rules in governed_duckdb_tool.py find the governed cutoff in the SQL
text. These tests prove the cases where the cutoff is present but not in force:
OR, NOT, UNION and friends. Each bypass below was accepted by the regex rules
alone, and returns rows outside the segment when executed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governed_duckdb_tool import GovernanceViolation, GovernedDuckDBTool, validate_plan  # noqa: E402
from langchain_context_chain import resolve_context  # noqa: E402
from semantic_contracts import DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT, MetricContract, PlannedQuery  # noqa: E402
from sql_predicates import CategoricalPredicate, NumericPredicate, UnsupportedSql, inspect_sql  # noqa: E402
from synthetic_data import build_database  # noqa: E402

QUERY = "Compare the profile of our highest-value enterprise accounts at the license level versus the parent organization level."


@pytest.fixture(scope="module")
def db():
    conn, _ = build_database(seed=42, verbose=False)
    return conn


def _columns(contract: MetricContract) -> str:
    return f"{contract.entity_id_column}, {contract.metric_expression} AS metric_value, {', '.join(contract.profile_attributes)}"


def _plan(contract: MetricContract, where: str, *, table: str | None = None, tail: str = "") -> PlannedQuery:
    source = table or contract.source_table
    return PlannedQuery(
        lens=contract.lens,
        contract_id=contract.contract_id,
        contract_version=contract.version,
        threshold_name="high_value_floor",
        sql=f"SELECT {_columns(contract)} FROM {source} WHERE {where}{tail}",
        rationale="test",
    )


def _failed(plan: PlannedQuery, contract: MetricContract) -> dict[str, str]:
    return {c.name: c.detail for c in validate_plan(plan, contract) if not c.passed}


LICENSE_FLOOR = "license_revenue >= 5000"


def test_canonical_plan_passes_for_both_contracts():
    """The plan every existing test and the live planner produces is still accepted."""
    for contract in (DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT):
        threshold = contract.thresholds[0]
        assert _failed(_plan(contract, threshold.as_sql()), contract) == {}


@pytest.mark.parametrize(
    "where",
    [
        f"{LICENSE_FLOOR} OR 1 = 1",
        f"{LICENSE_FLOOR} OR deployment_channel = 'self_serve'",
        f"({LICENSE_FLOOR}) OR (tenure_years >= 0)",
        "NOT (license_revenue >= 5000)",
        f"{LICENSE_FLOOR} AND (1 = 1 OR tenure_years > 3)",
    ],
)
def test_widened_or_inverted_filters_are_refused(where):
    """OR and NOT keep the cutoff in the text and drop it from the segment."""
    failed = _failed(_plan(DEPARTMENT_CONTRACT, where), DEPARTMENT_CONTRACT)
    assert "where_clause" in failed


def test_refusal_prevents_execution_and_is_audited(db):
    """Through the tool: no rows are returned, and the audit record names the rule that refused it."""
    tool = GovernedDuckDBTool(db, resolve_context(QUERY))
    with pytest.raises(GovernanceViolation, match="where_clause"):
        tool.execute(_plan(DEPARTMENT_CONTRACT, f"{LICENSE_FLOOR} OR 1 = 1"))
    record = tool.audit_trail[-1]
    assert record.executed is False and record.row_count == 0
    assert any(c.name == "where_clause" and not c.passed for c in record.checks)


def test_bypass_would_have_returned_every_license(db):
    """Why the rule matters: the refused SQL, run directly, ignores the floor entirely."""
    total = db.execute("SELECT COUNT(*) FROM licenses").fetchone()[0]
    governed = db.execute(f"SELECT COUNT(*) FROM licenses WHERE {LICENSE_FLOOR}").fetchone()[0]
    widened = db.execute(f"SELECT COUNT(*) FROM licenses WHERE {LICENSE_FLOOR} OR 1 = 1").fetchone()[0]
    assert widened == total and governed < total


def test_reversed_and_decimal_forms_of_the_governed_cutoff_are_accepted():
    """Equivalent spellings of the floor pass; the rule is about meaning, not text."""
    for where in ("5000 <= license_revenue", "license_revenue >= 5000.0", "licenses.license_revenue >= 5000"):
        assert _failed(_plan(DEPARTMENT_CONTRACT, where), DEPARTMENT_CONTRACT) == {}, where


def test_wrong_operator_or_value_is_refused():
    for where in ("license_revenue > 5000", "license_revenue <= 5000", "license_revenue >= 4999"):
        assert "where_clause" in _failed(_plan(DEPARTMENT_CONTRACT, where), DEPARTMENT_CONTRACT), where


def test_named_threshold_must_actually_be_applied():
    """A plan that cites a threshold it does not apply is refused, even if it applies some other governed one."""
    plan = _plan(DEPARTMENT_CONTRACT, LICENSE_FLOOR).model_copy(update={"threshold_name": "no_such_threshold"})
    assert "does not exist" in _failed(plan, DEPARTMENT_CONTRACT)["where_clause"]


@pytest.mark.parametrize(
    "sql, reason",
    [
        (f"SELECT license_id FROM licenses WHERE {LICENSE_FLOOR} UNION ALL SELECT license_id FROM licenses", "set operations"),
        (f"WITH c AS (SELECT * FROM licenses) SELECT license_id FROM c WHERE {LICENSE_FLOOR}", "common table"),
        (f"SELECT license_id FROM licenses WHERE {LICENSE_FLOOR} LIMIT 5", "LIMIT"),
        (f"SELECT license_id FROM licenses WHERE {LICENSE_FLOOR} GROUP BY license_id HAVING COUNT(*) > 0", "HAVING"),
        (f"SELECT license_id FROM licenses USING SAMPLE 10 PERCENT", "SAMPLE"),
        (f"SELECT license_id FROM read_csv('x.csv') WHERE {LICENSE_FLOOR}", "not supported"),
    ],
)
def test_constructs_that_change_the_segment_without_a_threshold_are_refused(sql, reason):
    with pytest.raises(UnsupportedSql, match=reason):
        inspect_sql(sql)


def test_only_governed_tables_may_be_read():
    """The decomposition tables and system catalogs are not sources a segment plan may touch."""
    plan = _plan(DEPARTMENT_CONTRACT, LICENSE_FLOOR).model_copy(
        update={"sql": f"SELECT license_id FROM licenses WHERE {LICENSE_FLOOR} AND license_id IN (SELECT license_id FROM license_movements)"}
    )
    # IN (subquery) is itself outside the subset; either reason is a refusal.
    assert "where_clause" in _failed(plan, DEPARTMENT_CONTRACT)

    direct = _plan(DEPARTMENT_CONTRACT, LICENSE_FLOOR, table="licenses l JOIN license_movements m ON m.license_id = l.license_id")
    assert "not governed sources" in _failed(direct, DEPARTMENT_CONTRACT)["where_clause"]


def test_aggregated_cross_grain_subquery_is_still_accepted():
    """The one legitimate cross-table shape (aggregate the foreign grain, then join) keeps working."""
    sql = (
        "SELECT o.organization_id, l.license_total FROM organizations o JOIN "
        "(SELECT organization_id, SUM(license_revenue) AS license_total FROM licenses GROUP BY organization_id) l "
        "ON l.organization_id = o.organization_id WHERE platform_revenue >= 25000"
    )
    plan = _plan(ENTERPRISE_CONTRACT, "platform_revenue >= 25000").model_copy(update={"sql": sql})
    assert _failed(plan, ENTERPRISE_CONTRACT) == {}


def test_inspection_reports_predicates_by_position():
    """Top-level and nested predicates are kept apart, and text predicates are recognised for the dimension rules."""
    inspected = inspect_sql(
        "SELECT o.organization_id FROM organizations o JOIN "
        "(SELECT organization_id FROM licenses WHERE license_revenue >= 5000) l ON l.organization_id = o.organization_id "
        "WHERE platform_revenue >= 25000 AND org_tier IN ('smb', 'enterprise') AND deployment_channel = 'direct_sales'"
    )
    assert inspected.top_level == (
        NumericPredicate("platform_revenue", ">=", 25000.0),
        CategoricalPredicate("org_tier", ("smb", "enterprise")),
        CategoricalPredicate("deployment_channel", ("direct_sales",)),
    )
    assert inspected.nested == (NumericPredicate("license_revenue", ">=", 5000.0),)
    assert inspected.tables == ("organizations", "licenses")
