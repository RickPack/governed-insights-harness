"""
tests/test_planner_scope.py — question-driven scoping, and the allowlist that governs it.

The planner is useful where a contract cannot decide alone: turning "direct
sales only" into a restriction. These tests pin the other half of that
bargain. The model may only name a column and values the contract lists, the
SQL must apply exactly what the plan declares, and the baseline, salience and
decomposition all follow the restriction. No network and no key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decomposition import decompose  # noqa: E402
from governed_duckdb_tool import GovernanceViolation, GovernedDuckDBTool, validate_plan  # noqa: E402
from langchain_context_chain import resolve_context  # noqa: E402
from narrative_agent import NarrativeBrief  # noqa: E402
from pipeline import run_dual_lens_pipeline  # noqa: E402
from salience import EmptySegmentError, compute_salience  # noqa: E402
from semantic_contracts import (  # noqa: E402
    DEPARTMENT_CONTRACT,
    ENTERPRISE_CONTRACT,
    FILTERABLE_DIMENSIONS,
    DimensionDefinition,
    DimensionFilter,
    DualLensPlan,
    GovernedPlan,
    MetricContract,
    PlannedQuery,
)
from synthetic_data import DEPLOYMENT_CHANNELS, MATURITY_BANDS, ORG_TIERS, build_database  # noqa: E402

QUERY = (
    "Compare the profile of our highest-value enterprise accounts, "
    "evaluated at the license level versus the parent organization level."
)
DIRECT = DimensionFilter(column="deployment_channel", values=("direct_sales",))


class StructuredFakeChatModel(FakeListChatModel):
    """FakeListChatModel with with_structured_output, as in test_pipeline.py."""

    def with_structured_output(self, schema, **kwargs) -> Runnable:  # type: ignore[override]
        def parse(message: AIMessage):
            return schema.model_validate(json.loads(message.content))

        return self | RunnableLambda(parse)


@pytest.fixture(scope="module")
def db():
    return build_database(seed=42, verbose=False)


@pytest.fixture
def context():
    return resolve_context(QUERY)


def _sql(contract: MetricContract, extra: str = "") -> str:
    return (
        f"SELECT {contract.entity_id_column}, {contract.metric_expression} AS metric_value, "
        f"{', '.join(contract.profile_attributes)} FROM {contract.source_table} "
        f"WHERE {contract.thresholds[0].as_sql()}{extra}"
    )


def _planned(contract: MetricContract, extra: str = "", **fields) -> PlannedQuery:
    return PlannedQuery(
        lens=contract.lens,
        contract_id=contract.contract_id,
        contract_version=contract.version,
        threshold_name="high_value_floor",
        sql=_sql(contract, extra),
        rationale="test",
        **fields,
    )


def _plan_dict(*, extra: str = " AND deployment_channel = 'direct_sales'", **fields) -> dict:
    plans = {}
    for contract in (DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT):
        plans[contract.lens] = _planned(contract, extra, **fields).model_dump(mode="json")
    return plans


def _failed(plan: PlannedQuery, contract: MetricContract) -> dict[str, str]:
    return {c.name: c.detail for c in validate_plan(plan, contract) if not c.passed}


# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------


def test_allowlist_equals_what_the_generator_can_produce(db):
    """A dimension listing a value the data never has, or missing one it does, would silently mis-scope questions."""
    conn, _ = db
    expected = {
        "deployment_channel": set(DEPLOYMENT_CHANNELS),
        "org_tier": set(ORG_TIERS),
        "org_maturity_band": set(MATURITY_BANDS),
    }
    assert {d.column: set(d.values) for d in FILTERABLE_DIMENSIONS} == expected
    for table in ("licenses", "organizations"):
        for column, values in expected.items():
            stored = {r[0] for r in conn.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()}
            assert stored <= values, (table, column, stored - values)


def test_contract_rejects_a_dimension_that_is_not_a_profile_column():
    base = DEPARTMENT_CONTRACT.model_dump()
    bad = DimensionDefinition(column="license_id", values=("L0001",), description="not profiled")
    with pytest.raises(ValidationError, match="not profile attributes"):
        MetricContract.model_validate({**base, "filterable_dimensions": (bad,)})
    with pytest.raises(ValidationError, match="only once"):
        MetricContract.model_validate({**base, "filterable_dimensions": FILTERABLE_DIMENSIONS + FILTERABLE_DIMENSIONS[:1]})


def test_filter_renders_safe_sql():
    """Values are quoted with doubled quotes and columns must be plain identifiers, so a filter cannot end its own literal."""
    hostile = DimensionFilter(column="deployment_channel", values=("x' OR '1'='1",))
    assert hostile.as_sql() == "deployment_channel = 'x'' OR ''1''=''1'"
    with pytest.raises(ValidationError):
        DimensionFilter(column="deployment_channel; DROP TABLE licenses", values=("x",))
    assert DimensionFilter(column="org_tier", values=("smb", "enterprise")).as_sql() == "org_tier IN ('smb', 'enterprise')"


def test_prompt_lists_the_governed_dimensions_and_the_tier_caveat():
    text = DEPARTMENT_CONTRACT.describe_for_prompt()
    assert "governed dimensions" in text and "direct_sales" in text
    assert "NOT the enterprise tier" in text


# ---------------------------------------------------------------------------
# Plan-level validation: hallucinated scope stops at the type boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filters, message",
    [
        ([{"column": "license_id", "values": ["L0001"]}], "not a governed dimension"),
        ([{"column": "deployment_channel", "values": ["reseller"]}], "does not allow"),
        (
            [{"column": "org_tier", "values": ["smb"]}, {"column": "org_tier", "values": ["enterprise"]}],
            "twice",
        ),
    ],
)
def test_governed_plan_rejects_ungoverned_scope(context, filters, message):
    plans = _plan_dict(extra="")
    plans["department"]["dimension_filters"] = filters
    with pytest.raises(ValidationError, match=message):
        GovernedPlan(context=context, plan=DualLensPlan.model_validate(plans))


def test_governed_plan_rejects_unknown_focus_attribute(context):
    plans = _plan_dict(extra="")
    plans["enterprise"]["focus_attributes"] = ["license_revenue"]
    with pytest.raises(ValidationError, match="not profile attributes"):
        GovernedPlan(context=context, plan=DualLensPlan.model_validate(plans))


def test_governed_plan_accepts_declared_scope(context):
    plans = _plan_dict(dimension_filters=[DIRECT.model_dump(mode="json")])
    governed = GovernedPlan(context=context, plan=DualLensPlan.model_validate(plans))
    assert governed.plan.department.dimension_filters == [DIRECT]


# ---------------------------------------------------------------------------
# The tool: SQL must apply exactly what the plan declares
# ---------------------------------------------------------------------------


def test_scoped_plan_executes_and_is_audited(db, context):
    conn, _ = db
    tool = GovernedDuckDBTool(conn, context)
    scoped = tool.execute(_planned(DEPARTMENT_CONTRACT, " AND deployment_channel = 'direct_sales'", dimension_filters=[DIRECT]))
    unscoped = tool.execute(_planned(DEPARTMENT_CONTRACT))
    frame = scoped.table.to_dataframe()
    assert 0 < scoped.row_count < unscoped.row_count
    assert set(frame["deployment_channel"]) == {"direct_sales"}
    check = next(c for c in tool.audit_trail[0].checks if c.name == "where_clause")
    assert check.passed and "deployment channel is direct sales" in check.detail


@pytest.mark.parametrize(
    "extra, filters, reason",
    [
        (" AND deployment_channel = 'direct_sales'", [], "do not match"),
        ("", [DIRECT], "do not match"),
        (" AND deployment_channel = 'partner_channel'", [DIRECT], "do not match"),
        (" AND deployment_channel = 'reseller'", [], "not allowed"),
        (" AND license_id = 'L0001'", [], "not a governed dimension"),
        (" AND deployment_channel = 'direct_sales' AND deployment_channel = 'self_serve'", [DIRECT], "twice"),
        (" OR deployment_channel = 'self_serve'", [], "OR is not allowed"),
        (" AND deployment_channel <> 'self_serve'", [], "NOT"),
        (" AND deployment_channel NOT IN ('self_serve')", [], "NOT"),
        (" AND deployment_channel LIKE 'direct%'", [], "not allowed"),
        (" AND lower(deployment_channel) = 'direct_sales'", [], "one column and one constant"),
    ],
)
def test_sql_that_disagrees_with_the_declared_scope_is_refused(extra, filters, reason):
    failed = _failed(_planned(DEPARTMENT_CONTRACT, extra, dimension_filters=filters), DEPARTMENT_CONTRACT)
    assert "where_clause" in failed and reason.lower() in failed["where_clause"].lower(), failed


def test_dimension_filter_inside_a_subquery_is_refused():
    """A filter the baseline cannot see would make the segment and its baseline disagree."""
    sql = (
        "SELECT o.organization_id, o.platform_revenue AS metric_value FROM organizations o JOIN "
        "(SELECT organization_id FROM licenses WHERE deployment_channel = 'direct_sales' GROUP BY organization_id) l "
        "ON l.organization_id = o.organization_id WHERE platform_revenue >= 25000"
    )
    plan = _planned(ENTERPRISE_CONTRACT).model_copy(update={"sql": sql})
    assert "top-level WHERE" in _failed(plan, ENTERPRISE_CONTRACT)["where_clause"]


def test_refused_scope_never_executes(db, context):
    tool = GovernedDuckDBTool(db[0], context)
    with pytest.raises(GovernanceViolation, match="where_clause"):
        tool.execute(_planned(DEPARTMENT_CONTRACT, " AND deployment_channel = 'reseller'"))
    assert tool.audit_trail[-1].executed is False


# ---------------------------------------------------------------------------
# Scope reaches the baseline, the salience ranking and the decomposition
# ---------------------------------------------------------------------------


def test_salience_baseline_is_the_scoped_population(db):
    conn, _ = db
    scope = (DIRECT,)
    ranking = compute_salience(conn, DEPARTMENT_CONTRACT, _sql(DEPARTMENT_CONTRACT, " AND deployment_channel = 'direct_sales'"), scope=scope)
    in_scope = conn.execute("SELECT COUNT(*) FROM licenses WHERE deployment_channel = 'direct_sales'").fetchone()[0]
    assert ranking.baseline_size == in_scope < conn.execute("SELECT COUNT(*) FROM licenses").fetchone()[0]
    assert 0 < ranking.segment_size < ranking.baseline_size
    assert ranking.scope == scope and "deployment channel is direct sales" in ranking.scope_text()
    channel = next(s for s in ranking.scores if s.attribute == "deployment_channel" and s.level == "direct_sales")
    # Within the scope the restricted attribute cannot distinguish the segment: 100% against 100%.
    assert (channel.segment_value, channel.baseline_value, channel.direction) == (100.0, 100.0, "similar")


def test_focus_attributes_limit_what_is_profiled(db):
    conn, _ = db
    ranking = compute_salience(
        conn, DEPARTMENT_CONTRACT, _sql(DEPARTMENT_CONTRACT), attributes=("tenure_years", DEPARTMENT_CONTRACT.metric_expression)
    )
    assert {s.attribute for s in ranking.scores} == {"tenure_years", "license_revenue"}


def test_empty_segment_fails_closed_with_a_reason(db):
    conn, _ = db
    empty = DimensionFilter(column="org_maturity_band", values=("0-2",))
    sql = _sql(DEPARTMENT_CONTRACT, " AND org_maturity_band = '0-2'")
    assert conn.execute(f"SELECT COUNT(*) FROM ({sql}) t").fetchone()[0] == 0  # precondition: no high-value licenses here
    with pytest.raises(EmptySegmentError, match="segment is empty: 0 of") as caught:
        compute_salience(conn, DEPARTMENT_CONTRACT, sql, scope=(empty,))
    assert "org maturity band is 0-2" in str(caught.value)


def test_scoped_decomposition_reconciles_and_matches_an_independent_sum(db):
    conn, _ = db
    full = decompose(conn, "department")
    scoped = decompose(conn, "department", scope=(DIRECT,))
    in_scope = {r[0] for r in conn.execute("SELECT license_id FROM licenses WHERE deployment_channel = 'direct_sales'").fetchall()}
    expected = [r for r in full.rows if r.entity_id in in_scope]
    assert scoped.outcome.passed and scoped.entities == len(expected) < full.entities
    assert scoped.reported_change == pytest.approx(sum(r.reported_change for r in expected))
    assert scoped.components.churn == pytest.approx(sum(r.components.churn for r in expected))
    assert scoped.components.net == pytest.approx(scoped.reported_change)
    assert "restricted to deployment channel is direct sales" in scoped.render()
    assert full.scope == () and "restricted" not in full.scope_note


def test_scoped_organization_decomposition_uses_organization_attributes(db):
    conn, _ = db
    scope = (DimensionFilter(column="org_tier", values=("mid_market", "enterprise")),)
    full = decompose(conn, "enterprise")
    scoped = decompose(conn, "enterprise", scope=scope)
    tiers = dict(conn.execute("SELECT organization_id, org_tier FROM organizations").fetchall())
    expected = [r for r in full.rows if tiers[r.entity_id] in ("mid_market", "enterprise")]
    assert scoped.entities == len(expected) < full.entities
    assert scoped.reported_change == pytest.approx(sum(r.reported_change for r in expected))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _deliverable(department_ranking, enterprise_ranking) -> dict:
    return {
        "department_summary": {
            "headline": "Direct sales licenses above the floor differ mainly by tenure.",
            "narrative": "This lens covers direct sales licenses only, and its baseline is the same scope.",
            "top_attributes": ["tenure_years"],
        },
        "enterprise_summary": {
            "headline": "Direct sales organizations above the floor differ by module breadth.",
            "narrative": "This lens covers direct sales organizations only, and its baseline is the same scope.",
            "top_attributes": ["module_count"],
        },
        "reconciliation_memo": (
            "The lenses differ by grain, metric and threshold; neither is wrong. "
            "The question also asked about startups, which no governed dimension covers, so that restriction was not applied."
        ),
        "cited_metrics": [
            {"label": "department segment size", "lens": "department", "value": department_ranking.segment_size},
            {"label": "department baseline size", "lens": "department", "value": department_ranking.baseline_size},
            {"label": "enterprise segment size", "lens": "enterprise", "value": enterprise_ranking.segment_size},
        ],
    }


def test_end_to_end_scoped_run_carries_scope_through_every_stage(db):
    conn, summary = db
    dept_ranking = compute_salience(conn, DEPARTMENT_CONTRACT, _sql(DEPARTMENT_CONTRACT, " AND deployment_channel = 'direct_sales'"), scope=(DIRECT,))
    ent_ranking = compute_salience(conn, ENTERPRISE_CONTRACT, _sql(ENTERPRISE_CONTRACT, " AND deployment_channel = 'direct_sales'"), scope=(DIRECT,))
    plans = _plan_dict(dimension_filters=[DIRECT.model_dump(mode="json")], unapplied_qualifiers=["startups"])

    run = run_dual_lens_pipeline(
        QUERY,
        chat_model=StructuredFakeChatModel(responses=[json.dumps(plans)]),
        narrative_model=TestModel(custom_output_args=_deliverable(dept_ranking, ent_ranking)),
        conn=conn,
        dataset_summary=summary,
    )

    assert run.gate.passed and not run.narrative_suppressed
    assert run.department_salience.scope == (DIRECT,) and run.enterprise_salience.scope == (DIRECT,)
    assert run.department_decomposition.scope == (DIRECT,) and run.enterprise_decomposition.scope == (DIRECT,)
    assert run.unapplied_qualifiers == ["startups"]
    assert set(run.department_result.table.to_dataframe()["deployment_channel"]) == {"direct_sales"}
    assert all(r.executed and any(c.name == "where_clause" and c.passed for c in r.checks) for r in run.audit_trail)
    assert len(run.audit_trail) == 2

    brief = NarrativeBrief(
        context=run.governed_plan.context,
        department_result=run.department_result,
        enterprise_result=run.enterprise_result,
        department_salience=run.department_salience,
        enterprise_salience=run.enterprise_salience,
        department_decomposition=run.department_decomposition,
        enterprise_decomposition=run.enterprise_decomposition,
        unapplied_qualifiers=run.unapplied_qualifiers,
    ).render()
    assert "NOT APPLIED" in brief and "'startups'" in brief
    assert "restricts this lens to deployment channel is direct sales" in brief


def test_unscoped_question_is_unchanged(db):
    """The default question restricts nothing: no scope anywhere, and the brief has no scope or NOT APPLIED lines."""
    conn, summary = db
    plans = _plan_dict(extra="")
    dept = compute_salience(conn, DEPARTMENT_CONTRACT, _sql(DEPARTMENT_CONTRACT))
    ent = compute_salience(conn, ENTERPRISE_CONTRACT, _sql(ENTERPRISE_CONTRACT))
    deliverable = _deliverable(dept, ent)
    deliverable["department_summary"]["narrative"] = "This lens covers every license above the floor."
    deliverable["enterprise_summary"]["narrative"] = "This lens covers every organization above the floor."
    deliverable["reconciliation_memo"] = "The lenses differ by grain, metric and threshold; neither is wrong."
    run = run_dual_lens_pipeline(
        QUERY,
        chat_model=StructuredFakeChatModel(responses=[json.dumps(plans)]),
        narrative_model=TestModel(custom_output_args=deliverable),
        conn=conn,
        dataset_summary=summary,
    )
    assert run.gate.passed
    assert run.department_salience.scope == () and run.enterprise_decomposition.scope == () and run.unapplied_qualifiers == []
    assert run.department_salience.baseline_size == summary.licenses


def test_renderers_show_scope_and_unapplied_qualifiers(db, capsys):
    """A reader of the notebook or the demo can see what the question was narrowed to, and what could not be applied."""
    from executive_render import render_executive_html
    from run_demo import print_run

    conn, summary = db
    dept = compute_salience(conn, DEPARTMENT_CONTRACT, _sql(DEPARTMENT_CONTRACT, " AND deployment_channel = 'direct_sales'"), scope=(DIRECT,))
    ent = compute_salience(conn, ENTERPRISE_CONTRACT, _sql(ENTERPRISE_CONTRACT, " AND deployment_channel = 'direct_sales'"), scope=(DIRECT,))
    plans = _plan_dict(
        dimension_filters=[DIRECT.model_dump(mode="json")], unapplied_qualifiers=["startups"], focus_attributes=["tenure_years"]
    )
    run = run_dual_lens_pipeline(
        QUERY,
        chat_model=StructuredFakeChatModel(responses=[json.dumps(plans)]),
        narrative_model=TestModel(custom_output_args=_deliverable(dept, ent)),
        conn=conn,
        dataset_summary=summary,
    )
    html = render_executive_html(run)
    assert "Scope:" in html and "deployment channel is direct sales" in html and "Not applied." in html

    print_run(run)
    out = capsys.readouterr().out
    assert "scope: deployment channel is direct sales" in out and "focus: tenure_years" in out
    assert "NOT APPLIED (no governed dimension covers): 'startups'" in out
