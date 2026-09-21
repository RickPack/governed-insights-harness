"""
tests/test_pipeline.py — the governance layer under test with no network and no API key.

The language model is mocked at the model boundary only. The planner mock is a
FakeListChatModel subclass whose `with_structured_output` returns a Runnable
(the fake model piped into a Pydantic parser), so LCEL composition with the
pipe operator is exercised for real. The narrative mock is PydanticAI's
TestModel, which drives the agent loop with a fixed structured output.
Nothing else is mocked: DuckDB, validators, salience and the gate all run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

# Make the repository root importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decomposition import decompose  # noqa: E402
from governed_duckdb_tool import (  # noqa: E402
    CitedMetric,
    FactBase,
    GovernanceViolation,
    GovernedDuckDBTool,
    build_fact_base,
    uncited_prose_figures,
    verify_cited_metrics,
)
from langchain_context_chain import build_planning_chain, resolve_context  # noqa: E402
from narrative_agent import NarrativeBrief  # noqa: E402
from pipeline import run_dual_lens_pipeline  # noqa: E402
from salience import compute_salience  # noqa: E402
from semantic_contracts import (  # noqa: E402
    DEPARTMENT_CONTRACT,
    ENTERPRISE_CONTRACT,
    ContextResolutionError,
    DualLensPlan,
    MetricContract,
    PlannedQuery,
    ThresholdDefinition,
)
from synthetic_data import build_database  # noqa: E402

QUERY = (
    "Compare the profile of our highest-value enterprise accounts, "
    "evaluated at the license level versus the parent organization level."
)


# ---------------------------------------------------------------------------
# Fixtures and mocks
# ---------------------------------------------------------------------------


class StructuredFakeChatModel(FakeListChatModel):
    """A FakeListChatModel that supports with_structured_output.

    The fake emits its canned JSON text; the returned Runnable parses that text
    into the requested Pydantic schema. This mirrors what a real provider does
    and keeps the chain composition identical to production.
    """

    def with_structured_output(self, schema, **kwargs) -> Runnable:  # type: ignore[override]
        def parse(message: AIMessage):
            return schema.model_validate(json.loads(message.content))

        return self | RunnableLambda(parse)


def _segment_sql(contract: MetricContract) -> str:
    """The canonical plan for a contract: entity id, metric, profile attributes, governed filter."""
    return (
        f"SELECT {contract.entity_id_column}, {contract.metric_expression} AS metric_value, "
        f"{', '.join(contract.profile_attributes)} FROM {contract.source_table} "
        f"WHERE {contract.thresholds[0].as_sql()}"
    )


def _plan_dict() -> dict:
    return {
        "department": {
            "lens": "department",
            "contract_id": DEPARTMENT_CONTRACT.contract_id,
            "contract_version": DEPARTMENT_CONTRACT.version,
            "threshold_name": "high_value_floor",
            "sql": _segment_sql(DEPARTMENT_CONTRACT),
            "rationale": "Reads licenses at license grain with the governed floor.",
        },
        "enterprise": {
            "lens": "enterprise",
            "contract_id": ENTERPRISE_CONTRACT.contract_id,
            "contract_version": ENTERPRISE_CONTRACT.version,
            "threshold_name": "high_value_floor",
            "sql": _segment_sql(ENTERPRISE_CONTRACT),
            "rationale": "Reads organizations at organization grain with the governed floor.",
        },
    }


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Every test runs as if no key were present; nothing here may reach a provider."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(scope="module")
def db():
    conn, summary = build_database(seed=42, verbose=False)
    return conn, summary


@pytest.fixture
def context():
    return resolve_context(QUERY)


@pytest.fixture
def good_plan() -> DualLensPlan:
    return DualLensPlan.model_validate(_plan_dict())


@pytest.fixture
def fake_planner() -> StructuredFakeChatModel:
    return StructuredFakeChatModel(responses=[json.dumps(_plan_dict())])


@pytest.fixture
def tool(db, context) -> GovernedDuckDBTool:
    return GovernedDuckDBTool(db[0], context)


# ---------------------------------------------------------------------------
# 1–3: contracts and context resolution
# ---------------------------------------------------------------------------


def test_both_lenses_resolve_from_query(context):
    """A value question resolves one contract per lens, at two different grains."""
    assert context.department_contract.entity_grain == "license"
    assert context.enterprise_contract.entity_grain == "organization"
    assert context.department_segment.contract_id == context.department_contract.contract_id


def test_unmatched_query_raises_context_resolution_error():
    """Fail closed: an unrelated question is refused rather than answered under a guessed definition."""
    with pytest.raises(ContextResolutionError):
        resolve_context("What was the uptime of the platform last quarter?")


def test_malformed_contract_is_unconstructable():
    """A contract with an organization key on a license grain, or a blank metric, never exists as an object."""
    base = DEPARTMENT_CONTRACT.model_dump()
    with pytest.raises(ValidationError):
        MetricContract.model_validate({**base, "entity_id_column": "organization_id"})
    with pytest.raises(ValidationError):
        MetricContract.model_validate({**base, "metric_expression": "   "})
    with pytest.raises(ValidationError):
        MetricContract.model_validate({**base, "version": "v2"})
    with pytest.raises(ValidationError):
        ThresholdDefinition(name="x", column="license_revenue", operator="~", value=1, description="bad operator")


# ---------------------------------------------------------------------------
# 4–5: LCEL composition and the structured-output boundary
# ---------------------------------------------------------------------------


def test_lcel_chain_composes_and_parses_structured_output(fake_planner):
    """The pipe-composed chain returns a GovernedPlan whose plans cite the resolved contracts."""
    chain = build_planning_chain(fake_planner)
    governed = chain.invoke(QUERY)
    assert governed.plan.department.contract_version == DEPARTMENT_CONTRACT.version
    assert governed.plan.enterprise.contract_version == ENTERPRISE_CONTRACT.version
    assert "organizations" in governed.plan.enterprise.sql


def test_malformed_model_output_fails_at_pydantic_boundary():
    """Missing lens, wrong lens label, or a non-SELECT statement: all rejected before any SQL runs."""
    missing_lens = {"department": _plan_dict()["department"]}
    chain = build_planning_chain(StructuredFakeChatModel(responses=[json.dumps(missing_lens)]))
    with pytest.raises(ValidationError):
        chain.invoke(QUERY)

    swapped = _plan_dict()
    swapped["enterprise"]["lens"] = "department"
    with pytest.raises(ValidationError):
        DualLensPlan.model_validate(swapped)

    with pytest.raises(ValidationError):
        PlannedQuery.model_validate({**_plan_dict()["department"], "sql": "DROP TABLE licenses"})


# ---------------------------------------------------------------------------
# 6–8: pre-execution validator
# ---------------------------------------------------------------------------


def test_rejects_raw_row_count_denominator(tool, good_plan):
    """Grain discipline: a ratio over COUNT(*) is refused, and the refusal is audited."""
    bad = good_plan.department.model_copy(
        update={"sql": "SELECT COUNT(*) * 1.0 / COUNT(license_id) AS share FROM licenses WHERE license_revenue >= 5000"}
    )
    with pytest.raises(GovernanceViolation, match="grain_discipline"):
        tool.execute(bad)
    assert tool.audit_trail[-1].executed is False
    assert any(c.name == "grain_discipline" and not c.passed for c in tool.audit_trail[-1].checks)


def test_rejects_ungoverned_threshold(tool, good_plan):
    """A cutoff the contract does not define is refused even though it is syntactically valid SQL."""
    bad = good_plan.department.model_copy(
        update={"sql": "SELECT license_id FROM licenses WHERE license_revenue >= 7500"}
    )
    with pytest.raises(GovernanceViolation, match="governed_thresholds"):
        tool.execute(bad)


def test_intercepts_cross_grain_join(tool, good_plan):
    """Joining the license table into an organization query without aggregation is refused; with aggregation it runs."""
    naive = good_plan.enterprise.model_copy(
        update={
            "sql": "SELECT o.organization_id FROM organizations o JOIN licenses l ON l.organization_id = o.organization_id "
            "WHERE platform_revenue >= 25000"
        }
    )
    with pytest.raises(GovernanceViolation, match="cross_grain_join"):
        tool.execute(naive)

    aggregated = good_plan.enterprise.model_copy(
        update={
            "sql": "SELECT o.organization_id, l.license_total FROM organizations o JOIN "
            "(SELECT organization_id, SUM(license_revenue) AS license_total FROM licenses GROUP BY organization_id) l "
            "ON l.organization_id = o.organization_id WHERE platform_revenue >= 25000"
        }
    )
    result = tool.execute(aggregated)
    assert result.row_count > 0


# ---------------------------------------------------------------------------
# 9–12: zero-token-math gate
# ---------------------------------------------------------------------------


@pytest.fixture
def fact_base(db, context, tool, good_plan) -> FactBase:
    """The per-lens fact base exactly as the pipeline builds it: row counts, salience statistics, thresholds."""
    conn, _ = db
    department_result = tool.execute(good_plan.department)
    enterprise_result = tool.execute(good_plan.enterprise)
    return build_fact_base(
        context,
        department_result,
        enterprise_result,
        compute_salience(conn, DEPARTMENT_CONTRACT, good_plan.department.sql),
        compute_salience(conn, ENTERPRISE_CONTRACT, good_plan.enterprise.sql),
    )


def test_gate_catches_unverifiable_figure(fact_base):
    """A cited value that no query produced fails the gate, and the failure message names the gate."""
    outcome = verify_cited_metrics(
        [CitedMetric(label="invented mean tenure", lens="department", value=99.9)], fact_base
    )
    assert outcome.passed is False
    assert "ZERO-TOKEN-MATH GATE" in outcome.detail
    assert outcome.unverifiable[0].value == 99.9


def test_gate_passes_clean_deliverable(db, good_plan, fact_base):
    """Figures copied from the brief for the right lens all trace, so the gate passes."""
    ranking = compute_salience(db[0], ENTERPRISE_CONTRACT, good_plan.enterprise.sql)
    module_score = next(s for s in ranking.scores if s.attribute == "module_count")
    cited = [
        CitedMetric(label="enterprise segment size", lens="enterprise", value=ranking.segment_size),
        CitedMetric(label="enterprise mean module count", lens="enterprise", value=module_score.segment_value),
        CitedMetric(label="baseline mean module count", lens="enterprise", value=module_score.baseline_value),
        CitedMetric(label="enterprise floor", lens="enterprise", value=ENTERPRISE_CONTRACT.thresholds[0].value),
    ]
    prose = (
        f"Organizations in this segment hold {module_score.segment_value:,.1f} modules against "
        f"{module_score.baseline_value:,.1f} at baseline, above the {ENTERPRISE_CONTRACT.thresholds[0].value:,.1f} floor."
    )
    outcome = verify_cited_metrics(cited, fact_base, prose=prose)
    assert outcome.passed is True
    assert outcome.checked == 4 and outcome.uncited_prose == []


def test_gate_rejects_wrong_lens_and_coincidental_values(db, good_plan, fact_base):
    """A real enterprise figure cited under the department lens fails; so does a fabricated value that
    merely equals some row's cell, because the fact base holds only what the brief showed the model."""
    ranking = compute_salience(db[0], ENTERPRISE_CONTRACT, good_plan.enterprise.sql)
    revenue_score = next(s for s in ranking.scores if s.attribute == "platform_revenue")
    assert revenue_score.segment_value in fact_base.enterprise
    wrong_lens = verify_cited_metrics(
        [CitedMetric(label="platform revenue", lens="department", value=revenue_score.segment_value)], fact_base
    )
    assert wrong_lens.passed is False and wrong_lens.unverifiable[0].lens == "department"

    # 14.0 is a tenure_years cell in the license table, but no brief figure. It must not pass.
    cell_value = db[0].execute("SELECT tenure_years FROM licenses WHERE tenure_years = 14 LIMIT 1").fetchone()
    assert cell_value is not None
    coincidence = verify_cited_metrics([CitedMetric(label="tenure", lens="enterprise", value=14.0)], fact_base)
    assert coincidence.passed is False


def test_gate_prose_sweep_catches_uncited_formatted_figure(fact_base):
    """A formatted figure written into the prose but left out of cited_metrics fails the gate.
    Plain integers, version strings and category ranges are never flagged."""
    cited = [CitedMetric(label="department floor", lens="department", value=5000.0)]
    prose = "Licenses above the 5,000.0 floor average 8,789.5 in revenue (contract v1.2.0, maturity band 6-10, 21 licenses)."
    assert uncited_prose_figures(prose, cited) == [8789.5]
    outcome = verify_cited_metrics(cited, fact_base, prose=prose)
    assert outcome.passed is False and outcome.uncited_prose == [8789.5]
    assert outcome.unverifiable == []


# ---------------------------------------------------------------------------
# 13: salience against a known fixture
# ---------------------------------------------------------------------------


def test_salience_ranking_numeric_and_categorical_fixture():
    """A hand-built table with a known contrast: numeric uses Cohen's d, categorical uses percentage points."""
    import duckdb

    conn = duckdb.connect(":memory:")
    # Ten licenses. Five in the segment (revenue >= 5000) all have tenure 20 and channel
    # 'direct_sales'; five outside have tenure 10 and channel 'self_serve'. Every other
    # attribute is constant.
    conn.execute(
        """
        CREATE TABLE licenses AS
        SELECT 'L' || i AS license_id,
               CASE WHEN i <= 5 THEN 6000.0 ELSE 1000.0 END AS license_revenue,
               '6-10' AS org_maturity_band,
               CASE WHEN i <= 5 THEN 20 ELSE 10 END + (i % 2) AS tenure_years,
               2 AS module_count,
               CASE WHEN i <= 5 THEN 'direct_sales' ELSE 'self_serve' END AS deployment_channel,
               'mid_market' AS org_tier
        FROM range(1, 11) t(i)
        """
    )
    ranking = compute_salience(conn, DEPARTMENT_CONTRACT, _segment_sql(DEPARTMENT_CONTRACT))

    assert ranking.segment_size == 5 and ranking.baseline_size == 10
    by_name = {s.display_name: s for s in ranking.scores}

    tenure = by_name["tenure_years"]
    assert tenure.method == "cohens_d"
    assert tenure.segment_value == pytest.approx(20.6) and tenure.baseline_value == pytest.approx(15.5)
    assert tenure.direction == "higher" and tenure.effect_size > 0.8
    assert ranking.lead_insight().startswith(
        "The most distinctive quality of this segment is deployment channel = direct sales: 100.0% of the segment"
    )

    direct_sales = by_name["deployment_channel = direct_sales"]
    assert direct_sales.method == "percentage_point_delta"
    assert direct_sales.segment_value == 100.0 and direct_sales.baseline_value == 50.0
    assert direct_sales.raw_difference == 50.0 and direct_sales.direction == "higher"

    # Constant attributes are 'similar' with zero effect, and never use the wrong branch.
    assert by_name["module_count"].effect_size == 0.0 and by_name["module_count"].direction == "similar"
    assert by_name["org_maturity_band = 6-10"].method == "percentage_point_delta"
    # The ranking is ordered by absolute effect size, most distinctive first.
    effects = [abs(s.effect_size) for s in ranking.scores]
    assert effects == sorted(effects, reverse=True)


# ---------------------------------------------------------------------------
# 14: end-to-end happy path
# ---------------------------------------------------------------------------


def test_end_to_end_happy_path(db, fake_planner):
    """Question in; both summaries, a memo, a passing gate and a complete audit trail out. No network."""
    conn, summary = db
    department_salience = compute_salience(conn, DEPARTMENT_CONTRACT, _segment_sql(DEPARTMENT_CONTRACT))
    tenure = next(s for s in department_salience.scores if s.attribute == "tenure_years")
    deliverable = {
        "department_summary": {
            "headline": "High value license agreements are long-tenured relationships.",
            "narrative": f"Mean tenure is {tenure.segment_value:,.1f} years against {tenure.baseline_value:,.1f}.",
            "top_attributes": ["tenure_years", "org_maturity_band = 20+"],
        },
        "enterprise_summary": {
            "headline": "High value organizations are defined by module breadth.",
            "narrative": "Organizations in this segment use far more modules than baseline (contract v2.0.0).",
            "top_attributes": ["module_count", "org_tier = enterprise"],
        },
        "reconciliation_memo": "The lenses differ by grain, metric and threshold; neither is wrong.",
        "cited_metrics": [
            {"label": "department mean tenure", "lens": "department", "value": tenure.segment_value},
            {"label": "baseline mean tenure", "lens": "department", "value": tenure.baseline_value},
        ],
    }
    run = run_dual_lens_pipeline(
        QUERY,
        chat_model=fake_planner,
        narrative_model=TestModel(custom_output_args=deliverable),
        conn=conn,
        dataset_summary=summary,
    )
    assert os.environ.get("GOOGLE_API_KEY") is None
    assert run.deliverable is not None and run.narrative_suppressed is False
    assert run.gate.passed and run.narrative_attempts == 1
    assert run.deliverable.department_summary.headline.startswith("High value license agreements")
    assert run.deliverable.enterprise_summary.top_attributes[0] == "module_count"
    assert "neither is wrong" in run.deliverable.reconciliation_memo
    assert run.department_result.row_count > 0 and run.enterprise_result.row_count > 0
    assert len(run.audit_trail) == 2 and all(r.executed for r in run.audit_trail)
    assert run.department_salience.scores[0].attribute in ("license_revenue", "tenure_years")
    assert run.enterprise_salience.lens == "enterprise"

    # Telemetry records observed behaviour only.
    telemetry = run.telemetry
    assert telemetry.llm_calls == 2  # one planner call, one narrative output
    assert telemetry.governed_queries == 2 == len(run.audit_trail)
    assert telemetry.deterministic_operations == 2 + 2 + 2 + 1
    assert set(telemetry.stage_ms) == {"plan", "execute", "salience", "decomposition", "narrative"}
    assert all(ms >= 0 for ms in telemetry.stage_ms.values())
    assert telemetry.total_ms == pytest.approx(sum(telemetry.stage_ms.values()))
    assert all(r.duration_ms >= 0 for r in run.audit_trail)


# 8: narration of the revenue decomposition goes through the existing gate


def test_gate_checks_decomposition_figures(db, context, tool, good_plan):
    """A figure present in the decomposition table passes; one absent from it fails; the brief shows the same figures."""
    conn, _ = db
    dept_decomposition = decompose(conn, "department")
    ent_decomposition = decompose(conn, "enterprise")
    department_salience = compute_salience(conn, DEPARTMENT_CONTRACT, good_plan.department.sql)
    enterprise_salience = compute_salience(conn, ENTERPRISE_CONTRACT, good_plan.enterprise.sql)
    department_result = tool.execute(good_plan.department)
    enterprise_result = tool.execute(good_plan.enterprise)
    fact_base = build_fact_base(
        context, department_result, enterprise_result, department_salience, enterprise_salience,
        dept_decomposition, ent_decomposition,
    )

    churn = ent_decomposition.components.churn
    present = verify_cited_metrics(
        [CitedMetric(label="enterprise churn", lens="enterprise", value=churn)],
        fact_base,
        prose=f"Churn removed {churn:,.1f} of seat-license revenue.",
    )
    assert present.passed is True

    # A derived figure the table does not contain (churn net of contraction) must fail.
    derived = churn - ent_decomposition.components.contraction
    absent = verify_cited_metrics(
        [CitedMetric(label="net churn", lens="enterprise", value=derived)], fact_base
    )
    assert absent.passed is False

    # Lens discipline: the migration-in figure differs by lens, so the department value fails under enterprise.
    assert dept_decomposition.components.migration_in != ent_decomposition.components.migration_in
    wrong_lens = verify_cited_metrics(
        [CitedMetric(label="migration in", lens="enterprise", value=dept_decomposition.components.migration_in)],
        fact_base,
    )
    assert wrong_lens.passed is False

    brief = NarrativeBrief(
        context=context,
        department_result=department_result,
        enterprise_result=enterprise_result,
        department_salience=department_salience,
        enterprise_salience=enterprise_salience,
        department_decomposition=dept_decomposition,
        enterprise_decomposition=ent_decomposition,
    ).render()
    assert f"churn {churn:,.1f}" in brief
    assert "does not reconcile to the full change" in brief


def test_refused_plan_is_audited_with_a_duration(tool, good_plan):
    """A refused call still writes an audit record, and it carries a non-negative duration."""
    bad = good_plan.department.model_copy(update={"sql": good_plan.department.sql.replace("5000", "4999")})
    with pytest.raises(GovernanceViolation):
        tool.execute(bad)
    record = tool.audit_trail[-1]
    assert record.executed is False and record.duration_ms >= 0
