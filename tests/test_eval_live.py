"""
tests/test_eval_live.py — the live-evaluation script, tested with no model and no key.

The script measures a real model. What can be tested offline is everything
around the measurement: the interval arithmetic, how a run is scored, how each
kind of failure is classified, that the report survives empty data, and that
the script refuses to spend API calls unless told to.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_live  # noqa: E402
from decomposition import ReconciliationOutcome, ReconciliationError  # noqa: E402
from eval_live import CASES, EvalCase, RunRecord, run_case, score_run, summarize, wilson_interval  # noqa: E402
from governed_duckdb_tool import GovernanceViolation  # noqa: E402
from pipeline import run_dual_lens_pipeline  # noqa: E402
from salience import EmptySegmentError  # noqa: E402
from semantic_contracts import (  # noqa: E402
    DEPARTMENT_CONTRACT,
    ENTERPRISE_CONTRACT,
    ContextResolutionError,
    DimensionFilter,
    PlannedQuery,
)
from synthetic_data import build_database  # noqa: E402

DIRECT = DimensionFilter(column="deployment_channel", values=("direct_sales",))


class StructuredFakeChatModel(FakeListChatModel):
    def with_structured_output(self, schema, **kwargs) -> Runnable:  # type: ignore[override]
        def parse(message: AIMessage):
            return schema.model_validate(json.loads(message.content))

        return self | RunnableLambda(parse)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(scope="module")
def db():
    return build_database(seed=42, verbose=False)


def _planned(contract, *, scoped: bool, **fields) -> dict:
    extra = " AND deployment_channel = 'direct_sales'" if scoped else ""
    return PlannedQuery(
        lens=contract.lens,
        contract_id=contract.contract_id,
        contract_version=contract.version,
        threshold_name="high_value_floor",
        sql=(
            f"SELECT {contract.entity_id_column}, {contract.metric_expression} AS metric_value, "
            f"{', '.join(contract.profile_attributes)} FROM {contract.source_table} "
            f"WHERE {contract.thresholds[0].as_sql()}{extra}"
        ),
        rationale="test",
        dimension_filters=[DIRECT] if scoped else [],
        **fields,
    ).model_dump(mode="json")


def _run(db, *, scoped: bool, deliverable: dict | None = None, **fields):
    conn, summary = db
    plans = {c.lens: _planned(c, scoped=scoped, **fields) for c in (DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT)}
    grounded = deliverable or {
        "department_summary": {"headline": "A headline.", "narrative": "No figures here.", "top_attributes": ["tenure_years"]},
        "enterprise_summary": {"headline": "A headline.", "narrative": "No figures here.", "top_attributes": ["module_count"]},
        "reconciliation_memo": "The lenses differ by grain, metric and threshold; neither is wrong.",
        "cited_metrics": [],
    }
    return run_dual_lens_pipeline(
        "Compare the profile of our highest-value enterprise accounts, evaluated at the license level versus the parent organization level.",
        chat_model=StructuredFakeChatModel(responses=[json.dumps(plans)]),
        narrative_model=TestModel(custom_output_args=grounded),
        conn=conn,
        dataset_summary=summary,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "successes, n, low, high",
    [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634), (18, 20, 0.6990, 0.9721)],
)
def test_wilson_interval_matches_published_values(successes, n, low, high):
    got_low, got_high = wilson_interval(successes, n)
    assert got_low == pytest.approx(low, abs=5e-4) and got_high == pytest.approx(high, abs=5e-4)


def test_wilson_interval_with_no_data_is_undefined_not_zero():
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_summarize_reports_rates_with_denominators():
    records = [
        RunRecord("c", 1, "ok", scope_correct=True, unapplied_ok=True, focus_ok=True, gate_passed=True, first_attempt_pass=True, llm_calls=2, total_ms=100),
        RunRecord("c", 2, "ok", scope_correct=False, unapplied_ok=True, focus_ok=True, gate_passed=True, first_attempt_pass=False, llm_calls=3, total_ms=300),
        RunRecord("c", 3, "ok", scope_correct=True, unapplied_ok=True, focus_ok=True, gate_passed=False, first_attempt_pass=False, llm_calls=4, total_ms=200),
        RunRecord("c", 4, "validator_refused", detail="refused"),
    ]
    s = summarize(records)
    assert s["runs"] == 4 and s["outcomes"]["ok"] == 3 and s["outcomes"]["validator_refused"] == 1
    assert (s["completion"]["successes"], s["completion"]["n"]) == (3, 4)
    assert (s["scope_correct"]["successes"], s["scope_correct"]["n"]) == (2, 3)
    assert (s["gate_first_attempt"]["successes"], s["gate_eventually"]["successes"], s["suppressed"]["successes"]) == (1, 2, 1)
    assert s["median_ms"] == 200 and s["max_ms"] == 300 and s["mean_llm_calls"] == pytest.approx(3.0)


def test_report_survives_empty_and_all_failed_data():
    meta = {"finished_utc": "now", "planner_model": "p", "narrative_model": "n", "runs_per_case": 1, "seed": 42}
    assert "n/a" in eval_live.format_report(CASES[:1], [], meta)
    failed = [RunRecord(CASES[0].case_id, 1, "error", detail="quota")]
    report = eval_live.format_report(CASES[:1], failed, meta)
    assert "error=1" in report and "Read with care" in report


# ---------------------------------------------------------------------------
# Scoring one run
# ---------------------------------------------------------------------------

DEMO = next(c for c in CASES if c.case_id == "demo")
DIRECT_CASE = next(c for c in CASES if c.case_id == "direct_sales")
FOCUS_CASE = next(c for c in CASES if c.case_id == "focus_tenure")
UNGOVERNED = next(c for c in CASES if c.case_id == "ungoverned_industry")


def test_scope_is_scored_against_what_the_case_expects(db):
    scoped = _run(db, scoped=True)
    unscoped = _run(db, scoped=False)
    assert score_run(DIRECT_CASE, 1, scoped).scope_correct is True
    assert score_run(DIRECT_CASE, 1, unscoped).scope_correct is False  # forgot the restriction
    assert score_run(DEMO, 1, unscoped).scope_correct is True
    over_eager = score_run(DEMO, 1, scoped)
    assert over_eager.scope_correct is False and over_eager.produced_scope["department"] == ["deployment_channel=direct_sales"]


def test_unapplied_qualifier_and_focus_are_scored(db):
    said_so = _run(db, scoped=False, unapplied_qualifiers=["healthcare"])
    silent = _run(db, scoped=False)
    assert score_run(UNGOVERNED, 1, said_so).unapplied_ok is True
    assert score_run(UNGOVERNED, 1, silent).unapplied_ok is False
    assert score_run(DEMO, 1, said_so).unapplied_ok is False  # claimed a qualifier the question never had

    focused = _run(db, scoped=False, focus_attributes=["tenure_years"])
    assert score_run(FOCUS_CASE, 1, focused).focus_ok is True
    assert score_run(FOCUS_CASE, 1, silent).focus_ok is False


def test_gate_outcome_is_recorded_including_suppression(db):
    passed = score_run(DEMO, 1, _run(db, scoped=False))
    assert passed.gate_passed and passed.first_attempt_pass and passed.narrative_attempts == 1
    fabricated = {
        "department_summary": {"headline": "H.", "narrative": "N.", "top_attributes": ["tenure_years"]},
        "enterprise_summary": {"headline": "H.", "narrative": "N.", "top_attributes": ["module_count"]},
        "reconciliation_memo": "M.",
        "cited_metrics": [{"label": "invented", "lens": "department", "value": 99999.9}],
    }
    suppressed = score_run(DEMO, 1, _run(db, scoped=False, deliverable=fabricated))
    assert suppressed.gate_passed is False and suppressed.first_attempt_pass is False


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _validation_error() -> ValidationError:
    try:
        PlannedQuery.model_validate({"lens": "nope"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


def test_every_failure_is_classified_and_none_stops_the_run():
    outcome = ReconciliationOutcome(passed=False, tolerance=0.01, rows_checked=1)
    failures = iter(
        [
            ReconciliationError(outcome, "department", "2025-Q4"),
            GovernanceViolation("refused"),
            EmptySegmentError("empty"),
            ContextResolutionError("no contract"),
            _validation_error(),
            RuntimeError("quota exceeded"),
        ]
    )

    def runner(_question: str):
        raise next(failures)

    records = run_case(DEMO, 6, runner)
    assert [r.outcome for r in records] == [
        "reconciliation_error",
        "validator_refused",
        "empty_segment",
        "context_refused",
        "planner_schema_error",
        "error",
    ]
    assert "RuntimeError: quota exceeded" in records[-1].detail
    assert all(r.scope_correct is None and r.gate_passed is None for r in records)


# ---------------------------------------------------------------------------
# The eval set itself, and the command line
# ---------------------------------------------------------------------------


def test_eval_set_is_consistent_with_the_governed_vocabulary():
    """An expected filter the contract would refuse, or a focus it does not profile, would make a correct model look wrong."""
    allowed = DEPARTMENT_CONTRACT.dimension_values()
    assert len({c.case_id for c in CASES}) == len(CASES)
    for case in CASES:
        for column, values in case.expected_scope.items():
            assert column in allowed and values <= allowed[column], case.case_id
        assert case.expected_focus <= set(DEPARTMENT_CONTRACT.profile_attributes), case.case_id
    assert DEMO.expected_scope == {} and not DEMO.expects_unapplied


def test_script_spends_nothing_without_a_key_or_without_yes(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise AssertionError("the pipeline must not be called")

    monkeypatch.setattr(eval_live, "run_dual_lens_pipeline", boom)
    assert eval_live.main(["--runs", "2"]) == 2
    assert "GOOGLE_API_KEY is not set" in capsys.readouterr().out

    monkeypatch.setenv("GOOGLE_API_KEY", "not-a-real-key")
    assert eval_live.main(["--runs", "2"]) == 0
    printed = capsys.readouterr().out
    assert "Nothing has been sent" in printed and f"{len(CASES) * 2} pipeline run(s)" in printed


def test_script_rejects_bad_arguments(capsys):
    assert eval_live.main(["--cases", "nope"]) == 2
    assert "Unknown case id" in capsys.readouterr().out
    assert eval_live.main(["--runs", "0"]) == 2


def test_script_writes_results_with_an_injected_runner(db, tmp_path, capsys):
    good = _run(db, scoped=True)
    out = tmp_path / "results.json"
    code = eval_live.main(["--cases", "direct_sales", "--runs", "2", "--out", str(out), "--yes"], runner=lambda _q: good)
    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["meta"]["runs_per_case"] == 2 and len(saved["records"]) == 2
    assert all(r["scope_correct"] is True for r in saved["records"])
    assert "scope correct     : 2/2" in capsys.readouterr().out
