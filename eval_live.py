"""
eval_live.py — measure the live pipeline instead of asserting things about it.

WHY THIS EXISTS
---------------
The offline tests mock the model, so they prove the governance layer holds when
a model is wrong or right in a specific way. They cannot say how often a real
model is right. That is a measurement: run a fixed set of questions many times,
record what happened, and report rates with intervals.

What is measured, per question and overall:

  * Completion: how often planning, validation, execution and decomposition
    finish, and how the runs that do not finish fail (a schema error at the
    type boundary, a validator refusal, an empty segment, an infrastructure error).
  * Scope: among completed runs, whether the planner produced exactly the
    governed filters the question calls for, no filters when it calls for none
    (the demo question says "enterprise accounts" and must not become a tier
    filter), an unapplied qualifier when no dimension covers a restriction, and
    the focus attributes when the question names them.
  * Gate: how often the narrative passes the zero-token-math gate on the first
    attempt, eventually after retries, or is suppressed.
  * Cost: model calls and wall-clock time, as observed.

Proportions carry a 95% Wilson interval. With a few runs per question the
intervals are wide, and the report says so. Results describe one model, one
synthetic dataset and this question set; they are not a general claim.

SAFETY
------
The script spends API calls only when given --yes. Without it, it prints the
plan (questions, runs, and the range of model calls it would make) and exits.
The scoring and statistics are plain functions, tested offline in
tests/test_eval_live.py with the pipeline replaced by a stub.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from decomposition import ReconciliationError
from governed_duckdb_tool import GovernanceViolation
from pipeline import PipelineRun, run_dual_lens_pipeline
from salience import EmptySegmentError
from semantic_contracts import ContextResolutionError

Scope = dict[str, frozenset[str]]
Runner = Callable[[str], PipelineRun]

DEFAULT_RUNS = 5
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class EvalCase:
    """One question and what a correct plan for it looks like."""

    case_id: str
    question: str
    expected_scope: Scope = field(default_factory=dict)
    expects_unapplied: bool = False
    expected_focus: frozenset[str] = frozenset()
    why: str = ""


_BASE = "Compare the profile of our highest-value enterprise accounts, evaluated at the license level versus the parent organization level"

CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="demo",
        question=f"{_BASE}.",
        why="No restriction is asked for. 'Enterprise accounts' names the account base and must not become an enterprise-tier filter.",
    ),
    EvalCase(
        case_id="direct_sales",
        question=f"{_BASE}, looking only at accounts sold through direct sales.",
        expected_scope={"deployment_channel": frozenset({"direct_sales"})},
        why="One governed restriction, stated in plain words.",
    ),
    EvalCase(
        case_id="partner_midmarket",
        question=f"{_BASE}, restricted to mid-market customers that came through the partner channel.",
        expected_scope={
            "org_tier": frozenset({"mid_market"}),
            "deployment_channel": frozenset({"partner_channel"}),
        },
        why="Two governed restrictions at once.",
    ),
    EvalCase(
        case_id="ungoverned_industry",
        question=f"{_BASE}, but only healthcare companies.",
        expects_unapplied=True,
        why="No governed dimension covers industry. The planner must say so instead of inventing a filter.",
    ),
    EvalCase(
        case_id="focus_tenure",
        question=f"{_BASE}. Focus on how long they have been customers.",
        expected_focus=frozenset({"tenure_years"}),
        why="Names an attribute to concentrate on; no population restriction.",
    ),
)

_OUTCOMES = (
    "ok",
    "planner_schema_error",
    "validator_refused",
    "reconciliation_error",
    "empty_segment",
    "context_refused",
    "error",
)


@dataclass
class RunRecord:
    """What one live run did. JSON-serialisable."""

    case_id: str
    repetition: int
    outcome: str
    detail: str = ""
    scope_correct: bool | None = None
    unapplied_ok: bool | None = None
    focus_ok: bool | None = None
    produced_scope: dict[str, list[str]] = field(default_factory=dict)
    gate_passed: bool | None = None
    first_attempt_pass: bool | None = None
    narrative_attempts: int | None = None
    llm_calls: int | None = None
    total_ms: float | None = None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion. (nan, nan) when there is no data."""
    if n <= 0:
        return (math.nan, math.nan)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(successes: int, n: int) -> dict:
    low, high = wilson_interval(successes, n)
    return {"successes": successes, "n": n, "rate": (successes / n) if n else math.nan, "low": low, "high": high}


# ---------------------------------------------------------------------------
# Running and scoring
# ---------------------------------------------------------------------------


def _scope_of(plan) -> Scope:
    return {f.column: frozenset(f.values) for f in plan.dimension_filters}


def score_run(case: EvalCase, repetition: int, run: PipelineRun) -> RunRecord:
    """Score a completed run against what the case expects."""
    department, enterprise = run.governed_plan.plan.department, run.governed_plan.plan.enterprise
    produced = {"department": _scope_of(department), "enterprise": _scope_of(enterprise)}
    scope_correct = all(scope == case.expected_scope for scope in produced.values())
    focus = {"department": frozenset(department.focus_attributes), "enterprise": frozenset(enterprise.focus_attributes)}
    return RunRecord(
        case_id=case.case_id,
        repetition=repetition,
        outcome="ok",
        scope_correct=scope_correct,
        unapplied_ok=bool(run.unapplied_qualifiers) == case.expects_unapplied,
        focus_ok=all(f == case.expected_focus for f in focus.values()),
        produced_scope={lens: sorted(f"{c}={'|'.join(sorted(v))}" for c, v in scope.items()) for lens, scope in produced.items()},
        gate_passed=run.gate.passed,
        first_attempt_pass=run.gate.passed and run.narrative_attempts == 1,
        narrative_attempts=run.narrative_attempts,
        llm_calls=run.telemetry.llm_calls,
        total_ms=run.telemetry.total_ms,
    )


def run_case(case: EvalCase, repetitions: int, runner: Runner) -> list[RunRecord]:
    """Run one case `repetitions` times, classifying every failure instead of stopping at it."""
    records: list[RunRecord] = []
    for repetition in range(1, repetitions + 1):
        started = time.perf_counter()
        try:
            records.append(score_run(case, repetition, runner(case.question)))
            continue
        except ReconciliationError as exc:
            outcome = "reconciliation_error"
            detail = str(exc)
        except GovernanceViolation as exc:
            outcome, detail = "validator_refused", str(exc)
        except EmptySegmentError as exc:
            outcome, detail = "empty_segment", str(exc)
        except ContextResolutionError as exc:
            outcome, detail = "context_refused", str(exc)
        except ValidationError as exc:
            outcome, detail = "planner_schema_error", exc.errors()[0]["msg"] if exc.errors() else str(exc)
        except Exception as exc:  # provider, network or quota errors: recorded, not hidden
            outcome, detail = "error", f"{type(exc).__name__}: {exc}"
        records.append(
            RunRecord(
                case_id=case.case_id,
                repetition=repetition,
                outcome=outcome,
                detail=detail[:300],
                total_ms=(time.perf_counter() - started) * 1000,
            )
        )
    return records


def summarize(records: list[RunRecord]) -> dict:
    """Rates with Wilson intervals, the outcome mix, and cost, for a set of records."""
    completed = [r for r in records if r.outcome == "ok"]
    gated = [r for r in completed if r.gate_passed is not None]
    times = [r.total_ms for r in completed if r.total_ms is not None]
    calls = [r.llm_calls for r in completed if r.llm_calls is not None]
    return {
        "runs": len(records),
        "outcomes": {name: sum(r.outcome == name for r in records) for name in _OUTCOMES},
        "completion": _rate(len(completed), len(records)),
        "scope_correct": _rate(sum(bool(r.scope_correct) for r in completed), len(completed)),
        "unapplied_correct": _rate(sum(bool(r.unapplied_ok) for r in completed), len(completed)),
        "focus_correct": _rate(sum(bool(r.focus_ok) for r in completed), len(completed)),
        "gate_first_attempt": _rate(sum(bool(r.first_attempt_pass) for r in gated), len(gated)),
        "gate_eventually": _rate(sum(bool(r.gate_passed) for r in gated), len(gated)),
        "suppressed": _rate(sum(not r.gate_passed for r in gated), len(gated)),
        "median_ms": statistics.median(times) if times else math.nan,
        "max_ms": max(times) if times else math.nan,
        "mean_llm_calls": statistics.fmean(calls) if calls else math.nan,
    }


def _fmt_rate(block: dict) -> str:
    if not block["n"]:
        return "n/a"
    return f"{block['successes']}/{block['n']} ({block['rate']:.0%}; 95% CI {block['low']:.0%}-{block['high']:.0%})"


def format_report(cases: tuple[EvalCase, ...], records: list[RunRecord], meta: dict) -> str:
    """A plain-text report. Ends with what the numbers do and do not show."""
    lines = [
        f"Live evaluation, {meta['finished_utc']}",
        f"  planner model {meta['planner_model']}; narrative model {meta['narrative_model']}",
        f"  {len(cases)} question(s) x {meta['runs_per_case']} run(s); synthetic data, seed {meta['seed']}",
        "",
    ]
    for case in cases:
        rows = [r for r in records if r.case_id == case.case_id]
        s = summarize(rows)
        mix = ", ".join(f"{k}={v}" for k, v in s["outcomes"].items() if v)
        lines += [
            f"[{case.case_id}] {case.why}",
            f"  outcomes          : {mix or 'none'}",
            f"  scope correct     : {_fmt_rate(s['scope_correct'])}",
            f"  unapplied correct : {_fmt_rate(s['unapplied_correct'])}",
            f"  focus correct     : {_fmt_rate(s['focus_correct'])}",
            f"  gate, 1st attempt : {_fmt_rate(s['gate_first_attempt'])}",
            f"  gate, eventually  : {_fmt_rate(s['gate_eventually'])}",
        ]
        wrong = Counter(json.dumps(r.produced_scope, sort_keys=True) for r in rows if r.outcome == "ok" and r.scope_correct is False)
        for shown, count in wrong.most_common(3):
            lines.append(f"  scope mismatch x{count}: {shown}")
        lines.append("")
    overall = summarize(records)
    lines += [
        "All questions",
        f"  completion        : {_fmt_rate(overall['completion'])}",
        f"  scope correct     : {_fmt_rate(overall['scope_correct'])}",
        f"  gate, 1st attempt : {_fmt_rate(overall['gate_first_attempt'])}",
        f"  gate, eventually  : {_fmt_rate(overall['gate_eventually'])}",
        f"  suppressed        : {_fmt_rate(overall['suppressed'])}",
        f"  model calls / run : {overall['mean_llm_calls']:.2f}   median wall-clock {overall['median_ms']:,.0f} ms"
        if overall["runs"] and not math.isnan(overall["mean_llm_calls"])
        else "  model calls / run : n/a",
        "",
        "Read with care: intervals are wide at this many runs, the data is synthetic, the question set is small and hand-written,",
        "and one model was used. Scope is scored only for runs that completed planning and execution.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _plan_text(cases: tuple[EvalCase, ...], runs: int) -> str:
    total = len(cases) * runs
    return (
        f"Would run {len(cases)} question(s) x {runs} run(s) = {total} pipeline run(s).\n"
        f"Each run makes 1 planner call plus 1 to 3 narrative calls, so about {total * 2} to {total * 4} model calls.\n"
        "Nothing has been sent. Re-run with --yes to spend those calls."
    )


def main(argv: list[str] | None = None, runner: Runner | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the live pipeline over a fixed question set.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="repetitions per question")
    parser.add_argument("--cases", nargs="*", help="case ids to run (default: all)")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON results")
    parser.add_argument("--yes", action="store_true", help="actually call the model; without it the plan is printed and nothing is sent")
    args = parser.parse_args(argv)

    known = {c.case_id: c for c in CASES}
    unknown = [c for c in (args.cases or []) if c not in known]
    if unknown:
        print(f"Unknown case id(s): {', '.join(unknown)}. Known: {', '.join(known)}.")
        return 2
    cases = tuple(known[c] for c in args.cases) if args.cases else CASES
    if args.runs < 1:
        print("--runs must be at least 1.")
        return 2

    if runner is None:
        if not os.environ.get("GOOGLE_API_KEY"):
            print("GOOGLE_API_KEY is not set. The live evaluation needs it; nothing was sent.")
            return 2
        if not args.yes:
            print(_plan_text(cases, args.runs))
            return 0
        runner = lambda question: run_dual_lens_pipeline(question, verbose=False)  # noqa: E731

    from langchain_context_chain import DEFAULT_MODEL_NAME
    from narrative_agent import DEFAULT_NARRATIVE_MODEL

    records: list[RunRecord] = []
    for case in cases:
        print(f"running {case.case_id} x {args.runs} ...", flush=True)
        records.extend(run_case(case, args.runs, runner))

    meta = {
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "planner_model": DEFAULT_MODEL_NAME,
        "narrative_model": DEFAULT_NARRATIVE_MODEL,
        "runs_per_case": args.runs,
        "seed": 42,
    }
    print()
    print(format_report(cases, records, meta))

    out = args.out or Path("eval_runs") / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "cases": [{"case_id": c.case_id, "question": c.question} for c in cases],
        "records": [asdict(r) for r in records],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
