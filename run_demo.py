"""
run_demo.py — the whole pipeline on one question, printed stage by stage.

WHY THIS DESIGN
---------------
A demo that prints only the final narrative asks the reader to trust it. This
one prints every intermediate artifact in the order it was produced: the
question, both resolved contracts, both generated SQL statements, both DuckDB
result sets, both salience rankings, the gate outcomes, both lens summaries,
and the reconciliation memo. A reviewer can stop at any stage and check the
next one by hand. That is the point of a governed pipeline: the trail is the
product, the prose is the summary of it.

Requires GOOGLE_API_KEY for the two live model calls (planning and
synthesis). Exits with a clear message if the key is absent; the test suite
covers the same path with mocked models.
"""

from __future__ import annotations

import os
import sys

from tabulate import tabulate

from pipeline import PipelineRun, run_dual_lens_pipeline
from salience import SalienceRanking
from semantic_contracts import ContextResolutionError, MetricContract
from governed_duckdb_tool import GovernanceViolation

DEMO_QUESTION = (
    "Compare the profile of our highest-value enterprise accounts, "
    "evaluated at the license level versus the parent organization level."
)

_RULE = "=" * 78


def money(value: float) -> str:
    """Thousands separators, one decimal place, no currency symbol (renders cleanly everywhere)."""
    return f"{value:,.1f}"


def _banner(title: str) -> None:
    print(f"\n{_RULE}\n  {title}\n{_RULE}")


def _print_contract(contract: MetricContract) -> None:
    print(f"[{contract.lens.upper()} LENS]  {contract.contract_id}  v{contract.version}")
    print(f"  grain      : {contract.entity_grain} (one row per {contract.entity_id_column})")
    print(f"  table      : {contract.source_table}")
    print(f"  metric     : {contract.metric_name} = {contract.metric_expression}")
    for t in contract.thresholds:
        print(f"  threshold  : {t.name} -> {t.column} {t.operator} {money(t.value)}")
    print(f"  meaning    : {contract.description}")
    print()


def _print_result(run: PipelineRun, lens: str, limit: int = 8) -> None:
    result = run.result_for(lens)
    frame = result.table.to_dataframe().sort_values("metric_value", ascending=False).head(limit)
    frame = frame.assign(metric_value=frame["metric_value"].map(money))
    print(f"[{lens.upper()} LENS]  {result.row_count} rows  (showing top {min(limit, result.row_count)} by metric)")
    print(tabulate(frame, headers="keys", tablefmt="simple", showindex=False))
    print()


def _print_salience(ranking: SalienceRanking, top_n: int = 6) -> None:
    rows = []
    for score in ranking.top(top_n):
        unit = "%" if score.method == "percentage_point_delta" else ""
        arrow = {"higher": "^ higher", "lower": "v lower", "similar": "= similar"}[score.direction]
        rows.append(
            [
                score.display_name,
                f"{money(score.segment_value)}{unit}",
                f"{money(score.baseline_value)}{unit}",
                f"{score.effect_size:+.2f}",
                arrow,
                "Cohen's d" if score.method == "cohens_d" else "pp delta",
            ]
        )
    print(
        f"[{ranking.lens.upper()} LENS]  segment {ranking.segment_size} of {ranking.baseline_size} "
        f"(contract {ranking.contract_id} v{ranking.contract_version})"
    )
    print(tabulate(rows, headers=["attribute", "segment", "baseline", "effect", "direction", "method"], tablefmt="simple"))
    print()


def print_run(run: PipelineRun) -> None:
    """Render a completed run in the order the pipeline produced it."""
    _banner("1. INPUT QUESTION")
    print(run.query)
    print()
    print(run.dataset.render())

    _banner("2. RESOLVED CONTRACTS (both lenses, retrieved deterministically)")
    _print_contract(run.governed_plan.context.department_contract)
    _print_contract(run.governed_plan.context.enterprise_contract)

    _banner("3. GENERATED SQL (structured output, validated before execution)")
    for lens in ("department", "enterprise"):
        plan = run.governed_plan.plan.plan_for(lens)
        print(f"[{lens.upper()} LENS]  threshold={plan.threshold_name}")
        print(f"  {plan.sql}")
        print(f"  rationale: {plan.rationale}\n")

    _banner("4. DUCKDB RESULTS (deterministic execution)")
    _print_result(run, "department")
    _print_result(run, "enterprise")

    _banner("5. SALIENCE RANKINGS (what differentiates each segment from its own baseline)")
    _print_salience(run.department_salience)
    _print_salience(run.enterprise_salience)

    _banner("6. GATE OUTCOMES")
    for record in run.audit_trail:
        status = "executed" if record.executed else "REFUSED"
        checks = ", ".join(f"{c.name}={'pass' if c.passed else 'FAIL'}" for c in record.checks)
        print(f"[{record.lens.upper()} LENS]  {status}  rows={record.row_count}  {checks}")
    print(f"\n{run.gate.detail}")
    print(f"narrative attempts validated: {run.narrative_attempts}")

    if run.deliverable is None:
        _banner("7. NARRATIVE SUPPRESSED (fail-closed)")
        print("The narrative did not pass the zero-token-math gate. The deterministic result sets above stand on their own.")
        return

    d = run.deliverable
    _banner("7. LENS SUMMARIES")
    for lens, summary in (("department", d.department_summary), ("enterprise", d.enterprise_summary)):
        print(f"[{lens.upper()} LENS]  {summary.headline}")
        print(f"  {summary.narrative}")
        print(f"  top attributes: {', '.join(summary.top_attributes)}\n")

    _banner("8. RECONCILIATION MEMO")
    print(d.reconciliation_memo)
    print()
    print("Cited metrics (each verified against executed results):")
    for metric in d.cited_metrics:
        print(f"  - {metric.label} [{metric.lens}] = {money(metric.value)}")


def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set. The live demo needs it for the planning and synthesis calls.")
        print("Set it with: setx GOOGLE_API_KEY \"your-key\" (then reopen the shell), or export it on Unix.")
        print("The test suite (python -m pytest tests) runs the same pipeline with mocked models.")
        return 2

    try:
        run = run_dual_lens_pipeline(DEMO_QUESTION, verbose=False)
    except ContextResolutionError as exc:
        print(f"Context resolution refused the question: {exc}")
        return 1
    except GovernanceViolation as exc:
        print(f"Pre-execution validator refused a plan: {exc}")
        return 1

    print_run(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
