"""
executive_render.py — the run as a VP would want to see it.

WHY THIS DESIGN
---------------
The pipeline produces typed artifacts; this module turns one PipelineRun into
a single self-contained HTML fragment for a notebook. Rendering lives here,
not in notebook cells, for the same reason orchestration lives in
pipeline.py: one implementation, readable in one pass, that the notebook
imports rather than reinvents.

Information hierarchy is deliberate. The eye lands on the key finding, then
on the two lenses side by side, then on the memo that reconciles them, then
on the salience tables that justify the memo, and only then on the governance
badge and the audit trail. Proof is available, but it does not compete with
the finding for attention.

No raw dollar signs or underscores appear in the rendered prose, so the output
displays the same way in Colab and in a GitHub-rendered notebook.
"""

from __future__ import annotations

from html import escape

from governed_duckdb_tool import AuditRecord
from pipeline import PipelineRun
from salience import SalienceRanking, SalienceScore

# One palette, used consistently: ink for text, two lens hues, a green for
# verified, an amber for direction "lower", and a muted grey for chrome.
_CSS = """
<style>
.gih { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1f2937; max-width: 1080px; line-height: 1.45; overflow-wrap: normal; word-break: normal; hyphens: none; }
.gih pre, .gih code { overflow-wrap: normal; word-break: normal; }
.gih h1 { font-size: 1.6rem; margin: 0 0 .25rem 0; letter-spacing: -.01em; }
.gih .sub { color: #6b7280; font-size: .95rem; margin-bottom: 1.25rem; }
.gih .callout { border-left: 5px solid #0f766e; background: #f0fdfa; padding: .9rem 1.1rem; margin: 1rem 0 1.5rem 0; border-radius: 4px; }
.gih .callout .label { text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; color: #0f766e; font-weight: 600; }
.gih .callout p { margin: .35rem 0 0 0; font-size: 1.05rem; }
.gih .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.gih .lens { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.1rem; }
.gih .lens.dept { border-top: 4px solid #1d4ed8; }
.gih .lens.ent { border-top: 4px solid #7c3aed; }
.gih .lens .tag { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
.gih .lens.dept .tag { color: #1d4ed8; } .gih .lens.ent .tag { color: #7c3aed; }
.gih .lens h3 { margin: .3rem 0 .5rem 0; font-size: 1.05rem; }
.gih .lens .meta { color: #6b7280; font-size: .85rem; margin-bottom: .6rem; }
.gih .lens .meta b { color: #374151; }
.gih .lens .insight { margin: .2rem 0 .6rem 0; padding: .5rem .7rem; background: #f9fafb; border-left: 3px solid #9ca3af; font-weight: 600; font-size: .95rem; }
.gih table.cmp { width: 100%; border-collapse: collapse; margin: 1.25rem 0; font-size: .92rem; }
.gih table.cmp th, .gih table.cmp td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid #e5e7eb; vertical-align: top; }
.gih table.cmp th { background: #f9fafb; font-weight: 600; color: #374151; }
.gih table.cmp td.k { color: #6b7280; width: 22%; }
.gih .memo { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 1rem 1.2rem; margin: 1.25rem 0; }
.gih .memo .label { text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; color: #b45309; font-weight: 700; }
.gih .memo p { margin: .4rem 0 0 0; }
.gih table.sal { width: 100%; border-collapse: collapse; font-size: .9rem; margin-top: .5rem; }
.gih table.sal th { text-align: left; font-weight: 600; color: #6b7280; font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; padding: .35rem .4rem; border-bottom: 1px solid #e5e7eb; }
.gih table.sal td { padding: .4rem .4rem; border-bottom: 1px solid #f3f4f6; }
.gih table.sal tr.rank1 td { font-weight: 700; font-size: 1.0rem; }
.gih table.sal tr.rank2 td { font-weight: 600; }
.gih .bar { display: inline-block; height: 10px; border-radius: 3px; vertical-align: middle; margin-right: .4rem; }
.gih .bar.higher { background: #0f766e; } .gih .bar.lower { background: #d97706; } .gih .bar.similar { background: #9ca3af; }
.gih .dir { font-size: .78rem; font-weight: 700; padding: .1rem .4rem; border-radius: 999px; }
.gih .dir.higher { background: #ccfbf1; color: #0f766e; } .gih .dir.lower { background: #fef3c7; color: #b45309; } .gih .dir.similar { background: #f3f4f6; color: #6b7280; }
.gih .method { color: #9ca3af; font-size: .75rem; }
.gih .badge { display: inline-flex; align-items: center; gap: .5rem; border-radius: 999px; padding: .4rem .9rem; font-weight: 600; font-size: .9rem; margin: 1.25rem 0 .5rem 0; }
.gih .badge.pass { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.gih .badge.fail { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.gih details { border: 1px solid #e5e7eb; border-radius: 8px; padding: .6rem 1rem; margin-top: .75rem; background: #fafafa; }
.gih summary { cursor: pointer; font-weight: 600; color: #374151; }
.gih pre { background: #111827; color: #e5e7eb; padding: .7rem .9rem; border-radius: 6px; font-size: .8rem; overflow-x: auto; white-space: pre-wrap; }
.gih .check { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem; }
.gih .check .ok { color: #166534; } .gih .check .bad { color: #991b1b; }
.gih .suppressed { background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0; }
</style>
"""


def money(value: float) -> str:
    """Thousands separators, one decimal place, no currency symbol."""
    return f"{value:,.1f}"


def _plain(text: str) -> str:
    """Escape HTML and replace underscores with spaces so attribute names read as words."""
    return escape(text.replace("_", " "))


def _lens_card(run: PipelineRun, lens: str) -> str:
    contract = run.governed_plan.context.contract_for(lens)
    ranking = run.salience_for(lens)
    summary = None if run.deliverable is None else (
        run.deliverable.department_summary if lens == "department" else run.deliverable.enterprise_summary
    )
    css = "dept" if lens == "department" else "ent"
    title = "Department lens" if lens == "department" else "Enterprise lens"
    headline = _plain(summary.headline) if summary else _plain(f"{ranking.segment_size} of {ranking.baseline_size} {contract.entity_grain}s selected")
    narrative = f"<p>{_plain(summary.narrative)}</p>" if summary else ""
    threshold = contract.thresholds[0]
    # The lead insight is computed, not written: it is the top salience score as a sentence.
    insight = f'<p class="insight">{escape(ranking.lead_insight())}</p>'
    return (
        f'<div class="lens {css}"><div class="tag">{title}</div>'
        f"<h3>{headline}</h3>"
        f'<div class="meta"><b>Grain:</b> {contract.entity_grain} &nbsp;·&nbsp; '
        f"<b>Metric:</b> {_plain(contract.metric_name)} &nbsp;·&nbsp; "
        f"<b>Floor:</b> {money(threshold.value)} &nbsp;·&nbsp; "
        f"<b>Contract:</b> {_plain(contract.contract_id)} v{contract.version}<br>"
        f"<b>Segment:</b> {ranking.segment_size} of {ranking.baseline_size} {contract.entity_grain}s</div>"
        f"{insight}{narrative}</div>"
    )


def _comparison_table(run: PipelineRun) -> str:
    d, e = run.governed_plan.context.department_contract, run.governed_plan.context.enterprise_contract
    ds, es = run.department_salience, run.enterprise_salience

    def top3(r: SalienceRanking) -> str:
        return ", ".join(_plain(s.display_name) for s in r.top(3))

    rows = [
        ("Entity grain", d.entity_grain, e.entity_grain),
        ("Metric", _plain(d.metric_name), _plain(e.metric_name)),
        ("Governed floor", money(d.thresholds[0].value), money(e.thresholds[0].value)),
        ("Contract version", f"{_plain(d.contract_id)} v{d.version}", f"{_plain(e.contract_id)} v{e.version}"),
        ("Segment size", f"{ds.segment_size} of {ds.baseline_size} licenses", f"{es.segment_size} of {es.baseline_size} organizations"),
        ("Most distinctive", top3(ds), top3(es)),
    ]
    body = "".join(f'<tr><td class="k">{k}</td><td>{a}</td><td>{b}</td></tr>' for k, a, b in rows)
    return (
        '<table class="cmp"><thead><tr><th></th><th style="color:#1d4ed8">Department lens</th>'
        '<th style="color:#7c3aed">Enterprise lens</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _salience_table(ranking: SalienceRanking, top_n: int = 8) -> str:
    scores = ranking.top(top_n)
    max_effect = max((abs(s.effect_size) for s in scores), default=1.0) or 1.0

    def row(i: int, s: SalienceScore) -> str:
        unit = "%" if s.method == "percentage_point_delta" else ""
        width = int(120 * abs(s.effect_size) / max_effect)
        arrow = {"higher": "▲ higher", "lower": "▼ lower", "similar": "≈ similar"}[s.direction]
        method = "Cohen's d" if s.method == "cohens_d" else "pp delta"
        rank_class = f"rank{i}" if i <= 2 else ""
        return (
            f'<tr class="{rank_class}"><td style="color:#9ca3af">{i}</td><td>{_plain(s.display_name)}</td>'
            f"<td>{money(s.segment_value)}{unit}</td><td>{money(s.baseline_value)}{unit}</td>"
            f'<td><span class="bar {s.direction}" style="width:{width}px"></span>{s.effect_size:+.2f}</td>'
            f'<td><span class="dir {s.direction}">{arrow}</span></td>'
            f'<td class="method">{method}</td></tr>'
        )

    title = "Department lens" if ranking.lens == "department" else "Enterprise lens"
    color = "#1d4ed8" if ranking.lens == "department" else "#7c3aed"
    body = "".join(row(i, s) for i, s in enumerate(scores, start=1))
    return (
        f'<div class="lens" style="border-top-color:{color}"><div class="tag" style="color:{color}">{title} · what makes this segment distinctive</div>'
        '<table class="sal"><thead><tr><th>#</th><th>Attribute</th><th>Segment</th><th>Baseline</th>'
        "<th>Effect size</th><th>Direction</th><th>Method</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _audit_record(record: AuditRecord) -> str:
    checks = " &nbsp; ".join(
        f'<span class="{"ok" if c.passed else "bad"}">{c.name}: {"pass" if c.passed else "FAIL"}</span>'
        for c in record.checks
    )
    status = "executed" if record.executed else "REFUSED"
    return (
        f"<p><b>{record.lens} lens</b> · {_plain(record.contract_id)} v{record.contract_version} · {status} · "
        f"{record.row_count} rows · {record.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>"
        f'<div class="check">{checks}</div><pre>{escape(record.sql)}</pre>'
    )


def render_executive_html(run: PipelineRun, title: str = "Highest-value enterprise accounts: two governed lenses") -> str:
    """Return the complete HTML fragment for one run. Pass to IPython.display.HTML."""
    context = run.governed_plan.context
    gate_class = "pass" if run.gate.passed else "fail"
    gate_text = (
        f"Zero-token-math gate passed · {run.gate.checked} cited figures verified · "
        f"{run.narrative_attempts} narrative attempt(s) validated"
        if run.gate.passed
        else f"Zero-token-math gate failed · narrative suppressed · {len(run.gate.unverifiable)} unverifiable figure(s)"
    )

    if run.deliverable is not None:
        finding = (
            f'<div class="callout"><div class="label">Key finding</div>'
            f"<p>{_plain(run.deliverable.department_summary.headline)}<br>"
            f"{_plain(run.deliverable.enterprise_summary.headline)}</p></div>"
        )
        memo = (
            f'<div class="memo"><div class="label">Reconciliation memo · why the two answers differ</div>'
            f"<p>{_plain(run.deliverable.reconciliation_memo)}</p></div>"
        )
    else:
        finding = (
            '<div class="suppressed"><b>Narrative suppressed.</b> The synthesis stage could not produce prose whose '
            "cited figures all trace to executed results. The deterministic result sets and salience rankings below "
            "stand on their own.</div>"
        )
        memo = ""

    cited = ""
    if run.deliverable is not None:
        items = "".join(
            f"<li>{_plain(m.label)} <span style='color:#6b7280'>[{m.lens}]</span> = {money(m.value)}</li>"
            for m in run.deliverable.cited_metrics
        )
        cited = f"<details><summary>Cited metrics checked by the gate ({len(run.deliverable.cited_metrics)})</summary><ul>{items}</ul></details>"

    audit = "".join(_audit_record(r) for r in run.audit_trail)

    return (
        _CSS
        + '<div class="gih">'
        + f"<h1>{escape(title)}</h1>"
        + f'<div class="sub">Question: {escape(context.query)}<br>{escape(run.dataset.render().splitlines()[0])}</div>'
        + finding
        + '<div class="grid">' + _lens_card(run, "department") + _lens_card(run, "enterprise") + "</div>"
        + _comparison_table(run)
        + memo
        + '<div class="grid">' + _salience_table(run.department_salience) + _salience_table(run.enterprise_salience) + "</div>"
        + f'<div class="badge {gate_class}">{"✔" if run.gate.passed else "✖"} {gate_text}</div>'
        + cited
        + f"<details><summary>Audit trail · {len(run.audit_trail)} governed tool call(s)</summary>{audit}</details>"
        + "</div>"
    )
