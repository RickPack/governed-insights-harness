"""
build_notebook.py — generates governed_insights_langchain.ipynb with nbformat.

WHY THIS DESIGN
---------------
Notebook JSON is easy to corrupt by hand: one unescaped quote or newline and
the file silently fails to open in Colab. Building the notebook with
nbformat's constructors and validating it before writing means the artifact
in the repository is always well-formed, and the notebook's content lives
here as reviewable Python rather than as escaped strings in JSON.

The notebook is a guided tour, not a script dump. Each markdown cell poses
the question the next code cell answers, in this order: the ambiguity problem,
how each lens computes its own answer, the two answers side by side, why they
differ, and proof that every number is real.

This script writes the notebook with empty outputs. The committed copy is
then executed in place (jupyter nbconvert --execute --inplace) so a reader on
GitHub sees real numbers without opening Colab, and a dated note is inserted
saying which run produced them. Regenerating with this script clears those
outputs; re-execute before committing.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path(__file__).with_name("governed_insights_langchain.ipynb")
REPO_URL = "https://github.com/RickPack/governed-insights-harness"

# Pinned versions mirror requirements.txt. Kept explicit here so the notebook
# is self-contained when opened directly in Colab.
PINS = (
    "langchain==1.4.1 langchain-core==1.6.3 langchain-google-genai==4.4.0 google-genai==2.24.0 "
    "pydantic-ai==2.45.0 pydantic==2.13.4 duckdb==1.5.5 pandas==2.3.3 tabulate==0.10.0 nest_asyncio==1.6.0"
)


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(text.strip("\n"))


CELLS = [
    # ------------------------------------------------------------------ 0
    md(f"""
# Governed Insights Harness — LangChain + PydanticAI

**Author:** Rick Pack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({REPO_URL.replace("https://github.com/", "https://colab.research.google.com/github/")}/blob/main/governed_insights_langchain.ipynb)

A probabilistic model sits between a business question and a revenue answer. This notebook shows one way to make that safe:
every boundary the model touches is a typed Pydantic contract, the model plans and writes but never computes, and every figure
in the final narrative is traced back to an executed query before anyone reads it.

Two frameworks share the work, on purpose:

- **LangChain (LCEL)** orchestrates the deterministic part: contract retrieval, prompt templating, structured query planning for two lenses.
- **PydanticAI** runs the one stage that needs a typed agent loop with validation and retry: turning verified results into an executive narrative.
- **Plain Pydantic models** are the contracts both frameworks carry. The framework is an implementation choice. The governance layer is not.

All data is synthetic. Runtime for the whole notebook is about two minutes.
"""),
    # ------------------------------------------------------------------ 1
    md("""
## Setup

Pinned packages and a clone of the repository so the notebook imports the same modules a reviewer can read on GitHub.
"""),
    code(f"""
%pip install -q {PINS}

import os, sys, subprocess, pathlib

# Keep PydanticAI's start-up banner out of the notebook output.
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

# Jupyter/Colab kernels already run an asyncio event loop. PydanticAI's
# synchronous agent call needs one of its own, which raises "event loop is
# already running" unless the loop is patched to allow nesting.
import nest_asyncio
nest_asyncio.apply()

REPO_DIR = pathlib.Path("governed-insights-harness")
if not REPO_DIR.exists() and not pathlib.Path("semantic_contracts.py").exists():
    subprocess.run(["git", "clone", "--quiet", "{REPO_URL}", str(REPO_DIR)], check=True)
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR.resolve()))

import logging
logging.getLogger("google_genai").setLevel(logging.ERROR)  # quiet an advisory log line from the SDK
print("Ready.")
"""),
    # ------------------------------------------------------------------ 2
    md("""
The pipeline makes exactly two model calls: one to plan queries, one to write the narrative. Both use Gemini. Store your key as a Colab secret named `GOOGLE_API_KEY`, or paste it when prompted.
"""),
    code("""
import os
from getpass import getpass

if not os.environ.get("GOOGLE_API_KEY"):
    try:
        from google.colab import userdata  # type: ignore
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    except Exception:
        os.environ["GOOGLE_API_KEY"] = getpass("Gemini API key: ")
print("Key present:", bool(os.environ.get("GOOGLE_API_KEY")))
"""),
    # ------------------------------------------------------------------ 3
    md("""
## 1. The ambiguity problem

A VP of Product asks a simple question in a quarterly review: *which of our enterprise accounts are our highest-value customers,
and what do they look like?*

Two teams answer. The first is measured on direct seat-license volume, so it pulls every organization with active licenses at or
above the floor and profiles those. Its answer: massive legacy accounts, mostly financial or industrial giants, using a single
core module heavily.

The second team owns consolidated platform usage. It pulls every parent organization whose combined workflow volume across API
calls, analytics modules, and secondary seats clears a different floor. Its answer: mid-market tech companies, holding eight or
nine integrated products, with no single module dominating.

Both teams are right. The populations barely overlap. And the meeting spends its remaining time arguing over which answer to
believe, because the question never specified which definition of value it meant, and nothing in either team's tooling recorded
which definition was used.

This is not a data quality problem. It is a **semantic governance** problem, and it is exactly the kind of ambiguity a
conversational analytics layer will hit on its first day in production. A model asked "who are our best customers" will pick a
definition. It will pick fluently. It will not tell you it picked.

Here is a synthetic dataset built to make that divergence visible. Two linked tables, one per grain, with a deliberate contrast:
some organizations hold one large legacy license and little else; others hold many mid-sized integrated products and no single
large license.
"""),
    code("""
from synthetic_data import build_database

conn, dataset = build_database(seed=42)
display(conn.execute("SELECT * FROM licenses ORDER BY license_revenue DESC LIMIT 5").df())
display(conn.execute("SELECT * FROM organizations ORDER BY platform_revenue DESC LIMIT 5").df())
"""),
    # ------------------------------------------------------------------ 4
    md("""
## 2. Two contracts, one question

Rather than choosing a definition, the system holds both as **versioned Pydantic contracts**. Each states its grain, its metric,
its source table and the only numeric cutoffs a query may use. The retriever matches the question to both, or refuses.

What does a contract look like when the model reads it? And what happens if someone tries to author a contract with a grain error?
"""),
    code("""
from pydantic import ValidationError
from langchain_context_chain import resolve_context
from semantic_contracts import DEPARTMENT_CONTRACT, MetricContract, ContextResolutionError

QUESTION = ("Compare the profile of our highest-value enterprise accounts, "
            "evaluated at the license level versus the parent organization level.")

context = resolve_context(QUESTION)
print(context.department_contract.describe_for_prompt(), end="\\n\\n")
print(context.enterprise_contract.describe_for_prompt(), end="\\n\\n")

# A contract with an organization key on a license grain cannot be instantiated. Invalid definitions are unconstructable.
try:
    MetricContract.model_validate({**DEPARTMENT_CONTRACT.model_dump(), "entity_id_column": "organization_id"})
except ValidationError as exc:
    print("Rejected at construction:", exc.errors()[0]["msg"])

# A question that matches no contract is refused rather than guessed.
try:
    resolve_context("What was the platform's uptime last quarter?")
except ContextResolutionError as exc:
    print("Refused:", exc)
"""),
    # ------------------------------------------------------------------ 5
    md("""
## 3. The planner sees both contracts, labelled

This is the LCEL chain. The question and the resolved context flow into a prompt that presents both contracts under explicit lens
labels, and the model is bound to a `DualLensPlan` schema: one query plan per lens, in one structured object. It cannot blend the
grains within a single query because each plan is tied to one contract, and the final `GovernedPlan` validator checks that tie.

What SQL does the model propose for each lens?
"""),
    code("""
from langchain_context_chain import build_planning_chain, default_chat_model

chain = build_planning_chain(default_chat_model())
governed_plan = chain.invoke(QUESTION)

for lens in ("department", "enterprise"):
    plan = governed_plan.plan.plan_for(lens)
    print(f"[{lens.upper()} LENS] cites {plan.contract_id} v{plan.contract_version}, threshold {plan.threshold_name}")
    print("  ", plan.sql)
    print("   rationale:", plan.rationale, end="\\n\\n")
"""),
    # ------------------------------------------------------------------ 6
    md("""
## 4. Deterministic execution behind a governed tool

The plans now pass through a LangChain tool that validates before it executes. Four rules run on every call: the plan reads its own
contract's table; any ratio divides by `COUNT(DISTINCT entity_id)`; every numeric cutoff matches a governed threshold; and any reach
across grains is aggregated first. DuckDB then produces the only numbers this notebook will ever show.

Both lenses execute below. Then a deliberately ungoverned plan is submitted to show what a refusal looks like.
"""),
    code("""
from governed_duckdb_tool import GovernedDuckDBTool, GovernanceViolation

tool = GovernedDuckDBTool(conn, governed_plan.context)
department_result = tool.tool.invoke({"plan": governed_plan.plan.department.model_dump()})
enterprise_result = tool.tool.invoke({"plan": governed_plan.plan.enterprise.model_dump()})

print(f"Department lens: {department_result.row_count} licenses")
display(department_result.table.to_dataframe().sort_values("metric_value", ascending=False).head(6))
print(f"Enterprise lens: {enterprise_result.row_count} organizations")
display(enterprise_result.table.to_dataframe().sort_values("metric_value", ascending=False).head(6))

# A cutoff nobody governed. Syntactically fine; refused before execution and recorded in the audit trail.
rogue = governed_plan.plan.department.model_copy(update={"sql": "SELECT license_id FROM licenses WHERE license_revenue >= 7500"})
try:
    tool.execute(rogue)
except GovernanceViolation as exc:
    print("\\nRefused:", exc)
"""),
    # ------------------------------------------------------------------ 7
    md("""
## 5. What is actually interesting about each segment?

Computing every attribute of a segment is easy. The harder question is which attributes *differentiate* it from its baseline.
Salience is computed deterministically and separately for each lens, against that lens's own grain. Numeric attributes use Cohen's d;
categorical attributes use the percentage-point delta at each level. Numeric distance math never touches a category code. The one-line
"most distinctive quality" sentence above each table is also computed, not written: it is the top-ranked score rendered as prose.

The model has not been involved yet. These rankings are pure SQL and Python.
"""),
    code("""
import pandas as pd
from salience import compute_salience

department_salience = compute_salience(conn, context.department_contract, governed_plan.plan.department.sql)
enterprise_salience = compute_salience(conn, context.enterprise_contract, governed_plan.plan.enterprise.sql)

def salience_frame(ranking, n=6):
    rows = [{"attribute": s.display_name, "segment": s.segment_value, "baseline": s.baseline_value,
             "effect": s.effect_size, "direction": s.direction, "method": s.method} for s in ranking.top(n)]
    return pd.DataFrame(rows)

print(f"Department lens: segment {department_salience.segment_size} of {department_salience.baseline_size} licenses")
print(" ", department_salience.lead_insight())
display(salience_frame(department_salience))
print(f"Enterprise lens: segment {enterprise_salience.segment_size} of {enterprise_salience.baseline_size} organizations")
print(" ", enterprise_salience.lead_insight())
display(salience_frame(enterprise_salience))
"""),
    # ------------------------------------------------------------------ 8
    md("""
## 6. Synthesis, with a gate on the way out

Now the second model call, and the only place PydanticAI is used. The agent receives both verified result sets and both salience
rankings, and must return an `ExecutiveDeliverable`: a summary per lens, a reconciliation memo, and a typed `cited_metrics` list.

The **zero-token-math gate** runs as the agent's output validator. Every value in `cited_metrics` is checked, for the lens it
claims, against a fact base holding exactly the figures the brief showed the model: segment sizes, salience statistics and the
governed thresholds. A second pass confirms every formatted figure in the prose is in that typed list. A figure that fails either
check is sent back to the model as a retry with the failing values named. If retries run out, the narrative is suppressed and the
deterministic results stand on their own.
"""),
    code("""
from governed_duckdb_tool import build_fact_base
from narrative_agent import NarrativeBrief, synthesize_narrative

# The fact base is exactly what the brief shows the model, per lens: segment sizes,
# salience statistics, and the governed thresholds. Nothing the model was not shown is in it.
fact_base = build_fact_base(context, department_result, enterprise_result, department_salience, enterprise_salience)
brief = NarrativeBrief(context=context, department_result=department_result, enterprise_result=enterprise_result,
                       department_salience=department_salience, enterprise_salience=enterprise_salience)

narrative = synthesize_narrative(brief, fact_base)
print(narrative.gate.detail)
print(f"Model outputs validated before acceptance: {narrative.attempts}")
if narrative.deliverable:
    print("\\nReconciliation memo:\\n" + narrative.deliverable.reconciliation_memo)
"""),
    # ------------------------------------------------------------------ 9
    md("""
## 7. The two answers, side by side

Everything above assembled into one typed `PipelineRun` and rendered for an executive reader: the finding, the two lenses, the memo
that reconciles them, the salience tables that justify the memo, and the governance proof underneath.
"""),
    code("""
from IPython.display import HTML
from pipeline import PipelineRun
from executive_render import render_executive_html

run = PipelineRun(
    query=QUESTION, dataset=dataset, governed_plan=governed_plan,
    department_result=department_result, enterprise_result=enterprise_result,
    department_salience=department_salience, enterprise_salience=enterprise_salience,
    audit_trail=tool.audit_trail, gate=narrative.gate, deliverable=narrative.deliverable,
    narrative_suppressed=narrative.suppressed, narrative_attempts=narrative.attempts,
)
HTML(render_executive_html(run))
"""),
    # ------------------------------------------------------------------ 10
    md("""
## 8. Proof that every number is real

The narrative above cites figures only through a typed list, so the gate's primary check is an exact comparison of numbers. To see
it work, take the verified deliverable and tamper with it twice: add a fabricated cited value, then write a figure into the prose
without citing it.
"""),
    code("""
from governed_duckdb_tool import CitedMetric, verify_cited_metrics

if narrative.deliverable:
    deliverable = narrative.deliverable
    print(verify_cited_metrics(deliverable.cited_metrics, fact_base, prose=deliverable.prose()).detail, end="\\n\\n")

    # Tamper one: a cited figure nothing computed.
    fabricated = list(deliverable.cited_metrics) + [
        CitedMetric(label="a figure the model made up", lens="enterprise", value=12345.6)
    ]
    print(verify_cited_metrics(fabricated, fact_base, prose=deliverable.prose()).detail, end="\\n\\n")

    # Tamper two: a figure written into the prose but left out of cited_metrics.
    smuggled_prose = deliverable.prose() + " Retention in this segment stands at 97.3 percent."
    print(verify_cited_metrics(deliverable.cited_metrics, fact_base, prose=smuggled_prose).detail)
"""),
    # ------------------------------------------------------------------ 11
    md("""
## What this demonstrates

Two teams gave two answers to one question. Neither was wrong. The failure was that the question had no governed definition, so the
disagreement surfaced in a meeting instead of in a memo. This pipeline turns that disagreement into a first-class output: both definitions
run, both are shown, and the model's job is to explain the gap, not to pick a winner.

The model never touched a number. Contracts are versioned in Git. Plans are validated before the database sees them. Salience is
arithmetic. The narrative's figures are typed and checked. When the gate fails, the prose disappears and the tables remain.

**What a team would build next:** a contract registry with an approval workflow, a third lens for the renewals and contract-value view,
salience over time rather than at a point, and a gate that also verifies directional claims ("higher than baseline") against the
salience ranking.
"""),
]


def build() -> None:
    notebook = new_notebook(cells=CELLS)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.metadata["colab"] = {"provenance": [], "name": "governed_insights_langchain.ipynb"}

    # Validate before writing: an invalid notebook never reaches disk.
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT.name}: {len(notebook.cells)} cells, nbformat validation passed.")


if __name__ == "__main__":
    build()
