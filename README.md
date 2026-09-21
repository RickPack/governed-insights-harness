# Governed Insights Harness

**Author:** Rick Pack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RickPack/governed-insights-harness/blob/main/governed_insights_langchain.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL%20orchestration-1c3c3c.svg)](https://python.langchain.com/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-narrative%20synthesis-e92063.svg)](https://ai.pydantic.dev/)
[![DuckDB](https://img.shields.io/badge/DuckDB-deterministic%20execution-fff000.svg)](https://duckdb.org/)

> **Note on repository contributors:** GitHub's contributor graph for this repository has shown an entry labeled "BMTEORG." I do
> not know why this appears. BMTEORG is the domain for a youth mentorship non-profit of which I am a member, and it has no
> connection to this project's authorship or content.

A pattern for putting a language model between a business question and a revenue answer without letting the model touch a number.
Two competing definitions of the same term run side by side from versioned contracts, every boundary the model crosses is a typed
Pydantic model, and every figure in the final narrative is traced back to an executed query before anyone reads it.

All data is synthetic. The problem is a general enterprise analytics challenge, not a description of any particular organisation.

---

## The moment this is built for

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

## What this repository does about it

The harness treats the ambiguity as the deliverable rather than as noise to resolve.

1. **Both definitions exist as versioned Pydantic contracts.** A contract states its grain (license or organization), its metric
   expression, its source table, and the only numeric thresholds a query may apply. Changing a business definition is a Git diff
   with a version bump and a reviewer, not a prompt edit.
2. **The question resolves to both contracts, or is refused.** A deterministic retriever matches the question against the contract
   registry. If either lens has no matching contract, the pipeline raises `ContextResolutionError` rather than guessing.
3. **One structured planning call produces one query plan per lens.** The model sees both contracts explicitly labelled and must return
   a `DualLensPlan`. A plan that applies the organization threshold to the license table is a visible mismatch between the plan and the
   contract it cites, and it is rejected at the type boundary. The plan can also carry a governed restriction: "direct sales only"
   becomes a `deployment_channel` filter if, and only if, the contract lists that column and value.
4. **A governed DuckDB tool validates before it executes.** Grain discipline (ratios divide by `COUNT(DISTINCT entity_id)`), governed
   thresholds only, no cross-grain joins without intermediate aggregation, and a WHERE clause that is an AND of governed predicates,
   read from DuckDB's parsed syntax tree rather than from the SQL text. Every call writes a typed audit record.
5. **Salience is arithmetic, computed per lens.** For each lens, every profile attribute is scored against that lens's own baseline.
   Numeric attributes use Cohen's d; categorical attributes use the percentage-point delta at each level. The top-ranked attribute
   other than the selection metric becomes a one-sentence lead insight ("the most distinctive quality of this segment is module
   count: 8.9 vs 4.7 baseline"), rendered from the score, not written by the model. The model receives a ranked list it did not
   compute and cannot alter.
6. **A typed agent writes the narrative, and a gate checks it on the way out.** A PydanticAI agent produces a summary per lens, a
   reconciliation memo explaining why the lenses diverge, and a typed `cited_metrics` list. The zero-token-math gate verifies every
   cited value, for the lens it claims, against the figures the model was actually shown, then confirms every formatted figure in the
   prose is in that list. A failure is sent back to the model as a retry with the failing figures named. If retries run out, the
   narrative is suppressed and the deterministic tables stand on their own.

The VP gets both populations side by side, a memo that explains the gap in terms of grain, metric and threshold, a salience table per
lens showing what makes each population distinctive, and a governance badge saying every number was verified.

## Why Pydantic, not just prompts

This is the repository's thesis, and it is visible in every file: **Pydantic is the deterministic contract layer.**

A prompt can ask a model to respect a grain. A Pydantic validator can make a contract with the wrong grain impossible to construct.
Those are different guarantees. The harness relies on the second kind at every boundary the model touches:

- **Contracts are unconstructable when invalid.** `MetricContract` rejects a non-semantic version string, a blank metric expression, an
  entity key that is not an identifier column, and a key that does not match the declared grain. These are field and model validators,
  so the check runs at construction, not at query time.
- **Plans are validated before the database sees them.** `PlannedQuery` accepts only a single `SELECT`; `DualLensPlan` requires each
  plan to be labelled with its own lens; `GovernedPlan` requires each plan to cite the exact contract id and version the retriever
  resolved. A hallucinated citation stops here.
- **The narrative cites figures through a typed field, not prose.** `ExecutiveDeliverable.cited_metrics` is a list of
  `CitedMetric(label, lens, value: float)`. The zero-token-math gate compares each typed float, for the lens it claims, against a
  `FactBase` holding exactly the figures the model was shown for that lens: segment and baseline sizes, salience statistics and the
  governed thresholds. Raw result rows are deliberately excluded, because the model never sees rows, so a fabricated value that happens
  to equal some row's cell is still a fabrication and still fails. Matching is to one decimal place with no relative tolerance. Because
  the primary check is a comparison of typed numbers rather than a regular expression over text, a year, a rank or a version number in
  the prose can never trigger a false positive. A second, narrower pass then confirms every executive-formatted figure in the prose
  (thousands separators or decimals) appears in the typed list, so a number cannot route around the gate by being left out of it.
  **The type system is what makes the gate exact; the prose sweep is what makes it complete.**
- **Field descriptions are the model's instructions.** Every field on every model carries `Field(description=...)`. Those descriptions
  are what the model reads through structured output binding, so the schema and the prompt cannot drift apart.

No bare dictionaries cross a stage boundary. Contracts, resolved context, plans, execution results, salience scores, audit records,
the deliverable and the gate outcome are all Pydantic models.

## LangChain for orchestration, PydanticAI for synthesis

Two frameworks share the work. This is a deliberate division, not indecision.

**LangChain LCEL owns orchestration.** Contract retrieval, prompt templating, structured query planning for both lenses, tool binding
and chain composition are a fixed, inspectable sequence of steps. LCEL's pipe operator makes that sequence readable in one pass:

```python
{"query": RunnablePassthrough(), "context": context_retriever}
| RunnablePassthrough.assign(plan=RunnableLambda(_prompt_inputs) | prompt_template | planner)
| RunnableLambda(_bind_plan_to_context)
```

where `planner` is the chat model bound with `.with_structured_output(DualLensPlan)`. No LangGraph, no multi-agent orchestration.
A reader can trace the chain from question to governed plan without a diagram.

**PydanticAI owns narrative synthesis.** The final stage turns two verified result sets and two salience rankings into an executive
comparison and a reconciliation memo. That is constrained generation: the output must fit a typed model with separate fields per lens,
cite figures only in a typed list, and survive the gate. When it does not, the right response is to hand the validation error back to
the model and let it try again. A typed agent loop with output validation and automatic retry is PydanticAI's native strength, and it is
a genuine reason to keep it for this one stage rather than bolting a retry loop onto a chain by hand.

**Plain Pydantic models are what both frameworks carry.** The orchestration framework is an implementation choice. The governance layer
is not.

## Architecture

```
                         business question
                                |
                                v
  +---------------------------------------------------------------+
  |  LANGCHAIN LCEL                                                |
  |                                                                |
  |  context_retriever  (RunnableLambda, deterministic)            |
  |      matches question -> CONTRACT_REGISTRY                     |
  |      returns ResolvedContext  [PYDANTIC: both lenses or raise] |
  |                                |                               |
  |          +---------------------+---------------------+         |
  |          v                                           v         |
  |   department contract                       enterprise contract|
  |   (license grain, v1.2.0)                 (organization grain, v2.0.0)
  |          |                                           |         |
  |          +----------> prompt_template <--------------+         |
  |                       (both contracts, labelled)               |
  |                                |                               |
  |                                v                               |
  |                 planner.with_structured_output(DualLensPlan)   |
  |                       [PYDANTIC: one PlannedQuery per lens]    |
  |                                |                               |
  |                                v                               |
  |                       GovernedPlan                             |
  |                       [PYDANTIC: plans cite resolved contracts]|
  +---------------------------------------------------------------+
                                |
             +------------------+------------------+
             v                                     v
  +------------------------+           +------------------------+
  |  GOVERNED DUCKDB TOOL  |           |  GOVERNED DUCKDB TOOL  |
  |  department plan       |           |  enterprise plan       |
  |  validate:             |           |  validate:             |
  |   - source table       |           |   - source table       |
  |   - grain discipline   |           |   - grain discipline   |
  |   - governed thresholds|           |   - governed thresholds|
  |   - cross-grain joins  |           |   - cross-grain joins  |
  |  execute -> ExecutionResult        |  execute -> ExecutionResult
  |  [PYDANTIC + AuditRecord]          |  [PYDANTIC + AuditRecord]
  +------------------------+           +------------------------+
             |                                     |
             v                                     v
  +------------------------+           +------------------------+
  |  SALIENCE (SQL+Python) |           |  SALIENCE (SQL+Python) |
  |  vs. all licenses      |           |  vs. all organizations |
  |  numeric  -> Cohen's d |           |  numeric  -> Cohen's d |
  |  category -> pp delta  |           |  category -> pp delta  |
  |  -> SalienceRanking    |           |  -> SalienceRanking    |
  |  [PYDANTIC]            |           |  [PYDANTIC]            |
  +------------------------+           +------------------------+
             |                                     |
             +------------------+------------------+
                                v
  +---------------------------------------------------------------+
  |  PYDANTICAI AGENT                                              |
  |  NarrativeBrief in  [PYDANTIC]                                 |
  |  ExecutiveDeliverable out  [PYDANTIC]                          |
  |     department_summary | enterprise_summary                    |
  |     reconciliation_memo | cited_metrics: list[CitedMetric]     |
  |                                |                               |
  |                                v                               |
  |     ZERO-TOKEN-MATH GATE  (output validator)                   |
  |     every cited value  <-->  executed results + salience       |
  |     fail -> ModelRetry with failing figures named              |
  |     retries exhausted -> narrative suppressed, tables surfaced |
  +---------------------------------------------------------------+
                                |
                                v
              executive rendering + audit trail
```

Every bracket marked `[PYDANTIC]` is a boundary where a malformed object cannot pass. The model is involved at exactly two points:
producing `DualLensPlan` and producing `ExecutiveDeliverable`. Everything between and after is deterministic.

## Repository layout

| File | Role |
|---|---|
| `semantic_contracts.py` | Versioned, owned contracts, plan models, `ContextResolutionError`, and the metric-name collision check. The deterministic contract layer. |
| `synthetic_data.py` | Seeded two-grain DuckDB dataset with a deliberate contrast between the lenses, plus license snapshots and movement events for one period. |
| `decomposition.py` | Deterministic revenue movement per lens, reconciled by a fail-closed gate. No model involved. |
| `langchain_context_chain.py` | LCEL chain: retrieval, prompt, structured planning, governed plan. |
| `governed_duckdb_tool.py` | Pre-execution validator, DuckDB tool, audit records, the zero-token-math gate. |
| `sql_predicates.py` | Parse-based inspection of planned SQL: which predicates and tables a plan actually uses. Refuses OR, NOT, UNION, LIMIT and the like. |
| `eval_live.py` | Live evaluation: runs a fixed question set several times, scores scope and gate outcomes, reports rates with Wilson intervals. Spends nothing without `--yes`. |
| `tests/test_seed_sweep.py` | Forty-one cases (parametrised) checking twenty seeds: decompositions reconcile, segments are non-empty, the lenses stay different, and a seed reproduces every table. |
| `tests/test_eval_live.py` | Fifteen cases (parametrised) for the interval arithmetic, scoring, failure classification, the eval set's consistency with the contract vocabulary, and the no-spend guard. |
| `tests/test_planner_scope.py` | Thirty-one cases (parametrised) for governed, question-driven scoping: the allowlist, plan validation, refusals, scoped baseline and decomposition, an empty segment, and an end-to-end run. |
| `salience.py` | Per-lens attribute salience; Cohen's d for numeric, percentage-point delta for categorical. |
| `narrative_agent.py` | PydanticAI agent with the gate wired in as an output validator; fails closed. |
| `pipeline.py` | The one composition function both entry points call; returns a typed `PipelineRun`. |
| `executive_render.py` | HTML rendering of a run for the notebook. |
| `run_demo.py` | Prints every stage of one run in order. |
| `build_notebook.py` | Generates the Colab notebook with nbformat and validates it. |
| `execute_notebook.py` | Regenerates, executes, verifies and date-stamps the notebook so GitHub shows real outputs. |
| `governed_insights_langchain.ipynb` | The guided tour, committed with executed outputs. Generated; do not hand-edit. |
| `tests/test_pipeline.py` | Sixteen cases, models mocked at the boundary, no network, no key. |
| `tests/test_decomposition.py` | Ten cases (some run under both lenses) for classification, grain, reconciliation and the floor view. |
| `equivalence.py` | Paired equivalence check for a proposed floor change: Tango score interval, TOST decision, seeded power. Illustrative. |
| `tests/test_equivalence.py` | Twelve cases: the R test vector, orientation, decision rule, discordant counts, seeded power and size. |
| `tests/test_contracts.py` | Five cases for contract ownership and metric-name collisions. |
| `tests/test_sql_governance.py` | Twenty cases (parametrised) for the parse-based WHERE-clause check, including the filters that used to slip past the text rules. |

## Quickstart

```bash
git clone https://github.com/RickPack/governed-insights-harness.git
cd governed-insights-harness
pip install -r requirements.txt

# Tests run with no network and no API key. A GitHub Actions workflow runs the same command on every push and pull request.
python -m pytest tests -q

# The live demo needs a Gemini key.
export GOOGLE_API_KEY="your-key"        # PowerShell: setx GOOGLE_API_KEY "your-key", then reopen the shell
python run_demo.py

# Measure the live pipeline over a fixed question set. Prints the plan and sends nothing until you add --yes.
python eval_live.py --runs 5
python eval_live.py --runs 5 --yes

# Refresh the committed notebook's outputs (regenerate, execute, verify, date-stamp).
# The notebook execution packages are pinned in requirements.txt alongside everything else.
python execute_notebook.py
```

Or open the notebook in Colab with the badge above and store the key as a Colab secret named `GOOGLE_API_KEY`.

The demo asks one question and prints, in order: the question, both resolved contracts, both generated SQL statements, both DuckDB
result sets, both salience rankings, the reconciled revenue movement for each lens, the validator and gate outcomes with run telemetry, both lens summaries, and the reconciliation memo.

## Measuring the live pipeline

The offline tests mock the model, so they show the governance layer holds when a model is wrong or right in a specific way. They
cannot show how often a real model is right. `eval_live.py` measures that: five hand-written questions (an unrestricted one, two with
governed restrictions, one asking for a restriction no dimension covers, one naming an attribute to focus on), each run several times.
It reports how many runs complete and how the rest fail, how often the planner produced exactly the right scope (including none for the
demo question, whose words "enterprise accounts" must not become a tier filter), how often the narrative passes the gate on the first
attempt or is suppressed, and the model calls and wall-clock time observed. Proportions carry a 95% Wilson interval.

No results are published here. The script has been tested offline (interval arithmetic, scoring, failure classification, and a guard
that spends nothing without `--yes`), but I have not run it against the live model, and any figures would describe one model, one
synthetic dataset and a small question set at a small number of runs, so the intervals would be wide.

## Sample run

This is real output from `python run_demo.py`, captured 2026-09-21 against Gemini (`gemini-3.6-flash`), seed 42. The full captured
stdout, unedited, is in [SAMPLE_RUN.md](SAMPLE_RUN.md). The committed notebook carries
outputs from its own live run (also 2026-09-21, model output differs run to run; the deterministic numbers do not), date-stamped, so
the results are visible on GitHub without opening Colab.

| | Department lens | Enterprise lens |
|---|---|---|
| Grain | License (`license_id`) | Organization (`organization_id`) |
| Segment size | 21 of 119 licenses | 13 of 50 organizations |
| Governed floor | 5,000.0 | 25,000.0 |
| Segment mean metric | 8,789.5 | 30,638.3 |
| Baseline mean metric | 3,221.3 | 13,246.7 |
| Lead insight (computed, not written) | Tenure: 17.5 yrs vs 7.8 yrs baseline (Cohen's d +1.55) | Module count: 8.9 vs 4.7 baseline (Cohen's d +1.37) |
| Seat-license revenue movement, 2025-Q4 | Change 32,370.7 across 125 licenses (119 current plus 6 churned): new 46,863.3, expansion 14,996.2, contraction 9,020.4, churn 20,184.3, migration in 1,697.9, migration out 1,981.8. Reconciled. | Same change across 50 organizations; migrations between licenses of one organization net to zero, leaving migration in 395.2 and out 679.2. Reconciled. |
| Zero-token-math gate | Passed: 46 of 46 cited figures traced for their lens (both lenses together), prose fully cited, 1 attempt | Same run, same gate |

**Reconciliation memo (verbatim from the run):** "The department lens operates at the license grain, evaluating annual seat-license revenue against a threshold of 5,000 to isolate large individual contracts. In contrast, the enterprise lens operates at the parent organization grain, applying a threshold of 25,000 across total consolidated platform revenue, which incorporates API call volume and analytics modules alongside seat licenses. Consequently, the enterprise lens captures accounts that achieve high aggregate value through product breadth across multiple integrated modules, even when individual licenses remain below 5,000. Conversely, the department lens highlights legacy licenses with long tenure regardless of whether the parent entity holds additional modules. Neither view is wrong; they answer different questions for direct license management versus executive platform relationship oversight."

Numbers depend on the random seed and on what the model plans and writes, so a different run (or a different question) will not
reproduce these exactly. To generate a fresh run yourself, open the notebook in Colab using the badge at the top of this file, or
run `python run_demo.py` locally with `GOOGLE_API_KEY` set, as described in Quickstart above.

## What the tests prove

Sixteen cases in `tests/test_pipeline.py`, all passing with no network access and no API key present. The planner is mocked with a
`FakeListChatModel` subclass whose `with_structured_output` returns a real Runnable, so LCEL composition with the pipe operator is
exercised for real. The narrative model is PydanticAI's `TestModel`. Nothing else is mocked.

- Both lenses resolve from a value question; an unrelated question raises `ContextResolutionError`.
- A contract with a grain error, a blank metric or a non-semantic version cannot be constructed.
- The chain composes and parses structured output; malformed model output fails at the Pydantic boundary.
- The validator rejects raw-row-count denominators, ungoverned thresholds and unaggregated cross-grain joins, and accepts an aggregated join.
- The gate catches a fabricated figure and passes a clean deliverable whose prose and typed list agree.
- The gate rejects a real enterprise figure cited under the department lens, and rejects a fabricated value that merely equals some
  row's cell, because the fact base holds only what the brief showed the model.
- The gate's prose sweep catches a formatted figure written into the narrative but left out of `cited_metrics`, while ignoring plain
  integers, version strings and category ranges.
- Salience against a hand-built fixture uses Cohen's d for numeric and percentage points for categorical attributes, ranks by absolute
  effect, and renders the lead insight from the top score.
- The end-to-end path produces both summaries, a memo, a passing gate and a complete audit trail.
- A figure quoted from the revenue decomposition passes the same gate; a derived figure the table does not contain, or a figure cited under the wrong lens, fails it.
- Every audit record, including a refused plan, carries a duration; run telemetry reports the calls and stage timings it observed.

`tests/test_sql_governance.py` adds twenty more (parametrised) for the WHERE-clause check. It includes `WHERE floor OR 1 = 1` and `WHERE NOT (floor)`, which the earlier text-matching rules accepted and which return rows outside the segment; a test runs the widened SQL directly to show the floor is defeated, and another confirms the tool now refuses it and audits the refusal. The other three files add 27 cases, also offline. `tests/test_decomposition.py` covers a clean reconciliation, an injected discrepancy that fails closed, internal versus cross-organization migration, the grain invariant, contraction versus churn, a migration with a missing destination, and a period with no movement. `tests/test_contracts.py` covers owners and the collision check. `tests/test_equivalence.py` checks the Tango interval against a numeric vector from R's `PropCIs::scoreci.mp`, the orientation of the estimate, the decision rule, that discordant counts are always reported, and that the seeded Monte Carlo power and size are reproducible.

## Design decisions worth knowing

- **The contrast between the lenses is engineered, and checked beyond one seed.** The generator builds archetypes so the two
  definitions of "high value" disagree, which is the point of the demonstration. That would be a weak result if it were an accident of
  seed 42, so `tests/test_seed_sweep.py` checks twenty seeds: the organizations each lens selects overlap by less than half, each lens
  finds organizations the other misses, every decomposition reconciles and no segment is empty. In a 200-seed exploration outside the
  test suite the overlap ranged from 0.04 to 0.32. This shows the generator behaves as designed. It says nothing about real data,
  where the lenses may agree. The narrative prompt asks the model to explain why the lenses differ, and I have not tested how the
  pipeline behaves when they do not; that is a real gap.

- **What the planner does that the contract cannot.** For the demo question, nothing: the correct SQL follows from the contract, and
  the validator refuses anything else, so a model that is right reproduces the contract's SQL. An earlier version of this repository
  had exactly that weakness. The planner earns its place where a question restricts the population in words a contract cannot
  anticipate ("direct sales only", "mid-market customers", "focus on tenure"). Each contract lists the columns a question may restrict
  by and the only values allowed (`filterable_dimensions`). The model proposes filters; a type-boundary check refuses any column or
  value outside that list; the tool refuses SQL that applies anything other than the filters the plan declares; and the baseline,
  salience ranking and revenue decomposition are all computed over the same restricted population, so a "direct sales" segment is
  not reported as distinctively direct-sales by construction. A qualifier that no governed dimension covers ("startups") is not
  guessed at: it is listed in `unapplied_qualifiers`, shown to the reader and stated in the memo. Whether a live model fills these
  slots reliably is a measurement, not something the offline tests can show.

- **The validator reads the syntax tree, not the SQL text.** The first version matched the governed cutoff with regular expressions.
  Probing it showed the cutoff can be present in the text and defeated in effect: `WHERE license_revenue >= 5000 OR 1 = 1` returned all
  119 licenses instead of 21, and `NOT (license_revenue >= 5000)` returned the complement. `sql_predicates.py` now asks DuckDB's own
  parser for the tree and accepts a small subset: one plain SELECT, governed base tables, and a WHERE that is an AND of governed
  predicates. Anything else is refused with the reason. The cost is that a legitimate but unusual plan (a CTE, a `LIMIT`) is refused
  rather than guessed at; the planner prompt says to emit only the supported shape.

- **Organization platform revenue is materialised on the organization table.** Each contract reads one table at one grain. A reviewer
  can reproduce either lens with plain SQL, and the cross-grain join rule has a clean definition of "foreign table".
- **Categorical salience carries Cohen's h alongside the percentage-point delta.** The delta is what a reader understands. Cohen's h is
  the arcsine-transformed proportion difference, on the same standardised scale as Cohen's d, so a category level and a numeric
  attribute can be ranked in one list.
- **The gate's fact base is the brief, nothing more.** It holds, per lens, the segment and baseline sizes, the salience statistics
  and the governed thresholds: exactly what the model was shown. A narrative that mentions the 5,000.0 floor is citing a versioned
  contract, not inventing a number. Raw result rows are excluded because the model never saw them. Matching is to one decimal
  place with no relative tolerance, because one part in a thousand of a five-figure revenue is room for a fabricated number.
  Decomposition components that are arithmetically identical at both lenses — the total seat-license revenue pool is the same
  regardless of grain — pass the wrong-lens check by design; the check catches lens-specific figures like migration amounts.
- **The lead insight skips the selection metric.** A segment selected for high revenue has high revenue by construction. The
  sentence a product leader repeats is the attribute that differs *given* the selection: tenure for the license lens, module
  count for the organization lens.
- **The committed notebook is executed, and says so.** `execute_notebook.py` regenerates the notebook, runs every cell with a fresh
  kernel, refuses to proceed on any error, and inserts a dated note naming the model. A reader on GitHub sees real numbers and
  knows which run produced them.
- **Revenue movement is explained, then reconciled, then narrated.** `decomposition.py` splits the change in seat-license revenue
  into new, expansion, contraction, churn and migration, using classified events. Reported change comes from period snapshots;
  components come only from the events; no component is ever computed as a residual. If they disagree by 0.01 or more for any license
  or organization, the run stops with a typed `ReconciliationError` before any narration. The narrative can then quote only figures
  from that table, through the same gate as everything else.
  - *Scope:* seat-license revenue only. Organization platform revenue also holds API and analytics revenue that has no license, so the
    organization-lens decomposition explains the license part of the change and is labelled that way.
  - *Migrations:* a move between two licenses of one organization is internal and nets to zero at the organization lens; a move across
    organizations is external for both. A migration with a missing or unknown destination is treated as external and flagged, never
    silently internal.
  - *Floor interaction:* reconciliation runs on the full population. A floor-filtered view is shown only as a labelled subset,
    because a churned license ends at zero and so falls below any floor; a filtered view cannot reconcile to the full change. The
    governed floors are unchanged.
  - *Data:* one synthetic period (2025-Q4) drawn from a separate seeded generator, so the original tables are unchanged. Snapshots
    and events are independent inputs, which is what lets the gate fail. Real data would bring late events, restatements and
    multiple periods; none is modelled here.
- **Telemetry reports what happened, nothing more.** Each audit record has a duration, and each run reports the model calls made,
  the deterministic operations executed and the time per stage. No baseline was measured, so the repository makes no claim about
  savings.
- **Contracts have owners, and one name means one definition.** Each contract names the team accountable for it
  (`license-ops`, `account-strategy`). Ownership is metadata and changes no behaviour. A test fails if two contracts share a metric
  name but define it differently.
- **A floor change can be checked for equivalence before it ships.** `equivalence.py` asks whether a candidate definition (here, the
  department contract with a 5,500 floor, defined only in that module and not registered) puts the same licenses in the segment as
  the current one, within a margin. Each license is in or out under each definition, so the data are paired binary observations. It
  computes the Tango (1998) score interval for the difference of paired proportions and calls the definitions equivalent when the
  90% interval lies inside the margin, which is the two one-sided tests procedure. The primary margin is ±0.05, with ±0.02 as a
  sensitivity margin. The harness borrows the equivalence-testing idea from Lo et al. (2025); it does not reproduce or validate their
  method.
  - *Read the discordant counts.* Equivalent aggregate rates can hide large disagreement about individual licenses, so every result
    prints b (in the segment under the current definition only), c (candidate only) and n beside the interval.
  - *Orientation:* the estimate is candidate minus current, (c − b) / n. R's `scoreci.mp(x, y, n)` estimates (y − x) / n, so
    `tango_interval(b, c, n)` matches `scoreci.mp(x = b, y = c, n = n)`. The unit test pins this against a vector from R.
  - *Power and size:* a seeded Monte Carlo reports power at a true difference of zero for the observed sample size and discordance,
    with 0.80 as the bar for calling the check adequately powered, and checks that the chance of declaring equivalence at the
    margin boundary stays near the 0.05 target.
  - *Framing:* illustrative only. The synthetic data are fully enumerated, so there is no sampling and the interval describes a
    data-generating process. The check is designed for a monitoring sample of a real book, not for this dataset.
- **Fail closed, twice.** An unmatched question is refused. A narrative that cannot be verified is suppressed. In both cases the
  system prefers to say less rather than to say something it cannot trace.

## References

- Lo, V. S. Y., Datta, S., & Salami, Y. (2025). Bringing practical statistical science to AI and predictive model fairness testing.
  *AI and Ethics, 5*, 2149-2164. https://doi.org/10.1007/s43681-024-00518-2
- Tango, T. (1998). Equivalence test and confidence interval for the difference in proportions for the paired-sample design.
  *Statistics in Medicine, 17*(8), 891-908.
- Pack, R., Lo, V., & Yao, P. (in preparation). Fair mentor matching at enterprise scale: Gale-Shapley pairing, NLP, and prespecified
  statistical validation. Joint Statistical Meetings 2026.

## Future extensions

What a team would build next, in the order it would pay off:

1. **A contract registry with an approval workflow.** Contracts already live in Git; the next step is a pull-request template that
   requires a business owner's sign-off on any version bump, and a changelog rendered into the executive output.
2. **A third lens.** A renewals and contract-value view, which recognises revenue on a different calendar, would exercise the
   retriever's ability to resolve more than two contracts and the memo's ability to reconcile three.
3. **Directional claims under the gate.** Today the gate verifies figures. A second check could verify that every "higher than baseline"
   in the prose matches the sign of the corresponding salience score.
4. **Salience over time.** Point-in-time salience says what distinguishes a segment now. A monthly series would say what is changing.
5. **Contract-aware retrieval at scale.** The keyword matcher is right for a two-contract registry. A hundred-contract registry would
   want embedding retrieval, still with the same fail-closed rule when confidence is low.

## License

MIT. See `LICENSE`.
