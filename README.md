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
   contract it cites, and it is rejected at the type boundary.
4. **A governed DuckDB tool validates before it executes.** Grain discipline (ratios divide by `COUNT(DISTINCT entity_id)`), governed
   thresholds only, no cross-grain joins without intermediate aggregation. Every call writes a typed audit record.
5. **Salience is arithmetic, computed per lens.** For each lens, every profile attribute is scored against that lens's own baseline.
   Numeric attributes use Cohen's d; categorical attributes use the percentage-point delta at each level. The model receives a ranked
   list it did not compute and cannot alter.
6. **A typed agent writes the narrative, and a gate checks it on the way out.** A PydanticAI agent produces a summary per lens, a
   reconciliation memo explaining why the lenses diverge, and a typed `cited_metrics` list. The zero-token-math gate verifies every
   cited value against the executed results. A failure is sent back to the model as a retry with the failing figures named. If retries
   run out, the narrative is suppressed and the deterministic tables stand on their own.

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
  `CitedMetric(label, lens, value: float)`. The zero-token-math gate compares typed floats against the union of executed result cells,
  row counts, salience statistics and governed thresholds. Because it is a comparison of numbers rather than a regular expression over
  text, a year, a rank or a version number in the prose can never trigger a false positive, and a metric can never hide in the prose
  unchecked. **The type system is what makes the gate reliable.**
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
| `semantic_contracts.py` | Versioned contracts, plan models, `ContextResolutionError`. The deterministic contract layer. |
| `synthetic_data.py` | Seeded two-grain DuckDB dataset with a deliberate contrast between the lenses. |
| `langchain_context_chain.py` | LCEL chain: retrieval, prompt, structured planning, governed plan. |
| `governed_duckdb_tool.py` | Pre-execution validator, DuckDB tool, audit records, the zero-token-math gate. |
| `salience.py` | Per-lens attribute salience; Cohen's d for numeric, percentage-point delta for categorical. |
| `narrative_agent.py` | PydanticAI agent with the gate wired in as an output validator; fails closed. |
| `pipeline.py` | The one composition function both entry points call; returns a typed `PipelineRun`. |
| `executive_render.py` | HTML rendering of a run for the notebook. |
| `run_demo.py` | Prints every stage of one run in order. |
| `build_notebook.py` | Generates the Colab notebook with nbformat and validates it. |
| `governed_insights_langchain.ipynb` | The guided tour. Generated; do not hand-edit. |
| `tests/test_pipeline.py` | Twelve cases, models mocked at the boundary, no network, no key. |

## Quickstart

```bash
git clone https://github.com/RickPack/governed-insights-harness.git
cd governed-insights-harness
pip install -r requirements.txt

# Tests run with no network and no API key.
python -m pytest tests -q

# The live demo needs a Gemini key.
export GOOGLE_API_KEY="your-key"        # PowerShell: setx GOOGLE_API_KEY "your-key", then reopen the shell
python run_demo.py
```

Or open the notebook in Colab with the badge above and store the key as a Colab secret named `GOOGLE_API_KEY`.

The demo asks one question and prints, in order: the question, both resolved contracts, both generated SQL statements, both DuckDB
result sets, both salience rankings, the validator and gate outcomes, both lens summaries, and the reconciliation memo.

## What the tests prove

Twelve cases in `tests/test_pipeline.py`, all passing with no network access and no API key present. The planner is mocked with a
`FakeListChatModel` subclass whose `with_structured_output` returns a real Runnable, so LCEL composition with the pipe operator is
exercised for real. The narrative model is PydanticAI's `TestModel`. Nothing else is mocked.

- Both lenses resolve from a value question; an unrelated question raises `ContextResolutionError`.
- A contract with a grain error, a blank metric or a non-semantic version cannot be constructed.
- The chain composes and parses structured output; malformed model output fails at the Pydantic boundary.
- The validator rejects raw-row-count denominators, ungoverned thresholds and unaggregated cross-grain joins, and accepts an aggregated join.
- The gate catches an unverifiable figure and passes a clean deliverable.
- Salience against a hand-built fixture uses Cohen's d for numeric and percentage points for categorical attributes, and ranks by absolute effect.
- The end-to-end path produces both summaries, a memo, a passing gate and a complete audit trail.

## Design decisions worth knowing

- **Organization platform revenue is materialised on the organization table.** Each contract reads one table at one grain. A reviewer
  can reproduce either lens with plain SQL, and the cross-grain join rule has a clean definition of "foreign table".
- **Categorical salience carries Cohen's h alongside the percentage-point delta.** The delta is what a reader understands. Cohen's h is
  the arcsine-transformed proportion difference, on the same standardised scale as Cohen's d, so a category level and a numeric
  attribute can be ranked in one list.
- **The gate's fact base includes governed thresholds.** A narrative that mentions the 5,000.0 floor is citing a versioned contract,
  not inventing a number. Thresholds are deterministic facts and belong in the base.
- **Fail closed, twice.** An unmatched question is refused. A narrative that cannot be verified is suppressed. In both cases the
  system prefers to say less rather than to say something it cannot trace.

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
