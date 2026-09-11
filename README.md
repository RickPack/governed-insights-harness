# Governed Insights Harness (GIH)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RickPack/governed-insights-harness/blob/main/governed_insights_harness.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-Orchestration-darkgreen.svg)](https://github.com/pydantic/pydantic-ai)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory%20SQL-yellow.svg)](https://duckdb.org/)

A type-safe analytical execution harness demonstrating **semantic entity disambiguation** and **model-risk governance** for conversational analytics in enterprise environments.

---

## The Enterprise Challenge

Enterprise conversational BI and analytics agents frequently stall in regulated corporate environments due to two architectural pitfalls:

1. **Cross-Departmental Semantic Divergence:** Business units define identical business entities with conflicting grains and business rules. 
   * **The Department Lens:** Evaluates performance at the single account/contract level (`account_id`), prioritizing immediate fee generation (`fee_amount`).
   * **The Enterprise Lens:** Evaluates performance at the consolidated household relationship level (`household_party_id`), prioritizing lifetime relationship value (`total_revenue`).
   When asked open questions like *"Which customers generate the most revenue?"*, unconstrained agents conflate these entity grains and present conflicting or misleading executive metrics.

2. **Model-Risk Arithmetic Hallucinations:** Large language models predict next tokens rather than executing deterministic mathematics. Permitting models to perform mental calculations in narrative prose generates unverified figures that fail basic Model Risk Management (MRM) standards.

---

## Architectural Pattern

The harness constrains the model strictly to **query planning and contract discovery**. Business logic runs deterministically in an isolated database engine, and narratives pass through automated assertion gates before presentation.

```
                  ┌─────────────────────────────────┐
                  │        Executive Query          │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │    PydanticAI Query Planner     │
                  │   (Constrained Query Router)    │
                  └───────┬─────────────────┬───────┘
                          │                 │
             [Contract: Department]   [Contract: Enterprise]
                          │                 │
                          ▼                 ▼
                  ┌─────────────────────────────────┐
                  │   Deterministic Engine (DuckDB) │
                  │  (Executes Analytical Aggs/SQL) │
                  └────────────────┬────────────────┘
                                   │ Raw Query Result Sets
                                   ▼
                  ┌─────────────────────────────────┐
                  │    Automated Governance Gates   │
                  ├─────────────────────────────────┤
                  │ 1. Tool Coverage Audit          │
                  │ 2. Zero-Token-Math Traceability │
                  └───────┬─────────────────┬───────┘
                          │                 │
                    [Gates Pass]      [Gate Failure]
                          │                 │
                          ▼                 ▼
        ┌───────────────────────────┐     ┌───────────────────────────┐
        │ Verified Executive Output │     │   Fail-Closed Fallback    │
        ├───────────────────────────┤     ├───────────────────────────┤
        │ • Dual-Lens Comparative   │     │ • Suppress Narrative Memo │
        │   Profile Summary         │     │ • Alert Audit Trace Log   │
        │ • Semantic Reconciliation │     │ • Output Audited Raw      │
        │   Executive Memo          │     │   DataFrames Directly     │
        │ • Audited Metrics Trace   │     └───────────────────────────┘
        └───────────────────────────┘
```

---

## Core Technical Capabilities

* **Decoupled Semantic Contracts:** Business logic, table schemas, entity grains, and metric definitions are exposed declaratively through tools (`list_contracts`). The underlying agent dynamically matches business scope rather than relying on brittle, monolithic system prompts.
* **Zero-Token-Math Gate:** All numerical metrics referenced in generated narrative memos must be explicitly traced back to the output cells of executed SQL result sets. Uncited or hallucinated figures are rejected.
* **Fail-Closed Circuit Breaker:** When model outputs fail verification or schema constraints, the harness suppresses narrative assertions and defaults to displaying raw, auditable data tables.
* **Native Concurrency:** Built with asynchronous Python (`async`/`await`) to integrate cleanly into modern microservice architectures, enterprise orchestration layers, and API endpoints.

---

## The Semantic Divergence Demo (Synthetic Data)

The repository includes a seeded, synthetic dataset featuring **50 enterprise households**[cite: 1] linked to **136 individual department accounts**:

| Segment Archetype | Department Profile (`account_id`) | Enterprise Profile (`household_party_id`) |
| :--- | :--- | :--- |
| **Institutional Single** | High fee per account ($42K–$130K), single product contract[cite: 1]. | Modest enterprise footprint (~1.0x fee multiple)[cite: 1]. |
| **Multi-Product Enterprise** | Modest fee per sub-account ($8K–$26K), private wealth structure[cite: 1]. | Top enterprise revenue tier across 4–7 multi-product holdings (5.5x–9.0x multiple)[cite: 1]. |
| **Retail Core** | Baseline single accounts, self-directed commissions[cite: 1]. | Baseline household lifetime revenue[cite: 1]. |

**The Analytical Result:** When querying top customers, the harness proves that both answers are correct within their respective business contracts: the Department Lens highlights *Institutional Contracts*, while the Enterprise Lens highlights *Multi-Product Family Offices*. The system generates a cross-functional memo explaining the grain divergence rather than forcing an artificial choice.

---

## Live Colab Quickstart

1. Launch the notebook directly in [Google Colab](https://colab.research.google.com/github/RickPack/governed-insights-harness/blob/main/governed_insights_harness.ipynb).
2. Open the **Secrets (🔑)** panel on the left sidebar in Colab:
   * Key: `GEMINI_API_KEY`
   * Value: Your Google AI Studio API key.
3. Click **Runtime** > **Run all** (`Ctrl + F9` or `Cmd + F9`).
4. Review the generated execution trace, dual-lens comparative segment table, and the semantic reconciliation memo.

---

## Strategic Enterprise Roadmap

In a complete enterprise implementation, the harness extends across four primary production layers:

1. **Centralized Semantic Layer Ingestion:** Replace localized contract tools with dynamic connectors to enterprise metric stores (e.g., dbt Semantic Layer, Cube, or enterprise data catalogs).
2. **Advanced Model Risk Management (MRM) Invariants:**
   * *Hierarchy Boundary Validation:* Programmatic assertions ensuring departmental rollups do not exceed parent relationship totals.
   * *Grain Denominator Audits:* Pre-execution AST checks ensuring ratios aggregate across distinct entity IDs (`COUNT(DISTINCT entity_id)`) rather than raw transaction rows.
3. **Golden Query Benchmark Suite:** CI/CD integration running automated regression suites across 50+ golden business questions to monitor schema drift, join accuracy, and planner quality across foundation model updates.
4. **Tenant-Aware Access Controls:** Dynamic schema masking and contract filtering based on user authentication scope.

---

## License

Distributed under the MIT License. Open for adaptation, research, and enterprise implementation.
