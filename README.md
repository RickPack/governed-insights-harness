# Governed Insights Harness (GIH)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RickPack/governed-insights-harness/blob/main/governed_insights_harness.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, type-safe execution runtime demonstrating **semantic entity disambiguation** and **model-risk governance** for enterprise conversational analytics.

---

## The Enterprise Challenge

Enterprise conversational BI initiatives often struggle when scaled across business units because departments maintain conflicting definitions for identical terms:
* **The Department Lens:** Measures performance at the single account/contract level (`account_id`) focused on direct management fees.
* **The Enterprise Lens:** Measures performance at the consolidated household relationship level (`household_party_id`) across all product lines.

When an executive asks: *"What qualities describe the customers who produce the most revenue per customer for our department vs. the entire business?"*, naive LLM agents conflate the entity grains and hallucinate arithmetic in narrative text.

---

## Architectural Pattern
