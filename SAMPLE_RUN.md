# Sample run

This is real, unedited output from `python run_demo.py`, captured on **2026-09-21 08:31 UTC** against Gemini
(`gemini-3.6-flash`), seed 42. Every number below comes from the executed DuckDB queries, the salience computations and the
revenue decomposition, not from the model. The two SDK advisory log lines Google's client prints (about which credential it picked
and about automatic function calling) go to stderr and were not captured; the output below is exactly what the script printed to
stdout.

The "most distinctive quality" line above each salience table is computed from the top-ranked score other than the selection
metric; it is not written by the model. Section 6 is the revenue decomposition: it reconciled on the full population before any
narration ran, and the narrative may quote only its figures. It covers 125 licenses because the 119 current licenses are joined by
6 that churned during the period and so no longer appear in the licenses table; the summary line "119 licenses" above it counts only
the current book. The gate line in section 7 reports both checks: every cited figure traced to the fact base for its lens, and every
formatted figure in the prose was present in the typed cited_metrics list. The telemetry line reports what this run observed, with
no comparison to any other design. Wall-clock times vary from run to run.

To reproduce this yourself or see fresh numbers on a different seed or question, open the notebook in Colab — see the badge and
Quickstart in [README.md](README.md) — and run all cells, or run `python run_demo.py` locally with `GOOGLE_API_KEY` set.

```

==============================================================================
  1. INPUT QUESTION
==============================================================================
Compare the profile of our highest-value enterprise accounts, evaluated at the license level versus the parent organization level.

Synthetic dataset: 50 organizations, 119 licenses (both=4, breadth=10, concentrated=9, mass=27)
  License lens      : 21 licenses with seat-license revenue >= 5,000.0
  Organization lens : 13 organizations with consolidated platform revenue >= 25,000.0
  Overlap           : 4 organizations qualify under both lenses

==============================================================================
  2. RESOLVED CONTRACTS (both lenses, retrieved deterministically)
==============================================================================
[DEPARTMENT LENS]  license_seat_revenue  v1.2.0
  grain      : license (one row per license_id)
  table      : licenses
  metric     : annual seat-license revenue = license_revenue
  threshold  : high_value_floor -> license_revenue >= 5,000.0
  meaning    : Value is the seat-license revenue a single license agreement generates in a year. This is the view a team measured on direct license volume uses. It does not consolidate multiple license agreements held by one parent organization, and it excludes API and analytics module revenue.

[ENTERPRISE LENS]  organization_platform_revenue  v2.0.0
  grain      : organization (one row per organization_id)
  table      : organizations
  metric     : consolidated platform revenue = platform_revenue
  threshold  : high_value_floor -> platform_revenue >= 25,000.0
  meaning    : Value is the total revenue the platform earns from every seat license, API call volume and analytics module a parent organization holds. This is the consolidated relationship view. An organization can qualify through breadth of integrated products even when no single license is large.


==============================================================================
  3. GENERATED SQL (structured output, validated before execution)
==============================================================================
[DEPARTMENT LENS]  threshold=high_value_floor
  SELECT license_id, license_revenue AS metric_value, org_maturity_band, tenure_years, module_count, deployment_channel, org_tier FROM licenses WHERE license_revenue >= 5000
  rationale: Selects license-level revenue and profile attributes from the licenses table where annual seat-license revenue meets or exceeds the governed high_value_floor threshold of 5000.

[ENTERPRISE LENS]  threshold=high_value_floor
  SELECT organization_id, platform_revenue AS metric_value, org_maturity_band, tenure_years, module_count, deployment_channel, org_tier FROM organizations WHERE platform_revenue >= 25000
  rationale: Selects consolidated platform revenue and profile attributes from the organizations table where platform revenue meets or exceeds the governed high_value_floor threshold of 25000.


==============================================================================
  4. DUCKDB RESULTS (deterministic execution)
==============================================================================
[DEPARTMENT LENS]  21 rows  (showing top 8 by metric)
license_id      metric_value  org_maturity_band      tenure_years    module_count  deployment_channel    org_tier
------------  --------------  -------------------  --------------  --------------  --------------------  ----------
L0007                12127.9  20+                              14               2  direct_sales          mid_market
L0005                11752.6  20+                              24               1  self_serve            enterprise
L0001                11470.8  20+                              19               1  direct_sales          mid_market
L0011                11049.2  20+                              14               2  partner_channel       enterprise
L0010                10631.3  20+                              15               2  direct_sales          mid_market
L0004                10245.6  20+                              25               1  partner_channel       enterprise
L0009                 9950.1  20+                              24               2  self_serve            enterprise
L0006                 9912.9  6-10                             17               1  self_serve            enterprise

[ENTERPRISE LENS]  13 rows  (showing top 8 by metric)
organization_id      metric_value  org_maturity_band      tenure_years    module_count  deployment_channel    org_tier
-----------------  --------------  -------------------  --------------  --------------  --------------------  ----------
O023                      37523.3  6-10                             19               8  direct_sales          enterprise
O022                      33979.9  11-20                            14               7  self_serve            mid_market
O016                      32773.2  3-5                               7               8  direct_sales          mid_market
O020                      32058.2  3-5                              11               8  direct_sales          enterprise
O021                      31162    6-10                             19               5  partner_channel       mid_market
O019                      31017.2  6-10                              8              10  direct_sales          enterprise
O017                      30920.4  3-5                              14               9  direct_sales          mid_market
O014                      30407.5  3-5                              12              14  partner_channel       mid_market


==============================================================================
  5. SALIENCE RANKINGS (what differentiates each segment from its own baseline)
==============================================================================
[DEPARTMENT LENS]  segment 21 of 119 (contract license_seat_revenue v1.2.0)
  The most distinctive quality of this segment is tenure years: 17.5 vs 7.8 baseline (higher; Cohen's d +1.55).
attribute                segment    baseline      effect  direction    method
-----------------------  ---------  ----------  --------  -----------  ---------
license_revenue          8,789.5    3,221.3         1.99  ^ higher     Cohen's d
tenure_years             17.5       7.8             1.55  ^ higher     Cohen's d
org_tier = smb           0.0%       41.2%          -1.39  v lower      pp delta
org_maturity_band = 0-2  0.0%       14.3%          -0.78  v lower      pp delta
org_maturity_band = 20+  42.9%      11.8%           0.73  ^ higher     pp delta
org_maturity_band = 3-5  9.5%       37.0%          -0.68  v lower      pp delta

[ENTERPRISE LENS]  segment 13 of 50 (contract organization_platform_revenue v2.0.0)
  The most distinctive quality of this segment is module count: 8.9 vs 4.7 baseline (higher; Cohen's d +1.37).
attribute                          segment    baseline      effect  direction    method
---------------------------------  ---------  ----------  --------  -----------  ---------
platform_revenue                   30,638.3   13,246.7        1.65  ^ higher     Cohen's d
module_count                       8.9        4.7             1.37  ^ higher     Cohen's d
org_tier = smb                     7.7%       46.0%          -0.93  v lower      pp delta
org_maturity_band = 0-2            0.0%       16.0%          -0.82  v lower      pp delta
deployment_channel = self_serve    15.4%      46.0%          -0.68  v lower      pp delta
deployment_channel = direct_sales  61.5%      38.0%           0.48  ^ higher     pp delta


==============================================================================
  6. REVENUE MOVEMENT (deterministic decomposition, reconciled before any narration)
==============================================================================
DEPARTMENT LENS revenue movement, 2025-Q4 (seat-license revenue; every movement counts)
  begin 350,962.3 -> end 383,333.0; change 32,370.7 across 125 licenses
  new 46,863.3 | expansion 14,996.2 | contraction 9,020.4 | churn 20,184.3 | migration in 1,697.9 | migration out 1,981.8
  reconciliation: passed on 125 rows (tolerance 0.01)
  labelled subset, does not reconcile to the full change: licenses with end-of-period seat-license revenue >= 5,000; 21 licenses, change 29,772.0

ENTERPRISE LENS revenue movement, 2025-Q4 (seat-license part of platform revenue only; migrations inside one organization net to zero)
  begin 350,962.3 -> end 383,333.0; change 32,370.7 across 50 organizations
  new 46,863.3 | expansion 14,996.2 | contraction 9,020.4 | churn 20,184.3 | migration in 395.2 | migration out 679.2
  reconciliation: passed on 50 rows (tolerance 0.01)
  labelled subset, does not reconcile to the full change: organizations with consolidated platform revenue >= 25,000; 13 organizations, change 23,559.4


==============================================================================
  7. GATE OUTCOMES
==============================================================================
[DEPARTMENT LENS]  executed  rows=21  source_table=pass, grain_discipline=pass, governed_thresholds=pass, cross_grain_join=pass
[ENTERPRISE LENS]  executed  rows=13  source_table=pass, grain_discipline=pass, governed_thresholds=pass, cross_grain_join=pass

ZERO-TOKEN-MATH GATE PASSED: all 46 cited figure(s) trace to executed results for their lens, and every formatted figure in the prose is cited.
narrative attempts validated: 1
telemetry (observed): 2 model calls, 7 deterministic operations (2 governed queries, 2 salience rankings, 2 decompositions, 1 gate evaluation(s)); wall-clock 38,683 ms
  by stage: plan 7,744 ms, execute 8 ms, salience 94 ms, decomposition 33 ms, narrative 30,804 ms

==============================================================================
  8. LENS SUMMARIES
==============================================================================
[DEPARTMENT LENS]  The Department Lens highlights long-tenured, highly mature single license agreements that yield high annual seat-license revenue.
  The department lens evaluates single licenses, identifying 21 high-value licenses out of 119 total licenses. These high-value licenses average 8,789.5 in annual seat-license revenue compared to the baseline average of 3,221.3, with an average tenure of 17.5 years versus 7.8 years for the baseline. Organizations with maturity band 20+ represent 42.9% of this segment compared to 11.8% of the baseline, while small and medium business accounts comprise 0.0% compared to 41.2% in the baseline. Over 2025-Q4, seat-license revenue across 125 licenses began at 350,962.3 and ended at 383,333.0, representing a change of 32,370.7, driven by 46,863.3 in new revenue, 14,996.2 in expansion, 9,020.4 in contraction, 20,184.3 in churn, 1,697.9 in migration in, and 1,981.8 in migration out; the labelled subset of 21 licenses with ending seat-license revenue at or above 5,000 contributed a change of 29,772.0, which does not reconcile to the full change.
  top attributes: license revenue, tenure years, org maturity band = 20+

[ENTERPRISE LENS]  The Enterprise Lens identifies consolidated parent organizations with extensive product adoption and direct sales engagement generating high platform revenue.
  The enterprise lens evaluates consolidated parent organizations, identifying 13 high-value organizations out of 50 total organizations. These segment organizations average 30,638.3 in platform revenue compared to 13,246.7 for the baseline, and utilize an average module count of 8.9 compared to 4.7 for the baseline. Direct sales represent 61.5% of deployment channels in this segment versus 38.0% in the baseline, while self-serve deployment accounts for 15.4% versus 46.0% for the baseline, and small and medium business tier accounts comprise 7.7% versus 46.0% for the baseline. In 2025-Q4, seat-license revenue across 50 organizations moved from a beginning figure of 350,962.3 to an ending figure of 383,333.0, yielding a net change of 32,370.7, with new additions of 46,863.3, expansion of 14,996.2, contraction of 9,020.4, churn of 20,184.3, migration in of 395.2, and migration out of 679.2; the labelled subset of 13 organizations with ending platform revenue at or above 25,000 accounted for a change of 23,559.4, which does not reconcile to the full change.
  top attributes: platform revenue, module count, deployment channel = direct sales


==============================================================================
  9. RECONCILIATION MEMO
==============================================================================
The department lens operates at the license grain, evaluating annual seat-license revenue against a threshold of 5,000 to isolate large individual contracts. In contrast, the enterprise lens operates at the parent organization grain, applying a threshold of 25,000 across total consolidated platform revenue, which incorporates API call volume and analytics modules alongside seat licenses. Consequently, the enterprise lens captures accounts that achieve high aggregate value through product breadth across multiple integrated modules, even when individual licenses remain below 5,000. Conversely, the department lens highlights legacy licenses with long tenure regardless of whether the parent entity holds additional modules. Neither view is wrong; they answer different questions for direct license management versus executive platform relationship oversight.

Cited metrics (each verified against executed results):
  - department high value license count [department] = 21.0
  - department total licenses [department] = 119.0
  - department segment mean license revenue [department] = 8,789.5
  - department baseline mean license revenue [department] = 3,221.3
  - department segment mean tenure years [department] = 17.5
  - department baseline mean tenure years [department] = 7.8
  - department segment org maturity band 20+ percentage [department] = 42.9
  - department baseline org maturity band 20+ percentage [department] = 11.8
  - department segment org tier smb percentage [department] = 0.0
  - department baseline org tier smb percentage [department] = 41.2
  - department movement total licenses [department] = 125.0
  - department movement begin revenue [department] = 350,962.3
  - department movement end revenue [department] = 383,333.0
  - department movement revenue change [department] = 32,370.7
  - department movement new revenue [department] = 46,863.3
  - department movement expansion revenue [department] = 14,996.2
  - department movement contraction revenue [department] = 9,020.4
  - department movement churn revenue [department] = 20,184.3
  - department movement migration in revenue [department] = 1,697.9
  - department movement migration out revenue [department] = 1,981.8
  - department threshold license revenue [department] = 5,000.0
  - department labelled subset revenue change [department] = 29,772.0
  - enterprise high value organization count [enterprise] = 13.0
  - enterprise total organizations [enterprise] = 50.0
  - enterprise segment mean platform revenue [enterprise] = 30,638.3
  - enterprise baseline mean platform revenue [enterprise] = 13,246.7
  - enterprise segment mean module count [enterprise] = 8.9
  - enterprise baseline mean module count [enterprise] = 4.7
  - enterprise segment deployment channel direct sales percentage [enterprise] = 61.5
  - enterprise baseline deployment channel direct sales percentage [enterprise] = 38.0
  - enterprise segment deployment channel self serve percentage [enterprise] = 15.4
  - enterprise baseline deployment channel self serve percentage [enterprise] = 46.0
  - enterprise segment org tier smb percentage [enterprise] = 7.7
  - enterprise baseline org tier smb percentage [enterprise] = 46.0
  - enterprise movement total organizations [enterprise] = 50.0
  - enterprise movement begin revenue [enterprise] = 350,962.3
  - enterprise movement end revenue [enterprise] = 383,333.0
  - enterprise movement revenue change [enterprise] = 32,370.7
  - enterprise movement new revenue [enterprise] = 46,863.3
  - enterprise movement expansion revenue [enterprise] = 14,996.2
  - enterprise movement contraction revenue [enterprise] = 9,020.4
  - enterprise movement churn revenue [enterprise] = 20,184.3
  - enterprise movement migration in revenue [enterprise] = 395.2
  - enterprise movement migration out revenue [enterprise] = 679.2
  - enterprise threshold platform revenue [enterprise] = 25,000.0
  - enterprise labelled subset revenue change [enterprise] = 23,559.4
```
