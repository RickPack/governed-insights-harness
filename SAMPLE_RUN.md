# Sample run

This is real, unedited output from `python run_demo.py`, captured on **2026-09-18 12:35 UTC** against Gemini
(`gemini-3.6-flash`), seed 42. Every number below comes from the executed DuckDB queries and salience computations, not from the
model. The two SDK advisory log lines Google's client prints (about which credential it picked and about automatic function
calling) were removed; nothing else was edited.

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
  rationale: This query selects all requested profile attributes and license_revenue aliased as metric_value from the licenses source table, filtering for high value licenses using the exact threshold license_revenue >= 5000.

[ENTERPRISE LENS]  threshold=high_value_floor
  SELECT organization_id, platform_revenue AS metric_value, org_maturity_band, tenure_years, module_count, deployment_channel, org_tier FROM organizations WHERE platform_revenue >= 25000
  rationale: This query selects all requested profile attributes and platform_revenue aliased as metric_value from the organizations source table, filtering for high value organizations using the exact threshold platform_revenue >= 25000.


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
attribute                segment    baseline      effect  direction    method
-----------------------  ---------  ----------  --------  -----------  ---------
license_revenue          8,789.5    3,221.3         1.99  ^ higher     Cohen's d
tenure_years             17.5       7.8             1.56  ^ higher     Cohen's d
org_tier = smb           0.0%       41.2%          -1.39  v lower      pp delta
org_maturity_band = 0-2  0.0%       14.3%          -0.78  v lower      pp delta
org_maturity_band = 20+  42.9%      11.8%           0.73  ^ higher     pp delta
org_maturity_band = 3-5  9.5%       37.0%          -0.68  v lower      pp delta

[ENTERPRISE LENS]  segment 13 of 50 (contract organization_platform_revenue v2.0.0)
attribute                        segment    baseline      effect  direction    method
-------------------------------  ---------  ----------  --------  -----------  ---------
platform_revenue                 30,638.3   13,246.7        1.65  ^ higher     Cohen's d
module_count                     8.9        4.7             1.37  ^ higher     Cohen's d
org_tier = smb                   7.7%       46.0%          -0.93  v lower      pp delta
org_maturity_band = 0-2          0.0%       16.0%          -0.82  v lower      pp delta
deployment_channel = self_serve  15.4%      46.0%          -0.68  v lower      pp delta
org_maturity_band = 6-10         46.2%      24.0%           0.47  ^ higher     pp delta


==============================================================================
  6. GATE OUTCOMES
==============================================================================
[DEPARTMENT LENS]  executed  rows=21  source_table=pass, grain_discipline=pass, governed_thresholds=pass, cross_grain_join=pass
[ENTERPRISE LENS]  executed  rows=13  source_table=pass, grain_discipline=pass, governed_thresholds=pass, cross_grain_join=pass

ZERO-TOKEN-MATH GATE PASSED: all 18 cited figure(s) trace to executed results.
narrative attempts validated: 1

==============================================================================
  7. LENS SUMMARIES
==============================================================================
[DEPARTMENT LENS]  The high-value license segment represents long-tenured enterprise agreements driving substantial annual seat revenue.
  Evaluated at the individual contract level, this segment generates an average license revenue of 8,789.5 compared to the baseline average of 3,221.3. These accounts exhibit exceptional longevity with a mean tenure of 17.5 years versus the baseline of 7.8 years. Highly mature organizations dominate this group, with 42.9% falling in the 20+ organization maturity band compared to 11.8% in the baseline, while small business accounts comprise 0.0% of the segment compared to 41.2% overall.
  top attributes: license revenue, tenure years, organization maturity band = 20+

[ENTERPRISE LENS]  The top enterprise segment highlights cross-product platform adoption across multi-module account relationships.
  Evaluated at the parent organization level, high-value accounts deliver a mean consolidated platform revenue of 30,638.3 compared to the baseline average of 13,246.7. These enterprise accounts demonstrate broader platform engagement with an average module count of 8.9 compared to 4.7 across the baseline. Mid-stage organizations lead this category, as 46.2% belong to the 6-10 organization maturity band compared to 24.0% in the baseline, while self-serve deployment accounts for only 15.4% compared to 46.0% overall.
  top attributes: platform revenue, module count, organization maturity band = 6-10


==============================================================================
  8. RECONCILIATION MEMO
==============================================================================
The department lens and enterprise lens surface different account populations because they analyze high value using distinct grains, metrics, and qualification thresholds. The department lens operates at the license contract grain, applying a threshold of annual license revenue of at least 5,000.0 to capture single large seat agreements held by long-tenured clients. In contrast, the enterprise lens operates at the parent organization grain, consolidating platform revenue across seat licenses, API usage, and analytics modules with a threshold of at least 25,000.0 to highlight multi-product organization breadth. Neither lens is incorrect, as they answer different questions for license-level contract tracking versus holistically managed enterprise account strategy.

Cited metrics (each verified against executed results):
  - department lens mean license revenue [department] = 8,789.5
  - department lens baseline mean license revenue [department] = 3,221.3
  - department lens mean tenure years [department] = 17.5
  - department lens baseline mean tenure years [department] = 7.8
  - department lens organization maturity band = 20+ percentage [department] = 42.9
  - department lens baseline organization maturity band = 20+ percentage [department] = 11.8
  - department lens organization tier = smb percentage [department] = 0.0
  - department lens baseline organization tier = smb percentage [department] = 41.2
  - department lens license revenue threshold [department] = 5,000.0
  - enterprise lens mean platform revenue [enterprise] = 30,638.3
  - enterprise lens baseline mean platform revenue [enterprise] = 13,246.7
  - enterprise lens mean module count [enterprise] = 8.9
  - enterprise lens baseline mean module count [enterprise] = 4.7
  - enterprise lens organization maturity band = 6-10 percentage [enterprise] = 46.2
  - enterprise lens baseline organization maturity band = 6-10 percentage [enterprise] = 24.0
  - enterprise lens deployment channel = self serve percentage [enterprise] = 15.4
  - enterprise lens baseline deployment channel = self serve percentage [enterprise] = 46.0
  - enterprise lens platform revenue threshold [enterprise] = 25,000.0
```
