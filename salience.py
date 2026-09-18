"""
salience.py — deterministic attribute salience, computed per lens.

WHY THIS DESIGN
---------------
Computing every attribute of a segment is easy. Deciding which attributes
actually distinguish that segment from its baseline is the harder problem, and
it is the one an executive is asking about when they say "what is interesting
about these customers". The obvious alternative is to hand the model the two
profile tables and ask it to pick out the differences. That reintroduces
token math at the exact point where precision matters most.

Salience here is pure SQL and Python. The model receives a ranked list it did
not compute and cannot alter. Two properties matter:

  * Salience branches on dtype. Numeric attributes (tenure, module count,
    revenue) use the standardised mean difference, Cohen's d, with a pooled
    standard deviation. Categorical attributes (org maturity band, deployment
    channel, org tier) use the percentage-point delta between the segment's
    share at each level and the baseline's share. Numeric distance math is
    never applied to a category code. Both branches return the same
    SalienceScore model, so downstream ranking and rendering are uniform.

  * Salience is computed independently for each lens against its own
    baseline (the full population at that lens's grain). A license-grain
    segment is compared with all licenses; an organization-grain segment with
    all organizations. Comparing across grains would be a grain error of its own.

For ranking, categorical scores also carry Cohen's h (the arcsine-transformed
difference in proportions), which lives on the same standardised scale as
Cohen's d. The percentage-point delta is the figure a reader understands;
Cohen's h is what lets a category level and a numeric attribute be ranked in
one list without comparing apples to percentages.
"""

from __future__ import annotations

import math
from typing import Literal

import duckdb
from pydantic import BaseModel, Field

from semantic_contracts import Lens, MetricContract

SalienceMethod = Literal["cohens_d", "percentage_point_delta"]
Direction = Literal["higher", "lower", "similar"]

# Below this absolute effect size the attribute is reported as 'similar' to
# baseline. 0.2 is the conventional floor for a small effect.
_SIMILAR_THRESHOLD = 0.2

# DuckDB type names treated as categorical. Everything else numeric-looking is numeric.
_CATEGORICAL_TYPES = {"VARCHAR", "TEXT", "STRING", "CHAR", "BOOLEAN", "ENUM"}


class SalienceScore(BaseModel):
    """One attribute (or one level of a categorical attribute) compared with baseline."""

    attribute: str = Field(description="Attribute column name.")
    level: str | None = Field(description="Category level for categorical attributes; None for numeric.")
    method: SalienceMethod = Field(description="Which branch computed the score: Cohen's d or percentage-point delta.")
    segment_value: float = Field(description="Segment mean (numeric) or segment share in percent (categorical).")
    baseline_value: float = Field(description="Baseline mean (numeric) or baseline share in percent (categorical).")
    raw_difference: float = Field(description="segment_value minus baseline_value, in the attribute's own units.")
    effect_size: float = Field(
        description="Standardised effect: Cohen's d for numeric, Cohen's h for categorical. Used for ranking."
    )
    direction: Direction = Field(description="Whether the segment is higher, lower, or similar to baseline.")

    @property
    def display_name(self) -> str:
        return self.attribute if self.level is None else f"{self.attribute} = {self.level}"


class SalienceRanking(BaseModel):
    """All scores for one lens, ordered by absolute effect size, most distinctive first."""

    lens: Lens = Field(description="Lens the ranking belongs to.")
    contract_id: str = Field(description="Contract that defined the segment.")
    contract_version: str = Field(description="Version of that contract.")
    segment_size: int = Field(description="Distinct entities in the segment.")
    baseline_size: int = Field(description="Distinct entities in the baseline (the full grain).")
    scores: list[SalienceScore] = Field(description="Scores sorted by absolute effect size, descending.")

    def top(self, n: int = 5) -> list[SalienceScore]:
        return self.scores[:n]

    def numeric_facts(self) -> list[float]:
        """Every number in the ranking, so the zero-token-math gate can verify figures the narrative cites."""
        facts: list[float] = [float(self.segment_size), float(self.baseline_size)]
        for score in self.scores:
            facts.extend([score.segment_value, score.baseline_value, score.raw_difference, score.effect_size])
        return facts


# ---------------------------------------------------------------------------
# Effect-size math (pure Python)
# ---------------------------------------------------------------------------


def cohens_d(mean_s: float, sd_s: float, n_s: int, mean_b: float, sd_b: float, n_b: int) -> float:
    """Standardised mean difference with a pooled standard deviation. Zero if there is no spread."""
    if n_s < 2 or n_b < 2:
        return 0.0
    pooled_var = ((n_s - 1) * sd_s**2 + (n_b - 1) * sd_b**2) / (n_s + n_b - 2)
    if pooled_var <= 0:
        return 0.0
    return (mean_s - mean_b) / math.sqrt(pooled_var)


def cohens_h(p_s: float, p_b: float) -> float:
    """Arcsine-transformed difference in proportions; the standardised analogue of Cohen's d for shares."""
    clamp = lambda p: min(1.0, max(0.0, p))  # noqa: E731 — tiny inline guard against float drift
    return 2 * math.asin(math.sqrt(clamp(p_s))) - 2 * math.asin(math.sqrt(clamp(p_b)))


def _direction(effect: float) -> Direction:
    if abs(effect) < _SIMILAR_THRESHOLD:
        return "similar"
    return "higher" if effect > 0 else "lower"


# ---------------------------------------------------------------------------
# SQL-backed profiling
# ---------------------------------------------------------------------------


def _column_types(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """Column -> DuckDB type name, so the salience branch is chosen from the schema, not guessed from values."""
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?", [table]
    ).fetchall()
    return {name: dtype.upper() for name, dtype in rows}


def _numeric_score(
    conn: duckdb.DuckDBPyConnection, contract: MetricContract, segment_sql: str, attribute: str
) -> SalienceScore:
    """Cohen's d branch. Segment and baseline stats are computed in DuckDB in one statement."""
    entity = contract.entity_id_column
    table = contract.source_table
    # Baseline is the whole grain; segment is entities selected by the validated plan.
    sql = f"""
        WITH seg AS (SELECT {entity} FROM ({segment_sql}) AS plan_result)
        SELECT
            AVG(CASE WHEN b.{entity} IN (SELECT {entity} FROM seg) THEN b.{attribute} END)              AS seg_mean,
            STDDEV_SAMP(CASE WHEN b.{entity} IN (SELECT {entity} FROM seg) THEN b.{attribute} END)      AS seg_sd,
            COUNT(DISTINCT CASE WHEN b.{entity} IN (SELECT {entity} FROM seg) THEN b.{entity} END)      AS seg_n,
            AVG(b.{attribute})                                                                         AS base_mean,
            STDDEV_SAMP(b.{attribute})                                                                 AS base_sd,
            COUNT(DISTINCT b.{entity})                                                                 AS base_n
        FROM {table} AS b
    """
    seg_mean, seg_sd, seg_n, base_mean, base_sd, base_n = conn.execute(sql).fetchone()
    seg_mean, base_mean = round(float(seg_mean), 1), round(float(base_mean), 1)
    d = cohens_d(float(seg_mean), float(seg_sd or 0.0), int(seg_n), float(base_mean), float(base_sd or 0.0), int(base_n))
    d = round(d, 2)
    return SalienceScore(
        attribute=attribute,
        level=None,
        method="cohens_d",
        segment_value=seg_mean,
        baseline_value=base_mean,
        raw_difference=round(seg_mean - base_mean, 1),
        effect_size=d,
        direction=_direction(d),
    )


def _categorical_scores(
    conn: duckdb.DuckDBPyConnection, contract: MetricContract, segment_sql: str, attribute: str
) -> list[SalienceScore]:
    """Percentage-point-delta branch. One score per level, shares computed over COUNT(DISTINCT entity)."""
    entity = contract.entity_id_column
    table = contract.source_table
    sql = f"""
        WITH seg AS (SELECT {entity} FROM ({segment_sql}) AS plan_result),
        totals AS (
            SELECT
                COUNT(DISTINCT {entity})                                                    AS base_n,
                COUNT(DISTINCT CASE WHEN {entity} IN (SELECT {entity} FROM seg) THEN {entity} END) AS seg_n
            FROM {table}
        )
        SELECT
            b.{attribute}                                                                                AS level,
            COUNT(DISTINCT CASE WHEN b.{entity} IN (SELECT {entity} FROM seg) THEN b.{entity} END) * 1.0
                / (SELECT seg_n FROM totals)                                                             AS seg_share,
            COUNT(DISTINCT b.{entity}) * 1.0 / (SELECT base_n FROM totals)                               AS base_share
        FROM {table} AS b
        GROUP BY b.{attribute}
        ORDER BY b.{attribute}
    """
    scores: list[SalienceScore] = []
    for level, seg_share, base_share in conn.execute(sql).fetchall():
        seg_pct = round(float(seg_share or 0.0) * 100, 1)
        base_pct = round(float(base_share or 0.0) * 100, 1)
        h = round(cohens_h(seg_pct / 100, base_pct / 100), 2)
        scores.append(
            SalienceScore(
                attribute=attribute,
                level=str(level),
                method="percentage_point_delta",
                segment_value=seg_pct,
                baseline_value=base_pct,
                raw_difference=round(seg_pct - base_pct, 1),
                effect_size=h,
                direction=_direction(h),
            )
        )
    return scores


def compute_salience(
    conn: duckdb.DuckDBPyConnection,
    contract: MetricContract,
    segment_sql: str,
    attributes: tuple[str, ...] | None = None,
) -> SalienceRanking:
    """Rank how each profile attribute differentiates the segment from its own grain's baseline.

    `segment_sql` must already have passed the pre-execution validator; this
    function trusts it only as far as wrapping it in a subquery.
    """
    attributes = attributes or contract.profile_attributes + (contract.metric_expression,)
    types = _column_types(conn, contract.source_table)

    scores: list[SalienceScore] = []
    for attribute in attributes:
        dtype = types.get(attribute)
        if dtype is None:
            raise KeyError(f"attribute {attribute!r} is not a column of {contract.source_table}")
        # The branch is chosen from the schema type. Categorical columns never see Cohen's d.
        if dtype in _CATEGORICAL_TYPES:
            scores.extend(_categorical_scores(conn, contract, segment_sql, attribute))
        else:
            scores.append(_numeric_score(conn, contract, segment_sql, attribute))

    entity = contract.entity_id_column
    segment_size = conn.execute(
        f"SELECT COUNT(DISTINCT {entity}) FROM ({segment_sql}) AS plan_result"
    ).fetchone()[0]
    baseline_size = conn.execute(f"SELECT COUNT(DISTINCT {entity}) FROM {contract.source_table}").fetchone()[0]

    scores.sort(key=lambda s: abs(s.effect_size), reverse=True)
    return SalienceRanking(
        lens=contract.lens,
        contract_id=contract.contract_id,
        contract_version=contract.version,
        segment_size=int(segment_size),
        baseline_size=int(baseline_size),
        scores=scores,
    )
