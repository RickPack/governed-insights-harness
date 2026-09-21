"""
decomposition.py — deterministic period-over-period seat-license revenue decomposition.

WHY THIS EXISTS
---------------
"Revenue moved" is only useful when the movement is explained, and an
explanation that does not add up is worse than none. This module explains the
change in seat-license revenue between the start and end of a period using
classified movement events, then checks that the explanation reconciles to the
change reported by the snapshots. It uses no language model: every number is
SQL and arithmetic, so the narrative layer can only quote it.

THE IDENTITY
------------
    change = new + expansion - contraction - churn + migration_in - migration_out

* Reported change comes from snapshots: end_revenue - begin_revenue.
* Components come only from classified movement events.
* No component is ever computed as a residual. If the two disagree by 0.01 or
  more for any license or organization, the gate fails closed with
  ReconciliationError (a GovernanceViolation) and nothing is narrated.

TWO LENSES
----------
* department (license grain): every event counts.
* enterprise (organization grain): a migration between two licenses of the SAME
  organization is INTERNAL and contributes zero. Migrations that cross an
  organization boundary remain. A migration whose counterparty is missing or
  cannot be resolved is EXTERNAL and is flagged, never silently internal.

SCOPE
-----
Seat-license revenue only. Organization platform_revenue also holds API and
analytics revenue that has no license, so the organization-lens decomposition
explains the license part of the change and is labelled that way.

FLOOR INTERACTION
-----------------
Reconciliation runs on the FULL population. A floor-filtered view is shown only
as a labelled subset with its membership rule stated. It cannot reconcile to the
full change: churned licenses end at 0 and therefore fall below any floor. The
governed floors are not changed.

This module deliberately reads the database directly rather than going through
GovernedDuckDBTool, so it adds nothing to the per-query audit trail.
"""

from __future__ import annotations

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from governed_duckdb_tool import GovernanceViolation
from semantic_contracts import DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT, DimensionFilter, Lens

DEFAULT_PERIOD = "2025-Q4"
RECONCILIATION_TOLERANCE = 0.01

# Fixed SQL. Nothing is interpolated; the period is filtered after the query.
# scope is 'internal' when the counterparty license sits in the same organization,
# 'external' when it sits in a different one, 'unresolved' when the counterparty
# is missing or unknown (treated as external and flagged), and 'none' for
# non-migration events.
_CLASSIFIED_MOVEMENTS_CTE = """
WITH classified AS (
    SELECT
        m.movement_id,
        m.license_id,
        m.period,
        m.movement_type,
        m.amount,
        s.organization_id,
        CASE
            WHEN m.movement_type NOT IN ('MIGRATION_IN', 'MIGRATION_OUT') THEN 'none'
            WHEN c.organization_id IS NULL THEN 'unresolved'
            WHEN c.organization_id = s.organization_id THEN 'internal'
            ELSE 'external'
        END AS scope
    FROM license_movements m
    LEFT JOIN license_snapshots s ON s.license_id = m.license_id AND s.period = m.period
    LEFT JOIN license_snapshots c ON c.license_id = m.destination_license_id AND c.period = m.period
)
"""

_COMPONENT_COLUMNS = """
    COALESCE(SUM(amount) FILTER (WHERE movement_type = 'NEW'), 0) AS new_revenue,
    COALESCE(SUM(amount) FILTER (WHERE movement_type = 'EXPANSION'), 0) AS expansion,
    COALESCE(SUM(amount) FILTER (WHERE movement_type = 'CONTRACTION'), 0) AS contraction,
    COALESCE(SUM(amount) FILTER (WHERE movement_type = 'CHURN'), 0) AS churn,
    COALESCE(SUM(amount) FILTER (WHERE movement_type = 'MIGRATION_IN'), 0) AS migration_in,
    COALESCE(SUM(amount) FILTER (WHERE movement_type = 'MIGRATION_OUT'), 0) AS migration_out
"""

_LICENSE_SQL = (
    _CLASSIFIED_MOVEMENTS_CTE
    + f"""
, movement_totals AS (
    SELECT license_id, period, {_COMPONENT_COLUMNS}
    FROM classified
    GROUP BY license_id, period
)
SELECT
    s.license_id AS entity_id, s.period, s.begin_revenue, s.end_revenue,
    COALESCE(t.new_revenue, 0) AS new_revenue, COALESCE(t.expansion, 0) AS expansion,
    COALESCE(t.contraction, 0) AS contraction, COALESCE(t.churn, 0) AS churn,
    COALESCE(t.migration_in, 0) AS migration_in, COALESCE(t.migration_out, 0) AS migration_out
FROM license_snapshots s
LEFT JOIN movement_totals t ON t.license_id = s.license_id AND t.period = s.period
ORDER BY s.period, s.license_id
"""
)

# Organization lens: internal migrations are excluded before aggregating.
_ORGANIZATION_SQL = (
    _CLASSIFIED_MOVEMENTS_CTE
    + f"""
, movement_totals AS (
    SELECT organization_id, period, {_COMPONENT_COLUMNS}
    FROM classified
    WHERE NOT (movement_type IN ('MIGRATION_IN', 'MIGRATION_OUT') AND scope = 'internal')
    GROUP BY organization_id, period
),
snapshot_totals AS (
    SELECT organization_id, period, SUM(begin_revenue) AS begin_revenue, SUM(end_revenue) AS end_revenue
    FROM license_snapshots
    GROUP BY organization_id, period
)
SELECT
    s.organization_id AS entity_id, s.period, s.begin_revenue, s.end_revenue,
    COALESCE(t.new_revenue, 0) AS new_revenue, COALESCE(t.expansion, 0) AS expansion,
    COALESCE(t.contraction, 0) AS contraction, COALESCE(t.churn, 0) AS churn,
    COALESCE(t.migration_in, 0) AS migration_in, COALESCE(t.migration_out, 0) AS migration_out,
    o.platform_revenue
FROM snapshot_totals s
LEFT JOIN movement_totals t ON t.organization_id = s.organization_id AND t.period = s.period
LEFT JOIN organizations o ON o.organization_id = s.organization_id
ORDER BY s.period, s.organization_id
"""
)

# Movements the snapshots cannot account for, and migrations with no usable counterparty.
_ORPHAN_SQL = _CLASSIFIED_MOVEMENTS_CTE + "SELECT movement_id, license_id, period FROM classified WHERE organization_id IS NULL ORDER BY movement_id"
_UNRESOLVED_SQL = (
    _CLASSIFIED_MOVEMENTS_CTE
    + "SELECT movement_id, license_id, period FROM classified WHERE scope = 'unresolved' AND organization_id IS NOT NULL ORDER BY movement_id"
)


# ---------------------------------------------------------------------------
# Typed outputs
# ---------------------------------------------------------------------------


class Components(BaseModel):
    """Classified movement totals. Every amount is non-negative; the name gives the sign."""

    model_config = ConfigDict(frozen=True)

    new: float
    expansion: float
    contraction: float
    churn: float
    migration_in: float
    migration_out: float

    @property
    def net(self) -> float:
        """Signed sum: new + expansion - contraction - churn + migration_in - migration_out."""
        return self.new + self.expansion - self.contraction - self.churn + self.migration_in - self.migration_out


class DecompositionRow(BaseModel):
    """One license or organization for one period."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    period: str
    begin_revenue: float
    end_revenue: float
    reported_change: float = Field(description="end_revenue - begin_revenue, from snapshots.")
    components: Components
    discrepancy: float = Field(description="reported_change minus the signed sum of components. Must be ~0.")

    @property
    def reconciles(self) -> bool:
        return abs(self.discrepancy) < RECONCILIATION_TOLERANCE


class MovementFlag(BaseModel):
    """A movement that needed a judgment call, surfaced instead of hidden."""

    model_config = ConfigDict(frozen=True)

    movement_id: str
    license_id: str
    reason: str


class ReconciliationOutcome(BaseModel):
    """Typed result of the fail-closed reconciliation gate."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    tolerance: float
    rows_checked: int
    failures: tuple[DecompositionRow, ...] = ()
    orphan_movements: tuple[MovementFlag, ...] = ()


class FloorView(BaseModel):
    """A floor-filtered subset, shown labelled. It is NOT expected to reconcile to the full change."""

    model_config = ConfigDict(frozen=True)

    membership_rule: str
    members: int
    reported_change: float
    components: Components


class LensDecomposition(BaseModel):
    """Decomposition for one lens and one period, reconciled on the full population."""

    model_config = ConfigDict(frozen=True)

    lens: Lens
    grain: str
    period: str
    scope_note: str
    scope: tuple[DimensionFilter, ...] = Field(
        default=(), description="Governed restrictions the question applied. Empty means the full population."
    )
    entities: int
    begin_revenue: float
    end_revenue: float
    reported_change: float
    components: Components
    outcome: ReconciliationOutcome
    flags: tuple[MovementFlag, ...] = ()
    floor_view: FloorView
    rows: tuple[DecompositionRow, ...] = Field(default=(), repr=False)

    def figures(self) -> list[float]:
        """Every number this decomposition presents, for the narration FactBase."""
        c = self.components
        fv = self.floor_view
        return [
            self.begin_revenue, self.end_revenue, self.reported_change,
            c.new, c.expansion, c.contraction, c.churn, c.migration_in, c.migration_out,
            float(self.entities), float(self.outcome.rows_checked), self.outcome.tolerance,
            float(fv.members), fv.reported_change,
        ]

    def render(self) -> str:
        """Plain text, one decimal. These are the figures a narrative may quote."""
        c = self.components
        fv = self.floor_view
        return (
            f"{self.lens.upper()} LENS revenue movement, {self.period} ({self.scope_note})\n"
            f"  begin {self.begin_revenue:,.1f} -> end {self.end_revenue:,.1f}; change {self.reported_change:,.1f} "
            f"across {self.entities} {self.grain}s\n"
            f"  new {c.new:,.1f} | expansion {c.expansion:,.1f} | contraction {c.contraction:,.1f} | "
            f"churn {c.churn:,.1f} | migration in {c.migration_in:,.1f} | migration out {c.migration_out:,.1f}\n"
            f"  reconciliation: {'passed' if self.outcome.passed else 'FAILED'} on {self.outcome.rows_checked} rows "
            f"(tolerance {self.outcome.tolerance})\n"
            f"  labelled subset, does not reconcile to the full change: {fv.membership_rule}; "
            f"{fv.members} {self.grain}s, change {fv.reported_change:,.1f}"
        )


class ReconciliationError(GovernanceViolation):
    """Raised when components do not add up to the snapshot change. Carries the typed outcome."""

    def __init__(self, outcome: ReconciliationOutcome, lens: Lens, period: str):
        self.outcome = outcome
        self.lens = lens
        self.period = period
        worst = max(outcome.failures, key=lambda r: abs(r.discrepancy), default=None)
        detail = (
            f"{len(outcome.failures)} failing row(s), worst {worst.entity_id} off by {worst.discrepancy:,.2f}"
            if worst
            else f"{len(outcome.orphan_movements)} movement(s) match no snapshot"
        )
        super().__init__(f"{lens} decomposition for {period} does not reconcile: {detail}. Nothing is narrated.")


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _row(record: tuple) -> DecompositionRow:
    entity_id, period, begin, end, new, exp, con, churn, mig_in, mig_out = record[:10]
    components = Components(
        new=new, expansion=exp, contraction=con, churn=churn, migration_in=mig_in, migration_out=mig_out
    )
    change = end - begin
    return DecompositionRow(
        entity_id=entity_id,
        period=period,
        begin_revenue=begin,
        end_revenue=end,
        reported_change=change,
        components=components,
        discrepancy=change - components.net,
    )


def _sum_components(rows: list[DecompositionRow]) -> Components:
    return Components(
        new=sum(r.components.new for r in rows),
        expansion=sum(r.components.expansion for r in rows),
        contraction=sum(r.components.contraction for r in rows),
        churn=sum(r.components.churn for r in rows),
        migration_in=sum(r.components.migration_in for r in rows),
        migration_out=sum(r.components.migration_out for r in rows),
    )


def decompose(
    conn: duckdb.DuckDBPyConnection,
    lens: Lens,
    period: str = DEFAULT_PERIOD,
    scope: tuple[DimensionFilter, ...] = (),
) -> LensDecomposition:
    """Decompose one lens for one period, or raise ReconciliationError.

    Reconciliation is checked on every license (department) or organization
    (enterprise) in the population before anything is returned. That is the
    full population unless the question restricted it (`scope`), in which case
    it is the same restricted population the plan's baseline uses; each entity
    still reconciles on its own, so a restriction cannot hide a discrepancy.
    """
    if lens == "department":
        sql, grain = _LICENSE_SQL, "license"
        scope_note = "seat-license revenue; every movement counts"
        floor = DEPARTMENT_CONTRACT.threshold("high_value_floor")
        floor_rule = f"licenses with end-of-period seat-license revenue {floor.operator} {floor.value:,.0f}"
        source = DEPARTMENT_CONTRACT
    else:
        sql, grain = _ORGANIZATION_SQL, "organization"
        scope_note = "seat-license part of platform revenue only; migrations inside one organization net to zero"
        floor = ENTERPRISE_CONTRACT.threshold("high_value_floor")
        floor_rule = f"organizations with consolidated platform revenue {floor.operator} {floor.value:,.0f}"
        source = ENTERPRISE_CONTRACT

    records = [r for r in conn.execute(sql).fetchall() if r[1] == period]
    if scope:
        predicate = " AND ".join(f.as_sql() for f in scope)
        in_scope = {
            row[0]
            for row in conn.execute(f"SELECT {source.entity_id_column} FROM {source.source_table} WHERE {predicate}").fetchall()
        }
        records = [r for r in records if r[0] in in_scope]
        scope_note += f"; restricted to {'; '.join(f.as_text() for f in scope)}"
    rows = [_row(r) for r in records]
    orphans = tuple(
        MovementFlag(movement_id=m, license_id=lid, reason="movement matches no license snapshot")
        for m, lid, p in conn.execute(_ORPHAN_SQL).fetchall()
        if p == period
    )
    failures = tuple(r for r in rows if not r.reconciles)
    outcome = ReconciliationOutcome(
        passed=not failures and not orphans,
        tolerance=RECONCILIATION_TOLERANCE,
        rows_checked=len(rows),
        failures=failures,
        orphan_movements=orphans,
    )
    if not outcome.passed:
        raise ReconciliationError(outcome, lens, period)

    flags = tuple(
        MovementFlag(
            movement_id=m, license_id=lid, reason="migration counterparty missing or unresolved; counted as external"
        )
        for m, lid, p in conn.execute(_UNRESOLVED_SQL).fetchall()
        if p == period
    )

    if lens == "department":
        members = [r for r in rows if r.end_revenue >= floor.value]
    else:
        platform = {r[0]: r[10] for r in records}
        members = [r for r in rows if platform[r.entity_id] is not None and platform[r.entity_id] >= floor.value]

    return LensDecomposition(
        lens=lens,
        grain=grain,
        period=period,
        scope_note=scope_note,
        scope=scope,
        entities=len(rows),
        begin_revenue=sum(r.begin_revenue for r in rows),
        end_revenue=sum(r.end_revenue for r in rows),
        reported_change=sum(r.reported_change for r in rows),
        components=_sum_components(rows),
        outcome=outcome,
        flags=flags,
        floor_view=FloorView(
            membership_rule=floor_rule,
            members=len(members),
            reported_change=sum(r.reported_change for r in members),
            components=_sum_components(members),
        ),
        rows=tuple(rows),
    )
