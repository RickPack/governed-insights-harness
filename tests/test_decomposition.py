"""
tests/test_decomposition.py — the revenue decomposition and its reconciliation gate, offline.

Small hand-built databases make each classification rule visible. The generated
synthetic database is used for the population-level checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decomposition import (  # noqa: E402
    DEFAULT_PERIOD,
    ReconciliationError,
    decompose,
)
from governed_duckdb_tool import GovernanceViolation  # noqa: E402
from synthetic_data import build_database  # noqa: E402

P = DEFAULT_PERIOD


def make_db(snapshots, movements=(), platform=None):
    """snapshots: (license_id, org, begin, end). movements: (id, license, type, amount, counterparty)."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE license_snapshots (license_id VARCHAR, organization_id VARCHAR, period VARCHAR, "
        "begin_revenue DOUBLE, end_revenue DOUBLE)"
    )
    conn.execute(
        "CREATE TABLE license_movements (movement_id VARCHAR, license_id VARCHAR, period VARCHAR, "
        "movement_type VARCHAR, amount DOUBLE, destination_license_id VARCHAR)"
    )
    conn.execute("CREATE TABLE organizations (organization_id VARCHAR, platform_revenue DOUBLE)")
    for lid, org, begin, end in snapshots:
        conn.execute("INSERT INTO license_snapshots VALUES (?, ?, ?, ?, ?)", [lid, org, P, begin, end])
    for mid, lid, kind, amount, other in movements:
        conn.execute("INSERT INTO license_movements VALUES (?, ?, ?, ?, ?, ?)", [mid, lid, P, kind, amount, other])
    for org in {s[1] for s in snapshots}:
        total = sum(s[3] for s in snapshots if s[1] == org)
        conn.execute("INSERT INTO organizations VALUES (?, ?)", [org, (platform or {}).get(org, total)])
    return conn


@pytest.fixture(scope="module")
def synthetic():
    conn, _ = build_database(verbose=False)
    return conn


# 1 ----------------------------------------------------------------------------


@pytest.mark.parametrize("lens", ["department", "enterprise"])
def test_reconciliation_passes_on_clean_synthetic_data(synthetic, lens):
    result = decompose(synthetic, lens)
    assert result.outcome.passed
    assert result.outcome.rows_checked == result.entities > 0
    assert abs(result.reported_change - result.components.net) < 0.01 * result.entities


@pytest.mark.parametrize("lens", ["department", "enterprise"])
def test_injected_discrepancy_fails_closed(lens):
    conn, _ = build_database(verbose=False)
    conn.execute("UPDATE license_snapshots SET begin_revenue = begin_revenue + 50 WHERE license_id = 'L0001'")
    with pytest.raises(ReconciliationError) as caught:
        decompose(conn, lens)
    assert isinstance(caught.value, GovernanceViolation)
    outcome = caught.value.outcome
    assert not outcome.passed
    assert len(outcome.failures) == 1
    assert outcome.failures[0].entity_id in {"L0001", conn.execute("SELECT organization_id FROM license_snapshots WHERE license_id='L0001'").fetchone()[0]}
    assert abs(outcome.failures[0].discrepancy) == pytest.approx(50, abs=0.01)


def test_movement_without_snapshot_fails_closed():
    conn = make_db([("L1", "O1", 100.0, 100.0)], [("M1", "L9", "EXPANSION", 10.0, None)])
    with pytest.raises(ReconciliationError) as caught:
        decompose(conn, "department")
    assert caught.value.outcome.orphan_movements[0].license_id == "L9"


# 2 ----------------------------------------------------------------------------


def test_internal_migration_visible_at_license_lens_and_zero_at_organization_lens():
    conn = make_db(
        [("L1", "O1", 1000.0, 900.0), ("L2", "O1", 500.0, 600.0)],
        [("M1", "L1", "MIGRATION_OUT", 100.0, "L2"), ("M2", "L2", "MIGRATION_IN", 100.0, "L1")],
    )
    dept = decompose(conn, "department")
    assert dept.components.migration_out == 100.0
    assert dept.components.migration_in == 100.0
    ent = decompose(conn, "enterprise")
    assert ent.components.migration_out == 0.0
    assert ent.components.migration_in == 0.0
    assert ent.reported_change == dept.reported_change == 0.0
    assert ent.outcome.passed


# 3 ----------------------------------------------------------------------------


def test_grain_invariant_organization_equals_sum_of_its_licenses(synthetic):
    dept = decompose(synthetic, "department")
    ent = decompose(synthetic, "enterprise")
    org_of = dict(synthetic.execute("SELECT license_id, organization_id FROM license_snapshots").fetchall())
    internal_moves = {
        (m, lid)
        for m, lid, other in synthetic.execute(
            "SELECT movement_id, license_id, destination_license_id FROM license_movements "
            "WHERE movement_type IN ('MIGRATION_IN','MIGRATION_OUT')"
        ).fetchall()
        if other in org_of and org_of[other] == org_of[lid]
    }
    assert internal_moves, "synthetic data should contain an internal migration"

    expected = {"new": 0.0, "expansion": 0.0, "contraction": 0.0, "churn": 0.0, "migration_in": 0.0, "migration_out": 0.0}
    names = {
        "NEW": "new", "EXPANSION": "expansion", "CONTRACTION": "contraction",
        "CHURN": "churn", "MIGRATION_IN": "migration_in", "MIGRATION_OUT": "migration_out",
    }
    for mid, lid, kind, amount in synthetic.execute(
        "SELECT movement_id, license_id, movement_type, amount FROM license_movements"
    ).fetchall():
        if (mid, lid) in internal_moves:
            continue
        expected[names[kind]] += amount
    for name, value in expected.items():
        assert getattr(ent.components, name) == pytest.approx(value, abs=0.01)
    assert ent.reported_change == pytest.approx(dept.reported_change, abs=0.01)
    assert ent.components.net == pytest.approx(dept.components.net - (
        (dept.components.migration_in - ent.components.migration_in)
        - (dept.components.migration_out - ent.components.migration_out)
    ), abs=0.01)


# 4 ----------------------------------------------------------------------------


def test_cross_organization_migration_is_external_for_both_organizations():
    conn = make_db(
        [("L1", "O1", 1000.0, 900.0), ("L2", "O2", 500.0, 600.0)],
        [("M1", "L1", "MIGRATION_OUT", 100.0, "L2"), ("M2", "L2", "MIGRATION_IN", 100.0, "L1")],
    )
    ent = decompose(conn, "enterprise")
    by_org = {r.entity_id: r for r in ent.rows}
    assert by_org["O1"].components.migration_out == 100.0
    assert by_org["O2"].components.migration_in == 100.0
    assert by_org["O1"].reported_change == -100.0
    assert by_org["O2"].reported_change == 100.0
    assert not ent.flags


# 5 ----------------------------------------------------------------------------


def test_contraction_to_nonzero_versus_churn_to_zero():
    conn = make_db(
        [("L1", "O1", 1000.0, 400.0), ("L2", "O1", 700.0, 0.0)],
        [("M1", "L1", "CONTRACTION", 600.0, None), ("M2", "L2", "CHURN", 700.0, None)],
    )
    dept = decompose(conn, "department")
    assert dept.components.contraction == 600.0
    assert dept.components.churn == 700.0
    assert dept.reported_change == -1300.0
    assert dept.outcome.passed


# 6 ----------------------------------------------------------------------------


@pytest.mark.parametrize("counterparty", [None, "L404"])
def test_migration_with_missing_destination_is_external_and_flagged(counterparty):
    conn = make_db(
        [("L1", "O1", 1000.0, 900.0)],
        [("M1", "L1", "MIGRATION_OUT", 100.0, counterparty)],
    )
    ent = decompose(conn, "enterprise")
    assert ent.components.migration_out == 100.0  # counted, not treated as internal
    assert [f.movement_id for f in ent.flags] == ["M1"]
    assert "external" in ent.flags[0].reason


# 7 ----------------------------------------------------------------------------


@pytest.mark.parametrize("lens", ["department", "enterprise"])
def test_zero_movement_period(lens):
    conn = make_db([("L1", "O1", 800.0, 800.0), ("L2", "O2", 300.0, 300.0)])
    result = decompose(conn, lens)
    assert result.reported_change == 0.0
    assert result.components.net == 0.0
    assert all(getattr(result.components, n) == 0.0 for n in type(result.components).model_fields)


# Floor view ------------------------------------------------------------------


def test_floor_view_is_a_labelled_subset_that_does_not_claim_to_reconcile():
    conn = make_db(
        [("L1", "O1", 6000.0, 6000.0), ("L2", "O1", 2000.0, 0.0)],
        [("M1", "L2", "CHURN", 2000.0, None)],
    )
    dept = decompose(conn, "department")
    assert dept.reported_change == -2000.0
    assert dept.floor_view.members == 1
    assert dept.floor_view.reported_change == 0.0  # churned license fell below the floor
    assert "5,000" in dept.floor_view.membership_rule
