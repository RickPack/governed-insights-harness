"""
tests/test_seed_sweep.py — the dataset's properties hold beyond seed 42.

Every other test uses one seed, so a property could be an accident of that
draw. The two lenses are engineered to disagree, and the decomposition is
engineered to reconcile. These tests check both claims across many seeds, so
"engineered by design" is something the suite verifies rather than assumes.
The data is synthetic; this shows the generator behaves, not that any real
data would.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decomposition import decompose  # noqa: E402
from salience import compute_salience  # noqa: E402
from semantic_contracts import DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT  # noqa: E402
from synthetic_data import build_database  # noqa: E402

SEEDS = range(1, 21)

# The lens contrast must not depend on the seed: measured over 200 seeds the
# overlap of the organizations each lens selects was between 0.04 and 0.32.
MAX_OVERLAP = 0.5


def _segment_sql(contract) -> str:
    return (
        f"SELECT {contract.entity_id_column}, {contract.metric_expression} AS metric_value, "
        f"{', '.join(contract.profile_attributes)} FROM {contract.source_table} WHERE {contract.thresholds[0].as_sql()}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_decomposition_reconciles_and_segments_exist_for_every_seed(seed):
    conn, _ = build_database(seed=seed, verbose=False)
    for contract in (DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT):
        decomposition = decompose(conn, contract.lens)
        assert decomposition.outcome.passed and decomposition.outcome.rows_checked == decomposition.entities
        assert all(r.begin_revenue >= 0 and r.end_revenue >= 0 for r in decomposition.rows)
        ranking = compute_salience(conn, contract, _segment_sql(contract))  # raises EmptySegmentError if none
        assert 0 < ranking.segment_size < ranking.baseline_size


@pytest.mark.parametrize("seed", SEEDS)
def test_the_two_lenses_select_different_organizations_for_every_seed(seed):
    conn, _ = build_database(seed=seed, verbose=False)
    by_license = {r[0] for r in conn.execute("SELECT DISTINCT organization_id FROM licenses WHERE license_revenue >= 5000").fetchall()}
    by_platform = {r[0] for r in conn.execute("SELECT organization_id FROM organizations WHERE platform_revenue >= 25000").fetchall()}
    overlap = len(by_license & by_platform) / len(by_license | by_platform)
    assert overlap < MAX_OVERLAP, f"seed {seed}: the lenses agree too closely ({overlap:.2f})"
    assert by_license - by_platform and by_platform - by_license  # each lens finds organizations the other misses


def _snapshot(conn) -> dict[str, list]:
    tables = ("licenses", "organizations", "license_snapshots", "license_movements")
    return {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall() for t in tables}


def test_a_seed_reproduces_every_table_and_different_seeds_differ():
    first, _ = build_database(seed=7, verbose=False)
    second, _ = build_database(seed=7, verbose=False)
    other, _ = build_database(seed=8, verbose=False)
    assert _snapshot(first) == _snapshot(second)
    assert _snapshot(first) != _snapshot(other)
