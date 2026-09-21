"""
tests/test_equivalence.py — the paired equivalence check (Tango 1998 score interval), offline.

The numeric vector below comes from the author's own R runs of PropCIs::scoreci.mp and
checks this function only; it says nothing about any real definition.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equivalence import (  # noqa: E402
    CANDIDATE_DEPARTMENT_CONTRACT,
    DEFAULT_ALPHA,
    POWER_GATE,
    compare_contract_segments,
    paired_equivalence,
    power_check,
    tango_interval,
)
from semantic_contracts import CONTRACT_REGISTRY, DEPARTMENT_CONTRACT  # noqa: E402
from synthetic_data import build_database  # noqa: E402


# --- the numeric vector and orientation ------------------------------------------------


def test_tango_interval_matches_the_r_vector():
    """scoreci.mp(25, 28, 618) at 90% -> [-0.0148, +0.0247]."""
    lower, upper = tango_interval(25, 28, 618, conf_level=0.90)
    assert round(lower, 4) == -0.0148
    assert round(upper, 4) == 0.0247


def test_orientation_is_candidate_minus_current():
    """x=b (current only), y=c (candidate only), estimand (y - x)/n. Swapping mirrors the interval."""
    lower, upper = tango_interval(25, 28, 618)
    swapped_lower, swapped_upper = tango_interval(28, 25, 618)
    assert swapped_lower == pytest.approx(-upper, abs=1e-9)
    assert swapped_upper == pytest.approx(-lower, abs=1e-9)
    result = paired_equivalence(b=25, c=28, n=618)
    assert result.estimate == pytest.approx(3 / 618)  # candidate includes 3 more licenses than current
    assert result.lower < result.estimate < result.upper


def test_interval_widens_with_confidence_and_stays_in_range():
    narrow = tango_interval(25, 28, 618, conf_level=0.90)
    wide = tango_interval(25, 28, 618, conf_level=0.95)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]
    for b, c, n in ((0, 0, 100), (3, 0, 125), (0, 40, 100), (50, 50, 100)):
        lower, upper = tango_interval(b, c, n)
        assert -1 <= lower <= (c - b) / n <= upper <= 1


def test_invalid_counts_are_rejected():
    for args in ((-1, 0, 10), (6, 6, 10), (0, 0, 0)):
        with pytest.raises(ValueError):
            tango_interval(*args)


# --- the decision rule and the discordant counts ---------------------------------------


def test_decision_rule_is_interval_inside_margin():
    assert paired_equivalence(25, 28, 618, margin=0.05).equivalent is True
    # Upper limit 0.0247 exceeds 0.02, so the tighter margin is not met.
    assert paired_equivalence(25, 28, 618, margin=0.02).equivalent is False
    assert paired_equivalence(0, 40, 100, margin=0.05).equivalent is False  # shifted well outside


def test_discordant_counts_are_reported_with_the_interval():
    """Equal aggregate rates can hide large disagreement: b = c = 30 gives an estimate of 0."""
    result = paired_equivalence(30, 30, 600)
    assert result.estimate == 0.0 and result.equivalent is True
    assert (result.current_only, result.candidate_only, result.n, result.discordant) == (30, 30, 600, 60)
    text = result.render()
    assert "b=30" in text and "c=30" in text and "n=600" in text and "90% interval" in text


# --- seeded Monte Carlo ----------------------------------------------------------------


def test_power_is_reproducible_for_a_fixed_seed():
    first = power_check(618, 53 / 618, reps=500, seed=7)
    second = power_check(618, 53 / 618, reps=500, seed=7)
    other = power_check(618, 53 / 618, reps=500, seed=8)
    assert first == second
    assert first.power_gate == POWER_GATE
    assert first.adequately_powered is True
    assert (first.power, first.boundary_size) != (other.power, other.boundary_size)


def test_underpowered_design_is_flagged():
    check = power_check(60, 0.10, reps=500, seed=7)
    assert check.power < POWER_GATE and check.adequately_powered is False


def test_size_at_the_margin_boundary_is_near_alpha():
    reps = 2000
    tolerance = 3 * math.sqrt(DEFAULT_ALPHA * (1 - DEFAULT_ALPHA) / reps)
    for n, rate in ((618, 53 / 618), (300, 0.10)):
        check = power_check(n, rate, reps=reps, seed=11)
        assert check.boundary_size is not None
        assert check.boundary_size <= DEFAULT_ALPHA + tolerance


def test_boundary_size_is_undefined_when_discordance_is_below_the_margin():
    assert power_check(200, 0.02, reps=200, seed=3).boundary_size is None


# --- contract comparison ---------------------------------------------------------------


def test_candidate_is_not_registered_and_the_governed_floor_is_unchanged():
    assert CANDIDATE_DEPARTMENT_CONTRACT not in CONTRACT_REGISTRY
    assert DEPARTMENT_CONTRACT.threshold("high_value_floor").value == 5000
    assert CANDIDATE_DEPARTMENT_CONTRACT.threshold("high_value_floor").value == 5500


def test_report_pairs_membership_per_license_on_the_synthetic_book():
    conn, _ = build_database(verbose=False)
    between = conn.execute("SELECT count(*) FROM licenses WHERE license_revenue >= 5000 AND license_revenue < 5500").fetchone()[0]
    n = conn.execute("SELECT count(*) FROM licenses").fetchone()[0]
    report = compare_contract_segments(conn, reps=200)
    assert report.primary.current_only == between  # a stricter candidate only ever removes licenses
    assert report.primary.candidate_only == 0
    assert report.primary.n == n
    assert report.primary.lower <= report.primary.estimate <= report.primary.upper
    text = report.render()
    assert "synthetic" in text and "monitoring sample" in text and f"b={between}" in text
