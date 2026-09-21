"""
equivalence.py — a paired equivalence check for a contract version bump.

THE QUESTION
------------
A contract's threshold is about to change. Is the candidate definition
equivalent to the current one, within a stated margin, on which licenses fall
in the segment? Each license is either in or out of the segment under the
current definition and under the candidate, so the data are paired binary
observations. The harness borrows the equivalence-testing idea from Lo,
Datta and Salami (2025); it does not reproduce or validate their method.

THE METHOD
----------
Tango (1998): a score confidence interval for the difference of two paired
proportions, using the restricted maximum-likelihood estimate of the nuisance
parameter. Equivalence is declared when the (1 - 2 * alpha) interval lies
inside [-margin, +margin]. That is the two one-sided tests (TOST) procedure
read off an interval: with alpha = 0.05 the interval is 90%.

ORIENTATION (read this before comparing with R)
-----------------------------------------------
    b = licenses IN the segment under the CURRENT definition only
    c = licenses IN the segment under the CANDIDATE definition only
    estimand = (c - b) / n  =  candidate rate minus current rate

R's PropCIs::scoreci.mp(x, y, n) estimates (y - x) / n, so
tango_interval(b, c, n) corresponds to scoreci.mp(x=b, y=c, n=n).
Swapping the two arguments negates and mirrors the interval.

ALWAYS READ THE DISCORDANT COUNTS
---------------------------------
Two definitions can have nearly equal aggregate rates while disagreeing about
many individual licenses: b = 30 and c = 30 gives an estimate of exactly 0.
Every result therefore carries b, c and n beside the interval, and the render
prints them together.

FRAMING
-------
The synthetic data here are fully enumerated: every license is observed, so
there is no sampling and the interval describes a data-generating process
rather than an estimate about a population. The check is designed for a
monitoring sample of a real book. Nothing here demonstrates equivalence of
any real definition; it shows the mechanics under stated assumptions.

The candidate definition below is defined in this module only. It is not
registered, so the governed floor is unchanged.
"""

from __future__ import annotations

import math
import operator
import random
from functools import lru_cache
from statistics import NormalDist

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from semantic_contracts import DEPARTMENT_CONTRACT, MetricContract, ThresholdDefinition

PRIMARY_MARGIN = 0.05
SENSITIVITY_MARGIN = 0.02
DEFAULT_ALPHA = 0.05
POWER_GATE = 0.80

_OPERATORS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "=": operator.eq}

# The current definition is the registered department contract. The candidate is
# the same contract with a higher floor; it exists only inside this module.
CANDIDATE_DEPARTMENT_CONTRACT: MetricContract = DEPARTMENT_CONTRACT.model_copy(
    update={
        "version": "1.3.0",
        "thresholds": (
            ThresholdDefinition(
                name="high_value_floor",
                column="license_revenue",
                operator=">=",
                value=5500,
                description="Candidate: a license agreement is high value at 5,500 or more in annual seat-license revenue.",
            ),
        ),
    }
)


# ---------------------------------------------------------------------------
# Tango (1998) score interval
# ---------------------------------------------------------------------------


def _xlogy(x: float, y: float) -> float:
    if x == 0:
        return 0.0
    return x * math.log(y) if y > 0 else -math.inf


def _restricted_mle(b: int, c: int, n: int, theta: float) -> float:
    """Restricted MLE of P(current-only) when P(candidate-only) - P(current-only) = theta.

    Maximises b*ln(p) + c*ln(p + theta) + (n - b - c)*ln(1 - 2p - theta).
    Setting the derivative to zero gives 2n p^2 - B p - b theta (1 - theta) = 0,
    with B = b(1 - 3 theta) + c(1 - theta) - 2 (n - b - c) theta. Of the (up to
    two) roots inside the valid range, the one with the higher likelihood is used.
    """
    a = n - b - c
    big_b = b * (1 - 3 * theta) + c * (1 - theta) - 2 * a * theta
    const = -b * theta * (1 - theta)
    disc = big_b * big_b - 8 * n * const
    lo, hi = max(0.0, -theta), (1 - theta) / 2
    if disc < 0:
        disc = 0.0
    roots = [(big_b + s * math.sqrt(disc)) / (4 * n) for s in (1, -1)]
    valid = [p for p in roots if lo - 1e-12 <= p <= hi + 1e-12]
    if not valid:
        valid = [min(max(roots[0], lo), hi)]

    def loglik(p: float) -> float:
        p = min(max(p, lo), hi)
        return _xlogy(b, p) + _xlogy(c, p + theta) + _xlogy(a, 1 - 2 * p - theta)

    return min(max(max(valid, key=loglik), lo), hi)


def _score_z(b: int, c: int, n: int, theta: float) -> float:
    """Tango's score statistic for H0: (c - b)/n population difference equals theta."""
    numerator = (c - b) - n * theta
    p = _restricted_mle(b, c, n, theta)
    variance = n * (2 * p + theta - theta * theta)
    if variance <= 1e-15:
        return 0.0 if abs(numerator) < 1e-9 else math.copysign(math.inf, numerator)
    return numerator / math.sqrt(variance)


def _validate_counts(b: int, c: int, n: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if b < 0 or c < 0 or b + c > n:
        raise ValueError(f"discordant counts must satisfy 0 <= b, c and b + c <= n (got b={b}, c={c}, n={n})")


def tango_interval(b: int, c: int, n: int, conf_level: float = 0.90) -> tuple[float, float]:
    """Tango (1998) score interval for (c - b) / n. Orientation: see the module docstring.

    Equals PropCIs::scoreci.mp(x=b, y=c, n=n, conf.level=conf_level).
    """
    _validate_counts(b, c, n)
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1")
    z = NormalDist().inv_cdf(1 - (1 - conf_level) / 2)
    estimate = (c - b) / n
    eps = 1e-9

    def solve(low: float, high: float, target: float) -> float:
        # Z(theta) falls as theta rises, so bisect for Z(theta) == target.
        if _score_z(b, c, n, low) < target:
            return low
        if _score_z(b, c, n, high) > target:
            return high
        for _ in range(80):
            mid = (low + high) / 2
            if _score_z(b, c, n, mid) > target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    lower = solve(-1 + eps, estimate, z)
    upper = solve(estimate, 1 - eps, -z)
    return max(-1.0, lower), min(1.0, upper)


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------


class PairedEquivalenceResult(BaseModel):
    """One equivalence decision. The discordant counts are part of the result, not an extra."""

    model_config = ConfigDict(frozen=True)

    n: int = Field(description="Paired observations (licenses).")
    current_only: int = Field(description="b: in the segment under the current definition only.")
    candidate_only: int = Field(description="c: in the segment under the candidate definition only.")
    estimate: float = Field(description="(c - b) / n: candidate rate minus current rate.")
    lower: float = Field(description="Lower limit of the Tango score interval.")
    upper: float = Field(description="Upper limit of the Tango score interval.")
    margin: float = Field(description="Equivalence margin; equivalent when the interval is inside [-margin, +margin].")
    alpha: float = Field(description="Per-side significance level; the interval is 100 * (1 - 2 * alpha) percent.")
    equivalent: bool = Field(description="True when the whole interval lies inside the margin.")

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> "PairedEquivalenceResult":
        _validate_counts(self.current_only, self.candidate_only, self.n)
        return self

    @property
    def discordant(self) -> int:
        return self.current_only + self.candidate_only

    def render(self) -> str:
        verdict = "equivalent" if self.equivalent else "not shown equivalent"
        return (
            f"margin +/-{self.margin:.2f}: {verdict}. estimate {self.estimate:+.4f}, "
            f"{100 * (1 - 2 * self.alpha):.0f}% interval [{self.lower:+.4f}, {self.upper:+.4f}]; "
            f"discordant b={self.current_only} (current only), c={self.candidate_only} (candidate only), n={self.n}"
        )


def paired_equivalence(
    b: int, c: int, n: int, margin: float = PRIMARY_MARGIN, alpha: float = DEFAULT_ALPHA
) -> PairedEquivalenceResult:
    """Decide equivalence: the (1 - 2 alpha) Tango interval must sit inside [-margin, +margin]."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    lower, upper = tango_interval(b, c, n, conf_level=1 - 2 * alpha)
    return PairedEquivalenceResult(
        n=n,
        current_only=b,
        candidate_only=c,
        estimate=(c - b) / n,
        lower=lower,
        upper=upper,
        margin=margin,
        alpha=alpha,
        equivalent=(lower > -margin and upper < margin),
    )


# ---------------------------------------------------------------------------
# Seeded Monte Carlo: power and size
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _declares_equivalent(b: int, c: int, n: int, margin: float, alpha: float) -> bool:
    lower, upper = tango_interval(b, c, n, conf_level=1 - 2 * alpha)
    return lower > -margin and upper < margin


def _simulate(n: int, p_current_only: float, p_candidate_only: float, margin: float, alpha: float, reps: int, seed: int) -> float:
    """Fraction of simulated samples of size n in which equivalence is declared."""
    rng = random.Random(seed)
    edge = p_current_only + p_candidate_only
    hits = 0
    for _ in range(reps):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            if u < p_current_only:
                b += 1
            elif u < edge:
                c += 1
        if _declares_equivalent(b, c, n, margin, alpha):
            hits += 1
    return hits / reps


class PowerCheck(BaseModel):
    """Seeded Monte Carlo power at a true difference of 0, and size at the margin boundary."""

    model_config = ConfigDict(frozen=True)

    n: int
    discordant_rate: float = Field(description="Assumed (b + c) / n. Taken from the observed sample.")
    margin: float
    alpha: float
    reps: int
    seed: int
    power: float = Field(description="P(declare equivalence) when the true difference is 0.")
    power_gate: float = Field(description="Minimum power to call the check adequately powered.")
    adequately_powered: bool
    boundary_size: float | None = Field(
        description="Worst-case P(declare equivalence) with the true difference AT +/-margin; should be <= alpha. "
        "None when the discordant rate is below the margin, where the boundary is unreachable."
    )

    def render(self) -> str:
        size = "n/a" if self.boundary_size is None else f"{self.boundary_size:.3f}"
        return (
            f"power at true difference 0: {self.power:.3f} (gate {self.power_gate:.2f}, "
            f"{'adequately powered' if self.adequately_powered else 'NOT adequately powered'}); "
            f"size at the +/-{self.margin:.2f} boundary: {size} (target <= {self.alpha:.2f}); "
            f"{self.reps} draws, seed {self.seed}, discordant rate {self.discordant_rate:.3f}"
        )


def power_check(
    n: int,
    discordant_rate: float,
    margin: float = PRIMARY_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    reps: int = 1000,
    seed: int = 20260921,
) -> PowerCheck:
    """Monte Carlo power at true difference 0, plus size at the boundary. Reproducible for a fixed seed."""
    if not 0 <= discordant_rate <= 1:
        raise ValueError("discordant_rate must be between 0 and 1")
    half = discordant_rate / 2
    power = _simulate(n, half, half, margin, alpha, reps, seed)
    size: float | None = None
    if discordant_rate >= margin:
        p_small, p_large = (discordant_rate - margin) / 2, (discordant_rate + margin) / 2
        size = max(
            _simulate(n, p_small, p_large, margin, alpha, reps, seed + 1),  # true difference +margin
            _simulate(n, p_large, p_small, margin, alpha, reps, seed + 2),  # true difference -margin
        )
    return PowerCheck(
        n=n,
        discordant_rate=discordant_rate,
        margin=margin,
        alpha=alpha,
        reps=reps,
        seed=seed,
        power=power,
        power_gate=POWER_GATE,
        adequately_powered=power >= POWER_GATE,
        boundary_size=size,
    )


# ---------------------------------------------------------------------------
# Contract comparison
# ---------------------------------------------------------------------------


class EquivalenceReport(BaseModel):
    """Everything a reviewer needs to read a proposed floor change."""

    model_config = ConfigDict(frozen=True)

    current: str
    candidate: str
    primary: PairedEquivalenceResult
    sensitivity: PairedEquivalenceResult
    power: PowerCheck
    framing: str = (
        "Illustrative. The data are synthetic and fully enumerated, so the interval describes a data-generating "
        "process, not a sample estimate. The check is designed for a monitoring sample of a real book."
    )

    def render(self) -> str:
        return "\n".join(
            [
                f"Paired equivalence: {self.current} vs {self.candidate}",
                f"  primary     {self.primary.render()}",
                f"  sensitivity {self.sensitivity.render()}",
                f"  {self.power.render()}",
                f"  {self.framing}",
            ]
        )


def _label(contract: MetricContract) -> str:
    return f"{contract.contract_id} v{contract.version} (floor {contract.thresholds[0].value:,.0f})"


def _membership(contract: MetricContract, revenue: float) -> bool:
    threshold = contract.threshold("high_value_floor")
    return _OPERATORS[threshold.operator](revenue, threshold.value)


def compare_contract_segments(
    conn: duckdb.DuckDBPyConnection,
    current: MetricContract = DEPARTMENT_CONTRACT,
    candidate: MetricContract = CANDIDATE_DEPARTMENT_CONTRACT,
    margin: float = PRIMARY_MARGIN,
    sensitivity_margin: float = SENSITIVITY_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    reps: int = 1000,
    seed: int = 20260921,
) -> EquivalenceReport:
    """Pair each license's segment membership under two license-grain contracts and test equivalence."""
    for contract in (current, candidate):
        threshold = contract.threshold("high_value_floor")
        if contract.source_table != "licenses" or threshold.column != "license_revenue":
            raise ValueError(f"{contract.contract_id} is not a license_revenue contract on the licenses table")

    revenues = [row[0] for row in conn.execute("SELECT license_revenue FROM licenses").fetchall()]
    b = sum(1 for r in revenues if _membership(current, r) and not _membership(candidate, r))
    c = sum(1 for r in revenues if _membership(candidate, r) and not _membership(current, r))
    n = len(revenues)
    return EquivalenceReport(
        current=_label(current),
        candidate=_label(candidate),
        primary=paired_equivalence(b, c, n, margin, alpha),
        sensitivity=paired_equivalence(b, c, n, sensitivity_margin, alpha),
        power=power_check(n, (b + c) / n, margin, alpha, reps, seed),
    )
