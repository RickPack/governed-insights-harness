"""
synthetic_data.py — a seeded, two-grain dataset with a deliberate contrast.

WHY THIS DESIGN
---------------
A demo dataset that makes both definitions of "high value" pick the same
accounts would prove nothing. This generator seeds a defensible contrast that
any enterprise SaaS platform would recognise:

  * Concentrated organizations hold one large legacy license agreement. They
    are high value at the LICENSE grain (a single license clears the revenue
    floor) but ordinary at the ORGANIZATION grain, because they use few
    integrated modules and little API or analytics-module volume.

  * Breadth organizations hold several mid-sized license agreements plus API
    usage and analytics modules. No single license clears the revenue floor,
    so the license lens misses them, but their consolidated platform revenue
    clears the organization floor.

The two grains are linked by a shared organization key so that either lens
can be reproduced by a reviewer with plain SQL. Organization revenue is
materialised on the organization table (rather than computed by a join at
query time) so that each contract reads exactly one table at exactly one
grain, which is what the cross-grain join validator expects.

Everything is generated from `random.Random(seed)`; the same seed yields the
same tables byte for byte, so every number in the notebook is reproducible.
All data is synthetic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

# Grain floors mirror the governed thresholds in semantic_contracts.py. They are
# repeated here only for the summary print; the contracts remain the source of truth.
LICENSE_FLOOR = 5_000.0
ORGANIZATION_FLOOR = 25_000.0

MATURITY_BANDS = ("0-2", "3-5", "6-10", "11-20", "20+")
DEPLOYMENT_CHANNELS = ("direct_sales", "self_serve", "partner_channel")
ORG_TIERS = ("smb", "mid_market", "enterprise")


@dataclass(frozen=True)
class _Archetype:
    """Generation parameters for one organization archetype. Internal to this module."""

    name: str
    count: int
    licenses_range: tuple[int, int]
    license_revenue_range: tuple[float, float]      # per-license annual seat-license revenue
    api_usage_range: tuple[float, float]        # organization-level API usage revenue
    analytics_range: tuple[float, float]        # organization-level analytics module revenue
    tenure_range: tuple[int, int]
    maturity_weights: tuple[float, ...]         # weights over MATURITY_BANDS
    channel_weights: tuple[float, ...]          # weights over DEPLOYMENT_CHANNELS
    tier_weights: tuple[float, ...]             # weights over ORG_TIERS


# The contrast lives in these four rows. Concentrated organizations are the
# license lens's favourites; breadth organizations are the enterprise lens's.
_ARCHETYPES: tuple[_Archetype, ...] = (
    _Archetype(
        name="concentrated",
        count=9,
        licenses_range=(1, 2),
        license_revenue_range=(5_800, 13_500),
        api_usage_range=(0, 2_500),
        analytics_range=(0, 1_500),
        tenure_range=(16, 32),
        maturity_weights=(0.0, 0.05, 0.15, 0.30, 0.50),
        channel_weights=(0.25, 0.65, 0.10),
        tier_weights=(0.0, 0.35, 0.65),
    ),
    _Archetype(
        name="breadth",
        count=10,
        licenses_range=(4, 6),
        license_revenue_range=(1_800, 4_600),
        api_usage_range=(8_000, 15_000),
        analytics_range=(3_000, 7_500),
        tenure_range=(6, 15),
        maturity_weights=(0.05, 0.30, 0.45, 0.15, 0.05),
        channel_weights=(0.70, 0.05, 0.25),
        tier_weights=(0.05, 0.60, 0.35),
    ),
    _Archetype(
        name="both",
        count=4,
        licenses_range=(2, 3),
        license_revenue_range=(4_000, 9_500),
        api_usage_range=(5_000, 12_000),
        analytics_range=(2_000, 6_000),
        tenure_range=(10, 22),
        maturity_weights=(0.0, 0.10, 0.35, 0.40, 0.15),
        channel_weights=(0.55, 0.15, 0.30),
        tier_weights=(0.0, 0.30, 0.70),
    ),
    _Archetype(
        name="mass",
        count=27,
        licenses_range=(1, 3),
        license_revenue_range=(250, 2_400),
        api_usage_range=(0, 1_800),
        analytics_range=(0, 1_200),
        tenure_range=(1, 12),
        maturity_weights=(0.30, 0.30, 0.20, 0.15, 0.05),
        channel_weights=(0.35, 0.60, 0.05),
        tier_weights=(0.75, 0.22, 0.03),
    ),
)


class TierSummary(BaseModel):
    """What the generated dataset looks like, printed on init so the data story is visible."""

    organizations: int = Field(description="Total organizations generated.")
    licenses: int = Field(description="Total license agreements generated.")
    organizations_by_archetype: dict[str, int] = Field(description="Organization count per generation archetype.")
    licenses_over_floor: int = Field(description="License agreements whose seat-license revenue meets the license-grain floor.")
    organizations_over_floor: int = Field(description="Organizations whose consolidated platform revenue meets the organization-grain floor.")
    organizations_in_both: int = Field(description="Organizations that qualify under both lenses.")

    def render(self) -> str:
        """Plain-text summary; no raw dollar signs so it renders cleanly in notebooks."""
        archetypes = ", ".join(f"{k}={v}" for k, v in self.organizations_by_archetype.items())
        return (
            f"Synthetic dataset: {self.organizations} organizations, {self.licenses} licenses ({archetypes})\n"
            f"  License lens      : {self.licenses_over_floor} licenses with seat-license revenue >= {LICENSE_FLOOR:,.1f}\n"
            f"  Organization lens : {self.organizations_over_floor} organizations with consolidated platform revenue >= {ORGANIZATION_FLOOR:,.1f}\n"
            f"  Overlap           : {self.organizations_in_both} organizations qualify under both lenses"
        )


def _choice(rng: random.Random, options: tuple[str, ...], weights: tuple[float, ...]) -> str:
    return rng.choices(options, weights=weights, k=1)[0]


def _generate_frames(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the license and organization frames from one seeded RNG."""
    rng = random.Random(seed)
    organizations: list[dict] = []
    licenses: list[dict] = []
    organization_counter = 0
    license_counter = 0

    for archetype in _ARCHETYPES:
        for _ in range(archetype.count):
            organization_counter += 1
            organization_id = f"O{organization_counter:03d}"

            # Organization-level attributes are drawn once and shared by its licenses,
            # with light per-license variation so the two grains are linked but not identical.
            maturity_band = _choice(rng, MATURITY_BANDS, archetype.maturity_weights)
            channel = _choice(rng, DEPLOYMENT_CHANNELS, archetype.channel_weights)
            org_tier = _choice(rng, ORG_TIERS, archetype.tier_weights)
            organization_tenure = rng.randint(*archetype.tenure_range)

            n_licenses = rng.randint(*archetype.licenses_range)
            organization_license_total = 0.0
            organization_modules = 0
            for _ in range(n_licenses):
                license_counter += 1
                revenue = round(rng.uniform(*archetype.license_revenue_range), 2)
                # Breadth organizations attach more modules per license by construction.
                modules = rng.randint(2, 3) if archetype.name in ("breadth", "both") else rng.randint(1, 2)
                licenses.append(
                    {
                        "license_id": f"L{license_counter:04d}",
                        "organization_id": organization_id,
                        "license_revenue": revenue,
                        "org_maturity_band": maturity_band,
                        # License tenure never exceeds the relationship tenure.
                        "tenure_years": max(1, organization_tenure - rng.randint(0, 4)),
                        "module_count": modules,
                        "deployment_channel": channel if rng.random() < 0.85 else _choice(rng, DEPLOYMENT_CHANNELS, archetype.channel_weights),
                        "org_tier": org_tier,
                    }
                )
                organization_license_total += revenue
                organization_modules += modules

            api_usage = round(rng.uniform(*archetype.api_usage_range), 2)
            analytics = round(rng.uniform(*archetype.analytics_range), 2)
            organizations.append(
                {
                    "organization_id": organization_id,
                    "archetype": archetype.name,
                    "license_count": n_licenses,
                    "license_revenue_total": round(organization_license_total, 2),
                    "api_usage_revenue": api_usage,
                    "analytics_module_revenue": analytics,
                    # Materialised so the organization contract reads one table at one grain.
                    "platform_revenue": round(organization_license_total + api_usage + analytics, 2),
                    "org_maturity_band": maturity_band,
                    "tenure_years": organization_tenure,
                    "module_count": organization_modules,
                    "deployment_channel": channel,
                    "org_tier": org_tier,
                }
            )

    return pd.DataFrame(licenses), pd.DataFrame(organizations)


def build_database(seed: int = 42, verbose: bool = True) -> tuple[duckdb.DuckDBPyConnection, TierSummary]:
    """Create an in-memory DuckDB with `licenses` and `organizations` tables.

    Returns the connection and a TierSummary. The summary is printed when
    verbose is True so the shape of the data is visible before any query runs.
    """
    licenses_df, organizations_df = _generate_frames(seed)

    conn = duckdb.connect(database=":memory:")
    # Registering the frames then copying into real tables keeps the connection
    # independent of the pandas objects' lifetime.
    conn.register("licenses_src", licenses_df)
    conn.register("organizations_src", organizations_df)
    conn.execute("CREATE TABLE licenses AS SELECT * FROM licenses_src")
    conn.execute("CREATE TABLE organizations AS SELECT * FROM organizations_src")
    conn.unregister("licenses_src")
    conn.unregister("organizations_src")

    summary = TierSummary(
        organizations=len(organizations_df),
        licenses=len(licenses_df),
        organizations_by_archetype=organizations_df["archetype"].value_counts().sort_index().to_dict(),
        licenses_over_floor=int((licenses_df["license_revenue"] >= LICENSE_FLOOR).sum()),
        organizations_over_floor=int((organizations_df["platform_revenue"] >= ORGANIZATION_FLOOR).sum()),
        organizations_in_both=int(
            organizations_df.loc[organizations_df["platform_revenue"] >= ORGANIZATION_FLOOR, "organization_id"]
            .isin(licenses_df.loc[licenses_df["license_revenue"] >= LICENSE_FLOOR, "organization_id"])
            .sum()
        ),
    )
    if verbose:
        print(summary.render())
    return conn, summary


if __name__ == "__main__":
    build_database()
