"""
tests/test_contracts.py — contract governance metadata, offline.

Ownership is metadata. These tests pin it and guard the one rule that matters
across contracts: one metric name, one definition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_contracts import (  # noqa: E402
    CONTRACT_REGISTRY,
    DEPARTMENT_CONTRACT,
    ENTERPRISE_CONTRACT,
    MetricContract,
    metric_collisions,
)


def test_owners_and_versions_are_pinned():
    assert (DEPARTMENT_CONTRACT.owner, DEPARTMENT_CONTRACT.version) == ("license-ops", "1.2.0")
    assert (ENTERPRISE_CONTRACT.owner, ENTERPRISE_CONTRACT.version) == ("account-strategy", "2.0.0")


def test_a_contract_without_an_owner_cannot_exist():
    base = DEPARTMENT_CONTRACT.model_dump()
    with pytest.raises(ValidationError):
        MetricContract.model_validate({**base, "owner": "   "})
    with pytest.raises(ValidationError):
        MetricContract.model_validate({k: v for k, v in base.items() if k != "owner"})


def test_registry_has_no_metric_name_collision():
    assert metric_collisions(CONTRACT_REGISTRY) == []


def test_same_metric_name_with_a_different_definition_is_reported():
    rival = MetricContract.model_validate(
        {
            **DEPARTMENT_CONTRACT.model_dump(),
            "contract_id": "license_seat_revenue_alt",
            "version": "1.0.0",
            "owner": "finance-ops",
            "metric_expression": "license_revenue * 1.1",
        }
    )
    problems = metric_collisions([*CONTRACT_REGISTRY, rival])
    assert len(problems) == 1
    assert "license-ops" in problems[0] and "finance-ops" in problems[0]


def test_same_metric_name_with_the_same_definition_is_not_a_collision():
    twin = MetricContract.model_validate(
        {**DEPARTMENT_CONTRACT.model_dump(), "version": "1.3.0", "owner": "finance-ops", "description": "Reworded."}
    )
    assert metric_collisions([*CONTRACT_REGISTRY, twin]) == []
