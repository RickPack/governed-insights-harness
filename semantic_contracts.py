"""
semantic_contracts.py — the deterministic contract layer.

WHY THIS DESIGN
---------------
Business definitions in an enterprise drift. "High value account" means one
thing to a team that is measured on direct seat-license volume and another
thing to the platform view that consolidates a parent organization's total
usage. Neither definition is wrong; they answer different questions.

The obvious alternative is to bury those definitions in a prompt: "when the
user says high value, assume revenue over 5,000." That works until the
threshold changes, and then nobody can say which answers were produced under
which rule. Prompts are not versioned artifacts with reviewers.

Contracts here are declarative, versioned Pydantic models. A business
definition changes in Git — with a diff, a reviewer, and a version bump —
without touching a prompt or retraining anything. The language model reads the
contract through structured-output field descriptions; it never invents one.

The validators on these models are the first governance boundary. A contract
with a malformed grain, an empty metric expression, or an entity key that is
not an identifier column cannot be instantiated. Invalid definitions are not
caught at runtime; they are unconstructable.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Vocabulary shared by every model in the pipeline.
# ---------------------------------------------------------------------------

# The two lenses the system always evaluates side by side. "department" is the
# license-grain view; "enterprise" is the organization-grain view.
Lens = Literal["department", "enterprise"]

# The only grains a contract may declare. Anything else fails at construction.
EntityGrain = Literal["license", "organization"]

# Comparison operators a threshold may use. Kept small so the SQL validator can
# recognise every governed threshold literally.
ThresholdOperator = Literal[">=", ">", "<=", "<", "="]

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class ContextResolutionError(Exception):
    """Raised when a business question matches no governed contract.

    Fail-closed by design: the pipeline would rather stop than guess which
    definition of a term the user meant.
    """


# ---------------------------------------------------------------------------
# Contract models
# ---------------------------------------------------------------------------


class ThresholdDefinition(BaseModel):
    """One governed cutoff a query plan is allowed to apply.

    Every numeric comparison in generated SQL must trace back to one of these.
    The pre-execution validator rejects any literal it cannot find here.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Stable identifier for this threshold, e.g. 'high_value_floor'.")
    column: str = Field(description="Column the threshold applies to, as it appears in the source table.")
    operator: ThresholdOperator = Field(description="Comparison operator applied between the column and the value.")
    value: float = Field(description="The governed cutoff value. The only numeric literal a plan may use for this column.")
    description: str = Field(description="Plain-English meaning of the cutoff, for reviewers and for the planner.")

    def as_sql(self) -> str:
        """Render the threshold as a SQL predicate fragment, e.g. 'license_revenue >= 5000'."""
        # Integers are rendered without a trailing '.0' so the SQL reads naturally.
        literal = f"{self.value:g}"
        return f"{self.column} {self.operator} {literal}"


class MetricContract(BaseModel):
    """A versioned, reviewable definition of one business metric at one grain.

    The contract is the single source of truth for: which table to read, what
    the entity is, how the metric is computed, and which cutoffs are governed.
    """

    model_config = ConfigDict(frozen=True)

    contract_id: str = Field(description="Stable identifier, e.g. 'license_seat_revenue'.")
    lens: Lens = Field(description="Which lens this contract serves: 'department' or 'enterprise'.")
    version: str = Field(description="Semantic version of this definition, e.g. '2.0.0'. Bumped on any rule change.")
    scope: str = Field(description="Business scope the definition applies to, e.g. 'Enterprise SaaS account base'.")
    entity_grain: EntityGrain = Field(description="The unit each row represents: 'license' or 'organization'.")
    entity_id_column: str = Field(description="Column that uniquely identifies one entity at this grain. Must end in '_id'.")
    source_table: str = Field(description="The one table this contract reads from.")
    metric_name: str = Field(description="Business name of the metric, e.g. 'annual seat-license revenue'.")
    metric_expression: str = Field(description="SQL expression computing the metric for one entity row.")
    thresholds: tuple[ThresholdDefinition, ...] = Field(
        description="Governed cutoffs available to a query plan. A plan may not invent others."
    )
    profile_attributes: tuple[str, ...] = Field(
        description="Attribute columns available for profiling at this grain."
    )
    keywords: tuple[str, ...] = Field(
        description="Lower-case phrases that indicate a question falls under this contract."
    )
    description: str = Field(description="Plain-English statement of what the metric means and does not mean.")

    # -- Field validators: contract integrity is enforced at construction. -----

    @field_validator("version")
    @classmethod
    def _version_is_semver(cls, value: str) -> str:
        """A contract without a real version cannot be audited; reject anything that is not MAJOR.MINOR.PATCH."""
        if not _SEMVER.match(value):
            raise ValueError(f"version must be semantic (MAJOR.MINOR.PATCH), got {value!r}")
        return value

    @field_validator("metric_expression", "source_table", "metric_name", "scope")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        """Blank strings satisfy the type checker but not the business; reject them."""
        if not value.strip():
            raise ValueError("field must not be blank")
        return value.strip()

    @field_validator("entity_id_column")
    @classmethod
    def _entity_key_is_identifier(cls, value: str) -> str:
        """The grain discipline rule divides by COUNT(DISTINCT entity_id); the key must be recognisably an id column."""
        if not value.endswith("_id"):
            raise ValueError(f"entity_id_column must name an identifier column ending in '_id', got {value!r}")
        return value

    @field_validator("thresholds")
    @classmethod
    def _at_least_one_threshold(cls, value: tuple[ThresholdDefinition, ...]) -> tuple[ThresholdDefinition, ...]:
        """A contract with no governed cutoff leaves the planner free to invent one, which defeats the point."""
        if not value:
            raise ValueError("a contract must define at least one governed threshold")
        return value

    @model_validator(mode="after")
    def _grain_matches_key(self) -> "MetricContract":
        """The entity key must belong to the declared grain: a license contract keyed on organization_id is a grain error."""
        expected = f"{self.entity_grain}_id"
        if self.entity_id_column != expected:
            raise ValueError(
                f"entity_id_column {self.entity_id_column!r} does not match grain {self.entity_grain!r} (expected {expected!r})"
            )
        return self

    # -- Helpers used by the retriever, the prompt, and the validator. --------

    def matches(self, query: str) -> bool:
        """True when any contract keyword appears in the (lower-cased) question."""
        lowered = query.lower()
        return any(keyword in lowered for keyword in self.keywords)

    def threshold(self, name: str) -> ThresholdDefinition:
        """Look up a governed threshold by name; KeyError if the plan references one that does not exist."""
        for threshold in self.thresholds:
            if threshold.name == name:
                return threshold
        raise KeyError(f"contract {self.contract_id} v{self.version} has no threshold named {name!r}")

    def describe_for_prompt(self) -> str:
        """Render the contract as labelled plain text the planner reads. Labels keep the two grains from blending."""
        thresholds = "\n".join(
            f"    - {t.name}: {t.as_sql()}  ({t.description})" for t in self.thresholds
        )
        return (
            f"[{self.lens.upper()} LENS] contract_id={self.contract_id} version={self.version}\n"
            f"  scope: {self.scope}\n"
            f"  entity grain: {self.entity_grain} (one row per {self.entity_id_column})\n"
            f"  source table: {self.source_table}\n"
            f"  metric: {self.metric_name} = {self.metric_expression}\n"
            f"  governed thresholds (the ONLY numeric cutoffs allowed):\n{thresholds}\n"
            f"  profile attributes: {', '.join(self.profile_attributes)}\n"
            f"  meaning: {self.description}"
        )


class CustomerSegmentDefinition(BaseModel):
    """How a contract's metric turns into a named population.

    Separating the segment from the metric lets one metric back several
    segments (top tier, at-risk, dormant) without duplicating the metric rule.
    """

    model_config = ConfigDict(frozen=True)

    segment_name: str = Field(description="Business name of the population, e.g. 'high value licenses'.")
    contract_id: str = Field(description="The MetricContract this segment is defined against.")
    contract_version: str = Field(description="Version of that contract the segment was authored for.")
    threshold_name: str = Field(description="Which governed threshold on the contract selects the segment.")
    selection_rule: str = Field(description="Plain-English statement of who is in the segment.")

    @field_validator("contract_version")
    @classmethod
    def _version_is_semver(cls, value: str) -> str:
        if not _SEMVER.match(value):
            raise ValueError(f"contract_version must be semantic, got {value!r}")
        return value


# ---------------------------------------------------------------------------
# Retrieval output: what the context retriever hands to the planner.
# ---------------------------------------------------------------------------


class ResolvedContext(BaseModel):
    """Both contracts and both segment definitions that apply to one question.

    The retriever always resolves both lenses or raises ContextResolutionError.
    Half a context (one lens) is not a valid state: the whole point of the
    pipeline is the side-by-side comparison.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="The business question, verbatim.")
    department_contract: MetricContract = Field(description="License-grain contract that matched the question.")
    enterprise_contract: MetricContract = Field(description="Organization-grain contract that matched the question.")
    department_segment: CustomerSegmentDefinition = Field(description="Segment definition for the department lens.")
    enterprise_segment: CustomerSegmentDefinition = Field(description="Segment definition for the enterprise lens.")

    @model_validator(mode="after")
    def _lenses_are_distinct_and_correct(self) -> "ResolvedContext":
        """Each slot must hold the lens it claims. A retriever bug that put the same contract in both slots stops here."""
        if self.department_contract.lens != "department" or self.enterprise_contract.lens != "enterprise":
            raise ValueError("resolved contracts are assigned to the wrong lens slots")
        if self.department_contract.entity_grain == self.enterprise_contract.entity_grain:
            raise ValueError("dual-lens context requires two different grains")
        return self

    def contract_for(self, lens: Lens) -> MetricContract:
        return self.department_contract if lens == "department" else self.enterprise_contract

    def segment_for(self, lens: Lens) -> CustomerSegmentDefinition:
        return self.department_segment if lens == "department" else self.enterprise_segment


# ---------------------------------------------------------------------------
# Planner output: what the model is allowed to emit.
# ---------------------------------------------------------------------------


class PlannedQuery(BaseModel):
    """One executable query plan for one lens.

    This is the structured-output surface the language model fills in. The
    field descriptions are the model's instructions; the validators are the
    boundary a malformed plan cannot cross.
    """

    lens: Lens = Field(description="The lens this plan serves. Must match the contract it cites.")
    contract_id: str = Field(description="contract_id of the governing MetricContract, copied exactly.")
    contract_version: str = Field(description="version of the governing MetricContract, copied exactly.")
    threshold_name: str = Field(description="Name of the governed threshold used to select the segment.")
    sql: str = Field(
        description=(
            "A single DuckDB SELECT statement reading only the contract's source_table. "
            "It must select the entity id column, the metric expression aliased as metric_value, "
            "and every profile attribute, filtered by the governed threshold. "
            "Numeric cutoffs must be copied from the contract's thresholds verbatim."
        )
    )
    rationale: str = Field(description="One or two sentences explaining how the plan follows the contract.")

    @field_validator("sql")
    @classmethod
    def _single_select_statement(cls, value: str) -> str:
        """Only read-only, single-statement SQL may reach the validator. Anything else fails at the type boundary."""
        cleaned = value.strip().rstrip(";").strip()
        if not cleaned.lower().startswith("select"):
            raise ValueError("sql must be a single SELECT statement")
        if ";" in cleaned:
            raise ValueError("sql must contain exactly one statement")
        return cleaned

    @field_validator("contract_version")
    @classmethod
    def _version_is_semver(cls, value: str) -> str:
        if not _SEMVER.match(value):
            raise ValueError(f"contract_version must be semantic, got {value!r}")
        return value


class DualLensPlan(BaseModel):
    """The planner's complete answer: one PlannedQuery per lens.

    The model receives both contracts labelled and must return both plans in
    one structured object. That single constraint is what stops it blending
    a license threshold into an organization query: each plan is tied to one
    lens, and the validator checks the tie.
    """

    department: PlannedQuery = Field(description="Query plan for the license-grain department lens.")
    enterprise: PlannedQuery = Field(description="Query plan for the organization-grain enterprise lens.")

    @model_validator(mode="after")
    def _plans_serve_their_lens(self) -> "DualLensPlan":
        if self.department.lens != "department":
            raise ValueError("department plan is labelled with the wrong lens")
        if self.enterprise.lens != "enterprise":
            raise ValueError("enterprise plan is labelled with the wrong lens")
        return self

    def plan_for(self, lens: Lens) -> PlannedQuery:
        return self.department if lens == "department" else self.enterprise


class GovernedPlan(BaseModel):
    """The chain's final output: resolved context plus the plans generated against it.

    Carrying both together means every downstream stage can check a plan
    against the contract it claims to follow, without a second lookup.
    """

    model_config = ConfigDict(frozen=True)

    context: ResolvedContext = Field(description="Both resolved contracts and segment definitions.")
    plan: DualLensPlan = Field(description="Both generated query plans.")

    @model_validator(mode="after")
    def _plans_cite_resolved_contracts(self) -> "GovernedPlan":
        """A plan that cites a contract id or version the retriever did not resolve is a hallucinated citation. Stop it here."""
        for lens in ("department", "enterprise"):
            contract = self.context.contract_for(lens)
            planned = self.plan.plan_for(lens)
            if planned.contract_id != contract.contract_id or planned.contract_version != contract.version:
                raise ValueError(
                    f"{lens} plan cites {planned.contract_id} v{planned.contract_version}, "
                    f"but the resolved contract is {contract.contract_id} v{contract.version}"
                )
        return self


# ---------------------------------------------------------------------------
# The contract registry: two competing definitions of "high value account".
# ---------------------------------------------------------------------------

PROFILE_ATTRIBUTES: tuple[str, ...] = (
    "org_maturity_band",
    "tenure_years",
    "module_count",
    "deployment_channel",
    "org_tier",
)

# Shared vocabulary that signals the question is about account value. Both
# contracts listen for the same phrases on purpose: the ambiguity is the point,
# and the system resolves it by running both definitions rather than picking one.
_VALUE_KEYWORDS: tuple[str, ...] = (
    "high value",
    "highest value",
    "high-value",
    "most valuable",
    "top customers",
    "best customers",
    "top accounts",
    "high revenue",
    "enterprise accounts",
)

DEPARTMENT_CONTRACT = MetricContract(
    contract_id="license_seat_revenue",
    lens="department",
    version="1.2.0",
    scope="Enterprise SaaS account base, evaluated per license agreement",
    entity_grain="license",
    entity_id_column="license_id",
    source_table="licenses",
    metric_name="annual seat-license revenue",
    metric_expression="license_revenue",
    thresholds=(
        ThresholdDefinition(
            name="high_value_floor",
            column="license_revenue",
            operator=">=",
            value=5000,
            description="A license agreement is high value when its annual seat-license revenue is at least 5,000.",
        ),
    ),
    profile_attributes=PROFILE_ATTRIBUTES,
    keywords=_VALUE_KEYWORDS + ("license level", "seat license revenue", "per license", "seat-license volume"),
    description=(
        "Value is the seat-license revenue a single license agreement generates in a year. This is the "
        "view a team measured on direct license volume uses. It does not consolidate multiple license "
        "agreements held by one parent organization, and it excludes API and analytics module revenue."
    ),
)

ENTERPRISE_CONTRACT = MetricContract(
    contract_id="organization_platform_revenue",
    lens="enterprise",
    version="2.0.0",
    scope="Enterprise SaaS account base, evaluated per parent organization",
    entity_grain="organization",
    entity_id_column="organization_id",
    source_table="organizations",
    metric_name="consolidated platform revenue",
    metric_expression="platform_revenue",
    thresholds=(
        ThresholdDefinition(
            name="high_value_floor",
            column="platform_revenue",
            operator=">=",
            value=25000,
            description=(
                "An organization is high value when its consolidated platform revenue across seat "
                "licenses, API usage and analytics modules is at least 25,000."
            ),
        ),
    ),
    profile_attributes=PROFILE_ATTRIBUTES,
    keywords=_VALUE_KEYWORDS
    + ("organization level", "consolidated platform revenue", "per organization", "parent organization", "consolidated"),
    description=(
        "Value is the total revenue the platform earns from every seat license, API call volume and "
        "analytics module a parent organization holds. This is the consolidated relationship view. An "
        "organization can qualify through breadth of integrated products even when no single license is large."
    ),
)

DEPARTMENT_SEGMENT = CustomerSegmentDefinition(
    segment_name="high value license agreements",
    contract_id=DEPARTMENT_CONTRACT.contract_id,
    contract_version=DEPARTMENT_CONTRACT.version,
    threshold_name="high_value_floor",
    selection_rule="License agreements whose annual seat-license revenue meets the governed high_value_floor.",
)

ENTERPRISE_SEGMENT = CustomerSegmentDefinition(
    segment_name="high value organizations",
    contract_id=ENTERPRISE_CONTRACT.contract_id,
    contract_version=ENTERPRISE_CONTRACT.version,
    threshold_name="high_value_floor",
    selection_rule="Organizations whose consolidated platform revenue meets the governed high_value_floor.",
)

# The registry the retriever searches. Adding a third definition is a new
# entry here and a version bump, not a prompt edit.
CONTRACT_REGISTRY: tuple[MetricContract, ...] = (DEPARTMENT_CONTRACT, ENTERPRISE_CONTRACT)
SEGMENT_REGISTRY: dict[str, CustomerSegmentDefinition] = {
    DEPARTMENT_CONTRACT.contract_id: DEPARTMENT_SEGMENT,
    ENTERPRISE_CONTRACT.contract_id: ENTERPRISE_SEGMENT,
}
