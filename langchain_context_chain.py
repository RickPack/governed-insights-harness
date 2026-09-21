"""
langchain_context_chain.py — LCEL orchestration from question to governed plan.

WHY THIS DESIGN
---------------
Two decisions shape this module.

(a) The model receives BOTH contracts, explicitly labelled by lens, in one
    prompt, and must return BOTH plans in one structured object. The obvious
    alternative is to call the planner twice, once per lens. That is simpler
    to write and worse to govern: with two calls, nothing stops the second
    call from being influenced by the first, and there is no single object a
    reviewer can inspect to confirm the model kept the grains apart. With one
    labelled prompt and one DualLensPlan, a plan that applies the organization
    threshold to the license table is visible as a mismatch between a plan
    and the contract it cites, and GovernedPlan's validator rejects it.

(b) The planner is bound with `.with_structured_output(DualLensPlan)`. The
    alternative is to ask for SQL in free text and parse it. Structured output
    binding means the model's reply is validated by Pydantic before this
    module ever sees it: a malformed plan fails at the type boundary, not in
    the database and not in a downstream string parser.

LCEL is used deliberately for this stage. Retrieve context, render a prompt,
bind a schema, validate the result: that is a fixed, inspectable sequence of
steps, and a reader can trace it in one pass by following the pipe operator.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from semantic_contracts import (
    CONTRACT_REGISTRY,
    SEGMENT_REGISTRY,
    ContextResolutionError,
    DualLensPlan,
    GovernedPlan,
    MetricContract,
    ResolvedContext,
)

# The default Gemini model. Overridable so the notebook and tests can swap it.
DEFAULT_MODEL_NAME = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# Stage 1 — context retrieval (deterministic; no model involved)
# ---------------------------------------------------------------------------


def resolve_context(query: str) -> ResolvedContext:
    """Match the question against the contract registry and return both lenses.

    Fail-closed: if either lens has no matching contract the whole request is
    refused with ContextResolutionError. The pipeline never answers a value
    question under one definition while silently ignoring the other.
    """
    matched: dict[str, MetricContract] = {}
    for contract in CONTRACT_REGISTRY:
        if contract.matches(query):
            # First match per lens wins; the registry is small and curated.
            matched.setdefault(contract.lens, contract)

    missing = [lens for lens in ("department", "enterprise") if lens not in matched]
    if missing:
        raise ContextResolutionError(
            f"No governed contract matches this question for lens(es): {', '.join(missing)}. "
            f"Question: {query!r}. Refusing to guess a definition."
        )

    department = matched["department"]
    enterprise = matched["enterprise"]
    return ResolvedContext(
        query=query,
        department_contract=department,
        enterprise_contract=enterprise,
        department_segment=SEGMENT_REGISTRY[department.contract_id],
        enterprise_segment=SEGMENT_REGISTRY[enterprise.contract_id],
    )


# A RunnableLambda so the retriever composes with the pipe operator like any other stage.
context_retriever: Runnable[str, ResolvedContext] = RunnableLambda(resolve_context).with_config(
    run_name="context_retriever"
)


# ---------------------------------------------------------------------------
# Stage 2 — prompt template
# ---------------------------------------------------------------------------

# The system message states the rules; the human message carries the labelled
# contracts. Threshold values are rendered from the contract, never typed here.
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query planner for a governed analytics system. You plan; you never compute.\n"
            "You will receive one business question and TWO metric contracts, one per lens.\n"
            "Return one query plan per lens. Rules:\n"
            "1. Each plan reads ONLY its own contract's source_table. Never join the two tables.\n"
            "2. Each plan selects the contract's entity_id_column, the metric_expression aliased as "
            "metric_value, and every profile attribute listed on the contract.\n"
            "3. Filter with the governed threshold rendered exactly as the contract states it. "
            "Do not invent, round, or adjust any numeric cutoff.\n"
            "4. Copy contract_id and version into the plan exactly.\n"
            "5. If a ratio is ever needed, the denominator must be COUNT(DISTINCT entity_id_column).\n"
            "6. Emit a single plain SELECT statement per lens with no trailing semicolon. The WHERE clause must be the "
            "governed threshold and nothing else, or AND-joined governed predicates. Do not use OR, NOT, UNION, WITH, "
            "HAVING, LIMIT or table functions; a plan that does is refused before it runs.\n"
            "7. Restrict the population only when the question explicitly asks. If it does, and a governed dimension "
            "covers the restriction, put it in dimension_filters (column and allowed values copied exactly from the "
            "contract) AND apply the same restriction in the SQL WHERE clause, AND-joined with the threshold, for both "
            "lenses. If the question adds a restriction no governed dimension covers, do NOT invent a filter: list it "
            "in unapplied_qualifiers.\n"
            "8. Most questions restrict nothing. Leave dimension_filters and unapplied_qualifiers empty unless the "
            "question clearly restricts the population. The phrase 'enterprise accounts' names the account base, "
            "not the enterprise customer tier.\n"
            "9. Use focus_attributes only when the question asks to concentrate on particular profile attributes; "
            "otherwise leave it empty. The SQL still selects every profile attribute.",
        ),
        (
            "human",
            "Business question:\n{query}\n\n"
            "Department lens contract:\n{department_contract}\n\n"
            "Enterprise lens contract:\n{enterprise_contract}\n\n"
            "Department segment: {department_segment}\n"
            "Enterprise segment: {enterprise_segment}",
        ),
    ]
)


def _prompt_inputs(inputs: Mapping[str, Any]) -> dict[str, str]:
    """Flatten the resolved context into the string slots the prompt template expects.

    The mapping in and out is LCEL's own carrier between RunnableParallel and
    ChatPromptTemplate; the typed objects live inside it and are what get read.
    """
    context: ResolvedContext = inputs["context"]
    return {
        "query": inputs["query"],
        "department_contract": context.department_contract.describe_for_prompt(),
        "enterprise_contract": context.enterprise_contract.describe_for_prompt(),
        "department_segment": (
            f"{context.department_segment.segment_name} — {context.department_segment.selection_rule} "
            f"(threshold: {context.department_segment.threshold_name})"
        ),
        "enterprise_segment": (
            f"{context.enterprise_segment.segment_name} — {context.enterprise_segment.selection_rule} "
            f"(threshold: {context.enterprise_segment.threshold_name})"
        ),
    }


# ---------------------------------------------------------------------------
# Stage 3 — planner and chain assembly
# ---------------------------------------------------------------------------


def default_chat_model(model_name: str = DEFAULT_MODEL_NAME) -> BaseChatModel:
    """Construct the Gemini chat model. Imported lazily so tests never need the provider package on the path.

    No sampling parameters are set: current Gemini models use fixed sampling
    defaults, and determinism in this pipeline comes from the validators and
    the database, not from the model's temperature.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set; cannot construct the planner model.")
    return ChatGoogleGenerativeAI(model=model_name)


def _bind_plan_to_context(inputs: Mapping[str, Any]) -> GovernedPlan:
    """Assemble the final GovernedPlan. Its validator checks every plan cites the contract that was resolved."""
    return GovernedPlan(context=inputs["context"], plan=inputs["plan"])


def build_planning_chain(chat_model: BaseChatModel) -> Runnable[str, GovernedPlan]:
    """Compose the LCEL chain: question -> resolved context -> prompt -> structured plan -> governed plan.

    The chat model is injected so tests can pass a FakeListChatModel or a
    RunnableLambda-backed stand-in; the composition is identical either way.
    """
    # Structured output binding: the model's reply is parsed into DualLensPlan
    # by Pydantic before anything downstream sees it.
    planner = chat_model.with_structured_output(DualLensPlan)

    return (
        {"query": RunnablePassthrough(), "context": context_retriever}
        | RunnablePassthrough.assign(plan=RunnableLambda(_prompt_inputs) | prompt_template | planner)
        | RunnableLambda(_bind_plan_to_context)
    ).with_config(run_name="governed_planning_chain")
