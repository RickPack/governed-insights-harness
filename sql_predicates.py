"""
sql_predicates.py — parse-based inspection of planned SQL.

WHY THIS DESIGN
---------------
The first validator rules read the SQL text with regular expressions. A regex
finds the governed cutoff in `WHERE license_revenue >= 5000 OR 1 = 1` and
passes it, even though the predicate no longer selects anything: every license
comes back. It also passes `NOT (license_revenue >= 5000)`, which selects the
complement. Text matching cannot tell a predicate that is present from a
predicate that is in force.

This module asks DuckDB's own parser for the syntax tree instead, and accepts
only a small, provable subset of SQL:

  * the top-level statement is one plain SELECT (no UNION, no CTE);
  * FROM reads only base tables, joins and subqueries (no table functions);
  * every WHERE clause is a conjunction (AND) of simple predicates, each one a
    column compared with a constant;
  * no HAVING, QUALIFY, SAMPLE or LIMIT, all of which change which rows count
    as the segment without appearing as a threshold.

Anything outside that subset raises UnsupportedSql. The validator turns that
into a refusal with the reason, so an unusual plan is refused rather than
guessed at. The parser is used only to parse: the connection holds no data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import duckdb

# Parse-only connection. json_serialize_sql never binds names, so no tables exist here.
_PARSER = duckdb.connect(":memory:")

_COMPARISONS = {
    "COMPARE_EQUAL": "=",
    "COMPARE_GREATERTHAN": ">",
    "COMPARE_GREATERTHANOREQUALTO": ">=",
    "COMPARE_LESSTHAN": "<",
    "COMPARE_LESSTHANOREQUALTO": "<=",
}
# `5000 <= license_revenue` means `license_revenue >= 5000`.
_FLIPPED = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "="}
_INTEGER_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
}
_FLOAT_TYPES = {"FLOAT", "DOUBLE"}


class UnsupportedSql(ValueError):
    """The plan uses SQL this validator cannot prove safe. It is refused, never guessed at."""


@dataclass(frozen=True)
class NumericPredicate:
    """`column <operator> number`, with the column on the left."""

    column: str
    operator: str
    value: float


@dataclass(frozen=True)
class CategoricalPredicate:
    """`column = 'text'` or `column IN ('a', 'b')`."""

    column: str
    values: tuple[str, ...]


Predicate = NumericPredicate | CategoricalPredicate


@dataclass(frozen=True)
class SqlInspection:
    """What a plan's SQL actually does, read from its syntax tree."""

    top_level: tuple[Predicate, ...]
    nested: tuple[Predicate, ...]
    tables: tuple[str, ...]


def _walk(obj: Any) -> Iterator[dict]:
    """Every dict anywhere in a nested JSON structure."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)


def parse_select(sql: str) -> dict:
    """The syntax tree of exactly one plain SELECT, or UnsupportedSql."""
    try:
        raw = _PARSER.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
    except duckdb.Error as exc:  # pragma: no cover - json_serialize_sql reports errors in-band
        raise UnsupportedSql(f"SQL could not be parsed: {exc}") from exc
    tree = json.loads(raw)
    if tree.get("error"):
        raise UnsupportedSql(f"SQL does not parse: {tree.get('error_message')}")
    statements = tree["statements"]
    if len(statements) != 1:
        raise UnsupportedSql("exactly one statement is allowed")
    node = statements[0]["node"]
    if node["type"] != "SELECT_NODE":
        raise UnsupportedSql("set operations (UNION, INTERSECT, EXCEPT) are not supported; use one plain SELECT")
    return node


def _is_column(expr: dict) -> bool:
    return expr.get("class") == "COLUMN_REF"


def _is_constant(expr: dict) -> bool:
    return expr.get("class") == "CONSTANT"


def _column_name(expr: dict) -> str:
    return expr["column_names"][-1].lower()


def _constant_value(expr: dict) -> float | str:
    """A constant as a Python number or string. NULL and unusual types are refused."""
    payload = expr["value"]
    if payload["is_null"]:
        raise UnsupportedSql("comparison with NULL is not supported")
    type_id = payload["type"]["id"]
    value = payload["value"]
    if type_id == "VARCHAR":
        return str(value)
    if type_id in _INTEGER_TYPES:
        return float(value)
    if type_id == "DECIMAL":
        return float(value) / (10 ** payload["type"]["type_info"]["scale"])
    if type_id in _FLOAT_TYPES:
        return float(value)
    raise UnsupportedSql(f"constant of type {type_id} is not supported in a filter")


def _describe(expr: dict) -> str:
    """A short reason naming the construct that is not allowed."""
    kind = expr.get("type", "?")
    if kind == "CONJUNCTION_OR":
        return "OR is not allowed; a segment filter must be AND-joined governed predicates"
    if kind in ("COMPARE_NOT_IN", "COMPARE_NOTEQUAL", "COMPARE_DISTINCT_FROM", "COMPARE_NOT_DISTINCT_FROM"):
        return f"{kind} is not allowed; only = and IN on text, and governed comparisons on numbers"
    if kind == "FUNCTION":
        return f"function or operator {expr.get('function_name')!r} is not allowed in a filter"
    if expr.get("class") == "OPERATOR":
        return f"{kind} is not allowed in a filter"
    return f"{expr.get('class', '?')}/{kind} is not allowed in a filter"


def classify(expr: dict) -> Predicate:
    """One WHERE conjunct as a NumericPredicate or CategoricalPredicate, or UnsupportedSql."""
    kind = expr.get("type")
    if kind in _COMPARISONS:
        operator = _COMPARISONS[kind]
        left, right = expr["left"], expr["right"]
        if _is_column(left) and _is_constant(right):
            column, constant = left, right
        elif _is_constant(left) and _is_column(right):
            column, constant, operator = right, left, _FLIPPED[operator]
        else:
            raise UnsupportedSql("a comparison must be between one column and one constant")
        value = _constant_value(constant)
        if isinstance(value, str):
            if operator != "=":
                raise UnsupportedSql(f"text can only be compared with = or IN, not {operator}")
            return CategoricalPredicate(_column_name(column), (value,))
        return NumericPredicate(_column_name(column), operator, value)
    if kind == "COMPARE_IN":
        first, rest = expr["children"][0], expr["children"][1:]
        if not _is_column(first) or not all(_is_constant(item) for item in rest):
            raise UnsupportedSql("IN must compare one column with a list of constants")
        values = [_constant_value(item) for item in rest]
        if not all(isinstance(v, str) for v in values):
            raise UnsupportedSql("IN lists must contain text values")
        return CategoricalPredicate(_column_name(first), tuple(str(v) for v in values))
    raise UnsupportedSql(_describe(expr))


def _conjuncts(expr: dict | None) -> list[dict]:
    """Flatten nested AND nodes. Anything else, including OR, is a single conjunct for classify to judge."""
    if expr is None:
        return []
    if expr.get("type") == "CONJUNCTION_AND":
        flat: list[dict] = []
        for child in expr["children"]:
            flat.extend(_conjuncts(child))
        return flat
    return [expr]


def _from_tables(table: dict | None) -> list[str]:
    """Base table names in a FROM tree. Table functions, VALUES and the like are refused."""
    if table is None:
        return []
    kind = table["type"]
    if kind == "BASE_TABLE":
        if table.get("schema_name") or table.get("catalog_name"):
            raise UnsupportedSql("schema-qualified table names are not supported")
        return [table["table_name"].lower()]
    if kind == "JOIN":
        return _from_tables(table["left"]) + _from_tables(table["right"])
    if kind == "SUBQUERY":
        return []  # its SELECT is inspected on its own
    raise UnsupportedSql(f"FROM clause of kind {kind} is not supported; read base tables only")


def inspect_sql(sql: str) -> SqlInspection:
    """Parse a plan's SQL and return its predicates and tables, or raise UnsupportedSql."""
    root = parse_select(sql)
    top_level: list[Predicate] = []
    nested: list[Predicate] = []
    tables: list[str] = []

    for node in _walk(root):
        node_type = node.get("type")
        if node_type == "SET_OPERATION_NODE":
            raise UnsupportedSql("set operations (UNION, INTERSECT, EXCEPT) are not supported; use one plain SELECT")
        if node_type != "SELECT_NODE":
            continue
        if node["cte_map"]["map"]:
            raise UnsupportedSql("WITH (common table expressions) is not supported; use one plain SELECT")
        if node.get("having") is not None:
            raise UnsupportedSql("HAVING is not supported; it filters rows outside the governed predicates")
        if node.get("qualify") is not None:
            raise UnsupportedSql("QUALIFY is not supported")
        if node.get("sample") is not None:
            raise UnsupportedSql("SAMPLE is not supported; a segment must not be a random subset")
        for modifier in node.get("modifiers", []):
            if "LIMIT" in modifier["type"]:
                raise UnsupportedSql("LIMIT is not supported; it silently truncates the segment")
        tables.extend(_from_tables(node.get("from_table")))
        target = top_level if node is root else nested
        target.extend(classify(conjunct) for conjunct in _conjuncts(node.get("where_clause")))

    return SqlInspection(top_level=tuple(top_level), nested=tuple(nested), tables=tuple(tables))
