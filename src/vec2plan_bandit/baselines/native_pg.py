from __future__ import annotations

import re

from vec2plan_bandit.paths.sql import RenderedSQL, _semijoin_filter_clauses
from vec2plan_bandit.queries.loader import QueryRecord


_FROM_RELATION = re.compile(
    r'\bFROM\s+(?:"[^"]+"\.)?(?P<table>"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)'
    r'(?:\s+(?:AS\s+)?(?P<alias>"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?',
    re.IGNORECASE,
)
_LEMBED = re.compile(
    r"lembed\(\s*'[^']+'\s*,\s*'(?:''|[^'])*'\s*\)",
    re.IGNORECASE,
)
_LIMIT = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _top_level_keyword(sql: str, keyword: str, *, start: int = 0) -> int:
    """Find a SQL keyword outside parentheses and quoted strings."""
    target = keyword.upper()
    depth = 0
    quote: str | None = None
    index = start
    while index < len(sql):
        char = sql[index]
        if quote:
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced SQL parentheses")
        elif depth == 0 and sql[index:index + len(target)].upper() == target:
            left_ok = index == 0 or not (sql[index - 1].isalnum() or sql[index - 1] == "_")
            end = index + len(target)
            right_ok = end == len(sql) or not (sql[end].isalnum() or sql[end] == "_")
            if left_ok and right_ok:
                return index
        index += 1
    return -1


def _first_projection_end(projection: str) -> int:
    depth = 0
    quote: str | None = None
    for index, char in enumerate(projection):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return index
    return len(projection)


def render_native_pg(query: QueryRecord) -> RenderedSQL:
    """Minimally adapt the source SQL into an ID-only native PG baseline.

    The original joins, filters, ordering and LIMIT remain.  PostgreSQL receives
    no path-specific planner settings.  Native PG is never registered as a
    selectable path or assigned an action_id.
    """
    # The four arXiv source files use an annotation `a.k = N` instead of a
    # PostgreSQL LIMIT and contain an embedding MATCH expression. Translate
    # their already-parsed predicates into ordinary PostgreSQL, leaving the
    # planner and pgvector settings at their native defaults.
    if " MATCH " in query.source_sql.upper() or not _LIMIT.search(query.source_sql):
        where_sql = " AND ".join(_semijoin_filter_clauses(query))
        sql = (
            "SELECT a.id AS article_id FROM arxiv_exp.articles a "
            f"WHERE {where_sql} "
            f"ORDER BY a.{query.embedding_column} <=> %(query_embedding)s::vector(384), a.id "
            f"LIMIT {query.k};"
        )
        return RenderedSQL("PG_NATIVE_SQL", (), sql)
    sql = query.source_sql.replace("<->", "<=>")
    select_at = _top_level_keyword(sql, "SELECT")
    from_at = _top_level_keyword(sql, "FROM", start=select_at + 6)
    if select_at < 0 or from_at < 0:
        raise ValueError(f"{query.query_id}: source SQL has no outer SELECT/FROM")
    match = _FROM_RELATION.match(sql, from_at)
    if match is None:
        raise ValueError(f"{query.query_id}: unsupported outer FROM relation")
    alias = match.group("alias")
    if alias and alias.strip('"').upper() in {"WHERE", "JOIN", "ORDER", "LIMIT", "GROUP", "HAVING"}:
        alias = None
    relation = alias or match.group("table")
    article_ref = f'"{relation.strip(chr(34))}"'
    sql = _LEMBED.sub("%(query_embedding)s::vector(384)", sql)
    limits = list(_LIMIT.finditer(sql))
    if not limits:
        raise ValueError(f"{query.query_id}: source SQL has no LIMIT")
    last = limits[-1]
    sql = sql[: last.start()] + f"LIMIT {query.k}" + sql[last.end() :]
    # Replace only the first outer projected value.  Keeping the remaining
    # projection preserves aliases such as ``distance`` used by ORDER BY.  A
    # top-level scan avoids rewriting SELECT statements inside CTEs.
    select_at = _top_level_keyword(sql, "SELECT")
    from_at = _top_level_keyword(sql, "FROM", start=select_at + 6)
    projection = sql[select_at + 6:from_at]
    first_end = _first_projection_end(projection)
    suffix = projection[first_end:] if first_end < len(projection) else " "
    replacement = f' {article_ref}."id" AS article_id{suffix}'
    sql = sql[:select_at + 6] + replacement + sql[from_at:]
    # The wrapper keeps distance available to the original ORDER BY while the
    # client receives only article IDs.
    wrapped = f"SELECT native_result.article_id FROM (\n{sql.rstrip().rstrip(';')}\n) AS native_result;"
    return RenderedSQL("PG_NATIVE_SQL", (), wrapped)


def execute_native_pg(collector: object, query: QueryRecord, vector_literal: str) -> tuple[list[int], float]:
    """Execute the external baseline through the collector's read-only PG client."""
    rendered = render_native_pg(query)
    pg = getattr(collector, "_pg", None)
    if pg is None:
        raise TypeError("collector does not expose the PostgreSQL execution boundary")
    return pg(rendered, vector_literal)
