from __future__ import annotations

from dataclasses import dataclass

from vec2plan_bandit.paths.registry import POST, PRE, REGISTRY
from vec2plan_bandit.queries.loader import QueryRecord


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _article_filter_clause(kind: str, values: list[str], alias: str, *, post_fence: bool = False) -> str | None:
    """Predicates on the article row, shared by exact PRE and HNSW POST."""
    for field in ("license", "comments", "journal_ref", "report_no"):
        column = f"({alias}.{field} || '')" if post_fence else f"{alias}.{field}"
        if kind == f"{field}_eq":
            return f"{column} = {literal(values[0])}"
        if kind == f"{field}_like":
            return f"{column} LIKE {literal(values[0])}"
        if kind == f"{field}_in":
            return f"{column} IN (" + ",".join(literal(value) for value in values) + ")"
    day = f"{alias}.update_date::date"
    if post_fence:
        day = f"({day} + 0)"
    if kind == "update_date_eq":
        return f"{day} = {literal(values[0])}::date"
    if kind == "update_date_le":
        return f"{day} <= {literal(values[0])}::date"
    if kind == "update_date_ge":
        return f"{day} >= {literal(values[0])}::date"
    if kind == "update_date_between":
        return f"{day} BETWEEN {literal(values[0])}::date AND {literal(values[1])}::date"
    if kind == "update_date_outside":
        return f"({day} <= {literal(values[0])}::date OR {day} >= {literal(values[1])}::date)"
    return None


@dataclass(frozen=True)
class RenderedSQL:
    path_id: str
    setup_sql: tuple[str, ...]
    sql: str
    pre_strategy: str | None = None
    pre_join_strategy: str | None = None


def pre_query_has_join(query: QueryRecord) -> bool:
    """Whether PRE's eligible-set SQL contains a relational join/subquery join."""
    return any(
        str(predicate.get("type", "")).startswith(("author_", "category_", "submitter_"))
        for predicate in query.predicates
    )


def render_candidate_filter(query: QueryRecord) -> str:
    """Filter FAISS candidates in PostgreSQL without changing FAISS rank."""
    clauses = " AND ".join(_semijoin_filter_clauses(query))
    return (
        "SELECT a.id AS article_id FROM arxiv_exp.articles a "
        "WHERE a.id = ANY(%(candidate_ids)s::bigint[]) AND "
        f"{clauses};"
    )


def render_eligible_ids(query: QueryRecord) -> str:
    """Return the complete eligible ID set for a query's metadata predicates."""
    where_sql = " AND ".join(_semijoin_filter_clauses(query))
    return (
        "SELECT a.id AS article_id FROM arxiv_exp.articles a "
        f"WHERE {where_sql} ORDER BY a.id;"
    )


def _filter_parts(query: QueryRecord, alias: str = "a") -> tuple[list[str], list[str]]:
    joins: list[str] = []
    clauses: list[str] = [f"{alias}.{query.embedding_column} IS NOT NULL"]
    for index, predicate in enumerate(query.predicates):
        kind, values = predicate["type"], predicate["values"]
        suffix = str(index)
        direct = _article_filter_clause(kind, values, alias)
        if direct is not None:
            clauses.append(direct)
            continue
        if kind.startswith("submitter_"):
            joins.append(f"JOIN arxiv_raw.submitters s{suffix} ON s{suffix}.id = {alias}.submitter_id")
            operator = "=" if kind == "submitter_eq" else "IN"
            rhs = literal(values[0]) if operator == "=" else "(" + ",".join(literal(v) for v in values) + ")"
            clauses.append(f"s{suffix}.name {operator} {rhs}")
        elif kind.startswith("author_"):
            joins.extend([
                f"JOIN arxiv_raw.article_authors aa{suffix} ON aa{suffix}.article_id = {alias}.id",
                f"JOIN arxiv_raw.authors au{suffix} ON au{suffix}.id = aa{suffix}.author_id",
            ])
            if kind == "author_eq": clauses.append(f"au{suffix}.name = {literal(values[0])}")
            elif kind == "author_in": clauses.append(f"au{suffix}.name IN (" + ",".join(literal(v) for v in values) + ")")
            elif kind == "author_like": clauses.append(f"au{suffix}.name LIKE {literal(values[0])}")
        elif kind.startswith("category_"):
            joins.extend([
                f"JOIN arxiv_raw.article_categories ac{suffix} ON ac{suffix}.article_id = {alias}.id",
                f"JOIN arxiv_raw.categories c{suffix} ON c{suffix}.id = ac{suffix}.category_id",
            ])
            if kind == "category_eq": clauses.append(f"c{suffix}.code = {literal(values[0])}")
            else: clauses.append(f"c{suffix}.code IN (" + ",".join(literal(v) for v in values) + ")")
        elif kind == "latest_update":
            clauses.append(f"{alias}.update_date = (SELECT max(v.created) FROM arxiv_raw.versions v WHERE v.article_id = {alias}.id)")
        elif kind == "update_date_range":
            clauses.append(
                f"{alias}.update_date >= {literal(values[0])} AND "
                f"{alias}.update_date < {literal(values[1])}"
            )
        else:
            raise ValueError(f"unsupported predicate: {kind}")
    return joins, clauses


def _semijoin_filter_clauses(
    query: QueryRecord, alias: str = "a", *, fence_article_id: bool = False
) -> list[str]:
    """Render filters without multiplying article rows.

    Keeping the distance-ordered article scan as the outer relation is essential
    for pgvector iterative HNSW scans. JOIN + DISTINCT makes PostgreSQL sort the
    filtered relation and can silently turn the post path into an exact scan.
    """
    clauses: list[str] = [f"{alias}.{query.embedding_column} IS NOT NULL"]
    article_id = f"({alias}.id + 0)" if fence_article_id else f"{alias}.id"
    for index, predicate in enumerate(query.predicates):
        kind, values = predicate["type"], predicate["values"]
        suffix = str(index)
        direct = _article_filter_clause(kind, values, alias)
        if direct is not None:
            clauses.append(direct)
            continue
        if kind.startswith("submitter_"):
            if kind == "submitter_eq":
                condition = f"s{suffix}.name = {literal(values[0])}"
            elif kind == "submitter_in":
                condition = f"s{suffix}.name IN (" + ",".join(literal(v) for v in values) + ")"
            else:
                raise ValueError(f"unsupported predicate: {kind}")
            clauses.append(
                f"EXISTS (SELECT 1 FROM arxiv_raw.submitters s{suffix} "
                f"WHERE s{suffix}.id = {alias}.submitter_id AND {condition})"
            )
        elif kind.startswith("author_"):
            if kind == "author_eq":
                condition = f"au{suffix}.name = {literal(values[0])}"
            elif kind == "author_in":
                condition = f"au{suffix}.name IN (" + ",".join(literal(v) for v in values) + ")"
            elif kind == "author_like":
                condition = f"au{suffix}.name LIKE {literal(values[0])}"
            else:
                raise ValueError(f"unsupported predicate: {kind}")
            clauses.append(
                "EXISTS (SELECT 1 "
                f"FROM arxiv_raw.article_authors aa{suffix} "
                f"JOIN arxiv_raw.authors au{suffix} ON au{suffix}.id = aa{suffix}.author_id "
                f"WHERE aa{suffix}.article_id = {article_id} AND {condition})"
            )
        elif kind.startswith("category_"):
            if kind == "category_eq":
                condition = f"c{suffix}.code = {literal(values[0])}"
            elif kind == "category_in":
                condition = f"c{suffix}.code IN (" + ",".join(literal(v) for v in values) + ")"
            else:
                raise ValueError(f"unsupported predicate: {kind}")
            clauses.append(
                "EXISTS (SELECT 1 "
                f"FROM arxiv_raw.article_categories ac{suffix} "
                f"JOIN arxiv_raw.categories c{suffix} ON c{suffix}.id = ac{suffix}.category_id "
                f"WHERE ac{suffix}.article_id = {article_id} AND {condition})"
            )
        elif kind == "latest_update":
            clauses.append(
                f"{alias}.update_date = (SELECT max(v.created) FROM arxiv_raw.versions v "
                # Keep the versions.article_id B-tree lookup.  The +0 fence is
                # only needed on many-to-many author/category membership; here
                # removing the ordinary lookup would turn the correlated exact
                # predicate into an avoidable full versions scan per article.
                f"WHERE v.article_id = {alias}.id)"
            )
        elif kind == "update_date_range":
            clauses.append(
                f"{alias}.update_date >= {literal(values[0])} AND "
                f"{alias}.update_date < {literal(values[1])}"
            )
        else:
            raise ValueError(f"unsupported predicate: {kind}")
    return clauses


def _post_filter_clauses(query: QueryRecord, alias: str = "a") -> list[str]:
    """Render metadata as scalar InitPlans and filters on the HNSW scan.

    ``+ 0`` is an intentional planner fence: it is value preserving for bigint
    keys but prevents their B-tree indexes from winning over the distance index.
    Every generated post plan must still pass the EXPLAIN gate before collection.
    """
    empty = "ARRAY[]::bigint[]"
    clauses: list[str] = [f"{alias}.{query.embedding_column} IS NOT NULL"]
    for index, predicate in enumerate(query.predicates):
        kind, values = predicate["type"], predicate["values"]
        suffix = str(index)
        # Preserve values/NULLs while fencing newly installed metadata indexes.
        # POST must remain HNSW + filter, not silently become exact Hash PRE.
        direct = _article_filter_clause(kind, values, alias, post_fence=True)
        if direct is not None:
            clauses.append(direct)
            continue
        if kind.startswith("submitter_"):
            if kind == "submitter_eq":
                condition = f"s{suffix}.name = {literal(values[0])}"
            elif kind == "submitter_in":
                condition = f"s{suffix}.name IN (" + ",".join(literal(v) for v in values) + ")"
            else:
                raise ValueError(f"unsupported predicate: {kind}")
            ids = (
                f"SELECT array_agg(s{suffix}.id)::bigint[] "
                f"FROM arxiv_raw.submitters s{suffix} WHERE {condition}"
            )
            clauses.append(
                f"({alias}.submitter_id + 0) = ANY (COALESCE(({ids}), {empty}))"
            )
        elif kind.startswith("author_"):
            if kind == "author_eq":
                condition = f"au{suffix}.name = {literal(values[0])}"
            elif kind == "author_in":
                condition = f"au{suffix}.name IN (" + ",".join(literal(v) for v in values) + ")"
            elif kind == "author_like":
                condition = f"au{suffix}.name LIKE {literal(values[0])}"
            else:
                raise ValueError(f"unsupported predicate: {kind}")
            ids = (
                f"SELECT array_agg(aa{suffix}.article_id)::bigint[] "
                f"FROM arxiv_raw.article_authors aa{suffix} "
                f"JOIN arxiv_raw.authors au{suffix} ON au{suffix}.id = aa{suffix}.author_id "
                f"WHERE {condition}"
            )
            clauses.append(f"({alias}.id + 0) = ANY (COALESCE(({ids}), {empty}))")
        elif kind.startswith("category_"):
            if kind == "category_eq":
                condition = f"c{suffix}.code = {literal(values[0])}"
            elif kind == "category_in":
                condition = f"c{suffix}.code IN (" + ",".join(literal(v) for v in values) + ")"
            else:
                raise ValueError(f"unsupported predicate: {kind}")
            ids = (
                f"SELECT array_agg(ac{suffix}.article_id)::bigint[] "
                f"FROM arxiv_raw.article_categories ac{suffix} "
                f"JOIN arxiv_raw.categories c{suffix} ON c{suffix}.id = ac{suffix}.category_id "
                f"WHERE {condition}"
            )
            clauses.append(f"({alias}.id + 0) = ANY (COALESCE(({ids}), {empty}))")
        elif kind == "latest_update":
            ids = (
                "SELECT array_agg(candidate.id)::bigint[] "
                "FROM arxiv_exp.articles candidate "
                "JOIN (SELECT v.article_id, max(v.created) AS latest_version_date "
                "FROM arxiv_raw.versions v GROUP BY v.article_id) lv "
                "ON lv.article_id = candidate.id "
                "WHERE candidate.update_date = lv.latest_version_date"
            )
            clauses.append(f"({alias}.id + 0) = ANY (COALESCE(({ids}), {empty}))")
        elif kind == "update_date_range":
            clauses.append(
                f"({alias}.update_date + INTERVAL '0 days') >= {literal(values[0])} AND "
                f"({alias}.update_date + INTERVAL '0 days') < {literal(values[1])}"
            )
        else:
            raise ValueError(f"unsupported predicate: {kind}")
    return clauses


def render_pre(
    query: QueryRecord,
    *,
    index_mode: str = "no_index",
    index_strategy: str | None = None,
    join_strategy: str = "all",
    ids_only: bool = True,
) -> RenderedSQL:
    """Render PRE as exact search over the complete eligible relation.

    ``index`` and ``no_index`` only control ordinary PostgreSQL access paths
    used to construct the eligible set. The distance ordering happens outside a
    materialized relation, so neither mode can use a vector HNSW index.
    """

    normalized = REGISTRY.normalize(
        PRE,
        {
            "index_mode": index_mode,
            **({"index_strategy": index_strategy} if index_mode == "index" and index_strategy is not None else {}),
            "join_strategy": join_strategy,
        },
    )
    pre_where_sql = " AND ".join(_semijoin_filter_clauses(query))
    projection = (
        "article_id"
        if ids_only
        else "article_id, distance"
    )
    order = "distance, article_id"
    sql = f"""WITH eligible AS MATERIALIZED (
 SELECT a.id AS article_id, a.{query.embedding_column} <=> %(query_embedding)s::vector(384) AS distance
 FROM arxiv_exp.articles a
 WHERE {pre_where_sql}
)
SELECT {projection}
FROM eligible
ORDER BY {order} LIMIT {query.k};"""
    access_setup: tuple[str, ...] = ()
    if normalized["index_mode"] == "no_index":
        access_setup = (
            "SET LOCAL enable_seqscan = on",
            "SET LOCAL enable_indexscan = off",
            "SET LOCAL enable_indexonlyscan = off",
            "SET LOCAL enable_bitmapscan = off",
        )
    elif normalized["index_strategy"] == "optimizer":
        access_setup = (
            "SET LOCAL enable_seqscan = on", "SET LOCAL enable_indexscan = on",
            "SET LOCAL enable_indexonlyscan = on", "SET LOCAL enable_bitmapscan = on",
        )
    elif normalized["index_strategy"] in {"btree", "hash"}:
        access_setup = (
            "SET LOCAL enable_seqscan = off",
            "SET LOCAL enable_indexscan = on",
            "SET LOCAL enable_indexonlyscan = off",
            "SET LOCAL enable_bitmapscan = off",
        )
    elif normalized["index_strategy"] == "bitmap":
        access_setup = (
            "SET LOCAL enable_seqscan = off",
            "SET LOCAL enable_indexscan = off",
            "SET LOCAL enable_indexonlyscan = off",
            "SET LOCAL enable_bitmapscan = on",
        )
    join_flags = {
        "all": ("on", "on", "on"),
        "hash": ("on", "off", "off"),
        "merge": ("off", "on", "off"),
        "nested_loop": ("off", "off", "on"),
    }
    hash_join, merge_join, nestloop = join_flags[normalized["join_strategy"]]
    join_setup = (
        f"SET LOCAL enable_hashjoin = {hash_join}",
        f"SET LOCAL enable_mergejoin = {merge_join}",
        f"SET LOCAL enable_nestloop = {nestloop}",
    )
    strategy = "scan" if normalized["index_mode"] == "no_index" else normalized["index_strategy"]
    return RenderedSQL(
        PRE,
        access_setup + join_setup,
        sql,
        strategy,
        normalized["join_strategy"],
    )


def render_post(
    query: QueryRecord,
    *,
    ef_search: int = 100,
    max_scan_tuples: int = 100_000,
    iterative_scan: str = "strict_order",
    scan_mem_multiplier: float = 2.0,
    ids_only: bool = True,
) -> RenderedSQL:
    params = REGISTRY.normalize(
        POST,
        {
            "ef_search": ef_search,
            "max_scan_tuples": max_scan_tuples,
        },
    )
    if iterative_scan not in {"strict_order", "relaxed_order"}:
        raise ValueError(f"invalid fixed POST iterative_scan: {iterative_scan}")
    if scan_mem_multiplier <= 0:
        raise ValueError("POST scan_mem_multiplier must be positive")
    post_where_sql = " AND ".join(_post_filter_clauses(query))
    distance = f"a.{query.embedding_column} <=> %(query_embedding)s::vector(384)"
    projection = "a.id AS article_id" if ids_only else f"a.id AS article_id, {distance} AS distance"
    order = distance if ids_only else "distance"
    sql = f"""SELECT {projection}
FROM arxiv_exp.articles a
WHERE {post_where_sql}
ORDER BY {order} LIMIT {query.k};"""
    setup = (
        "SET LOCAL enable_seqscan = off",
        f"SET LOCAL hnsw.ef_search = {params['ef_search']}",
        f"SET LOCAL hnsw.iterative_scan = '{iterative_scan}'",
        f"SET LOCAL hnsw.max_scan_tuples = {params['max_scan_tuples']}",
        f"SET LOCAL hnsw.scan_mem_multiplier = {scan_mem_multiplier:g}",
    )
    return RenderedSQL(POST, setup, sql)


def render(
    query: QueryRecord,
    path_id: str,
    *,
    params: dict[str, object] | None = None,
    ids_only: bool = True,
) -> RenderedSQL:
    values = dict(params or {})
    if path_id == PRE:
        return render_pre(query, ids_only=ids_only, **values)
    if path_id == POST:
        return render_post(query, ids_only=ids_only, **values)
    raise ValueError(f"SQL is not rendered for path {path_id}")


def render_exact_ground_truth(query: QueryRecord, *, ids_only: bool = True) -> RenderedSQL:
    """Offline exact evaluator; intentionally not a selectable path."""

    rendered = render_pre(query, index_mode="no_index", join_strategy="all", ids_only=ids_only)
    return RenderedSQL("EXACT_GROUND_TRUTH", rendered.setup_sql, rendered.sql)
