from __future__ import annotations

from typing import Any


def plan_nodes(payload: Any) -> list[dict[str, Any]]:
    root = payload[0]["Plan"] if isinstance(payload, list) else payload["Plan"]
    rows: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        rows.append(node)
        for child in node.get("Plans", []):
            visit(child)

    visit(root)
    return rows


def validate_pre_access_strategy(payload: Any, strategy: str, index_methods: dict[str, str]) -> None:
    """Fail if PostgreSQL ignored an experimental PRE access strategy."""
    nodes = plan_nodes(payload)
    indexed = [node for node in nodes if node.get("Node Type") in
               {"Index Scan", "Index Only Scan", "Bitmap Index Scan"}]
    all_names = [str(node["Index Name"]) for node in indexed if node.get("Index Name")]
    if any(index_methods.get(name) == "hnsw" for name in all_names):
        raise RuntimeError("PRE_USES_HNSW")
    # An index on authors/categories does not make the articles access path a
    # B-tree/Hash PRE.  Audit the eligible CTE's article alias specifically.
    producers = [node for node in nodes if "CTE eligible" in str(node.get("Subplan Name", ""))]
    access_nodes: list[dict[str, Any]] = []
    def visit(node: dict[str, Any]) -> None:
        access_nodes.append(node)
        for child in node.get("Plans", []):
            visit(child)
    for producer in producers:
        visit(producer)
    # The public-layout adapter expands arxiv_exp.articles into a subquery;
    # PostgreSQL then reports the physical table alias as "articles", not "a".
    article_nodes = [node for node in (access_nodes or nodes)
                     if node.get("Relation Name") == "articles"
                     and node.get("Alias") in {"a", "articles"}]
    article_indexed = [node for node in article_nodes if node.get("Node Type") in
                       {"Index Scan", "Index Only Scan"}]
    names = [str(node["Index Name"]) for node in article_indexed if node.get("Index Name")]
    if strategy == "scan":
        valid = bool(article_nodes) and not article_indexed and not any(
            node.get("Node Type") == "Bitmap Heap Scan" for node in article_nodes)
    elif strategy == "bitmap":
        valid = any(node.get("Node Type") == "Bitmap Heap Scan" for node in article_nodes)
    elif strategy in {"btree", "hash"}:
        valid = any(index_methods.get(name) == strategy for name in names)
        valid &= not any(node.get("Node Type") == "Bitmap Heap Scan" for node in article_nodes)
    else:
        raise ValueError(f"unknown PRE access strategy: {strategy}")
    if not valid:
        raise RuntimeError(
            f"PRE_STRATEGY_NOT_REALIZED:{strategy}; nodes="
            f"{[node.get('Node Type') for node in article_nodes]}; indexes={names}"
        )


def validate_pre_join_strategy(payload: Any, strategy: str) -> None:
    """Verify that a forced PRE join family is present in the eligible plan.

    ``all`` deliberately delegates the choice to PostgreSQL and therefore has
    no single node type to prove. Forced variants are accepted only when the
    requested physical join actually occurs; disabling GUCs alone is not proof.
    """
    expected = {
        "hash": "Hash Join",
        "merge": "Merge Join",
        "nested_loop": "Nested Loop",
    }
    if strategy == "all":
        return
    if strategy not in expected:
        raise ValueError(f"unknown PRE join strategy: {strategy}")
    nodes = plan_nodes(payload)
    producers = [
        node for node in nodes
        if "CTE eligible" in str(node.get("Subplan Name", ""))
    ]
    eligible_nodes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        eligible_nodes.append(node)
        for child in node.get("Plans", []):
            visit(child)

    for producer in producers:
        visit(producer)
    join_types = [
        str(node.get("Node Type"))
        for node in (eligible_nodes or nodes)
        if node.get("Node Type") in {"Hash Join", "Merge Join", "Nested Loop"}
    ]
    unexpected = sorted(set(join_types) - {expected[strategy]})
    if expected[strategy] not in join_types or unexpected:
        raise RuntimeError(
            f"PRE_JOIN_STRATEGY_NOT_REALIZED:{strategy}; joins={join_types}; "
            f"unexpected={unexpected}"
        )


def inspect_pre_plan(
    payload: Any, *, has_predicates: bool, embedding_column: str
) -> dict[str, Any]:
    nodes = plan_nodes(payload)
    hnsw = [
        node for node in nodes
        if "hnsw" in str(node.get("Index Name", "")).lower()
    ]
    expected = (
        "idx_arxiv_articles_abstract_hnsw_cosine"
        if embedding_column == "abstract_embedding"
        else "idx_arxiv_articles_title_hnsw_cosine"
    )
    names = [str(node.get("Index Name")) for node in hnsw]
    correct_column = not hnsw or all(name == expected for name in names)
    if not hnsw:
        classification = "EXACT_SCAN"
        placement = "FILTERED_RELATION_THEN_EXACT_DISTANCE_SORT"
    elif not has_predicates and correct_column:
        classification = "HNSW_PRE_FILTER_COMPATIBLE"
        placement = "NO_STRUCTURED_FILTER_PURE_VECTOR_SCAN"
    elif has_predicates and correct_column:
        # A global articles HNSW scan joined to the materialized eligible-ID CTE
        # applies membership after ANN ordering; it is not an ANN index over the
        # already-filtered relation.
        classification = "HNSW_POST_FILTER_BEHAVIOR"
        placement = "GLOBAL_HNSW_SCAN_THEN_ELIGIBLE_ID_JOIN"
    else:
        classification = "CONTRACT_VIOLATION"
        placement = "WRONG_EMBEDDING_INDEX"
    return {
        "classification": classification,
        "hnsw_used": bool(hnsw),
        "hnsw_index_names": names,
        "expected_hnsw_index": expected,
        "correct_embedding_index": correct_column,
        "filter_location": placement,
        "plan_node_types": [str(node.get("Node Type")) for node in nodes],
        "join_node_types": [
            str(node.get("Node Type")) for node in nodes
            if node.get("Node Type") in {"Hash Join", "Merge Join", "Nested Loop"}
        ],
        "rows_estimated": nodes[0].get("Plan Rows"),
        "estimated_total_cost": nodes[0].get("Total Cost"),
        "vector_operator": "<=>" if any("<=>" in str(node) for node in nodes) else None,
    }


def validate_pre_plan(
    payload: Any,
    *,
    has_predicates: bool,
    embedding_column: str,
    k: int,
) -> dict[str, Any]:
    inspected = inspect_pre_plan(
        payload,
        has_predicates=has_predicates,
        embedding_column=embedding_column,
    )
    nodes = plan_nodes(payload)
    cte_scans = [node for node in nodes if node.get("Node Type") == "CTE Scan"]
    materialized = any(node.get("CTE Name") == "eligible" for node in cte_scans)
    cte_producers = [
        node for node in nodes
        if "CTE eligible" in str(node.get("Subplan Name", ""))
    ]
    sort_keys = [str(key) for node in nodes for key in node.get("Sort Key", [])]
    root = nodes[0]
    limit_valid = root.get("Node Type") == "Limit" and int(root.get("Plan Rows", k)) <= k
    tie_break = any("article_id" in key or ".id" in key for key in sort_keys)
    distance_sort = any("<=>" in key for key in sort_keys)
    reasons = []
    if inspected["hnsw_used"]:
        reasons.append("PRE_USES_HNSW")
    if not inspected["correct_embedding_index"]:
        reasons.append("WRONG_EMBEDDING_INDEX")
    if not materialized or not cte_producers:
        reasons.append("MATERIALIZED_ELIGIBLE_RELATION_MISSING")
    if not distance_sort:
        reasons.append("DISTANCE_SORT_MISSING")
    if not tie_break:
        reasons.append("ARTICLE_ID_TIE_BREAK_MISSING")
    if not limit_valid:
        reasons.append("LIMIT_K_MISMATCH")
    filter_before_distance = (
        materialized
        and bool(cte_producers)
        and distance_sort
        and not inspected["hnsw_used"]
    )
    if not filter_before_distance:
        reasons.append("FILTER_NOT_PROVEN_BEFORE_DISTANCE")
    mode = "FILTERED_EXACT_PRE_V3" if has_predicates else "UNFILTERED_EXACT_PRE_V3"
    return {
        **inspected,
        "pre_mode": mode,
        "ann_allowed": False,
        "materialized_relation": materialized and bool(cte_producers),
        "filter_before_distance": filter_before_distance,
        "tie_break_present": tie_break,
        "limit_matches_k": limit_valid,
        "plan_valid": not reasons,
        "failure_reason": "|".join(dict.fromkeys(reasons)) or None,
    }
