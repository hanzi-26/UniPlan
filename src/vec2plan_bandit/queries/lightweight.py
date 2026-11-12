"""Shared SQL structure cache and conservative ANN simplification proof.

Only syntax and parameterized plans are cached. Vectors, scalar statistics,
eligibility sets, predictions and query results never enter these caches.
"""
from dataclasses import dataclass
from functools import lru_cache

import sqlglot
from sqlglot import exp

from vec2plan_bandit.queries.cosine_sql import DIALECT


@lru_cache(maxsize=1024)
def _parsed(source_sql):
    statements = sqlglot.parse(source_sql, read=DIALECT)
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError('only a single SELECT is accepted')
    return statements[0]


def parsed_select(source_sql):
    """Return an isolated tree: SQL adapters are allowed to rewrite their copy."""
    return _parsed(source_sql).copy()


@lru_cache(maxsize=2048)
def _routing_select(source_sql):
    try:
        return _parsed(source_sql)
    except sqlglot.errors.ParseError:
        from vec2plan_bandit.queries.arxiv_sql_adapter import structural_sql
        return _parsed(structural_sql(source_sql))


def routing_select(source_sql):
    """Isolated structure tree, including validated arXiv MATCH specifications.

    Keep this separate from parsed_select: structural normalization must never
    authorize a SQL rewrite, execution shortcut or unfiltered retrieval proof.
    """
    return _routing_select(source_sql).copy()


@lru_cache(maxsize=2048)
def sql_static_facts(source_sql):
    """Query-only feature terms; live selectivity and embeddings stay outside."""
    tree = _parsed(source_sql)
    return (len(list(tree.find_all(exp.Where))), len(list(tree.find_all(exp.Join))),
            len(list(tree.find_all(exp.Literal))), tree.find(exp.Between) is not None,
            tree.find(exp.Order) is not None)


def cached_plan(query, key, compile_plan):
    """Bounded per-query parameterized plan cache, invalidated by source edits."""
    state = getattr(query, '_lightweight_plans', None)
    if state is None or state[0] != query.source_sql:
        state = (query.source_sql, {})
        # QueryRecord is frozen; this cache is used only by mutable SQL adapters.
        query._lightweight_plans = state
    plans = state[1]
    if key not in plans:
        value = compile_plan()
        if len(plans) >= 32:
            plans.clear()
        plans[key] = value
    return plans[key]


@dataclass(frozen=True)
class RetrievalSemantics:
    direct_unfiltered: bool
    reason: str
    table: str | None = None
    column: str | None = None
    k: int | None = None
    order_required: bool = False


@lru_cache(maxsize=2048)
def retrieval_semantics(source_sql):
    """Prove a single unfiltered vector top-k, optionally through plain CTEs.

    No selectivity estimate is used as a proof. JOINs, scalar predicates,
    aggregates, offsets, subqueries and different output ordering fail closed.
    A runtime must additionally supply a valid snapshot row-identity mapping.
    This proves ID-only result semantics, not arbitrary output projections.
    """
    def reject(reason):
        return RetrievalSemantics(False, reason)
    try:
        tree = _parsed(source_sql)
        distances = list(tree.find_all(exp.Distance))
        if len(distances) != 1:
            return reject('not_single_vector')
        distance = distances[0]
        if (not isinstance(distance.this, exp.Column) or
                not isinstance(distance.expression, exp.Anonymous) or
                distance.expression.name.lower() != 'lembed'):
            return reject('unsupported_vector_expression')
        if any(tree.find(kind) for kind in (exp.Where, exp.Join, exp.Group, exp.Having,
                exp.Window, exp.Offset, exp.Subquery, exp.Unnest, exp.Lateral)):
            return reject('relational_work_required')
        if tree.find(exp.AggFunc):
            return reject('aggregation_required')
        if any(not (isinstance(f, exp.Anonymous) and f.name.lower() == 'lembed')
               for f in tree.find_all(exp.Func)):
            return reject('projection_function_requires_evaluation')
        ctes = {cte.alias_or_name: cte.this for cte in tree.find_all(exp.CTE)}
        if tree.args.get('with_') and tree.args['with_'].args.get('recursive'):
            return reject('recursive_cte')
        vector_select = distance.find_ancestor(exp.Select)
        aliases = {e.alias for e in vector_select.expressions
                   if isinstance(e, exp.Alias) and e.this == distance}
        limit = vector_select.args.get('limit')
        if limit is None or not isinstance(limit.expression, exp.Literal):
            return reject('missing_vector_limit')
        k = int(limit.expression.this)
        if k < 1:
            return reject('invalid_limit')
        current, visited, reached_vector = tree, set(), False
        while True:
            if not isinstance(current, exp.Select) or id(current) in visited:
                return reject('unsupported_lineage')
            visited.add(id(current))
            if current.args.get('distinct') or current.args.get('qualify'):
                return reject('row_set_operation')
            if current is vector_select:
                reached_vector = True
            elif reached_vector:
                return reject('input_cte_requires_evaluation')
            current_limit = current.args.get('limit')
            if current_limit and int(current_limit.expression.this) != k:
                return reject('different_outer_limit')
            order = current.args.get('order')
            if current is vector_select and not order:
                return reject('missing_distance_order')
            for ordered in order.expressions if order else ():
                value = ordered.this
                if ordered.args.get('desc') or not (value == distance or
                        isinstance(value, exp.Column) and value.name in aliases):
                    return reject('different_result_order')
            source = current.args.get('from_')
            if source is None or not isinstance(source.this, exp.Table):
                return reject('unsupported_lineage')
            relation = source.this
            if relation.name not in ctes:
                if not reached_vector or distance.this.table not in ('', relation.alias_or_name):
                    return reject('different_vector_relation')
                return RetrievalSemantics(True, 'single_unfiltered_topk', relation.name,
                                          distance.this.name, k, bool(tree.args.get('order')))
            current = ctes[relation.name]
    except (ValueError, TypeError, sqlglot.errors.SqlglotError):
        return reject('unsupported_sql')


def simple_retrieval(query):
    semantics = retrieval_semantics(query.source_sql)
    blocks = getattr(query, 'blocks', ())
    if (not semantics.direct_unfiltered or len(blocks) != 1 or
            query.k != semantics.k or blocks[0].k != semantics.k or
            blocks[0].table != semantics.table or blocks[0].column != semantics.column):
        return RetrievalSemantics(False, semantics.reason if not semantics.direct_unfiltered
                                  else 'adapter_semantics_mismatch')
    return semantics
