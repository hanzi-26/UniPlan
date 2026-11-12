"""Exact PRE reduction for a single vector relation followed by inner JOINs.

Every retained vector row has at least one qualifying joined row. Consequently
the first k joined rows can involve at most the first k qualifying vector rows,
provided distance and the vector row identity lead the final ordering. Restore
the original JOIN afterwards: never deduplicate its output or cache answers.
"""
import hashlib

from sqlglot import exp, parse_one


def reduce_join_topk(tree, query):
    from .relational_pre_assets import expand_identity_vars
    expand_identity_vars(tree)
    if (len(query.blocks) != 1 or tree.args.get('with_') or
            any(tree.args.get(k) for k in ('group', 'having', 'distinct', 'offset', 'qualify')) or
            tree.find(exp.AggFunc) or tree.find(exp.Window) or tree.find(exp.Subquery)):
        return []
    source, joins, order, limit = (tree.args.get(k) for k in ('from_', 'joins', 'order', 'limit'))
    if not source or not isinstance(source.this, exp.Table) or not joins or not order or not limit:
        return []
    base, block = source.this, query.blocks[0]
    if (base.name != block.table or base.alias_or_name != block.alias or
            not isinstance(limit.expression, exp.Literal) or not limit.expression.this.isdigit() or
            int(limit.expression.this) != block.k or any(j.args.get('side') or
            j.args.get('kind', '').upper() not in ('', 'INNER') or
            not isinstance(j.this, exp.Table) or j.args.get('on') is None for j in joins)):
        return []
    # Exactly the existing distance + complete joined identity ordering. Refuse
    # extra sort expressions, descending distances, and identities led by a
    # different relation. Identity ordering is text, matching the SQL adapter.
    if len(order.expressions) != 2 or order.expressions[0].args.get('desc'):
        return []
    first = order.expressions[0].this
    distance = next((v.parent for v in tree.find_all(exp.Var) if v.this == '%(v_0)s::vector'), None)
    if distance is None:
        return []
    if isinstance(first, exp.Column) and not first.table:
        projection = next((p for p in tree.expressions if p.alias == first.name), None)
        first = projection.this if projection is not None else first
    if first != distance or order.expressions[1].args.get('desc'):
        return []
    expected = parse_one(query.identity(tree), read='postgres')
    if order.expressions[1].this != expected:
        return []
    alias = base.alias_or_name
    # Require qualified columns: unqualified JOIN columns may bind differently
    # after moving the JOIN into EXISTS. Volatile functions cannot be repeated.
    if any(not c.table for c in tree.find_all(exp.Column) if c is not order.expressions[0].this):
        # The unqualified output alias used by ORDER BY is the sole exception.
        unknown = [c for c in tree.find_all(exp.Column) if not c.table and c is not order.expressions[0].this]
        if unknown:
            return []
    if any(f for f in tree.find_all(exp.Func) if not isinstance(f, (exp.Cast, exp.Array, exp.And, exp.Or, exp.Not))):
        return []
    needed = sorted({c.name for c in tree.find_all(exp.Column) if c.table == alias} - {'ctid'})
    if not needed or any(c not in query.columns[base.name] for c in needed):
        return []
    candidate = exp.select(*[exp.column(c, table=alias, quoted=True) for c in needed],
        exp.alias_(exp.column('ctid', table=alias, quoted=True), 'ctid', quoted=True)).from_(base.copy())
    exists = exp.select(exp.Literal.number(1)).from_(joins[0].this.copy())
    exists.set('joins', [j.copy() for j in joins[1:]])
    exists.where(joins[0].args['on'].copy(), copy=False)
    where = tree.args.get('where')
    predicates = list(where.this.flatten()) if where and isinstance(where.this, exp.And) else [where.this] if where else []
    for predicate in predicates:
        tables = {c.table for c in predicate.find_all(exp.Column)}
        if tables <= {alias}:
            candidate.where(predicate.copy(), append=True, copy=False)
        else:
            exists.where(predicate.copy(), append=True, copy=False)
    candidate.where(exp.Exists(this=exists), append=True, copy=False)
    candidate.set('order', exp.Order(expressions=[order.expressions[0].copy(),
        exp.Ordered(this=parse_one(f'"{alias.replace(chr(34), chr(34)*2)}".ctid::text', read='postgres'))]))
    # Use the distance itself rather than an output alias not projected here.
    candidate.args['order'].expressions[0].set('this', distance.copy())
    candidate.set('limit', limit.copy())
    name = '__v2p_pre_topk_' + hashlib.sha256(tree.sql().encode()).hexdigest()[:12]
    tree.set('with_', exp.With(expressions=[exp.CTE(this=candidate,
        alias=exp.TableAlias(this=exp.to_identifier(name, quoted=True)), materialized=True)]))
    tree.set('from_', exp.From(this=exp.Table(this=exp.to_identifier(name, quoted=True), alias=base.args.get('alias').copy()
        if base.args.get('alias') else exp.TableAlias(this=exp.to_identifier(alias, quoted=True)))))
    return [dict(kind='exact_topk_before_join', vector_alias=alias, limit=block.k,
        join_count=len(joins), preserves_join_multiplicity=True, cached_answers=False)]
