"""Exact bounded-key driven PRE joins, after unused-aggregate semijoin reduction.

The limited input is kept intact. Distinct join keys prevent multiplicity
inflation; original GROUP BY/order/limits and identity projections remain.
No vector distances, query results or eligible row sets are cached.
"""
import hashlib

from sqlglot import exp


def bounded_key_joins(tree, *, maximum_keys=64):
    ctes={}
    for cte in tree.find_all(exp.CTE):
        if cte.alias_or_name in ctes:
            return []  # Shadowed CTE names need scope resolution first.
        ctes[cte.alias_or_name]=cte.this
    changes=[]
    for select in list(tree.find_all(exp.Select)):
        source=select.args.get('from_')
        if (not source or not isinstance(source.this,exp.Table) or select.args.get('joins')
                or not select.args.get('group') or select.args.get('having')
                or select.args.get('with_') or select.args.get('qualify')
                or any(p.find(exp.AggFunc) or p.find(exp.Window) for p in select.expressions)):
            continue
        base=source.this;alias=base.alias_or_name
        where=select.args.get('where')
        if not where:
            continue
        predicates=list(where.this.flatten()) if isinstance(where.this,exp.And) else [where.this]
        semi=[p for p in predicates if isinstance(p,exp.Exists)]
        if len(semi)!=1:
            continue
        exists=semi[0].this
        child_source=exists.args.get('from_') if isinstance(exists,exp.Select) else None
        if (not child_source or not isinstance(child_source.this,exp.Table)
                or exists.args.get('joins') or any(exists.args.get(k) for k in
                ('group','having','limit','offset','order','with_','distinct'))):
            continue
        ref=child_source.this;bounded=ctes.get(ref.name)
        if ref.db or ref.catalog:
            continue
        limit=bounded.args.get('limit') if isinstance(bounded,exp.Select) else None
        if (not limit or not isinstance(limit.expression,exp.Literal) or limit.expression.is_string
                or not limit.expression.this.isdigit() or not 1<=int(limit.expression.this)<=maximum_keys
                or bounded.args.get('offset') or not bounded.args.get('order')):
            continue
        condition=exists.args.get('where')
        eq=condition.this if condition else None
        if not isinstance(eq,exp.EQ) or not all(isinstance(c,exp.Column) for c in (eq.this,eq.expression)):
            continue
        columns={c.table:c for c in (eq.this,eq.expression)}
        if alias==ref.alias_or_name or set(columns)!={alias,ref.alias_or_name}:
            continue
        others=[p for p in predicates if p is not semi[0]]
        if any(p.find(exp.Select) or any(c.table!=alias for c in p.find_all(exp.Column))
               # sqlglot also classifies boolean AND/OR/NOT as Func. These
               # deterministic predicates are safe inside the lateral input.
               or any(not isinstance(f,(exp.Cast,exp.And,exp.Or,exp.Not,exp.Between))
                      for f in p.find_all(exp.Func)) for p in others):
            continue
        # Only qualified base-column projections/grouping/order are supported.
        local=[n for n in select.walk() if n.find_ancestor(exp.Select) is select]
        if (any(isinstance(n,exp.Star) for n in local)
                or any(isinstance(n,exp.Column) and n.table!=alias for n in local
                       if n.find_ancestor(exp.Exists) is None)
                or any(isinstance(n,exp.Var) for n in local)):
            continue
        needed=sorted({n.name for n in local if isinstance(n,exp.Column) and n.table==alias})
        if not needed:
            continue
        name='__v2p_keys_'+hashlib.sha256(select.sql().encode()).hexdigest()[:10]
        if name in {n.alias_or_name for n in tree.find_all(exp.Table,exp.Subquery,exp.CTE)}:
            continue
        keys=(exp.select(exp.alias_(columns[ref.alias_or_name].copy(),'__key',quoted=True))
              .distinct().from_(ref.copy()))
        inner=exp.select(*[exp.alias_(exp.column(n,table=alias,quoted=True),n,quoted=True) for n in needed]).from_(base.copy())
        inner.where(exp.EQ(this=columns[alias].copy(),expression=exp.column('__key',table=name,quoted=True)),copy=False)
        for p in others:
            inner.where(p.copy(),append=True,copy=False)
        # OFFSET 0 is a semantics-preserving boundary for parameterized access.
        inner.set('offset',exp.Offset(expression=exp.Literal.number(0)))
        select.set('from_',exp.From(this=exp.Subquery(this=keys,
            alias=exp.TableAlias(this=exp.to_identifier(name,quoted=True)))))
        lateral=exp.Lateral(this=exp.Subquery(this=inner),
            alias=exp.TableAlias(this=exp.to_identifier(alias,quoted=True)))
        select.set('joins',[exp.Join(this=lateral,kind='CROSS')])
        select.set('where',None)
        changes.append(dict(kind='bounded_key_lateral_semijoin',cte=ref.name,
            maximum_keys=int(limit.expression.this),distinct_keys=True,
            grouping_preserved=True,limit_boundary_preserved=True,cached_answers=False))
    return changes
