"""Exact PRE rewrites for immutable relational benchmark snapshots.

No vector candidates, query answers or distance results are cached. A verified
monotone, complete row-ID/CTID mapping permits scalar range predicates to use
PostgreSQL TID range scans instead of joining millions of mapping rows. The
boundary lookup remains inside the timed SQL statement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlglot import exp, parse_one

VERSION = 'relational-pre-monotone-tid-v1'


def prove_mapping(conn, schema, table):
    """Read-only full proof. Missing/partial/unordered mappings fail closed."""
    from psycopg import sql
    from vec2plan_bandit.execution.relation_lineage import ctid_mapping_lineage, relation_info
    lineage = ctid_mapping_lineage(conn, schema, table)
    if lineage is None:
        return None
    oid, _ = relation_info(conn, schema, table)
    definition = conn.execute('SELECT pg_get_viewdef(%s,true)', (oid,)).fetchone()[0]
    tree = parse_one(definition, read='postgres')
    base = tree.args['from_'].this
    mapping = tree.args['joins'][0].this
    # Require unmodified base columns, one CTID and one synthetic row_id.
    for item in tree.expressions:
        column = item.this if isinstance(item, exp.Alias) else item
        if not isinstance(column, exp.Column):
            return None
        if column.table == base.alias_or_name:
            if item.alias_or_name != column.name:
                return None
        elif not (column.table == mapping.alias_or_name and column.name == 'row_id'
                  and item.alias_or_name == 'row_id'):
            return None
    b = sql.Identifier(lineage['base_schema'], lineage['base_table'])
    m = sql.Identifier(lineage['mapping_schema'], lineage['mapping_table'])
    proof = conn.execute(sql.SQL(
        'WITH ordered AS (SELECT source_ctid,row_id,lag(source_ctid) OVER (ORDER BY row_id) AS previous FROM {}) '
        'SELECT count(*),count(*) FILTER (WHERE source_ctid IS NULL OR row_id IS NULL '
        'OR previous >= source_ctid) FROM ordered').format(m)).fetchone()
    # PK(source_ctid) is proven by lineage. Equal counts plus absence of a
    # missing base CTID proves a bijection, including rows with NULL vectors.
    count = conn.execute(sql.SQL('SELECT count(*) FROM {}').format(b)).fetchone()[0]
    missing = conn.execute(sql.SQL(
        'SELECT EXISTS(SELECT 1 FROM {} b WHERE NOT EXISTS('
        'SELECT 1 FROM {} m WHERE m.source_ctid=b.ctid))').format(b, m)).fetchone()[0]
    if proof[1] or proof[0] != count or missing:
        return None
    files = conn.execute('SELECT pg_relation_filenode(%s),pg_relation_filenode(%s)',
        (f'{lineage["base_schema"]}."{lineage["base_table"]}"',
         f'{lineage["mapping_schema"]}."{lineage["mapping_table"]}"')).fetchone()
    return dict(lineage, version=VERSION, complete=True, strictly_monotone=True,
                rows=count, base_filenode=files[0], mapping_filenode=files[1],
                view_sha256=hashlib.sha256(definition.encode()).hexdigest(),
                snapshot_contract='immutable import; rebuild proof after any data/CTID change')


def prune_identity_projection(tree):
    """Drop presentation-only output; retain grouping/order alias dependencies."""
    if (any(tree.args.get(k) for k in ('distinct', 'group', 'having', 'qualify'))
            or any(n.find_ancestor(exp.Select) is tree for kind in (exp.AggFunc,exp.Window)
                   for n in tree.find_all(kind))):
        return
    clauses = [tree.args.get(k) for k in ('order', 'group', 'having', 'qualify')]
    # Positional references depend on original output positions.
    if any(isinstance(v, exp.Literal) and not v.is_string for clause in clauses if clause
           for v in clause.expressions if not isinstance(v, exp.Ordered)):
        return
    if tree.args.get('order') and any(isinstance(o.this, exp.Literal) for o in tree.args['order'].expressions):
        return
    names = {c.name for clause in clauses if clause for c in clause.find_all(exp.Column) if not c.table}
    tree.set('expressions', [p for p in tree.expressions if p.alias_or_name == '__bird_ids' or p.alias_or_name in names])


def rewrite_tree(tree, *, db_id, proofs):
    """Conservative scalar-only transformation; preserve every JOIN and LIMIT."""
    prune_identity_projection(tree)
    changes = []
    cte_names = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    for select in list(tree.find_all(exp.Select)):
        source = select.args.get('from_')
        refs = ([source.this] if source else []) + [j.this for j in select.args.get('joins', [])]
        where = select.args.get('where')
        if where is None:
            continue
        for ref in refs:
            if not isinstance(ref, exp.Table) or ref.name in cte_names:
                continue
            proof = proofs.get(f'{db_id}.{ref.name}')
            if not proof or proof.get('version') != VERSION or not proof.get('complete') or not proof.get('strictly_monotone'):
                continue
            alias = ref.alias_or_name
            if any(c.name == 'row_id' and (c.table == alias or not c.table)
                   and (c.find_ancestor(exp.Select) is not select or not c.table and len(refs) != 1)
                   for c in select.find_all(exp.Column)):
                continue
            # Qualified row_id only, except for unambiguous single-relation SQL.
            columns = [c for c in select.find_all(exp.Column) if c.name == 'row_id'
                       and c.find_ancestor(exp.Select) is select
                       and (c.table == alias or not c.table and len(refs) == 1)]
            if not columns:
                continue
            valid = []
            for column in columns:
                predicate = column.parent
                if (not isinstance(predicate, (exp.LTE, exp.LT, exp.GTE, exp.GT, exp.EQ))
                        or predicate.this is not column or not isinstance(predicate.expression, exp.Literal)
                        or predicate.expression.is_string or not predicate.expression.this.isdigit()):
                    break
                ancestor = predicate.parent
                while isinstance(ancestor, (exp.And, exp.Paren)):
                    ancestor = ancestor.parent
                if ancestor is not where:
                    break  # OR, outer-join conditions and nested scopes stay intact.
                valid.append((predicate, column, predicate.expression.this))
            else:
                # A CTE/star or other unqualified row_id reference would require
                # synthesizing row IDs. Keep the original view in those cases.
                if not valid or any(isinstance(c, exp.Star) for c in select.find_all(exp.Star)):
                    continue
                for predicate, column, value in valid:
                    operator = {exp.LTE:'<=', exp.LT:'<', exp.GTE:'>=', exp.GT:'>', exp.EQ:'='}[type(predicate)]
                    ascending = operator in ('>=', '>')
                    # With strict monotonicity, <= the last matching CTID (or
                    # >= the first) is exactly the original row_id predicate.
                    outer = '=' if operator == '=' else '>=' if ascending else '<='
                    order = 'ASC' if ascending else 'DESC'
                    q = lambda s: '"' + s.replace('"', '""') + '"'
                    sub = (f'SELECT source_ctid FROM {q(proof["mapping_schema"])}.{q(proof["mapping_table"])} '
                           f'WHERE row_id {operator} {value} ORDER BY row_id {order} LIMIT 1')
                    replacement = parse_one(f'{q(alias)}.ctid {outer} ({sub})', read='postgres')
                    predicate.replace(replacement)
                old = ref.name
                ref.set('this', exp.to_identifier(proof['base_table'], quoted=True))
                ref.set('db', exp.to_identifier(proof['base_schema'], quoted=True))
                if not ref.args.get('alias'):
                    ref.set('alias', exp.TableAlias(this=exp.to_identifier(alias, quoted=True)))
                changes.append(dict(table=old, alias=alias, predicates=len(valid)))
    return changes


def render_pre_sql(worker, query, vectors):
    options = worker.config.get('relational_pre', {})
    if not options.get('enabled', False):
        statement, binds = query.render(vectors, exact=True)
        return statement, binds, {}
    path = Path(options['proof_artifact'])
    cache = worker.__dict__.setdefault('_relational_pre_proofs', {})
    if str(path) not in cache:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != options['proof_sha256']:
            raise ValueError('PRE mapping proof changed')
        artifact = json.loads(raw)
        if artifact.get('version') != VERSION or artifact.get('snapshot_id') != worker.config['snapshot_id']:
            raise ValueError('PRE mapping proof snapshot mismatch')
        cache[str(path)] = artifact
    def compile_template():
        from vec2plan_bandit.queries.cosine_sql import DIALECT
        tree = query.tree(ids_only=True)
        from vec2plan_bandit.execution.relational_pre_assets import rewrite_covering, rewrite_unused_group_joins
        artifact = cache[str(path)]
        grouped = rewrite_unused_group_joins(tree)
        key_joins=[]
        if options.get('bounded_key_join',False):
            from vec2plan_bandit.execution.relational_key_join import bounded_key_joins
            key_joins=bounded_key_joins(tree)
        late_join = []
        if options.get('exact_join_topk', False):
            from vec2plan_bandit.execution.relational_topk import reduce_join_topk
            late_join = reduce_join_topk(tree, query)
        compact = rewrite_covering(tree, query, artifact.get('coverings', {}))
        changes = rewrite_tree(tree, db_id=query.db_id, proofs=artifact['mappings'])
        for node in list(tree.find_all(exp.Var)):
            if node.this.startswith('%(v_'):
                distance = node.parent
                distance.replace(exp.Paren(this=exp.Add(this=exp.Paren(this=distance.copy()), expression=exp.Literal.number('0.0'))))
        return tree.sql(dialect=DIALECT), changes, compact, grouped, late_join, key_joins
    from vec2plan_bandit.queries.lightweight import cached_plan
    statement, changes, compact, grouped, late_join, key_joins = cached_plan(query,
        ('relational_pre', options['proof_sha256'], bool(options.get('exact_join_topk', False)),
         bool(options.get('bounded_key_join',False))), compile_template)
    return statement, query.binds(vectors), dict(pre_optimization=VERSION, tid_range_rewrites=changes,
                                               compact_relation_rewrites=compact,
                                               group_join_semijoins=grouped,
                                               exact_topk_join_reductions=late_join,
                                               bounded_key_joins=key_joins,
                                               cached_results=False, boundary_lookup='inside timed SQL')
