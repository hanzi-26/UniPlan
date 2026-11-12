"""Offline narrow PG heap layouts for exact relational PRE, without ANN indexes."""
import hashlib
import json

from sqlglot import exp, parse_one

SOURCE_CTID = '__v2p_source_ctid'
VERSION = 'relational-pre-narrow-heap-v1'


def expand_identity_vars(tree):
    for node in list(tree.find_all(exp.Var)):
        # Query.identity also emits grouped-key arrays and CTE identity
        # concatenations. Leaving them opaque hides real column dependencies
        # from bounded-key PRE rewrites and covering projection analysis.
        if not node.this.startswith('%(v_') and (
                '.ctid' in node.this or node.this.startswith('ARRAY[') or '"__bird_ids"' in node.this):
            node.replace(parse_one(node.this, read='postgres'))


def rewrite_unused_group_joins(tree):
    """Reduce join fanout only when grouped aggregate values are not consumed.

    GROUP BY stays in place, so duplicate base rows and group identities retain
    their original meaning. Outer joins, HAVING, aggregate predicates/order,
    windows and bounded CTEs fail closed. This is a structural SQL rewrite.
    """
    from vec2plan_bandit.execution.relational_pre import prune_identity_projection
    prune_identity_projection(tree)
    expand_identity_vars(tree)
    changes=[]
    for cte in list(tree.find_all(exp.CTE)):
        select=cte.this
        if not isinstance(select,exp.Select) or not select.args.get('group'):
            continue
        if any(select.args.get(k) for k in ('having','order','limit','offset','qualify','distinct','with_')) or select.find(exp.Window):
            continue
        source=select.args.get('from_');joins=select.args.get('joins',[])
        if source is None or not isinstance(source.this,exp.Table) or not joins:
            continue
        base=source.this.alias_or_name
        groups=select.args['group'].expressions
        if not all(isinstance(c,exp.Column) and c.table==base for c in groups):
            continue
        if any(j.args.get('side') or j.args.get('kind','').upper() not in ('','INNER')
               or not isinstance(j.this,exp.Table) or j.args.get('on') is None for j in joins):
            continue
        aggregates=[p for p in select.expressions if p.find(exp.AggFunc)]
        if not aggregates or any(not p.alias for p in aggregates):
            continue
        names={p.alias for p in aggregates}
        aliases={ref.alias_or_name for ref in tree.find_all(exp.Table) if ref.name==cte.alias_or_name
                 and ref.find_ancestor(exp.CTE) is not cte}
        outside=[c for c in tree.find_all(exp.Column) if c.find_ancestor(exp.CTE) is not cte]
        if not aliases or any(c.name in names and (c.table in aliases or not c.table) for c in outside):
            continue
        # Non-aggregate projection expressions must depend only on group/base
        # columns; otherwise removing a joined relation could unbind a column.
        kept=[p for p in select.expressions if p not in aggregates]
        if any(c.table and c.table!=base for p in kept for c in p.find_all(exp.Column)):
            continue
        sub=exp.select(exp.Literal.number(1)).from_(joins[0].this.copy())
        sub.set('joins',[j.copy() for j in joins[1:]])
        sub.where(joins[0].args['on'].copy(),copy=False)
        select.set('joins',[])
        select.set('expressions',kept)
        select.where(exp.Exists(this=sub),append=True,copy=False)
        changes.append(dict(cte=cte.alias_or_name,removed_unused_aggregates=sorted(names),
                            join_count=len(joins),grouping_preserved=True))
    return changes


def covering_requirements(queries, *, minimum_points=200000):
    """Projection/filter/JOIN requirements only; no vectors, Recall or test labels."""
    from vec2plan_bandit.execution.relational_pre import prune_identity_projection
    result = {}
    for query in queries:
        tree = query.tree(ids_only=True)
        prune_identity_projection(tree)
        expand_identity_vars(tree)
        ctes = {c.alias_or_name for c in tree.find_all(exp.CTE)}
        for select in tree.find_all(exp.Select):
            source = select.args.get('from_')
            refs = ([source.this] if source else []) + [j.this for j in select.args.get('joins', [])]
            for ref in refs:
                if not isinstance(ref, exp.Table) or ref.name in ctes:
                    continue
                blocks = [b for b in query.blocks if b.table == ref.name and b.alias == ref.alias_or_name]
                if not blocks or len({b.column for b in blocks}) != 1:
                    continue
                block = blocks[0]
                if block.asset['estimated_points'] < minimum_points:
                    continue
                key = f'{query.db_id}.{ref.name}.{block.column}'
                cols = {c.name for c in select.find_all(exp.Column)
                        if c.table == ref.alias_or_name or not c.table and c.name in query.columns[ref.name]}
                cols.discard('ctid')
                cols &= set(query.columns[ref.name])
                cols.add(block.column)
                item = result.setdefault(key, dict(db_id=query.db_id, logical_table=ref.name,
                    vector_column=block.column, columns=set(), estimated_points=block.asset['estimated_points']))
                item['columns'].update(cols)
    for item in result.values():
        item['columns'] = sorted(item['columns'])
    return result


def scalar_requirements(queries, row_counts, *, minimum_points=200000):
    """Narrow large non-vector JOIN inputs too; wide vector heaps are costly."""
    from vec2plan_bandit.execution.relational_pre import prune_identity_projection
    from vec2plan_bandit.execution.relation_lineage import filter_join_columns
    index_columns=filter_join_columns(queries)
    result={}
    for query in queries:
        tree=query.tree(ids_only=True)
        prune_identity_projection(tree);rewrite_unused_group_joins(tree);expand_identity_vars(tree)
        ctes={c.alias_or_name for c in tree.find_all(exp.CTE)}
        for select in tree.find_all(exp.Select):
            source=select.args.get('from_');refs=([source.this] if source else [])+[j.this for j in select.args.get('joins',[])]
            for ref in refs:
                if not isinstance(ref,exp.Table) or ref.name in ctes or ref.name not in query.columns:
                    continue
                if any(b.table==ref.name and b.alias==ref.alias_or_name for b in query.blocks):
                    continue
                count=row_counts.get((query.db_id,ref.name),0)
                if count<minimum_points:
                    continue
                cols={c.name for c in select.find_all(exp.Column) if c.table==ref.alias_or_name
                      or not c.table and c.name in query.columns[ref.name]}
                cols &= set(query.columns[ref.name]);cols.discard('ctid')
                if not cols or any(c.endswith('_embedding') for c in cols):
                    continue
                key=f'{query.db_id}.{ref.name}.__scalar__'
                item=result.setdefault(key,dict(db_id=query.db_id,logical_table=ref.name,vector_column=None,
                    columns=set(),estimated_points=count,index_columns=set()))
                item['columns'].update(cols)
                item['index_columns'].update(index_columns.get((query.db_id,ref.name),set())&cols)
    for item in result.values():
        item['columns']=sorted(item['columns']);item['index_columns']=sorted(item['index_columns'])
    return result


def build_covering(conn, spec, *, source_schema, target_schema, snapshot_id):
    """Add an isolated regular table. Never modify source tables or old assets.

    CTAS retains duplicates and NULL vectors. The stored original CTID is used
    for every output identity and tie-break; the new heap's CTID is never used.
    No distance values, query results, or vector indexes are materialized.
    """
    from psycopg import sql
    from vec2plan_bandit.execution.relation_lineage import relation_info
    try:
        relation_info(conn,source_schema,spec['logical_table'])
    except ValueError:
        relation_info(conn,'public',spec['logical_table'])
        source_schema='public'
    digest = hashlib.sha256(json.dumps(dict(spec=spec,snapshot_id=snapshot_id,
        source_schema=source_schema,version=VERSION),sort_keys=True).encode()).hexdigest()
    name = 'pre_' + digest[:20]
    target = sql.Identifier(target_schema, name)
    marker = 'vec2plan-pre:' + digest
    conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(target_schema)))
    existing = conn.execute('SELECT c.oid,obj_description(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace '
                            'WHERE n.nspname=%s AND c.relname=%s', (target_schema,name)).fetchone()
    if existing:
        receipt = json.loads(existing[1] or '{}')
        if receipt.get('marker') != marker:
            raise ValueError('Refuse unrelated PRE covering relation')
        if 'row_id' in spec['columns'] and receipt.get('clustered_by')!='row_id':
            cluster_covering(conn,receipt)
        return receipt
    conn.execute("SET LOCAL work_mem='64MB'")
    conn.execute('SET LOCAL jit=off')
    conn.execute('SET LOCAL max_parallel_workers_per_gather=3')
    conn.execute('SET LOCAL statement_timeout=0')
    columns = sql.SQL(',').join(sql.SQL('b.{}').format(sql.Identifier(c)) for c in spec['columns'])
    command = sql.SQL('CREATE TABLE {} AS SELECT b.ctid AS {},{} FROM {} b').format(
        target,sql.Identifier(SOURCE_CTID),columns,sql.Identifier(source_schema,spec['logical_table']))
    cursor = conn.execute(command)
    count = cursor.rowcount
    if count < 0:
        count = conn.execute(sql.SQL('SELECT count(*) FROM {}').format(target)).fetchone()[0]
    scalar_indexes=[]
    for column in sorted(set(spec.get('index_columns',[]))|({'row_id'} if 'row_id' in spec['columns'] else set())):
        index_name=name+'_'+hashlib.sha256(column.encode()).hexdigest()[:8] if column!='row_id' else name+'_row_id'
        try:
            with conn.transaction():
                conn.execute(sql.SQL('CREATE INDEX {} ON {} ({})').format(sql.Identifier(index_name),target,sql.Identifier(column)))
            scalar_indexes.append(dict(name=index_name,method='btree',column=column))
        except Exception as error:
            if getattr(error,'sqlstate',None)!='54000':raise
    conn.execute(sql.SQL('ANALYZE {}').format(target))
    size = conn.execute('SELECT pg_total_relation_size(%s)', (f'{target_schema}.{name}',)).fetchone()[0]
    receipt = dict(spec, version=VERSION, marker=marker, schema=target_schema, table=name,
        rows=count, bytes=size, source_schema=source_schema, original_ctid_column=SOURCE_CTID,
        snapshot_id=snapshot_id, vector_indexes_created=0, scalar_indexes=scalar_indexes,cached_answers=False,
        source_contract='immutable database import; rebuild after source/CTID changes')
    conn.execute(sql.SQL('COMMENT ON TABLE {} IS {}').format(target,sql.Literal(json.dumps(receipt,sort_keys=True))))
    if 'row_id' in spec['columns']:
        cluster_covering(conn,receipt)
    return receipt


def cluster_covering(conn,receipt):
    """Order only our added heaps by scalar row_id; original CTIDs stay stored."""
    from psycopg import sql
    if receipt.get('version')!=VERSION or not receipt.get('marker','').startswith('vec2plan-pre:'):
        raise ValueError('Refuse to cluster an unrelated relation')
    target=sql.Identifier(receipt['schema'],receipt['table'])
    index=next(i['name'] for i in receipt['scalar_indexes'] if i['column']=='row_id')
    conn.execute("SET LOCAL maintenance_work_mem='256MB'")
    conn.execute('SET LOCAL statement_timeout=0')
    conn.execute(sql.SQL('CLUSTER {} USING {}').format(target,sql.Identifier(index)))
    conn.execute(sql.SQL('ANALYZE {}').format(target))
    receipt['clustered_by']='row_id'
    receipt['bytes']=conn.execute('SELECT pg_total_relation_size(%s)',(receipt['schema']+'.'+receipt['table'],)).fetchone()[0]
    conn.execute(sql.SQL('COMMENT ON TABLE {} IS {}').format(target,sql.Literal(json.dumps(receipt,sort_keys=True))))


def rewrite_covering(tree, query, receipts):
    from vec2plan_bandit.execution.relational_pre import prune_identity_projection
    prune_identity_projection(tree)
    expand_identity_vars(tree)
    ctes = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    changes = []
    for select in tree.find_all(exp.Select):
        source = select.args.get('from_')
        refs = ([source.this] if source else []) + [j.this for j in select.args.get('joins', [])]
        for ref in refs:
            if not isinstance(ref, exp.Table) or ref.name in ctes:
                continue
            alias, name = ref.alias_or_name, ref.name
            blocks = [b for b in query.blocks if b.alias == alias and b.table == name]
            if blocks and len({b.column for b in blocks}) != 1:
                continue
            key = f'{query.db_id}.{name}.{blocks[0].column if blocks else "__scalar__"}'
            receipt = receipts.get(key)
            if not receipt or receipt.get('version') != VERSION or receipt.get('cached_answers') is not False:
                continue
            # Resolve qualified correlated references without capturing an
            # inner relation that shadows the same alias.
            if select.find(exp.Star):
                continue
            columns=[]
            for column in select.find_all(exp.Column):
                scope=column.find_ancestor(exp.Select)
                if not column.table:
                    if scope is select and len(refs)==1 and column.name in query.columns[name]:columns.append(column)
                    continue
                if column.table!=alias:continue
                shadow=False
                while scope is not None and scope is not select:
                    child_from=scope.args.get('from_')
                    child_refs=([child_from.this] if child_from else [])+[j.this for j in scope.args.get('joins',[])]
                    if any(r.alias_or_name==alias for r in child_refs):shadow=True;break
                    scope=scope.find_ancestor(exp.Select)
                if scope is select and not shadow:columns.append(column)
            if any(c.name not in {*receipt['columns'],'ctid'} for c in columns):
                continue
            if any(not c.table and c.name == 'ctid' for c in select.find_all(exp.Column)):
                continue
            for column in columns:
                if column.name == 'ctid':
                    column.set('this',exp.to_identifier(SOURCE_CTID,quoted=True))
            ref.set('this',exp.to_identifier(receipt['table'],quoted=True))
            ref.set('db',exp.to_identifier(receipt['schema'],quoted=True))
            if not ref.args.get('alias'):
                ref.set('alias',exp.TableAlias(this=exp.to_identifier(alias,quoted=True)))
            changes.append(dict(logical_table=name,alias=alias,relation=receipt['schema']+'.'+receipt['table']))
    return changes
