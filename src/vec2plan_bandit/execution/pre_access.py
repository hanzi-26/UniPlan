"""Shared exact PRE access control and measured physical-plan verification."""
from __future__ import annotations
import hashlib
from vec2plan_bandit.paths.registry import pre_access_strategy
from vec2plan_bandit.execution.plan_semantics import plan_nodes


def pre_settings(params):
    access=pre_access_strategy(params)
    if access not in ('scan','btree','hash','bitmap','optimizer'):
        raise ValueError('Unsupported PRE access strategy')
    settings=[f'SET LOCAL enable_seqscan={"on" if access in ("scan","optimizer") else "off"}',
        f'SET LOCAL enable_indexscan={"on" if access in ("btree","hash","optimizer") else "off"}',
        f'SET LOCAL enable_indexonlyscan={"on" if access in ("btree","hash","optimizer") else "off"}',
        f'SET LOCAL enable_bitmapscan={"on" if access in ("bitmap","optimizer") else "off"}',
        'SET LOCAL max_parallel_workers_per_gather=0','SET LOCAL jit=off']
    for kind,option in {'hash':'enable_hashjoin','merge':'enable_mergejoin','nested_loop':'enable_nestloop'}.items():
        settings.append(f'SET LOCAL {option}={"on" if params["join_strategy"] in ("all",kind) else "off"}')
    return settings


def validate_plan(payload, params, index_methods, vector_tables, *, relation_lineage=None):
    """An installed index is insufficient: require actual vector-table access.

    Hash and B-tree may coexist. If PostgreSQL chooses the other access method,
    reject this action rather than silently relabeling it. Bitmap is an access
    plan, not a PostgreSQL index type. No forced join on a join-free query.
    """
    nodes=plan_nodes(payload)
    names=[n.get('Index Name') for n in nodes if n.get('Index Name')]
    if any(index_methods.get(n)=='hnsw' or 'hnsw' in n.lower() for n in names):
        raise ValueError('PRE_USES_VECTOR_ANN')
    access=pre_access_strategy(params)
    matched=[]
    for table in vector_tables:
        physical=(relation_lineage or {}).get(table,{}).get('base_table',table)
        scans=[n for n in nodes if n.get('Relation Name')==physical]
        if not scans:continue  # Unreferenced/eliminated CTEs have no physical access.
        indices=[n for n in scans if n.get('Node Type') in ('Index Scan','Index Only Scan')]
        if access=='optimizer':
            valid=True  # Mixed scalar/TID access; never claim a specific index.
        elif access=='scan':
            valid=bool(scans) and not indices and not any(n.get('Node Type')=='Bitmap Heap Scan' for n in scans)
        elif access=='bitmap':
            valid=any(n.get('Node Type')=='Bitmap Heap Scan' for n in scans)
        else:
            valid=any(index_methods.get(n.get('Index Name'))==access for n in indices)
        if not valid and access != 'scan' and relation_lineage and table in relation_lineage:
            valid = _mapping_access(nodes, relation_lineage[table], access, index_methods)
        matched.append(valid)
    if not matched and access in ('scan','optimizer') and params['join_strategy']=='all':
        # Constant-empty plans and unused vector CTEs perform no vector access.
        # Keep the exact default path; do not invent an indexed access or join.
        return
    if not matched or not all(matched):
        raise ValueError('PRE_ACCESS_NOT_REALIZED:'+access)
    join=params['join_strategy'];expected={'hash':'Hash Join','merge':'Merge Join','nested_loop':'Nested Loop'}
    actual=[n['Node Type'] for n in nodes if n.get('Node Type') in expected.values()]
    if join!='all' and (not actual or any(n!=expected[join] for n in actual)):
        raise ValueError('PRE_JOIN_NOT_REALIZED:'+join)


def _mapping_access(nodes, lineage, access, methods):
    import re
    if not lineage.get('mapping_unique') or lineage.get('join_key') != 'ctid=source_ctid':
        return False
    base, mapping = lineage['base_table'], lineage['mapping_table']
    for join in nodes:
        if join.get('Node Type') not in ('Hash Join','Merge Join','Nested Loop'):
            continue
        descendants = plan_nodes({'Plan': join})
        bases = [n for n in descendants if n.get('Relation Name') == base]
        maps = [n for n in descendants if n.get('Relation Name') == mapping]
        for mapped in maps:
            if not (mapped.get('Index Cond') or mapped.get('Filter') or mapped.get('Recheck Cond')):
                continue
            if access == 'bitmap':
                realized = mapped.get('Node Type') == 'Bitmap Heap Scan'
            else:
                realized = mapped.get('Node Type') in ('Index Scan','Index Only Scan') and methods.get(mapped.get('Index Name')) == access
            if not realized:
                continue
            alias = re.escape(mapped.get('Alias', mapping))
            for vector in bases:
                b = re.escape(vector.get('Alias', base))
                conditions = ' '.join(str(n.get(k,'')) for n in descendants
                    for k in ('Hash Cond','Merge Cond','Join Filter','TID Cond','Tid Cond','Index Cond')).replace('"','')
                lhs = rf'(?:{b}\.)?ctid'
                rhs = rf'{alias}\.source_ctid'
                if re.search(rf'\b(?:{lhs}\s*=\s*{rhs}|{rhs}\s*=\s*{lhs})\b', conditions):
                    return True
    return False


def install_scalar_indexes(conn, tables, *, schema='public', progress=None, resolve_views=False, column_filter=None):
    """Install scalar access assets offline; never create indexes in a request.

    Only built-in scalar types are considered. Oversized text cannot support a
    B-tree index and is recorded, while hash remains separately available.
    Existing ordinary and vector indexes are preserved.
    """
    from psycopg import sql
    records=[]
    if resolve_views:
        from vec2plan_bandit.execution.relation_lineage import index_targets
        targets=index_targets(conn,schema,tables)
    else:
        targets=[(schema,t,[t]) for t in sorted(set(tables))]
    for physical_schema,table,owners in targets:
        columns=conn.execute("SELECT a.attname,t.typname FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_type t ON t.oid=a.atttypid WHERE n.nspname=%s AND c.relname=%s AND a.attnum>0 AND NOT a.attisdropped",(physical_schema,table)).fetchall()
        for column,kind in columns:
            if column_filter is not None and not any(column in column_filter.get(owner,set()) for owner in owners):
                continue
            if kind not in ('int2','int4','int8','float4','float8','numeric','text','varchar','bpchar','bool','date','timestamp','timestamptz','uuid'):
                continue
            for method in ('btree','hash'):
                name='v2p_pre_'+method+'_'+hashlib.sha256(f'{physical_schema}.{table}.{column}'.encode()).hexdigest()[:20]
                if progress:progress(table=table,column=column,method=method)
                if resolve_views:
                    existing=conn.execute(
                        'SELECT ic.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indrelid '
                        'JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_class ic ON ic.oid=i.indexrelid '
                        'JOIN pg_am am ON am.oid=ic.relam JOIN pg_attribute a ON a.attrelid=c.oid '
                        'AND a.attnum=ANY(i.indkey) WHERE n.nspname=%s AND c.relname=%s '
                        'AND a.attname=%s AND am.amname=%s AND i.indisvalid AND i.indisready '
                        'AND i.indnkeyatts=1 AND i.indpred IS NULL AND i.indexprs IS NULL LIMIT 1',
                        (physical_schema,table,column,method)).fetchone()
                    if existing:
                        records.append({'schema':physical_schema,'table':table,'logical_tables':owners,
                            'column':column,'method':method,'index':existing[0],'status':'installed','reused_existing':True})
                        continue
                try:
                    with conn.transaction():
                        conn.execute(sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {}.{} USING {} ({})').format(sql.Identifier(name),sql.Identifier(physical_schema),sql.Identifier(table),sql.SQL(method),sql.Identifier(column)))
                    records.append({'schema':physical_schema,'table':table,'logical_tables':owners,'column':column,'method':method,'index':name,'status':'installed'})
                except Exception as error:
                    if resolve_views and getattr(error,'sqlstate',None) != '54000':
                        raise
                    records.append({'table':table,'column':column,'method':method,'status':'unsupported','error_type':type(error).__name__,'error':str(error)})
    return records
