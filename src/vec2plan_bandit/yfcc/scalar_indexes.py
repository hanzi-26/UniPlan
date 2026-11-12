"""Shared offline scalar assets for the existing exact PostgreSQL path.

No vector index, query answer cache, SQL rewrite, or selectable search arm.
"""
from psycopg import sql

INDEXES = (('equal', 'btree'), ('range', 'btree'), ('tags', 'gin'))


def index_name(column, method):
    if (column, method) not in INDEXES:
        raise ValueError('Only the three declared YFCC scalar indexes are allowed')
    return f'v2p_yfcc_{column}_{method}_v1'


def receipt(conn, name):
    row = conn.execute(
        "SELECT n.nspname,t.relname,am.amname,i.indisvalid,i.indisready,"
        "i.indnkeyatts,i.indnatts,i.indpred IS NULL,i.indexprs IS NULL,"
        "ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY k(num,ord) "
        "JOIN pg_attribute a ON a.attrelid=t.oid AND a.attnum=k.num ORDER BY k.ord),"
        "pg_get_indexdef(x.oid),pg_relation_size(x.oid) "
        "FROM pg_class x JOIN pg_namespace xn ON xn.oid=x.relnamespace "
        "JOIN pg_index i ON i.indexrelid=x.oid JOIN pg_class t ON t.oid=i.indrelid "
        "JOIN pg_namespace n ON n.oid=t.relnamespace JOIN pg_am am ON am.oid=x.relam "
        "WHERE xn.nspname='public' AND x.relname=%s", (name,)).fetchone()
    if row is None:
        return None
    return dict(zip(('schema','table','method','valid','ready','key_columns','total_columns',
                     'unconditional','plain_columns','columns','definition','bytes'), row))


def validate_receipt(value, column, method):
    expected = dict(schema='public', table='YFCCImages', method=method, valid=True,
                    ready=True, key_columns=1, total_columns=1, unconditional=True,
                    plain_columns=True, columns=[column])
    if value is None or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Scalar index absent, invalid or incompatible: '+index_name(column, method))


def install(conn, *, progress=None):
    """Call only offline, under the common experiment lock, in autocommit.

    Preserve all existing assets. A conflicting or interrupted invalid index
    causes an explicit failure instead of silently claiming it is installed.
    """
    if not conn.autocommit:
        raise ValueError('Concurrent scalar index installation requires autocommit')
    conn.execute('SET max_parallel_maintenance_workers=0')
    conn.execute("SET maintenance_work_mem='256MB'")
    conn.execute("SET lock_timeout='5s'")
    conn.execute('SET statement_timeout=0')
    result = []
    for column, method in INDEXES:
        name = index_name(column, method)
        if progress:
            progress(column=column, method=method, index=name)
        before = receipt(conn, name)
        if before is None:
            conn.execute(sql.SQL('CREATE INDEX CONCURRENTLY {} ON public.{} USING {} ({})').format(
                sql.Identifier(name), sql.Identifier('YFCCImages'), sql.SQL(method), sql.Identifier(column)))
        value = receipt(conn, name)
        validate_receipt(value, column, method)
        result.append(dict(name=name, reused_existing=before is not None, **value))
    conn.execute('ANALYZE public."YFCCImages" ("equal","range","tags")')
    return dict(version='yfcc-shared-scalar-indexes-v1', indexes=result,
                shared_by_all_methods=True, vector_indexes_created=False,
                query_answers_cached=False, table_rows_changed=False)


def optimizer_plan(worker, query, vectors):
    """Verify the existing optimizer PRE parameter against the actual SQL."""
    from vec2plan_bandit.execution.pre_access import validate_plan
    from vec2plan_bandit.execution.plan_semantics import plan_nodes
    from vec2plan_bandit.paths.registry import make_execution_config
    action = make_execution_config('PRE', dict(index_mode='index', index_strategy='optimizer', join_strategy='all'))
    statement, binds = query.render(vectors, exact=True)
    plan = worker.sql('EXPLAIN (FORMAT JSON) '+statement, binds, worker.pre_settings(action.params))[0][0]
    names = sorted({n['Index Name'] for n in plan_nodes(plan) if n.get('Index Name')})
    methods = dict(worker.sql('SELECT c.relname,am.amname FROM pg_class c JOIN pg_am am ON am.oid=c.relam '
                             'WHERE c.relname=ANY(%(index_names)s)', dict(index_names=names))) if names else {}
    validate_plan(plan, action.params, methods, {b.table for b in query.blocks})
    return action, plan, methods
