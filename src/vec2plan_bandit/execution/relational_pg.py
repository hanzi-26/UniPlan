"""Database-resident PRE/POST execution for relational SQL adapters.

The original SQL owns filtering, joins, vector-block LIMITs and row identities.
Only the final result leaves PostgreSQL; no candidate/eligibility ID roundtrip.
"""
from __future__ import annotations

from vec2plan_bandit.execution.detailed_timing import stage
from vec2plan_bandit.paths.registry import POST, PRE, REGISTRY

VERSION = 'relational-pg-pre-post-v1'


def post_settings(params, *, planner_mode='legacy'):
    values = REGISTRY.normalize(POST, params)
    if planner_mode not in ('legacy','optimizer'):
        raise ValueError('Unknown relational POST planner mode')
    return [f"SET LOCAL hnsw.ef_search={values['ef_search']}",
            f"SET LOCAL hnsw.max_scan_tuples={values['max_scan_tuples']}",
            "SET LOCAL hnsw.iterative_scan='strict_order'",
            'SET LOCAL hnsw.scan_mem_multiplier=2',
            'SET LOCAL enable_seqscan='+('on' if planner_mode=='optimizer' else 'off')]


def configured_post_settings(worker, params):
    return post_settings(params,planner_mode=worker.config.get('relational_pg',{}).get('post_planner_mode','legacy'))


def execute_pg_sql(worker, query, vectors, config):
    """One SELECT, using the existing rollback/deadline-aware PG transport.

    PRE controls ordinary access and joins while suppressing vector ANN in the
    rendered SQL. POST retains the native vector operator and its HNSW controls.
    Planner settings request a strategy; they are not evidence of a physical
    plan. Offline EXPLAIN remains the way to audit realized access paths.
    """
    path = config.path_id
    if path not in (PRE, POST):
        raise ValueError('Database-resident execution requires PRE or POST')
    values = REGISTRY.normalize(path, config.params)
    settings = worker.pre_settings(values) if path == PRE else configured_post_settings(worker, values)
    with stage(worker, 'sql_compile_ms'):
        if path == PRE and worker.config.get('relational_pre', {}).get('enabled', False):
            from vec2plan_bandit.execution.relational_pre import render_pre_sql
            statement, binds, pre_audit = render_pre_sql(worker, query, vectors)
        else:
            statement, binds = query.render(vectors, exact=path == PRE)
            pre_audit = {}
    rows = worker.sql(statement, binds, settings)
    with stage(worker, 'result_assembly_ms'):
        identities = [list(row[0]) for row in rows]
    return dict(row_ids=identities, path_id=path, requested_params=values,
                config_id=config.config_id, fallback_used=False, **pre_audit,
                distance_metric='cosine', result_identity='joined_row_primary_keys',
                execution_contract=VERSION, execution_backend='postgresql',
                postgres_selects=1, candidate_id_roundtrip=False,
                pg_planner_mode=worker.config.get('relational_pg',{}).get('post_planner_mode','legacy') if path==POST else 'verified_pre')


def explain_post_sql(worker, query, vectors, params=None):
    """Audit the same whole-query SQL and settings used by deployed POST."""
    statement, binds = query.render(vectors, exact=False)
    return worker.sql('EXPLAIN (FORMAT JSON) ' + statement, binds,
                      configured_post_settings(worker,params or {}))[0][0]


def post_plan_evidence(worker, plans):
    """Record realized index methods offline; planner flags alone prove nothing.

    Empty/point-lookup/small-relation plans may legitimately avoid ANN. Preserve
    that fact instead of labeling every POST request as an observed HNSW scan.
    """
    from vec2plan_bandit.execution.plan_semantics import plan_nodes
    nodes = [node for plan in plans for node in plan_nodes(plan)]
    names = sorted({node['Index Name'] for node in nodes if node.get('Index Name')})
    methods = dict(worker.sql(
        'SELECT c.relname,am.amname FROM pg_class c '
        'JOIN pg_am am ON am.oid=c.relam WHERE c.relname=ANY(%(index_names)s)',
        {'index_names': names})) if names else {}
    hnsw = sorted(name for name, method in methods.items() if method == 'hnsw')
    return dict(scope='offline_whole_query_explain', hnsw_used=bool(hnsw),
                hnsw_indexes=hnsw, index_methods=methods,
                node_types=[node['Node Type'] for node in nodes])
