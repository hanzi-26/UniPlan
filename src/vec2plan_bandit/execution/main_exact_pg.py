"""Opt-in main action: exact distance ordering with PostgreSQL's normal planner.

This action neither changes the PRE settings nor extends the baseline registry.
EXPLAIN verification is offline; every measured request renders fresh vectors.
"""
from dataclasses import dataclass

PG_EXACT = 'PG_EXACT_SQL'


@dataclass(frozen=True)
class MainExactPGConfig:
    path_id: str = PG_EXACT
    config_id: str = 'native-pg-exact'
    action_id: None = None

    @property
    def params(self):
        return {}


def execute_exact_pg(worker, query, vectors):
    from vec2plan_bandit.execution.detailed_timing import stage
    with stage(worker, 'sql_compile_ms'):
        statement, binds = query.render(vectors, exact=True)
    return dict(row_ids=[list(r[0]) for r in worker.sql(statement, binds)],
                path_id=PG_EXACT, requested_params={}, fallback_used=False)


def verify_exact_plan(worker, query, vectors):
    """Reject ANN or unresolved index nodes without executing the query plan."""
    statement, binds = query.render(vectors, exact=True)
    plan = worker.sql('EXPLAIN (FORMAT JSON) ' + statement, binds)[0][0]
    indexes = {}
    for name, method in worker.sql(
            'SELECT c.relname, a.amname FROM pg_catalog.pg_class c '
            'JOIN pg_catalog.pg_am a ON a.oid = c.relam WHERE c.relkind IN (\'i\', \'I\')'):
        indexes.setdefault(name, set()).add(method)
    used = {}
    def visit(value):
        if isinstance(value, dict):
            if 'Index Name' in value:
                name = value['Index Name']
                methods = indexes.get(name, set())
                if len(methods) != 1 or methods & {'hnsw', 'ivfflat'}:
                    raise ValueError(f'Exact candidate has ANN or unresolved index: {name}')
                used[name] = next(iter(methods))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(plan)
    return dict(plan=plan, indexes=used, no_ann=True, exact_renderer=True)
