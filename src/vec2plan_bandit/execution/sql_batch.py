"""Shared opt-in transport optimization; unchanged SELECT, settings and rollback."""
import math
import time


def execute_batched_local(connection, statement, parameters=None, settings=(), *, unified_resources=False):
    from vec2plan_bandit.execution.timing import evaluation_deadline_ns, EvaluationDeadlineExceeded
    from vec2plan_bandit.experiments.collector import _psycopg_sql

    commands = ['SET LOCAL max_parallel_workers_per_gather=0', 'SET LOCAL jit=off'] if unified_resources else []
    commands.extend(settings)
    if any(not isinstance(v, str) or not v.startswith('SET LOCAL ') or ';' in v for v in commands):
        raise ValueError('Only individual trusted SET LOCAL commands may be batched')
    try:
        with connection.cursor() as cursor:
            deadline = evaluation_deadline_ns()
            if deadline is not None:
                remaining = math.ceil((deadline - time.perf_counter_ns()) / 1e6)
                if remaining <= 0:
                    raise EvaluationDeadlineExceeded()
                commands.append(f'SET LOCAL statement_timeout={remaining}')
            if commands:
                cursor.execute('; '.join(commands))
            cursor.execute(_psycopg_sql(statement) if parameters else statement, parameters or None)
            return cursor.fetchall()
    finally:
        if not connection.closed:
            connection.rollback()
