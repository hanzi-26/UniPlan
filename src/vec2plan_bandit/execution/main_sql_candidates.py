"""Opt-in SQL candidates; original executors and baseline registry stay intact."""
from dataclasses import dataclass
import hashlib
import json
import math
import time

import numpy as np

from vec2plan_bandit.paths.registry import REGISTRY
from vec2plan_bandit.execution.relational_pg import configured_post_settings
from vec2plan_bandit.execution.timing import evaluation_deadline_ns, EvaluationDeadlineExceeded
from vec2plan_bandit.experiments.collector import _psycopg_sql

SQL_PATHS = ('FAISS_SQL', 'POST_SQL', 'PG_SQL', 'EXACT_SQL')


@dataclass(frozen=True)
class MainSQLConfig:
    path_id: str
    _params: tuple
    action_id: None = None

    @property
    def params(self):
        return dict(self._params)

    @property
    def config_id(self):
        payload = json.dumps([self.path_id, self.params], sort_keys=True)
        return 'main-sql-' + hashlib.sha256(payload.encode()).hexdigest()[:16]


def configuration(path, params):
    if path not in SQL_PATHS:
        raise ValueError('Unknown main SQL candidate')
    values = dict(params)
    if path == 'FAISS_SQL':
        mode = values.pop('search_mode', 'hnsw')
        if mode not in ('hnsw', 'flat'):
            raise ValueError('Unknown FAISS search mode')
        multiplier = values.get('candidate_multiplier', 4.)
        if (isinstance(multiplier, bool) or not isinstance(multiplier, (float, int))
                or not math.isfinite(multiplier) or not 1 <= multiplier <= 256):
            raise ValueError('Main FAISS candidate multiplier must be in [1, 256]')
        values['candidate_multiplier'] = min(multiplier, 50.)
        values = {**REGISTRY.normalize('FAISS', values), 'candidate_multiplier': float(multiplier), 'search_mode': mode}
        if mode == 'flat':
            values.pop('ef_search')  # Flat search has no graph traversal parameter.
    elif path == 'POST_SQL':
        values = REGISTRY.normalize('POST', values)
    elif values:
        raise ValueError('PG candidates have no path parameters')
    return MainSQLConfig(path, tuple(sorted(values.items())))


def sql_main(worker, statement, parameters, settings=()):
    """Batch only trusted SET LOCAL statements, preserving rollback/deadlines."""
    conn = worker.connection
    try:
        with conn.cursor() as cursor:
            commands = []
            if worker.config.get('runtime', {}).get('unified_resources', False):
                commands.extend(('SET LOCAL max_parallel_workers_per_gather=0', 'SET LOCAL jit=off'))
            commands.extend(settings)
            if any(not s.startswith('SET LOCAL ') or ';' in s for s in commands):
                raise ValueError('Only individual trusted SET LOCAL settings can be batched')
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
        if not conn.closed:
            conn.rollback()


def flat_index(worker, asset_key):
    """Build a small exact FAISS view from the frozen index, never from queries."""
    cache = worker.__dict__.setdefault('_main_sql_flat_indexes', {})
    if asset_key not in cache:
        import faiss
        original, ids = worker.indexes[asset_key]
        vectors = original.reconstruct_n(0, original.ntotal)
        flat = faiss.IndexFlatIP(original.d)
        flat.add(np.ascontiguousarray(vectors, dtype=np.float32))
        cache[asset_key] = flat
    return cache[asset_key]


def execute_sql_candidate(worker, query, vectors, config):
    params, path = config.params, config.path_id
    audit = dict(path_id=path, requested_params=params, config_id=config.config_id,
        fallback_used=False, distance_metric='cosine', result_identity='joined_row_primary_keys',
        execution_backend='postgresql', execution_contract='main-sql-candidates-v1')
    if path != 'FAISS_SQL':
        statement, binds = query.render(vectors, exact=path == 'EXACT_SQL')
        settings = configured_post_settings(worker, params) if path == 'POST_SQL' else ()
        return dict(audit, row_ids=[list(r[0]) for r in sql_main(worker, statement, binds, settings)],
                    postgres_selects=1, candidate_id_roundtrip=False)
    from vec2plan_bandit.queries.lightweight import simple_retrieval
    from vec2plan_bandit.execution.simple_ann import assemble_index_rows
    candidates, positions_by_block, indexes = [], [], []
    for block, vector in zip(query.blocks, vectors):
        index, ids = worker.indexes[block.asset_key]
        if params['search_mode'] == 'flat':
            index = flat_index(worker, block.asset_key)
        else:
            index.hnsw.efSearch = params['ef_search']
        count = min(index.ntotal, max(block.k, math.ceil(block.k * params['candidate_multiplier'])))
        _, positions = index.search(np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1), count)
        valid = [int(p) for p in positions[0] if p >= 0]
        candidates.append([int(ids[p]) for p in valid])
        positions_by_block.append(valid); indexes.append(index)
    semantics = simple_retrieval(query)
    if len(query.blocks) == 1 and semantics.direct_unfiltered:
        # Exact same tie/identity assembly contract as the existing simple path.
        positions = positions_by_block[0]
        ids = candidates[0]
        assembly = {}
        def reference_order():
            statement, binds = query.render(vectors, candidates=[ids], exact=True)
            return [list(r[0]) for r in sql_main(worker, statement, binds)]
        rows = assemble_index_rows(indexes[0], vectors[0], positions, [[i] for i in ids], k=query.k,
            reference_order=reference_order if semantics.order_required else None, audit=assembly)
        return dict(audit, row_ids=rows, candidate_counts=[len(c) for c in candidates],
            postgres_selects=int(assembly.get('reference_order_used', False)), **assembly)
    # Keep ALL retrieved candidates. Original SQL applies filters, JOINs and
    # each block's LIMIT; do not trim to k before relational eligibility.
    statement, binds = query.render(vectors, candidates=candidates, exact=True)
    rows = sql_main(worker, statement, binds)
    return dict(audit, row_ids=[list(r[0]) for r in rows],
        candidate_counts=[len(c) for c in candidates], postgres_selects=1,
        candidate_id_roundtrip=False, relational_eligibility='original_sql')
