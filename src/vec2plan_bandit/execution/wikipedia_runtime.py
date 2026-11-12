"""Cosine Wiki execution adapters; the shared CASH/MaxUCB policy remains unchanged."""
from __future__ import annotations
from vec2plan_bandit.execution.detailed_timing import stage
from pathlib import Path
import json
import math
import time
import numpy as np
import yaml

from vec2plan_bandit.services.database_layout import connection_parameters
from vec2plan_bandit.services.embedding import EmbeddingServiceClient
from vec2plan_bandit.services.vector_contract import validate_vector_contract
from vec2plan_bandit.experiments.collector import _psycopg_sql, _plan_nodes
from vec2plan_bandit.execution.timing import evaluation_deadline_ns
from vec2plan_bandit.paths.registry import pre_configuration_key
from vec2plan_bandit.policies.max_ucb import sample_path_configurations
from vec2plan_bandit.queries.wikipedia_sql_adapter import KEYS, quote
from vec2plan_bandit.execution.relational_pg import execute_pg_sql, explain_post_sql, post_settings

ROOT = Path(__file__).resolve().parents[3]


def elapsed(start):
    return (time.perf_counter_ns() - start) / 1e6


class WikiRuntime:
    def __init__(self, config):
        import faiss
        from qdrant_client import QdrantClient
        validate_vector_contract(config)
        self.config = config
        self.conn = None
        faiss.omp_set_num_threads(config.get('runtime', {}).get('cpu_threads', config.get('runtime', {}).get('recommended_m3_pro_threads', 6)))
        embed = config['embedding']
        import requests
        session = requests.Session()
        session.trust_env = embed.get('trust_env', True)
        self.embedding_session = session
        self.embedding = EmbeddingServiceClient(url=embed['url'], model=embed['model'],
             dimension=embed['dimension'], normalize=embed['normalized'], timeout_seconds=embed['timeout_seconds'], session=session)
        self.qdrant = QdrantClient(url=config['acorn']['url'], timeout=10,
            prefer_grpc=config['acorn'].get('prefer_grpc', True),
            grpc_port=config['acorn'].get('grpc_port', 6334), trust_env=config['acorn'].get('trust_env', True))
        self.assets = {}
        self.indexes = {}
        receipt = json.loads((ROOT / config['assets']['wiki_receipt']).read_text())
        if receipt['status'] != 'complete':
            raise ValueError('Wiki indexes are incomplete or retired')
        for asset in receipt['assets']:
            if asset.get('distance') != 'cosine':
                raise ValueError('Wiki indexes must use cosine distance')
            if asset['dimension'] != embed['dimension'] or asset['normalized'] != embed['normalized']:
                raise ValueError('Wiki asset encoder settings mismatch')
            if asset.get('embedding_model') != embed['model']:
                raise ValueError('Wiki index embedding model mismatch')
            key = f"{asset['table']}.{asset['column']}"
            index = faiss.read_index(asset['faiss_index'])
            ids = np.load(asset['row_ids'], allow_pickle=False)
            if index.d != embed['dimension'] or index.metric_type != faiss.METRIC_INNER_PRODUCT or index.ntotal != len(ids):
                raise ValueError('Wiki FAISS dimension/metric/ID mismatch')
            self.assets[key] = asset
            self.indexes[key] = (index, ids)

    @property
    def connection(self):
        if self.conn is None or self.conn.closed:
            import psycopg
            settings = yaml.safe_load((ROOT / self.config['database']['config_path']).read_text())
            values = connection_parameters(settings['database_connections']['postgresql'])
            values.pop('database', None)
            values['dbname'] = self.config['database']['name']
            self.conn = psycopg.connect(**values, connect_timeout=5,
                options='-c default_transaction_read_only=on -c search_path=public -c statement_timeout=10000')
        return self.conn

    def cancel_pending(self):
        if self.conn is not None and not self.conn.closed:
            try:
                self.conn.cancel_safe(timeout=1)
            finally:
                self.conn.close()
                self.conn = None

    def close(self):
        if self.conn is not None:
            self.conn.close()
        self.qdrant.close()

    def sql(self, statement, parameters=None, settings=()):
        conn = self.connection
        try:
            with conn.cursor() as cursor:
                if self.config.get('runtime', {}).get('unified_resources', False):
                    cursor.execute('SET LOCAL max_parallel_workers_per_gather=0')
                    cursor.execute('SET LOCAL jit=off')
                for setting in settings:
                    cursor.execute(setting)
                deadline = evaluation_deadline_ns()
                if deadline is not None:
                    remaining = max(1, math.ceil((deadline - time.perf_counter_ns()) / 1e6))
                    cursor.execute(f'SET LOCAL statement_timeout={remaining}')
                cursor.execute(_psycopg_sql(statement) if parameters else statement, parameters or None)
                return cursor.fetchall()
        finally:
            if not conn.closed:
                conn.rollback()

    def vectors(self, query):
        vectors = []
        for block in query.blocks:
            vector, _ = self.embedding.embed(block.text)
            vectors.append(vector)
        return vectors

    def eligible(self, query, vectors):
        return [[int(r[0]) for r in self.sql(*query.eligible_sql(b.number, vectors))] for b in query.blocks]

    def statistics(self, query, vectors):
        ids = self.eligible(query, vectors)
        counts = [len(i) for i in ids]
        total = sum(self.assets[b.asset_key]['points'] for b in query.blocks)
        return {'candidate_count': sum(counts), 'estimated_selectivity': sum(counts) / total if total else 0,
                'block_candidate_counts': counts}

    def pre_settings(self, params):
        if getattr(self, 'config', {}).get('runtime', {}).get('pre_access_enabled', False):
            from vec2plan_bandit.execution.pre_access import pre_settings
            return pre_settings(params)
        if params['index_mode'] != 'no_index':
            raise ValueError('Wiki scalar index strategies require an installed and verified scalar index')
        settings = ['SET LOCAL enable_indexscan=off', 'SET LOCAL enable_indexonlyscan=off',
                    'SET LOCAL enable_bitmapscan=off', 'SET LOCAL enable_seqscan=on']
        joins = {'hash': 'enable_hashjoin', 'merge': 'enable_mergejoin', 'nested_loop': 'enable_nestloop'}
        for kind, option in joins.items():
            settings.append(f"SET LOCAL {option}={'on' if params['join_strategy'] in ('all', kind) else 'off'}")
        return settings

    def pre_availability(self, query, vectors):
        sql, params = query.render(vectors, exact=True)
        result = {}
        for config in sample_path_configurations('PRE', 1, seed=42):
            if config.params['index_mode'] != 'no_index' and not self.config.get('runtime', {}).get('pre_access_enabled', False):
                continue
            payload = self.sql('EXPLAIN (FORMAT JSON) ' + sql, params, self.pre_settings(config.params))[0][0]
            if self.config.get('runtime', {}).get('pre_access_enabled', False):
                from vec2plan_bandit.execution.pre_access import validate_plan
                names=[n.get('Index Name') for n in _plan_nodes(payload) if n.get('Index Name')]
                methods=dict(self.sql('SELECT c.relname,am.amname FROM pg_class c JOIN pg_am am ON am.oid=c.relam WHERE c.relname=ANY(%(index_names)s)',{'index_names':names})) if names else {}
                try:
                    validate_plan(payload,config.params,methods,{b.table for b in query.blocks})
                except ValueError as error:
                    self.__dict__.setdefault('pre_rejections', {})[(query.query_id,pre_configuration_key(config.params))]=str(error)
                    continue
                result[pre_configuration_key(config.params)]=payload
                continue
            nodes = _plan_nodes(payload)
            joins = [n['Node Type'] for n in nodes if n['Node Type'] in ('Hash Join', 'Merge Join', 'Nested Loop')]
            requested = config.params['join_strategy']
            expected = {'hash': 'Hash Join', 'merge': 'Merge Join', 'nested_loop': 'Nested Loop'}.get(requested)
            if any('hnsw' in n.get('Index Name', '').lower() for n in nodes):
                raise ValueError('PRE unexpectedly uses HNSW')
            passed = requested == 'all' or bool(joins) and all(n == expected for n in joins)
            if passed:
                result[pre_configuration_key(config.params)] = payload
        if not result:
            raise ValueError('no valid exact PRE plan')
        return result

    def post_candidates(self, block, vector, eligible, params, *, explain=False):
        # Legacy block-level preparation probe, not the deployed POST executor.
        statement = (f'SELECT {quote(KEYS[block.table][0])} FROM {quote(block.table)} '
            f'WHERE {quote(KEYS[block.table][0])}=ANY(%(allowed)s::bigint[]) '
            f'ORDER BY {quote(block.column)} <=> %(vector)s::vector LIMIT %(k)s')
        values = {'allowed': eligible, 'vector': '[' + ','.join(str(float(v)) for v in vector) + ']', 'k': block.k}
        settings = post_settings(params)
        rows = self.sql(('EXPLAIN (FORMAT JSON) ' if explain else '') + statement, values, settings)
        return rows[0][0] if explain else [int(r[0]) for r in rows]

    def explain_post(self, query, vectors, params=None):
        return explain_post_sql(self, query, vectors, params)

    def execute(self, query, vectors, config):
        path, params = config.path_id, config.params
        if path in {'PRE', 'POST'}:
            return execute_pg_sql(self, query, vectors, config)
        audit = {'path_id': path, 'requested_params': dict(params), 'config_id': config.config_id,
                 'fallback_used': False, 'distance_metric': 'cosine', 'result_identity': 'joined_row_primary_keys'}
        if path in {'FAISS', 'ACORN'} and self.config.get('runtime', {}).get('lightweight_execution', True):
            from vec2plan_bandit.queries.lightweight import simple_retrieval
            with stage(self, 'semantic_check_ms'):
                semantics = simple_retrieval(query)
            audit['simplification_reason'] = semantics.reason
            if semantics.direct_unfiltered and len(KEYS.get(semantics.table, ())) == 1:
                return {**self.execute_simple_ann(query, vectors, config), **audit,
                        'execution_simplified': True}
        if path in {'FAISS', 'ACORN'}:
            # Exact relational eligibility is computed online, inside E2E. This
            # supports LIKE, ranges and cross-table joins without approximate
            # string matching or a different materialized dataset.
            eligible = self.eligible(query, vectors)
            candidates = []
            for block, vector, allowed in zip(query.blocks, vectors, eligible):
                if not allowed:
                    candidates.append([])
                    continue
                if path == 'FAISS':
                    index, row_ids = self.indexes[block.asset_key]
                    index.hnsw.efSearch = params['ef_search']
                    candidate_k = min(index.ntotal, max(block.k, math.ceil(block.k * params['candidate_multiplier'])))
                    with stage(self,'ann_search_ms'):
                        _, positions = index.search(vector.reshape(1, -1), candidate_k)
                    with stage(self,'id_mapping_ms'):
                        mapped=[int(row_ids[p]) for p in positions[0] if p >= 0]
                    with stage(self,'candidate_filter_ms'):
                        valid = set(allowed)
                        ids=[p for p in mapped if p in valid][:block.k]
                elif path == 'ACORN':
                    from qdrant_client import models
                    asset = self.assets[block.asset_key]
                    query_filter = None
                    if len(allowed) < asset['points']:
                        query_filter = models.Filter(must=[models.HasIdCondition(has_id=allowed)])
                    with stage(self,'ann_search_ms'):
                        points = self.qdrant.query_points(asset['collection'], query=vector.tolist(), query_filter=query_filter,
                            limit=block.k, with_payload=False, with_vectors=False,
                            search_params=models.SearchParams(hnsw_ef=params['hnsw_ef'], exact=False,
                                acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0))).points
                    ids = [int(p.id) for p in points]
                else:
                    raise ValueError('unknown Wiki execution path')
                candidates.append(ids)
            audit['candidate_counts'] = [len(i) for i in candidates]
            with stage(self,'sql_compile_ms'):
                statement, binds = query.render(vectors, candidates=candidates, exact=True)
            with stage(self,'result_assembly_ms'):
                rows = self.sql(statement, binds)
        else:
            raise ValueError('unknown Wiki execution path')
        # The first projection is the composite primary-key tuple. Keep row
        # multiplicity; collapsing to article_id loses JOIN semantics.
        with stage(self,'result_assembly_ms'):
            audit['row_ids'] = [list(r[0]) for r in rows]
        return audit

    def execute_simple_ann(self, query, vectors, config):
        from vec2plan_bandit.execution.simple_ann import search_simple_ann, assemble_index_rows
        from vec2plan_bandit.queries.lightweight import simple_retrieval
        block, vector = query.blocks[0], vectors[0]
        index, row_ids = self.indexes[block.asset_key]
        def remote():
            from qdrant_client import models
            result = self.qdrant.query_points(self.assets[block.asset_key]['collection'],
                query=vector.tolist(), query_filter=None, limit=block.k,
                with_payload=False, with_vectors=False,
                search_params=models.SearchParams(hnsw_ef=config.params['hnsw_ef'], exact=False,
                    acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0)))
            with stage(self, 'id_mapping_ms'):
                reverse = self.__dict__.setdefault('_simple_id_maps', {})
                key = (block.asset_key, id(row_ids))
                if key not in reverse:
                    reverse[key] = {int(value): i for i, value in enumerate(row_ids)}
                return [reverse[key][int(p.id)] for p in result.points]
        positions = search_simple_ann(self, index, vector, block.k, config, remote)
        with stage(self, 'id_mapping_ms'):
            identities = [[int(row_ids[i])] for i in positions]
        def reference_order():
            with stage(self, 'sql_compile_ms'):
                statement, binds = query.render(vectors, candidates=[[v[0] for v in identities]], exact=True)
            return [list(r[0]) for r in self.sql(statement, binds)]
        audit = {}
        with stage(self, 'result_assembly_ms'):
            rows = assemble_index_rows(index, vector, positions, identities, k=query.k,
                reference_order=reference_order if simple_retrieval(query).order_required else None,
                audit=audit)
        reference = audit.get('reference_order_used', False)
        return {'row_ids': rows, 'candidate_counts': [len(positions)],
                'return_to_pg': reference, 'postgres_queries': int(reference), **audit}

    def exact(self, query, vectors):
        statement, binds = query.render(vectors, exact=True)
        return [list(row[0]) for row in self.sql(statement, binds, self.pre_settings({'index_mode': 'no_index', 'join_strategy': 'all'}))]

    def native(self, query, vectors):
        with stage(self,'sql_compile_ms'):
            statement, binds = query.render(vectors)
        return {'row_ids': [list(r[0]) for r in self.sql(statement, binds)], 'path_id': 'PG_NATIVE_SQL',
                'requested_params': {}, 'fallback_used': False}
