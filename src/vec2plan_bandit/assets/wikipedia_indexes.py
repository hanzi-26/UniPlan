"""Build MiniLM Wiki cosine FAISS HNSW and Qdrant ACORN-1 assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import time
import hashlib

import faiss
import numpy as np
import psycopg
from psycopg import sql
from qdrant_client import QdrantClient, models
import yaml

from vec2plan_bandit.assets.wikipedia_pg import schema_from_queries
from vec2plan_bandit.services.database_layout import connection_parameters
from vec2plan_bandit.assets.build_resources import build_threads, configure_pg_build


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='wikipedia_minilm384')
    parser.add_argument('--folder', default='data/indexes/wiki-minilm-cosine')
    parser.add_argument('--dimension', type=int, default=384)
    parser.add_argument('--prefix', default='wiki_minilm_cosine')
    parser.add_argument('--schema', type=Path, default=root/'data/wiki/candidate_sql_cosine.json')
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:6333')
    parser.add_argument('--install-post-indexes', action='store_true')
    parser.add_argument('--threads', default='4', help='FAISS build threads: integer or auto (up to 32)')
    parser.add_argument('--pg-build-workers', help='PG parallel build workers: integer or auto; omit to retain server default')
    parser.add_argument('--maintenance-memory-mb', type=int, default=512)
    parser.add_argument('--qdrant-indexing-threads', type=int, default=2)
    parser.add_argument('--qdrant-optimization-threads', type=int,
                        help='Optimizer concurrency; omit to retain server default')
    parser.add_argument('--no-proxy', action='store_true')
    parser.add_argument('--normalized', action='store_true', default=True)
    args = parser.parse_args()
    threads = build_threads(args.threads)
    if args.qdrant_indexing_threads < 1 or args.maintenance_memory_mb < 1 or (
            args.qdrant_optimization_threads is not None and args.qdrant_optimization_threads < 1):
        parser.error('Build threads and maintenance memory must be positive')
    folder = root / args.folder
    folder.mkdir(parents=True, exist_ok=True)
    schema = schema_from_queries(args.schema)
    settings = yaml.safe_load((root / 'configs/database.yaml').read_text())
    params = connection_parameters(settings['database_connections']['postgresql'])
    params.pop('database', None)
    params['dbname'] = args.database
    client = QdrantClient(url=args.qdrant_url, timeout=120, trust_env=not args.no_proxy)
    if args.install_post_indexes:
        with psycopg.connect(**params) as conn:
            if args.pg_build_workers is None:
                conn.execute("SELECT set_config('maintenance_work_mem', %s, false)",
                             (f'{args.maintenance_memory_mb}MB',))
            else:
                settings = configure_pg_build(conn, workers=args.pg_build_workers,
                                               maintenance_memory_mb=args.maintenance_memory_mb)
                print(json.dumps({'stage': 'pg_build_settings', **settings}), flush=True)
            for table, columns in schema.items():
                for column, kind in columns:
                    if kind == 'vector':
                        name = f'idx_{table}_{column}_hnsw_cosine'
                        conn.execute(sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw ({} vector_cosine_ops)').format(
                            sql.Identifier(name), sql.Identifier(table), sql.Identifier(column)))
            conn.execute('ANALYZE')
    faiss.omp_set_num_threads(threads)
    assets = []
    with psycopg.connect(**params) as conn:
        conn.execute('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY')
        source_hash, status, corpus = conn.execute('SELECT source_sha256,status,report FROM _wikipedia_vec_import').fetchone()
        if status != 'complete':
            raise RuntimeError('PostgreSQL import is incomplete')
        from vec2plan_bandit.services.vector_contract import ENCODER_DIMENSIONS
        if (ENCODER_DIMENSIONS.get(corpus.get('embedding_model')) != args.dimension or
                corpus.get('dimension') != args.dimension):
            raise ValueError('Wiki asset dimensions must match the recorded corpus encoder')
        for table, columns in schema.items():
            vector_cols = [c for c, t in columns if t == 'vector']
            if not vector_cols:
                continue
            key = {'Articles': 'article_id', 'Headings': 'heading_id', 'Images': 'image_id', 'Paragraphs': 'paragraph_id'}[table]
            scalars = [(c, t) for c, t in columns if t != 'vector']
            for column in vector_cols:
                name = f'{args.prefix}_{table.lower()}_{column.lower()}_acorn_v1'
                destination = folder / table / column
                destination.mkdir(parents=True, exist_ok=True)
                vector_expr = sql.SQL('vector_send({})').format(sql.Identifier(column))
                statement = sql.SQL('SELECT {},{} FROM {} WHERE {} IS NOT NULL ORDER BY {}').format(
                    sql.SQL(',').join(sql.Identifier(c) for c, _ in scalars), vector_expr,
                    sql.Identifier(table), sql.Identifier(column), sql.Identifier(key))
                rows = conn.execute(statement).fetchall()
                ids = np.asarray([r[[c for c, _ in scalars].index(key)] for r in rows], dtype=np.int64)
                if not len(ids) or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
                    raise ValueError(f'invalid point IDs: {table}')
                vectors = []
                payloads = []
                for row in rows:
                    raw = bytes(row[-1])
                    if len(raw) != 4 + 4 * args.dimension or struct.unpack_from('>h', raw)[0] != args.dimension:
                        raise ValueError('invalid pgvector binary dimensions')
                    vectors.append(np.frombuffer(raw, dtype='>f4', offset=4).astype(np.float32))
                    payloads.append({c: value for (c, _), value in zip(scalars, row[:-1]) if value is not None})
                matrix = np.asarray(vectors, dtype=np.float32)
                if not np.all(np.isfinite(matrix)):
                    raise ValueError('nonfinite source vector')
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if np.any(norms <= 0):
                    raise ValueError('zero vectors cannot be indexed for cosine')
                matrix = np.ascontiguousarray(matrix / norms, dtype=np.float32)
                fingerprint = hashlib.sha256(ids.tobytes() + matrix.tobytes()).hexdigest()
                index = faiss.IndexHNSWFlat(args.dimension, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.add(matrix)
                temporary = destination / 'index.faiss.tmp'
                faiss.write_index(index, str(temporary))
                temporary.replace(destination / 'index.faiss')
                np.save(destination / 'row_ids.npy', ids)
                (destination / 'metadata.jsonl').write_text(''.join(json.dumps({'id': int(i), 'payload': p}, ensure_ascii=False) + '\n' for i, p in zip(ids, payloads)))
                if client.collection_exists(name):
                    raise FileExistsError(f'collection already exists; refusing overwrite: {name}')
                client.create_collection(name,
                    vectors_config=models.VectorParams(size=args.dimension, distance=models.Distance.COSINE, on_disk=True),
                    on_disk_payload=True,
                    hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200, on_disk=True, full_scan_threshold=10, max_indexing_threads=args.qdrant_indexing_threads),
                    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1, default_segment_number=1,
                        max_optimization_threads=args.qdrant_optimization_threads))
                payload_indexes = {}
                for field, kind in scalars:
                    if kind == 'bigint':
                        payload_indexes[field] = 'integer'
                        client.create_payload_index(name, field, models.PayloadSchemaType.INTEGER, wait=True)
                    elif field in {'title', 'filename', 'image_title', 'parsed_title', 'is_icon', 'on_commons', 'heading_text', 'url'}:
                        payload_indexes[field] = 'keyword'
                        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD, wait=True)
                for start in range(0, len(ids), 256):
                    client.upsert(name, [models.PointStruct(id=int(ids[i]), vector=matrix[i].tolist(), payload=payloads[i]) for i in range(start, min(start + 256, len(ids)))], wait=True)
                deadline = time.monotonic() + 600
                while True:
                    info = client.get_collection(name)
                    if str(info.status).lower() == 'red':
                        raise RuntimeError(f'Qdrant optimizer failed: {name}')
                    if str(info.status).lower() == 'green' and info.indexed_vectors_count >= len(ids):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'graph is still building: {name}')
                    time.sleep(2)
                if client.count(name, exact=True).count != len(ids):
                    raise ValueError('Qdrant count mismatch')
                # Re-read the serialized asset and validate against exact cosine.
                loaded = faiss.read_index(str(destination / 'index.faiss'))
                loaded.hnsw.efSearch = 200
                query = matrix[0]
                exact = 1.0 - matrix @ query
                k = min(5, len(ids))
                threshold = float(np.partition(exact, k - 1)[k - 1])
                _, positions = loaded.search(query.reshape(1, -1), k)
                if np.any(exact[positions[0]] > threshold + 1e-4):
                    raise ValueError('FAISS smoke search differs from exact top-k')
                response = client.query_points(name, query=query.tolist(), limit=k,
                    search_params=models.SearchParams(hnsw_ef=200, exact=False,
                        acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0)))
                by_id = {int(i): p for p, i in enumerate(ids)}
                if len(response.points) != k or any(exact[by_id[p.id]] > threshold + 1e-4 for p in response.points):
                    raise ValueError('Qdrant smoke search differs from exact top-k')
                filtered = client.query_points(name, query=query.tolist(), limit=1,
                    query_filter=models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=int(ids[0])))]),
                    search_params=models.SearchParams(exact=False, hnsw_ef=200, acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0)))
                if [p.id for p in filtered.points] != [int(ids[0])]:
                    raise ValueError('ACORN filtered smoke test failed')
                manifest = {'table': table, 'column': column, 'points': len(ids), 'dimension': args.dimension,
                    'row_id_column': key, 'distance': 'cosine', 'faiss_distance_output': 'cosine_similarity',
                    'normalized': args.normalized, 'embedding_model': corpus['embedding_model'], 'source_sha256': source_hash, 'vector_id_sha256': fingerprint,
                    'faiss_index': str(destination / 'index.faiss'), 'row_ids': str(destination / 'row_ids.npy'),
                    'metadata': str(destination / 'metadata.jsonl'), 'collection': name,
                    'payload_indexes': payload_indexes, 'm': 32, 'ef_construction': 200,
                    'indexed_vectors': info.indexed_vectors_count, 'full_scan_threshold_kb': 10,
                    'build_threads': threads, 'qdrant_indexing_threads': args.qdrant_indexing_threads,
                    'qdrant_optimization_threads': args.qdrant_optimization_threads,
                    'smoke_tests_passed': True}
                (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                assets.append(manifest)
                print(json.dumps({'stage': 'complete', 'table': table, 'column': column, 'points': len(ids)}), flush=True)
    receipt = {'status': 'complete', 'database': args.database, 'assets': assets,
               'embedding_model': corpus['embedding_model'], 'dimension':args.dimension, 'normalized':args.normalized,
               'asset_count': len(assets), 'total_vector_points': sum(a['points'] for a in assets),
               'qdrant_version': client.info().version, 'faiss_version': faiss.__version__,
               'acorn': {'enable': True, 'max_selectivity': 1.0},
               'distance': 'cosine',
               'limitation': 'Assets contain same-table payloads; cross-table filters and joins are evaluated by the SQL adapter.'}
    (folder / 'completed.json').write_text(json.dumps(receipt, indent=2) + '\n')
    client.close()
    print('All 6 Wikipedia FAISS and ACORN assets verified.', flush=True)


if __name__ == '__main__':
    main()
