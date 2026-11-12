"""Resumable, bounded-memory YFCC HNSW/ACORN construction from BigANN files.

Faiss shards use exact uint8->float32 conversion and squared L2. Search every
shard and merge by distance; row ID = shard start + local Faiss ID.
Qdrant Euclid returns L2 (take its square to compare to Faiss distances).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import traceback

import numpy as np


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def emit(folder, stage, **details):
    value = dict(stage=stage, pid=os.getpid(), time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details)
    write_json(folder / 'status.json', value)
    print(json.dumps(value), flush=True)


def u8bin(path):
    with path.open('rb') as handle:
        n, d = struct.unpack('<II', handle.read(8))
    if path.stat().st_size != 8 + n * d or n < 1 or d < 1:
        raise ValueError(f'Invalid u8bin: {path}')
    return np.memmap(path, dtype=np.uint8, offset=8, shape=(n, d), mode='r')


def metadata(path, n):
    with path.open('rb') as handle:
        rows, cols, nnz = struct.unpack('<3q', handle.read(24))
    if rows != n or path.stat().st_size != 24 + 8 * (rows + 1) + 8 * nnz:
        raise ValueError('Metadata dimensions/size do not match source')
    ptr = np.memmap(path, dtype='<i8', offset=24, shape=(rows + 1,), mode='r')
    tags = np.memmap(path, dtype='<i4', offset=24 + 8 * (rows + 1), shape=(nnz,), mode='r')
    if ptr[0] != 0 or ptr[-1] != nnz:
        raise ValueError('Invalid CSR offsets')
    for start in range(0, n, 100000):
        end = min(n, start + 100000)
        indices = tags[ptr[start]:ptr[end]]
        if np.any(np.diff(ptr[start:end + 1]) < 0) or np.any(indices < 0) or np.any(indices >= cols):
            raise ValueError('Invalid CSR contents')
    return ptr, tags


def source_signature(args):
    result = {}
    for name in ('source', 'metadata', 'equal', 'range'):
        path = getattr(args, name)
        stat = path.stat()
        with path.open('rb') as handle:
            prefix = handle.read(1024 * 1024)
        result[name] = dict(path=str(path.resolve()), bytes=stat.st_size,
                            mtime_ns=stat.st_mtime_ns, prefix_sha256=hashlib.sha256(prefix).hexdigest())
    return result


def contract(args, vectors):
    result = dict(version='yfcc-index-v1', count=len(vectors), dimension=vectors.shape[1],
                metric='cosine' if args.metric == 'cosine' else 'L2', normalized=args.metric == 'cosine', id_mapping='zero-based source row',
                sources=source_signature(args), m=args.m, ef_construction=args.ef_construction,
                shard_rows=args.shard_rows, collection=args.collection, url=args.url,
                grpc_port=args.grpc_port)
    if args.metric == 'cosine':
        result.update(max_segment_size=args.max_segment_size, target_segments=args.target_segments,
                      optimization_jobs=args.optimization_jobs)
    return result


def float_vectors(values, metric):
    result = np.array(values, dtype=np.float32, copy=True)
    if metric == 'cosine':
        norms = np.linalg.norm(result, axis=1)
        if np.any(norms == 0):
            raise ValueError('Cosine is undefined for zero vectors')
        result /= norms[:, None]
    return result


def build_faiss(args, folder, vectors, spec):
    import faiss
    faiss.omp_set_num_threads(args.threads)
    n, d = vectors.shape
    shards = []
    started = time.monotonic()
    for shard, start in enumerate(range(0, n, args.shard_rows)):
        end = min(n, start + args.shard_rows)
        target = folder / f'shard-{shard:04d}.faiss'
        receipt = target.with_suffix('.json')
        if target.exists() and receipt.exists():
            item = json.loads(receipt.read_text())
            if item['start'] != start or item['end'] != end or item['bytes'] != target.stat().st_size:
                raise ValueError(f'Invalid shard receipt {receipt}')
            shards.append(item)
            continue
        if target.exists():
            raise ValueError(f'Shard without receipt; inspect before resuming: {target}')
        # One shard at a time keeps vectors + graph below ~3 GiB by default.
        cosine = args.metric == 'cosine'
        index = faiss.IndexHNSWFlat(d, args.m, faiss.METRIC_INNER_PRODUCT if cosine else faiss.METRIC_L2)
        index.hnsw.efConstruction = args.ef_construction
        emit(folder, 'building', shard=shard, indexed=start, total=n, threads=args.threads)
        for batch_start in range(start, end, args.batch_rows):
            batch_end = min(end, batch_start + args.batch_rows)
            index.add(float_vectors(vectors[batch_start:batch_end], args.metric))
            emit(folder, 'building', shard=shard, indexed=batch_end, total=n,
                 elapsed_s=round(time.monotonic() - started, 1), threads=args.threads)
        if index.ntotal != end - start:
            raise ValueError('Faiss count mismatch')
        probe = float_vectors(vectors[start:start + min(8, end - start)], args.metric)
        index.hnsw.efSearch = 200
        distances, ids = index.search(probe, 1)
        if np.any(ids < 0) or not np.allclose(distances[:, 0], 1. if cosine else 0., atol=2e-5):
            raise ValueError('Faiss self-query smoke test failed')
        partial = target.with_suffix('.partial')
        faiss.write_index(index, str(partial))
        partial.replace(target)
        item = dict(file=target.name, start=start, end=end, count=end-start, bytes=target.stat().st_size)
        write_json(receipt, item)
        shards.append(item)
        del index
    write_json(folder / 'manifest.json', dict(**spec, index_type='IndexHNSWFlat',
               faiss_version=faiss.__version__, build_threads=args.threads, shards=shards,
               distance_output='cosine_similarity' if args.metric == 'cosine' else 'squared_L2',
               search_contract='search all shards; add start to IDs; merge descending similarity' if args.metric == 'cosine'
                   else 'search all shards; add start to IDs; merge ascending distance'))
    emit(folder, 'complete', indexed=n, total=n, shards=len(shards))


def scalar_bin(path, dtype, n):
    with path.open('rb') as handle:
        rows, cols = struct.unpack('<II', handle.read(8))
    if (rows, cols) != (n, 1) or path.stat().st_size != 8 + n * np.dtype(dtype).itemsize:
        raise ValueError(f'Invalid scalar data: {path}')
    return np.memmap(path, dtype=dtype, offset=8, shape=(n,), mode='r')


def local_graph_progress(folder, collection):
    """M=0 HNSW files count as indexed in Qdrant, but have no search graph."""
    segments = folder.parent / 'qdrant/storage/collections' / collection / '0/segments'
    if not segments.exists():
        return {}
    graphs, points = 0, 0
    for path in segments.glob('*/vector_index/hnsw_config.json'):
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # An optimizer can atomically replace this directory.
        if config.get('m', 0) > 0:
            graphs += 1
            points += config.get('indexed_vector_count', 0)
    return dict(graph_built_points=points, graph_built_segments=graphs)


def build_acorn(args, folder, vectors, spec):
    from qdrant_client import QdrantClient, models
    n, d = vectors.shape
    cosine = args.metric == 'cosine'
    distance = models.Distance.COSINE if cosine else models.Distance.EUCLID
    ptr, tags = metadata(args.metadata, n)
    equal = scalar_bin(args.equal, '<u2', n)
    ranges = scalar_bin(args.range, '<f4', n)
    if not np.all(np.isfinite(ranges)):
        raise ValueError('Nonfinite range attributes')
    client = QdrantClient(url=args.url, grpc_port=args.grpc_port, prefer_grpc=True,
                          timeout=600, trust_env=False)
    try:
        checkpoint = folder / 'upload.json'
        if not checkpoint.exists():
            if client.collection_exists(args.collection):
                raise ValueError('Collection exists without ownership checkpoint; refusing to modify')
            write_json(checkpoint, dict(next_row=0, collection=args.collection, creating=True))
        state = json.loads(checkpoint.read_text())
        if state['collection'] != args.collection:
            raise ValueError('Collection checkpoint mismatch')
        if state.get('creating') and not client.collection_exists(args.collection):
            client.create_collection(args.collection,
                vectors_config=models.VectorParams(size=d, distance=distance,
                    datatype=models.Datatype.FLOAT32 if cosine else models.Datatype.UINT8, on_disk=True),
                shard_number=1, on_disk_payload=True,
                hnsw_config=models.HnswConfigDiff(m=0, ef_construct=args.ef_construction,
                    full_scan_threshold=0, on_disk=True, max_indexing_threads=args.threads),
                optimizers_config=models.OptimizersConfigDiff(default_segment_number=args.target_segments,
                    memmap_threshold=20000, indexing_threshold=20000,
                    max_segment_size=args.max_segment_size, max_optimization_threads=args.optimization_jobs))
        if state.get('creating'):
            client.create_payload_index(args.collection, 'tags',
                models.IntegerIndexParams(type=models.IntegerIndexType.INTEGER,
                    lookup=True, range=False, on_disk=True), wait=True)
            client.create_payload_index(args.collection, 'equal', models.PayloadSchemaType.INTEGER, wait=True)
            client.create_payload_index(args.collection, 'range', models.PayloadSchemaType.FLOAT, wait=True)
            state = dict(next_row=0, collection=args.collection, creating=False)
            write_json(checkpoint, state)
        info = client.get_collection(args.collection)
        if info.config.params.vectors.size != d or info.config.params.vectors.distance != distance:
            raise ValueError('Collection vector config mismatch')
        started = time.monotonic()
        for start in range(state['next_row'], n, args.upload_rows):
            end = min(n, start + args.upload_rows)
            payload = [dict(tags=tags[ptr[row]:ptr[row + 1]].tolist(),
                            equal=int(equal[row]), range=float(ranges[row]))
                       for row in range(start, end)]
            # Bounded batches, gRPC transport, idempotent IDs and acknowledged writes.
            client.upsert(args.collection, models.Batch(ids=list(range(start, end)),
                vectors=float_vectors(vectors[start:end], args.metric).tolist(), payloads=payload), wait=True)
            write_json(checkpoint, dict(next_row=end, collection=args.collection, creating=False))
            if start == state['next_row'] or end % (args.upload_rows * 32) == 0 or end == n:
                emit(folder, 'uploading', uploaded=end, total=n, elapsed_s=round(time.monotonic()-started, 1))
        if client.count(args.collection, exact=True).count != n:
            raise ValueError('Qdrant point count mismatch')
        info = client.get_collection(args.collection)
        if (info.config.hnsw_config.m != args.m
                or info.config.hnsw_config.max_indexing_threads != args.threads
                or info.config.optimizer_config.indexing_threshold != 1):
            client.update_collection(args.collection,
                hnsw_config=models.HnswConfigDiff(m=args.m, max_indexing_threads=args.threads),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))
        while True:
            info = client.get_collection(args.collection)
            emit(folder, 'indexing', points=info.points_count, indexed=info.indexed_vectors_count,
                 total=n, status=str(info.status), optimizer=str(info.optimizer_status),
                 threads=info.config.hnsw_config.max_indexing_threads,
                 optimization_jobs=info.config.optimizer_config.max_optimization_threads,
                 **local_graph_progress(folder, args.collection))
            if info.optimizer_status != 'ok':
                raise RuntimeError(f'Qdrant optimizer failed: {info.optimizer_status}')
            if info.points_count == n and info.indexed_vectors_count == n and str(info.status).lower() == 'green':
                break
            time.sleep(30)
        first_tags = tags[ptr[0]:ptr[1]].tolist()
        if not first_tags:
            raise ValueError('Smoke probe requires a tag on row zero')
        result = client.query_points(args.collection, query=float_vectors(vectors[:1], args.metric)[0].tolist(),
            query_filter=models.Filter(must=[models.FieldCondition(key='tags', match=models.MatchValue(value=first_tags[0]))]),
            limit=1, with_payload=True,
            search_params=models.SearchParams(exact=False, hnsw_ef=200,
                acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0))).points
        if not result or abs(result[0].score - (1. if cosine else 0.)) > 0.001 or first_tags[0] not in result[0].payload['tags']:
            raise ValueError('ACORN filtered self-query failed')
        write_json(folder / 'manifest.json', dict(**spec, backend='Qdrant ACORN',
            datatype='float32' if cosine else 'uint8', build_threads=info.config.hnsw_config.max_indexing_threads,
            max_optimization_threads=info.config.optimizer_config.max_optimization_threads,
            indexed_vectors=info.indexed_vectors_count,
            distance_output='cosine_similarity' if cosine else 'L2', on_disk=True, acorn_search=dict(enable=True, max_selectivity=1.0)))
        emit(folder, 'complete', indexed=n, total=n, collection=args.collection)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['faiss', 'acorn'])
    parser.add_argument('--source', type=Path, default=Path('assets/yfcc/base.10M.1920d.u8bin'))
    parser.add_argument('--metadata', type=Path, default=Path('assets/yfcc/base.metadata.10M.spmat'))
    parser.add_argument('--equal', type=Path, default=Path('assets/yfcc/equal.u2bin'))
    parser.add_argument('--range', type=Path, default=Path('assets/yfcc/range.f4bin'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--ef-construction', type=int, default=200)
    parser.add_argument('--shard-rows', type=int, default=250000)
    parser.add_argument('--batch-rows', type=int, default=4096)
    parser.add_argument('--upload-rows', type=int, default=256)
    parser.add_argument('--url', default='http://127.0.0.1:19343')
    parser.add_argument('--grpc-port', type=int, default=19344)
    parser.add_argument('--collection', default='yfcc_10m_1920d_l2_acorn_v1')
    parser.add_argument('--metric', choices=('l2', 'cosine'), default='l2')
    parser.add_argument('--optimization-jobs', type=int, default=1)
    parser.add_argument('--max-segment-size', type=int, default=200000)
    parser.add_argument('--target-segments', type=int, default=16)
    args = parser.parse_args()
    for key in ('threads', 'm', 'ef_construction', 'shard_rows', 'batch_rows', 'upload_rows',
                'optimization_jobs', 'max_segment_size', 'target_segments'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        vectors = u8bin(args.source)
        spec = contract(args, vectors)
        contract_path = args.output / 'contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text()) != spec:
            raise ValueError('Source or build parameters changed; use a fresh output')
        write_json(contract_path, spec)
        emit(args.output, 'starting', backend=args.backend, total=len(vectors), dimension=vectors.shape[1])
        if args.backend == 'faiss':
            build_faiss(args, args.output, vectors, spec)
        else:
            build_acorn(args, args.output, vectors, spec)
    except BaseException as error:
        emit(args.output, 'failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
