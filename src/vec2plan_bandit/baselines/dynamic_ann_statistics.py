"""Label-free sampled cluster statistics and a documented density adaptation."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

VERSION = 'dynamicann-cluster-statistics-v1'
DENSITY_DEFINITION = 'rho=1/(1+sample_variance(L2_to_centroid)/max(mean(L2_to_centroid)^2,1e-12))'


def profile_index(index, index_path, *, sample_size=2048, clusters=16, seed=42):
    """Uniform point sample, independent of query IDs, filters and outcomes."""
    if sample_size < 2 or clusters < 1 or index.ntotal < 1:
        raise ValueError('invalid Dynamic ANN density profiling size')
    started = time.perf_counter()
    path = Path(index_path).resolve(); state = path.stat()
    points = np.random.default_rng(seed).choice(index.ntotal, min(sample_size, index.ntotal), replace=False)
    reconstruct = getattr(index, 'reconstruct_batch', None)
    vectors = (reconstruct(np.asarray(points, dtype=np.int64)) if reconstruct else
               [index.reconstruct(int(i)) for i in points])
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.all(np.isfinite(vectors)):
        raise ValueError('invalid vectors for density profiling')
    from sklearn.cluster import KMeans
    count = min(clusters, len(vectors), len(np.unique(vectors, axis=0)))
    model = KMeans(n_clusters=count, n_init=3, max_iter=50, random_state=seed).fit(vectors)
    descriptions = []
    for i, centroid in enumerate(model.cluster_centers_):
        distances = np.linalg.norm(vectors[model.labels_ == i] - centroid, axis=1)
        radius = float(np.mean(distances)); variance = float(np.var(distances, ddof=1)) if len(distances) > 1 else 0.
        descriptions.append(dict(sample_points=len(distances), radius_l2=radius,
            distance_variance=variance, density=float(1 / (1 + variance / max(radius ** 2, 1e-12)))))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_mtime_ns != state.st_mtime_ns or path.stat().st_size != state.st_size:
        raise ValueError('index changed during density profiling')
    return dict(total_rows=int(index.ntotal), embedding_dimension=int(vectors.shape[1]),
        index_path=str(path), index_size=state.st_size, index_mtime_ns=state.st_mtime_ns,
        index_sha256=digest.hexdigest(), sample_size=len(points), seed=seed,
        sample_positions_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        centroids=model.cluster_centers_.tolist(), clusters=descriptions,
        profile_time_ms=(time.perf_counter() - started) * 1000,
        input_vectors='stored normalized index vectors', query_labels_used=False)


class DensityCatalog:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text())
        if data.get('version') != VERSION or data.get('density_definition') != DENSITY_DEFINITION:
            raise ValueError('unsupported Dynamic ANN density artifact')
        self.assets = data['assets']
        self.centroids = {k: np.asarray(v['centroids'], dtype=np.float64) for k, v in self.assets.items()}
        self.validated = set()

    def resolve(self, key, vector, n, d, index_path):
        if key not in self.assets:
            raise ValueError(f'missing measured Dynamic ANN density for asset {key}')
        asset = self.assets[key]
        if (asset['total_rows'], asset['embedding_dimension']) != (n, d):
            raise ValueError('Dynamic ANN density corpus/dimension mismatch')
        if key not in self.validated:
            path = Path(index_path).resolve(); state = path.stat()
            if (str(path), state.st_size, state.st_mtime_ns) != (
                    asset['index_path'], asset['index_size'], asset['index_mtime_ns']):
                raise ValueError('Dynamic ANN density index snapshot changed; reprofile')
            self.validated.add(key)
        query = np.asarray(vector, dtype=np.float64)
        centroids = self.centroids[key]
        if query.shape != (d,) or centroids.ndim != 2 or centroids.shape[1] != d or not np.all(np.isfinite(query)):
            raise ValueError('invalid Dynamic ANN density query vector/centroids')
        cluster = int(np.argmin(np.sum((centroids - query) ** 2, axis=1)))
        return dict(density=asset['clusters'][cluster]['density'], density_cluster=cluster,
                    density_source='sampled_cluster_variance_proxy', density_definition=DENSITY_DEFINITION)


def index_inputs(worker, query):
    """Adapter capabilities only; no dataset-name routing decisions."""
    blocks = getattr(query, 'blocks', None)
    if blocks is None:
        key = query.embedding_column
        index, _ = worker._faiss_executor._load(key)
        path = worker._faiss_executor._assets[key].index_path
        return [(key, index, path, query.k)]
    result = []
    for block in blocks:
        key = block.asset_key
        if hasattr(worker, 'indexes'):
            index = worker.indexes[key][0]; path = worker.assets[key]['faiss_index']
        else:
            index = worker.cache[key]['index']; path = Path(worker.config['assets']['faiss_dir'])/key/'index.faiss'
        result.append((key, index, path, block.k))
    return result


def request_statistics(baseline, worker, query, vectors, statistics):
    statistics = dict(statistics or {})
    if baseline.density_mode == 'measured' and baseline._catalog is None:
        path = baseline.config.get('density_artifact')
        if not path:
            raise ValueError('measured Dynamic ANN density requires an offline density_artifact')
        baseline._catalog = DensityCatalog(path)
    result = []
    for i, (key, index, path, k) in enumerate(index_inputs(worker, query)):
        n, d = int(index.ntotal), int(index.d)
        counts = statistics.get('block_candidate_counts')
        sigma = (float(counts[i]) / n if counts is not None else statistics.get('estimated_selectivity'))
        block = dict(total_rows=n, embedding_dimension=d, k=k, estimated_selectivity=sigma)
        if baseline.density_mode == 'measured':
            block.update(baseline._catalog.resolve(key, vectors[i], n, d, path))
        else:
            block.update(density=baseline.fixed_density, density_source='fixed_ablation')
        result.append(block)
    statistics['dynamicann_blocks'] = result
    return statistics
