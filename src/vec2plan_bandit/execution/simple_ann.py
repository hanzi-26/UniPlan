"""Shared ID-only ANN primitives; callers prove semantics and provide identity.

FAISS and ACORN keep their existing search parameters and shortlist membership.
Both use the same local cosine ordering and row-identity materialization.
"""
import math
import numpy as np

from vec2plan_bandit.execution.detailed_timing import stage


def search_simple_ann(worker, index, vector, k, config, remote_search):
    if config.path_id == 'FAISS':
        index.hnsw.efSearch = config.params['ef_search']
        count = min(index.ntotal, max(k, math.ceil(k * config.params['candidate_multiplier'])))
        with stage(worker, 'ann_search_ms'):
            _, found = index.search(np.asarray(vector, dtype=np.float32).reshape(1, -1), count)
        # Keep the oversampled pool through reference/local reranking. Early
        # truncation loses near-tie candidates permanently, even on tiny data.
        return [int(i) for i in found[0] if i >= 0]
    if config.path_id == 'ACORN':
        with stage(worker, 'ann_search_ms'):
            return list(remote_search())[:k]
    raise ValueError('simple ANN only supports FAISS and ACORN')


def assemble_index_rows(index, vector, positions, identities, *, k, tie_key=tuple,
                        reference_order=None, audit=None):
    """Preserve shortlist membership, original row IDs and deterministic ties."""
    if len(positions) != len(identities):
        raise ValueError('incomplete ANN row-identity mapping')
    if not positions:
        return []
    reconstruct = getattr(index, 'reconstruct_batch', None)
    sources = (reconstruct(np.asarray(positions, dtype=np.int64)) if reconstruct else
               np.asarray([index.reconstruct(int(p)) for p in positions]))
    sources = np.asarray(sources, dtype=np.float64)
    query = np.asarray(vector, dtype=np.float64)
    # Match the existing local float64 reranker exactly (including reduction
    # order); a different vectorized reduction can reorder nearly equal scores.
    query_norm = np.linalg.norm(query)
    norms = np.asarray([np.linalg.norm(source) * query_norm for source in sources])
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError('invalid cosine vectors in ANN materialization')
    distances = 1. - np.asarray([np.dot(source, query) for source in sources]) / norms
    order = sorted(range(len(positions)), key=lambda i: (distances[i], tie_key(identities[i])))
    # A normalized float32 index can differ from the original vectors and from
    # float32 backend arithmetic. Do not silently change an explicit SQL order
    # on nearly equal scores. The reference callback orders ONLY this shortlist.
    guard = 8 * sources.shape[1] * np.finfo(np.float32).eps
    if reference_order is not None and len(order) > 1 and np.any(np.diff(distances[order]) <= guard):
        if audit is not None:
            audit['reference_order_used'] = True
        return reference_order()
    if audit is not None:
        audit['reference_order_used'] = False
    return [list(identities[i]) for i in order[:k]]
