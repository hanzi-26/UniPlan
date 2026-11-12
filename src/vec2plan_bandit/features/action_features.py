from __future__ import annotations

import math
from functools import lru_cache
import numpy as np

from vec2plan_bandit.paths.registry import ACTIONS, ExecutionConfig

ACTION_FEATURE_VERSION = "joint-action-v5-pre-access-join-strategy"


ACTION_FEATURE_NAMES = (
    "path_pre",
    "path_post",
    "path_faiss",
    "path_acorn",
    "pre_uses_index",
    "pre_btree",
    "pre_hash",
    "pre_bitmap",
    "pre_join_all",
    "pre_join_hash",
    "pre_join_merge",
    "pre_join_nested_loop",
    "log1p_ef_search",
    "log1p_acorn_hnsw_ef",
    "log1p_max_scan_tuples",
    "log1p_candidate_multiplier",
    "candidate_count_x_post_ef",
    "candidate_count_x_post_scan",
    "candidate_count_x_faiss_ef",
    "candidate_count_x_faiss_multiplier",
    "candidate_count_x_acorn_ef",
    "selectivity_x_acorn_ef",
    "no_filter_x_acorn",
    "candidate_count_x_pre_hash_join",
    "candidate_count_x_pre_merge_join",
    "candidate_count_x_pre_nested_loop",
    "selectivity_x_pre_hash_join",
    "selectivity_x_pre_merge_join",
    "selectivity_x_pre_nested_loop",
    "join_count_x_pre_hash_join",
    "join_count_x_pre_merge_join",
    "join_count_x_pre_nested_loop",
)


def joint_action_features(
    query_context: np.ndarray,
    config: ExecutionConfig,
    *,
    latency_cap_ms: float,
    remaining_budget_ms: float,
) -> np.ndarray:
    """Append path and parameter features; the deadline stays a runtime constraint.

    Budget arguments remain in the shared selector interface. They are not
    learned features: bootstrap collection cannot teach a constant budget column.
    """
    context = np.asarray(query_context, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(context)):
        raise ValueError("query context contains non-finite values")
    if config.action_id not in range(len(ACTIONS)):
        raise ValueError("unknown action_id")
    if ACTIONS[config.action_id].path_id != config.path_id:
        raise ValueError("action_id/path_id mismatch")
    if not math.isfinite(latency_cap_ms) or not math.isfinite(remaining_budget_ms) or latency_cap_ms <= 0 or remaining_budget_ms < 0:
        raise ValueError("invalid latency budget")
    path = np.zeros(len(ACTIONS), dtype=np.float64)
    path[config.action_id] = 1.0
    params = config.params
    values = np.asarray([
        float(params.get("index_mode") == "index"),
        float(params.get("index_strategy") == "btree"),
        float(params.get("index_strategy") == "hash"),
        float(params.get("index_strategy") == "bitmap"),
        float(params.get("join_strategy") == "all"),
        float(params.get("join_strategy") == "hash"),
        float(params.get("join_strategy") == "merge"),
        float(params.get("join_strategy") == "nested_loop"),
        math.log1p(float(params.get("ef_search", 0))) / math.log1p(800),
        math.log1p(float(params.get("hnsw_ef", 0))) / math.log1p(800),
        math.log1p(float(params.get("max_scan_tuples", 0))) / math.log1p(200_000),
        math.log1p(float(params.get("candidate_multiplier", 0))) / math.log1p(50),
    ])
    # FeaturePipeline context layout is intercept followed by base features.
    # These explicit parameter/query interactions matter in small data: the
    # cost of ef/scan depth changes with candidate count and selectivity.
    selectivity = context[10] if len(context) > 10 else 0.0
    log_candidate_count = context[12] if len(context) > 12 else 0.0
    no_filter = context[14] if len(context) > 14 else 0.0
    log_join_count = context[15] if len(context) > 15 else 0.0
    interactions = np.asarray([
        log_candidate_count * path[1] * values[8],
        log_candidate_count * path[1] * values[10],
        log_candidate_count * path[2] * values[8],
        log_candidate_count * path[2] * values[11],
        log_candidate_count * path[3] * values[9],
        selectivity * path[3] * values[9],
        no_filter * path[3],
        log_candidate_count * path[0] * values[5],
        log_candidate_count * path[0] * values[6],
        log_candidate_count * path[0] * values[7],
        selectivity * path[0] * values[5],
        selectivity * path[0] * values[6],
        selectivity * path[0] * values[7],
        log_join_count * path[0] * values[5],
        log_join_count * path[0] * values[6],
        log_join_count * path[0] * values[7],
    ])
    # Shared query terms transfer evidence across configurations, while the
    # Kronecker block lets the same query behave differently for each path.
    return np.concatenate((context, np.kron(path, context), path, values, interactions))


@lru_cache(maxsize=32)
def _static_action_values(params):
    return _action_values(dict(params))


def _action_values(p):
    return (
        float(p.get('index_mode') == 'index'), float(p.get('index_strategy') == 'btree'),
        float(p.get('index_strategy') == 'hash'), float(p.get('index_strategy') == 'bitmap'),
        float(p.get('join_strategy') == 'all'), float(p.get('join_strategy') == 'hash'),
        float(p.get('join_strategy') == 'merge'), float(p.get('join_strategy') == 'nested_loop'),
        math.log1p(float(p.get('ef_search', 0))) / math.log1p(800),
        math.log1p(float(p.get('hnsw_ef', 0))) / math.log1p(800),
        math.log1p(float(p.get('max_scan_tuples', 0))) / math.log1p(200_000),
        math.log1p(float(p.get('candidate_multiplier', 0))) / math.log1p(50))


def joint_action_feature_matrix(query_context, configs, *, latency_cap_ms, remaining_budget_ms):
    """Build the unchanged joint-action feature schema for all candidates at once."""
    context = np.asarray(query_context, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(context)):
        raise ValueError('query context contains non-finite values')
    if not math.isfinite(latency_cap_ms) or not math.isfinite(remaining_budget_ms) or latency_cap_ms <= 0 or remaining_budget_ms < 0:
        raise ValueError('invalid latency budget')
    configs = list(configs)
    n, d, paths = len(configs), len(context), len(ACTIONS)
    matrix = np.empty((n, d * (paths + 1) + len(ACTION_FEATURE_NAMES)), dtype=np.float64)
    matrix[:, :d] = context
    onehot = np.zeros((n, paths), dtype=np.float64)
    values = np.empty((n, 12), dtype=np.float64)
    for i, config in enumerate(configs):
        if config.action_id not in range(paths) or ACTIONS[config.action_id].path_id != config.path_id:
            raise ValueError('action_id/path_id mismatch')
        onehot[i, config.action_id] = 1.
        values[i] = (_static_action_values(tuple(sorted(config.params.items())))
                     if config.path_id == 'PRE' else _action_values(config.params))
    matrix[:, d:d*(paths+1)] = (onehot[:, :, None] * context[None, None, :]).reshape(n, paths*d)
    offset = d*(paths+1)
    matrix[:, offset:offset+paths] = onehot
    matrix[:, offset+paths:offset+paths+12] = values
    s = context[10] if d > 10 else 0.
    m = context[12] if d > 12 else 0.
    no_filter = context[14] if d > 14 else 0.
    joins = context[15] if d > 15 else 0.
    matrix[:, offset+paths+12:] = np.column_stack((
        m*onehot[:,1]*values[:,8], m*onehot[:,1]*values[:,10],
        m*onehot[:,2]*values[:,8], m*onehot[:,2]*values[:,11],
        m*onehot[:,3]*values[:,9], s*onehot[:,3]*values[:,9], no_filter*onehot[:,3],
        m*onehot[:,0]*values[:,5], m*onehot[:,0]*values[:,6], m*onehot[:,0]*values[:,7],
        s*onehot[:,0]*values[:,5], s*onehot[:,0]*values[:,6], s*onehot[:,0]*values[:,7],
        joins*onehot[:,0]*values[:,5], joins*onehot[:,0]*values[:,6], joins*onehot[:,0]*values[:,7]))
    return matrix
