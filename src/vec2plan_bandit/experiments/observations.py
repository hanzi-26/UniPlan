from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vec2plan_bandit.features.action_features import joint_action_features
from vec2plan_bandit.features.feature_pipeline import FeaturePipeline
from vec2plan_bandit.feedback.reward import REWARD_VERSION, reward_version
from vec2plan_bandit.paths.registry import make_execution_config
from vec2plan_bandit.queries.loader import QueryRecord


def build_training_observations(
    records: Sequence[QueryRecord],
    split: Mapping[str, Sequence[str]],
    matrix_rows: Iterable[Mapping[str, Any]],
    *,
    latency_cap_ms: float,
    recall_threshold: float = 0.9,
    statistics: Mapping[str, Mapping[str, Any]] | None = None,
    embeddings: Mapping[str, np.ndarray] | None = None,
) -> tuple[FeaturePipeline, list[dict[str, Any]]]:
    """Create the shared, leakage-safe representation used by every baseline."""
    by_id = {query.query_id: query for query in records}
    train_ids = [str(value) for value in split["train"]]
    if set(train_ids) - set(by_id):
        raise ValueError("train split references unknown queries")
    roles = {name: set(split.get(name, [])) for name in ("train", "dev", "test")}
    if any(queries - set(by_id) for queries in roles.values()):
        raise ValueError("split references unknown queries")
    for i, left in enumerate(roles):
        for right in list(roles)[i + 1:]:
            if roles[left] & roles[right]:
                raise ValueError("query splits overlap")
            left_groups = {by_id[q].group_id for q in roles[left] if q in by_id}
            right_groups = {by_id[q].group_id for q in roles[right] if q in by_id}
            if left_groups & right_groups:
                raise ValueError("query groups overlap across splits")
    pipeline = FeaturePipeline().fit(records, train_ids, split="train", statistics=statistics, embeddings=embeddings)
    contexts = pipeline.transform(records, statistics=statistics, embeddings=embeddings)
    observations: list[dict[str, Any]] = []
    for row in matrix_rows:
        query_id = str(row["query_id"])
        if query_id not in set(train_ids):
            continue
        if row.get("reward_version", REWARD_VERSION) != reward_version(recall_threshold):
            raise ValueError("obsolete reward version; recollect or re-adjudicate training observations")
        if float(row.get("recall_threshold", 0.9)) != recall_threshold:
            raise ValueError("training label Recall threshold does not match experiment")
        observed_cap = float(row.get("requested_budget_ms", latency_cap_ms))
        if abs(observed_cap - latency_cap_ms) > 1e-9:
            continue
        config = make_execution_config(str(row["path_id"]), row.get("requested_params"))
        supplied_id = row.get("config_id")
        if supplied_id is not None and str(supplied_id) != config.config_id:
            raise ValueError(f"configuration identity mismatch: {supplied_id}")
        context = contexts[query_id]
        features = joint_action_features(
            context,
            config,
            latency_cap_ms=latency_cap_ms,
            remaining_budget_ms=float(row.get("remaining_budget_ms", latency_cap_ms)),
        )
        latency_ms = float(row.get("e2e_latency_ms", float("nan")))
        quality_target = float(
            row.get("quality_pass") is True
            and bool(row.get("result_full_k", False))
            and not bool(row.get("fallback_used", False))
            and not bool(row.get("execution_details", {}).get("fallback_used", False))
            and not bool(row.get("error_type"))
        )
        if not np.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("training observations require finite nonnegative E2E latency")
        observations.append({
            "query_id": query_id,
            "group_id": by_id[query_id].group_id,
            "split": "train", "k": by_id[query_id].k,
            "query_context": context,
            "query_statistics": dict((statistics or {}).get(query_id) or {}),
            "path_id": config.path_id,
            "config_id": config.config_id,
            "requested_params": dict(config.params),
            "features": features,
            "reward": float(row["reward"]),
            "quality_target": quality_target,
            "e2e_latency_ms": latency_ms,
            "execution_ms": row.get('execution_ms',(row.get('spans_ms') or {}).get('execution_ms')),
            "execution_stages_ms":row.get('execution_stages_ms',row.get('execution_details',{}).get('execution_stages_ms',{})),
            "eligible_for_primary_quality": row.get('eligible_for_primary_quality',row.get('corpus_has_k',True)),
            "main_routing_facts": row.get('main_routing_facts'),
            "timed_out": bool(row.get('timed_out', False)),
            "latency_censored": bool(row.get('latency_censored', False)),
            "error_type": row.get('error_type'),
            "fallback_used": bool(row.get('fallback_used', False)),
        })
    if not observations:
        raise ValueError("training matrix contains no matching train observations")
    labels = np.asarray([row["reward"] for row in observations])
    if not np.all(np.isfinite(labels)) or np.any(np.abs(labels) > 1):
        raise ValueError("training rewards must be finite and in [-1, 1]")
    return pipeline, observations
