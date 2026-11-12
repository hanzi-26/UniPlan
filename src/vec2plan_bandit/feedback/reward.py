from __future__ import annotations

import math

REWARD_VERSION = "source-k-1-to-10-recall-0.9-e2e-soft-10s-v5"


def reward_version(recall_threshold: float = 0.9) -> str:
    return f"source-k-1-to-10-recall-{recall_threshold:g}-e2e-soft-10s-v5"


def compute_reward(
    *,
    e2e_latency_ms: float,
    recall_at_k: float | None,
    latency_cap_ms: float,
    success: bool,
    result_complete: bool | None = None,
    fallback_used: bool = False,
    timeout: bool = False,
    recall_threshold: float = 0.9,
    quality_pass: bool | None = None,
) -> float:
    """Quality-constrained latency reward used by collection and replay."""
    if not math.isfinite(float(latency_cap_ms)) or latency_cap_ms <= 0:
        raise ValueError("latency_cap_ms must be finite and positive")
    if not math.isfinite(float(recall_threshold)) or not 0 <= recall_threshold <= 1:
        raise ValueError("recall_threshold must be within [0, 1]")
    if (
        not success
        or result_complete is False
        or fallback_used
        or timeout
        or recall_at_k is None
    ):
        return -1.0
    try:
        latency = float(e2e_latency_ms)
        recall = float(recall_at_k)
    except (TypeError, ValueError):
        return -1.0
    if not math.isfinite(latency) or not math.isfinite(recall) or not 0 <= recall <= 1:
        return -1.0
    accepted_quality = recall >= recall_threshold and (quality_pass is None or bool(quality_pass))
    # Quality acceptance never overrides an explicitly incomplete top-k result.
    if not accepted_quality or latency < 0 or latency > latency_cap_ms:
        return -1.0
    return min(1.0, max(0.0, 1.0 - latency / latency_cap_ms))
