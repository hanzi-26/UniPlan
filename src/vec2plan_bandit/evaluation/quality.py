from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


RECALL_PASS_THRESHOLD = 0.9


@dataclass(frozen=True)
class QualityResult:
    k: int
    gt_size: int
    denominator: int
    hits_at_k: int
    recall_at_k: float | None
    quality_pass: bool | None
    corpus_has_k: bool


def required_hits(k: int, denominator: int | None = None, *, recall_threshold: float = RECALL_PASS_THRESHOLD) -> int:
    """Integer hit threshold; insufficient ground truth stays non-primary."""
    if not math.isfinite(recall_threshold) or not 0 <= recall_threshold <= 1:
        raise ValueError("recall_threshold must be within [0, 1]")
    if k not in range(1, 11):
        raise ValueError("query protocol requires 1 <= k <= 10")
    size = k if denominator is None else int(denominator)
    if size < 0 or size > k:
        raise ValueError("denominator must be between zero and k")
    if size == 0:
        return 0
    return math.ceil(recall_threshold * size - 1e-12)


def evaluate_small_k_quality(
    result_ids: Sequence[int],
    ground_truth_ids: Sequence[int],
    k: int,
    *,
    recall_threshold: float = RECALL_PASS_THRESHOLD,
) -> QualityResult:
    """Evaluate ordered top-k retrieval using unique article IDs."""
    if k not in range(1, 11):
        raise ValueError("query protocol requires 1 <= k <= 10")
    returned = [int(value) for value in result_ids[:k]]
    truth = [int(value) for value in ground_truth_ids[:k]]
    if len(returned) != len(set(returned)):
        raise ValueError("result_ids must be unique within top-k")
    if len(truth) != len(set(truth)):
        raise ValueError("ground_truth_ids must be unique within top-k")
    denominator = min(k, len(truth))
    if denominator == 0:
        return QualityResult(k, len(truth), 0, 0, None, None, False)
    hits = len(set(returned).intersection(truth))
    return QualityResult(
        k=k,
        gt_size=len(truth),
        denominator=denominator,
        hits_at_k=hits,
        recall_at_k=hits / denominator,
        quality_pass=hits >= required_hits(k, denominator, recall_threshold=recall_threshold),
        corpus_has_k=len(truth) >= k,
    )
