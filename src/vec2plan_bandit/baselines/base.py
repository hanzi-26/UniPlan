from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from vec2plan_bandit.paths.registry import ExecutionConfig
from vec2plan_bandit.queries.loader import QueryRecord


@dataclass(frozen=True)
class BaselineRequest:
    """Dataset-independent input shared by all path-selection baselines."""

    query: QueryRecord
    query_context: np.ndarray
    available_paths: Mapping[str, bool]
    latency_cap_ms: float
    remaining_budget_ms: float
    available_pre_configurations: frozenset[str] | None = None
    query_statistics: Mapping[str, Any] | None = None


@runtime_checkable
class Baseline(Protocol):
    """Minimal extension point for comparable path/parameter baselines."""

    name: str

    def fit(self, observations: Sequence[Mapping[str, Any]]) -> "Baseline": ...

    def select(self, request: BaselineRequest) -> ExecutionConfig: ...
