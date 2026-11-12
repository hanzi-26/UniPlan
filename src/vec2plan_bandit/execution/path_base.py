from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Protocol

import numpy as np

from vec2plan_bandit.queries.loader import QueryRecord


@dataclass(frozen=True)
class PathExecutionRequest:
    query: QueryRecord
    vector: np.ndarray
    vector_literal: str
    deadline_ns: int | None = None


@dataclass(frozen=True)
class PathExecutionResult:
    article_ids: list[int]
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)


class PathExecutor(Protocol):
    @property
    def path_id(self) -> str: ...

    def execute(
        self,
        request: PathExecutionRequest,
        params: Mapping[str, Any],
    ) -> PathExecutionResult: ...

    def available(self, *, probe: bool = False, query: QueryRecord | None = None) -> bool: ...


class PathUnavailableError(RuntimeError):
    pass


class PathDeadlineExceeded(TimeoutError):
    pass


def remaining_timeout_ms(deadline_ns: int | None) -> int | None:
    if deadline_ns is None:
        return None
    remaining = (deadline_ns - time.perf_counter_ns()) / 1_000_000
    if remaining <= 0:
        raise PathDeadlineExceeded("request deadline expired before path execution")
    return max(1, int(math.ceil(remaining)))
