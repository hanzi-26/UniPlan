from __future__ import annotations

from typing import Any, Iterable, Mapping

from vec2plan_bandit.execution.path_base import (
    PathExecutionRequest,
    PathExecutionResult,
    PathExecutor,
    PathUnavailableError,
)
from vec2plan_bandit.paths.registry import PATH_IDS, REGISTRY


class PathRouter:
    """Backend-neutral dispatcher for the four-path protocol."""

    def __init__(self, executors: Iterable[PathExecutor] = ()):
        self._executors: dict[str, PathExecutor] = {}
        for executor in executors:
            self.register(executor)

    def register(self, executor: PathExecutor, *, replace: bool = False) -> None:
        path_id = executor.path_id
        REGISTRY.get(path_id)
        if path_id in self._executors and not replace:
            raise ValueError(f"executor already registered for {path_id}")
        self._executors[path_id] = executor

    def available(self, *, probe: bool = False, query: Any | None = None) -> dict[str, bool]:
        readiness: dict[str, bool] = {}
        for path_id in PATH_IDS:
            executor = self._executors.get(path_id)
            if executor is None:
                readiness[path_id] = False
                continue
            checker = getattr(executor, "available", None)
            try:
                readiness[path_id] = True if checker is None else bool(
                    checker(probe=probe, query=query)
                )
            except Exception:
                readiness[path_id] = False
        return readiness

    def execute(
        self,
        path_id: str,
        request: PathExecutionRequest,
        params: Mapping[str, Any],
    ) -> PathExecutionResult:
        normalized = REGISTRY.normalize(path_id, params)
        try:
            executor = self._executors[path_id]
        except KeyError as error:
            raise PathUnavailableError(f"path executor is not configured: {path_id}") from error
        if not self.available(probe=False, query=request.query)[path_id]:
            raise PathUnavailableError(f"path is not operationally available: {path_id}")
        return executor.execute(request, normalized)
