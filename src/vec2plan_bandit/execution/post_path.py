from __future__ import annotations

from typing import Any, Mapping

from vec2plan_bandit.execution.path_base import (
    PathExecutionRequest, PathExecutionResult, remaining_timeout_ms,
)
from vec2plan_bandit.execution.pre_path import PostgresRunner
from vec2plan_bandit.paths.registry import POST, REGISTRY
from vec2plan_bandit.paths.sql import render_post


class PostPathExecutor:
    path_id = POST

    def __init__(self, pg_runner: PostgresRunner, availability_check: Any | None = None):
        self._pg_runner = pg_runner
        self._availability_check = availability_check

    def available(self, *, probe: bool = False, query: Any | None = None) -> bool:
        del query
        return True if self._availability_check is None else bool(self._availability_check(probe))

    def execute(
        self,
        request: PathExecutionRequest,
        params: Mapping[str, Any],
    ) -> PathExecutionResult:
        values = REGISTRY.normalize(POST, params)
        rendered = render_post(
            request.query,
            **values,
            iterative_scan="strict_order",
            scan_mem_multiplier=2.0,
        )
        timeout_ms = remaining_timeout_ms(request.deadline_ns)
        if timeout_ms is not None:
            from vec2plan_bandit.paths.sql import RenderedSQL
            rendered = RenderedSQL(
                rendered.path_id,
                rendered.setup_sql + (f"SET LOCAL statement_timeout = {timeout_ms}",),
                rendered.sql,
            )
        ids, latency = self._pg_runner(rendered, request.vector_literal)
        return PathExecutionResult(
            ids,
            latency,
            {
                **values,
                "iterative_scan": "strict_order",
                "scan_mem_multiplier": 2.0,
                "uses_hnsw": True,
                "result_complete": len(ids) >= request.query.k,
                "fallback_used": False,
            },
        )
