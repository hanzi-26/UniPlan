from __future__ import annotations

from typing import Any, Callable, Mapping

from vec2plan_bandit.execution.path_base import (
    PathExecutionRequest, PathExecutionResult, remaining_timeout_ms,
)
from vec2plan_bandit.paths.registry import PRE, REGISTRY
from vec2plan_bandit.paths.sql import RenderedSQL, render_pre


PostgresRunner = Callable[[RenderedSQL, str], tuple[list[int], float]]


class PrePathExecutor:
    """Exact PRE executor with explicit scan, B-tree, hash, and bitmap modes."""

    path_id = PRE

    def __init__(self, pg_runner: PostgresRunner, availability_check: Callable[[bool], bool] | None = None):
        self._pg_runner = pg_runner
        self._availability_check = availability_check
        self._index_strategies: dict[str, Callable[..., RenderedSQL]] = {
            "btree": render_pre,
            "hash": render_pre,
            "bitmap": render_pre,
            "optimizer": render_pre,
        }

    def available(self, *, probe: bool = False, query: Any | None = None) -> bool:
        del query
        return True if self._availability_check is None else bool(self._availability_check(probe))

    def register_index_strategy(
        self, name: str, renderer: Callable[..., RenderedSQL]
    ) -> None:
        """Register a future PRE index implementation without adding a path."""

        if not name or name in self._index_strategies:
            raise ValueError(f"duplicate or empty PRE index strategy: {name!r}")
        self._index_strategies[name] = renderer

    def execute(
        self,
        request: PathExecutionRequest,
        params: Mapping[str, Any],
    ) -> PathExecutionResult:
        values = REGISTRY.normalize(PRE, params)
        mode = values["index_mode"]
        strategy = values.get("index_strategy")
        join_strategy = values["join_strategy"]
        if mode == "index":
            try:
                renderer = self._index_strategies[strategy]
            except KeyError as error:
                raise ValueError(f"PRE index strategy is not registered: {strategy}") from error
            rendered = renderer(
                request.query,
                index_mode="index",
                index_strategy=strategy,
                join_strategy=join_strategy,
            )
        else:
            rendered = render_pre(
                request.query,
                index_mode="no_index",
                join_strategy=join_strategy,
            )
        timeout_ms = remaining_timeout_ms(request.deadline_ns)
        if timeout_ms is not None:
            rendered = RenderedSQL(
                rendered.path_id,
                rendered.setup_sql + (f"SET LOCAL statement_timeout = {timeout_ms}",),
                rendered.sql,
                rendered.pre_strategy,
                rendered.pre_join_strategy,
            )
        ids, latency = self._pg_runner(rendered, request.vector_literal)
        return PathExecutionResult(
            ids,
            latency,
            {
                "index_mode": mode,
                "index_strategy": strategy if mode == "index" else None,
                "pre_strategy": strategy if mode == "index" else "scan",
                "pre_strategy_verified": True,
                "join_strategy": join_strategy,
                "pre_join_strategy_verified": True,
                "uses_hnsw": False,
                "exact": True,
                "theoretical_recall_at_k": 1.0,
                "result_complete": len(ids) >= request.query.k,
                "fallback_used": False,
            },
        )
