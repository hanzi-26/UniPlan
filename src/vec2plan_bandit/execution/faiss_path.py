from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

import numpy as np

from vec2plan_bandit.execution.path_base import (
    PathExecutionRequest,
    PathExecutionResult,
    PathUnavailableError,
    remaining_timeout_ms,
)
from vec2plan_bandit.paths.registry import FAISS, REGISTRY
from vec2plan_bandit.queries.loader import QueryRecord
from vec2plan_bandit.execution.article_identity import ArticleIDMapping, materialize_article_ids


class EligibilityProvider(Protocol):
    def eligible_ids(
        self, query: QueryRecord, candidate_ids: list[int]
    ) -> set[int]: ...


@dataclass(frozen=True)
class FaissAsset:
    index_path: Path
    article_ids_path: Path


@dataclass(frozen=True)
class FaissCandidates:
    article_ids: list[int]
    scores: list[float]
    eligible_ids: set[int]
    candidate_count: int
    timings: dict = field(default_factory=dict)


class FaissPathExecutor:
    """FAISS backend isolated from routing and policy code."""

    path_id = FAISS

    def __init__(
        self,
        assets: Mapping[str, FaissAsset],
        eligibility_provider: EligibilityProvider,
        *,
        dimension: int = 384,
        faiss_module: Any | None = None,
        id_mappings: Mapping[str, ArticleIDMapping] | None = None,
    ):
        self._assets = dict(assets)
        self._eligibility_provider = eligibility_provider
        self._dimension = int(dimension)
        self._faiss = faiss_module
        self._cache: dict[str, tuple[Any, np.ndarray]] = {}
        self._id_mappings = dict(id_mappings or {})

    def available(self, *, probe: bool = False, query: QueryRecord | None = None) -> bool:
        columns = [query.embedding_column] if query is not None else list(self._assets)
        if not columns:
            return False
        for column in columns:
            asset = self._assets.get(column)
            if asset is None or not asset.index_path.is_file() or not asset.article_ids_path.is_file():
                return False
        if probe:
            try:
                for column in columns:
                    self._load(column)
            except (ImportError, OSError, ValueError, PathUnavailableError):
                return False
        if query is not None and query.predicates:
            checker = getattr(self._eligibility_provider, "available", None)
            if checker is not None and not bool(checker(query=query, probe=probe)):
                return False
        return True

    def preload(self, column: str) -> None:
        self._load(column)

    def release(self, column: str) -> None:
        self._cache.pop(column, None)

    def corpus_info(self, column: str) -> dict[str, int]:
        index, _ = self._load(column)
        return {"total_rows": int(index.ntotal), "embedding_dimension": int(index.d)}

    def search_candidates(self, request: PathExecutionRequest, *, candidate_count: int,
                          ef_search: int) -> FaissCandidates:
        """Shared ANN/filter primitive; iterative callers own their total budget.

        Counts are explicit, not limited by the single-action multiplier space.
        This does not expand the main policy's existing parameter space.
        """
        remaining_timeout_ms(request.deadline_ns)
        if candidate_count <= 0 or ef_search <= 0:
            raise ValueError("FAISS candidate count and ef_search must be positive")
        index, article_ids = self._load(request.query.embedding_column)
        count = min(int(index.ntotal), int(candidate_count))
        if count == 0:
            return FaissCandidates([], [], set(), 0)
        index.hnsw.efSearch = int(ef_search)
        search_start=time.perf_counter_ns()
        scores, positions = index.search(request.vector.reshape(1, -1), count)
        search_ms=(time.perf_counter_ns()-search_start)/1e6
        mapping_start=time.perf_counter_ns()
        remaining_timeout_ms(request.deadline_ns)
        valid = [(position, score) for position, score in zip(positions[0], scores[0]) if position != -1]
        ranked = self._id_mappings[request.query.embedding_column].decode([p for p,_ in valid])
        similarities = [float(score) for _,score in valid]
        mapping_ms=(time.perf_counter_ns()-mapping_start)/1e6
        filter_start=time.perf_counter_ns()
        eligible = (self._eligibility_provider.eligible_ids(request.query, ranked)
                    if request.query.predicates else set(ranked))
        remaining_timeout_ms(request.deadline_ns)
        return FaissCandidates(ranked, similarities, eligible, count, {'ann_search_ms':search_ms,'id_mapping_ms':mapping_ms,'candidate_filter_ms':(time.perf_counter_ns()-filter_start)/1e6})

    def _module(self) -> Any:
        if self._faiss is None:
            try:
                import faiss
            except ImportError as error:
                raise PathUnavailableError(
                    "FAISS path requires the optional faiss package"
                ) from error
            self._faiss = faiss
        return self._faiss

    def _load(self, column: str) -> tuple[Any, np.ndarray]:
        if column in self._cache:
            return self._cache[column]
        try:
            asset = self._assets[column]
        except KeyError as error:
            raise PathUnavailableError(f"no FAISS asset configured for {column}") from error
        if not asset.index_path.is_file() or not asset.article_ids_path.is_file():
            raise PathUnavailableError(
                f"FAISS index or article-ID mapping is missing for {column}: "
                f"{asset.index_path}, {asset.article_ids_path}"
            )
        module = self._module()
        index = module.read_index(str(asset.index_path))
        mapping = self._id_mappings.setdefault(column, ArticleIDMapping(asset.article_ids_path))
        article_ids = mapping.values
        if index.ntotal != len(article_ids) or index.d != self._dimension or not hasattr(index, "hnsw"):
            raise ValueError(f"invalid FAISS asset for {column}")
        if index.metric_type != module.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS path requires normalized vectors and inner product")
        self._cache[column] = (index, article_ids)
        return index, article_ids

    def execute(
        self,
        request: PathExecutionRequest,
        params: Mapping[str, Any],
    ) -> PathExecutionResult:
        started = time.perf_counter_ns()
        remaining_timeout_ms(request.deadline_ns)
        values = REGISTRY.normalize(FAISS, params)
        index, article_ids = self._load(request.query.embedding_column)
        candidate_count = min(
            int(index.ntotal),
            max(request.query.k, int(math.ceil(request.query.k * values["candidate_multiplier"]))),
        )
        batch = self.search_candidates(request, candidate_count=candidate_count, ef_search=values["ef_search"])
        assembly_start=time.perf_counter_ns()
        result = [article_id for article_id in batch.article_ids if article_id in batch.eligible_ids]
        result = materialize_article_ids(result, request.query.k)
        assembly_ms=(time.perf_counter_ns()-assembly_start)/1e6
        latency = (time.perf_counter_ns() - started) / 1_000_000
        return PathExecutionResult(
            result,
            latency,
            {
                **values,
                'execution_stages_ms': {**batch.timings,'result_assembly_ms':assembly_ms},
                "candidate_count": candidate_count,
                "uses_hnsw": True,
                "point_id_mapping": "ordinal_to_article_id",
                "result_identity": "articles.id",
                "return_to_pg": False,
                "result_complete": len(result) >= request.query.k,
                "fallback_used": False,
            },
        )
