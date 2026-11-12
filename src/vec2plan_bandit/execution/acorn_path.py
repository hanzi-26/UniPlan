from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

from vec2plan_bandit.execution.path_base import (
    PathExecutionRequest,
    PathExecutionResult,
    PathUnavailableError,
    remaining_timeout_ms,
)
from vec2plan_bandit.paths.registry import ACORN, REGISTRY
from vec2plan_bandit.queries.loader import QueryRecord
from vec2plan_bandit.execution.article_identity import ArticleIDMapping, materialize_article_ids, integer_id
from vec2plan_bandit.execution.metadata import VocabularyExpansionTooLarge


@dataclass(frozen=True)
class AcornPayloadSchema:
    submitter: str = "submitter_name"
    authors: str = "author_names"
    categories: str = "category_codes"
    update_date: str = "update_date"
    latest_update: str = "is_latest_update"
    license: str = "license"
    comments: str = "comments"
    journal_ref: str = "journal_ref"
    report_no: str = "report_no"


class AcornFilterCompiler(Protocol):
    def compile(self, query: QueryRecord, models: Any) -> Any | None: ...


class LazyQdrantClient:
    """Delay the optional import and socket setup until ACORN is executed."""

    def __init__(self, url: str, *, prefer_grpc: bool = False, grpc_port: int = 6334):
        self.url = url
        self.prefer_grpc = bool(prefer_grpc)
        self.grpc_port = int(grpc_port)
        self._client: Any | None = None
        self._readiness: bool | None = None

    def _connect(self) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise PathUnavailableError(
                "ACORN path requires qdrant-client and Qdrant >= 1.16"
            ) from error
        self._client = QdrantClient(
            url=self.url,
            grpc_port=self.grpc_port,
            prefer_grpc=self.prefer_grpc,
            timeout=600,
        )
        return self._client

    def query_points(self, **kwargs: Any) -> Any:
        if self._client is None:
            self._connect()
        return self._client.query_points(**kwargs)

    def available(self, *, probe: bool = False) -> bool:
        if not self.url:
            return False
        if not probe:
            return True
        if self._readiness is not None:
            return self._readiness
        try:
            if self._client is None:
                self._connect()
            self._client.get_collections()
            self._readiness = True
        except Exception:
            self._readiness = False
        return self._readiness

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                close()
            self._client = None


class QdrantPayloadFilterCompiler:
    """Translate QueryRecord predicates to the frozen Qdrant payload schema."""

    def __init__(self, schema: AcornPayloadSchema | None = None, *, metadata: Any | None = None,
                 id_mappings: Mapping[str, ArticleIDMapping] | None = None):
        self.schema = schema or AcornPayloadSchema()
        self.metadata = metadata
        self.id_mappings = id_mappings

    def compile(self, query: QueryRecord, models: Any) -> Any | None:
        must: list[Any] = []
        for predicate in query.predicates:
            kind = predicate["type"]
            values = list(predicate["values"])
            if kind == "submitter_eq":
                must.append(self._value(models, self.schema.submitter, values[0]))
            elif kind == "submitter_in":
                must.append(self._any(models, self.schema.submitter, values))
            elif kind == "author_eq":
                must.append(self._value(models, self.schema.authors, values[0]))
            elif kind == "author_in":
                must.append(self._any(models, self.schema.authors, values))
            elif kind == "author_like":
                if self.metadata is None:
                    raise ValueError("exact SQL LIKE requires the shared local metadata snapshot")
                names = self.metadata.author_values_like(str(values[0]))
                must.append(self._any(models, self.schema.authors, names)
                            if names else models.HasIdCondition(has_id=[]))
            elif kind == "category_eq":
                must.append(self._value(models, self.schema.categories, values[0]))
            elif kind == "category_in":
                must.append(self._any(models, self.schema.categories, values))
            elif kind == "update_date_range":
                must.append(
                    models.FieldCondition(
                        key=self.schema.update_date,
                        range=models.DatetimeRange(gte=values[0], lt=values[1]),
                    )
                )
            elif kind in {"update_date_eq", "update_date_le", "update_date_ge", "update_date_between", "update_date_outside"}:
                start = str(values[0])
                end = str(values[1]) if len(values) > 1 else start
                after_start = (date.fromisoformat(start) + timedelta(days=1)).isoformat()
                after_end = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
                def date_condition(**bounds: str) -> Any:
                    return models.FieldCondition(key=self.schema.update_date,
                        range=models.DatetimeRange(**bounds))
                if kind == "update_date_eq":
                    must.append(date_condition(gte=start, lt=after_start))
                elif kind == "update_date_le":
                    must.append(date_condition(lt=after_start))
                elif kind == "update_date_ge":
                    must.append(date_condition(gte=start))
                elif kind == "update_date_between":
                    must.append(date_condition(gte=start, lt=after_end))
                else:
                    must.append(models.Filter(should=[date_condition(lt=after_start), date_condition(gte=end)]))
            elif kind in {"license_eq", "comments_eq", "journal_ref_eq", "report_no_eq"}:
                field = kind.rsplit("_", 1)[0]
                must.append(self._value(models, getattr(self.schema, field), values[0]))
            elif kind == "license_in":
                must.append(self._any(models, self.schema.license, values))
            elif kind in {"comments_like", "journal_ref_like"}:
                if self.metadata is None:
                    raise ValueError("exact SQL LIKE requires the shared local metadata snapshot")
                field = kind.rsplit("_", 1)[0]
                try:
                    names = self.metadata.field_values_like(field, str(values[0]))
                except VocabularyExpansionTooLarge:
                    ids = self.metadata.field_ids_like(field,str(values[0]))
                    if self.id_mappings is not None:
                        mapping = self.id_mappings.get(query.embedding_column)
                        if mapping is None:
                            raise ValueError('exact LIKE ID filter requires the ACORN snapshot mapping')
                        ids = mapping.encode(ids)
                    else:
                        ids = list(ids)
                    must.append(models.HasIdCondition(has_id=ids))
                else:
                    must.append(self._any(models, getattr(self.schema, field), names)
                                if names else models.HasIdCondition(has_id=[]))
            elif kind == "latest_update":
                must.append(self._value(models, self.schema.latest_update, True))
            else:
                raise ValueError(f"unsupported ACORN predicate: {kind}")
        return models.Filter(must=must) if must else None

    @staticmethod
    def _value(models: Any, key: str, value: Any) -> Any:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    @staticmethod
    def _any(models: Any, key: str, values: list[Any]) -> Any:
        return models.FieldCondition(key=key, match=models.MatchAny(any=values))


class AcornPathExecutor:
    """Qdrant ACORN-1 executor with lazy optional dependencies."""

    path_id = ACORN

    def __init__(
        self,
        client: Any,
        collections: Mapping[str, str],
        *,
        filter_compiler: AcornFilterCompiler | None = None,
        models_module: Any | None = None,
        article_id_maps: Mapping[str, Path] | None = None,
        id_mappings: Mapping[str, ArticleIDMapping] | None = None,
    ):
        self._client = client
        self._collections = dict(collections)
        self._filter_compiler = filter_compiler or QdrantPayloadFilterCompiler()
        self._models = models_module
        if article_id_maps is not None and id_mappings is not None:
            raise ValueError('supply either shared ID mappings or legacy mapping paths')
        self._article_id_maps = dict(article_id_maps) if article_id_maps is not None else None
        self._id_mappings = (dict(id_mappings) if id_mappings is not None else
            {column:ArticleIDMapping(path) for column,path in (article_id_maps or {}).items()})
        self._ordinal_ids = id_mappings is not None or article_id_maps is not None

    def _id_map(self, column):
        if not self._ordinal_ids:
            return None
        if column not in self._id_mappings:
            raise PathUnavailableError(f'ACORN article-ID mapping missing for {column}')
        return self._id_mappings[column].values

    def available(self, *, probe: bool = False, query: QueryRecord | None = None) -> bool:
        if query is not None and query.embedding_column not in self._collections:
            return False
        if not self._collections:
            return False
        if self._ordinal_ids:
            columns = [query.embedding_column] if query is not None else list(self._collections)
            try:
                for column in columns:
                    if column not in self._id_mappings or not self._id_mappings[column].path.is_file():
                        return False
                    if probe:
                        self._id_map(column)
            except (OSError, ValueError):
                return False
        checker = getattr(self._client, "available", None)
        return bool(checker(probe=probe)) if checker is not None else True

    def _models_module(self) -> Any:
        if self._models is None:
            try:
                from qdrant_client import models
            except ImportError as error:
                raise PathUnavailableError(
                    "ACORN path requires qdrant-client and Qdrant >= 1.16"
                ) from error
            self._models = models
        return self._models

    def execute(
        self,
        request: PathExecutionRequest,
        params: Mapping[str, Any],
    ) -> PathExecutionResult:
        started = time.perf_counter_ns()
        remaining_timeout_ms(request.deadline_ns)
        values = REGISTRY.normalize(ACORN, params)
        models = self._models_module()
        try:
            collection_name = self._collections[request.query.embedding_column]
        except KeyError as error:
            raise PathUnavailableError(
                f"no ACORN collection configured for {request.query.embedding_column}"
            ) from error
        filter_start=time.perf_counter_ns()
        query_filter = self._filter_compiler.compile(request.query, models)
        filter_ms=(time.perf_counter_ns()-filter_start)/1e6
        search_params = models.SearchParams(
            hnsw_ef=int(values["hnsw_ef"]),
            exact=False,
            acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0),
        )
        # LIKE expansion can consume most of the request budget. Do not give
        # the remote search the stale budget from before filter preparation.
        timeout_ms = remaining_timeout_ms(request.deadline_ns)
        timeout = max(1, int(math.ceil(timeout_ms / 1000))) if timeout_ms is not None else None
        search_start=time.perf_counter_ns()
        response = self._client.query_points(
            collection_name=collection_name,
            query=request.vector.tolist(),
            query_filter=query_filter,
            search_params=search_params,
            limit=request.query.k,
            with_payload=False,
            with_vectors=False,
            **({"timeout": timeout} if timeout is not None else {}),
        )
        search_ms=(time.perf_counter_ns()-search_start)/1e6
        mapping_start=time.perf_counter_ns()
        points = getattr(response, "points", response)
        ids = [integer_id(point.id if hasattr(point, "id") else point["id"]) for point in points]
        mapping = self._id_map(request.query.embedding_column)
        if mapping is not None:
            ids = self._id_mappings[request.query.embedding_column].decode(ids)
        mapping_ms=(time.perf_counter_ns()-mapping_start)/1e6
        assembly_start=time.perf_counter_ns()
        ids = materialize_article_ids(ids, request.query.k)
        assembly_ms=(time.perf_counter_ns()-assembly_start)/1e6
        remaining_timeout_ms(request.deadline_ns)
        latency = (time.perf_counter_ns() - started) / 1_000_000
        return PathExecutionResult(
            ids,
            latency,
            {
                **values,
                "implementation": "QDRANT_ACORN1",
                "acorn_enabled": True,
                "max_selectivity": 1.0,
                "exact": False,
                "exact_fallback": False,
                'execution_stages_ms': {'filter_prepare_ms':filter_ms,'ann_search_ms':search_ms,
                    'id_mapping_ms':mapping_ms,'result_assembly_ms':assembly_ms},
                "point_id_mapping": 'ordinal_to_article_id' if mapping is not None else 'identity',
                "result_identity": "articles.id",
                "return_to_pg": False,
                "service_time_s": getattr(response, "time", None),
                "result_complete": len(ids) >= request.query.k,
                "fallback_used": False,
            },
        )
