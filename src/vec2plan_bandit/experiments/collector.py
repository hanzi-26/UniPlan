from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vec2plan_bandit.paths.registry import (
    ACORN, ACTIONS, FAISS, POST, PRE, PRE_ACCESS_STRATEGIES,
    PRE_JOIN_STRATEGIES, ExecutionConfig, default_execution_config,
    pre_configuration_key,
)
from vec2plan_bandit.paths.sql import (
    RenderedSQL, pre_query_has_join, render, render_candidate_filter,
)
from vec2plan_bandit.execution.acorn_path import AcornPathExecutor, AcornPayloadSchema, LazyQdrantClient, QdrantPayloadFilterCompiler
from vec2plan_bandit.execution.metadata import LocalMetadata
from vec2plan_bandit.execution.faiss_path import FaissAsset, FaissPathExecutor
from vec2plan_bandit.execution.article_identity import ArticleIDMapping
from vec2plan_bandit.execution.path_base import PathExecutionRequest
from vec2plan_bandit.execution.path_router import PathRouter
from vec2plan_bandit.execution.post_path import PostPathExecutor
from vec2plan_bandit.execution.pre_path import PrePathExecutor
from vec2plan_bandit.execution.plan_semantics import (
    validate_pre_access_strategy, validate_pre_join_strategy, validate_pre_plan,
)
from vec2plan_bandit.queries.loader import QueryRecord, load_source_queries
from vec2plan_bandit.services.embedding import EmbeddingServiceClient
from vec2plan_bandit.services.database_layout import connection_parameters, adapt_sql, migrated

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT_ROOT / "queries.json"
CONFIG = PROJECT_ROOT / "configs" / "experiment.json"


def _elapsed(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _psycopg_sql(sql: str) -> str:
    """Escape literal percent signs without altering named placeholders.

    Psycopg parses every ``%`` whenever parameters are supplied, including the
    wildcard characters in SQL LIKE literals. ``%%`` is its literal-percent
    escape; generated artifacts remain ordinary standalone SQL because this
    adapter is applied only at the execution boundary.
    """
    return re.sub(r"%(?!\([A-Za-z_][A-Za-z0-9_]*\)[sbt])", "%%", sql)


def _plan_nodes(payload: Any) -> list[dict[str, Any]]:
    root = payload[0]["Plan"] if isinstance(payload, list) else payload["Plan"]
    nodes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node.get("Plans", []):
            visit(child)

    visit(root)
    return nodes


@dataclass
class CompiledQuery:
    pg: dict[str, RenderedSQL]
    faiss_filter_sql: str
    translation_ms: float


class MatrixCollector:
    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        *,
        read_only: bool = False,
        ids_only: bool = True,
        db_config_path: Path | None = None,
    ):
        self.config = config
        self.run_dir = run_dir
        configured_db = self.config.get("database", {}).get("config_path")
        db_path = db_config_path or (
            Path(configured_db).expanduser() if configured_db else PROJECT_ROOT / "configs" / "database.yaml"
        )
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        self._db_path = db_path
        self._connection: Any | None = None
        self.read_only = bool(read_only)
        self.ids_only = bool(ids_only)
        embedding = self.config.get("embedding", {})
        self.embedding_client = EmbeddingServiceClient(
            url=str(embedding.get("url", "http://127.0.0.1:8333/embed")),
            model=str(embedding.get("model", "all-MiniLM-L6-v2")),
            dimension=int(embedding.get("dimension", 384)),
            normalize=bool(embedding.get("normalized", True)),
            timeout_seconds=float(embedding.get("timeout_seconds", 60)),
        )
        faiss_assets: dict[str, FaissAsset] = {}
        for column, asset in self.config.get("faiss_assets", {}).items():
            index_path = Path(asset["index"]).expanduser()
            ids_path = Path(asset["article_ids"]).expanduser()
            if not index_path.is_absolute():
                index_path = PROJECT_ROOT / index_path
            if not ids_path.is_absolute():
                ids_path = PROJECT_ROOT / ids_path
            faiss_assets[column] = FaissAsset(index_path, ids_path)
        acorn_config = self.config.get("acorn", {})
        ordinal_ids = acorn_config.get('point_ids_are_article_ids')
        strict_identity = config.get('runtime',{}).get('unified_resources',False)
        if acorn_config.get('collections') and (ordinal_ids is not None or strict_identity):
            if type(ordinal_ids) is not bool:
                raise ValueError('declare acorn.point_ids_are_article_ids explicitly; identity cannot be guessed')
        def identity_path(value):
            path = Path(value).expanduser()
            return path if path.is_absolute() else PROJECT_ROOT/path
        self.article_id_mappings = {}
        for column,asset in faiss_assets.items():
            source = config['faiss_assets'][column]
            require_proof = strict_identity and ordinal_ids is False
            receipt = source.get('id_mapping_receipt')
            if receipt is None and (asset.article_ids_path.parent/'id-mapping.json').is_file():
                receipt = str(asset.article_ids_path.parent/'id-mapping.json')
            acorn_receipt = acorn_config.get('identity_receipts',{}).get(column)
            self.article_id_mappings[column] = ArticleIDMapping(asset.article_ids_path,
                receipt=identity_path(receipt) if receipt else None,
                faiss_manifest=asset.index_path.parent/'manifest.json' if receipt else None,
                acorn_receipt=identity_path(acorn_receipt) if acorn_receipt else None,
                collection=acorn_config.get('collections',{}).get(column),require_proof=require_proof)
            if require_proof and (not receipt or not acorn_receipt):
                raise ValueError(f'{column}: supply FAISS mapping receipt and acorn.identity_receipts before evaluation')
        if ordinal_ids is False and set(acorn_config.get('collections',{}))-set(faiss_assets):
            raise ValueError('ordinal ACORN collections require the matching shared FAISS article-ID map')
        metadata_path = Path(config.get("metadata", {}).get("path", "data/metadata/metadata.sqlite"))
        if not metadata_path.is_absolute():
            metadata_path = PROJECT_ROOT / metadata_path
        extra_path = config.get("metadata", {}).get("extra_path")
        if extra_path is not None:
            extra_path = Path(extra_path)
            if not extra_path.is_absolute():
                extra_path = PROJECT_ROOT / extra_path
        self.local_metadata = LocalMetadata(metadata_path, extra_path=extra_path)
        if config.get('runtime', {}).get('unified_resources', False):
            import faiss
            faiss.omp_set_num_threads(config['runtime']['cpu_threads'])
        self._faiss_executor = FaissPathExecutor(
                faiss_assets,
                self.local_metadata,
                dimension=int(embedding.get("dimension", 384)),
                id_mappings=self.article_id_mappings,
            )
        self._qdrant_client = LazyQdrantClient(
            acorn_config.get("url", "http://127.0.0.1:6333"),
            prefer_grpc=bool(acorn_config.get("prefer_grpc", False)),
            grpc_port=int(acorn_config.get("grpc_port", 6334)),
        )
        self.path_router = PathRouter((
            PrePathExecutor(self._pg, self._pg_available),
            PostPathExecutor(self._pg, self._pg_available),
            self._faiss_executor,
            AcornPathExecutor(
                self._qdrant_client,
                acorn_config.get("collections", {}),
                filter_compiler=QdrantPayloadFilterCompiler(metadata=self.local_metadata,
                    schema=AcornPayloadSchema(**acorn_config.get('payload_schema', {})),
                    id_mappings=self.article_id_mappings if ordinal_ids is False else None),
                id_mappings=self.article_id_mappings if ordinal_ids is False else None,
            ),
        ))

    def close(self) -> None:
        self.local_metadata.close()
        self._qdrant_client.close()
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @property
    def connection(self) -> Any:
        """Open PostgreSQL only when PRE, POST, EXPLAIN, or PG filtering needs it."""
        if self._connection is not None:
            return self._connection
        if not self._db_path.is_file():
            raise FileNotFoundError(
                f"database config not found: {self._db_path}; set database.config_path"
            )
        import psycopg

        db_config = yaml.safe_load(self._db_path.read_text())
        pg = connection_parameters(db_config["database_connections"]["postgresql"])
        database = pg.pop("database", pg.pop("dbname", "postgres"))
        database = self.config.get('database', {}).get('name', database)
        pg.setdefault("host", "127.0.0.1")
        pg.setdefault("port", 5432)
        read_only_option = " -c default_transaction_read_only=on" if self.read_only else ""
        self._connection = psycopg.connect(
            dbname=database,
            connect_timeout=10,
            options=(
                "-c timezone=UTC -c search_path=arxiv_exp,arxiv_raw,public -c statement_timeout="
                + str(int(self.config["measurement"]["statement_timeout_ms"]))
                + read_only_option
            ),
            **pg,
        )
        return self._connection

    def _pg_available(self, probe: bool = False) -> bool:
        if not self._db_path.is_file():
            return False
        if not probe:
            return True
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
                if self.config.get("require_paths"):
                    schema = "public" if migrated(self.config) else "arxiv_exp"
                    raw_schema = "public" if migrated(self.config) else "arxiv_raw"
                    cursor.execute("SELECT indexname,indexdef FROM pg_indexes WHERE schemaname=%s AND tablename='articles'", (schema,))
                    indexes = dict(cursor.fetchall())
                    columns = set(self.config.get('database', {}).get('required_embedding_columns',
                        ['abstract_embedding', 'title_embedding']))
                    if not columns or columns - {'abstract_embedding', 'title_embedding'}:
                        raise ValueError('Invalid required embedding columns')
                    for column in columns:
                        declared = self.config.get('database', {}).get('post_index_names', {}).get(column)
                        definitions = [indexes.get(declared, '')] if declared else indexes.values()
                        if not any('using hnsw' in definition.lower() and
                                   f'({column} vector_cosine_ops)' in definition.lower().replace('"', '')
                                   for definition in definitions):
                            raise ValueError(f"missing or incompatible POST HNSW index for {column}")
                    cursor.execute("SELECT a.attname,format_type(a.atttypid,a.atttypmod) FROM pg_attribute a "
                                   "WHERE a.attrelid=%s::regclass AND a.attname IN ('abstract_embedding','title_embedding')", (schema + ".articles",))
                    types = dict(cursor.fetchall())
                    expected = f"vector({self.embedding_client.dimension})"
                    if not columns <= set(types) or any(types[column] != expected for column in columns):
                        raise ValueError("PostgreSQL vector dimensions disagree with embedding config")
                    cursor.execute("SHOW standard_conforming_strings")
                    if cursor.fetchone()[0] != "on":
                        raise ValueError("exact LIKE requires standard_conforming_strings=on")
                    cursor.execute("SELECT bool_and(c.collisdeterministic) FROM pg_attribute a "
                                   "JOIN pg_collation c ON c.oid=a.attcollation "
                                   "WHERE a.attrelid IN (%s::regclass,%s::regclass,%s::regclass) "
                                   "AND a.attname IN ('name','code')", tuple(raw_schema + "." + table for table in ("authors", "submitters", "categories")))
                    if cursor.fetchone()[0] is not True:
                        raise ValueError("local/Qdrant string filtering requires deterministic PostgreSQL collations")
            self.connection.rollback()
            return True
        except Exception:
            if self._connection is not None:
                try:
                    self._connection.rollback()
                except Exception:
                    pass
            return False

    def _begin(self, cursor: Any) -> None:
        if self.read_only:
            cursor.execute("BEGIN READ ONLY")
        if self.config.get('runtime', {}).get('unified_resources', False):
            cursor.execute('SET LOCAL max_parallel_workers_per_gather=0')
            cursor.execute('SET LOCAL jit=off')

    def compile(self, query: QueryRecord) -> CompiledQuery:
        started = time.perf_counter_ns()
        # These are precisely the normalized fields consumed by paths.sql.
        # Source wording, vector values and any measured outcomes are excluded.
        key = (query.embedding_column, query.k, self.ids_only,
               json.dumps(query.predicates, sort_keys=True, separators=(',', ':')))
        cache = self.__dict__.setdefault('_compiled_query_cache', OrderedDict())
        if key in cache:
            pg, filter_sql = cache[key]
            cache.move_to_end(key)
            return CompiledQuery(dict(pg), filter_sql, _elapsed(started))
        join_strategies = PRE_JOIN_STRATEGIES if pre_query_has_join(query) else ("all",)
        pg: dict[str, RenderedSQL] = {}
        for access_strategy in PRE_ACCESS_STRATEGIES:
            access_params = (
                {"index_mode": "no_index"}
                if access_strategy == "scan"
                else {"index_mode": "index", "index_strategy": access_strategy}
            )
            for join_strategy in join_strategies:
                params = {**access_params, "join_strategy": join_strategy}
                pg[f"{PRE}:{pre_configuration_key(params)}"] = render(
                    query, PRE, params=params, ids_only=self.ids_only,
                )
        pg[POST] = render(query, POST, ids_only=self.ids_only)
        filter_sql = render_candidate_filter(query)
        cache[key] = (dict(pg), filter_sql)
        if len(cache) > 128:
            cache.popitem(last=False)
        return CompiledQuery(pg, filter_sql, _elapsed(started))

    def available_pre_configurations(self, query: QueryRecord, vector_literal: str) -> frozenset[str]:
        """Discover realized PRE access/join pairs for this query inside E2E timing."""
        rendered_by_configuration = self.compile(query).pg
        available: set[str] = set()
        for key, rendered in rendered_by_configuration.items():
            if rendered.path_id != PRE:
                continue
            assert rendered.pre_strategy is not None
            assert rendered.pre_join_strategy is not None
            try:
                with self.connection.cursor() as cursor:
                    self._begin(cursor)
                    for statement in rendered.setup_sql:
                        cursor.execute(statement)
                    cursor.execute(
                        "EXPLAIN (FORMAT JSON) " + _psycopg_sql(adapt_sql(rendered.sql, self.config)),
                        {"query_embedding": vector_literal},
                    )
                    payload = cursor.fetchone()[0]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    names = [str(node["Index Name"]) for node in _plan_nodes(payload)
                             if node.get("Index Name")]
                    methods: dict[str, str] = {}
                    if names:
                        cursor.execute(
                            "SELECT c.relname, am.amname FROM pg_class c "
                            "JOIN pg_am am ON am.oid = c.relam WHERE c.relname = ANY(%s)",
                            (names,),
                        )
                        methods = dict(cursor.fetchall())
                    validate_pre_access_strategy(payload, rendered.pre_strategy, methods)
                    validate_pre_join_strategy(payload, rendered.pre_join_strategy)
                    available.add(
                        f"{rendered.pre_strategy}:{rendered.pre_join_strategy}"
                    )
            except RuntimeError as error:
                if not str(error).startswith(
                    ("PRE_STRATEGY_NOT_REALIZED", "PRE_JOIN_STRATEGY_NOT_REALIZED")
                ):
                    raise
            finally:
                self._finish_transaction()
        if "scan:all" not in available:
            raise RuntimeError("required exact PRE scan:all configuration is unavailable")
        return frozenset(available)

    def preload_faiss(self, column: str) -> float:
        """Load one index outside per-query timing and return preload milliseconds."""
        started = time.perf_counter_ns()
        self._faiss_executor.preload(column)
        return _elapsed(started)

    def release_faiss(self, column: str) -> None:
        self._faiss_executor.release(column)

    def embed(self, text: str) -> tuple[np.ndarray, float]:
        return self.embedding_client.embed(text)

    def estimate_statistics(self, query: QueryRecord) -> dict[str, float] | None:
        if not self.config.get("query_features", {}).get("local_statistics", False):
            return None
        return self.local_metadata.estimate(query)

    def preload_statistics(self) -> float:
        """Prepare label-free corpus assets; per-query estimation remains timed."""
        started = time.perf_counter_ns()
        if self.config.get("query_features", {}).get("local_statistics", False):
            self.local_metadata.prepare_statistics()
        guard=self.config.get('main_recall_guard',{})
        if guard.get('enabled',False) and (guard.get('cpu_exact_scan',False) or guard.get('cpu_exact_fallback',False)) and not hasattr(self,'exact_metadata'):
            from vec2plan_bandit.execution.exact_metadata import ExactMetadata
            self.exact_metadata=ExactMetadata(self.local_metadata,self.article_id_mappings['abstract_embedding'])
        return _elapsed(started)

    def _pg(self, rendered: RenderedSQL, vector_literal: str) -> tuple[list[int], float]:
        started = time.perf_counter_ns()
        connection = self.connection
        try:
            with connection.cursor() as cursor:
                self._begin(cursor)
                from vec2plan_bandit.execution.timing import evaluation_deadline_ns
                from vec2plan_bandit.execution.path_base import remaining_timeout_ms
                budget = remaining_timeout_ms(evaluation_deadline_ns())
                if budget is not None:
                    cursor.execute(f"SET LOCAL statement_timeout = {budget}")
                for statement in rendered.setup_sql:
                    cursor.execute(statement)
                if rendered.pre_strategy is not None:
                    cursor.execute(
                        "EXPLAIN (FORMAT JSON) " + _psycopg_sql(adapt_sql(rendered.sql, self.config)),
                        {"query_embedding": vector_literal},
                    )
                    planned = cursor.fetchone()[0]
                    if isinstance(planned, str):
                        planned = json.loads(planned)
                    index_names = [str(node["Index Name"]) for node in _plan_nodes(planned)
                                   if node.get("Index Name")]
                    index_methods: dict[str, str] = {}
                    if index_names:
                        cursor.execute(
                            "SELECT c.relname, am.amname FROM pg_class c "
                            "JOIN pg_am am ON am.oid = c.relam WHERE c.relname = ANY(%s)",
                            (index_names,),
                        )
                        index_methods = dict(cursor.fetchall())
                    validate_pre_access_strategy(planned, rendered.pre_strategy, index_methods)
                    if rendered.pre_join_strategy is not None:
                        validate_pre_join_strategy(planned, rendered.pre_join_strategy)
                cursor.execute(
                    _psycopg_sql(adapt_sql(rendered.sql, self.config)),
                    {"query_embedding": vector_literal},
                )
                ids = [int(row[0]) for row in cursor.fetchall()]
            return ids, _elapsed(started)
        except BaseException:
            # An alarm can interrupt libpq before it drains the command. A
            # ROLLBACK at that point both fails and hides the original timeout.
            # Closing the connection rolls back without issuing a new command.
            self.finish_cancelled_request()
            raise
        finally:
            if self._connection is connection and not connection.closed:
                try:
                    connection.rollback()
                except BaseException:
                    self.finish_cancelled_request()
                    raise

    def cancel_pending_request(self):
        """Called by the evaluation watchdog, never opens a new connection."""
        connection = self._connection
        if connection is not None and not connection.closed:
            try:
                connection.cancel_safe(timeout=1)
            except Exception:
                # The request may have finished or disconnected concurrently.
                pass

    def finish_cancelled_request(self):
        # A signal can interrupt libpq before it drains the cancelled command.
        # Reconnect on the next request instead of reusing a busy connection.
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                # Discard even when close fails; never replace the original
                # request error with a secondary connection-cleanup exception.
                pass

    def _finish_transaction(self):
        connection = self._connection
        if connection is None or connection.closed:
            return
        if sys.exc_info()[0] is not None:
            self.finish_cancelled_request()
            return
        try:
            connection.rollback()
        except BaseException:
            self.finish_cancelled_request()
            raise

    def explain(self, query: QueryRecord, compiled: CompiledQuery, vector_literal: str) -> dict[str, Any]:
        reports: dict[str, Any] = {}
        expected_index = (
            "idx_arxiv_articles_abstract_hnsw_cosine"
            if query.embedding_column == "abstract_embedding"
            else "idx_arxiv_articles_title_hnsw_cosine"
        )
        if migrated(self.config):
            expected_index = expected_index.replace("idx_arxiv_", "idx_public_")
        expected_index = self.config.get('database', {}).get('post_index_names', {}).get(
            query.embedding_column, expected_index)
        for plan_id, rendered in compiled.pg.items():
            path_id = rendered.path_id
            try:
                with self.connection.cursor() as cursor:
                    self._begin(cursor)
                    for statement in rendered.setup_sql:
                        cursor.execute(statement)
                    cursor.execute(
                        _psycopg_sql("EXPLAIN (FORMAT JSON, COSTS TRUE) " + adapt_sql(rendered.sql, self.config)),
                        {"query_embedding": vector_literal},
                    )
                    payload = cursor.fetchone()[0]
            finally:
                self.connection.rollback()
            nodes = _plan_nodes(payload)
            indexes = [str(node.get("Index Name")) for node in nodes if node.get("Index Name")]
            uses_hnsw = any("hnsw" in name.lower() for name in indexes)
            plan_semantics = None
            if path_id == PRE:
                plan_semantics = validate_pre_plan(
                    payload,
                    has_predicates=bool(query.predicates),
                    embedding_column=query.embedding_column,
                    k=query.k,
                )
                passed = plan_semantics["plan_valid"]
                reason = plan_semantics["failure_reason"]
                if passed and rendered.pre_strategy is not None:
                    with self.connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT c.relname, am.amname FROM pg_class c "
                            "JOIN pg_am am ON am.oid = c.relam WHERE c.relname = ANY(%s)",
                            (indexes,),
                        )
                        methods = dict(cursor.fetchall())
                    self.connection.rollback()
                    try:
                        validate_pre_access_strategy(payload, rendered.pre_strategy, methods)
                        if rendered.pre_join_strategy is not None:
                            validate_pre_join_strategy(payload, rendered.pre_join_strategy)
                    except RuntimeError as error:
                        passed = False
                        reason = str(error)
            elif path_id == POST:
                passed = expected_index in indexes
                reason = None if passed else f"expected {expected_index}, got {indexes}"
            else:
                passed = not uses_hnsw
                reason = None if passed else f"exact path unexpectedly used HNSW: {indexes}"
            normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            reports[plan_id] = {
                "passed": passed,
                "reason": reason,
                "uses_hnsw": uses_hnsw,
                "index_names": indexes,
                "fingerprint": hashlib.sha256(normalized.encode()).hexdigest(),
                "plan_semantics": plan_semantics,
                "plan": payload,
            }
            folder = (
                f"pre/{rendered.pre_strategy}/{rendered.pre_join_strategy}"
                if path_id == PRE else "post"
            )
            _atomic_json(
                self.run_dir / "explain" / folder / f"{query.query_id}.json",
                reports[plan_id],
            )
        return reports

    def execute(
        self,
        action_id: int,
        query: QueryRecord,
        compiled: CompiledQuery,
        vector: np.ndarray,
        vector_literal: str,
        execution_config: ExecutionConfig | None = None,
        *,
        deadline_ns: int | None = None,
    ) -> tuple[list[int], float, dict[str, Any]]:
        """Execute one of the four v3 paths through the modular router."""

        del compiled  # Each v3 executor owns its path-specific compilation.
        config = execution_config or default_execution_config(action_id)
        if config.action_id != action_id or config.path_id != ACTIONS[action_id].path_id:
            raise ValueError("execution configuration does not match selected path")
        result = self.path_router.execute(
            config.path_id,
            PathExecutionRequest(
                query=query,
                vector=vector,
                vector_literal=vector_literal,
                deadline_ns=deadline_ns,
            ),
            config.params,
        )
        return result.article_ids, result.latency_ms, {
            **result.details,
            "config_id": config.config_id,
            "resolved_params": dict(config.params),
        }

def _recall(result: list[int], ground_truth: list[int], k: int) -> float:
    denominator = min(k, len(ground_truth))
    if denominator == 0:
        return 1.0 if not result else 0.0
    return len(set(result[:k]).intersection(ground_truth[:k])) / denominator


def collect(
    run_dir: Path,
    query_ids: set[str] | None = None,
    latency_runs_override: int | None = None,
    ground_truth_by_query: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text())
    latency_runs = (
        int(latency_runs_override)
        if latency_runs_override is not None
        else int(config["measurement"]["latency_runs"])
    )
    if latency_runs <= 0:
        raise ValueError("latency_runs must be positive")
    records = load_source_queries(SOURCE)
    if query_ids is not None:
        records = [record for record in records if record.query_id in query_ids]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "query_ids": [record.query_id for record in records],
        "warmup_runs": config["measurement"]["warmup_runs"],
        "latency_runs": latency_runs,
        "configured_latency_runs": config["measurement"]["latency_runs"],
        "recall_runs": 1,
        "shared_translation_and_embedding": True,
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    collector = MatrixCollector(run_dir, config)
    rng = random.Random(int(config["seed"]))
    summary_rows: list[dict[str, Any]] = []
    try:
        for position, query in enumerate(records, 1):
            output_path = run_dir / "queries" / f"{query.query_id}.json"
            if output_path.exists():
                existing = json.loads(output_path.read_text())
                summary_rows.extend(existing["summary"])
                continue
            compiled = collector.compile(query)
            vector, embedding_ms = collector.embed(query.semantic_text)
            vector_literal = _vector_literal(vector)
            plans = collector.explain(query, compiled, vector_literal)
            required_plan_ids = {f"{PRE}:scan:all", POST}
            failed_plans = {
                key: value["reason"] for key, value in plans.items()
                if key in required_plan_ids and not value["passed"]
            }
            if failed_plans:
                raise RuntimeError(f"{query.query_id} EXPLAIN gate failed: {failed_plans}")
            action_ids = list(range(4))
            warmup_order = action_ids.copy()
            rng.shuffle(warmup_order)
            warmup_failures: dict[int, dict[str, Any]] = {}
            for action_id in warmup_order:
                started = time.perf_counter_ns()
                try:
                    _, path_ms, details = collector.execute(
                        action_id, query, compiled, vector, vector_literal
                    )
                    if not bool(details.get("result_complete")):
                        warmup_failures[action_id] = {
                            "error_type": "IncompleteWarmup",
                            "error_message": "adaptive search did not return k valid rows within its configured limits",
                            "path_latency_ms": path_ms,
                            "latency_censored": bool(details.get("timed_out", False)),
                            **details,
                        }
                except Exception as exc:
                    warmup_failures[action_id] = {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "path_latency_ms": _elapsed(started),
                        "latency_censored": True,
                        "result_complete": False,
                        "fallback_used": False,
                    }
            runs: list[dict[str, Any]] = []
            first_results: dict[int, list[int]] = {}
            first_success: dict[int, bool] = {}
            for repeat in range(1, latency_runs + 1):
                order = action_ids.copy()
                rng.shuffle(order)
                for sequence, action_id in enumerate(order, 1):
                    warmup_failure = warmup_failures.get(action_id)
                    if warmup_failure is not None:
                        ids = []
                        path_ms = float(warmup_failure["path_latency_ms"])
                        details = {
                            **warmup_failure,
                            "success": False,
                            "measurement_skipped_after_warmup_failure": True,
                        }
                    else:
                        started = time.perf_counter_ns()
                        try:
                            ids, path_ms, details = collector.execute(
                                action_id, query, compiled, vector, vector_literal
                            )
                            complete = bool(details.get("result_complete"))
                            details = {
                                **details,
                                "success": complete,
                                "error_type": None if complete else "IncompleteResult",
                                "error_message": None if complete else "did not return k valid rows",
                                "latency_censored": bool(details.get("timed_out", False)),
                                "measurement_skipped_after_warmup_failure": False,
                            }
                        except Exception as exc:
                            ids = []
                            path_ms = _elapsed(started)
                            details = {
                                "success": False,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "latency_censored": True,
                                "measurement_skipped_after_warmup_failure": False,
                                "result_complete": False,
                                "fallback_used": False,
                            }
                    first_results.setdefault(action_id, ids)
                    first_success.setdefault(action_id, bool(details["success"]))
                    runs.append({
                        "repeat": repeat,
                        "execution_sequence": sequence,
                        "action_id": action_id,
                        "path_id": ACTIONS[action_id].path_id,
                        "returned_ids": ids,
                        "rows_returned": len(ids),
                        "path_latency_ms": path_ms,
                        "sql_translation_latency_ms": compiled.translation_ms,
                        "embedding_latency_ms": embedding_ms,
                        "e2e_latency_ms": compiled.translation_ms + embedding_ms + path_ms,
                        **details,
                    })
            # Exact ground truth is evaluation-only in v3. No selectable path,
            # including ACORN, may be treated as truth.
            ground_truth = list((ground_truth_by_query or {}).get(query.query_id, []))
            ground_truth_valid = query.query_id in (ground_truth_by_query or {})
            query_summary: list[dict[str, Any]] = []
            for action_id in action_ids:
                action_runs = [row for row in runs if row["action_id"] == action_id]
                e2e = [float(row["e2e_latency_ms"]) for row in action_runs]
                row = {
                    "query_id": query.query_id,
                    "action_id": action_id,
                    "path_id": ACTIONS[action_id].path_id,
                    "k": query.k,
                    "embedding_column": query.embedding_column,
                    "recall_at_k_once": (
                        _recall(first_results.get(action_id, []), ground_truth, query.k)
                        if ground_truth_valid and first_success.get(action_id, False)
                        else None
                    ),
                    "p50_e2e_latency_ms": statistics.median(e2e),
                    "mean_e2e_latency_ms": statistics.fmean(e2e),
                    "min_e2e_latency_ms": min(e2e),
                    "max_e2e_latency_ms": max(e2e),
                    "complete_runs": sum(bool(item["result_complete"]) for item in action_runs),
                    "failed_runs": sum(not bool(item["success"]) for item in action_runs),
                    "measured_runs": len(action_runs),
                }
                query_summary.append(row)
                summary_rows.append(row)
            payload = {
                "query": query.to_dict(),
                "shared_sql_translation_latency_ms": compiled.translation_ms,
                "shared_embedding_latency_ms": embedding_ms,
                "plans": {key: {k: v for k, v in value.items() if k != "plan"} for key, value in plans.items()},
                "ground_truth_action_id": None,
                "ground_truth_source": "EXTERNAL_EXACT" if ground_truth_valid else None,
                "ground_truth_ids": ground_truth,
                "runs": runs,
                "summary": query_summary,
            }
            _atomic_json(output_path, payload)
            print(f"[{position}/{len(records)}] {query.query_id} complete", flush=True)
        columns = list(dict.fromkeys(key for row in summary_rows for key in row))
        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary_rows)
        _atomic_json(run_dir / "summary.json", summary_rows)
        manifest.update(status="COMPLETE", completed_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(run_dir / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="FAILED", failed_at=datetime.now(timezone.utc).isoformat(), error=f"{type(exc).__name__}: {exc}")
        _atomic_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        collector.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect the fresh four-path matrix")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--latency-runs", type=int)
    args = parser.parse_args(argv)
    run_dir = args.run_dir or PROJECT_ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = collect(
        run_dir,
        set(args.queries) if args.queries else None,
        latency_runs_override=args.latency_runs,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
