from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np

from vec2plan_bandit.baselines import builtin_baselines
from vec2plan_bandit.baselines.maxucb_bo import jev_router_from_config
from vec2plan_bandit.evaluation.quality import evaluate_small_k_quality
from vec2plan_bandit.execution.adaptive_runtime import (
    run_adaptive_request,
    run_fixed_configuration_request,
    run_native_baseline_request,
    run_selector_baseline_request,
)
from vec2plan_bandit.experiments.collector import MatrixCollector, _vector_literal
from vec2plan_bandit.experiments.configuration import (
    load_json as load_configuration,
    validate_baseline_config,
    validate_experiment_config,
)
from vec2plan_bandit.experiments.manifest import RunManifest
from vec2plan_bandit.experiments.observations import build_training_observations
from vec2plan_bandit.experiments.preflight import run_preflight
from vec2plan_bandit.experiments.protocol import bind_run, save_bundle, load_bundle
from vec2plan_bandit.experiments.results import JsonlResultStore
from vec2plan_bandit.features.action_features import ACTION_FEATURE_VERSION
from vec2plan_bandit.features.feature_pipeline import FeaturePipeline, load_frozen_embeddings
from vec2plan_bandit.feedback.reward import REWARD_VERSION, compute_reward, reward_version
from vec2plan_bandit.paths.registry import make_execution_config, pre_configuration_key
from vec2plan_bandit.policies.max_ucb import MaxUCB, initial_design
from vec2plan_bandit.policies.online_learning import OnlineLearner
from vec2plan_bandit.queries.loader import QueryRecord, load_source_queries
from vec2plan_bandit.surrogates.catboost_ensemble import ConstrainedCatBoostSurrogate
from vec2plan_bandit.experiments.workflow import prepare, train, benchmark, refit


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(args.config)
    if args.db_config_path is not None:
        config.setdefault("database", {})["config_path"] = str(args.db_config_path)
    for column, index_path, ids_path in (
        ("abstract_embedding", args.faiss_abstract_index, args.faiss_abstract_ids),
        ("title_embedding", args.faiss_title_index, args.faiss_title_ids),
    ):
        if index_path is None and ids_path is None:
            continue
        asset = config.setdefault("faiss_assets", {}).setdefault(column, {})
        if index_path is not None:
            asset["index"] = str(index_path)
        if ids_path is not None:
            asset["article_ids"] = str(ids_path)
    return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _queries(path: Path) -> list[QueryRecord]:
    try:
        loaded = load_source_queries(path)
    except ValueError as exc:
        rows = _read_jsonl(path)
        if not rows:
            raise exc
        loaded = [
            QueryRecord(
                query_id=str(row["query_id"]),
                group_id=str(row["group_id"]),
                question=str(row["question"]),
                semantic_text=str(row["semantic_text"]),
                embedding_column=str(row["embedding_column"]),
                k=int(row["k"]),
                predicates=tuple(row.get("predicates", [])),
                source_sql=str(row["source_sql"]),
                cosine_sql=str(row.get("cosine_sql", row["source_sql"].replace("<->", "<=>"))),
                candidate_sql=tuple(row.get("candidate_sql", [])),
                source_record_number=int(row["source_record_number"]),
                source_hash=str(row["source_hash"]),
            )
            for row in rows
        ]
    if any(query.k not in range(1, 11) for query in loaded):
        raise ValueError("every query must use k in 1..10; queries are never rewritten")
    return loaded


def _ground_truth(path: Path) -> dict[str, list[int]]:
    result = {}
    for row in _read_jsonl(path):
        if "topk_ids" in row:
            query_id = str(row["query_id"])
            ids = [int(value) for value in row["topk_ids"]]
            if query_id in result or len(ids) != len(set(ids)):
                raise ValueError("ground truth contains duplicate queries or article IDs")
            result[query_id] = ids
    return result


def _adjudicate(row: dict[str, Any], truth: dict[str, list[int]], cap_ms: float, recall_threshold: float = 0.9) -> dict[str, Any]:
    quality = evaluate_small_k_quality(
        row.get("article_ids", []), truth.get(str(row["query_id"]), []), int(row["k"]),
        recall_threshold=recall_threshold,
    )
    execution_success = not bool(row.get("error_type"))
    deadline_pass = bool(not row.get("timed_out", False) and np.isfinite(row["e2e_latency_ms"])
                         and 0 <= float(row["e2e_latency_ms"]) <= cap_ms)
    primary = quality.corpus_has_k and quality.quality_pass is not None
    complete = len(set(row.get("article_ids", [])[:int(row["k"])])) == int(row["k"])
    fallback_used = bool(row.get("fallback_used", False) or row.get("execution_details", {}).get("fallback_used", False))
    reward = compute_reward(
        e2e_latency_ms=float(row["e2e_latency_ms"]),
        recall_at_k=quality.recall_at_k,
        latency_cap_ms=cap_ms,
        success=execution_success,
        timeout=not deadline_pass,
        result_complete=complete,
        fallback_used=fallback_used,
        quality_pass=bool(quality.quality_pass) if primary else False,
        recall_threshold=recall_threshold,
    )
    return {
        **row,
        "gt_size": quality.gt_size,
        "hits_at_k": quality.hits_at_k,
        "recall_at_k": quality.recall_at_k,
        "recall_pass": quality.quality_pass,
        "quality_pass": bool(quality.quality_pass and complete) if quality.quality_pass is not None else None,
        "corpus_has_k": quality.corpus_has_k,
        "deadline_pass": deadline_pass,
        "result_full_k": complete,
        "joint_success": bool(primary and quality.quality_pass and complete and not fallback_used and deadline_pass and execution_success),
        "eligible_for_primary_quality": primary,
        "reward": reward,
        "reward_version": reward_version(recall_threshold),
        "recall_threshold": recall_threshold,
    }


def _selected_queries(
    records: Iterable[QueryRecord], split: dict[str, list[str]], roles: list[str]
) -> list[QueryRecord]:
    wanted = {query_id for role in roles for query_id in split[role]}
    return [query for query in records if query.query_id in wanted]


def _collector(output: Path, config: dict[str, Any]) -> MatrixCollector:
    worker = MatrixCollector(output / "runtime", config, read_only=True, ids_only=True)
    try:
        if config.get("require_paths"):
            ready = worker.path_router.available(probe=True)
            missing = [p for p in config["require_paths"] if not ready.get(p)]
            if missing:
                raise RuntimeError(f"formal experiment requires all configured paths: missing {missing}")
        return worker
    except Exception:
        worker.close()
        raise


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"requests": 0}
    latencies = [float(row["e2e_latency_ms"]) for row in rows]
    censored = any(row.get("latency_censored", False) for row in rows)
    completed_latencies = [float(row["e2e_latency_ms"]) for row in rows if not row.get("latency_censored", False)]
    reported_latencies = completed_latencies if censored else latencies
    quality_rows = [row for row in rows if row.get("eligible_for_primary_quality", True)]
    total_observed_ms = sum(latencies)
    serial_qps = (1000.0 * len(rows) / total_observed_ms
                  if all(np.isfinite(t) and t >= 0 for t in latencies) and total_observed_ms > 0 else None)
    return {
        "requests": len(rows),
        "serial_e2e_qps": serial_qps,
        "primary_quality_requests": len(quality_rows),
        "excluded_insufficient_ground_truth": len(rows) - len(quality_rows),
        "joint_success_rate": float(np.mean([bool(row["joint_success"]) for row in rows])),
        "joint_success_rate_all_requests": float(np.mean([bool(row["joint_success"]) for row in rows])),
        "joint_success_rate_primary_quality": float(np.mean([bool(row["joint_success"]) for row in quality_rows])) if quality_rows else None,
        "joint_success_rate_scope": "all_requests",
        "quality_pass_rate": float(np.mean([row.get("quality_pass") is True for row in quality_rows])) if quality_rows else None,
        "deadline_pass_rate": float(np.mean([bool(row["deadline_pass"]) for row in rows])),
        "e2e_latency_mean_ms": float(np.mean(reported_latencies)) if reported_latencies else None,
        "e2e_latency_p50_ms": float(np.percentile(reported_latencies, 50)) if reported_latencies else None,
        "e2e_latency_p95_ms": float(np.percentile(reported_latencies, 95)) if reported_latencies else None,
        **({"latency_summary_scope": "uncensored_requests_only", "censored_requests": len(rows)-len(completed_latencies),
            "timeout_rate": float(np.mean([bool(row.get("timed_out")) for row in rows])),
            "observed_request_duration_mean_ms": float(np.mean(latencies))} if censored else {}),
    }


def _write_evaluation_summary(
    adaptive_output: Path, native_output: Path, destination: Path
) -> None:
    adaptive_rows = _read_jsonl(adaptive_output)
    native_rows = _read_jsonl(native_output)
    adaptive_by_key = {
        (str(row["query_id"]), int(row["repeat"])): row for row in adaptive_rows
    }
    native_by_key = {
        (str(row["query_id"]), int(row["repeat"])): row for row in native_rows
    }
    common = sorted(set(adaptive_by_key).intersection(native_by_key))
    latency_deltas = [
        float(native_by_key[key]["e2e_latency_ms"])
        - float(adaptive_by_key[key]["e2e_latency_ms"])
        for key in common
    ]
    path_counts: dict[str, int] = defaultdict(int)
    for row in adaptive_rows:
        path_counts[str(row.get("path_id"))] += 1
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adaptive": _metric_summary(adaptive_rows),
        "pg_native_baseline": _metric_summary(native_rows),
        "paired_requests": len(common),
        "adaptive_faster_rate": float(np.mean([delta > 0 for delta in latency_deltas])),
        "median_native_minus_adaptive_ms": float(np.median(latency_deltas)),
        "adaptive_path_selection_counts": dict(path_counts),
        "native_is_external_baseline": True,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def plan(args: argparse.Namespace) -> None:
    records = _queries(args.query_source)
    split = _read_json(args.split_json)
    runtime = _runtime_config(args)
    design = runtime.get("initial_design", {})
    configs = collection_design(runtime, args.seed)
    counts = defaultdict(int)
    for config in configs:
        counts[config.path_id] += 1
    payload = {
        "status": "PLAN_ONLY",
        "query_count": len(records),
        "k_counts": {str(k): sum(q.k == k for q in records) for k in range(1, 11)},
        "path_configuration_counts": dict(counts),
        "total_path_configurations": len(configs),
        "native_is_external_baseline": True,
        "collect_roles": args.roles,
        "collect_query_count": len(_selected_queries(records, split, args.roles)),
        "planned_collect_requests": len(_selected_queries(records, split, args.roles)) * len(configs) * args.repeats,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def collection_design(config, seed):
    design = config.get('initial_design', {})
    configurations = list(initial_design(int(design.get('points_per_numeric_path', 8)),
        seed=int(design.get('seed', seed))))
    if config.get('collection', {}).get('include_default_actions', False):
        configurations.extend(make_execution_config(row['path_id'], row.get('params'))
                              for row in config['actions'].values())
    return tuple({item.config_id: item for item in configurations}.values())


def collect(args: argparse.Namespace) -> None:
    records = _queries(args.query_source)
    split = _read_json(args.split_json)
    truth = _ground_truth(args.ground_truth_jsonl)
    queries = _selected_queries(records, split, args.roles)
    missing = sorted({q.query_id for q in queries} - set(truth))
    if missing:
        raise ValueError(f"ground truth missing queries: {missing}")
    config = _runtime_config(args)
    output = args.output_dir / "configuration_matrix.jsonl"
    done = {
        (str(row["query_id"]), str(row["config_id"]), int(row["repeat"]))
        for row in _read_jsonl(output)
    } if output.exists() else set()
    bind_run(args, config, "collect")
    worker = _collector(args.output_dir, config)
    design = config.get("initial_design", {})
    bootstrap = collection_design(config, args.seed)
    availability_file = args.output_dir / "pre_configuration_availability.jsonl"
    pre_availability = {
        str(row["query_id"]): frozenset(row["available_configurations"])
        for row in _read_jsonl(availability_file)
    } if availability_file.exists() else {}
    try:
        for repeat in range(1, args.repeats + 1):
            for query in queries:
                if query.query_id not in pre_availability:
                    # Offline candidate feasibility is a design-stage cost;
                    # each measured configuration still embeds and times E2E.
                    vector, _ = worker.embed(query.semantic_text)
                    configurations = worker.available_pre_configurations(
                        query, _vector_literal(vector)
                    )
                    pre_availability[query.query_id] = configurations
                    _append(availability_file, {"query_id": query.query_id,
                                                "available_configurations": sorted(configurations)})
                allowed = pre_availability[query.query_id]
                configs = [item for item in bootstrap if item.path_id != "PRE" or
                           pre_configuration_key(item.params) in allowed]
                random.Random(f"{args.seed}:{query.query_id}:{repeat}").shuffle(configs)
                for execution_config in configs:
                    key = (query.query_id, execution_config.config_id, repeat)
                    if key in done:
                        continue
                    for _ in range(int(config.get("measurement", {}).get("warmup_runs", 0))):
                        try:
                            from vec2plan_bandit.experiments.workflow import budgeted_request
                            budgeted_request(lambda: run_fixed_configuration_request(query=query, collector=worker,
                                execution_config=execution_config, latency_cap_ms=args.latency_cap_ms),
                                query,args.latency_cap_ms,worker)
                        except Exception:
                            pass  # warm-up failures do not become training labels
                    started_ns = time.perf_counter_ns()
                    try:
                        from vec2plan_bandit.experiments.workflow import budgeted_request
                        row = budgeted_request(lambda: run_fixed_configuration_request(
                            query=query,
                            collector=worker,
                            execution_config=execution_config,
                            latency_cap_ms=args.latency_cap_ms,
                        ),query,args.latency_cap_ms,worker)
                    except Exception as exc:
                        row = {
                            "query_id": query.query_id,
                            "k": query.k,
                            "article_ids": [],
                            "path_id": execution_config.path_id,
                            "action_id": execution_config.action_id,
                            "config_id": execution_config.config_id,
                            "requested_params": execution_config.params,
                            "e2e_latency_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
                            "deadline_pass": (time.perf_counter_ns() - started_ns) / 1_000_000 < args.latency_cap_ms,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:500],
                        }
                    row.update(
                        path_id=execution_config.path_id,
                        action_id=execution_config.action_id,
                        config_id=execution_config.config_id,
                        requested_params=dict(execution_config.params),
                        repeat=repeat,
                        requested_budget_ms=args.latency_cap_ms,
                        measurement_protocol="LIVE_FIXED_CONFIG_E2E_V1",
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                    )
                    _append(output, _adjudicate(row, truth, args.latency_cap_ms, getattr(args, "recall_threshold", 0.9)))
                    done.add(key)
    finally:
        worker.close()


def _train_policy(
    records: list[QueryRecord],
    split: dict[str, list[str]],
    matrix: Path,
    latency_cap_ms: float,
    model_config: dict[str, Any],
    *,
    statistics: dict[str, Any] | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[FeaturePipeline, MaxUCB, ConstrainedCatBoostSurrogate]:
    pipeline, observations = build_training_observations(
        records, split, _read_jsonl(matrix), latency_cap_ms=latency_cap_ms,
        statistics=statistics, embeddings=embeddings,
        recall_threshold=float(model_config["reward"]["recall_threshold"]),
    )
    policy, surrogate = _fit_policy(observations, model_config)
    return pipeline, policy, surrogate


def _fit_policy(observations: list[dict[str, Any]], model_config: dict[str, Any]) -> tuple[MaxUCB, ConstrainedCatBoostSurrogate]:
    cat = model_config.get("catboost", {})
    options = dict(
        ensemble_size=int(cat.get("ensemble_size", 8)),
        iterations=int(cat.get("iterations", 300)),
        depth=int(cat.get("depth", 6)),
        learning_rate=float(cat.get("learning_rate", 0.05)),
        seed=int(cat.get("seed", 42)),
        uncertainty_floor=float(cat.get("uncertainty_floor", 1e-6)),
        task_type=str(cat.get("task_type", "CPU")),
        devices=cat.get("devices"),
    )
    x=np.vstack([row['features'] for row in observations])
    q=np.asarray([row['quality_target'] for row in observations])
    t=np.asarray([row['e2e_latency_ms'] for row in observations])
    if model_config.get('main_training',{}).get('enabled',False):
        from vec2plan_bandit.surrogates.main_cost import MainCostSurrogate,feedback_fields
        surrogate=MainCostSurrogate(**options,training_devices=
            model_config.get('runtime',{}).get('main_training_gpus',()))
        surrogate.fit_feedback(x,q,t,[feedback_fields(row) for row in observations])
    else:
        surrogate=ConstrainedCatBoostSurrogate(**options).fit(x,q,t)
    max_ucb = model_config.get("max_ucb", {})
    policy = MaxUCB(
        surrogate,
        beta_quality=float(max_ucb.get("beta_quality", 1.0)),
        beta_latency=float(max_ucb.get("beta_latency", 1.0)),
        kappa_quality=float(max_ucb.get("kappa_quality", 1.0)),
        kappa_latency=float(max_ucb.get("kappa_latency", 1.0)),
        quality_probability_min=float(max_ucb.get("quality_probability_min", 0.8)),
        gamma=float(max_ucb.get("gamma", 1.0)),
        candidates_per_numeric_path=int(max_ucb.get("candidates_per_numeric_path", 128)),
        seed=int(max_ucb.get("seed", 42)),
        fallback_quality_tolerance=float(max_ucb.get("fallback_quality_tolerance", 0.0)),
    )
    for row in observations:
        policy.update(row["path_id"])
    return policy, surrogate


def _feature_inputs(args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
    stats_path = getattr(args, "query_statistics", None)
    embeddings_path = getattr(args, "feature_embeddings", None)
    statistics = _read_json(stats_path) if stats_path else None
    config = _runtime_config(args)
    if config.get("query_features", {}).get("local_statistics", False):
        worker = MatrixCollector(args.output_dir / "feature-runtime", config, read_only=True)
        try:
            feature_records = _queries(args.query_source)
            if config.get("query_input", {}).get("contract") == "question_string_only":
                train_ids = set(_read_json(args.split_json)["train"])
                feature_records = [q for q in feature_records if q.query_id in train_ids]
            statistics = {q.query_id: worker.estimate_statistics(q) for q in feature_records}
        finally:
            worker.close()
    return statistics, load_frozen_embeddings(embeddings_path) if embeddings_path else None


def evaluate(args: argparse.Namespace) -> None:
    records = _queries(args.query_source)
    split = _read_json(args.split_json)
    truth = _ground_truth(args.ground_truth_jsonl)
    config = _runtime_config(args)
    bind_run(args, config, "evaluate")
    statistics, embeddings = _feature_inputs(args)
    if getattr(args, "checkpoint", None):
        pipeline, policy, surrogate = load_bundle(args.checkpoint, cap_ms=args.latency_cap_ms, train_ids=split["train"],
            expected={"training_source_hashes": {q.query_id: q.source_hash for q in records if q.query_id in split["train"]},
                      "snapshot_id": config.get("snapshot_id"), "embedding": config.get("embedding")})
    else:
        pipeline, policy, surrogate = _train_policy(
            records, split, args.training_matrix, args.latency_cap_ms, config,
            statistics=statistics, embeddings=embeddings,
        )
    queries = _selected_queries(records, split, args.roles)
    if "train" in args.roles or set(q.group_id for q in queries) & {q.group_id for q in records if q.query_id in split["train"]}:
        raise ValueError("frozen evaluation must not overlap training queries/groups")
    missing = sorted({q.query_id for q in queries} - set(truth))
    if missing:
        raise ValueError(f"ground truth missing queries: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(args.output_dir / "feature_pipeline.npz")
    surrogate.save(args.output_dir / "catboost_surrogate")
    adaptive_output = args.output_dir / "adaptive_live.jsonl"
    native_output = args.output_dir / "native_baseline_live.jsonl"
    adaptive_done = {
        (str(row["query_id"]), int(row["repeat"]))
        for row in _read_jsonl(adaptive_output)
    } if adaptive_output.exists() else set()
    native_done = {
        (str(row["query_id"]), int(row["repeat"]))
        for row in _read_jsonl(native_output)
    } if native_output.exists() else set()
    worker = _collector(args.output_dir, config)
    jev_router = jev_router_from_config(config.get("jev", {}))
    try:
        for repeat in range(1, args.repeats + 1):
            for query in queries:
                key = (query.query_id, repeat)
                if key not in adaptive_done:
                    for _ in range(int(config.get("measurement", {}).get("warmup_runs", 0))):
                        try:
                            run_adaptive_request(query=query, collector=worker, feature_pipeline=pipeline,
                                policy=policy, latency_cap_ms=args.latency_cap_ms, statistics=statistics,
                                exploration=False, jev_router=jev_router)
                        except Exception:
                            pass
                    started_ns = time.perf_counter_ns()
                    try:
                        adaptive = run_adaptive_request(
                            query=query,
                            collector=worker,
                            feature_pipeline=pipeline,
                            policy=policy,
                            latency_cap_ms=args.latency_cap_ms,
                            statistics=statistics,
                            exploration=False,
                            jev_router=jev_router,
                        )
                    except Exception as exc:
                        elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000
                        adaptive = {
                            "query_id": query.query_id, "k": query.k,
                            "article_ids": [], "path_id": None, "config_id": None,
                            "e2e_latency_ms": elapsed,
                            "deadline_pass": elapsed < args.latency_cap_ms,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:500],
                        }
                    adaptive.update(
                        repeat=repeat,
                        requested_budget_ms=args.latency_cap_ms,
                        measurement_protocol="LIVE_ADAPTIVE_E2E_V1",
                        evaluation_mode="frozen",
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                    )
                    _append(adaptive_output, _adjudicate(adaptive, truth, args.latency_cap_ms, getattr(args, "recall_threshold", 0.9)))
                    adaptive_done.add(key)

                if key not in native_done:
                    for _ in range(int(config.get("measurement", {}).get("warmup_runs", 0))):
                        try:
                            run_native_baseline_request(query=query, collector=worker, latency_cap_ms=args.latency_cap_ms)
                        except Exception:
                            pass
                    started_ns = time.perf_counter_ns()
                    try:
                        native = run_native_baseline_request(
                            query=query,
                            collector=worker,
                            latency_cap_ms=args.latency_cap_ms,
                        )
                    except Exception as exc:
                        elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000
                        native = {
                            "query_id": query.query_id, "k": query.k,
                            "article_ids": [], "baseline_id": "PG_NATIVE_SQL",
                            "e2e_latency_ms": elapsed,
                            "deadline_pass": elapsed < args.latency_cap_ms,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:500],
                        }
                    native.update(
                        repeat=repeat,
                        requested_budget_ms=args.latency_cap_ms,
                        measurement_protocol="LIVE_NATIVE_BASELINE_E2E_V1",
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                    )
                    _append(native_output, _adjudicate(native, truth, args.latency_cap_ms, getattr(args, "recall_threshold", 0.9)))
                    native_done.add(key)
    finally:
        if jev_router is not None:
            jev_router.close()
        worker.close()
    _write_evaluation_summary(
        adaptive_output, native_output, args.output_dir / "evaluation_summary.json"
    )


def _online_contract(args: argparse.Namespace, config: dict[str, Any]) -> str:
    """Bind a journal to its model, data, features, budgets and stream ordering."""
    paths = [args.query_source, args.split_json, args.training_matrix, args.ground_truth_jsonl,
             getattr(args, "query_statistics", None), getattr(args, "feature_embeddings", None)]
    payload = {
        "policy": MaxUCB.VERSION, "learner": OnlineLearner.VERSION, "config": config,
        "action_features": ACTION_FEATURE_VERSION, "reward": REWARD_VERSION,
        "roles": args.roles, "repeats": args.repeats, "cap_ms": args.latency_cap_ms,
        "inputs": [hashlib.sha256(p.read_bytes()).hexdigest() if p else None for p in paths],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _apply_online_feedback(learner: OnlineLearner, row: dict[str, Any]) -> None:
    from vec2plan_bandit.surrogates.main_cost import feedback_fields
    if row.get("path_id") is not None:
        learner.observe(observation_id=row["observation_id"], path_id=row["path_id"],
                        features=row["policy_features"],
                        quality_target=float(row.get("quality_pass") is True and row.get("result_full_k") is True
                                             and not row.get("fallback_used") and not row.get("error_type")),
                        e2e_latency_ms=float(row["e2e_latency_ms"]),feedback_metadata=feedback_fields(row))


def online(args: argparse.Namespace) -> None:
    """Prequential evaluation: predict, execute, score, then learn from that outcome.

    Ground truth is used only for post-execution experimental feedback. This is
    not a production claim that exact recall is freely available on live queries.
    """
    records = _queries(args.query_source)
    split = _read_json(args.split_json)
    if "train" in args.roles:
        raise ValueError("online stream must be held out from bootstrap train queries")
    queries = _selected_queries(records, split, args.roles)
    truth = _ground_truth(args.ground_truth_jsonl)
    if not queries or any(q.query_id not in truth for q in queries):
        raise ValueError("online stream must be nonempty with ground truth for every query")
    config = _runtime_config(args)
    statistics, embeddings = _feature_inputs(args)
    pipeline, observations = build_training_observations(
        records, split, _read_jsonl(args.training_matrix), latency_cap_ms=args.latency_cap_ms,
        statistics=statistics, embeddings=embeddings,
        recall_threshold=float(config["reward"]["recall_threshold"]),
    )
    policy, surrogate = _fit_policy(observations, config)
    learner = OnlineLearner(policy, observations, refit_every=config.get("online_learning", {}).get("refit_every", 8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "online_live.jsonl"
    contract = _online_contract(args, config)
    stream = [(repeat, query) for repeat in range(1, args.repeats + 1) for query in queries]
    previous = _read_jsonl(output) if output.exists() else []
    if len(previous) > len(stream):
        raise ValueError("online journal exceeds configured stream")
    # Replay in recorded order, including every refit boundary. Never update a
    # second time for a completed request after resuming an interrupted run.
    for index, row in enumerate(previous):
        repeat, query = stream[index]
        if row.get("contract_hash") != contract or row.get("observation_id") != f"{repeat}:{query.query_id}":
            raise ValueError("online journal contract/order mismatch; use a new output directory")
        _apply_online_feedback(learner, row)
    pipeline.save(args.output_dir / "feature_pipeline.npz")
    worker = _collector(args.output_dir, config)
    jev_router = jev_router_from_config(config.get("jev", {}))
    try:
        for repeat, query in stream[len(previous):]:
            started = time.perf_counter_ns()
            try:
                row = run_adaptive_request(query=query, collector=worker, feature_pipeline=pipeline,
                    policy=policy, statistics=statistics, latency_cap_ms=args.latency_cap_ms,
                    exploration=True, jev_router=jev_router)
            except Exception as error:
                elapsed = (time.perf_counter_ns() - started) / 1_000_000
                row = {"query_id": query.query_id, "k": query.k, "path_id": None, "article_ids": [],
                       "e2e_latency_ms": elapsed, "deadline_pass": elapsed < args.latency_cap_ms,
                       "error_type": type(error).__name__, "error_message": str(error)[:500]}
            row = _adjudicate(row, truth, args.latency_cap_ms, getattr(args, "recall_threshold", 0.9))
            row.update(repeat=repeat, observation_id=f"{repeat}:{query.query_id}", contract_hash=contract,
                       evaluation_mode="online_prequential", requested_budget_ms=args.latency_cap_ms,
                       recorded_at=datetime.now(timezone.utc).isoformat())
            # Durable outcome first; if fitting fails, resume replays this event.
            _append(output, row)
            _apply_online_feedback(learner, row)
        learner.save(args.output_dir / "online_checkpoint")
    finally:
        if jev_router is not None:
            jev_router.close()
        worker.close()
    rows = _read_jsonl(output)
    summary = {**_metric_summary(rows), "evaluation_mode": "online_prequential",
               "cumulative_reward": float(sum(row["reward"] for row in rows)),
               "refit_count": learner.refit_count, "policy_state": policy.state_dict(),
               "pending_refit_observations": len(learner.feedback_ids) % learner.refit_every}
    (args.output_dir / "online_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_baselines(args: argparse.Namespace) -> None:
    records = _queries(args.query_source)
    split = _read_json(args.split_json)
    truth = _ground_truth(args.ground_truth_jsonl)
    queries = _selected_queries(records, split, args.roles)
    missing = sorted({query.query_id for query in queries} - set(truth))
    if missing:
        raise ValueError(f"ground truth missing queries: {missing}")
    baseline_config = load_configuration(args.baseline_config)
    bind_run(args, _runtime_config(args), "baselines")
    validate_baseline_config(baseline_config)
    statistics, embeddings = _feature_inputs(args)
    pipeline, observations = build_training_observations(
        records,
        split,
        _read_jsonl(args.training_matrix),
        latency_cap_ms=args.latency_cap_ms,
        recall_threshold=float(_runtime_config(args)["reward"]["recall_threshold"]),
        statistics=statistics, embeddings=embeddings,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(args.output_dir / "baseline_feature_pipeline.npz")
    store = JsonlResultStore(args.output_dir / "baselines_live.jsonl")
    registry = builtin_baselines()
    worker = _collector(args.output_dir, _runtime_config(args))
    baselines_to_close: list[Any] = []
    try:
        for specification in baseline_config["baselines"]:
            method_id = str(specification["id"])
            name = str(specification["name"]).lower()
            if name == "text2vectorsql":
                from vec2plan_bandit.baselines.text2vectorsql import Text2VectorSQLTranslator
                baseline = Text2VectorSQLTranslator(**specification.get("config", {}))
            else:
                baseline = None if name == "native-pg" else registry.create(
                    name, specification.get("config", {})
                ).fit(observations)
            if baseline is not None:
                baselines_to_close.append(baseline)
            for repeat in range(1, args.repeats + 1):
                for query in queries:
                    if store.contains(method_id, query.query_id, repeat):
                        continue
                    started_ns = time.perf_counter_ns()
                    try:
                        if name == "native-pg":
                            row = run_native_baseline_request(
                                query=query, collector=worker,
                                latency_cap_ms=args.latency_cap_ms,
                            )
                        elif name == "text2vectorsql":
                            from vec2plan_bandit.baselines.text2vectorsql import run_text2vectorsql_request
                            row = run_text2vectorsql_request(query=query, collector=worker,
                                translator=baseline, latency_cap_ms=args.latency_cap_ms)
                        else:
                            assert baseline is not None
                            row = run_selector_baseline_request(
                                query=query, collector=worker,
                                feature_pipeline=pipeline, baseline=baseline,
                                latency_cap_ms=args.latency_cap_ms,
                                statistics=statistics,
                            )
                    except Exception as error:
                        elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000
                        row = {
                            "query_id": query.query_id,
                            "k": query.k,
                            "article_ids": [],
                            "path_id": None,
                            "config_id": None,
                            "e2e_latency_ms": elapsed,
                            "deadline_pass": elapsed < args.latency_cap_ms,
                            "error_type": type(error).__name__,
                            "error_message": str(error)[:500],
                        }
                    row.update(
                        method_id=method_id,
                        repeat=repeat,
                        requested_budget_ms=args.latency_cap_ms,
                        measurement_protocol="LIVE_BASELINE_E2E_V1",
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                    )
                    store.append(_adjudicate(row, truth, args.latency_cap_ms, getattr(args, "recall_threshold", 0.9)))
    finally:
        for baseline in baselines_to_close:
            close = getattr(baseline, "close", None)
            if close is not None:
                close()
        worker.close()
    rows = store.read()
    summary = {
        method_id: _metric_summary([row for row in rows if row["method_id"] == method_id])
        for method_id in sorted({str(row["method_id"]) for row in rows})
    }
    (args.output_dir / "baselines_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def preflight(args: argparse.Namespace) -> None:
    output = args.output_dir or PROJECT_ROOT / "outputs" / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    report = run_preflight(_runtime_config(args), output)
    (output / "preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["formal_ready"]:
        raise RuntimeError("formal preflight failed; see preflight.json")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("phase", choices=("preflight", "plan", "prepare", "collect", "train", "refit", "evaluate", "online", "baselines", "benchmark"))
    value.add_argument("--checkpoint", type=Path, help="Frozen bundle produced by train")
    value.add_argument("--query-limit", type=int, help="Limit dev benchmark queries for smoke testing")
    value.add_argument("--training-policy", choices=("cash-maxucb", "maxucb-bo"), default="cash-maxucb")
    value.add_argument("--feature-embeddings", type=Path, help="Aligned frozen query embeddings NPZ; PCA fitted on train only")
    value.add_argument("--query-statistics", type=Path, help="Pre-execution statistics keyed by query_id")
    value.add_argument("--query-source", type=Path)
    value.add_argument("--split-json", type=Path)
    value.add_argument("--ground-truth-jsonl", type=Path)
    value.add_argument("--training-matrix", type=Path)
    value.add_argument("--baseline-config", type=Path)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--db-config-path", type=Path)
    value.add_argument("--faiss-abstract-index", type=Path)
    value.add_argument("--faiss-abstract-ids", type=Path)
    value.add_argument("--faiss-title-index", type=Path)
    value.add_argument("--faiss-title-ids", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--roles", nargs="+", choices=("train", "dev", "test"))
    value.add_argument("--repeats", type=int, help="Defaults to measurement.latency_runs")
    value.add_argument("--latency-cap-ms", type=float,
                       help="Defaults to the configured training/evaluation budget")
    value.add_argument("--seed", type=int, default=42)
    return value


def main() -> int:
    arguments = parser()
    args = arguments.parse_args()
    if args.phase in {"baselines", "benchmark"} and args.baseline_config is None:
        arguments.error("--baseline-config is required for baselines and benchmark")
    if args.query_limit is not None and (args.query_limit <= 0 or args.phase != "benchmark" or args.roles != ["dev"]):
        raise ValueError("--query-limit requires a positive dev-only benchmark limit")
    if args.checkpoint and args.phase not in {"evaluate", "benchmark"}:
        raise ValueError("--checkpoint is supported only by evaluate/benchmark")
    if args.roles is None:
        args.roles = ["train"] if args.phase in {"collect", "train", "refit"} else (["train", "dev", "test"] if args.phase in {"plan", "prepare"} else (["dev"] if args.phase == "online" else ["test"]))
    if args.repeats is None:
        args.repeats = int(_runtime_config(args).get("measurement", {}).get("latency_runs", 5))
    runtime = _runtime_config(args)
    if args.latency_cap_ms is None:
        args.latency_cap_ms = float(runtime.get('training_latency_cap_ms',runtime['reward']['latency_cap_ms'])
            if args.phase in {'collect','train','refit'} else runtime['reward']['latency_cap_ms'])
    if args.repeats <= 0 or args.latency_cap_ms <= 0:
        raise ValueError("repeats and latency cap must be positive")
    validate_experiment_config(runtime)
    args.recall_threshold = float(runtime["reward"]["recall_threshold"])
    if args.phase == "preflight":
        preflight(args)
        return 0
    if args.query_source is None or args.split_json is None or args.output_dir is None:
        raise ValueError("--query-source, --split-json and --output-dir are required for this phase")
    if args.phase in {"collect", "train", "refit", "evaluate", "online", "baselines", "benchmark"} and args.ground_truth_jsonl is None:
        raise ValueError("--ground-truth-jsonl is required")
    from vec2plan_bandit.experiments.configuration import checkpoint_only_baselines
    frozen_only = args.phase == "benchmark" and args.checkpoint and checkpoint_only_baselines(load_configuration(args.baseline_config))
    if args.phase in {"train", "refit", "evaluate", "online", "baselines", "benchmark"} and args.training_matrix is None and not ((args.phase == "evaluate" and args.checkpoint) or frozen_only):
        raise ValueError("--training-matrix is required for evaluate/baselines")
    if args.phase == "plan":
        plan(args)
        return 0
    inputs = [args.config, args.query_source, args.split_json]
    inputs.extend(p for p in (args.feature_embeddings, args.query_statistics) if p is not None)
    if args.ground_truth_jsonl is not None:
        inputs.append(args.ground_truth_jsonl)
    if args.training_matrix is not None:
        inputs.append(args.training_matrix)
    if args.phase in {"baselines", "benchmark"}:
        inputs.append(args.baseline_config)
    manifest = RunManifest(
        args.output_dir / "run_manifest.json",
        project_root=PROJECT_ROOT,
        phase=args.phase,
        seed=args.seed,
        inputs=inputs,
    )
    try:
        {"collect": collect, "train": train, "refit": refit, "prepare": prepare, "benchmark": benchmark,
         "evaluate": evaluate, "online": online, "baselines": run_baselines}[args.phase](args)
    except BaseException as error:
        manifest.fail(error)
        raise
    manifest.complete()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
