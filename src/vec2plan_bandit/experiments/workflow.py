"""Offline preparation, train-only exploration, and interleaved frozen evaluation."""
from __future__ import annotations

import json
import hashlib
import random
import time
from datetime import datetime, timezone

import numpy as np


def refit(args):
    """Relabel existing train executions and fit offline, with no new queries."""
    from vec2plan_bandit.experiments import run as r
    from vec2plan_bandit.assets.indexes import sha256
    config = r._runtime_config(args)
    if args.roles != ["train"]:
        raise ValueError("offline refit uses train records only")
    r.bind_run(args, config, "refit")
    started = time.perf_counter()
    threshold = float(config["reward"]["recall_threshold"])
    records, split = r._queries(args.query_source), r._read_json(args.split_json)
    truth = r._ground_truth(args.ground_truth_jsonl)
    train_ids = set(split["train"])
    source = [row for row in r._read_jsonl(args.training_matrix) if row["query_id"] in train_ids]
    if any(row["query_id"] not in truth for row in source):
        raise ValueError("train ground truth is incomplete")
    labeled = [r._adjudicate(row, truth, args.latency_cap_ms, threshold) for row in source]
    stats, embeddings = r._feature_inputs(args)
    pipeline, observations = r.build_training_observations(records, split, labeled,
        latency_cap_ms=args.latency_cap_ms, recall_threshold=threshold,
        statistics=stats, embeddings=embeddings)
    with (args.output_dir / "shared_training.jsonl").open("w") as handle:
        for row in labeled:
            handle.write(json.dumps(row) + "\n")
    policy, _ = r._fit_policy(observations, config)
    r.save_bundle(args.output_dir / "frozen_model", pipeline, policy, {
        "cap_ms": args.latency_cap_ms, "recall_threshold": threshold,
        "training_query_ids": split["train"],
        "training_source_hashes": {q.query_id: q.source_hash for q in records if q.query_id in train_ids},
        "snapshot_id": config.get("snapshot_id"), "embedding": config.get("embedding")})
    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "training_mode": "offline_relabel_refit", "training_observations": len(observations),
        "training_queries": len({row["query_id"] for row in labeled}),
        "source_matrix": str(args.training_matrix), "source_sha256": sha256(args.training_matrix),
        "recall_threshold": threshold, "quality_probability_min": policy.quality_probability_min,
        "quality_pass_before": sum(row.get("quality_pass") is True for row in source),
        "quality_pass_after": sum(row.get("quality_pass") is True for row in labeled),
        "changed_rewards": sum(a["reward"] != b["reward"] for a, b in zip(source, labeled)),
        "new_execution_count": 0, "dev_test_labels_used": False,
        "training_ms": (time.perf_counter() - started) * 1000,
        "policy": policy.state_dict()}, indent=2))


def prepare(args):
    from vec2plan_bandit.experiments import run as r
    from vec2plan_bandit.experiments.collector import _vector_literal
    from vec2plan_bandit.paths.sql import render_exact_ground_truth
    config = r._runtime_config(args)
    r.bind_run(args, config, "prepare")
    records = r._queries(args.query_source)
    queries = r._selected_queries(records, r._read_json(args.split_json), args.roles)
    if not queries:
        raise ValueError("preparation requires nonempty queries")
    from vec2plan_bandit.experiments.preparation_cache import verified_cache,write_receipt
    cached,cache_receipt=verified_cache(config,queries,query_source=args.query_source,split_json=args.split_json)
    if cache_receipt:
        write_receipt(args.output_dir/'cache-reuse.json',cache_receipt)
    journal = args.output_dir / "prepared.jsonl"
    done = {row["query_id"] for row in r._read_jsonl(journal)} if journal.exists() else set()
    # Exact ground truth and PostgreSQL EXPLAIN do not depend on FAISS/Qdrant
    # readiness. This stage can run while the remote graph finishes optimizing.
    worker = r._collector(args.output_dir, {**config, "require_paths": []})
    try:
        if cached:
            train=set(r._read_json(args.split_json)['train'])
            probes=sorted([q for q in queries if q.query_id in train],
                          key=lambda q:hashlib.sha256(q.query_id.encode()).hexdigest())[:4]
            for query in probes:
                vector,_=worker.embed(query.semantic_text)
                if not np.allclose(vector,cached[query.query_id]['embedding'],atol=1e-6):
                    raise ValueError('Preparation cache encoder probe changed')
                ids,_=worker._pg(render_exact_ground_truth(query),_vector_literal(vector))
                if ids!=cached[query.query_id]['topk_ids'] or worker.estimate_statistics(query)!=cached[query.query_id]['statistics']:
                    raise ValueError('Preparation cache truth/statistics probe changed')
                cache_receipt['live_train_probes'].append(query.query_id)
                write_receipt(args.output_dir/'cache-reuse.json',cache_receipt)
        for query in queries:
            if query.query_id in done:
                continue
            vector=np.asarray(cached[query.query_id]['embedding'],dtype=np.float32) if cached else worker.embed(query.semantic_text)[0]
            literal = _vector_literal(vector)
            plans = worker.explain(query, worker.compile(query), literal)
            # Indexed PRE variants depend on the query and installed indexes.
            # Their failures are audited, but cannot block exact truth.
            # Older collectors exposed the exact scan path as ``PRE`` while
            # current collectors expose the concrete strategy key.  Accept
            # both so an existing preparation journal/worker remains usable.
            pre_plan = plans.get("PRE:scan:all", plans.get("PRE"))
            if pre_plan is None or not pre_plan["passed"] or not plans["POST"]["passed"]:
                raise RuntimeError(f"required PRE/POST plan semantics failed for {query.query_id}")
            if cached:
                row=dict(cached[query.query_id],preparation_reused_from=cache_receipt['source_output'])
            else:
                ids,elapsed=worker._pg(render_exact_ground_truth(query),literal)
                row={"query_id":query.query_id,"topk_ids":ids,"embedding":vector.tolist(),
                     "statistics":worker.estimate_statistics(query),"truth_ms":elapsed,
                     "tie_rule":"cosine_distance_then_article_id","source_hash":query.source_hash}
            r._append(journal,row)
    finally:
        worker.close()
    rows = r._read_jsonl(journal)
    with (args.output_dir / "ground_truth.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in ("query_id", "topk_ids", "source_hash", "tie_rule")}) + "\n")
    np.savez_compressed(args.output_dir / "query_embeddings.npz", query_ids=[row["query_id"] for row in rows],
                        embeddings=np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    (args.output_dir / "statistics.json").write_text(json.dumps({row["query_id"]: row["statistics"] for row in rows if row["statistics"] is not None}, indent=2))
    if cache_receipt:
        cache_receipt.update(plans_refreshed=True,prepared_queries=len(rows))
        write_receipt(args.output_dir/'cache-reuse.json',cache_receipt)


def train(args):
    from vec2plan_bandit.experiments import run as r
    started = time.perf_counter()
    if args.roles != ["train"]:
        raise ValueError("active training must use train queries only; never dev/test labels")
    config = r._runtime_config(args)
    r.bind_run(args, config, "train")
    records, split = r._queries(args.query_source), r._read_json(args.split_json)
    statistics, embeddings = r._feature_inputs(args)
    source_rows = r._read_jsonl(args.training_matrix)
    pipeline, observations = r.build_training_observations(records, split, source_rows,
        latency_cap_ms=args.latency_cap_ms, recall_threshold=float(config["reward"]["recall_threshold"]), statistics=statistics, embeddings=embeddings)
    training_policy = getattr(args, "training_policy", "cash-maxucb")
    refit_every = config.get("online_learning", {}).get("refit_every", 8)
    bo = None
    if training_policy == "maxucb-bo":
        from vec2plan_bandit.baselines.maxucb_bo import MaxUCBBayesianOptimizationBaseline
        options = config.get("max_ucb", {})
        bo = MaxUCBBayesianOptimizationBaseline(beta=options.get("beta", 1), gamma=options.get("gamma", 1),
            seed=options.get("seed", 42), candidates_per_numeric_path=options.get("candidates_per_numeric_path", 128),
            exploration=True).fit(observations)
        policy, learner = None, None
    else:
        policy, _ = r._fit_policy(observations, config)
        learner = r.OnlineLearner(policy, observations, refit_every=refit_every)
    initial_count = len(observations)
    def observe(row):
        if bo is None:
            r._apply_online_feedback(learner, row)
        elif row.get("path_id") is not None:
            observations.append({"query_id": row["query_id"], "path_id": row["path_id"],
                "features": row["policy_features"], "query_context": row["query_context"], "reward": row["reward"]})
            bo.path_counts[row["path_id"]] += 1
            if (len(observations) - initial_count) % refit_every == 0:
                bo.fit(observations)
    truth = r._ground_truth(args.ground_truth_jsonl)
    queries = r._selected_queries(records, split, ["train"])
    if any(q.query_id not in truth for q in queries):
        raise ValueError("training ground truth is incomplete")
    stream = []
    for repeat in range(1, args.repeats + 1):
        ordered = list(queries)
        random.Random(f"{getattr(args, 'seed', 42)}:{repeat}:train").shuffle(ordered)
        stream.extend((repeat, q) for q in ordered)
    output = args.output_dir / "training_live.jsonl"
    previous = r._read_jsonl(output) if output.exists() else []
    if len(previous) > len(stream):
        raise ValueError("training journal longer than requested stream")
    for row, (repeat, query) in zip(previous, stream):
        if row["observation_id"] != f"{repeat}:{query.query_id}":
            raise ValueError("training journal order mismatch")
        observe(row)
    worker = r._collector(args.output_dir, config)
    jev_router = r.jev_router_from_config(config.get("jev", {}))
    def execute(query):
        if bo is not None:
            return r.run_selector_baseline_request(query=query, collector=worker, feature_pipeline=pipeline,
                baseline=bo, latency_cap_ms=args.latency_cap_ms, statistics=statistics)
        return r.run_adaptive_request(query=query, collector=worker, feature_pipeline=pipeline,
            policy=policy, latency_cap_ms=args.latency_cap_ms, statistics=statistics,
            exploration=True, jev_router=jev_router)
    refit_ms = 0.0
    try:
        for repeat, query in stream[len(previous):]:
            for _ in range(int(config.get("measurement", {}).get("warmup_runs", 0))):
                budgeted_request(lambda: execute(query), query, args.latency_cap_ms,worker)
            row = budgeted_request(lambda: execute(query), query, args.latency_cap_ms,worker)
            row = r._adjudicate(row, truth, args.latency_cap_ms, float(config["reward"]["recall_threshold"]))
            row.update(repeat=repeat, observation_id=f"{repeat}:{query.query_id}",
                       requested_budget_ms=args.latency_cap_ms, evaluation_mode="train_exploration", training_policy=training_policy,
                       recorded_at=datetime.now(timezone.utc).isoformat())
            r._append(output, row)
            before = time.perf_counter()
            observe(row)
            refit_ms += (time.perf_counter() - before) * 1000
        # Include the final partial batch before publishing the frozen model.
        before = time.perf_counter()
        if bo is not None:
            bo.fit(observations)
        else:
            learner.refit()
        refit_ms += (time.perf_counter() - before) * 1000
        bundle = args.output_dir / "frozen_model"
        if bo is None and not bundle.exists():
            r.save_bundle(bundle, pipeline, policy, {"cap_ms": args.latency_cap_ms,
                "training_query_ids": split["train"], "training_source_hashes": {q.query_id: q.source_hash for q in queries},
                "snapshot_id": config.get("snapshot_id"), "embedding": config.get("embedding"),
                "recall_threshold": float(config["reward"]["recall_threshold"])})
        # All frozen methods can use exactly the same labeled training data.
        labeled = [row for row in source_rows if row["query_id"] in split["train"]]
        labeled += [row for row in r._read_jsonl(output) if row.get("path_id") is not None]
        with (args.output_dir / "shared_training.jsonl").open("w") as handle:
            for row in labeled:
                handle.write(json.dumps(row) + "\n")
        (args.output_dir / "training_summary.json").write_text(json.dumps({
            "training_policy": training_policy,
            "policy": policy.state_dict() if bo is None else {"path_counts": bo.path_counts, "round_index": sum(bo.path_counts.values())},
            "training_observations": len(learner.quality_targets) if bo is None else len(observations),
            "this_invocation_training_ms": (time.perf_counter() - started) * 1000,
            "this_invocation_feedback_and_final_refit_ms": refit_ms,
            "test_feedback_used": False}, indent=2))
    finally:
        if jev_router is not None:
            jev_router.close()
        worker.close()


def _outcome(call, query, cap):
    start = time.perf_counter_ns()
    try:
        return call()
    except Exception as error:
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        return {"query_id": query.query_id, "k": query.k, "article_ids": [], "path_id": None,
                "e2e_latency_ms": elapsed, "deadline_pass": elapsed <= cap,
                "error_type": type(error).__name__, "error_message": str(error)[:500]}


def budgeted_request(call,query,cap,worker):
    from vec2plan_bandit.execution.timing import run_with_deadline,evaluation_deadline_ns
    if evaluation_deadline_ns() is not None:
        return _outcome(call,query,cap)
    result=run_with_deadline(lambda:_outcome(call,query,cap),query,cap,
        getattr(worker,'cancel_pending_request',None))
    if result.get('timed_out') and hasattr(worker,'finish_cancelled_request'):
        worker.finish_cancelled_request()
    return result


def preload_benchmark_assets(worker,config,queries):
    """Initialize opt-in serving assets at the actual formal benchmark entry.

    A failed startup must stop the phase, rather than create an entire journal
    of fast failed requests. No query-result or eligibility caches are built.
    """
    guard=config.get('main_recall_guard',{})
    if guard.get('enabled',False) and (guard.get('cpu_exact_scan',False) or guard.get('cpu_exact_fallback',False)):
        elapsed=worker.preload_statistics()
        if getattr(worker,'exact_metadata',None) is None:
            raise RuntimeError('Formal CPU exact metadata initialization failed')
        return dict(exact_metadata_ready=True,startup_ms=elapsed,query_result_cache=False)
    if config.get('measurement',{}).get('preload_corpus_statistics',False):
        worker.estimate_statistics(queries[0])
    return dict(exact_metadata_ready=False,query_result_cache=False)


def configure_evaluation_probability(policy, config):
    """Apply a cell's probability to a model without changing its weights."""
    value = float(config.get('max_ucb', {}).get('quality_probability_min', 0.8))
    if not 0 <= value <= 1:
        raise ValueError('quality_probability_min must be within [0, 1]')
    calibration=config.get('main_probability_calibration',{})
    if calibration.get('enabled',False):
        import hashlib
        from pathlib import Path
        raw=Path(calibration['artifact']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=calibration['sha256']:
            raise ValueError('Probability admission calibration artifact changed')
        artifact=json.loads(raw)
        if (artifact.get('labels_role')!='dev_only' or artifact.get('test_labels_used') is not False
                or artifact.get('recall_threshold')!=config['reward']['recall_threshold']):
            raise ValueError('Probability admission calibration target/role mismatch')
        value=float(artifact['thresholds'][str(value)]['admission_probability_min'])
        if not float(config['max_ucb']['quality_probability_min'])<=value<=1:
            raise ValueError('Probability admission threshold must be at least the requested target')
    policy.quality_probability_min = value


def benchmark(args):
    """Shared-label, frozen, randomized method order; no test feedback updates."""
    from vec2plan_bandit.experiments import run as r
    config = r._runtime_config(args)
    training_cap = float(config.get('training_latency_cap_ms', args.latency_cap_ms))
    hard_deadline = bool(config.get("measurement", {}).get("hard_e2e_deadline", False))
    r.bind_run(args, config, "benchmark")
    records, split = r._queries(args.query_source), r._read_json(args.split_json)
    if "train" in args.roles:
        raise ValueError("benchmark must use held-out dev/test queries")
    from vec2plan_bandit.experiments.configuration import checkpoint_only_baselines
    baseline_config = r.load_configuration(args.baseline_config)
    r.validate_baseline_config(baseline_config)
    frozen_only = bool(getattr(args, "checkpoint", None) and checkpoint_only_baselines(baseline_config))
    fit_started = time.perf_counter()
    if frozen_only:
        stats = r._read_json(args.query_statistics) if getattr(args, "query_statistics", None) else None
        embeddings, observations = None, []
        feature_fit_ms = 0.0
    else:
        stats, embeddings = r._feature_inputs(args)
        pipeline, observations = r.build_training_observations(records, split, r._read_jsonl(args.training_matrix),
            latency_cap_ms=training_cap, recall_threshold=float(config["reward"]["recall_threshold"]), statistics=stats, embeddings=embeddings)
        feature_fit_ms = (time.perf_counter() - fit_started) * 1000
    fit_started = time.perf_counter()
    if getattr(args, "checkpoint", None):
        pipeline, policy, _ = r.load_bundle(args.checkpoint, cap_ms=training_cap, train_ids=split["train"],
            expected={"training_source_hashes": {q.query_id: q.source_hash for q in records if q.query_id in split["train"]},
                      "snapshot_id": config.get("snapshot_id"), "embedding": config.get("embedding"),
                "recall_threshold": float(config["reward"]["recall_threshold"])},
            expected_policy=config.get("max_ucb", {}) if frozen_only else None)
    else:
        policy, _ = r._fit_policy(observations, config)
    training_times = {"shared_feature_preparation": feature_fit_ms,
                      "cash_maxucb": (time.perf_counter() - fit_started) * 1000}
    if getattr(args, "checkpoint", None):
        training_times["cash_maxucb_checkpoint_load"] = training_times["cash_maxucb"]
        training_times["cash_maxucb"] = 0.0  # No main-model fit in this evaluation.
    configure_evaluation_probability(policy, config)
    from vec2plan_bandit.experiments.main_inference import configure_main_inference
    configure_main_inference(policy, config, args.output_dir, checkpoint=getattr(args,'checkpoint',None))
    main_method = str(config.get('main_method_id', 'cash_maxucb'))
    registry = r.builtin_baselines()
    baselines = {}
    frozen_controls = {}
    independent_training = False
    for spec in baseline_config["baselines"]:
        fit_started = time.perf_counter()
        method_observations = observations
        if spec.get("training_matrix"):
            from pathlib import Path
            independent_training = True
            _, method_observations = r.build_training_observations(records, split,
                r._read_jsonl(Path(spec["training_matrix"])), latency_cap_ms=args.latency_cap_ms,
                statistics=stats, embeddings=embeddings, recall_threshold=float(config["reward"]["recall_threshold"]))
            if len(method_observations) != len(observations):
                raise ValueError("independently trained methods must have equal labeled execution budgets")
        if spec["name"] == "frozen-policy":
            from pathlib import Path
            options = spec["config"]
            control_pipeline, control_policy, _ = r.load_bundle(Path(options["checkpoint"]),
                cap_ms=args.latency_cap_ms, train_ids=split["train"], expected_policy=options["max_ucb"],
                expected={"training_source_hashes": {q.query_id: q.source_hash for q in records if q.query_id in split["train"]},
                          "snapshot_id": config.get("snapshot_id"), "embedding": config.get("embedding"),
                          "recall_threshold": float(config["reward"]["recall_threshold"])})
            frozen_controls[spec["id"]] = (control_pipeline, control_policy)
            baselines[spec["id"]] = None
        elif spec["name"] == "text2vectorsql":
            from vec2plan_bandit.baselines.text2vectorsql import Text2VectorSQLTranslator
            options = dict(spec.get("config", {}))
            if hard_deadline:
                # Disconnect streaming HTTP at expiry; MLX closes generation
                # via its response context instead of continuing a full answer.
                options["stream"] = True
            baselines[spec["id"]] = Text2VectorSQLTranslator(**options)
        else:
            options = dict(spec.get("config", {}))
            if spec["name"] == "jev-router":
                options["recall_threshold"] = float(config["reward"]["recall_threshold"])
            baselines[spec["id"]] = None if spec["name"] == "native-pg" else registry.create(spec["name"], options).fit(method_observations)
        training_times[spec["id"]] = (time.perf_counter() - fit_started) * 1000
    if config.get('preserve_baseline_prediction_engine', False):
        for baseline in baselines.values():
            if hasattr(baseline, 'path_surrogate'):
                baseline.path_surrogate.use_shared_pool = False
    if main_method != 'cash_maxucb':
        training_times[main_method] = training_times.pop('cash_maxucb')
        if 'cash_maxucb_checkpoint_load' in training_times:
            training_times[main_method + '_checkpoint_load'] = training_times.pop('cash_maxucb_checkpoint_load')
    (args.output_dir / 'baseline_fit_audit.json').write_text(json.dumps({
        method: (baseline.fit_audit if hasattr(baseline, 'fit_audit') else
            {'selectivity_threshold': baseline.threshold, 'paired_train_queries': baseline.training_pairs,
             'training_labels': 'shared train only'})
        for method, baseline in baselines.items()
        if getattr(baseline, 'name', None) in {'dynamic-ann', 'selectivity-switch'}
    }, indent=2))
    (args.output_dir / "model_fit_times_ms.json").write_text(json.dumps(training_times, indent=2))
    (args.output_dir / "effective_policies.json").write_text(json.dumps({
        main_method: policy.state_dict(),
        **{method: item[1].state_dict() for method, item in frozen_controls.items()},
    }, indent=2))
    (args.output_dir / "evaluation_setup.json").write_text(json.dumps({
        "frozen_only": frozen_only, "shared_training_features_rebuilt": not frozen_only,
        "new_main_model_fit": not bool(getattr(args, "checkpoint", None))}, indent=2))
    if main_method in baselines:
        raise ValueError(f'{main_method} is reserved for the main method')
    queries = r._selected_queries(records, split, args.roles)
    if getattr(args, "query_limit", None) is not None:
        queries = sorted(queries, key=lambda q: q.query_id)[:args.query_limit]
    truth = r._ground_truth(args.ground_truth_jsonl)
    if not queries or any(q.query_id not in truth for q in queries):
        raise ValueError("benchmark requires nonempty held-out queries with ground truth")
    methods = [main_method, *baselines]
    worker = r._collector(args.output_dir, config)
    jev_router = r.jev_router_from_config(config.get("jev", {}))
    query_input = config.get("query_input", {})
    structurer = None
    if query_input.get("contract") == "question_string_only":
        if not hard_deadline:
            raise ValueError("question-only evaluation requires the complete-request deadline boundary")
        if set(methods) != {main_method, "maxucb_bo", "mlp", "lightgbm"}:
            raise ValueError("question-only comparison requires exactly the four requested selectors")
        if jev_router is not None:
            raise ValueError("question-only comparison uses the main method without JEV")
        from vec2plan_bandit.queries.natural_language import NaturalLanguageStructurer
        structurer = NaturalLanguageStructurer(**query_input["structurer"])
        if policy.quality_probability_min != float(config["max_ucb"]["quality_probability_min"]):
            raise ValueError("checkpoint quality probability does not match question-only configuration")
        # Persistent service assets, not per-query results, are loaded before steady-state timing.
        ready = worker.path_router.available(probe=True)
        if not all(ready.get(path, False) for path in config["require_paths"]):
            raise RuntimeError(f"question-only backend startup failed: {ready}")
        worker.embed("synthetic service warmup")
        # The metadata reservoir is a corpus-only persistent asset. Its initial
        # full snapshot scan must not be charged to whichever selector runs first.
        from vec2plan_bandit.queries.natural_language import query_from_fields
        worker.estimate_statistics(query_from_fields("synthetic statistics warmup", {
            "s": "synthetic statistics warmup", "e": "abstract", "k": 1, "p": []}))
        (args.output_dir / "runtime_startup.json").write_text(json.dumps({
            "paths": ready, "faiss_columns_preloaded": list(config["faiss_assets"]),
            "query_independent_metadata_reservoir_preloaded": True,
            "latency_scope": "steady-state request; persistent model/index startup excluded",
            "query_result_cache": False, "query_structuring_cache": False}, indent=2), encoding="utf-8")
    store = r.JsonlResultStore(args.output_dir / "benchmark_live.jsonl")
    startup=preload_benchmark_assets(worker,config,queries)
    if startup['exact_metadata_ready']:
        (args.output_dir/'exact_metadata_startup.json').write_text(json.dumps(startup,indent=2)+'\n')
    def execute(method, query):
        if structurer is not None:
            from vec2plan_bandit.execution.natural_language_runtime import run_question_request
            # Only this string crosses the evaluation-to-inference boundary.
            row = run_question_request(question=query.question, structurer=structurer,
                collector=worker, feature_pipeline=pipeline, latency_cap_ms=args.latency_cap_ms,
                policy=policy if method == main_method else None,
                baseline=baselines[method] if method != main_method else None)
            return row
        if method == main_method:
            return r.run_adaptive_request(query=query, collector=worker, feature_pipeline=pipeline,
                policy=policy, latency_cap_ms=args.latency_cap_ms, statistics=stats,
                exploration=False, jev_router=jev_router)
        if method in frozen_controls:
            control_pipeline, control_policy = frozen_controls[method]
            return r.run_adaptive_request(query=query, collector=worker, feature_pipeline=control_pipeline,
                policy=control_policy, latency_cap_ms=args.latency_cap_ms, statistics=stats,
                exploration=False)
        if baselines[method] is None:
            return r.run_native_baseline_request(query=query, collector=worker, latency_cap_ms=args.latency_cap_ms)
        from vec2plan_bandit.baselines.text2vectorsql import Text2VectorSQLTranslator, run_text2vectorsql_request
        if isinstance(baselines[method], Text2VectorSQLTranslator):
            return run_text2vectorsql_request(query=query, collector=worker,
                translator=baselines[method], latency_cap_ms=args.latency_cap_ms)
        return r.run_selector_baseline_request(query=query, collector=worker, feature_pipeline=pipeline,
            baseline=baselines[method], latency_cap_ms=args.latency_cap_ms, statistics=stats)
    state = policy.state_dict()
    control_states = {method: item[1].state_dict() for method, item in frozen_controls.items()}
    def measure(method, query):
        call = lambda: _outcome(lambda: execute(method, query), query, args.latency_cap_ms)
        if not hard_deadline:
            return call()
        from vec2plan_bandit.execution.timing import run_with_deadline
        row = run_with_deadline(call, query, args.latency_cap_ms,
                                getattr(worker, "cancel_pending_request", None))
        if row.get("timed_out"):
            cleanup = getattr(worker, "finish_cancelled_request", None)
            if cleanup is not None:
                cleanup()
        translator = baselines.get(method)
        if structurer is not None:
            # Gold k and predicates enter only the evaluator, after retrieval/timing.
            row.update(query_id=query.query_id, k=query.k,
                       input_contract="question_string_only", structure_audit=structurer.last_audit)
            parsed = structurer.last_audit.get("structured_query", {})
            row["structuring_success"] = structurer.last_audit.get("status") == "ok"
            row["parsed_k_correct"] = parsed.get("k") == query.k
            row["parsed_embedding_column_correct"] = parsed.get("embedding_column") == query.embedding_column
            def canonical(predicates):
                return sorted((str(p["type"]), tuple(sorted(p["values"]))) for p in predicates)
            row["parsed_predicates_correct"] = (row["structuring_success"] and
                canonical(parsed.get("predicates", [])) == canonical(query.predicates))
            row["parsed_metadata_exact"] = all(row[key] for key in (
                "structuring_success", "parsed_k_correct", "parsed_embedding_column_correct", "parsed_predicates_correct"))
            row["result_full_k"] = len(set(row.get("article_ids", [])[:query.k])) == query.k
        if method in baselines and hasattr(translator, "last_partial_sql"):
            row.setdefault("input_contract", "natural_language_plus_requested_k")
            if row.get("timed_out"):
                row["translation_audit"] = {"generated_sql_partial": translator.last_partial_sql,
                    "endpoint": translator.url, "requested_model": translator.model,
                    "model_identity": translator.model_identity, "model_revision": translator.model_revision}
        return row
    try:
        for repeat in range(1, args.repeats + 1):
            ordered = list(queries)
            random.Random(f"{getattr(args, 'seed', 42)}:{repeat}:queries").shuffle(ordered)
            for query in ordered:
                order = list(methods)
                random.Random(f"{getattr(args, 'seed', 42)}:{repeat}:{query.query_id}").shuffle(order)
                for method in order:
                    if store.contains(method, query.query_id, repeat):
                        continue
                    for _ in range(int(config.get("measurement", {}).get("warmup_runs", 0))):
                        measure(method, query)
                    row = measure(method, query)
                    row = r._adjudicate(row, truth, args.latency_cap_ms, float(config["reward"]["recall_threshold"]))
                    row.setdefault("input_contract", "parsed_query")
                    row["comparison_scope"] = ("natural_language_question_only_e2e"
                        if row["input_contract"] == "question_string_only"
                        else "natural_language_e2e_extension"
                        if row["input_contract"] == "natural_language_plus_requested_k"
                        else "parsed_query_execution")
                    row.update(method_id=method, repeat=repeat, requested_budget_ms=args.latency_cap_ms,
                               quality_probability_min=float(config.get("max_ucb", {}).get("quality_probability_min", 0.8)),
                               evaluation_mode="frozen_equal_budget" if independent_training else "frozen_shared_labels", seed=getattr(args, "seed", 42),
                               recorded_at=datetime.now(timezone.utc).isoformat())
                    if method==main_method:
                        row['policy_admission_probability_min']=policy.quality_probability_min
                    store.append(row)
                    if structurer is not None and (len(store.completed) == 1 or len(store.completed) % 25 == 0):
                        progress = {"completed_requests": len(store.completed),
                            "total_requests": len(queries) * args.repeats * len(methods),
                            "last_method": method, "last_query_id": query.query_id,
                            "last_repeat": repeat, "updated_at": datetime.now(timezone.utc).isoformat()}
                        temporary = args.output_dir / "progress.tmp"
                        temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                        temporary.replace(args.output_dir / "progress.json")
        if any(control_states[m] != p.state_dict() for m, (_, p) in frozen_controls.items()):
            raise ValueError("frozen control learned during evaluation")
        if state != policy.state_dict():
            raise AssertionError("frozen policy changed during benchmark")
    finally:
        if jev_router is not None:
            jev_router.close()
        for baseline in baselines.values():
            close = getattr(baseline, "close", None)
            if close is not None:
                close()
        worker.close()
    rows = store.read()
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps({method: r._metric_summary(
        [row for row in rows if row["method_id"] == method]) for method in methods}, indent=2))
    if structurer is not None:
        summaries = {}
        for method in methods:
            measured = [row for row in rows if row["method_id"] == method]
            stages = sorted({key for row in measured for key in row.get("spans_ms", {})})
            summaries[method] = {
                "requests": len(measured),
                **{key + "_rate": float(np.mean([row[key] for row in measured]))
                   for key in ("structuring_success", "parsed_k_correct", "parsed_embedding_column_correct",
                               "parsed_predicates_correct", "parsed_metadata_exact")},
                "all_request_stage_mean_ms": {stage: float(np.mean([
                    row.get("spans_ms", {}).get(stage, 0.0) for row in measured])) for stage in stages},
                "all_request_mean_recall_at_k": float(np.mean([row["recall_at_k"] for row in measured])),
                "timeout_rate": float(np.mean([row.get("timed_out", False) for row in measured])),
                "error_rate": float(np.mean([bool(row.get("error_type")) for row in measured])),
            }
        (args.output_dir / "question_only_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
