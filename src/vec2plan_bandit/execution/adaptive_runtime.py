from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vec2plan_bandit.baselines.base import Baseline, BaselineRequest
from vec2plan_bandit.baselines.maxucb_bo import JevPathRouter, JevRoutingError
from vec2plan_bandit.baselines.native_pg import render_native_pg
from vec2plan_bandit.execution.timing import RequestTimer, evaluation_deadline_ns, record_request_audit
from vec2plan_bandit.experiments.collector import MatrixCollector, _vector_literal
from vec2plan_bandit.features.feature_pipeline import FeaturePipeline
from vec2plan_bandit.features.action_features import joint_action_features
from vec2plan_bandit.paths.registry import ACORN, FAISS, ExecutionConfig, make_execution_config, pre_configuration_key
from vec2plan_bandit.paths.sql import RenderedSQL
from vec2plan_bandit.policies.max_ucb import MaxUCB
from vec2plan_bandit.queries.loader import QueryRecord


def _unique_ids(values: list[int], k: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        article_id = int(value)
        if article_id not in seen:
            result.append(article_id)
            seen.add(article_id)
        if len(result) >= k:
            break
    return result


def _statistics(collector: Any, query: QueryRecord, supplied: Any) -> Any:
    estimator = getattr(collector, "estimate_statistics", None)
    estimated = estimator(query) if estimator is not None else None
    return {query.query_id: estimated} if estimated is not None else supplied


def _available_paths(collector: Any, query: QueryRecord) -> dict[str, bool]:
    """Operational readiness plus the method's semantic applicability mask."""
    available = dict(collector.path_router.available(probe=False, query=query))
    # ACORN is a filtered-ANN algorithm. For an unfiltered query it adds graph
    # and remote-service cost without providing filtering benefit.
    if not query.predicates:
        available[ACORN] = False
    return available


def _available_pre_configurations(collector: Any, query: QueryRecord, vector_literal: str) -> frozenset[str] | None:
    auditor = getattr(collector, "available_pre_configurations", None)
    return auditor(query, vector_literal) if auditor is not None else None


def _validate_main_decision_scope(config: Mapping[str, Any], decision: Any) -> str:
    """A native PG early exit belongs only to the calibrated small model."""
    allowed=config.get('main_execution',{}).get('allowed_paths')
    small_pg=(decision.config.path_id=='PG_NATIVE_SQL'
        and config.get('main_execution',{}).get('allow_small_model_pg',False)
        and config.get('main_lightweight_selector',{}).get('enabled',False)
        and config.get('main_lightweight_selector',{}).get('native_pg_early',False)
        and getattr(decision,'selection_mode',None)=='lightweight')
    if allowed is not None and decision.config.path_id not in allowed and not small_pg:
        raise ValueError('Selected path is outside the configured main execution scope: '+decision.config.path_id)
    return 'small_model_pg' if small_pg else 'main_model'


def run_adaptive_request(
    *,
    query: QueryRecord,
    collector: MatrixCollector,
    feature_pipeline: FeaturePipeline,
    policy: MaxUCB,
    latency_cap_ms: float,
    statistics: Mapping[str, Mapping[str, Any]] | None = None,
    exploration: bool = True,
    jev_router: JevPathRouter | None = None,
    request_timer: RequestTimer | None = None,
    compile_all_paths: bool = True,
    acceleration_router: Any = None,
    _embedded_vector: Any = None,
    _disable_acceleration: bool = False,
    _remaining_budget_ms: float | None = None,
    _reuse_statistics: bool = False,
) -> dict[str, Any]:
    """Run one real request: embed, choose one configuration, execute, return IDs."""
    runtime_config = getattr(collector, 'config', {})
    allowed=runtime_config.get('main_execution',{}).get('allowed_paths')
    if allowed is not None:
        if not allowed or not set(allowed).issubset({'PRE','POST','FAISS','ACORN'}):
            raise ValueError('Main execution scope supports only PRE/POST/FAISS/ACORN')
        if runtime_config.get('main_recall_guard',{}).get('enabled',False):
            raise ValueError('Exact/profile guard is outside the configured four-path scope')
    if not exploration and not _disable_acceleration:
        from vec2plan_bandit.policies.optional_acceleration import configured_acceleration, describe_query
        from vec2plan_bandit.execution.optional_acceleration import accelerated_request
        router = acceleration_router or configured_acceleration(runtime_config)
        if router is not None:
            routing_description = None
            routing_statistics = None
            def description():
                nonlocal routing_description, routing_statistics
                current = _statistics(collector, query, statistics)
                routing_statistics = current
                routing_description = describe_query(query, (current or {}).get(query.query_id),
                    corpus_size=runtime_config.get('optional_acceleration', {}).get('corpus_size_by_column',{}).get(
                        query.embedding_column,runtime_config.get('optional_acceleration', {}).get('corpus_size')))
                return routing_description

            def execute_fast(rule, vector):
                config = make_execution_config(rule['path_id'], rule['params']) if rule['path_id'] != 'PG_NATIVE_SQL' else None
                literal = _vector_literal(vector)
                if config is None:
                    ids, _ = collector._pg(render_native_pg(query), literal)
                    details = {}
                else:
                    compiled = collector.compile(query) if compile_all_paths and config.path_id in {'PRE', 'POST'} else None
                    ids, _, details = collector.execute(config.action_id, query, compiled, vector, literal,
                        config, deadline_ns=evaluation_deadline_ns())
                return {'article_ids': _unique_ids(ids, query.k), 'execution_details': details}

            def configurations(vector):
                selected = router.rules.get((routing_description or {}).get('key'))
                if selected is None or selected['path_id'] != 'PRE':
                    return None
                eligible = _available_pre_configurations(collector, query, _vector_literal(vector))
                if eligible is not None and pre_configuration_key(selected['params']) in eligible:
                    return {selected['config_id']}
                return set()

            def paths():
                available = dict(collector.path_router.available(probe=False, query=query))
                available['PG_NATIVE_SQL'] = hasattr(collector, '_pg') and (available.get('PRE', False) or available.get('POST', False))
                return available

            return accelerated_request(query=query, router=router, description=description,
                available_paths=paths,
                available_configurations=configurations, embed=lambda: collector.embed(query.semantic_text)[0],
                execute_fast=execute_fast, request_main=lambda vector, remaining: run_adaptive_request(
                    query=query, collector=collector, feature_pipeline=feature_pipeline, policy=policy,
                    latency_cap_ms=latency_cap_ms, statistics=routing_statistics, exploration=False,
                    jev_router=jev_router, compile_all_paths=compile_all_paths,
                    _embedded_vector=vector, _disable_acceleration=True, _remaining_budget_ms=remaining,
                    _reuse_statistics=True),
                latency_cap_ms=latency_cap_ms, id_field='article_ids', timer=request_timer,
                recall_threshold=runtime_config.get('reward', {}).get('recall_threshold'),
                budget_faiss_threshold_ms=(runtime_config.get('budget_faiss', {}).get('threshold_ms', 100.)
                    if runtime_config.get('budget_faiss', {}).get('enabled', False) else None),
                cancel_pending=getattr(collector, 'cancel_pending_request', None))
    rule = runtime_config.get('rule_based_routing', {})
    if not _disable_acceleration and rule.get('unfiltered_to_faiss', False) and not query.predicates and not exploration:
        if not collector.path_router.available(probe=False, query=query).get(FAISS, False):
            raise RuntimeError('Unfiltered FAISS routing requires an operational FAISS path')
        config = make_execution_config(FAISS, rule.get('faiss_params',
            {'ef_search': 100, 'candidate_multiplier': 4.0}))
        record_request_audit(routing_rule='unfiltered_to_faiss', rule_based_routing=True,
                             main_pipeline_stage='rule_based_faiss')
        row = run_fixed_configuration_request(query=query, collector=collector,
            execution_config=config, latency_cap_ms=latency_cap_ms,
            request_timer=request_timer, compile_all_paths=False)
        row.update(routing_rule='unfiltered_to_faiss', rule_based_routing=True,
                   policy_version='arxiv-rule-faiss-v1', exploration=False, jev=None,
                   main_pipeline_stage='rule_based_faiss')
        return row
    timer = request_timer if request_timer is not None else RequestTimer(
        max(.001, _remaining_budget_ms) if _remaining_budget_ms is not None else latency_cap_ms)
    if _embedded_vector is None:
        with timer.span("embedding_ms"):
            vector, _service_reported_ms = collector.embed(query.semantic_text)
    else:
        vector = _embedded_vector
    cpu_guard=runtime_config.get('main_recall_guard',{})
    if not exploration and cpu_guard.get('enabled',False) and cpu_guard.get('profile_early_gate',False):
        from vec2plan_bandit.execution.main_recall_guard import run_profile_gate
        return run_profile_gate(collector,query,vector,_vector_literal(vector),timer)
    if not exploration and cpu_guard.get('enabled',False) and cpu_guard.get('cpu_exact_scan',False):
        if (runtime_config.get('reward',{}).get('recall_threshold')!=1.
                or not runtime_config.get('snapshot_id','').startswith('vectorsqlbench-arxiv-')):
            raise ValueError('Strict mean-recall CPU execution is restricted to arXiv recall=1')
        from vec2plan_bandit.execution.main_recall_guard import execute_exact_cpu
        record_request_audit(path_id='MAIN_EXACT',config_id='main-exact-cpu-subset-v1',
                             main_selection_mode='strict_mean_recall_cpu',
                             main_selection={'reason':'mean_recall_one_requires_complete_recall'},
                             requested_params={},policy_features=[],query_context=[])
        ids=[];details={};errors={}
        try:
            with timer.span('execution_ms'):
                ids,details=execute_exact_cpu(collector,query,vector,_vector_literal(vector))
            with timer.span('postprocess_ms'):
                ids=_unique_ids(ids,query.k)
        except Exception as error:
            errors=dict(error_type=type(error).__name__,error_message=str(error)[:500])
        finally:
            record_request_audit(spans_ms=dict(timer.spans_ms),execution_ms=timer.spans_ms.get('execution_ms'))
        return dict(query_id=query.query_id,k=query.k,article_ids=ids,path_id='MAIN_EXACT',action_id=None,
                    config_id='main-exact-cpu-subset-v1',requested_params={},execution_details=details,
                    main_selection_mode='strict_mean_recall_cpu',policy_version='strict-mean-recall-cpu-v1',
                    fallback_used=False,result_full_k=len(ids)>=query.k,result_fill_count=len(ids),
                    **timer.finish(),**errors)
    with timer.span("feature_ms"):
        query_statistics = statistics if _reuse_statistics else _statistics(collector, query, statistics)
        embeddings = {query.query_id: vector} if feature_pipeline.embedding_enabled_ else None
        context = feature_pipeline.transform_one(
            query, statistics=query_statistics, embeddings=embeddings
        )
    with timer.span("sql_compile_ms"):
        compiled = collector.compile(query) if compile_all_paths else None
        vector_literal = _vector_literal(vector)
    with timer.span("policy_ms"):
        availability = _available_paths(collector, query)
        required_path=runtime_config.get('main_execution',{}).get('required_path')
        if required_path is not None:
            if required_path not in ('PRE','POST','FAISS','ACORN'):
                raise ValueError('Required path is outside PRE/POST/FAISS/ACORN')
            availability={p:bool(ready and p==required_path) for p,ready in availability.items()}
        pre_configurations = _available_pre_configurations(collector, query, vector_literal)
        jev_decision = None
        jev_error: dict[str, Any] | None = None
        jev_fallback = False
        if jev_router is not None:
            try:
                jev_decision = jev_router.route(BaselineRequest(
                    query=query,
                    query_context=context,
                    available_paths=availability,
                    latency_cap_ms=latency_cap_ms,
                    remaining_budget_ms=timer.remaining_ms,
                    available_pre_configurations=pre_configurations,
                    query_statistics=(query_statistics or {}).get(query.query_id),
                ))
            except JevRoutingError as error:
                jev_error = dict(error.audit)
                if jev_router.failure_policy != "no_prior":
                    raise
                jev_fallback = True
        selection_options = {}
        if runtime_config.get('main_training',{}).get('enabled',False):
            from vec2plan_bandit.experiments.main_inference import runtime_routing_facts
            facts=runtime_routing_facts(query,(query_statistics or {}).get(query.query_id),collector,runtime_config)
            selection_options['request_facts']=facts
            record_request_audit(main_routing_facts=facts)
        if getattr(policy,'collect_main_routing_facts',False):
            from vec2plan_bandit.experiments.main_inference import main_request_options
            mapping = getattr(collector,'article_id_mappings',{}).get(query.embedding_column)
            selection_options = main_request_options(policy,query,(query_statistics or {}).get(query.query_id),
                corpus_size=len(mapping.values) if mapping is not None else None)
        if getattr(policy,'main_lightweight',None) is not None:
            selection_options['native_pg_available'] = (callable(getattr(collector,'_pg',None))
                and (availability.get('PRE',False) or availability.get('POST',False)))
        if runtime_config.get('main_parameter_tuning',{}).get('enabled',False):
            from vec2plan_bandit.policies.parameter_profiles import keys as profile_keys
            selection_options['parameter_profile_keys']=tuple(profile_keys(query,selection_options.get('request_facts')))
        decision = policy.select(
            context,
            request_key=query.query_id,
            latency_cap_ms=latency_cap_ms,
            remaining_budget_ms=timer.remaining_ms,
            available=availability,
            available_pre_configurations=pre_configurations,
            exploration=exploration,
            path_priors=jev_decision.probabilities if jev_decision else None,
            path_prior_weight=jev_router.path_prior_weight if jev_router else 0.0,
            **selection_options,
        )
        guard_options = runtime_config.get('main_recall_guard', {})
        guarded = guard_options.get('enabled', False) and not exploration
        if guarded:
            if (runtime_config.get('reward', {}).get('recall_threshold') != 1.
                    or not runtime_config.get('snapshot_id', '').startswith('vectorsqlbench-arxiv-')):
                raise ValueError('Recall guard requires the arXiv recall-one protocol')
            from vec2plan_bandit.execution.main_recall_guard import load_guard, guard_decision
            artifact = load_guard(guard_options['artifact'], guard_options['sha256'])
            decision, evidence = guard_decision(decision, query,
                runtime_config['max_ucb']['quality_probability_min'], artifact, availability)
            record_request_audit(main_recall_guard=evidence,
                                 model_prediction_scope='original_configuration_before_guard')
    error_details: dict[str, Any] = {}
    pipeline_stage=_validate_main_decision_scope(runtime_config,decision)
    details: dict[str, Any] = {}
    article_ids: list[int] = []
    record_request_audit(path_id=decision.config.path_id, action_id=decision.config.action_id,
                         config_id=decision.config.config_id, requested_params=dict(decision.config.params),
                         policy_features=list(decision.selected_features), query_context=context.tolist(),
                         main_routing_facts=selection_options.get('request_facts'), exploration=exploration,
                         main_selection_mode=getattr(decision,'selection_mode','full_maxucb'),
                         main_selection=dict(getattr(decision,'selection_evidence',None) or {}),
                         constraints_satisfied=decision.constraints_satisfied,
                         policy_fallback_reason=decision.fallback_reason,
                         main_pipeline_stage=pipeline_stage)
    try:
        if decision.config.path_id == 'PG_NATIVE_SQL':
            with timer.span('native_sql_compile_ms'):
                native_sql = render_native_pg(query)
        with timer.span("execution_ms"):
            if decision.config.path_id == 'MAIN_EXACT':
                from vec2plan_bandit.execution.main_recall_guard import execute_exact
                ids, details = execute_exact(collector, query, vector, vector_literal,
                    selection_options.get('request_facts'))
            elif decision.config.path_id == 'PG_NATIVE_SQL':
                ids, _path_ms = collector._pg(native_sql, vector_literal)
                details = dict(implementation='PG_NATIVE_SQL',fallback_used=False,
                    main_selection_mode=decision.selection_mode)
            else:
                if runtime_config.get('main_execution',{}).get('adaptive_fill',False):
                    from vec2plan_bandit.execution.main_adaptive_fill import execute_with_fill
                    def execute_candidate(c):
                        values,_,extra=collector.execute(c.action_id,query,compiled,vector,vector_literal,c,
                            deadline_ns=evaluation_deadline_ns())
                        return {'article_ids':values,'execution_details':extra}
                    filled=execute_with_fill(execute_candidate,decision.config,k=query.k,id_field='article_ids',
                        remaining_ms=lambda:timer.remaining_ms,facts=selection_options.get('request_facts'),enabled=True)
                    ids=filled['article_ids'];details={**filled.get('execution_details',{}),
                        **{k:v for k,v in filled.items() if k not in ('article_ids','execution_details')}}
                else:
                    ids, _path_ms, details = collector.execute(
                        decision.config.action_id,query,compiled,vector,vector_literal,decision.config,
                        deadline_ns=evaluation_deadline_ns())
            if guarded and decision.config.path_id != 'MAIN_EXACT' and len(set(ids)) < query.k:
                from vec2plan_bandit.execution.main_recall_guard import execute_exact
                original_details = details
                ids, details = execute_exact(collector, query, vector, vector_literal,
                    selection_options.get('request_facts'))
                details.update(underfilled_profile=original_details,
                               actual_execution_path='MAIN_EXACT', exact_fallback=True)
        with timer.span("postprocess_ms"):
            article_ids = _unique_ids(ids, query.k)
    except Exception as error:
        # Preserve the chosen path and features so failed executions provide
        # negative feedback rather than disappearing from the online history.
        error_details = {"error_type": type(error).__name__, "error_message": str(error)[:500]}
    finally:
        # The hard deadline raises BaseException and bypasses the normal return.
        # Publish closed spans while unwinding, without converting cancellation
        # into a successful result or dropping the selected action's feedback.
        record_request_audit(spans_ms=dict(timer.spans_ms),
                             execution_ms=timer.spans_ms.get('execution_ms'))
    timing = timer.finish()
    return {
        "query_id": query.query_id,
        "k": query.k,
        "article_ids": article_ids,
        "path_id": decision.config.path_id,
        "action_id": decision.config.action_id,
        "config_id": decision.config.config_id,
        "main_selection_mode": getattr(decision,'selection_mode','full_maxucb'),
        "main_pipeline_stage": pipeline_stage,
        "main_selection": dict(getattr(decision,'selection_evidence',None) or {}),
        "main_routing_facts":selection_options.get('request_facts'),
        "requested_params": dict(decision.config.params),
        "execution_details": details,
        "policy_version": decision.policy_version,
        "policy_score": decision.score,
        "predicted_quality_probability": decision.predicted_quality_probability,
        "quality_uncertainty": decision.quality_uncertainty,
        "predicted_e2e_latency_ms": decision.predicted_e2e_latency_ms,
        "latency_log_uncertainty": decision.latency_log_uncertainty,
        "quality_lcb": decision.quality_lcb,
        "quality_ucb": decision.quality_ucb,
        "latency_lcb_ms": decision.latency_lcb_ms,
        "latency_ucb_ms": decision.latency_ucb_ms,
        "constraints_satisfied": decision.constraints_satisfied,
        "policy_fallback_reason": decision.fallback_reason,
        "feasible_candidate_count": decision.feasible_candidate_count,
        "best_config_by_path": decision.best_by_path,
        "evaluated_candidates": decision.evaluated_candidates,
        "decision_id": decision.decision_id,
        "round_index": decision.round_index,
        "path_counts": decision.path_counts,
        "scores_by_path": decision.scores_by_path,
        "candidate_seeds": decision.candidate_seeds,
        "policy_features": list(decision.selected_features),
        "query_context": context.tolist(),
        "exploration": exploration,
        "jev": ({
            "chosen_path": jev_decision.chosen_path,
            "path_probabilities": jev_decision.probabilities,
            "confidence": jev_decision.confidence,
            "request_id": jev_decision.request_id,
            "latency_ms": jev_decision.latency_ms,
            "model": jev_router.model,
            "resolved_model": jev_decision.resolved_model,
            "request_id_missing": jev_decision.request_id is None,
            "path_prior_weight": jev_router.path_prior_weight,
            "prompt_version": jev_decision.prompt_version,
            "state_sha256": jev_decision.state_sha256,
            "source": jev_decision.source,
            "effective_timeout_seconds": jev_decision.effective_timeout_seconds,
            "status": "ok",
        } if jev_decision is not None and jev_router is not None else jev_error),
        "jev_fallback_used": jev_fallback,
        "fallback_used": bool(jev_fallback or details.get("fallback_used", False)),
        "result_fill_count": len(article_ids),
        "result_full_k": len(article_ids) >= query.k,
        **timing,
        **error_details,
    }


def run_native_baseline_request(
    *,
    query: QueryRecord,
    collector: MatrixCollector,
    latency_cap_ms: float,
) -> dict[str, Any]:
    """Run native PostgreSQL as an external baseline with no policy cost."""
    timer = RequestTimer(latency_cap_ms)
    with timer.span("embedding_ms"):
        vector, _service_reported_ms = collector.embed(query.semantic_text)
    with timer.span("sql_compile_ms"):
        rendered = render_native_pg(query)
        vector_literal = _vector_literal(vector)
    with timer.span("execution_ms"):
        ids, _path_ms = collector._pg(rendered, vector_literal)
    with timer.span("postprocess_ms"):
        article_ids = _unique_ids(ids, query.k)
    timing = timer.finish()
    return {
        "query_id": query.query_id,
        "k": query.k,
        "article_ids": article_ids,
        "baseline_id": "PG_NATIVE_SQL",
        "result_fill_count": len(article_ids),
        "result_full_k": len(article_ids) >= query.k,
        **timing,
    }


def run_fixed_configuration_request(
    *,
    query: QueryRecord,
    collector: MatrixCollector,
    execution_config: ExecutionConfig,
    latency_cap_ms: float,
    request_timer: RequestTimer | None = None,
    compile_all_paths: bool = True,
) -> dict[str, Any]:
    """Collect one live E2E label for a fixed path configuration."""
    timer = request_timer if request_timer is not None else RequestTimer(latency_cap_ms)
    # Keep the attempted action even if embedding or statistics exhausts the budget.
    record_request_audit(path_id=execution_config.path_id, action_id=execution_config.action_id,
        config_id=execution_config.config_id, requested_params=dict(execution_config.params))
    with timer.span("embedding_ms"):
        vector, _service_reported_ms = collector.embed(query.semantic_text)
    with timer.span("sql_compile_ms"):
        compiled = collector.compile(query) if compile_all_paths else None
        vector_literal = _vector_literal(vector)
    facts=None
    if getattr(collector,'config',{}).get('main_training',{}).get('enabled',False):
        from vec2plan_bandit.experiments.main_inference import runtime_routing_facts
        with timer.span('main_statistics_ms'):
            facts=runtime_routing_facts(query,collector.estimate_statistics(query),collector,collector.config)
        record_request_audit(main_routing_facts=facts)
    with timer.span("execution_ms"):
        ids, _path_ms, details = collector.execute(
            execution_config.action_id,
            query,
            compiled,
            vector,
            vector_literal,
            execution_config,
            deadline_ns=evaluation_deadline_ns(),
        )
    with timer.span("postprocess_ms"):
        article_ids = _unique_ids(ids, query.k)
    timing = timer.finish()
    return {
        "query_id": query.query_id,
        "k": query.k,
        "article_ids": article_ids,
        "path_id": execution_config.path_id,
        "action_id": execution_config.action_id,
        "config_id": execution_config.config_id,
        "requested_params": dict(execution_config.params),
        "main_routing_facts":facts,
        "execution_details": details,
        "result_fill_count": len(article_ids),
        "result_full_k": len(article_ids) >= query.k,
        **timing,
    }


def run_selector_baseline_request(
    *,
    query: QueryRecord,
    collector: MatrixCollector,
    feature_pipeline: FeaturePipeline,
    baseline: Baseline,
    latency_cap_ms: float,
    statistics: Mapping[str, Mapping[str, Any]] | None = None,
    request_timer: RequestTimer | None = None,
    compile_all_paths: bool = True,
) -> dict[str, Any]:
    """Execute any registered selector with the same end-to-end timing boundary."""
    timer = request_timer if request_timer is not None else RequestTimer(latency_cap_ms)
    with timer.span("embedding_ms"):
        vector, _service_reported_ms = collector.embed(query.semantic_text)
    with timer.span("feature_ms"):
        query_statistics = _statistics(collector, query, statistics)
        if getattr(baseline, "requires_context", True):
            embeddings = {query.query_id: vector} if feature_pipeline.embedding_enabled_ else None
            context = feature_pipeline.transform_one(query, statistics=query_statistics, embeddings=embeddings)
        else:
            context = np.empty(0)
        selection_statistics = (query_statistics or {}).get(query.query_id)
        prepare_statistics = getattr(baseline, 'request_statistics', None)
        if prepare_statistics is not None:
            selection_statistics = prepare_statistics(collector, query, [vector], selection_statistics)
        execution_statistics = getattr(baseline, "execution_statistics", None)
        if execution_statistics is not None:
            selection_statistics = dict(selection_statistics or {})
            selection_statistics.update(execution_statistics(collector, query))
    try:
        with timer.span("policy_ms"):
            # Pure controls must actually execute their declared path, including
            # ACORN for unfiltered queries. Only adaptive selectors apply the
            # optimization-specific unfiltered-ACORN applicability mask.
            availability = (dict(collector.path_router.available(probe=False, query=query))
                            if not getattr(baseline, "requires_context", True)
                            else _available_paths(collector, query))
            pre_configurations = _available_pre_configurations(collector, query, _vector_literal(vector))
            config = baseline.select(BaselineRequest(
                query=query,
                query_context=context,
                available_paths=availability,
                latency_cap_ms=latency_cap_ms,
                remaining_budget_ms=timer.remaining_ms,
                available_pre_configurations=pre_configurations,
                query_statistics=selection_statistics,
            ))
    except JevRoutingError as error:
        # Preserve JEV service/replay failures as first-class benchmark rows;
        # the outer generic catcher cannot recover baseline audit metadata.
        timing = timer.finish()
        audit = dict(error.audit)
        return {
            "query_id": query.query_id,
            "k": query.k,
            "article_ids": [],
            "baseline_id": baseline.name,
            "path_id": None,
            "jev": audit,
            "jev_error": audit,
            "jev_fallback_used": False,
            "fallback_used": False,
            "result_fill_count": 0,
            "result_full_k": False,
            "error_type": "JevRoutingError",
            "error_message": str(error)[:500],
            **timing,
        }
    with timer.span("sql_compile_ms"):
        compiled = collector.compile(query) if compile_all_paths else None
        vector_literal = _vector_literal(vector)
    article_ids, details, error_details = [], {}, {}
    record_request_audit(path_id=config.path_id, action_id=config.action_id,
                         config_id=config.config_id, requested_params=dict(config.params),
                         baseline_id=baseline.name, **getattr(baseline, "last_selection_metadata", {}))
    try:
        with timer.span("execution_ms"):
            execute_request = getattr(baseline, "execute_request", None)
            if execute_request is not None:
                ids, _path_ms, details = execute_request(
                    collector=collector, query=query, compiled=compiled, vector=vector,
                    vector_literal=vector_literal, execution_config=config,
                    deadline_ns=evaluation_deadline_ns())
            else:
                ids, _path_ms, details = collector.execute(
                    config.action_id, query, compiled, vector, vector_literal, config,
                    deadline_ns=evaluation_deadline_ns(),
                )
        with timer.span("postprocess_ms"):
            article_ids = _unique_ids(ids, query.k)
    except Exception as error:
        error_details = {"error_type": type(error).__name__, "error_message": str(error)[:500]}
        details.update(getattr(baseline, "last_execution_details", {}))
    timing = timer.finish()
    return {
        "query_id": query.query_id,
        "k": query.k,
        "article_ids": article_ids,
        "baseline_id": baseline.name,
        "path_id": config.path_id,
        "fallback_used": bool(details.get("fallback_used", False)),
        "action_id": config.action_id,
        "config_id": config.config_id,
        "requested_params": dict(config.params),
        "execution_details": details,
        "query_context": context.tolist(),
        "policy_features": joint_action_features(context, config, latency_cap_ms=latency_cap_ms,
                                                  remaining_budget_ms=timer.remaining_ms).tolist(),
        "result_fill_count": len(article_ids),
        "result_full_k": len(article_ids) >= query.k,
        **dict(getattr(baseline, "last_selection_metadata", {})),
        **timing,
        **error_details,
    }
