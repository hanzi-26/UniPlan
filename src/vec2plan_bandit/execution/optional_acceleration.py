"""Shared optional fast-route lifecycle for every dataset runtime."""
from __future__ import annotations

from vec2plan_bandit.execution.timing import (
    RequestTimer, evaluation_deadline_ns, record_request_audit, run_with_deadline,
    EvaluationDeadlineExceeded,
)
from vec2plan_bandit.paths.registry import make_execution_config


def execute_action(worker, query, vectors, rule):
    if rule['path_id'] == 'PG_NATIVE_SQL':
        return worker.native(query, vectors)
    return worker.execute(query, vectors, make_execution_config(rule['path_id'], rule['params']))


def accelerated_request(*, query, router, description, available_paths, embed,
                        execute_fast, request_main, latency_cap_ms, id_field='row_ids',
                        available_configurations=None, cancel_pending=None, timer=None,
                        recall_threshold=None, budget_faiss_threshold_ms=None):
    """Embed once, optionally execute a validated action, otherwise run main.

    description and availability are callbacks: their actual costs belong to
    the request. request_main receives the SAME vectors and remaining budget.
    It must use the frozen main policy with acceleration disabled. Its full
    request cost (including execution) is recorded as one outer span; internal
    timings remain nested in main_request_timings and are never double counted.
    An existing hard deadline is reused; this function never installs a nested
    signal timer or retries after a deadline has expired.
    """
    timer = timer or RequestTimer(latency_cap_ms)
    audit = {'router_hit': False, 'router_fast_accepted': False, 'router_fallback': False,
             'policy_skipped': False, 'acceleration_version': router.artifact['version']}

    def call():
        with timer.span('router_ms'):
            desc, paths = description(), available_paths()
            remaining = timer.remaining_ms
            audit.update(router_remaining_budget_ms=remaining,
                         budget_faiss_threshold_ms=budget_faiss_threshold_ms)
            if budget_faiss_threshold_ms is not None:
                decision = router.select_budget_faiss(desc, remaining_ms=remaining,
                    threshold_ms=budget_faiss_threshold_ms, available_paths=paths,
                    recall_threshold=recall_threshold)
            else:
                decision = router.select(desc, available_paths=paths,
                    recall_threshold=recall_threshold, latency_cap_ms=remaining)
            rule = decision.rule
            audit.update(router_reason=decision.reason, router_hit=rule is not None)
        record_request_audit(**audit)
        if timer.remaining_ms <= 0:
            raise EvaluationDeadlineExceeded()
        with timer.span('embedding_ms'):
            vectors = embed()
        if rule is not None and available_configurations is not None:
            with timer.span('applicability_ms'):
                try:
                    configurations = available_configurations(vectors)
                except Exception as error:
                    rule = None
                    audit.update(router_hit=False, router_reason='fast_applicability_check_failed',
                                 applicability_error_type=type(error).__name__)
                else:
                    if configurations is not None and rule['config_id'] not in configurations:
                        rule = None
                        audit.update(router_hit=False, router_reason='fast_configuration_unavailable')
            record_request_audit(**audit)
        if rule is not None:
            if timer.remaining_ms <= 0:
                raise EvaluationDeadlineExceeded()
            audit.update(path_id=rule['path_id'], config_id=rule['config_id'], requested_params=rule['params'])
            record_request_audit(**audit)
            with timer.span('fast_execution_ms'):
                try:
                    fast = execute_fast(rule, vectors)
                except Exception as error:
                    fast = {id_field: [], 'error_type': type(error).__name__, 'error_message': str(error)}
            if (not fast.get('error_type') and not fast.get('timed_out') and
                    fast.get('execution_success', True) and len(fast.get(id_field, [])) >= query.k):
                audit.update(router_fast_accepted=True, policy_skipped=True)
                record_request_audit(**audit)
                return {**fast, **audit}
            audit['fast_execution_stages_ms']=fast.get('execution_stages_ms', {})
            audit.update(router_fallback=True, router_reason='fast_error_or_underfilled',
                         fast_path_id=rule['path_id'], fast_error_type=fast.get('error_type'),
                         fast_error_message=fast.get('error_message'))
            record_request_audit(**audit)
        if timer.remaining_ms <= 0:
            raise EvaluationDeadlineExceeded()
        with timer.span('main_request_ms'):
            main = request_main(vectors, timer.remaining_ms)
        # Main's selected action replaces the attempted action in the final row.
        for name in ('path_id', 'config_id', 'requested_params'):
            if name in main:
                audit[name] = main[name]
        record_request_audit(**audit)
        return {**main, **audit,
                'main_request_timings': main.get('timings', main.get('spans_ms', {}))}

    def guarded():
        try:
            return call()
        except Exception as error:
            import requests
            if isinstance(error, requests.RequestException):
                raise
            return {id_field: [], 'error_type': type(error).__name__, 'error_message': str(error), **audit}

    if evaluation_deadline_ns() is None:
        row = run_with_deadline(guarded, query, timer.remaining_ms, cancel_pending)
    else:
        row = guarded()
    timing = timer.finish()
    total = timing['e2e_latency_ms']
    spans = dict(timer.spans_ms)
    spans['others_ms'] = max(0., total - sum(spans.values()))
    if row.get('timed_out'):
        row[id_field] = []
    row.update(query_id=query.query_id, k=query.k, e2e_latency_ms=total,
               result_full_k=len(row.get(id_field, [])) >= query.k,
               result_fill_count=len(row.get(id_field, [])), acceleration_spans_ms=spans,
               deadline_pass=not row.get('timed_out', False) and total < latency_cap_ms,
               router_ms=spans.get('router_ms', 0.) + spans.get('applicability_ms', 0.),
               embedding_ms=spans.get('embedding_ms', 0.))
    return row
