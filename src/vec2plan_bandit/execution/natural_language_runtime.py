"""One request clock from an unparsed question to IDs for all four selectors."""
from __future__ import annotations

import hashlib

from vec2plan_bandit.execution.adaptive_runtime import run_adaptive_request, run_selector_baseline_request
from vec2plan_bandit.execution.timing import RequestTimer, record_request_audit


def run_question_request(*, question, structurer, collector, feature_pipeline,
                         latency_cap_ms, policy=None, baseline=None):
    if (policy is None) == (baseline is None):
        raise ValueError("provide exactly one main policy or baseline")
    timer = RequestTimer(latency_cap_ms)
    structurer.last_audit = {}
    identity = "N" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    record_request_audit(input_contract="question_string_only", spans_ms=timer.spans_ms)
    try:
        with timer.span("structured_query_ms"):
            query = structurer.structure(question)
        record_request_audit(executed_k=query.k,
            structured_query={"semantic_text": query.semantic_text,
                "embedding_column": query.embedding_column, "k": query.k,
                "predicates": list(query.predicates)})
        options = dict(query=query, collector=collector, feature_pipeline=feature_pipeline,
                       latency_cap_ms=latency_cap_ms, statistics=None,
                       request_timer=timer, compile_all_paths=False)
        if policy is not None:
            row = run_adaptive_request(**options, policy=policy, exploration=False, jev_router=None)
        else:
            row = run_selector_baseline_request(**options, baseline=baseline)
        row["executed_k"] = query.k
    except Exception as error:
        row = {"query_id": identity, "k": 1, "article_ids": [], "path_id": None,
               "error_type": type(error).__name__, "error_message": str(error)[:500],
               "result_fill_count": 0, "result_full_k": False, **timer.finish()}
    row.update(input_contract="question_string_only", structure_audit=structurer.last_audit)
    return row
