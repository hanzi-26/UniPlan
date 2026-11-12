"""Opt-in train-only risk guard and exact execution over existing assets.

Approximate profiles have empirical group-level risk estimates, not a coverage
guarantee. Probability one always requires exact execution. No ground truth,
query-result cache, or new vector index is used at request time.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from vec2plan_bandit.paths.registry import make_execution_config
from vec2plan_bandit.paths.sql import RenderedSQL, render_pre
from vec2plan_bandit.execution.timing import evaluation_deadline_ns, record_request_audit
from vec2plan_bandit.execution.path_base import remaining_timeout_ms

VERSION = 'arxiv-train-profile-recall-guard-v1'
PROFILES = {
    'FAISS': {'ef_search': 800, 'candidate_multiplier': 50.},
    'ACORN': {'hnsw_ef': 800},
    'POST': {'ef_search': 800, 'max_scan_tuples': 200000},
}


def bucket(query):
    if not query.predicates:
        return 'unfiltered'
    if any(p['type'].startswith(('author_', 'submitter_', 'category_')) for p in query.predicates):
        return 'joined'
    return 'filtered'


def risk_lower_bound(values, z=1.2815515655446004):
    """Wilson score on independent semantic-group all-query successes.

    This is a conservative empirical selection score. Groups and deployment
    data need not be IID, so it is deliberately not a coverage certificate.
    """
    n = len(values)
    if not n:
        return 0.
    p = sum(values) / n
    return max(0., (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n))


def fit_guard(rows, queries, train_ids, *, recall_threshold=1.):
    train_ids = set(train_ids)
    by_id = {q.query_id: q for q in queries}
    grouped = {}
    seen = set()
    for row in rows:
        qid, path = row['query_id'], row['path_id']
        if qid not in train_ids or row.get('recall_threshold') != recall_threshold:
            raise ValueError('Guard calibration must use matching-recall train labels only')
        if row.get('requested_budget_ms') != 5000:
            raise ValueError('Guard calibration budget mismatch')
        if path not in PROFILES or dict(row['requested_params']) != PROFILES[path]:
            raise ValueError('Unmeasured guard profile')
        if (qid, path) in seen:
            raise ValueError('Duplicate profile/query observation')
        seen.add((qid, path))
        query = by_id[qid]
        key = (bucket(query), path)
        grouped.setdefault(key, []).append((query.group_id, row))
    profiles = {}
    for (kind, path), samples in grouped.items():
        groups = {}
        for group, row in samples:
            ok = bool(row.get('joint_success') and row.get('result_full_k')
                      and not row.get('timed_out') and not row.get('error_type'))
            groups[group] = groups.get(group, True) and ok
        profiles.setdefault(kind, {})[path] = dict(params=PROFILES[path], groups=len(groups),
            successful_groups=sum(groups.values()), rows=len(samples),
            lower_score=risk_lower_bound(list(groups.values())),
            mean_ms=float(np.mean([r['e2e_latency_ms'] for _, r in samples])))
    return dict(version=VERSION, labels_role='train_only_new_profile_executions',
                train_ids=sorted(train_ids), dev_test_labels_used=False, profiles=profiles,recall_threshold=recall_threshold,
                evidence='group-weighted empirical risk score; no deployment coverage guarantee')


@lru_cache(maxsize=8)
def load_guard(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Recall guard artifact changed')
    result = json.loads(raw)
    if result['version'] != VERSION or result.get('dev_test_labels_used') is not False:
        raise ValueError('Invalid recall guard artifact')
    return result


@dataclass(frozen=True)
class ExactConfig:
    path_id: str = 'MAIN_EXACT'
    config_id: str = 'main-exact-existing-storage-v1'
    action_id: None = None

    @property
    def params(self):
        return {}


def guard_decision(decision, query, probability, artifact, available):
    measured = artifact['profiles'].get(bucket(query), {})
    allowed = {p: v for p, v in measured.items() if available.get(p, False)
               and v['groups'] >= 5 and v['lower_score'] >= probability}
    exact = probability >= 1. or not allowed
    if exact:
        config = ExactConfig()
        evidence = dict(mode='exact_required', reason='probability_one' if probability >= 1. else 'no_calibrated_profile')
    else:
        path = decision.config.path_id
        if path not in allowed:
            path = min(allowed, key=lambda p: allowed[p]['mean_ms'])
        config = make_execution_config(path, allowed[path]['params'])
        evidence = dict(mode='calibrated_profile', path=path, **allowed[path])
    evidence.update(original_config_id=decision.config.config_id,
                    original_path=decision.config.path_id,
                    original_constraints_satisfied=decision.constraints_satisfied,
                    probability_target=probability, recall_target=1.)
    # Original model predictions are retained for audit, and are explicitly
    # marked as applying to the original configuration, not the replacement.
    return replace(decision, config=config, constraints_satisfied=False,
                   selection_mode='main_recall_guard',
                   selection_evidence={'recall_guard': evidence}), evidence


def execute_exact(collector, query, vector, vector_literal, facts=None):
    """Complete global flat prefixes + exact metadata, or materialized SQL.

    IndexFlatIP is the existing HNSW backing storage, downcast without copying.
    Each returned global prefix is exhaustive; filtering a sufficiently long
    prefix therefore gives the exact filtered top-k. No ANN stability heuristic.
    """
    guard=getattr(collector,'config',{}).get('main_recall_guard',{})
    if guard.get('cpu_exact_scan',False) or guard.get('cpu_exact_fallback',False):
        return execute_exact_cpu(collector,query,vector,vector_literal)
    remaining_timeout_ms(evaluation_deadline_ns())
    selectivity = (facts or {}).get('selectivity', 1.)
    attempts = []
    if not query.predicates or selectivity >= .02:
        executor = collector._faiss_executor
        index, _ = executor._load(query.embedding_column)
        module = executor._module()
        flat = module.downcast_index(index.storage)
        if (not isinstance(flat, module.IndexFlat) or flat.metric_type != module.METRIC_INNER_PRODUCT
                or flat.ntotal != index.ntotal or flat.code_size != 4 * flat.d):
            raise ValueError('Exact scan requires the existing complete IndexFlatIP storage')
        mapping = collector.article_id_mappings[query.embedding_column]
        count = min(flat.ntotal, max(query.k + 1, int(math.ceil(query.k / max(selectivity, .02))) * 2))
        while count <= 8192 or count == flat.ntotal:
            remaining_timeout_ms(evaluation_deadline_ns())
            started = time.perf_counter_ns()
            scores, positions = flat.search(np.asarray(vector, dtype=np.float32).reshape(1, -1), count)
            remaining_timeout_ms(evaluation_deadline_ns())
            ids = mapping.decode([int(p) for p in positions[0] if p >= 0])
            pairs = sorted(zip(ids, scores[0][:len(ids)]), key=lambda pair: (-float(pair[1]), pair[0]))
            eligible = (executor._eligibility_provider.eligible_ids(query, ids) if query.predicates else set(ids))
            selected = [(i, s) for i, s in pairs if i in eligible]
            attempts.append(dict(count=count, elapsed_ms=(time.perf_counter_ns()-started)/1e6))
            record_request_audit(main_exact_attempts=attempts)
            # Include complete boundary ties before applying the logical-ID tie break.
            if count == flat.ntotal or (len(selected) >= query.k and selected[query.k-1][1] > scores[0][-1]):
                return [i for i, _ in selected[:query.k]], dict(exact=True, uses_hnsw=False,
                    implementation='existing_faiss_flat_storage', new_index_built=False,
                    exact_prefix_attempts=attempts, result_complete=len(selected) >= query.k)
            next_count = min(flat.ntotal, count * 4)
            if next_count == count:
                break
            count = next_count
    remaining_timeout_ms(evaluation_deadline_ns())
    # Existing Qdrant payload indexes provide exact membership for sparse
    # filters; exact=True scans all eligible vectors rather than the ANN graph.
    # Keep the ACORN baseline's exact=False executor untouched.
    if hasattr(collector, '_qdrant_client'):
        executor = collector.path_router._executors['ACORN']
        models = executor._models_module()
        query_filter = executor._filter_compiler.compile(query, models)
        timeout_ms = remaining_timeout_ms(evaluation_deadline_ns())
        response = collector._qdrant_client.query_points(
            collection_name=executor._collections[query.embedding_column],
            query=np.asarray(vector,dtype=np.float32).tolist(),query_filter=query_filter,
            search_params=models.SearchParams(exact=True),limit=query.k,
            with_payload=False,with_vectors=False,
            **({'timeout':max(1,math.ceil(timeout_ms/1000))} if timeout_ms is not None else {}))
        remaining_timeout_ms(evaluation_deadline_ns())
        points = getattr(response,'points',response)
        ids = [int(p.id if hasattr(p,'id') else p['id']) for p in points]
        if executor._id_map(query.embedding_column) is not None:
            ids = collector.article_id_mappings[query.embedding_column].decode(ids)
        return ids,dict(exact=True,uses_hnsw=False,implementation='existing_qdrant_exact_filtered_scan',
                        new_index_built=False,exact_prefix_attempts=attempts,
                        result_complete=len(ids)>=query.k)
    # MATERIALIZED is the ANN fence. Leave scalar indexes and join planning
    # available instead of copying PRE-Scan's forced sequential-scan settings.
    base = render_pre(query, index_mode='no_index', join_strategy='all')
    rendered = RenderedSQL('MAIN_EXACT', (), base.sql)
    ids, _ = collector._pg(rendered, vector_literal)
    return ids, dict(exact=True, uses_hnsw=False, implementation='materialized_exact_sql',
                     new_index_built=False, exact_prefix_attempts=attempts,
                     result_complete=len(ids) >= query.k)


def execute_exact_cpu(collector, query, vector, vector_literal):
    """Score exactly the complete eligible subset, once, on the existing CPU assets."""
    remaining_timeout_ms(evaluation_deadline_ns())
    executor=collector._faiss_executor
    index,_=executor._load(query.embedding_column)
    module=executor._module()
    flat=module.downcast_index(index.storage)
    if (not isinstance(flat,module.IndexFlat) or flat.metric_type!=module.METRIC_INNER_PRODUCT
            or flat.code_size!=flat.d*4 or flat.ntotal!=index.ntotal):
        raise ValueError('Exact CPU execution requires complete full-precision flat IP storage')
    mapping=collector.article_id_mappings[query.embedding_column]
    if not query.predicates:
        scores,positions=flat.search(np.asarray(vector,dtype=np.float32).reshape(1,-1),min(query.k+1,flat.ntotal))
        # Generic tie-safe fallback retains exact ordering even for duplicate vectors.
        if len(scores[0])>query.k and scores[0][query.k-1]==scores[0][query.k]:
            count=len(scores[0])
            while count<flat.ntotal and scores[0][query.k-1]==scores[0][-1]:
                count=min(flat.ntotal,count*4)
                remaining_timeout_ms(evaluation_deadline_ns())
                scores,positions=flat.search(np.asarray(vector,dtype=np.float32).reshape(1,-1),count)
        ids=mapping.decode([int(p) for p in positions[0] if p>=0])
        pairs=sorted(zip(ids,scores[0]),key=lambda item:(-float(item[1]),item[0]))
        remaining_timeout_ms(evaluation_deadline_ns())
        return [i for i,_ in pairs[:query.k]],dict(exact=True,uses_hnsw=False,
            implementation='cpu_exact_flat_unfiltered',eligible_count=flat.ntotal,
            distance_evaluations=flat.ntotal,new_index_built=False)
    metadata=getattr(collector,'exact_metadata',None)
    if metadata is None:raise ValueError('Exact scalar metadata must be preloaded outside request timing')
    started=time.perf_counter_ns()
    positions=metadata.eligible(query)
    filter_ms=(time.perf_counter_ns()-started)/1e6
    remaining_timeout_ms(evaluation_deadline_ns())
    if positions is None:
        # Relational predicates stay with the proven Qdrant filter compiler.
        # Never first run multiple full scans with a fabricated selectivity=1.
        options=collector.config['main_recall_guard']
        # Call the legacy exact backend without mutating the shared config.
        return execute_exact_remote(collector,query,vector,filter_ms)
    if not len(positions):
        return [],dict(exact=True,uses_hnsw=False,implementation='cpu_exact_scalar_subset',
                       eligible_count=0,distance_evaluations=0,filter_ms=filter_ms,new_index_built=False)
    q=np.ascontiguousarray(vector,dtype=np.float32)
    positions=np.ascontiguousarray(positions,dtype=np.int64)
    distances=np.empty(len(positions),dtype=np.float32)
    started=time.perf_counter_ns()
    flat.compute_distance_subset(1,module.swig_ptr(q),len(positions),
                                 module.swig_ptr(distances),module.swig_ptr(positions))
    remaining_timeout_ms(evaluation_deadline_ns())
    count=min(query.k,len(positions))
    chosen=np.argpartition(-distances,count-1)[:count]
    # Include all boundary ties before the logical-ID tie break.
    boundary=float(np.min(distances[chosen]))
    chosen=np.flatnonzero(distances>=boundary)
    ids=mapping.values[positions[chosen]]
    order=np.lexsort((ids,-distances[chosen]))[:count]
    result=[int(i) for i in ids[order]]
    return result,dict(exact=True,uses_hnsw=False,implementation='cpu_exact_scalar_subset',
        eligible_count=len(positions),distance_evaluations=len(positions),filter_ms=filter_ms,
        score_ms=(time.perf_counter_ns()-started)/1e6,new_index_built=False)


def run_profile_gate(collector,query,vector,vector_literal,timer):
    """Admit train-measured profiles, otherwise score the eligible CPU subset.

    This opt-in early stage replaces optimistic model admission with a frozen
    group-level risk score. It reads no query IDs, truth or held-out outcomes.
    """
    config=collector.config;options=config['main_recall_guard']
    artifact=load_guard(options['artifact'],options['sha256'])
    target=float(config['reward']['recall_threshold'])
    probability=float(config['max_ucb']['quality_probability_min'])
    if (artifact.get('recall_threshold',1.)!=target or not 0<target<=1
            or not config.get('snapshot_id','').startswith('vectorsqlbench-arxiv-')):
        raise ValueError('Profile risk gate target/dataset mismatch')
    available=collector.path_router.available(probe=False,query=query)
    profiles=artifact['profiles'].get(bucket(query),{})
    allowed={p:v for p,v in profiles.items() if available.get(p,False)
        and v['groups']>=5 and v['lower_score']>=probability and probability<1.}
    path=min(allowed,key=lambda p:allowed[p]['mean_ms']) if allowed else 'MAIN_EXACT'
    params=dict(allowed[path]['params']) if allowed else {}
    selected=make_execution_config(path,params) if allowed else ExactConfig()
    evidence=dict(mode='train_profile_risk_gate',bucket=bucket(query),recall_target=target,
        probability_target=probability,admitted_paths=sorted(allowed),
        lower_score=allowed[path]['lower_score'] if allowed else None,
        reason='measured_profile_admitted' if allowed else 'no_admitted_profile_cpu_exact')
    record_request_audit(path_id=path,config_id=selected.config_id,requested_params=params,
        main_selection_mode='train_profile_risk_gate',main_selection=evidence,
        policy_features=[],query_context=[])
    ids=[];details={};errors={};fallback=False
    try:
        with timer.span('execution_ms'):
            if path=='MAIN_EXACT':
                ids,details=execute_exact_cpu(collector,query,vector,vector_literal)
            else:
                ids,_,details=collector.execute(selected.action_id,query,None,vector,vector_literal,selected,
                    deadline_ns=evaluation_deadline_ns())
                if len(set(ids))<query.k:
                    previous=details
                    ids,details=execute_exact_cpu(collector,query,vector,vector_literal)
                    details.update(underfilled_profile=previous,actual_execution_path='MAIN_EXACT',exact_fallback=True)
                    fallback=True
        seen=set();ids=[int(i) for i in ids if not (int(i) in seen or seen.add(int(i)))][:query.k]
    except Exception as error:
        errors=dict(error_type=type(error).__name__,error_message=str(error)[:500])
    finally:
        record_request_audit(spans_ms=dict(timer.spans_ms),execution_ms=timer.spans_ms.get('execution_ms'))
    return dict(query_id=query.query_id,k=query.k,article_ids=ids,path_id=path,action_id=selected.action_id,
        config_id=selected.config_id,requested_params=params,execution_details=details,
        main_selection_mode='train_profile_risk_gate',main_selection=evidence,
        policy_version='arxiv-train-profile-cpu-fallback-v1',fallback_used=fallback,
        result_full_k=len(ids)>=query.k,result_fill_count=len(ids),**timer.finish(),**errors)


def execute_exact_remote(collector,query,vector,filter_ms=0.):
    executor=collector.path_router._executors['ACORN']
    models=executor._models_module()
    query_filter=executor._filter_compiler.compile(query,models)
    timeout_ms=remaining_timeout_ms(evaluation_deadline_ns())
    response=collector._qdrant_client.query_points(collection_name=executor._collections[query.embedding_column],
        query=np.asarray(vector,dtype=np.float32).tolist(),query_filter=query_filter,
        search_params=models.SearchParams(exact=True),limit=query.k,with_payload=False,with_vectors=False,
        **({'timeout':max(1,math.ceil(timeout_ms/1000))} if timeout_ms is not None else {}))
    remaining_timeout_ms(evaluation_deadline_ns())
    points=getattr(response,'points',response)
    ids=[int(p.id if hasattr(p,'id') else p['id']) for p in points]
    if executor._id_map(query.embedding_column) is not None:
        ids=collector.article_id_mappings[query.embedding_column].decode(ids)
    return ids,dict(exact=True,uses_hnsw=False,implementation='cpu_exact_qdrant_relational',
                    filter_ms=filter_ms,new_index_built=False)
