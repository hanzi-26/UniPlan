"""Shared BIRD sampling, training and frozen evaluation from the deployed runner."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np

from vec2plan_bandit.queries.wikipedia_sql_adapter import load_queries, grouped_split, MODEL
from vec2plan_bandit.features.wikipedia_features import WikiFeaturePipeline
from vec2plan_bandit.features.action_features import joint_action_features
from vec2plan_bandit.execution.wikipedia_runtime import WikiRuntime
from vec2plan_bandit.execution.timing import run_with_deadline, evaluation_deadline_ns
from vec2plan_bandit.policies.max_ucb import initial_design
from vec2plan_bandit.paths.registry import pre_configuration_key, make_execution_config
from vec2plan_bandit.policies.online_learning import OnlineLearner
from vec2plan_bandit.experiments.run import _fit_policy, _metric_summary
from vec2plan_bandit.feedback.reward import compute_reward
from vec2plan_bandit.baselines import builtin_baselines
from vec2plan_bandit.baselines.base import BaselineRequest

ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()


def score(row, query, truth, config):
    expected = Counter(tuple(v) for v in truth)
    actual = Counter(tuple(v) for v in row.get('row_ids', []))
    recall = sum((expected & actual).values()) / sum(expected.values()) if expected else None
    evaluable = len(truth) >= query.k and query.k > 0
    complete = len(row.get('row_ids', [])) >= query.k
    success = not row.get('error_type') and not row.get('timed_out')
    quality = bool(evaluable and success and complete and recall >= config['reward']['recall_threshold'])
    cap = config['reward']['latency_cap_ms']
    row.update(recall_at_k=recall, eligible_for_primary_quality=evaluable,
               ground_truth_row_count=len(truth),
               deadline_pass=not row.get('timed_out', False) and row['e2e_latency_ms'] <= cap,
               result_full_k=complete, quality_pass=quality,
               quality_target=int(quality), joint_success=quality and row['e2e_latency_ms'] <= cap,
               quality_success=quality, execution_success=success,
               reward=compute_reward(e2e_latency_ms=row['e2e_latency_ms'], recall_at_k=recall,
                    latency_cap_ms=cap, success=success, result_complete=complete,
                    timeout=row.get('timed_out', False), recall_threshold=config['reward']['recall_threshold']))
    return row


class SharedExperiment:
    def request(self, query, pipeline, stats, *, config=None, selector=None, baseline=None, exploration=False, key='', native=False, context_required=True,
                acceleration_router=None, _vectors=None, _disable_acceleration=False,
                _remaining_budget_ms=None):
        if selector is not None and not exploration and not _disable_acceleration:
            from vec2plan_bandit.policies.optional_acceleration import configured_acceleration, describe_query
            from vec2plan_bandit.execution.optional_acceleration import accelerated_request, execute_action
            router = acceleration_router or configured_acceleration(self.config)
            if router is not None:
                worker = self.runtime()
                if self.config.get('runtime', {}).get('detailed_timing', False):
                    from vec2plan_bandit.execution.detailed_timing import instrument_runtime
                    instrument_runtime(worker)
                n = sum(getattr(b, 'asset', {}).get('estimated_points',
                    getattr(worker, 'assets', {}).get(b.asset_key, {}).get('points', 0)) for b in query.blocks)
                def configurations(vectors):
                    pre = frozenset(self.prepared[query.query_id]['available_pre'])
                    return {r['config_id'] for r in router.artifact['rules']
                        if r['path_id'] != 'PRE' or pre_configuration_key(r['params']) in pre}
                row = accelerated_request(query=query, router=router,
                    description=lambda: describe_query(query, stats.get(query.query_id), corpus_size=n),
                    available_paths=lambda: {p: True for p in ('PRE', 'POST', 'FAISS', 'ACORN', 'PG_NATIVE_SQL')},
                    available_configurations=configurations, embed=lambda: worker.vectors(query),
                    execute_fast=lambda rule, vectors: execute_action(worker, query, vectors, rule),
                    request_main=lambda vectors, remaining: SharedExperiment.request(self, query, pipeline, stats,
                        selector=selector, key=key, _vectors=vectors, _disable_acceleration=True,
                        _remaining_budget_ms=remaining),
                    latency_cap_ms=self.config['reward']['latency_cap_ms'], cancel_pending=worker.cancel_pending,
                    recall_threshold=self.config['reward']['recall_threshold'],
                    budget_faiss_threshold_ms=(self.config.get('budget_faiss', {}).get('threshold_ms', 100.)
                        if self.config.get('budget_faiss', {}).get('enabled', False) else None))
                outer = row['acceleration_spans_ms']
                nested = row.get('main_request_timings', {})
                durations = {'embedding': row['embedding_ms'], 'context_features': nested.get('context_features', 0.),
                    'policy_selection': row['router_ms'] + nested.get('policy_selection', 0.),
                    'execution': outer.get('fast_execution_ms', 0.) + nested.get('execution', 0.)}
                durations['others'] = max(0., row['e2e_latency_ms'] - sum(durations.values()))
                row.update(spans_ms=durations, **{k + '_ms': v for k, v in durations.items()},
                    timing_protocol='bird-five-spans-v1', context_statistics_source='prepared_snapshot')
                return score(row, query, self.prepared[query.query_id]['topk_row_ids'], self.config)
        from vec2plan_bandit.execution.timing import RequestTimer
        worker = self.runtime()
        if self.config.get('runtime', {}).get('detailed_timing', False):
            from vec2plan_bandit.execution.detailed_timing import instrument_runtime
            instrument_runtime(worker)
        cap = self.config['reward']['latency_cap_ms']
        timer = RequestTimer(max(.001, _remaining_budget_ms) if _remaining_budget_ms is not None else cap)
        audit = {}
        fields = ('embedding', 'context_features', 'policy_selection', 'execution')
        def call():
            try:
                if _vectors is None:
                    with timer.span('embedding'):
                        vectors = worker.vectors(query)
                else:
                    vectors = _vectors
                if native:
                    audit.update(path_id='PG_NATIVE_SQL', config_id='native-pg', requested_params={})
                    with timer.span('execution'):
                        allowed=self.config.get('main_execution',{}).get('allowed_paths')
                        if selector is not None and allowed is not None and selected.path_id not in allowed:
                            raise ValueError('Selected main path is outside PRE/POST/FAISS/ACORN: '+selected.path_id)
                        row = worker.native(query, vectors)
                else:
                    selected = config
                    context = None
                    needs_context = context_required and (baseline is None or getattr(baseline, 'requires_context', True))
                    prepare_statistics = getattr(baseline, 'request_statistics', None)
                    selection_statistics = stats.get(query.query_id, {})
                    if needs_context or prepare_statistics is not None:
                        with timer.span('context_features'):
                            if needs_context:
                                mean = np.mean(vectors, axis=0)
                                context = pipeline.transform_one(query, statistics=stats,
                                    embeddings={query.query_id: mean / np.linalg.norm(mean)})
                            if prepare_statistics is not None:
                                selection_statistics = prepare_statistics(worker, query, vectors, selection_statistics)
                    available_pre = frozenset(self.prepared[query.query_id]['available_pre'])
                    with timer.span('policy_selection'):
                        training_facts = None
                        training_facts_ms=0.
                        if baseline is None and self.config.get('main_training',{}).get('enabled',False):
                            from vec2plan_bandit.experiments.main_inference import runtime_routing_facts
                            facts_start=time.perf_counter_ns()
                            training_facts=runtime_routing_facts(query,selection_statistics,worker,self.config,
                                                               getattr(self,'corpus_sizes',{}))
                            training_facts_ms=(time.perf_counter_ns()-facts_start)/1e6
                            audit['main_routing_facts']=training_facts
                        if selector is not None:
                            selection_options = {'request_facts':training_facts} if training_facts is not None else {}
                            if (training_facts is not None and self.config.get('runtime',{}).get('reuse_main_routing_facts',False)
                                    and getattr(selector,'collect_main_routing_facts',False)):
                                selection_options.update(request_facts=training_facts,request_facts_ms=training_facts_ms)
                            elif getattr(selector,'collect_main_routing_facts',False):
                                from vec2plan_bandit.experiments.main_inference import main_request_options
                                sizes = getattr(self,'corpus_sizes',{})
                                n = (sum(sizes[b.asset_key] for b in query.blocks)
                                     if sizes and all(b.asset_key in sizes for b in query.blocks) else None)
                                selection_options.update(main_request_options(selector,query,selection_statistics,corpus_size=n))
                            if getattr(selector,'main_lightweight',None) is not None:
                                selection_options['native_pg_available'] = callable(getattr(worker,'native',None))
                                if getattr(selector, 'main_sql_candidates_early', False):
                                    selection_options['sql_candidates_available'] = (
                                        (hasattr(worker, 'indexes') or callable(getattr(worker, 'execute_main_sql_candidate', None)))
                                        and callable(getattr(query, 'render', None)))
                                if getattr(selector, 'main_exact_pg_early', False):
                                    selection_options['exact_pg_available'] = (
                                        callable(getattr(worker, 'sql', None)) and callable(getattr(query, 'render', None)))
                            decision = selector.select(context, latency_cap_ms=cap,
                                remaining_budget_ms=timer.remaining_ms,
                                available={p:(self.config.get('main_execution',{}).get('required_path') in (None,p)) for p in ('PRE','POST','FAISS','ACORN')},
                                request_key=key, available_pre_configurations=available_pre, exploration=exploration,
                                **selection_options)
                            selected = decision.config
                            audit.update(main_selection=dict(getattr(decision,'selection_evidence',None) or {}),
                                         main_selection_mode=getattr(decision,'selection_mode','full_maxucb'))
                            audit['policy_stages_ms'] = dict(getattr(selector, 'last_timings', {}))
                            audit.update({k: getattr(decision, k) for k in ('constraints_satisfied','quality_lcb','predicted_e2e_latency_ms','feasible_candidate_count','evaluated_candidates',
                                'fallback_reason','fallback_candidate_count')})
                        elif baseline is not None:
                            selected = baseline.select(BaselineRequest(query, context if context is not None else np.empty(0),
                                {p:True for p in ('PRE','POST','FAISS','ACORN')},
                                cap, timer.remaining_ms, available_pre, selection_statistics))
                            audit.update(baseline_id=getattr(baseline, 'name', type(baseline).__name__),
                                **getattr(baseline, 'last_selection_metadata', {}))
                        if selected.path_id == 'PRE' and pre_configuration_key(selected.params) not in available_pre:
                            raise ValueError('unverified PRE configuration')
                        audit.update(path_id=selected.path_id, requested_params=dict(selected.params), config_id=selected.config_id)
                        compact_early = (selector is not None and not exploration and (
                            getattr(selector, 'main_compact_early', False)
                            and audit.get('main_selection_mode') == 'empirical_early_fallback'
                            or getattr(getattr(selector,'main_lightweight',None),'compact_early_features',False)
                            and audit.get('main_selection_mode') == 'lightweight'))
                        if context is not None and selected.path_id != 'PG_NATIVE_SQL' and not compact_early:
                            audit.update(query_context=context.tolist(), policy_features=joint_action_features(context, selected,
                                latency_cap_ms=cap, remaining_budget_ms=cap).tolist())
                    from vec2plan_bandit.execution.timing import record_request_audit
                    record_request_audit(**audit)
                    with timer.span('execution'):
                        if selector is not None and selected.path_id == 'PG_NATIVE_SQL':
                            row = worker.native(query, vectors)
                        elif selected.path_id == 'PG_EXACT_SQL':
                            if selector is None or not getattr(selector, 'main_exact_pg_early', False):
                                raise ValueError('Exact PG requires explicit main-policy opt-in')
                            from vec2plan_bandit.execution.main_exact_pg import execute_exact_pg
                            row = execute_exact_pg(worker, query, vectors)
                        elif selected.path_id in ('FAISS_SQL', 'POST_SQL', 'PG_SQL', 'EXACT_SQL'):
                            if selector is None or not getattr(selector, 'main_sql_candidates_early', False):
                                raise ValueError('SQL candidates require explicit main-policy opt-in')
                            from vec2plan_bandit.execution.main_sql_candidates import execute_sql_candidate
                            adapter = getattr(worker, 'execute_main_sql_candidate', None)
                            row = (adapter(query, vectors, selected) if callable(adapter)
                                   else execute_sql_candidate(worker, query, vectors, selected))
                        elif selector is not None and self.config.get('main_execution',{}).get('adaptive_fill',False):
                            from vec2plan_bandit.execution.main_adaptive_fill import execute_with_fill
                            row=execute_with_fill(lambda c:worker.execute(query,vectors,c),selected,k=query.k,
                                remaining_ms=lambda:timer.remaining_ms,facts=audit.get('main_routing_facts'),enabled=True,
                                allowed_configurations=(selector.main_configuration_bank['FAISS']
                                    if getattr(selector,'main_configuration_bank',None) is not None else None))
                        else:
                            row = worker.execute(query, vectors, selected)
                return {**row, **audit}
            except Exception as error:
                import requests
                if isinstance(error, requests.RequestException):
                    raise  # Service failure must not become a model-quality label.
                return {**audit, 'row_ids': [], 'error_type':type(error).__name__, 'error_message':str(error)}
        row = (run_with_deadline(call, query, cap, worker.cancel_pending)
               if evaluation_deadline_ns() is None else call())
        if 'e2e_latency_ms' not in row:
            row.update(e2e_latency_ms=timer.total_ms, timed_out=False)
        if row.get('timed_out'):
            row['row_ids'] = []
        durations = {name: timer.spans_ms.get(name, 0.0) for name in fields}
        durations['others'] = max(0.0, row['e2e_latency_ms'] - sum(durations.values()))
        row.update({name + '_ms': value for name, value in durations.items()})
        row.update(query_id=query.query_id, k=query.k, spans_ms=durations,
            timing_protocol='bird-five-spans-v1', context_statistics_source='prepared_snapshot', **audit)
        return score(row, query, self.prepared[query.query_id]['topk_row_ids'], self.config)

    def collect(self):
        pipeline, stats, _ = self.features()
        journal = self.output / 'collect/configuration_matrix.jsonl'
        done = {(r['query_id'],r['config_id']) for r in read_rows(journal)}
        design = initial_design(self.config['initial_design']['points_per_numeric_path'], seed=self.dataset['seed'])
        stream = [(self.by_id[q], c) for q in self.split['train'] for c in design
                  if c.path_id != 'PRE' or pre_configuration_key(c.params) in self.prepared[q]['available_pre']]
        random.Random(self.dataset['seed']).shuffle(stream)
        for query, config in stream:
            if (query.query_id,config.config_id) in done:
                continue
            for _ in range(self.config['measurement']['warmup_runs']):
                self.request(query,pipeline,stats,config=config)
            row = self.request(query,pipeline,stats,config=config)
            row.update(evaluation_mode='train_bootstrap')
            if 'policy_features' not in row: raise RuntimeError(f'{query.query_id}: collection failed before selection; inspect {row}')
            append(journal,row)
        print(f'Collected {len(read_rows(journal))} train bootstrap observations',flush=True)

    def train(self):
        pipeline, stats, _ = self.features()
        bootstrap = read_rows(self.output / 'collect/configuration_matrix.jsonl')
        if any(r['query_id'] not in self.split['train'] for r in bootstrap):
            raise ValueError('held-out labels in train')
        observations = [self.observation(r) for r in bootstrap]
        policy, _ = _fit_policy(observations,self.config)
        learner = OnlineLearner(policy,observations,refit_every=self.config['online_learning']['refit_every'])
        journal = self.output / 'train/training_live.jsonl'
        stream = list(self.split['train'])
        random.Random(self.dataset['seed']).shuffle(stream)
        previous = read_rows(journal)
        if [r['query_id'] for r in previous] != stream[:len(previous)]:
            raise ValueError('training journal order mismatch')
        def observe(row):
            from vec2plan_bandit.surrogates.main_cost import feedback_fields
            learner.observe(observation_id=row['query_id'],path_id=row['path_id'],features=row['policy_features'],
                 quality_target=row['quality_target'],e2e_latency_ms=row['e2e_latency_ms'],feedback_metadata=feedback_fields(row))
        for row in previous:
            observe(row)
        for key in stream[len(previous):]:
            query = self.by_id[key]
            for _ in range(self.config['measurement']['warmup_runs']):
                self.request(query,pipeline,stats,selector=policy,exploration=True,key=key+':warmup')
            row = self.request(query,pipeline,stats,selector=policy,exploration=True,key=key)
            row.update(evaluation_mode='train_exploration')
            if 'policy_features' not in row:
                raise RuntimeError(f'{key}: train selection failed before features were produced')
            append(journal,row)
            observe(row)
            print(f'Train {key}: {row["path_id"]}, quality={row["quality_target"]}',flush=True)
        learner.refit()
        policy.surrogate.save(self.output / 'train/frozen_model/surrogate')
        write(self.output / 'train/frozen_model/policy.json',policy.state_dict())
        write(self.output / 'train/frozen_model/provenance.json',{'training_query_ids':self.split['train'],
             'source_hashes': {q:self.by_id[q].source_hash for q in self.split['train']},'embedding':self.config['embedding'],
             'held_out_labels_used':False,'refit_every':learner.refit_every,'online_refits':learner.refit_count})
        rows = bootstrap + read_rows(journal)
        destination = self.output / 'train/shared_training.jsonl'
        destination.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print('BIRD train model frozen',flush=True)

    @staticmethod
    def observation(row):
        return {**{k:row.get(k) for k in ('execution_ms','timed_out','latency_censored','error_type',
                'eligible_for_primary_quality','main_routing_facts','execution_stages_ms')},
                'query_id':row['query_id'],'features':row['policy_features'],'query_context':row['query_context'],
                'path_id':row['path_id'],'quality_target':row['quality_target'],'e2e_latency_ms':row['e2e_latency_ms'],
                'reward':row['reward']}

    def benchmark(self, role):
        from vec2plan_bandit.surrogates.catboost_ensemble import ConstrainedCatBoostSurrogate
        from vec2plan_bandit.policies.max_ucb import MaxUCB
        pipeline, stats, _ = self.features()
        directory = self.output / 'train/frozen_model'
        state = json.loads((directory/'policy.json').read_text())
        policy = MaxUCB(ConstrainedCatBoostSurrogate.load(directory/'surrogate'), **self.config['max_ucb'])
        policy.load_state_dict(state)
        # Loading a supported legacy checkpoint normalizes its version and
        # default settings. Compare against the loaded state to detect changes
        # during evaluation without mistaking that migration for learning.
        frozen_state = policy.state_dict()
        observations = [self.observation(r) for r in read_rows(self.output/'train/shared_training.jsonl')]
        baseline_config = json.loads((ROOT/self.dataset['baselines']).read_text())
        registry = builtin_baselines()
        baselines = {}
        for spec in baseline_config['baselines']:
            if spec['name'] in ('fixed-wiki','native-pg'):
                baselines[spec['id']] = spec
            else:
                options = dict(spec['config'])
                if spec['name'] == 'laya-router':
                    options['recall_threshold'] = float(self.config['reward']['recall_threshold'])
                baselines[spec['id']] = registry.create(spec['name'], options).fit(observations)
        main_method = self.dataset.get('main_method', 'text2plan')
        methods = [main_method,*baselines]
        journal = self.output/role/'benchmark_live.jsonl'
        done = {(r['query_id'],r['method_id'],r['repeat']) for r in read_rows(journal)}
        write(self.output/role/'effective_policy.json',state)
        for repeat in range(1,self.dataset['repeats']+1):
            stream = [(self.by_id[q],m) for q in self.split[role] for m in methods]
            random.Random(f'{self.dataset["seed"]}:{role}:{repeat}').shuffle(stream)
            for query,method in stream:
                if (query.query_id,method,repeat) in done:
                    continue
                kwargs = {'key':f'{role}:{repeat}:{query.query_id}'}
                if method==main_method:
                    kwargs['selector']=policy
                elif isinstance(baselines[method],dict):
                    spec=baselines[method]
                    if spec['name']=='native-pg':
                        kwargs.update(native=True, context_required=False)
                    else:
                        params={'index_mode':'no_index','join_strategy':'all'} if spec['config']['path_id']=='PRE' else None
                        kwargs.update(config=make_execution_config(spec['config']['path_id'],spec['config'].get('params',params)), context_required=False)
                else:
                    kwargs['baseline']=baselines[method]
                for _ in range(self.config['measurement']['warmup_runs']):
                    self.request(query,pipeline,stats,**kwargs)
                row=self.request(query,pipeline,stats,**kwargs)
                row.update(method_id=method,repeat=repeat,evaluation_mode='frozen',split=role,
                     recall_threshold=self.config['reward']['recall_threshold'],quality_probability_min=policy.quality_probability_min)
                append(journal,row)
            print(f'{role} repeat {repeat} complete',flush=True)
        if policy.state_dict()!=frozen_state:
            raise ValueError('held-out benchmark mutated frozen policy')
        rows=read_rows(journal)
        summaries = {}
        for method in methods:
            values = [r for r in rows if r['method_id'] == method]
            summary = _metric_summary(values)
            recalls = [r['recall_at_k'] for r in values if r['eligible_for_primary_quality']]
            summary['mean_recall_at_k'] = float(np.mean(recalls)) if recalls else None
            summary['timeout_rate'] = float(np.mean([r.get('timed_out', False) for r in values]))
            summary['timing_ms'] = {name: {'mean': float(np.mean([r[name + '_ms'] for r in values])), 'median': float(np.median([r[name + '_ms'] for r in values])), 'p95': float(np.percentile([r[name + '_ms'] for r in values],95))} for name in ('embedding','context_features','policy_selection','execution','others')}
            summary['timing_requests'] = len(values)
            summary['observed_request_duration_mean_ms'] = float(np.mean([r['e2e_latency_ms'] for r in values]))
            summary['timing_summary_scope'] = 'all observed requests; timed-out spans are partial durations'
            summary['timing_completed_ms'] = {name: {'mean': float(np.mean([r[name + '_ms'] for r in values if not r.get('latency_censored')])) if any(not r.get('latency_censored') for r in values) else None} for name in ('embedding','context_features','policy_selection','execution','others')}
            summaries[method] = summary
        write(self.output/role/'summary.json', summaries)
        self.report()
