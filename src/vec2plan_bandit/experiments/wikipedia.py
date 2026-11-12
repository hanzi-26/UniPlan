"""Isolated Wiki experiment using the existing optimizer, surrogate and learner."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np

from vec2plan_bandit.queries.wikipedia_sql_adapter import load_queries, grouped_split, semantic_query_groups, MODEL
from vec2plan_bandit.features.wikipedia_features import WikiFeaturePipeline
from vec2plan_bandit.features.action_features import joint_action_features
from vec2plan_bandit.execution.wikipedia_runtime import WikiRuntime
from vec2plan_bandit.execution.timing import run_with_deadline, record_request_audit
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


def decision_audit(decision):
    """Audit v4 and v5 decisions without changing the selected configuration.

    v4 falls back to its highest-quality candidate when none is feasible; it
    has no quality-tolerance relaxation. Do not report the configured v5
    tolerance as though v4 had applied it.
    """
    fields = {k: getattr(decision, k) for k in (
        'constraints_satisfied', 'quality_lcb', 'predicted_e2e_latency_ms',
        'feasible_candidate_count', 'evaluated_candidates')}
    fields['fallback_reason'] = getattr(decision, 'fallback_reason',
        None if decision.constraints_satisfied else 'no_feasible_candidate_best_quality')
    fields['fallback_quality_tolerance'] = getattr(decision, 'fallback_quality_tolerance', 0.0)
    fields['policy_version'] = decision.policy_version
    return fields


def restore_policy(surrogate, config, state):
    from vec2plan_bandit.policies.max_ucb import MaxUCB
    # v4 checkpoints have no v5-only fallback_quality_tolerance setting.
    # Load exactly the settings actually recorded by the trained policy.
    policy = MaxUCB(surrogate, **{k: v for k, v in config.items() if k in state})
    policy.load_state_dict(state)
    return policy


@contextmanager
def request_stage(audit, name, enabled):
    """BIRD-only, non-overlapping wall-clock spans, including failed attempts.

    Publish in finally so hard-deadline cancellation retains the partial span.
    A not-started stage stays None, not a fabricated zero-duration observation.
    """
    if not enabled:
        yield
        return
    started = time.perf_counter_ns()
    status = 'interrupted'
    try:
        yield
        status = 'completed'
    except Exception:
        status = 'failed'
        raise
    finally:
        audit[f'{name}_ms'] = (time.perf_counter_ns() - started) / 1e6
        audit['stage_timing_status'][name] = status
        record_request_audit(**audit)


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


class WikiExperiment:
    record_stage_timings = False

    def __init__(self, dataset_path):
        self.dataset = json.loads(dataset_path.read_text())
        self.config = json.loads((ROOT / self.dataset['experiment_config']).read_text())
        self.output = ROOT / self.dataset['output_dir']
        self.data = (ROOT / self.dataset['queries']).parent
        self.output.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        source = Path(self.dataset['source']).expanduser()
        self.queries = load_queries(source, self.config['embedding']['model'])
        self.by_id = {q.query_id: q for q in self.queries}
        self.routable = [q for q in self.queries if q.blocks]
        split_path = ROOT / self.dataset['splits']
        if self.dataset.get('split_protocol') == 'provided-semantic-grouped-v1':
            from vec2plan_bandit.queries.grouped_stratification import validate_grouped_split
            proposed = json.loads(split_path.read_text())
            validate_grouped_split(proposed, semantic_query_groups(self.routable))
        else:
            proposed = grouped_split(self.routable, self.dataset['seed'])
        if split_path.exists() and json.loads(split_path.read_text()) != proposed:
            raise ValueError('existing Wiki split differs; use a new dataset directory')
        write(split_path, proposed)
        self.split = proposed
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        contract = {'source_sha256': source_hash, 'config': self.config, 'dataset': self.dataset,
                    'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                    'split_snapshot': self.split, 'adapter': 'wiki-sql-v1', 'source_hashes': {q.query_id: q.source_hash for q in self.queries}}
        contract['implementation_sha256'] = {
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                Path(__file__), ROOT / 'src/vec2plan_bandit/queries/wikipedia_sql_adapter.py',
                ROOT / 'src/vec2plan_bandit/execution/wikipedia_runtime.py',
                ROOT / 'src/vec2plan_bandit/features/wikipedia_features.py',
                ROOT / 'src/vec2plan_bandit/policies/max_ucb.py',
                ROOT / 'src/vec2plan_bandit/surrogates/catboost_ensemble.py',
                ROOT / self.dataset['baselines'])}
        contract_path = self.output / 'run_contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('Wiki run contract changed; use a new output directory')
        write(contract_path, contract)
        (self.data / 'candidate_sql.json').write_bytes(source.read_bytes())
        (ROOT / self.dataset['queries']).write_text(''.join(json.dumps(asdict(q), ensure_ascii=False) + '\n' for q in self.queries))
        self.worker = None
        self.prepared = {r['query_id']: r for r in read_rows(self.output / 'prepare/prepared.jsonl')}

    def runtime(self):
        if self.worker is None:
            self.worker = WikiRuntime(self.config)
        return self.worker

    def plan(self):
        reference = self.dataset.get('method_reference')
        if reference:
            latest = json.loads((ROOT / reference).read_text())
            for field in ('catboost', 'max_ucb', 'parameter_spaces', 'initial_design', 'online_learning', 'reward', 'quality'):
                if latest[field] != self.config[field]:
                    raise ValueError(f'Wiki method differs from configured reference: {field}')
        result = {'source_queries': len(self.queries), 'vector_queries': len(self.routable),
                  'relational_only_queries': [q.query_id for q in self.queries if not q.blocks],
                  'split_counts': {r: len(ids) for r, ids in self.split.items()},
                  'method': 'CASH + MaxUCB + CatBoost', 'max_ucb': self.config['max_ucb'],
                  'distance': 'cosine', 'embedding_model': self.config['embedding']['model'], 'dimension': self.config['embedding']['dimension'], 'normalized': self.config['embedding']['normalized'],
                  'repeats': self.dataset['repeats'], 'warmup_runs': self.config['measurement']['warmup_runs'],
                  'training_labels': 'Wiki train only; no arXiv checkpoint or held-out feedback',
                  'return_contract': 'composite row primary-key tuples, preserving join multiplicity',
                  'limitations': ['same path/config applied to all vector blocks; no joint plan-space expansion',
                                 'cross-table eligible IDs materialized online in PostgreSQL and included in E2E',
                                 'PRE scalar index modes unavailable unless installed and plan-verified',
                                 'unreferenced vector CTEs retained; adapter overhead included',
                                 'relational-only queries audited with native PostgreSQL, excluded from routing metrics']}
        write(self.output / 'experiment_plan.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    def prepare(self):
        worker = self.runtime()
        health = worker.embedding.health()
        if self.config['embedding']['model'] not in health.get('loaded_models', []):
            raise ValueError('configured Wiki embedding service is unavailable')
        # Compare the raw encoder against existing source headings. The source
        # is a quantized SQLite-lembed release; exact byte equality is not expected.
        rows = worker.sql('SELECT heading_text, heading_text_embedding::text FROM "Headings" WHERE heading_text_embedding IS NOT NULL LIMIT 3')
        checks = []
        for text, serialized in rows:
            vector, _ = worker.embedding.embed(text)
            reference = np.asarray(json.loads(serialized))
            cosine = float(np.dot(vector, reference) / (np.linalg.norm(vector) * np.linalg.norm(reference)))
            ratio = float(np.linalg.norm(vector) / np.linalg.norm(reference))
            checks.append({'text': text, 'cosine': cosine, 'norm_ratio': ratio})
        write(self.output / 'prepare/encoder_validation.json', checks)
        # Released CLIP vectors have non-unit norms. Cosine retrieval compares
        # directions, so require encoder agreement without rejecting equivalent
        # vectors solely because the query client normalizes its output.
        if any(not np.isfinite(c['cosine']) or c['cosine'] < .98 for c in checks):
            raise ValueError('Wiki encoder disagrees with source vectors; refusing benchmark')
        for query in self.queries:
            if query.query_id in self.prepared:
                continue
            vectors = worker.vectors(query)
            truth = worker.exact(query, vectors)
            # Ensure the row-ID adapter keeps the number and identities of the
            # original projected result, including duplicate JOIN rows.
            source_sql, source_params = query.render(vectors, exact=True, ids_only=False)
            original = worker.sql(source_sql, source_params)
            from vec2plan_bandit.experiments.ground_truth_identity import projection_rows_equal
            if not projection_rows_equal(query, [list(r[-1]) for r in original], truth):
                raise ValueError(f'{query.query_id}: source projection and ID projection differ')
            pre_plans = worker.pre_availability(query, vectors) if query.blocks else {}
            post_plans = [worker.explain_post(query, vectors)] if query.blocks else []
            from vec2plan_bandit.execution.relational_pg import VERSION, post_plan_evidence
            stats = worker.statistics(query, vectors)
            row = {'query_id': query.query_id, 'source_hash': query.source_hash, 'topk_row_ids': truth,
                   'vectors': [v.tolist() for v in vectors], 'statistics': stats,
                   'available_pre': list(pre_plans), 'pre_plans': pre_plans, 'post_plans': post_plans,
                   'post_plan_scope': 'whole_query_pg_sql', 'pg_execution_contract': VERSION,
                   'post_plan_evidence': post_plan_evidence(worker, post_plans),
                   'ground_truth_below_k': len(truth) < query.k,
                   'tie_rule': 'source ORDER BY followed by composite local row primary keys'}
            append(self.output / 'prepare/prepared.jsonl', row)
            self.prepared[query.query_id] = row
            print(f'Prepared {query.query_id}: {len(truth)} rows, {len(query.blocks)} vector blocks', flush=True)
        write(self.output / 'prepare/coverage.json', {'total': len(self.queries), 'prepared':len(self.prepared),
               'routable':len(self.routable), 'below_k':[q.query_id for q in self.routable if self.prepared[q.query_id]['ground_truth_below_k']]})

    def features(self):
        stats = {q: row['statistics'] for q, row in self.prepared.items()}
        embeddings = {q: np.mean(row['vectors'], axis=0) for q, row in self.prepared.items() if row['vectors']}
        model_path = self.output / 'train/features.npz'
        if model_path.exists():
            pipeline = WikiFeaturePipeline.load(model_path)
        else:
            pipeline = WikiFeaturePipeline().fit(self.routable, self.split['train'], statistics=stats, embeddings=embeddings)
            pipeline.save(model_path)
            write(self.output / 'train/feature_audit.json', pipeline.audit())
        return pipeline, stats, embeddings

    def request(self, query, pipeline, stats, *, config=None, selector=None, baseline=None, exploration=False, key='', native=False):
        worker = self.runtime()
        cap = self.config['reward']['latency_cap_ms']
        audit = {}
        timing = self.record_stage_timings
        if timing:
            audit.update(stage_timing_protocol='bird-request-stages-v1',
                stage_timing_status={name: 'not_started' for name in
                    ('embedding', 'context_features', 'policy', 'execution')},
                embedding_ms=None, context_features_ms=None, policy_ms=None, execution_ms=None,
                policy_invoked=False)
            if native or (selector is None and baseline is None):
                audit['policy_ms'] = 0.0
                audit['stage_timing_status']['policy'] = 'skipped'
            if native:
                audit['context_features_ms'] = 0.0
                audit['stage_timing_status']['context_features'] = 'skipped'
        def call():
            start = time.perf_counter_ns()
            try:
                if timing:
                    record_request_audit(**audit)
                # Every measurement embeds afresh; the prepared vectors supply
                # truth/training PCA only and are never a latency shortcut.
                with request_stage(audit, 'embedding', timing):
                    vectors = worker.vectors(query)
                embedding_ms = (time.perf_counter_ns() - start) / 1e6
                if native:
                    with request_stage(audit, 'execution', timing):
                        row = worker.native(query, vectors)
                else:
                    with request_stage(audit, 'context_features', timing):
                        context = pipeline.transform_one(query, statistics=stats, embeddings={query.query_id: np.mean(vectors, axis=0)})
                    available_pre = frozenset(self.prepared[query.query_id]['available_pre'])
                    selected = config
                    if selector is not None:
                        if timing:
                            audit['policy_invoked'] = True
                        with request_stage(audit, 'policy', timing):
                            decision = selector.select(context, latency_cap_ms=cap,
                                remaining_budget_ms=max(0, cap - (time.perf_counter_ns()-start)/1e6),
                                request_key=key, available_pre_configurations=available_pre, exploration=exploration)
                        selected = decision.config
                        audit.update(decision_audit(decision))
                    elif baseline is not None:
                        if timing:
                            audit['policy_invoked'] = True
                        with request_stage(audit, 'policy', timing):
                            selected = baseline.select(BaselineRequest(query, context, {p:True for p in ('PRE','POST','FAISS','ACORN')},
                                 cap, max(0,cap-(time.perf_counter_ns()-start)/1e6), available_pre, stats[query.query_id]))
                    if selected.path_id == 'PRE' and pre_configuration_key(selected.params) not in available_pre:
                        raise ValueError('unverified PRE configuration')
                    audit.update(path_id=selected.path_id, requested_params=dict(selected.params), config_id=selected.config_id,
                         query_context=context.tolist(), policy_features=joint_action_features(context, selected,
                             latency_cap_ms=cap, remaining_budget_ms=cap).tolist())
                    record_request_audit(**audit)
                    with request_stage(audit, 'execution', timing):
                        row = worker.execute(query, vectors, selected)
                return {**row, **audit, 'embedding_ms': audit['embedding_ms'] if timing else embedding_ms}
            except Exception as error:
                return {**audit, 'row_ids': [], 'error_type':type(error).__name__, 'error_message':str(error)}
        row = run_with_deadline(call, query, cap, worker.cancel_pending)
        if row.get('timed_out'):
            row['row_ids'] = []
        row.update(query_id=query.query_id, k=query.k, **audit)
        if timing:
            accounted = sum(audit[f'{name}_ms'] or 0.0 for name in audit['stage_timing_status'])
            row['other_ms'] = max(0.0, row['e2e_latency_ms'] - accounted)
            row['stage_timing_complete'] = (not row.get('error_type') and not row.get('timed_out')
                and all(status in ('completed', 'skipped')
                    for status in audit['stage_timing_status'].values()))
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
            write(self.output/'status.json', {'stage':'collect_running', 'query_id':query.query_id,
                'completed':len(done), 'total':len(stream)})
            for _ in range(self.config['measurement']['warmup_runs']):
                self.request(query,pipeline,stats,config=config)
            row = self.request(query,pipeline,stats,config=config)
            row.update(evaluation_mode='train_bootstrap')
            append(journal,row)
            done.add((query.query_id,config.config_id))
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
            write(self.output / 'status.json', {'stage': 'train_query', 'query_id': key,
                'completed': len(learner.feedback_ids), 'total': len(stream)})
            for _ in range(self.config['measurement']['warmup_runs']):
                self.request(query,pipeline,stats,selector=policy,exploration=True,key=key+':warmup')
            row = self.request(query,pipeline,stats,selector=policy,exploration=True,key=key)
            row.update(evaluation_mode='train_exploration')
            if 'policy_features' not in row:
                append(self.output / 'train/selection_failures.jsonl', row)
                raise RuntimeError(f'{key}: train selection failed before features were produced: '
                    f'{row.get("error_type")}: {row.get("error_message")}')
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
        print('Wiki train model frozen',flush=True)

    @staticmethod
    def observation(row):
        return {**{k:row.get(k) for k in ('execution_ms','timed_out','latency_censored','error_type',
                'eligible_for_primary_quality','main_routing_facts','execution_stages_ms')},
                'query_id':row['query_id'],'features':row['policy_features'],'query_context':row['query_context'],
                'path_id':row['path_id'],'quality_target':row['quality_target'],'e2e_latency_ms':row['e2e_latency_ms'],
                'reward':row['reward']}

    def benchmark(self, role):
        from vec2plan_bandit.surrogates.catboost_ensemble import ConstrainedCatBoostSurrogate
        pipeline, stats, _ = self.features()
        directory = self.output / 'train/frozen_model'
        state = json.loads((directory/'policy.json').read_text())
        policy = restore_policy(ConstrainedCatBoostSurrogate.load(directory/'surrogate'),
            self.config['max_ucb'], state)
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
        methods = ['cash_maxucb',*baselines]
        journal = self.output/role/'benchmark_live.jsonl'
        done = {(r['query_id'],r['method_id'],r['repeat']) for r in read_rows(journal)}
        write(self.output/role/'effective_policy.json',state)
        for repeat in range(1,self.dataset['repeats']+1):
            stream = [(self.by_id[q],m) for q in self.split[role] for m in methods]
            random.Random(f'{self.dataset["seed"]}:{role}:{repeat}').shuffle(stream)
            for query,method in stream:
                if (query.query_id,method,repeat) in done:
                    continue
                write(self.output / 'status.json', {'stage': role + '_query', 'query_id': query.query_id,
                    'method_id': method, 'repeat': repeat, 'completed': len(done),
                    'total': len(self.split[role]) * len(methods) * self.dataset['repeats']})
                kwargs = {'key':f'{role}:{repeat}:{query.query_id}'}
                if method=='cash_maxucb':
                    kwargs['selector']=policy
                elif isinstance(baselines[method],dict):
                    spec=baselines[method]
                    if spec['name']=='native-pg':
                        kwargs['native']=True
                    else:
                        params={'index_mode':'no_index','join_strategy':'all'} if spec['config']['path_id']=='PRE' else None
                        kwargs['config']=make_execution_config(spec['config']['path_id'],params)
                else:
                    kwargs['baseline']=baselines[method]
                for _ in range(self.config['measurement']['warmup_runs']):
                    self.request(query,pipeline,stats,**kwargs)
                row=self.request(query,pipeline,stats,**kwargs)
                row.update(method_id=method,repeat=repeat,evaluation_mode='frozen',split=role,
                     recall_threshold=self.config['reward']['recall_threshold'],quality_probability_min=policy.quality_probability_min)
                append(journal,row)
                done.add((query.query_id, method, repeat))
            print(f'{role} repeat {repeat} complete',flush=True)
        if policy.state_dict()!=state:
            raise ValueError('held-out benchmark mutated frozen policy')
        rows=read_rows(journal)
        summaries = {}
        for method in methods:
            values = [r for r in rows if r['method_id'] == method]
            summary = _metric_summary(values)
            recalls = [r['recall_at_k'] for r in values if r['eligible_for_primary_quality']]
            summary['mean_recall_at_k'] = float(np.mean(recalls)) if recalls else None
            summary['timeout_rate'] = float(np.mean([r.get('timed_out', False) for r in values]))
            summaries[method] = summary
        write(self.output/role/'summary.json', summaries)
        self.report()

    def report(self):
        lines=['# Wikipedia-Vec 实验结果','',
               f"CASH + MaxUCB + CatBoost：Recall≥{self.config['reward']['recall_threshold']}、完整k、质量概率门槛{self.config['max_ucb']['quality_probability_min']}、{self.config['reward']['latency_cap_ms']/1000:g}秒E2E。",
               f"Wiki 使用{self.config['embedding']['model']}，{self.config['embedding']['dimension']}维，normalized={self.config['embedding']['normalized']}，Cosine距离。按语义分组划分train/dev/test，仅train反馈用于训练；dev/test均冻结。",
               f'查询覆盖：{len(self.queries)}条SQL，{len(self.routable)}条向量查询；划分数量：'+json.dumps({r:len(v) for r,v in self.split.items()})+'。',
               '返回复合主键行标识，保留JOIN重复行；Recall按最终行标识的多重集合计算，完整k沿用SQL外层LIMIT（无外层LIMIT时使用向量块最大k）。',
               '真值少于k的查询保留在全请求统计中，但排除于可评估质量口径；纯关系查询单独审计。','']
        for role in ('dev','test'):
            path=self.output/role/'summary.json'
            if not path.exists():continue
            lines += [f'## {role}','', '| 方法 | 请求 | 平均Recall | 可评估联合达标率 | 超时率 | E2E P50 ms | E2E P95 ms |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for method,s in json.loads(path.read_text()).items():
                def f(key,percent=False):
                    v=s.get(key)
                    return '—' if v is None else (f'{v:.2%}' if percent else f'{v:.2f}')
                lines.append(f'| {method} | {s["requests"]} | {f("mean_recall_at_k",True)} | {f("joint_success_rate_primary_quality",True)} | {f("timeout_rate",True)} | {f("e2e_latency_p50_ms")} | {f("e2e_latency_p95_ms")} |')
            lines += ['']
        lines += ['延迟分位数仅含完成请求，应同时查看超时率。每条记录均重新embedding，包含特征、策略推理、关系过滤、ANN、SQL回表与结果整理。',
                  '四路复用原参数空间；PRE仅开放实际执行计划验证通过的scan/join配置。跨表过滤通过在线PG物化eligible IDs，开销计入E2E。',
                  '多向量查询为各块使用同一配置，不扩展优化动作空间。对照仅含本地纯路径、PG原生、MaxUCB-BO、MLP、LightGBM，未包含JEV/Text2VectorSQL。',
                  '数据仅100篇文章、6763个向量点；本次结果不能直接外推到大规模Wiki。']
        (self.output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='plan',choices=['plan','prepare','collect','train','dev','test','report','all'])
    parser.add_argument('--dataset',type=Path,required=True)
    args=parser.parse_args()
    run=WikiExperiment(args.dataset)
    stages=['plan','prepare','collect','train','dev','test'] if args.stage=='all' else [args.stage]
    try:
        for stage in stages:
            write(run.output/'status.json',{'stage':stage+'_running'})
            if stage in ('dev','test'):run.benchmark(stage)
            else:getattr(run,stage)()
            write(run.output/'status.json',{'stage':stage+'_complete'})
        write(run.output/'status.json',{'stage':'complete','completed_stages':stages})
    except BaseException as error:
        write(run.output/'status.json',{'stage':'failed','phase':stage,'error_type':type(error).__name__,'error':str(error)})
        raise
    finally:
        if run.worker is not None:run.worker.close()


if __name__=='__main__':
    main()
