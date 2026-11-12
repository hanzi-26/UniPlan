"""Wiki experiment with one shared optional-acceleration algorithm and paired ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from vec2plan_bandit.bird.shared_experiment import SharedExperiment
from vec2plan_bandit.experiments.wikipedia import WikiExperiment, ROOT, write, read_rows, append, restore_policy
from vec2plan_bandit.experiments.optional_acceleration import calibration_row
from vec2plan_bandit.experiments.run import _metric_summary
from vec2plan_bandit.surrogates.catboost_ensemble import ConstrainedCatBoostSurrogate
from vec2plan_bandit.paths.registry import make_execution_config, pre_configuration_key
from vec2plan_bandit.policies.optional_acceleration import OptionalAccelerationRouter, VERSION, FIELDS, describe_query, fit_acceleration
from vec2plan_bandit.baselines import builtin_baselines


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_actions():
    """The same bounded candidate grid for every encoder/data source."""
    configs = [make_execution_config('PRE', {'index_mode': 'no_index', 'join_strategy': 'all'}),
               make_execution_config('POST'), make_execution_config('ACORN')]
    configs += [make_execution_config('FAISS', {'ef_search': ef, 'candidate_multiplier': mult})
                for ef, mult in ((100, 4.), (200, 8.), (400, 16.))]
    return [None, *configs]  # None represents native SQL, not a corpus-specific rule.


class WikiOptionalExperiment(WikiExperiment):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.acceleration_router = None
        self.disabled_acceleration = False
        self.budget_faiss = self.config.get('budget_faiss', {}).get('enabled', False)
        corpus = json.loads((ROOT/self.config['assets']['wiki_receipt']).read_text())
        self.corpus_sizes = {f"{a['table']}.{a['column']}": a['points'] for a in corpus['assets']}
        implementation = [Path(__file__), ROOT/'src/vec2plan_bandit/policies/optional_acceleration.py',
            ROOT/'src/vec2plan_bandit/execution/optional_acceleration.py',
            ROOT/'src/vec2plan_bandit/bird/shared_experiment.py',
            ROOT/'src/vec2plan_bandit/experiments/run.py',
            ROOT/'src/vec2plan_bandit/features/action_features.py',
            ROOT/'src/vec2plan_bandit/policies/max_ucb.py',
            ROOT/'src/vec2plan_bandit/surrogates/catboost_ensemble.py']
        self.binding = {'source_sha256': sha(ROOT/self.dataset['source']),
            'splits_sha256': sha(ROOT/self.dataset['splits']),
            'asset_receipt_sha256': sha(ROOT/self.config['assets']['wiki_receipt']),
            'encoder': self.config['embedding']}
        contract = {'protocol': VERSION, 'binding': self.binding,
            'implementation': {str(p.relative_to(ROOT)): sha(p) for p in implementation},
            'candidate_actions': [dict(path_id=c.path_id, params=dict(c.params), config_id=c.config_id)
                if c else dict(path_id='PG_NATIVE_SQL', params={}, config_id='native-pg') for c in self.calibration_actions()],
            'options': self.config['acceleration_calibration'], 'test_used_for_calibration': False,
            'main_comparison': ['cash_maxucb', 'cash_maxucb_no_acceleration']}
        path = self.output/'acceleration_contract.json'
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError('Acceleration experiment contract changed; use a new directory')
        write(path, contract)

    def calibration_actions(self):
        actions = candidate_actions()
        paths = self.config['acceleration_calibration'].get('candidate_paths')
        if paths is not None and (not paths or set(paths) - {'PG_NATIVE_SQL','PRE','POST','FAISS','ACORN'}):
            raise ValueError('Invalid acceleration candidate paths')
        if self.budget_faiss:
            paths = ['FAISS']
        return [a for a in actions if (a.path_id if a else 'PG_NATIVE_SQL') in paths] if paths is not None else actions

    def describe(self, query, stats):
        return describe_query(query, stats.get(query.query_id),
            corpus_size=sum(self.corpus_sizes[b.asset_key] for b in query.blocks))

    def request(self, query, pipeline, stats, *, config=None, selector=None, baseline=None,
                exploration=False, key='', native=False):
        return SharedExperiment.request(self, query, pipeline, stats, config=config,
            selector=selector, baseline=baseline, exploration=exploration, key=key, native=native,
            acceleration_router=None if self.disabled_acceleration else self.acceleration_router)

    def load_frozen(self,calibrating=False):
        folder = self.output/'train/frozen_model'
        policy = restore_policy(ConstrainedCatBoostSurrogate.load(folder/'surrogate'),
            self.config['max_ucb'], json.loads((folder/'policy.json').read_text()))
        from vec2plan_bandit.experiments.main_inference import configure_main_inference
        configure_main_inference(policy, self.config, self.output, checkpoint=folder,calibrating=calibrating)
        self.binding['frozen_main_sha256'] = sha(folder/'policy.json')
        self.binding['training_sha256'] = sha(self.output/'train/shared_training.jsonl')
        self.binding['feature_pipeline_sha256'] = sha(self.output/'train/features.npz')
        self.binding['surrogate_sha256'] = {str(p.relative_to(folder)): sha(p)
                                          for p in (folder/'surrogate').rglob('*') if p.is_file()}
        return policy

    def measurement_router(self, action, desc):
        """Temporary forced action for cost collection; never a deployed rule.

        Only calibration requests use this object. A failed/underfilled action is
        explicitly marked execution_success=False even if main fallback succeeds.
        Final acceptance always goes through the shared fit_acceleration gate.
        """
        rule = {'key': desc['key'], 'path_id': action.path_id if action else 'PG_NATIVE_SQL',
            'config_id': action.config_id if action else 'native-pg',
            'params': dict(action.params) if action else {}, 'calibrated_e2e_max_ms': 0.}
        for field in FIELDS:
            rule[field+'_min'] = rule[field+'_max'] = desc[field]
        return OptionalAccelerationRouter({'version': VERSION, 'binding': self.binding,
            'rules': [rule], 'fit_protocol': {'recall_threshold': self.config['reward']['recall_threshold'],
                                            'budget_faiss': self.budget_faiss,
                                            'scope': 'temporary forced calibration action'}}, binding=self.binding)

    def calibrate(self):
        policy = (self.load_frozen(calibrating=True) if self.config.get('main_selector_calibration',{}).get('enabled')
                  else self.load_frozen())
        state = policy.state_dict()
        pipeline, stats, _ = self.features()
        journal = self.output/'calibration/measurements.jsonl'
        rows = read_rows(journal)
        done = {(r['split'], r['query_id'], r['mode'], r.get('config_id'), r['repeat']) for r in rows}
        options = self.config['acceleration_calibration']
        tasks = []
        for role in ('train', 'dev'):
            for qid in self.split[role]:
                q, desc = self.by_id[qid], self.describe(self.by_id[qid], stats)
                if desc is None:
                    continue
                for repeat in range(1, options['repeats']+1):
                    tasks.append((role, qid, repeat, 'main', None))
                    for action in self.calibration_actions():
                        if action and action.path_id == 'PRE' and pre_configuration_key(action.params) not in self.prepared[qid]['available_pre']:
                            continue
                        tasks.append((role, qid, repeat, 'candidate', action))
        random.Random(self.dataset['seed']).shuffle(tasks)
        completed = False
        try:
            for role, qid, repeat, mode, action in tasks:
                cid = (action.config_id if action else 'native-pg') if mode == 'candidate' else None
                if (role, qid, mode, cid, repeat) in done:
                    continue
                q = self.by_id[qid]
                desc = self.describe(q, stats)
                self.acceleration_router = self.measurement_router(action, desc) if mode == 'candidate' else None
                write(self.output/'status.json', {'stage': 'calibrating', 'completed': len(done),
                    'total': len(tasks), 'query_id': qid, 'split': role, 'mode': mode})
                key = f'calibration:{role}:{repeat}:{qid}'
                for _ in range(self.config['measurement']['warmup_runs']):
                    self.request(q, pipeline, stats, selector=policy, key=key)
                row = self.request(q, pipeline, stats, selector=policy, key=key)
                facts = None
                if self.config.get('main_lightweight_selector',{}).get('collect_facts',False):
                    from vec2plan_bandit.policies.main_lightweight import routing_facts
                    facts = routing_facts(q,stats.get(qid),embedding_dimension=self.config['embedding']['dimension'],
                                          corpus_size=desc['n'])
                observation = calibration_row(row, query=q, split=role, repeat=repeat, mode=mode,
                    description=desc, routing_overhead_ms=row.get('router_ms', 0.),
                    e2e_includes_routing=True,routing_facts=facts)
                if mode == 'candidate':
                    observation.update(path_id=action.path_id if action else 'PG_NATIVE_SQL',
                        config_id=cid, requested_params=dict(action.params) if action else {},
                        execution_success=bool(row.get('router_fast_accepted')), router_fallback=row.get('router_fallback', False))
                else:
                    # Main identities have no fixed configuration: policy may choose any action.
                    observation.pop('config_id', None)
                append(journal, observation)
                done.add((role, qid, mode, cid, repeat))
            if policy.state_dict() != state:
                raise ValueError('Calibration mutated the frozen main policy')
            artifact = fit_acceleration(read_rows(journal), binding=self.binding,
                train_ids=self.split['train'], dev_ids=self.split['dev'],
                recall_threshold=self.config['reward']['recall_threshold'],
                latency_cap_ms=self.config['reward']['latency_cap_ms'], budget_faiss=self.budget_faiss,
                **{k: options[k] for k in ('min_relative_gain', 'min_train_groups', 'min_dev_groups')})
            write(self.output/'router.json', artifact)
            self.acceleration_router = OptionalAccelerationRouter(artifact, binding=self.binding)
            coverage = {role: sum(self.acceleration_router.select(self.describe(self.by_id[qid], stats),
                available_paths={p: True for p in ('PG_NATIVE_SQL','PRE','POST','FAISS','ACORN')}).rule is not None
                for qid in self.split[role]) for role in ('train','dev')}
            write(self.output/'calibration/summary.json', {'measurements': len(done), 'rules': len(artifact['rules']),
                'coverage_train_dev': coverage, 'rule_sha256': sha(self.output/'router.json')})
            from vec2plan_bandit.experiments.main_inference import fit_auto_main_selector
            fit_auto_main_selector(self.config,self.output,self.split)
            completed = True
        finally:
            # Temporary forced-action objects must never escape calibration.
            if not completed:
                self.acceleration_router = None

    def benchmark(self, role):
        policy = self.load_frozen()
        state = policy.state_dict()
        router_path = self.output/'router.json'
        self.acceleration_router = OptionalAccelerationRouter.load(router_path, binding=self.binding)
        if self.config.get('main_selector_calibration',{}).get('enabled'):
            self.acceleration_router=None  # The calibrated v2 main gate owns early decisions.
        router_hash = sha(router_path)
        pipeline, stats, _ = self.features()
        registry = builtin_baselines()
        observations = [self.observation(r) for r in read_rows(self.output/'train/shared_training.jsonl')]
        requested_methods = self.dataset.get('evaluation_methods')
        baselines = {spec['id']: (spec if spec['name'] in ('fixed-wiki','native-pg') else
            registry.create(spec['name'], spec['config']).fit(observations))
            for spec in json.loads((ROOT/self.dataset['baselines']).read_text())['baselines']
            if requested_methods is None or spec['id'] in requested_methods}
        write(self.output/role/'dynamicann-fit-audit.json', {key: baseline.fit_audit
            for key, baseline in baselines.items() if getattr(baseline, 'name', None) == 'dynamic-ann'})
        # Keep the baseline's original ndarray prediction engine.
        for baseline in baselines.values():
            if hasattr(baseline, 'path_surrogate'):
                baseline.path_surrogate.use_shared_pool = self.config.get('runtime', {}).get('baseline_shared_pool', False)
        methods = requested_methods or ['cash_maxucb', 'cash_maxucb_no_acceleration', *baselines]
        if len(methods) != len(set(methods)) or set(methods) - {'cash_maxucb', 'cash_maxucb_no_acceleration', 'cash_maxucb_legacy', *baselines}:
            raise ValueError('Unknown or duplicate evaluation methods')
        journal = self.output/role/'benchmark_live.jsonl'
        existing = read_rows(journal)
        done = {(r['query_id'], r['method_id'], r['repeat']) for r in existing}
        if len(existing) != len(done) or any(r['query_id'] not in self.split[role] or r['method_id'] not in methods for r in existing):
            raise ValueError('Invalid benchmark resume journal')
        for repeat in range(1, self.dataset['repeats']+1):
            tasks = [(qid, method) for qid in self.split[role] for method in methods]
            random.Random(f'{self.dataset["seed"]}:{role}:{repeat}').shuffle(tasks)
            for qid, method in tasks:
                if (qid, method, repeat) in done:
                    continue
                kwargs = {'key': f'{role}:{repeat}:{qid}'}
                optimized = method != 'cash_maxucb_legacy'
                policy.use_batched_action_features = optimized
                policy.surrogate.use_shared_pool = optimized
                policy.surrogate.use_fast_inference = optimized
                policy.use_main_lightweight = optimized
                self.disabled_acceleration = method not in ('cash_maxucb', 'cash_maxucb_legacy')
                if method.startswith('cash_maxucb'):
                    kwargs['selector'] = policy
                elif isinstance(baselines[method], dict):
                    spec = baselines[method]
                    if spec['name'] == 'native-pg':
                        kwargs['native'] = True
                    else:
                        kwargs['config'] = make_execution_config(spec['config']['path_id'])
                else:
                    kwargs['baseline'] = baselines[method]
                write(self.output/'status.json', {'stage': role+'_running', 'completed': len(done),
                    'total': len(self.split[role])*len(methods)*self.dataset['repeats'],
                    'method_id': method, 'query_id': qid, 'repeat': repeat})
                q = self.by_id[qid]
                for _ in range(self.config['measurement']['warmup_runs']):
                    self.request(q, pipeline, stats, **kwargs)
                row = self.request(q, pipeline, stats, **kwargs)
                row.update(method_id=method, repeat=repeat, split=role, evaluation_mode='frozen', router_sha256=router_hash,
                    recall_threshold=self.config['reward']['recall_threshold'],
                    quality_probability_min=self.config['max_ucb']['quality_probability_min'],
                    inference_engine=(('policy-skipped' if row.get('policy_skipped') else
                        row.get('main_inference_engine', 'main-frozen-forest-v1' if
                            getattr(policy.surrogate,'fast_inference',None) else 'matrix-shared-pool-v1'))
                        if optimized else 'legacy-numpy-v1')
                        if method.startswith('cash_maxucb') else 'original-baseline-v1')
                append(journal, row)
                done.add((qid, method, repeat))
        self.disabled_acceleration = False
        if policy.state_dict() != state or sha(router_path) != router_hash:
            raise ValueError('Model or rules changed during frozen evaluation')
        summaries = {}
        for method in methods:
            values = [r for r in read_rows(journal) if r['method_id'] == method]
            recalls = [r['recall_at_k'] for r in values if r['eligible_for_primary_quality']]
            summary = _metric_summary(values)
            summary.update(mean_recall_at_k=float(np.mean(recalls)) if recalls else None,
                timeout_rate=float(np.mean([r.get('timed_out', False) for r in values])),
                router_hits=sum(r.get('router_hit', False) for r in values),
                fast_accepted=sum(r.get('router_fast_accepted', False) for r in values),
                router_fallbacks=sum(r.get('router_fallback', False) for r in values))
            summaries[method] = summary
        write(self.output/role/'summary.json', summaries)
        self.report()

    def report(self):
        super().report()
        rows = []
        for role in ('dev','test'):
            path = self.output/role/'summary.json'
            if path.exists():
                for method, summary in json.loads(path.read_text()).items():
                    rows.append({'split': role, 'method_id': method,
                        **{k: v for k,v in summary.items() if not isinstance(v, (dict,list))}})
        if rows:
            fields = sorted(set().union(*(r.keys() for r in rows)))
            with (self.output/'comparison.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        path = self.output/'report.md'
        text = path.read_text()
        if self.dataset.get('evaluation_methods'):
            text += '\n本次评测方法：' + '、'.join(self.dataset['evaluation_methods']) + '。所有列出的方法均同轮重新测量。主策略复用v4冻结模型、split和前置规则；cash_maxucb使用特征矩阵化与共享Pool。\n'
            if any(not m.startswith('cash_maxucb') for m in self.dataset['evaluation_methods']):
                text += 'baseline配置保持原样，仅使用相同train数据拟合；其推理接口保留原实现，不使用主策略新增的共享Pool或前置规则。\n'
        text += '\n主实验使用官方 LAION CLIP/512；补充实验使用 MiniLM/384，具体模型见 run_contract.json。\n'
        if self.budget_faiss:
            text += '预算低于100ms时，仅启用train/dev通过召回和完整k验证的FAISS配置。主策略概率门槛0.8；规则采用全部校准观测通过的经验验证，不代表概率保证。\n'
            if self.dataset.get('evaluation_methods'):
                text += '本次复用v4冻结模型，未重新训练；原训练预算10000ms，评测预算100ms。\n'
            elif 'training_latency_cap_ms' in self.config:
                text += f"按新版split重新训练模型和校准规则；训练预算{self.config['training_latency_cap_ms']}ms，评测预算{self.config['reward']['latency_cap_ms']}ms。\n"
                text += 'split根据结构与无质量标签统计重新分层。本次为查看旧test后调整设计的探索实验，不能视为新的未接触测试集。\n'
            else:
                text += '共享v2冻结模型（原训练预算10秒），本次请求总预算100ms。test是已查看过的固定split，本次属于消融实验。\n'
        else:
            text += f"前置候选路径：{self.config['acceleration_calibration'].get('candidate_paths', '全部注册路径')}。train选择参数，dev仅接受或拒绝；所有校准观测满足召回与完整k条件，train/dev均要求至少{self.config['acceleration_calibration']['min_relative_gain']:.0%}相对E2E收益。\n"
            text += '启用条件为质量验证和实测收益，不依赖剩余预算低于100ms。test是此前已查看的固定split，本次为探索对照。\n'
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan','prepare','collect','train','calibrate','dev','test','report','all'], default='all', nargs='?')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reuse-frozen', action='store_true', help='Reuse copied, verified prepare/train artifacts')
    parser.add_argument('--reuse-rules', action='store_true', help='Reuse verified frozen rules without recalibration')
    args = parser.parse_args()
    run = WikiOptionalExperiment(args.dataset)
    stages = ['plan','prepare','collect','train','calibrate','dev','test','report'] if args.stage == 'all' else [args.stage]
    if args.reuse_frozen:
        receipt = json.loads((run.output/'reuse_receipt.json').read_text())
        for relative, digest in receipt['copied_sha256'].items():
            if sha(run.output/relative) != digest:
                raise ValueError('Reused artifact changed: ' + relative)
        run.load_frozen()
        stages = [s for s in stages if s not in ('prepare','collect','train')]
    if args.reuse_rules:
        if not args.reuse_frozen or 'router.json' not in receipt['copied_sha256']:
            raise ValueError('Rule reuse requires verified frozen-model and router receipts')
        OptionalAccelerationRouter.load(run.output/'router.json', binding=run.binding)
        stages = [s for s in stages if s != 'calibrate']
    stage = 'initializing'
    try:
        for stage in stages:
            write(run.output/'status.json', {'stage': stage+'_running'})
            cap = run.config['reward']['latency_cap_ms']
            try:
                if stage in ('collect', 'train'):
                    run.config['reward']['latency_cap_ms'] = run.config.get('training_latency_cap_ms', cap)
                if stage in ('dev','test'):
                    run.benchmark(stage)
                else:
                    getattr(run, stage)()
            finally:
                run.config['reward']['latency_cap_ms'] = cap
        write(run.output/'status.json', {'stage':'complete', 'completed_stages':stages})
    except BaseException as error:
        write(run.output/'status.json', {'stage':'failed', 'phase':stage,
            'error_type':type(error).__name__, 'error':str(error)})
        raise
    finally:
        if run.worker is not None:
            run.worker.close()


if __name__ == '__main__':
    main()
