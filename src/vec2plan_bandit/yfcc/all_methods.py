"""All ten methods, with dataset-only DynamicANN statistics adaptation."""
import copy
import json
import random

import numpy as np

from vec2plan_bandit.baselines import builtin_baselines
from vec2plan_bandit.baselines.dynamic_ann import DynamicANNBaseline
from vec2plan_bandit.baselines.dynamic_ann_statistics import DensityCatalog,VERSION,DENSITY_DEFINITION,profile_index
from vec2plan_bandit.experiments.wikipedia import append,read_rows
from vec2plan_bandit.paths.registry import make_execution_config
from vec2plan_bandit.policies.main_lightweight import MainNativePGConfig
from .experiment import Experiment
from .data import write,sha

LABELS={'cash_maxucb':'Text2Plan','maxucb_bo':'MaxUCB+BO','mlp':'MLP','lightgbm':'LightGBM',
    'pure_pre':'PRE','pure_post':'POST','pure_faiss':'FAISS','pure_acorn':'ACORN','dynamic_ann':'DynamicANN','pg_native':'PostgreSQL',
    'jev_router':'JEV'}


class CorpusView:
    """Read-only source vector view for profiling; builds no ANN index."""
    def __init__(self,raw):self.raw=raw;self.ntotal,self.d=raw.shape
    def reconstruct_batch(self,ids):return self.raw[ids].astype('float32')


class YFCCDynamicANN(DynamicANNBaseline):
    def request_statistics(self,worker,query,vectors,statistics):
        stats=dict(statistics or {})
        if self._catalog is None:self._catalog=DensityCatalog(self.config['density_artifact'])
        block=dict(total_rows=len(worker.raw),embedding_dimension=1920,k=query.k,
            estimated_selectivity=stats['estimated_selectivity'])
        block.update(self._catalog.resolve('YFCCImages.image_vec',vectors[0],len(worker.raw),1920,worker.shards/'manifest.json'))
        stats['dynamicann_blocks']=[block];return stats


class AllMethodsExperiment(Experiment):
    @property
    def extra(self):return self.output/'all-methods'

    def measure10(self,q,pipeline,stats,**kwargs):
        key=kwargs.pop('key','')
        warm=self.request(q,pipeline,stats,key=key+':warmup',**kwargs)
        return self.request(q,pipeline,stats,key=key,**kwargs)

    @property
    def labels(self):
        return json.loads((self.extra/'contract.json').read_text()).get('labels',LABELS)

    def baseline_setup(self):
        from joblib.externals import cloudpickle
        from threadpoolctl import threadpool_limits
        contract=json.loads((self.extra/'contract.json').read_text())
        specs=json.loads((self.extra/'baselines.json').read_text())['baselines']
        if 'dynamic_ann' not in self.labels:
            observations=[self.observation(r) for r in read_rows(self.output/'train/shared_training.jsonl')]
            if any(r['query_id'] not in self.split['train'] for r in observations):raise ValueError('Held-out baseline training data')
            models={};registry=builtin_baselines()
            for spec in specs:
                write(self.output/'status.json',dict(stage='baseline_setup',substage='fit_'+spec['id']))
                with threadpool_limits(limits=4):models[spec['id']]=registry.create(spec['name'],spec['config']).fit(observations)
            artifact=self.extra/'frozen-baselines.pkl'
            with artifact.open('wb') as stream:cloudpickle.dump(models,stream)
            write(self.extra/'baseline-artifact-manifest.json',{str(artifact):sha(artifact)})
            write(self.extra/'baseline-setup.complete.json',dict(methods=list(models),training_only=True,
                training_query_ids=self.split['train'],test_labels_used=False))
            return
        pipeline,stats,_=self.features();worker=self.runtime()
        density=self.extra/'density.json'
        if not density.exists():
            write(self.output/'status.json',dict(stage='baseline_setup',substage='dynamicann_density',completed=0,total=1))
            with threadpool_limits(limits=4):
                asset=profile_index(CorpusView(worker.raw),worker.shards/'manifest.json',
                    sample_size=contract['density_sample_size'],clusters=contract['density_clusters'],seed=42)
            asset['input_vectors']='original uint8 corpus converted to float32; raw L2, no normalization'
            write(density,dict(version=VERSION,density_definition=DENSITY_DEFINITION,assets={'YFCCImages.image_vec':asset},query_labels_used=False))
        dynamic_spec=next(s for s in specs if s['id']=='dynamic_ann')
        options={**dynamic_spec['config']};options.pop('calibration_artifact',None)
        probe=YFCCDynamicANN({**options,'coefficient_mode':'paper'}).fit([])
        dynamic_path=self.extra/'dynamicann-train-measurements.jsonl'
        done={(r['query_id'],r['path_id'],r['repeat']) for r in read_rows(dynamic_path)}
        ids=sorted(self.split['train']);random.Random(42).shuffle(ids);ids=ids[:contract['dynamicann_train_queries']]
        tasks=[(qid,action,repeat) for qid in ids for action in (probe.pre,probe.post) for repeat in range(1,contract['dynamicann_repeats']+1)]
        random.Random(42).shuffle(tasks)
        old_cap=self.config['reward']['latency_cap_ms'];self.config['reward']['latency_cap_ms']=contract['dynamicann_offline_budget_ms']
        try:
            for qid,action,repeat in tasks:
                if (qid,action.path_id,repeat) in done:continue
                write(self.output/'status.json',dict(stage='baseline_setup',substage='dynamicann_calibration',completed=len(done),total=len(tasks)))
                q=self.by_id[qid]
                dynamic_stats=probe.request_statistics(worker,q,worker.vectors(q),stats[qid])
                row=self.measure10(q,pipeline,stats,config=action,key=f'dynamicann-fit:{qid}:{action.path_id}:{repeat}')
                row.update(query_statistics=dynamic_stats,group_id=q.group_id,split='train',repeat=repeat,
                    k=q.k,requested_params=action.params,path_id=action.path_id,config_id=action.config_id,
                    offline_calibration_budget_ms=contract['dynamicann_offline_budget_ms'])
                append(dynamic_path,row);done.add((qid,action.path_id,repeat))
        finally:self.config['reward']['latency_cap_ms']=old_cap
        dynamic=YFCCDynamicANN(options).fit(read_rows(dynamic_path))
        dynamic.save(self.extra/'dynamicann-calibrated.json')
        observations=[self.observation(r) for r in read_rows(self.output/'train/shared_training.jsonl')]
        if any(r['query_id'] not in self.split['train'] for r in observations):raise ValueError('Held-out baseline training data')
        models={};registry=builtin_baselines()
        for spec in specs:
            if spec['id'] not in ('maxucb_bo','mlp','lightgbm','jev_router'):continue
            write(self.output/'status.json',dict(stage='baseline_setup',substage='fit_'+spec['id']))
            with threadpool_limits(limits=4):models[spec['id']]=registry.create(spec['name'],spec['config']).fit(observations)
        models['dynamic_ann']=YFCCDynamicANN(dynamic_spec['config']).fit([])
        with (self.extra/'frozen-baselines.pkl').open('wb') as stream:cloudpickle.dump(models,stream)
        write(self.extra/'baseline-artifact-manifest.json',{str(p):sha(p) for p in
            (density,self.extra/'dynamicann-calibrated.json',self.extra/'frozen-baselines.pkl')})
        write(self.extra/'baseline-setup.complete.json',dict(methods=list(models),training_only=True,
            training_query_ids=self.split['train'],dynamicann_query_ids=ids,
            dynamicann_offline_budget_ms=contract['dynamicann_offline_budget_ms'],formal_budget_ms=old_cap,
            dynamicann_adapter='existing project cost model; raw YFCC corpus density; unindexed PG PRE/POST',
            test_labels_used=False))

    def evaluate(self,role):
        from joblib.externals import cloudpickle
        from vec2plan_bandit.experiments.run import _metric_summary
        for path,digest in json.loads((self.extra/'baseline-artifact-manifest.json').read_text()).items():
            if sha(path)!=digest:raise ValueError('Frozen baseline artifact changed')
        self.config['main_competitive_early']['sha256']=json.loads((self.output/'calibration.complete.json').read_text())['artifact_sha256']
        policy=self.policy();state=copy.deepcopy(policy.state_dict());pipeline,stats,_=self.features()
        with (self.extra/'frozen-baselines.pkl').open('rb') as stream:models=cloudpickle.load(stream)
        model_digest=sha(self.extra/'frozen-baselines.pkl')
        fixed={'pure_pre':make_execution_config('PRE'),'pure_post':make_execution_config('POST'),
            'pure_faiss':make_execution_config('FAISS'),'pure_acorn':make_execution_config('ACORN')}
        journal=self.output/role/'benchmark_live.jsonl';existing=read_rows(journal)
        done={(r['method_id'],r['query_id'],r['repeat']) for r in existing}
        tasks=[(qid,repeat,method) for qid in self.split[role] for repeat in range(1,self.dataset['repeats']+1) for method in self.labels]
        random.Random(42).shuffle(tasks)
        for qid,repeat,method in tasks:
            if (method,qid,repeat) in done:continue
            write(self.output/'status.json',dict(stage=role,completed=len(done),total=len(tasks),query_id=qid,method=method))
            kwargs={'selector':policy} if method=='cash_maxucb' else ({'config':fixed[method],'context_required':False}
                if method in fixed else ({'native':True,'context_required':False} if method=='pg_native' else {'baseline':models[method]}))
            row=self.measure10(self.by_id[qid],pipeline,stats,key=f'{role}:{qid}:{repeat}:{method}',**kwargs)
            row.update(split=role,repeat=repeat,method_id=method,method_label=self.labels[method]);append(journal,row)
            done.add((method,qid,repeat))
        if state!=policy.state_dict() or sha(self.extra/'frozen-baselines.pkl')!=model_digest:raise ValueError('Frozen evaluation changed a model')
        rows=read_rows(journal)
        if len(rows)!=len(tasks):raise ValueError('Incomplete or duplicated ten-method evaluation')
        summaries={}
        for method,label in self.labels.items():
            rr=[r for r in rows if r['method_id']==method];durations=[r['e2e_latency_ms'] for r in rr]
            summaries[method]=dict(_metric_summary(rr),label=label,requests=len(rr),
                observed_request_duration_mean_ms=float(np.mean(durations)),
                all_request_p50_ms=float(np.percentile(durations,50)),
                serial_e2e_qps=1000*len(durations)/sum(durations) if sum(durations)>0 else None,
                all_request_p95_ms=float(np.percentile(durations,95)),
                timeout_rate=float(np.mean([r.get('timed_out',False) for r in rr])))
        write(self.output/role/'summary.json',summaries)
        write(self.output/f'{role}.complete.json',dict(requests=len(rows),methods=list(self.labels)))
