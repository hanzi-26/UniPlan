"""Dataset adaptation of the frozen Wiki final method, no YFCC test feedback."""
import copy
import json
from pathlib import Path
import random

import numpy as np

from vec2plan_bandit.bird.shared_experiment import SharedExperiment
from vec2plan_bandit.experiments.wikipedia import read_rows,append,restore_policy
from vec2plan_bandit.features.wikipedia_features import WikiFeaturePipeline
from vec2plan_bandit.features.feature_pipeline import FeaturePipeline
from vec2plan_bandit.features.projector import TrainOnlyPCA
from vec2plan_bandit.surrogates.catboost_ensemble import ConstrainedCatBoostSurrogate
from vec2plan_bandit.experiments.main_inference import configure_main_inference,runtime_routing_facts
from vec2plan_bandit.paths.registry import make_execution_config
from vec2plan_bandit.policies.main_lightweight import MainNativePGConfig
from vec2plan_bandit.execution.main_sql_candidates import configuration
from .data import load_queries,write,sha
from .runtime import Runtime


class YFCCFeatures(WikiFeaturePipeline):
    def fit(self,records,train_query_ids,*,statistics=None,embeddings=None,split='train'):
        FeaturePipeline.fit(self,records,train_query_ids,statistics=statistics,embeddings=None,split=split)
        matrix=np.vstack([embeddings[q] for q in self.fit_query_ids]).astype('float64')
        if matrix.shape[1] not in (192,1920):raise ValueError('Expected 192D or 1920D features')
        # Shared request construction normalizes embeddings for context/PCA,
        # independently of the distance used by the execution backend.
        matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
        self.pca_=TrainOnlyPCA(self.pca_dimensions).fit(matrix)
        projected=self.pca_.transform(matrix)
        self.mean_=np.concatenate((self.mean_,projected.mean(axis=0)))
        self.scale_=np.concatenate((self.scale_,np.where(projected.std(axis=0)>1e-12,projected.std(axis=0),1.)))
        self.embedding_enabled_=True;self.feature_version=f'yfcc-wiki-normalized-context-{matrix.shape[1]}d-v3'
        return self


class Experiment(SharedExperiment):
    def __init__(self,output):
        self.output=Path(output);self.config=json.loads((self.output/'experiment.json').read_text())
        self.dataset=json.loads((self.output/'dataset.json').read_text())
        self.queries=load_queries(self.dataset['source'],metric=self.config.get('distance_metric','cosine').lower(),expected_count=self.dataset.get('query_count',500));self.routable=self.queries
        self.by_id={q.query_id:q for q in self.queries};self.split=json.loads((self.output/'splits.json').read_text())
        self.prepared={r['query_id']:r for r in read_rows(self.output/'prepare/prepared.jsonl')}
        self.worker=None;self.corpus_sizes={'YFCCImages.image_vec':10000000}

    def runtime(self):
        if self.worker is None:
            write(self.output/'runtime-startup.json',dict(stage='initializing',faiss_load_mode=self.config['yfcc'].get('faiss_load_mode','one_shard')))
            self.worker=Runtime(self.config)
            write(self.output/'runtime-startup.json',dict(stage='ready',**self.worker.initialization))
        return self.worker

    def request(self,*args,**kwargs):
        row=super().request(*args,**kwargs)
        key=kwargs.get('key','')
        if key.endswith(':warmup'):
            append(self.output/'warmups.jsonl',dict(query_id=args[0].query_id,key=key,
                e2e_latency_ms=row['e2e_latency_ms'],timed_out=row.get('timed_out',False),
                error_type=row.get('error_type'),yfcc_execution=row.get('yfcc_execution')))
        return row

    def collect(self):
        from vec2plan_bandit.policies.max_ucb import initial_design
        from vec2plan_bandit.paths.registry import pre_configuration_key
        pipeline,stats,_=self.features();journal=self.output/'collect/configuration_matrix.jsonl'
        done={(r['query_id'],r['config_id']) for r in read_rows(journal)}
        design=initial_design(self.config['initial_design']['points_per_numeric_path'],seed=self.dataset['seed'])
        tasks=[(self.by_id[qid],c) for qid in self.split['train'] for c in design
            if c.path_id!='PRE' or pre_configuration_key(c.params) in self.prepared[qid]['available_pre']]
        random.Random(self.dataset['seed']).shuffle(tasks)
        for query,config in tasks:
            if (query.query_id,config.config_id) in done:continue
            write(self.output/'status.json',dict(stage='collect',completed=len(done),total=len(tasks),query_id=query.query_id,path=config.path_id))
            row=self.measure(query,pipeline,stats,config=config,key=f'bootstrap:{query.query_id}:{config.config_id}')
            row['evaluation_mode']='train_bootstrap'
            if 'policy_features' not in row:raise RuntimeError('Collection failed before selection: '+str(row))
            append(journal,row);done.add((query.query_id,config.config_id))

    def features(self):
        stats={q:r['statistics'] for q,r in self.prepared.items()}
        embeddings={q:np.asarray(r['vectors'][0]) for q,r in self.prepared.items()}
        path=self.output/'train/features.npz'
        if path.exists():pipeline=YFCCFeatures.load(path)
        else:
            pipeline=YFCCFeatures().fit(self.queries,self.split['train'],statistics=stats,embeddings=embeddings)
            pipeline.save(path);write(self.output/'train/feature_audit.json',pipeline.audit())
        return pipeline,stats,embeddings

    def plans(self):
        worker=self.runtime()
        for q in self.queries:
            row=self.prepared[q.query_id]
            row['available_pre']=list(worker.pre_availability(q,worker.vectors(q)))
            write(self.output/'status.json',dict(stage='verifying_pre_plans',query_id=q.query_id))
        (self.output/'prepare/prepared.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.prepared.values()))
        write(self.output/'plans.complete.json',dict(queries=len(self.queries)))

    def policy(self,calibrating=False):
        checkpoint=self.output/'train/frozen_model'
        policy=restore_policy(ConstrainedCatBoostSurrogate.load(checkpoint/'surrogate'),self.config['max_ucb'],json.loads((checkpoint/'policy.json').read_text()))
        configure_main_inference(policy,self.config,self.output,checkpoint=checkpoint,calibrating=calibrating)
        return policy

    def measure(self,q,pipeline,stats,*,config=None,selector=None,native=False,key=''):
        # Persist warmup outcomes; no slow method is selectively removed.
        warm=self.request(q,pipeline,stats,config=config,selector=selector,native=native,key=key+':warmup')
        return self.request(q,pipeline,stats,config=config,selector=selector,native=native,key=key)

    def calibrate_final(self):
        from vec2plan_bandit.policies.scoped_early import fit_scoped_early
        from vec2plan_bandit.policies.anchor_competition import fit_anchor_competition
        from vec2plan_bandit.experiments.main_priority import fit_priority
        from types import SimpleNamespace
        policy=self.policy(calibrating=True);pipeline,stats,_=self.features()
        binding=json.loads((self.output/'main-lightweight-binding.json').read_text())
        # Same eight main candidates as the released Wiki anchor competition.
        actions=[configuration('FAISS_SQL',dict(search_mode='hnsw',ef_search=100,candidate_multiplier=4.))]
        actions += [configuration('FAISS_SQL',dict(search_mode='flat',candidate_multiplier=float(m))) for m in (4,16,256)]
        actions += [configuration('POST_SQL',dict(ef_search=e,max_scan_tuples=100000)) for e in (100,400)]
        actions += [configuration('PG_SQL',{}),configuration('EXACT_SQL',{})]
        fallback_actions=[make_execution_config('PRE'),make_execution_config('POST'),MainNativePGConfig()]
        specs=[('main',None),*[('candidate',a) for a in fallback_actions],
               *[('sql_candidate',a) for a in actions],('anchor',make_execution_config('POST')),('anchor',MainNativePGConfig())]
        journal=self.output/'calibration/measurements.jsonl';existing=read_rows(journal)
        done={(r['split'],r['query_id'],r['mode'],r.get('config_id'),r['repeat']) for r in existing}
        tasks=[(role,qid,repeat,mode,a) for role in ('train','dev') for qid in self.split[role]
               for repeat in range(1,4) for mode,a in specs]
        random.Random(42).shuffle(tasks)
        for role,qid,repeat,mode,action in tasks:
            cid=action.config_id if action else None
            if (role,qid,mode,cid,repeat) in done:continue
            q=self.by_id[qid]
            if action and action.path_id=='PRE':
                from vec2plan_bandit.paths.registry import pre_configuration_key
                if pre_configuration_key(action.params) not in self.prepared[qid]['available_pre']:
                    raise ValueError('Default PRE plan unavailable; cannot silently skip a matched candidate')
            write(self.output/'status.json',dict(stage='calibrating',completed=len(done),total=len(tasks),query_id=qid))
            if mode=='sql_candidate':
                # Shared request requires a selector opt-in for SQL-only actions.
                class Forced:
                    main_sql_candidates_early=True
                    main_compact_early=True
                    def select(self,*args,**kwargs):
                        return SimpleNamespace(config=action,constraints_satisfied=True,quality_lcb=0.,predicted_e2e_latency_ms=0.,feasible_candidate_count=1,evaluated_candidates=1,selection_mode='empirical_early_fallback')
                selector=Forced();row=self.measure(q,pipeline,stats,selector=selector,key=f'cal:{role}:{qid}:{repeat}:{cid}')
            else:
                row=self.measure(q,pipeline,stats,selector=policy if mode=='main' else None,
                    config=action if action and action.path_id!='PG_NATIVE_SQL' else None,
                    native=bool(action and action.path_id=='PG_NATIVE_SQL'),key=f'cal:{role}:{qid}:{repeat}:{cid}')
            f=runtime_routing_facts(q,stats[qid],self.runtime(),self.config,self.corpus_sizes)
            row.update(split=role,repeat=repeat,mode=mode,config_id=cid,routing_facts=f,
                semantic_group=q.group_id,measurement_scope='e2e',source_hash=q.source_hash)
            if action:row.update(path_id=action.path_id,requested_params=action.params)
            append(journal,row);done.add((role,qid,mode,cid,repeat))
        rows=read_rows(journal);truth={qid:self.prepared[qid]['topk_row_ids'] for role in ('train','dev') for qid in self.split[role]}
        fallback=fit_scoped_early([r for r in rows if r['mode'] in ('main','candidate')],truth_by_query=truth,
            train_ids=self.split['train'],dev_ids=self.split['dev'],binding=binding,coarse_partition_nodes=[dict(leaf=0)])
        anchor_rows=[]
        for r in rows:
            if r['mode'] in ('sql_candidate','anchor'):
                r=copy.deepcopy(r)
                if r['mode']=='sql_candidate':r['mode']='candidate'
                anchor_rows.append(r)
        artifact=fit_anchor_competition(anchor_rows,truth_by_query=truth,train_ids=self.split['train'],dev_ids=self.split['dev'],binding=binding,fallback=fallback,recall_threshold=self.config['reward']['recall_threshold'])
        write(self.output/'competitive-early.json',artifact)
        write(self.output/'main-selector.json',fit_priority([],train_ids=self.split['train'],dev_ids=self.split['dev'],binding=binding,recall_threshold=self.config['reward']['recall_threshold'],probability_min=self.config['max_ucb']['quality_probability_min'],latency_cap_ms=self.config['reward']['latency_cap_ms']))
        # Avoid rewriting the frozen config: the artifact hash is bound by a
        # separate manifest and validated by the phase supervisor.
        write(self.output/'calibration.complete.json',dict(rows=len(rows),artifact_sha256=sha(self.output/'competitive-early.json'),test_labels_used=False))

    def evaluate(self,role):
        from vec2plan_bandit.experiments.run import _metric_summary
        self.config['main_competitive_early']['sha256']=json.loads((self.output/'calibration.complete.json').read_text())['artifact_sha256']
        policy=self.policy();state=copy.deepcopy(policy.state_dict());pipeline,stats,_=self.features()
        journal=self.output/role/'benchmark_live.jsonl';rows=read_rows(journal)
        done={(r['method_id'],r['query_id'],r['repeat']) for r in rows}
        specs={'cash_maxucb':None,'pure_pre':make_execution_config('PRE'),'pure_post':make_execution_config('POST'),
               'pure_faiss':make_execution_config('FAISS'),'pure_acorn':make_execution_config('ACORN'),'pg_native':MainNativePGConfig()}
        tasks=[(qid,repeat,method) for qid in self.split[role] for repeat in range(1,6) for method in specs]
        random.Random(42).shuffle(tasks)
        for qid,repeat,method in tasks:
            if (method,qid,repeat) in done:continue
            write(self.output/'status.json',dict(stage=role,completed=len(done),total=len(tasks),query_id=qid,method=method))
            q=self.by_id[qid];a=specs[method]
            row=self.measure(q,pipeline,stats,selector=policy if a is None else None,
                config=a if a and a.path_id!='PG_NATIVE_SQL' else None,native=method=='pg_native',key=f'{role}:{qid}:{repeat}:{method}')
            row.update(split=role,repeat=repeat,method_id=method);append(journal,row);done.add((method,qid,repeat))
        if state!=policy.state_dict():raise ValueError('Evaluation mutated the model')
        rows=read_rows(journal)
        if len(rows)!=len(tasks):raise ValueError('Incomplete evaluation')
        summaries={}
        for method in specs:
            rr=[r for r in rows if r['method_id']==method];duration=[r['e2e_latency_ms'] for r in rr]
            summaries[method]=dict(_metric_summary(rr),observed_request_duration_mean_ms=float(np.mean(duration)),
                all_request_p50_ms=float(np.percentile(duration,50)),all_request_p95_ms=float(np.percentile(duration,95)),timeout_rate=float(np.mean([r.get('timed_out',False) for r in rr])))
        write(self.output/role/'summary.json',summaries)
        write(self.output/f'{role}.complete.json',dict(requests=len(rows),methods=list(specs)))
