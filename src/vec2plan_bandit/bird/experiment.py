import json,hashlib,time,os,fcntl,traceback
from pathlib import Path
import numpy as np
from .shared_experiment import SharedExperiment,write,append,read_rows
from vec2plan_bandit.queries.wikipedia_sql_adapter import grouped_split
from vec2plan_bandit.features.feature_pipeline import FeaturePipeline
from vec2plan_bandit.experiments.collector import _plan_nodes
from .adapter import Query,parse
from .runtime import Runtime
from sqlglot import exp
from vec2plan_bandit.queries.lightweight import sql_static_facts

class BirdFeatures(FeaturePipeline):
 def _base_vector(self,q,statistics):
  # Shared feature contract; dataset-specific column flag is deliberately zero.
  v=super()._base_vector(q,statistics);where_count,join_count,*_=sql_static_facts(q.source_sql)
  v[5]=where_count;v[14]=np.log1p(join_count);v[15]=bool(join_count)
  return v

class Experiment(SharedExperiment):
 def __init__(self,path):
  self.dataset=json.loads(path.read_text());self.config=json.loads(Path(self.dataset['experiment_config']).read_text());self.output=Path(self.dataset['output_dir']);self.output.mkdir(parents=True,exist_ok=True)
  self.worker=None;self.prepared={r['query_id']:r for r in read_rows(self.output/'prepare/prepared.jsonl')}
  rows=json.loads(Path(self.dataset['source']).read_text());metadata={r['db_id']:r for r in json.loads(Path(self.dataset['metadata']).read_text())};assets=json.loads(Path(self.dataset['catalog']).read_text())['assets']
  schema_path=Path(self.dataset.get('schema',self.output/'database-columns.json'))
  if schema_path.exists():
   for db,columns in json.loads(schema_path.read_text()).items():metadata[db]['_server_columns']=columns
  queries=[];errors=[]
  for i,r in enumerate(rows,1):
   try:
    q=Query(i,r,metadata[r['db_id']],assets)
    q.deterministic_limit_ordering=self.config.get('runtime',{}).get('deterministic_limit_ordering',False)
    q.render([np.zeros(384) for b in q.blocks],exact=True)
    for b in q.blocks:q.eligible_sql(b,[np.zeros(384) for b in q.blocks])
    queries.append(q)
   except Exception as e:errors.append({'query_id':f'B{i:03d}','error':str(e)})
  write(self.output/'query-adapter-audit.json',{'source_queries':len(rows),'accepted':len(queries),'errors':errors})
  if errors:raise ValueError(f'{len(errors)} queries require additional SQL support; see query-adapter-audit.json')
  self.queries=queries;self.by_id={q.query_id:q for q in queries};self.routable=[q for q in queries if q.blocks]
  split_path=Path(self.dataset['splits'])
  if self.dataset.get('split_policy')=='curated_frozen':
   from vec2plan_bandit.queries.wikipedia_sql_adapter import semantic_query_groups
   from vec2plan_bandit.queries.grouped_stratification import validate_grouped_split
   self.split=json.loads(split_path.read_text())
   validate_grouped_split(self.split,semantic_query_groups(self.routable))
  else:self.split=grouped_split(self.routable,self.dataset['seed'])
  if split_path.exists() and json.loads(split_path.read_text())!=self.split:raise ValueError('Existing BIRD split changed; use a new output directory')
  write(split_path,self.split)
  contract={'dataset':self.dataset,'config':self.config,'source_sha256':hashlib.sha256(Path(self.dataset['source']).read_bytes()).hexdigest(),'methods':['cash_maxucb','maxucb_bo','mlp','lightgbm'],'split_counts':{k:len(v) for k,v in self.split.items()},'identity':'CTID text tuples propagated through CTEs; preserve join multiplicity','gpu':'MiniLM encoder on CUDA_VISIBLE_DEVICES=1; original CPU selectors and HNSW retained','test_labels_used_for_training':False,'source_queries':len(self.queries),'vector_queries':len(self.routable),'relational_only_queries':[q.query_id for q in self.queries if not q.blocks],'tie_rule':'ORDER BY distance then CTID/group identity; equal distances with different row identities count as strict recall misses','probability_constraint':'quality_probability_min=0.85 applies to constrained main method; baselines retain original selection logic'}
  contract['methods']=[self.dataset.get('main_method','text2plan'),*[b['id'] for b in json.loads(Path(self.dataset['baselines']).read_text())['baselines']]]
  contract['probability_constraint']=f"quality_probability_min={self.config['max_ucb']['quality_probability_min']} applies to text2plan; baselines retain original selection logic"
  contract['gpu']={'embedding':self.config['embedding']['url'],'catboost':self.config['catboost'].get('task_type','CPU'),'catboost_devices':self.config['catboost'].get('devices'),'retrieval':'PostgreSQL, CPU FAISS HNSW, Qdrant ACORN'}
  contract['timing_protocol']='bird-five-spans-v1'
  contract['context_statistics_source']='prepared snapshot; transform/PCA measured online'
  contract['native_pg']='cosine SQL with row identity projection; PostgreSQL default access and join settings'
  contract['ann_execution']=self.config.get('ann_execution',{'mode':'database_join'})
  contract['implementation_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
  contract['baseline_sha256']=hashlib.sha256(Path(self.dataset['baselines']).read_bytes()).hexdigest()
  contract_path=self.output/'run_contract.json'
  if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
   previous=json.loads(contract_path.read_text());changed=[k for k in sorted(set(previous)|set(contract)) if previous.get(k)!=contract.get(k)]
   implementations=[name for name in sorted(set(previous.get('implementation_sha256',{}))|set(contract['implementation_sha256'])) if previous.get('implementation_sha256',{}).get(name)!=contract['implementation_sha256'].get(name)]
   write(self.output/'run-contract-mismatch.json',{'changed_fields':changed,'changed_implementation_files':implementations,'existing_contract_preserved':True})
   raise ValueError(f'BIRD run contract changed: {changed}; implementation files: {implementations}; use a new frozen source/output directory')
  write(contract_path,contract)
 def runtime(self):
  if self.worker is None:self.worker=Runtime(self.config)
  return self.worker
 def request(self,query,*args,**kwargs):
  if self.config.get('relational_pre',{}).get('enabled',False):
   fixed=kwargs.get('config');selector=kwargs.get('selector')
   pg_only=(kwargs.get('native',False) or fixed is not None and fixed.path_id in ('PRE','POST')
    or selector is not None and self.config.get('main_execution',{}).get('required_path')=='PRE')
   self.runtime().prepare_query(query,load_ann=not pg_only)
  else:self.runtime().prepare_query(query)
  return super().request(query,*args,**kwargs)
 def prepare(self):
  worker=self.runtime()
  preparation_vectors=None
  if self.dataset.get('preparation_embeddings'):
   from vec2plan_bandit.execution.preparation_embeddings import load_preparation_embeddings
   preparation_vectors=load_preparation_embeddings(self.dataset['preparation_embeddings'],
       embedding=self.config['embedding'],source=self.dataset['source'],queries=self.routable)
   for key in [q for q in self.split['train'] if q in preparation_vectors][:2]:
    if not np.allclose(worker.vectors(self.by_id[key]),preparation_vectors[key],atol=1e-5):
     raise ValueError('Preparation embeddings differ from the live encoder')
  if self.dataset.get('preparation_source') and not self.prepared:
   source=Path(self.dataset['preparation_source'])
   previous=json.loads((source.parent.parent/'run_contract.json').read_text())
   current_hash=hashlib.sha256(Path(self.dataset['source']).read_bytes()).hexdigest()
   if previous['source_sha256']!=current_hash or previous['config']['distance_metric']!='cosine':raise ValueError('Preparation source differs from current BIRD queries/distance')
   for key in ('model','dimension','normalized'):
    if previous['config']['embedding'][key]!=self.config['embedding'][key]:raise ValueError('Preparation encoder differs')
   rows=read_rows(source)
   if {r['query_id'] for r in rows}!={q.query_id for q in self.routable}:raise ValueError('Preparation query coverage differs')
   for row in rows:
    q=self.by_id[row['query_id']]
    if row['source_hash']!=q.source_hash or len(row['vectors'])!=len(q.blocks):raise ValueError('Preparation query/vector mismatch')
    vectors=np.asarray(row['vectors'])
    if vectors.shape!=(len(q.blocks),384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-3):raise ValueError('Invalid preparation vectors')
   # Verify fresh HTTP embeddings and physical row identities on train probes.
   for key in self.dataset.get('preparation_probe_queries',['B114','B115']):
    q=self.by_id[key]
    if key not in self.split['train']:raise ValueError('Preparation probes must use train queries')
    reference=next(r for r in rows if r['query_id']==key)
    worker.prepare_query(q);vectors=worker.vectors(q)
    if not np.allclose(vectors,reference['vectors'],atol=1e-5) or worker.exact(q,vectors)!=reference['topk_row_ids']:raise ValueError('Preparation snapshot/encoder probe changed')
   destination=self.output/'prepare/prepared.jsonl';destination.parent.mkdir(parents=True,exist_ok=True)
   import shutil
   shutil.copyfile(source,destination)
   self.prepared={r['query_id']:r for r in rows}
   write(self.output/'prepare/reuse-manifest.json',{'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'records':len(rows),'scope':'exact truth, vectors, statistics and verified plans only; new train labels and model are collected separately','encoder_and_identity_probes':self.dataset.get('preparation_probe_queries',['B114','B115'])})
   print(f'Reused {len(rows)} verified BIRD preparation records; no training model reused',flush=True)
  for q in self.routable:
   if q.query_id in self.prepared:continue
   worker.prepare_query(q)
   vectors=preparation_vectors[q.query_id] if preparation_vectors is not None else worker.vectors(q)
   truth=worker.exact(q,vectors)
   source,binds=q.render(vectors,exact=True,ids_only=False);original=worker.sql(source,binds,worker.pre_settings({'index_mode':'no_index','join_strategy':'all'}))
   from vec2plan_bandit.experiments.ground_truth_identity import projection_rows_equal
   if not projection_rows_equal(q,[list(r[-1]) for r in original],truth):raise ValueError(q.query_id+': projection identities differ')
   pre=worker.pre_availability(q,vectors)
   # Audit the deployed whole-query SQL, not a legacy block probe with ID arrays.
   post=[worker.explain_post(q,vectors)] if q.blocks else []
   from vec2plan_bandit.execution.relational_pg import VERSION,post_plan_evidence
   row={'query_id':q.query_id,'source_hash':q.source_hash,'topk_row_ids':truth,'vectors':[v.tolist() for v in vectors],'statistics':worker.statistics(q,vectors),'available_pre':list(pre),'pre_plans':pre,'post_plans':post,'post_plan_scope':'whole_query_pg_sql','post_plan_evidence':post_plan_evidence(worker,post),'pg_execution_contract':VERSION,'ground_truth_below_k':len(truth)<q.k}
   append(self.output/'prepare/prepared.jsonl',row);self.prepared[q.query_id]=row
   print('Prepared',q.query_id,'truth rows',len(truth),flush=True)
  write(self.output/'prepare/coverage.json',{'total':len(self.queries),'prepared':len(self.prepared),'routable':len(self.routable)})
 def features(self):
  FeatureClass=BirdFeatures
  if self.config.get('runtime',{}).get('relational_plan_features',False):
   from vec2plan_bandit.features.bird_relational import BirdRelationalFeatures
   FeatureClass=BirdRelationalFeatures
  if self.config.get('runtime',{}).get('relational_plan_profiles',False):
   from vec2plan_bandit.features.bird_relational import BirdRelationalPlanFeatures
   FeatureClass=BirdRelationalPlanFeatures
  stats={q:r['statistics'] for q,r in self.prepared.items()};embeddings={q:np.mean(r['vectors'],axis=0) for q,r in self.prepared.items() if r['vectors']}
  embeddings={q:v/np.linalg.norm(v) for q,v in embeddings.items()}
  model=self.output/'train/features.npz'
  if model.exists():pipeline=FeatureClass.load(model)
  else:
   options={'feature_version':'pre-execution-relational-plan-v1'} if FeatureClass is not BirdFeatures else {}
   if self.config.get('runtime',{}).get('relational_plan_profiles',False):options['feature_version']='pre-execution-relational-plan-profiles-v2'
   pipeline=FeatureClass(**options).fit(self.routable,self.split['train'],statistics=stats,embeddings=embeddings);pipeline.save(model);write(self.output/'train/feature_audit.json',pipeline.audit())
  return pipeline,stats,embeddings
 def report(self):
  import csv
  results={role:json.loads((self.output/role/'summary.json').read_text()) for role in ['dev','test'] if (self.output/role/'summary.json').exists()}
  write(self.output/'results.json',results)
  fields=['split','method','requests','quality_pass_rate','joint_success_rate','timeout_rate','e2e_latency_mean_ms','observed_e2e_mean_ms',*[n+'_mean_ms' for n in ('embedding','context_features','policy_selection','execution','others')]]
  with (self.output/'comparison.csv').open('w',newline='') as handle:
   writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
   for role,summaries in results.items():
    for method,summary in summaries.items():
     writer.writerow({'split':role,'method':method,**{k:summary.get(k) for k in fields[2:7]},'observed_e2e_mean_ms':summary['observed_request_duration_mean_ms'],**{n+'_mean_ms':v['mean'] for n,v in summary.get('timing_ms',{}).items()}})
