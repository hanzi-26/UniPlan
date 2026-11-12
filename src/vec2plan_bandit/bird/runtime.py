from vec2plan_bandit.execution.detailed_timing import stage
import json,sqlite3,time,math
from pathlib import Path
import numpy as np
import psycopg
from psycopg import sql
from qdrant_client import QdrantClient,models
from vec2plan_bandit.assets.bird import pg_parameters,_json_value
from vec2plan_bandit.experiments.collector import _psycopg_sql,_plan_nodes
from vec2plan_bandit.paths.registry import pre_configuration_key
from vec2plan_bandit.execution.wikipedia_runtime import WikiRuntime
from .adapter import quote
from vec2plan_bandit.execution.relational_pg import execute_pg_sql, explain_post_sql, post_settings

class Runtime:
 def pre_settings(self,params):
  settings=WikiRuntime.pre_settings(self,params)
  options=self.config.get('relational_pre',{})
  if options.get('enabled',False):
   workers=options.get('parallel_workers',0)
   if type(workers) is not int or not 0<=workers<=3:raise ValueError('PRE parallel workers must be 0..3')
   settings=settings+[f'SET LOCAL max_parallel_workers_per_gather={workers}',"SET LOCAL work_mem='32MB'"]
  return settings
 def __init__(self,config):
  import faiss
  from vec2plan_bandit.services.embedding import EmbeddingServiceClient
  import requests
  self.config=config;self.conn=None;self.db=None;self.faiss=faiss;faiss.omp_set_num_threads(config.get('runtime', {}).get('cpu_threads',4))
  self.http=requests.Session();self.http.trust_env=False
  embedding=config['embedding']
  self.embedding=EmbeddingServiceClient(url=embedding['url'],model=embedding['model'],dimension=embedding['dimension'],normalize=embedding['normalized'],timeout_seconds=embedding['timeout_seconds'],session=self.http)
  self.qdrant=QdrantClient(url=config['acorn']['url'],timeout=60,prefer_grpc=False,trust_env=False)
  self.cache={}
  self.covering=None
  self.sql_query_count=0
 def open(self,db):
  if self.db!=db or self.conn is None or self.conn.closed:
   if self.conn:self.conn.close()
   params=pg_parameters(Path(self.config['database']['config_path']));params['dbname']=db
   options='-c default_transaction_read_only=on -c statement_timeout=600000'
   namespace=self.config['database'].get('search_path_schema')
   if namespace:
    import re
    if not re.fullmatch('[a-z][a-z0-9_]*',namespace):raise ValueError('Invalid experiment schema')
    options+=' -c search_path='+namespace+',public'
   self.conn=psycopg.connect(**params,options=options);self.db=db
 def sql(self,statement,parameters=None,settings=()):
  parallel=self.config.get('runtime',{}).get('symmetric_pg_parallel_workers')
  if parallel is not None:
   if type(parallel) is not int or not 0<=parallel<=3:raise ValueError('PG worker budget must be 0..3')
   settings=[f'SET LOCAL max_parallel_workers_per_gather={parallel}',"SET LOCAL work_mem='32MB'"]+list(settings)
  if self.config.get('runtime',{}).get('batch_local_settings',False):
   from vec2plan_bandit.execution.sql_batch import execute_batched_local
   self.sql_query_count=getattr(self,'sql_query_count',0)+1
   if isinstance(statement,sql.Composable):statement=statement.as_string(self.conn)
   return execute_batched_local(self.conn,statement,parameters,settings,
     unified_resources=self.config.get('runtime',{}).get('unified_resources',False))
  self.sql_query_count=getattr(self,'sql_query_count',0)+1
  if isinstance(statement,sql.Composable):statement=statement.as_string(self.conn)
  from vec2plan_bandit.execution.timing import evaluation_deadline_ns
  try:
   with self.conn.cursor() as c:
    if self.config.get('runtime',{}).get('unified_resources',False):
     c.execute('SET LOCAL max_parallel_workers_per_gather=0');c.execute('SET LOCAL jit=off')
    for setting in settings:c.execute(setting)
    deadline=evaluation_deadline_ns()
    if deadline is not None:c.execute(f'SET LOCAL statement_timeout={max(1,math.ceil((deadline-time.perf_counter_ns())/1e6))}')
    c.execute(_psycopg_sql(statement) if parameters else statement,parameters or None);return c.fetchall()
  finally:
   if not self.conn.closed:self.conn.rollback()
 def prepare_query(self,q,*,load_ann=True):
  self.open(q.db_id)
  if not load_ann:
   if self.config.get('runtime',{}).get('release_ann_on_pg',True):
    for key in list(self.cache):self.cache.pop(key)['mapping'].close()
   return
  wanted={b.asset_key for b in q.blocks}
  for k in list(self.cache):
   if k not in wanted:self.cache.pop(k)['mapping'].close()
  for b in q.blocks:
   if b.asset_key in self.cache:continue
   folder=Path(self.config['assets']['faiss_dir'])/b.asset_key
   mapping=sqlite3.connect(f'file:{folder}/rows.sqlite?mode=ro',uri=True,check_same_thread=False)
   manifest=json.loads((folder/'manifest.json').read_text())
   self.cache[b.asset_key]={'index':self.faiss.read_index(str(folder/'index.faiss')),'mapping':mapping,'manifest':manifest,'reverse':None}
   duplicate=self.config.get('raw_duplicate_identities',{})
   if duplicate.get('enabled',False):
    from .duplicate_vectors import DuplicateIdentities
    from vec2plan_bandit.assets.bird import sha256
    root=Path(duplicate['directory'])
    if sha256(root/'manifest.json')!=duplicate['sha256']:raise ValueError('Duplicate metadata manifest changed')
    receipt=json.loads((root/'manifest.json').read_text())
    if b.asset_key in receipt['assets']:
     binding=dict(index_manifest_sha256=sha256(folder/'manifest.json'),
       covering_manifest_sha256=sha256(Path(self.config['ann_execution']['snapshot_dir'])/'manifest.json'))
     self.cache[b.asset_key]['duplicates']=DuplicateIdentities(root/b.asset_key,binding=binding,
       count=self.cache[b.asset_key]['index'].ntotal)
 def vectors(self,q):
  return [self.embedding.embed(b.text)[0] for b in q.blocks]
 def keys(self,b,ids):
  from vec2plan_bandit.execution.identity import fetch_point_values
  rows=fetch_point_values(self.cache[b.asset_key]['mapping'],'rows','key_json',ids)
  return [json.loads(rows[int(i)]) for i in ids if int(i) in rows]
 @staticmethod
 def canonical(k):return json.dumps(k,sort_keys=True,ensure_ascii=False,separators=(',',':'))
 def eligible(self,q,vectors):
  out=[]
  for b in q.blocks:
   stmt=q.eligible_sql(b,vectors)
   fields=b.asset['primary_key'] or ['ctid']
   out.append(None if stmt is None else [{k:_json_value(v) for k,v in zip(fields,row)} for row in self.sql(*stmt)])
  return out
 def statistics(self,q,vectors):
  allowed=self.eligible(q,vectors);counts=[self.cache[b.asset_key]['index'].ntotal if a is None else len(a) for b,a in zip(q.blocks,allowed)]
  total=sum(self.cache[b.asset_key]['index'].ntotal for b in q.blocks)
  return {'candidate_count':sum(counts),'estimated_selectivity':sum(counts)/total if total else 0,'block_candidate_counts':counts}
 def pre_availability(self,q,vectors):
  from vec2plan_bandit.policies.max_ucb import sample_path_configurations
  from vec2plan_bandit.execution.relational_pre import render_pre_sql
  statement,binds,pre_audit=render_pre_sql(self,q,vectors);out={}
  configs=sample_path_configurations('PRE',1,seed=42)
  if self.config.get('relational_pre',{}).get('enabled',False):
   from vec2plan_bandit.paths.registry import include_optimizer_pre
   configs=include_optimizer_pre(configs,{f'optimizer:{j}' for j in ('all','hash','merge','nested_loop')})
  for c in configs:
   if c.params['index_mode']!='no_index' and not self.config.get('runtime',{}).get('pre_access_enabled',False):continue
   plan=self.sql('EXPLAIN (FORMAT JSON) '+statement,binds,self.pre_settings(c.params))[0][0]
   if self.config.get('runtime',{}).get('pre_access_enabled',False):
    from vec2plan_bandit.execution.pre_access import validate_plan
    names=[n.get('Index Name') for n in _plan_nodes(plan) if n.get('Index Name')]
    methods=dict(self.sql('SELECT c.relname,am.amname FROM pg_class c JOIN pg_am am ON am.oid=c.relam WHERE c.relname=ANY(%(index_names)s)',{'index_names':names})) if names else {}
    lineage={}
    if self.config.get('runtime',{}).get('view_aware_pre',False):
     from vec2plan_bandit.execution.relation_lineage import ctid_mapping_lineage
     cache=self.__dict__.setdefault('_pre_lineage',{})
     schema=self.config['database'].get('search_path_schema','public')
     for b in q.blocks:
      key=(q.db_id,schema,b.table)
      if key not in cache:cache[key]=ctid_mapping_lineage(self.conn,schema,b.table)
      if cache[key] is not None:lineage[b.table]=cache[key]
     self.conn.rollback()
    tables={b.table for b in q.blocks}|{r['relation'].split('.')[-1] for r in pre_audit.get('compact_relation_rewrites',[])}
    try:validate_plan(plan,c.params,methods,tables,relation_lineage=lineage)
    except ValueError as error:
     self.__dict__.setdefault('pre_rejections',{})[(q.query_id,pre_configuration_key(c.params))]=str(error);continue
    out[pre_configuration_key(c.params)]=plan;continue
   joins=[n['Node Type'] for n in _plan_nodes(plan) if n['Node Type'] in ('Hash Join','Merge Join','Nested Loop')]
   expected={'hash':'Hash Join','merge':'Merge Join','nested_loop':'Nested Loop'}.get(c.params['join_strategy'])
   if c.params['join_strategy']=='all' or joins and all(j==expected for j in joins):out[pre_configuration_key(c.params)]=plan
  if not out:raise ValueError('No verified PRE plan')
  return out
 def post_candidates(self,b,vector,allowed,params,explain=False):
  # Legacy block-level preparation probe; deployed POST uses whole-query SQL.
  fields=b.asset['primary_key'] or ['ctid'];parts=[]
  for key in fields:parts.extend(["'"+key.replace("'","''")+"'",quote(key)+('::text' if key=='ctid' else '')])
  statement='SELECT '+','.join(quote(k)+('::text' if k=='ctid' else '') for k in fields)+' FROM '+quote(b.table)
  binds={'v':'['+','.join(str(float(x)) for x in vector)+']','k':b.k}
  if allowed is not None:
   statement+=' WHERE jsonb_build_object('+','.join(parts)+') = ANY(%(allowed)s::jsonb[])';binds['allowed']=[json.dumps(k) for k in allowed]
  statement+=' ORDER BY '+quote(b.column)+' <=> %(v)s::vector LIMIT %(k)s'
  settings=post_settings(params)
  rows=self.sql(('EXPLAIN (FORMAT JSON) ' if explain else '')+statement,binds,settings)
  return rows[0][0] if explain else [{k:_json_value(v) for k,v in zip(fields,r)} for r in rows]
 def explain_post(self,q,vectors,params=None):
  self.open(q.db_id)
  return explain_post_sql(self,q,vectors,params)
 def execute(self,q,vectors,c):
  if c.path_id in {'PRE','POST'}:
   if (self.config.get('relational_pre',{}).get('enabled',False)
       and self.config.get('runtime',{}).get('release_ann_on_pg',True)):
    # These paths use no external ANN asset. Release an unused multi-GB FAISS
    # heap before the PG scan, for every selector, rather than competing with
    # PostgreSQL's file cache. This cost is inside the timed execution.
    for key in list(self.cache):self.cache.pop(key)['mapping'].close()
   self.open(q.db_id)
   return execute_pg_sql(self,q,vectors,c)
  if c.path_id in {'FAISS','ACORN'} and self.config.get('ann_execution',{}).get('mode')=='covering_ids':
   return self.execute_covering(q,vectors,c)
  self.open(q.db_id);path=c.path_id;p=c.params
  if path in {'FAISS','ACORN'}:
   allowed=self.eligible(q,vectors);candidates=[]
   for b,vector,a in zip(q.blocks,vectors,allowed):
    asset=self.cache[b.asset_key]
    if a==[]:candidates.append([]);continue
    if path=='FAISS':
     index=asset['index'];index.hnsw.efSearch=p['ef_search'];count=min(index.ntotal,max(b.k,math.ceil(b.k*p['candidate_multiplier'])))
     with stage(self,'ann_search_ms'):
      _,ids=index.search(np.asarray(vector,dtype=np.float32).reshape(1,-1),count)
     keys=self.keys(b,[i for i in ids[0] if i>=0])
     if a is not None:
      valid={self.canonical(k) for k in a};keys=[k for k in keys if self.canonical(k) in valid]
     # SQL applies the block's original ORDER BY/LIMIT after reranking this
     # pool. Truncating here discards near ties before reference arithmetic.
    elif path=='ACORN':
     filt=None
     if a is not None:
      if asset['reverse'] is None:asset['reverse']={self.canonical(json.loads(k)):i for i,k in asset['mapping'].execute('SELECT point_id,key_json FROM rows')}
      ids=[asset['reverse'][self.canonical(k)] for k in a if self.canonical(k) in asset['reverse']]
      if not ids:candidates.append([]);continue
      filt=models.Filter(must=[models.HasIdCondition(has_id=ids)])
     collection='bird_'+b.asset_key.replace('-','_')+'_acorn_v1'
     with stage(self,'ann_search_ms'):
      points=self.qdrant.query_points(collection,query=vector.tolist(),query_filter=filt,limit=b.k,with_payload=False,with_vectors=False,search_params=models.SearchParams(hnsw_ef=p['hnsw_ef'],exact=False,acorn=models.AcornSearchParams(enable=True,max_selectivity=1.0))).points
     keys=self.keys(b,[x.id for x in points])
    else:raise ValueError(path)
    candidates.append(keys)
   with stage(self,'sql_compile_ms'):statement,binds=q.render(vectors,candidates=candidates,exact=True)
   with stage(self,'result_assembly_ms'):rows=self.sql(statement,binds)
  else:raise ValueError('Unknown relational execution path')
  return {'row_ids':[list(r[0]) for r in rows],'path_id':path,'requested_params':dict(p),'config_id':c.config_id,'fallback_used':False,'distance_metric':'cosine'}
 def execute_covering(self,q,vectors,c):
  import locale
  from .covering import CoveringRuntime
  from vec2plan_bandit.assets.bird import sha256
  if self.covering is None:
   ann=self.config['ann_execution']
   self.covering=CoveringRuntime(ann['snapshot_dir'],source_sha256=sha256(Path(ann['source'])),catalog_sha256=sha256(Path(ann['catalog'])),schema_sha256=sha256(Path(ann['schema'])) if ann.get('schema') else None)
   if ann.get('query_id_map'):
    # Curated records retain SQL/source hashes but have dense new IDs. Only
    # remap label-free representation hints; never mutate the shared snapshot.
    old=self.covering.manifest.get('eligibility_estimates',{})
    ids=json.loads(Path(ann['query_id_map']).read_text())
    self.covering.manifest={**self.covering.manifest,'eligibility_estimates':{
     r['query_id']:old[r['original_query_id']] for r in ids if r['original_query_id'] in old}}
  before=self.sql_query_count
  from vec2plan_bandit.queries.lightweight import simple_retrieval
  with stage(self,'semantic_check_ms'):
   simple=(self.config.get('runtime',{}).get('lightweight_execution',True) and
     simple_retrieval(q).direct_unfiltered)
  self.covering.bind(q,vectors,self.cache,simple=simple)
  if simple:
   from vec2plan_bandit.execution.simple_ann import search_simple_ann,assemble_index_rows
   import locale
   block=q.blocks[0];vector=vectors[0];index=self.cache[block.asset_key]['index']
   def remote():
    result=self.qdrant.query_points('bird_'+block.asset_key.replace('-','_')+'_acorn_v1',
      query=vector.tolist(),query_filter=None,limit=block.k,with_payload=False,with_vectors=False,
      search_params=models.SearchParams(hnsw_ef=c.params['hnsw_ef'],exact=False,
        acorn=models.AcornSearchParams(enable=True,max_selectivity=1.0)))
    return [int(p.id) for p in result.points]
   ids=search_simple_ann(self,index,vector,block.k,c,remote)
   duplicates=self.cache[block.asset_key].get('duplicates') if c.path_id=='FAISS' else None
   duplicate_expanded=False
   if duplicates is not None:
    with stage(self,'id_mapping_ms'):ids,duplicate_expanded=duplicates.expand(ids,k=q.k)
   with stage(self,'id_mapping_ms'):
    ctids=self.covering.ctids(block,ids)
   with stage(self,'result_assembly_ms'):
    row_ids=assemble_index_rows(index,vector,ids,[[v] for v in ctids],k=q.k,
      tie_key=lambda row:(locale.strxfrm(row[0]),row[0]))
   if self.sql_query_count!=before:raise RuntimeError('Simple ANN unexpectedly queried PostgreSQL')
   return {'row_ids':row_ids,'path_id':c.path_id,'requested_params':dict(c.params),
     'config_id':c.config_id,'fallback_used':False,'distance_metric':'cosine',
     'ann_result_mode':'covering_ids','postgres_queries':0,'return_to_pg':False,
     'ann_point_ids':[ids],'execution_simplified':True,
     'raw_duplicate_identity_expansion':duplicate_expanded,
     'simplification_reason':'single_unfiltered_topk'}
  candidates=[];search_ms=0;eligibility_ms=0
  for block,vector in zip(q.blocks,vectors):
   asset=self.cache[block.asset_key]
   if c.path_id=='FAISS':
    index=asset['index'];index.hnsw.efSearch=c.params['ef_search']
    count=min(index.ntotal,max(block.k,math.ceil(block.k*c.params['candidate_multiplier'])))
    started=time.perf_counter_ns()
    with stage(self,'ann_search_ms'):
     _,ids=index.search(np.asarray(vector,dtype=np.float32).reshape(1,-1),count)
    ids=[int(i) for i in ids[0] if i>=0];search_ms+=(time.perf_counter_ns()-started)/1e6
    started=time.perf_counter_ns()
    with stage(self,'candidate_filter_ms'):valid=self.covering.filter_ids(block,ids)
    eligibility_ms+=(time.perf_counter_ns()-started)/1e6
    eligible=[i for i in ids if i in valid]
    ctids=self.covering.ctids(block,eligible)
    candidates.append([i for _,_,_,i in sorted((self.covering.distance(block.number,ctid),locale.strxfrm(ctid),ctid,i)
                      for ctid,i in zip(ctids,eligible))][:block.k])
   else:
    started=time.perf_counter_ns()
    with stage(self,'filter_prepare_ms'):allowed=self.covering.eligible_ids(block)
    eligibility_ms+=(time.perf_counter_ns()-started)/1e6
    if allowed==[]:candidates.append([]);continue
    if allowed is None:filt=None
    elif block.number in self.covering.excluded:filt=models.Filter(must_not=[models.HasIdCondition(has_id=allowed)])
    else:filt=models.Filter(must=[models.HasIdCondition(has_id=allowed)])
    started=time.perf_counter_ns()
    with stage(self,'ann_search_ms'):
     points=self.qdrant.query_points('bird_'+block.asset_key.replace('-','_')+'_acorn_v1',query=vector.tolist(),
       query_filter=filt,limit=block.k,with_payload=False,with_vectors=False,
       search_params=models.SearchParams(hnsw_ef=c.params['hnsw_ef'],exact=False,
        acorn=models.AcornSearchParams(enable=True,max_selectivity=1.0))).points
    search_ms+=(time.perf_counter_ns()-started)/1e6;candidates.append([int(p.id) for p in points])
  started=time.perf_counter_ns()
  with stage(self,'result_assembly_ms'):row_ids=self.covering.assemble(candidates)
  materialization_ms=(time.perf_counter_ns()-started)/1e6
  if self.sql_query_count!=before:raise RuntimeError('Covering ANN unexpectedly queried PostgreSQL')
  return {'row_ids':row_ids,'path_id':c.path_id,'requested_params':dict(c.params),'config_id':c.config_id,
    'fallback_used':False,'distance_metric':'cosine','ann_result_mode':'covering_ids',
    'postgres_queries':0,'return_to_pg':False,'ann_point_ids':candidates,
    'ann_search_ms':search_ms,'local_eligibility_ms':eligibility_ms,'local_materialization_ms':materialization_ms}
 def exact(self,q,vectors):return [list(r[0]) for r in self.sql(*q.render(vectors,exact=True),settings=self.pre_settings({'index_mode':'no_index','join_strategy':'all'}))]
 def native(self,q,vectors):
  self.open(q.db_id)
  # Retain vector operators and let PostgreSQL choose its own access/join plan.
  with stage(self,'sql_compile_ms'):statement,binds=q.render(vectors,exact=False)
  rows=self.sql(statement,binds)
  return {'row_ids':[list(r[0]) for r in rows],'path_id':'PG_NATIVE_SQL',
          'requested_params':{},'config_id':'native-pg','fallback_used':False,'distance_metric':'cosine'}
 def cancel_pending(self):
  if self.covering:self.covering.interrupt()
  if self.conn and not self.conn.closed:self.conn.cancel_safe(timeout=1);self.conn.close()
 def close(self):
  if self.covering:self.covering.close()
  if self.conn:self.conn.close()
  for asset in self.cache.values():asset['mapping'].close()
  self.cache.clear();self.http.close()
  self.qdrant.close()
