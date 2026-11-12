"""Cosine YFCC execution, retaining Wiki's SQL candidate/PG contracts."""
import json
import math
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed,wait

import numpy as np

from vec2plan_bandit.execution.wikipedia_runtime import WikiRuntime
from vec2plan_bandit.execution.relational_pg import configured_post_settings,execute_pg_sql
from vec2plan_bandit.execution.timing import evaluation_deadline_ns,EvaluationDeadlineExceeded
from .data import normalize,vectors,COLLECTION,write
from .execution_support import ExecutionSupport
from .shard_merge import merge_shards


def check_deadline():
    deadline=evaluation_deadline_ns()
    if deadline is not None and time.perf_counter_ns()>=deadline:raise EvaluationDeadlineExceeded()


class Runtime(ExecutionSupport,WikiRuntime):
    def __init__(self,config):
        import faiss
        from qdrant_client import QdrantClient
        self.config=config;self.conn=None
        self._init_execution_support(config)
        self.shard_workers=int(config['yfcc'].get('shard_workers',1))
        if not 1<=self.shard_workers<=4:raise ValueError('Shard workers must fit the four-CPU budget')
        self.search_pool=ThreadPoolExecutor(max_workers=self.shard_workers) if self.shard_workers>1 else None
        self.metric=config.get('distance_metric','cosine').lower()
        faiss.omp_set_num_threads(4)
        self.dimension=config.get('embedding',{}).get('dimension',1920)
        self.raw=vectors() if self.dimension==1920 else vectors(self.dimension);self.shards=Path(config['yfcc']['faiss_dir'])
        self.collection=config['yfcc'].get('collection',COLLECTION)
        self.assets={'YFCCImages.image_vec':dict(points=10000000,collection=self.collection)}
        self.corpus_sizes={'YFCCImages.image_vec':10000000}
        self.cache=None
        self.mapped_indexes={}
        self.manifest=json.loads((self.shards/'manifest.json').read_text())
        if self.manifest['metric'].lower()!=self.metric or self.manifest['normalized']!=(self.metric=='cosine'):raise ValueError('Index metric mismatch')
        self.initialization=dict(faiss_load_mode=config['yfcc'].get('faiss_load_mode','one_shard'),mapped_shards=0)
        if self.initialization['faiss_load_mode']=='mmap_all':
            started=time.perf_counter()
            for shard in self.manifest['shards']:
                index=faiss.read_index(str(self.shards/shard['file']),faiss.IO_FLAG_MMAP_IFC|faiss.IO_FLAG_READ_ONLY)
                self.validate_index(index,shard)
                self.mapped_indexes[shard['file']]=index
                status=config['yfcc'].get('runtime_status_path')
                if status:write(status,dict(stage='initializing',faiss_load_mode='mmap_all',
                    mapped_shards=len(self.mapped_indexes),total_shards=len(self.manifest['shards']),
                    elapsed_s=time.perf_counter()-started))
            self.initialization.update(mapped_shards=len(self.mapped_indexes),faiss_initialization_seconds=time.perf_counter()-started,
                initialization_scope='offline runtime startup; query page faults remain inside E2E')
        self.qdrant=(None if config['yfcc'].get('defer_qdrant_client',False) else self._qdrant_client())

    def _qdrant_client(self):
        from qdrant_client import QdrantClient
        config=self.config
        return QdrantClient(url=config['acorn']['url'],grpc_port=config['acorn']['grpc_port'],prefer_grpc=True,trust_env=False,timeout=6,check_compatibility=False)

    def validate_index(self,index,shard):
        import faiss
        expected=faiss.METRIC_INNER_PRODUCT if self.metric=='cosine' else faiss.METRIC_L2
        if index.d!=getattr(self,'dimension',1920) or index.ntotal!=shard['count'] or index.metric_type!=expected:
            raise ValueError('Faiss shard dimensions, row count or metric mismatch')

    def close(self):
        try:
            if self.search_pool is not None:self.search_pool.shutdown(wait=True,cancel_futures=True)
            if self.qdrant is not None:super().close()
            elif self.conn is not None:self.conn.close()
        finally:
            self.mapped_indexes.clear();self.cache=None

    def vectors(self,query):
        value=self.raw[query.item['reference_vector_name']]
        return [normalize(value) if self.metric=='cosine' else value.astype('float32')]

    def pre_availability(self, query, values):
        result = super().pre_availability(query, values)
        if self.config.get('yfcc', {}).get('scalar_index_optimizer_pre', False):
            if not self.config.get('runtime', {}).get('pre_access_enabled', False):
                raise ValueError('YFCC scalar optimizer PRE requires verified scalar access')
            from .scalar_indexes import optimizer_plan
            from vec2plan_bandit.paths.registry import pre_configuration_key
            action, plan, _ = optimizer_plan(self, query, values)
            result[pre_configuration_key(action.params)] = plan
        return result

    def ann(self,vector,count,ef):
        import faiss
        self._begin_execution()
        if getattr(self,'search_pool',None) is not None:
            return self._parallel_ann(vector,count,ef)
        candidates=[]
        with self._stage('faiss_search_ms'):
            for shard in self.manifest['shards']:
                check_deadline()
                name=shard['file']
                if name in self.mapped_indexes:index=self.mapped_indexes[name]
                elif self.cache is None or self.cache[0]!=name:
                    self.cache=None
                    index=faiss.read_index(str(self.shards/name));self.cache=(name,index)
                    self.validate_index(index,shard)
                else:index=self.cache[1]
                index.hnsw.efSearch=ef
                check_deadline()
                scores,ids=index.search(vector.reshape(1,-1),min(count,index.ntotal))
                candidates.append((scores[0],ids[0],shard['start']))
        with self._stage('faiss_merge_ms'):result=merge_shards(candidates,count,self.metric)
        self._audit_values(shards_completed=len(self.manifest['shards']),shard_workers=1,ann_candidates=len(result))
        return result

    def _parallel_ann(self,vector,count,ef):
        import faiss
        deadline=evaluation_deadline_ns();cancelled=threading.Event();self._active_search=cancelled
        def search(shard):
            if cancelled.is_set() or (deadline is not None and time.perf_counter_ns()>=deadline):return None
            # Faiss OpenMP settings are thread-local. Each shard gets one
            # native thread, at most four shard tasks are active.
            faiss.omp_set_num_threads(1)
            index=self.mapped_indexes[shard['file']];index.hnsw.efSearch=ef
            scores,ids=index.search(vector.reshape(1,-1),min(count,index.ntotal))
            return shard,scores[0],ids[0]
        futures=[];candidates=[];completed=0
        with self._stage('faiss_search_ms'):
            try:
                for shard in self.manifest['shards']:futures.append(self.search_pool.submit(search,shard))
                for future in as_completed(futures):
                    check_deadline();result=future.result()
                    if result is None:continue
                    shard,scores,ids=result;completed+=1
                    candidates.append((scores,ids,shard['start']))
            finally:
                cancelled.set()
                for future in futures:future.cancel()
                # Native searches cannot be interrupted safely. Drain active
                # tasks before another request mutates efSearch or uses CPUs.
                with self._stage('faiss_drain_ms'):wait(futures)
                self._active_search=None
                self._audit_values(shards_completed=completed,shard_workers=self.shard_workers)
        check_deadline()
        with self._stage('faiss_merge_ms'):result=merge_shards(candidates,count,self.metric)
        self._audit_values(ann_candidates=len(result));return result

    def flat(self,vector,count):
        import faiss
        candidates=[]
        for start in range(0,len(self.raw),32768):
            check_deadline();raw=self.raw[start:start+32768]
            block=normalize(raw) if self.metric=='cosine' else raw.astype('float32')
            index=faiss.IndexFlatIP(block.shape[1]) if self.metric=='cosine' else faiss.IndexFlatL2(block.shape[1]);index.add(block)
            scores,ids=index.search(vector.reshape(1,-1),min(count,len(block)))
            candidates.extend((float(s) if self.metric=='cosine' else -float(s),int(i)+start) for s,i in zip(scores[0],ids[0]) if i>=0)
            candidates=sorted(candidates,key=lambda p:(-p[0],p[1]))[:count]
        return [i for _,i in candidates]

    def candidate_rows(self,query,vector,candidates):
        """Equivalent single-table filters and exact L2 rerank without a new ID index."""
        from .assets import scalar,tags
        if not hasattr(self,'_conn_lock'):self._begin_execution()
        if not hasattr(self,'scalars'):
            self.scalars=(scalar('equal.u2bin','<u2'),scalar('range.f4bin','<f4'),*tags())
        eq,ranges,ptr,tv=self.scalars
        ids=np.unique(np.asarray(candidates,dtype='int64'))
        self._audit_values(input_candidates=len(ids))
        with self._stage('candidate_filter_ms'):
            for f in query.item['scalar_filters']:
                check_deadline();key,op,v=f['field'],f['operator'],f['value']
                if key=='equal':ids=ids[eq[ids]==v]
                elif key=='tags':ids=ids[np.asarray([int(v) in tv[ptr[i]:ptr[i+1]] for i in ids],dtype=bool)]
                else:
                    compare={'>':np.greater,'>=':np.greater_equal,'<':np.less,'<=':np.less_equal}[op]
                    ids=ids[compare(ranges[ids].astype('float64'),v)]
        self._audit_values(filtered_candidates=len(ids))
        with self._stage('source_vector_lookup_ms'):raw=self.raw[ids].astype('float64')
        with self._stage('candidate_rerank_ms'):
            distance=np.sum((raw-vector.astype('float64'))**2,axis=1)
            rows=[[int(i)] for i in ids[np.lexsort((ids,distance))[:query.k]]]
        self._audit_values(returned_rows=len(rows),candidate_underfill=len(rows)<query.k)
        return rows

    def execute(self,query,values,config):
        self._begin_execution()
        if config.path_id in ('PRE','POST'):
            row=execute_pg_sql(self,query,values,config);row['distance_metric']=self.metric
            if self.metric=='l2':row['postgres_ann_available']=False
            return row
        params=config.params
        if config.path_id=='FAISS':
            count=max(query.k,math.ceil(query.k*params['candidate_multiplier']))
            candidates=self.ann(values[0],count,params['ef_search'])
            if self.metric=='l2':return dict(row_ids=self.candidate_rows(query,values[0],candidates),path_id='FAISS',requested_params=params,fallback_used=False,distance_metric='l2')
            statement,binds=query.render(values,candidates=[candidates],exact=True)
            return dict(row_ids=[list(r[0]) for r in self.sql(statement,binds)],path_id='FAISS',requested_params=params,fallback_used=False)
        if config.path_id!='ACORN':raise ValueError('Unsupported YFCC path')
        if self.qdrant is None:self.qdrant=self._qdrant_client()
        from qdrant_client import models
        must=[]
        for f in query.item['scalar_filters']:
            key,op,v=f['field'],f['operator'],f['value']
            if key in ('equal','tags'):
                condition=models.FieldCondition(key=key,match=models.MatchValue(value=int(v)))
            else:
                name={'>':'gt','>=':'gte','<':'lt','<=':'lte'}[op]
                condition=models.FieldCondition(key=key,range=models.Range(**{name:float(v)}))
            must.append(condition)
        # Native payload filters are equivalent for this single-table schema.
        max_selectivity=self.config.get('yfcc',{}).get('acorn_max_selectivity',1.0)
        acorn_enabled=bool(must) and self.config.get('yfcc',{}).get('acorn_enabled',True)
        self._audit_values(acorn_enabled=acorn_enabled,acorn_max_selectivity=max_selectivity)
        deadline=evaluation_deadline_ns()
        remaining=max(1,math.ceil((deadline-time.perf_counter_ns())/1e9)) if deadline else None
        with self._stage('acorn_search_rpc_ms'):
            try:
                result=self.qdrant.query_points(self.collection,query=values[0].tolist(),limit=query.k,
                    query_filter=models.Filter(must=must) if must else None,with_payload=False,with_vectors=False,timeout=remaining,
                    search_params=models.SearchParams(hnsw_ef=params['hnsw_ef'],exact=False,
                        acorn=models.AcornSearchParams(enable=acorn_enabled,max_selectivity=max_selectivity))).points
            except Exception as error:
                import grpc
                if isinstance(error,grpc.RpcError) and error.code()==grpc.StatusCode.DEADLINE_EXCEEDED and deadline is not None:
                    raise EvaluationDeadlineExceeded() from error
                raise
        candidates=[int(p.id) for p in result]
        if self.metric=='l2':return dict(row_ids=self.candidate_rows(query,values[0],candidates),path_id='ACORN',requested_params=params,fallback_used=False,distance_metric='l2')
        statement,binds=query.render(values,candidates=[candidates],exact=True)
        return dict(row_ids=[list(r[0]) for r in self.sql(statement,binds)],path_id='ACORN',requested_params=params,fallback_used=False)

    def execute_main_sql_candidate(self,query,values,config):
        self._begin_execution()
        path,params=config.path_id,config.params
        if path=='EXACT_SQL' and self.config.get('yfcc',{}).get('native_exact_filter',False):
            if self.metric!='l2':raise ValueError('Native exact filter requires raw L2')
            from .assets import scalar,tags
            from .exact import filtered_l2
            if not hasattr(self,'scalars'):
                self.scalars=(scalar('equal.u2bin','<u2'),scalar('range.f4bin','<f4'),*tags())
            with self._stage('native_exact_filter_search_ms'):
                row=filtered_l2(self.raw,self.scalars,query,values[0],check=check_deadline)
            self._audit_values(exact_certificate=row['exact_certificate'])
            return dict(row,path_id=path,requested_params=params,fallback_used=False,
                distance_metric='l2',execution_backend='native_filtered_exact',
                execution_contract='yfcc-native-exact-filter-v1')
        if path=='FAISS_SQL':
            count=max(query.k,math.ceil(query.k*params['candidate_multiplier']))
            candidates=(self.flat(values[0],count) if params['search_mode']=='flat'
                        else self.ann(values[0],count,params['ef_search']))
            if self.metric=='l2':return dict(row_ids=self.candidate_rows(query,values[0],candidates),path_id=path,requested_params=params,fallback_used=False,distance_metric='l2')
            statement,binds=query.render(values,candidates=[candidates],exact=True);settings=()
        else:
            statement,binds=query.render(values,exact=path=='EXACT_SQL')
            settings=configured_post_settings(self,params) if path=='POST_SQL' else ()
        return dict(row_ids=[list(r[0]) for r in self.sql(statement,binds,settings)],path_id=path,
                    requested_params=params,fallback_used=False,distance_metric=self.metric,postgres_ann_available=self.metric!='l2')

    def native(self,query,values):
        self._begin_execution()
        return super().native(query,values)
