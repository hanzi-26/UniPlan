"""Bounded-memory PostgreSQL import and cosine exact-truth preparation."""
import json
from pathlib import Path
import struct
import time

import numpy as np

from .data import SOURCE,DATABASE,normalize,vectors,pg_parameters,write,load_queries


def scalar(name,dtype):
    p=SOURCE/name
    with p.open('rb') as f:n,d=struct.unpack('<II',f.read(8))
    if (n,d)!=(10000000,1):raise ValueError('Unexpected scalar shape')
    return np.memmap(p,dtype=dtype,mode='r',offset=8,shape=(n,))


def tags():
    p=SOURCE/'base.metadata.10M.spmat'
    with p.open('rb') as f:n,d,nnz=struct.unpack('<3q',f.read(24))
    ptr=np.memmap(p,dtype='<i8',mode='r',offset=24,shape=(n+1,))
    values=np.memmap(p,dtype='<i4',mode='r',offset=24+8*(n+1),shape=(nnz,))
    if n!=10000000 or ptr[-1]!=nnz:raise ValueError('Invalid metadata')
    return ptr,values


def binary_rows(start,values,equal,ranges,ptr,tag_values,*,metric='cosine'):
    """Postgres COPY binary protocol and pgvector's int16-dim/float32 format."""
    out=bytearray(b'PGCOPY\n\xff\r\n\x00'+struct.pack('!ii',0,0))
    vecs=(normalize(values) if metric=='cosine' else np.asarray(values,dtype='float32')).astype('>f4');dimension=vecs.shape[1]
    for offset,v in enumerate(vecs):
        row=start+offset
        tv=tag_values[ptr[row]:ptr[row+1]]
        out+=struct.pack('!hiqi',5,8,row,4+4*dimension)
        out+=struct.pack('!hh',dimension,0)+v.tobytes()
        out+=struct.pack('!iii',4,int(equal[row]),4)+struct.pack('!f',float(ranges[row]))
        out+=struct.pack('!i',20+8*len(tv))+struct.pack('!iiiii',1,0,23,len(tv),1)
        for tag in tv:out+=struct.pack('!ii',4,int(tag))
    out+=struct.pack('!h',-1)
    return out


def import_postgres(output,config_path,*,database=DATABASE,metric='cosine',create_indexes=True,dimension=1920):
    import psycopg
    output=Path(output);status=output/'postgres/status.json'
    with psycopg.connect(**pg_parameters(config_path,database='postgres'),autocommit=True) as admin:
        exists=admin.execute('SELECT 1 FROM pg_database WHERE datname=%s',(database,)).fetchone()
        from psycopg import sql
        if not exists:admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(database)))
    with psycopg.connect(**pg_parameters(config_path,database=database)) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.execute('CREATE TABLE IF NOT EXISTS import_progress (singleton boolean NOT NULL DEFAULT true CHECK(singleton), next_row bigint NOT NULL, source_contract text NOT NULL)')
        primary=' PRIMARY KEY' if create_indexes else ''
        if dimension not in (192,1920):raise ValueError('Unsupported YFCC dimension')
        conn.execute('CREATE TABLE IF NOT EXISTS "YFCCImages" (id bigint'+primary+f',image_vec vector({dimension}) NOT NULL,"equal" integer NOT NULL,"range" real NOT NULL,tags integer[] NOT NULL)')
        contract=json.dumps(dict(source=str(SOURCE/('base.10M.1920d.u8bin' if dimension==1920 else 'base.10M.u8bin')),metric=metric,normalized=metric=='cosine',zero_based=True),sort_keys=True)
        conn.execute('INSERT INTO import_progress SELECT true,0,%s WHERE NOT EXISTS (SELECT 1 FROM import_progress)',(contract,))
        next_row,previous=conn.execute('SELECT next_row,source_contract FROM import_progress FOR UPDATE').fetchone()
        if previous!=contract:raise ValueError('Import contract mismatch')
        conn.commit()
        raw=vectors() if dimension==1920 else vectors(dimension);equal=scalar('equal.u2bin','<u2');ranges=scalar('range.f4bin','<f4');ptr,tv=tags()
        started=time.monotonic()
        for start in range(next_row,len(raw),8192):
            end=min(len(raw),start+8192)
            with conn.cursor().copy('COPY "YFCCImages" (id,image_vec,"equal","range",tags) FROM STDIN WITH (FORMAT BINARY)') as copy:
                copy.write(binary_rows(start,raw[start:end],equal,ranges,ptr,tv,metric=metric))
            conn.execute('UPDATE import_progress SET next_row=%s',(end,));conn.commit()
            write(status,dict(stage='importing',rows=end,total=len(raw),elapsed_s=time.monotonic()-started))
        if conn.execute('SELECT count(*) FROM "YFCCImages"').fetchone()[0]!=len(raw):raise ValueError('PostgreSQL count mismatch')
        write(output/'postgres/import.complete.json',dict(count=len(raw),dimension=dimension,metric=metric,database=database))
        if not create_indexes:
            conn.execute('ANALYZE "YFCCImages"');conn.commit()
            write(output/'postgres/manifest.json',dict(database=database,count=len(raw),dimension=dimension,metric=metric,indexes=[],new_index_construction=False,postgres_ann_available=False))
    write(status,dict(stage='import_complete',rows=10000000))


def postgres_indexes(output,config_path):
    import psycopg
    output=Path(output)
    with psycopg.connect(**pg_parameters(config_path),autocommit=True) as conn:
        conn.execute("SET maintenance_work_mem='768MB'")
        conn.execute('SET max_parallel_maintenance_workers=0')
        indexes=[('yfcc_equal_btree','CREATE INDEX yfcc_equal_btree ON "YFCCImages" ("equal")'),
                 ('yfcc_range_btree','CREATE INDEX yfcc_range_btree ON "YFCCImages" ("range")'),
                 ('yfcc_tags_gin','CREATE INDEX yfcc_tags_gin ON "YFCCImages" USING gin(tags)'),
                 ('yfcc_image_cosine_hnsw','CREATE INDEX yfcc_image_cosine_hnsw ON "YFCCImages" USING hnsw(image_vec vector_cosine_ops) WITH (m=32,ef_construction=200)')]
        for name,statement in indexes:
            row=conn.execute('SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid WHERE c.relname=%s',(name,)).fetchone()
            if row:
                if not row[0]:raise ValueError('Invalid existing index '+name)
                continue
            write(output/'postgres/status.json',dict(stage='building_index',index=name,backend_pid=conn.info.backend_pid))
            conn.execute(statement)
        conn.execute('ANALYZE "YFCCImages"')
        write(output/'postgres/manifest.json',dict(database=DATABASE,count=10000000,dimension=1920,metric='cosine',indexes=[n for n,_ in indexes],build_memory_mb=768,parallel_workers=0))
        write(output/'postgres/status.json',dict(stage='complete'))


def exact_truth(output,query_source,*,metric='cosine',expected_count=500,dimension=1920):
    """Full corpus cosine, float64 accumulation, bounded CPU batches, stable ID ties."""
    from threadpoolctl import threadpool_limits
    output=Path(output);queries=load_queries(query_source,metric=metric,expected_count=expected_count);raw=vectors() if dimension==1920 else vectors(dimension)
    ptr,tv=tags();eq=scalar('equal.u2bin','<u2');ranges=scalar('range.f4bin','<f4')
    tag_ids={}
    wanted={int(f['value']) for q in queries for f in q.item['scalar_filters'] if f['field']=='tags'}
    for tag in sorted(wanted):
        positions=np.flatnonzero(tv==tag)
        tag_ids[tag]=np.unique(np.searchsorted(ptr,positions,side='right')-1)
    # Compare exactly the normalized float32 values stored by every backend.
    selected=raw[[q.item['reference_vector_name'] for q in queries]]
    q32=normalize(selected) if metric=='cosine' else selected.astype('float32')
    qv=q32.astype('float64')
    if metric=='cosine':qv/=np.linalg.norm(qv,axis=1,keepdims=True)
    best=[(np.empty(0,dtype='int64'),np.empty(0)) for q in queries]
    counts=np.zeros(len(queries),dtype='int64')
    started=time.monotonic()
    with threadpool_limits(limits=4):
        for start in range(0,len(raw),32768):
            end=min(len(raw),start+32768);ids=np.arange(start,end,dtype='int64')
            block=(normalize(raw[start:end]) if metric=='cosine' else raw[start:end]).astype('float64')
            if metric=='cosine':block/=np.linalg.norm(block,axis=1,keepdims=True)
            scores=qv@block.T
            if metric=='l2':scores=2*scores-np.sum(qv*qv,axis=1)[:,None]-np.sum(block*block,axis=1)[None,:]
            for i,q in enumerate(queries):
                keep=np.ones(end-start,dtype=bool)
                for f in q.item['scalar_filters']:
                    v=f['value'];field=f['field'];op=f['operator']
                    if field=='equal':keep &= eq[start:end]==v
                    elif field=='tags':keep &= np.isin(ids,tag_ids[int(v)],assume_unique=True)
                    else:
                        vals=ranges[start:end].astype('float64')
                        keep &= {'>':np.greater,'>=':np.greater_equal,'<':np.less,'<=':np.less_equal}[op](vals,v)
                counts[i]+=keep.sum();positions=np.flatnonzero(keep);s=scores[i,positions]
                if len(s)>q.k:
                    cut=np.partition(s,-q.k)[-q.k];positions=positions[s>=cut];s=scores[i,positions]
                old_ids,old_s=best[i];merged_ids=np.concatenate((old_ids,ids[positions]));merged_s=np.concatenate((old_s,s))
                order=np.lexsort((merged_ids,-merged_s))[:q.k];best[i]=(merged_ids[order],merged_s[order])
            write(output/'truth/status.json',dict(stage='exact_'+metric+'_scan',scanned=end,total=len(raw),elapsed_s=time.monotonic()-started))
    prepared=[]
    for q,(ids,s),count,vector in zip(queries,best,counts,q32):
        prepared.append(dict(query_id=q.query_id,topk_row_ids=[[int(i)] for i in ids],similarities=s.tolist(),
            vectors=[vector.astype('float32').tolist()],statistics=dict(corpus_size=len(raw),candidate_count=int(count),estimated_selectivity=float(count)/len(raw)),
            available_pre=[],ground_truth_below_k=int(count)<q.k,source_hash=q.source_hash))
    target=output/'prepare/prepared.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(r)+'\n' for r in prepared))
    (output/'truth/prepared.jsonl').write_bytes(target.read_bytes())
    write(output/'truth/manifest.json',dict(count=len(queries),corpus=len(raw),metric=metric,normalization='float32 corpus then float64 cosine accumulation' if metric=='cosine' else 'none; exact uint8 values, float64 squared L2',score_convention='cosine' if metric=='cosine' else 'negative_squared_L2',tie_rule='score_desc_then_id_asc',source_query_count=len(queries)))
    write(output/'truth/status.json',dict(stage='complete'))
