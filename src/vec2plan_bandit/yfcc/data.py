from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import struct

import numpy as np

SOURCE = Path(os.environ.get('YFCC_ASSET_DIR', 'assets/yfcc'))
DATABASE = 'yfcc1920_cosine_v1'
COLLECTION = 'yfcc_10m_1920d_cosine_acorn_v1'


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vectors(dimension=1920):
    if dimension not in (192,1920):raise ValueError('Unsupported YFCC dimension')
    path = SOURCE / ('base.10M.1920d.u8bin' if dimension==1920 else 'base.10M.u8bin')
    with path.open('rb') as f: n,d = struct.unpack('<II',f.read(8))
    if (n,d) != (10000000,dimension) or path.stat().st_size != 8+n*d:
        raise ValueError('Unexpected YFCC source shape')
    return np.memmap(path,dtype='uint8',mode='r',offset=8,shape=(n,d))


def normalize(values):
    result = np.array(values,dtype='float32',copy=True)
    norm = np.linalg.norm(result,axis=-1,keepdims=True)
    if np.any(norm==0): raise ValueError('Cosine requires nonzero vectors')
    return result/norm


def pg_parameters(config_path, *, database=DATABASE):
    import yaml
    data = yaml.safe_load(Path(config_path).read_text())['database_connections']['postgresql']
    return dict(host=data['host'],port=data['port'],user=data['user'],password=data['password'],
                dbname=database,connect_timeout=10,application_name='yfcc-wiki-cosine-v1')


def filters_sql(filters):
    parts,binds=[],{}
    for i,f in enumerate(filters):
        field,op=f['field'],f['operator']; key=f'f{i}'
        if f.get('logic')!='and': raise ValueError('Only AND filters supported')
        if field=='tags' and op=='contains':
            parts.append(f'"tags" @> ARRAY[%({key})s]::integer[]')
            binds[key]=int(f['value'])
        elif field=='equal' and op=='==':
            parts.append(f'"equal" = %({key})s'); binds[key]=int(f['value'])
        elif field=='range' and op in ('>','>=','<','<='):
            parts.append(f'"range" {op} %({key})s'); binds[key]=float(f['value'])
        else: raise ValueError(f'Unsupported filter {field} {op}')
    return ' AND '.join(parts) or 'true',binds


@dataclass(frozen=True)
class Block:
    text: str
    k: int
    table: str='YFCCImages'
    column: str='image_vec'
    alias: str='y'
    number: int=0
    asset_key: str='YFCCImages.image_vec'


@dataclass
class Query:
    item: dict
    metric: str='cosine'

    @property
    def operator(self):
        if self.metric not in ('cosine','l2'):raise ValueError('Unsupported metric')
        return '<->' if self.metric=='l2' else '<=>'

    @property
    def query_id(self): return self.item['query_id']
    @property
    def group_id(self): return 'reference-'+str(self.item['reference_vector_name'])
    @property
    def question(self): return self.item['question']
    @property
    def k(self): return self.item['k']
    @property
    def semantic_text(self): return 'reference-image:'+str(self.item['reference_vector_name'])
    @property
    def embedding_column(self): return 'image_vec'
    @property
    def blocks(self): return (Block(self.semantic_text,self.k),)
    @property
    def source_hash(self): return hashlib.sha256(json.dumps(self.item,sort_keys=True).encode()).hexdigest()
    @property
    def predicates(self): return ()
    @property
    def source_sql(self):
        where,binds=filters_sql(self.item['scalar_filters'])
        for key,value in binds.items():where=where.replace(f'%({key})s',str(value))
        # Source SQL is used only for structural features. Bound values stay out.
        return f'SELECT id FROM "YFCCImages" WHERE {where} ORDER BY image_vec {self.operator} lembed(\'reference\') LIMIT {self.k}'

    def render(self, values, candidates=None, exact=False, ids_only=True):
        where,binds=filters_sql(self.item['scalar_filters'])
        binds['v']='['+','.join(map(str,map(float,values[0])))+']'
        distance=f'image_vec {self.operator} %(v)s::vector'
        if exact: distance='('+distance+') + 0.0'
        if candidates is not None:
            where+=' AND id = ANY(%(candidates)s::bigint[])'; binds['candidates']=candidates[0]
        return f'SELECT ARRAY[id]::bigint[] FROM "YFCCImages" WHERE {where} ORDER BY {distance},id LIMIT {self.k}',binds


def load_queries(path,metric='cosine',*,expected_count=500):
    rows=json.loads(Path(path).read_text())
    if len(rows)!=expected_count or len({r['query_id'] for r in rows})!=expected_count:
        raise ValueError(f'Expected {expected_count} unique queries')
    for r in rows:
        if not 0<=r['reference_vector_name']<10000000 or not 1<=r['k']<=10 or r['limit']!=r['k']:
            raise ValueError('Invalid row ID or k')
        filters_sql(r['scalar_filters'])
    return [Query(r,metric=metric) for r in rows]


def validate_splits(queries,split):
    if set(split)!={'train','dev','test'} or not all(split.values()):
        raise ValueError('Expected nonempty train/dev/test partitions')
    assigned=[qid for ids in split.values() for qid in ids]
    if len(assigned)!=len(set(assigned)) or set(assigned)!={q.query_id for q in queries}:
        raise ValueError('Query partitions must be disjoint and cover the query source')
    refs={}
    by_id={q.query_id:q for q in queries}
    for role,ids in split.items():
        for qid in ids:
            ref=by_id[qid].item['reference_vector_name']
            if ref in refs and refs[ref]!=role:raise ValueError('Reference crosses splits')
            refs[ref]=role
    return split


def split_queries(queries, seed=42):
    # Each reference image belongs to one role. Allocate proportional integer
    # quotas per (k, template, selectivity), then balance the remaining slots.
    from collections import defaultdict
    groups=defaultdict(list)
    for q in queries: groups[q.k].append(q)
    result={r:[] for r in ('train','dev','test')}
    rng=random.Random(seed)
    for k,items in sorted(groups.items()):
        if len(items)!=50: raise ValueError('Expected 50 queries per k')
        buckets=defaultdict(list)
        for q in sorted(items,key=lambda q:q.query_id):buckets[(q.item['template_category'],q.item['selectivity_band'])].append(q)
        local={r:[] for r in result};pending=[]
        for key in sorted(buckets):
            b=buckets[key];rng.shuffle(b);cursor=0
            for role,fraction in [('train',.6),('dev',.2),('test',.2)]:
                size=int(len(b)*fraction);local[role].extend(q.query_id for q in b[cursor:cursor+size]);cursor+=size
            pending.extend(b[cursor:])
        rng.shuffle(pending);target=dict(train=30,dev=10,test=10)
        for q in pending:
            role=max(local,key=lambda r:target[r]-len(local[r]));local[role].append(q.query_id)
        for role,ids in local.items():result[role].extend(ids)
    roles={qid:role for role,ids in result.items() for qid in ids}
    refs={}
    for q in queries:
        ref=q.item['reference_vector_name']
        if ref in refs and refs[ref]!=roles[q.query_id]:raise ValueError('Reference crosses splits')
        refs[ref]=roles[q.query_id]
    return {role:sorted(ids) for role,ids in result.items()}
