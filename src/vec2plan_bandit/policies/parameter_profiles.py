"""Four-path parameter admission from train/dev executions, never test labels."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from vec2plan_bandit.paths.registry import make_execution_config

VERSION='four-path-parameter-profiles-v1'


def keys(query,facts):
    signature=','.join(sorted(p['type'] for p in query.predicates)) or 'unfiltered'
    if facts is None or facts.get('selectivity') is None:
        return [signature+':unknown',signature,'GLOBAL']
    selectivity=float((facts or {}).get('selectivity',1.))
    level=min(0,max(-6,int(math.floor(math.log10(max(1e-6,selectivity))))))
    return [signature+':s'+str(level),signature,'GLOBAL']


def fit(rows,queries,profiles,train_ids,dev_ids,*,recall_threshold=1.):
    if not 0<recall_threshold<=1:raise ValueError('Invalid profile Recall target')
    by_id={q.query_id:q for q in queries};train=set(train_ids);dev=set(dev_ids)
    expected=train|dev;profile_ids={c.config_id for c in profiles};buckets=defaultdict(list)
    seen=set()
    for row in rows:
        qid=row['query_id'];cid=row['config_id']
        if qid not in expected or cid not in profile_ids:raise ValueError('Profile labels outside train/dev')
        if row.get('recall_threshold')!=recall_threshold or row.get('requested_budget_ms')!=5000:
            raise ValueError('Profile Recall/deadline mismatch')
        if (qid,cid) in seen:raise ValueError('Duplicate profile/query measurement')
        seen.add((qid,cid))
        for key in keys(by_id[qid],row.get('main_routing_facts')):buckets[key].append(row)
    admitted={};coverage={}
    for key,values in buckets.items():
        ids={r['query_id'] for r in values};selected=[]
        for cid in sorted(profile_ids):
            sample=[r for r in values if r['config_id']==cid]
            if ({r['query_id'] for r in sample}==ids and sample
                    and all(r.get('joint_success') and r.get('recall_at_k',0)>=recall_threshold for r in sample)):
                selected.append(cid)
        admitted[key]=selected;coverage[key]=dict(queries=len(ids),train=len(ids&train),dev=len(ids&dev))
    return dict(version=VERSION,labels_role='train_dev_only',test_labels_used=False,recall_threshold=recall_threshold,
        profiles=[dict(path_id=c.path_id,params=dict(c.params),config_id=c.config_id) for c in profiles],
        admitted=admitted,coverage=coverage,evidence='Zero observed train/dev failures; empirical admission, not a coverage certificate')


def load(options):
    raw=Path(options['artifact']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=options['sha256']:raise ValueError('Parameter profiles changed')
    artifact=json.loads(raw)
    if artifact.get('version')!=VERSION or artifact.get('test_labels_used') is not False:
        raise ValueError('Invalid parameter profile artifact')
    for p in artifact['profiles']:
        c=make_execution_config(p['path_id'],p['params'])
        if c.config_id!=p['config_id']:raise ValueError('Profile configuration mismatch')
    return artifact


def admitted_candidates(artifact,profile_keys):
    unknown=any(key.endswith(':unknown') for key in profile_keys)
    for key in profile_keys:
        coverage=artifact.get('coverage',{}).get(key,{})
        # One lucky ANN measurement must not certify an entire new cohort.
        # Require observations in both roles before trusting its admission.
        if (coverage.get('queries',0)<3 or coverage.get('train',0)<1
                or coverage.get('dev',0)<1):
            continue
        ids=artifact['admitted'].get(key,[])
        selected=[make_execution_config(p['path_id'],p['params']) for p in artifact['profiles']
                  if p['config_id'] in ids and (not unknown or p['path_id']=='PRE')]
        if selected:return selected,dict(cohort=key,coverage=coverage,
            evidence='zero_observed_train_dev_failures',statistical_probability_one_certificate=False)
    # Missing estimates and unsupported cohorts have no empirical ANN recall
    # evidence. Use the existing exact PRE path; keep its real deadline failures.
    # This is neither a new search path nor a claim of a latency guarantee.
    return [make_execution_config('PRE',dict(index_mode='no_index',join_strategy='all'))],dict(
        cohort=None,evidence='exact_pre_fallback',statistical_probability_one_certificate=False)


def candidates(artifact,profile_keys):
    return admitted_candidates(artifact,profile_keys)[0]
