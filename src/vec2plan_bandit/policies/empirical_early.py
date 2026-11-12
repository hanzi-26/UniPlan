"""Explicit empirical early fallback, independent of dataset names.

Train picks parameters; dev accepts/rejects. A finite empirical route never
pretends that its quality lower bound meets probability=1.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from .main_lightweight import FACT_FIELDS, PG_NATIVE, LightweightSelection, main_configuration, valid_facts

VERSION = 'main-empirical-early-fallback-v1'


def in_scope(facts, scope):
    return (valid_facts(facts) and facts['vector_blocks'] == 1 and
        (scope == 'single_vector' or (scope == 'simple_unfiltered' and facts.get('direct_unfiltered') is True)))


def fit_empirical_early(rows, *, train_ids, dev_ids, binding, recall_threshold,
                        min_train_groups=5, min_dev_groups=2, routing_allowance_ms=.5):
    if not 0 < recall_threshold <= 1 or min_train_groups < 1 or min_dev_groups < 1:
        raise ValueError('Invalid empirical early fit settings')
    roles = {'train':set(train_ids),'dev':set(dev_ids)}
    if roles['train'] & roles['dev']:raise ValueError('Overlapping train/dev')
    records=defaultdict(list);facts={};groups={};group_role={};seen=set()
    for row in rows:
        role,qid=row['split'],row['query_id']
        if role not in roles or qid not in roles[role] or row.get('measurement_scope')!='e2e':
            raise ValueError('Only paired train/dev calibration observations accepted')
        group=row['semantic_group']
        if group in group_role and group_role[group]!=role:
            raise ValueError('Semantic group crosses train/dev')
        group_role[group]=role
        f=row.get('routing_facts')
        if not valid_facts(f):continue
        if qid in facts and (facts[qid]!=f or groups[qid]!=group):raise ValueError('Inconsistent facts')
        facts[qid],groups[qid]=f,group
        if row['mode'] not in ('main','candidate'):raise ValueError('Unknown calibration mode')
        cid='__main__' if row['mode']=='main' else row['config_id']
        key=(role,qid,cid,row['repeat'])
        if key in seen:raise ValueError('Duplicate calibration observation')
        seen.add(key);records[(role,qid,cid)].append(row)

    def certify(role,action,scope):
        fast,main=[],[]
        for qid in sorted(roles[role]):
            if qid not in facts or not in_scope(facts[qid],scope):continue
            a,b=records.get((role,qid,action.config_id),[]),records.get((role,qid,'__main__'),[])
            if not a or not b or {r['repeat'] for r in a}!={r['repeat'] for r in b}:
                return None,'missing_paired_observations'
            if all(not r.get('eligible_for_primary_quality') for r in a):continue
            if not all(r.get('eligible_for_primary_quality') and r.get('result_full_k')
                and r.get('execution_success',True) and not r.get('timed_out')
                and not r.get('latency_censored') and not r.get('error_type')
                and not r.get('fallback_used') and not r.get('router_fallback')
                and r.get('recall_at_k') is not None and r['recall_at_k']>=recall_threshold
                and 0<=r['e2e_latency_ms']+routing_allowance_ms<5000 for r in a):
                return None,'quality_or_deadline_failure'
            fast.extend(a);main.extend(b)
        support=len({r['semantic_group'] for r in fast})
        if support < (min_train_groups if role=='train' else min_dev_groups):
            return None,'insufficient_independent_groups'
        def mean(rs):
            by_query=defaultdict(list);by_group=defaultdict(list)
            for r in rs:by_query[r['query_id']].append(r['e2e_latency_ms'])
            for q,values in by_query.items():by_group[groups[q]].append(statistics.median(values))
            return statistics.mean(statistics.mean(v) for v in by_group.values())
        if mean(fast)+routing_allowance_ms>=mean(main):return None,'no_paired_mean_gain'
        return dict(rows=fast,groups=support,mean_ms=mean(fast)+routing_allowance_ms,
            upper_ms=max(r['e2e_latency_ms'] for r in fast)+routing_allowance_ms),None

    rules,rejected,audit=[],[],[]
    actions=[('simple_unfiltered',[main_configuration('FAISS',{'ef_search':ef,'candidate_multiplier':mult})
        for ef,mult in [(100,4.),(200,8.),(400,16.)]]),('single_vector',[main_configuration(PG_NATIVE,{})])]
    for scope,choices in actions:
        candidates=[]
        for action in choices:
            result,error=certify('train',action,scope)
            audit.append(dict(scope=scope,params=action.params,train_rejection=error))
            if result:candidates.append((result['mean_ms'],action.config_id,action,result))
        if not candidates:
            rejected.append(dict(scope=scope,reason='no_train_candidate'));continue
        _,_,action,train=min(candidates,key=lambda v:(v[0],v[1]))
        dev,error=certify('dev',action,scope)
        if error:
            rejected.append(dict(scope=scope,reason='dev_'+error));continue
        signatures={r['routing_facts']['semantic_signature'] for r in train['rows']} & {
            r['routing_facts']['semantic_signature'] for r in dev['rows']}
        if not signatures:
            rejected.append(dict(scope=scope,reason='no_shared_semantics'));continue
        fs=[r['routing_facts'] for r in train['rows']+dev['rows']]
        rules.append(dict(path_id=action.path_id,params=dict(action.params),config_id=action.config_id,
            scope=scope,acceptance='empirical_best_effort',quality_mean=1.,
            quality_lower_bound=min(n/(n+1.6448536269514722**2) for n in [train['groups'],dev['groups']]),
            train_groups=train['groups'],dev_groups=dev['groups'],
            min_observed_recall=min(r['recall_at_k'] for r in train['rows']+dev['rows']),
            semantic_signatures=sorted(signatures),
            bounds={k:[min(f[k] for f in fs),max(f[k] for f in fs)] for k in FACT_FIELDS},
            e2e_mean_ms=max(train['mean_ms'],dev['mean_ms']),e2e_upper_ms=max(train['upper_ms'],dev['upper_ms'])))
    return dict(version=VERSION,binding=binding,rules=rules,rejected=rejected,
        protocol=dict(recall_threshold=recall_threshold,min_train_groups=min_train_groups,
            min_dev_groups=min_dev_groups,routing_allowance_ms=routing_allowance_ms,
            parameter_selection='train',enable_or_reject='dev',test_labels_used=False,
            evidence='all eligible paired train/dev observations pass; empirical only',
            cost_reference='original calibration main; fresh overall dev must validate new policy gain',
            train_grid_audit=audit))


class EmpiricalEarlySelector:
    def __init__(self,artifact,*,binding):
        if artifact.get('version')!=VERSION or artifact.get('binding')!=binding:
            raise ValueError('Empirical early binding/version mismatch')
        protocol=artifact['protocol']
        if (protocol.get('parameter_selection')!='train' or protocol.get('enable_or_reject')!='dev'
                or protocol.get('test_labels_used') is not False):
            raise ValueError('Empirical early needs train/dev evidence only')
        self.artifact=artifact;self.overhead_budget_ms=protocol['routing_allowance_ms']
        if not 0<self.overhead_budget_ms<=5:raise ValueError('Invalid empirical routing allowance')
        if len(artifact['rules'])>2:raise ValueError('Too many empirical rules')
        for rule in artifact['rules']:
            config=main_configuration(rule['path_id'],rule['params'])
            if (rule['path_id'] not in ('FAISS',PG_NATIVE) or config.config_id!=rule['config_id']
                    or rule['scope']!=('simple_unfiltered' if rule['path_id']=='FAISS' else 'single_vector')
                    or rule.get('acceptance')!='empirical_best_effort'
                    or rule['quality_mean']!=1 or not 0<=rule['quality_lower_bound']<1
                    or rule['min_observed_recall']<protocol['recall_threshold']
                    or rule['train_groups']<protocol['min_train_groups']
                    or rule['dev_groups']<protocol['min_dev_groups']
                    or not rule['semantic_signatures']):
                raise ValueError('Invalid empirical rule')
            for key in FACT_FIELDS:
                lo,hi=rule['bounds'][key]
                if not all(math.isfinite(v) and v>=0 for v in [lo,hi]) or lo>hi:
                    raise ValueError('Invalid empirical rule bounds')

    @classmethod
    def load(cls,path,*,binding,sha256):
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:
            raise ValueError('Empirical early artifact changed')
        return cls(json.loads(path.read_text()),binding=binding)

    def propose(self,facts,*,available,available_pre_configurations,recall_threshold,
                probability_min,latency_cap_ms,remaining_budget_ms,purpose='early'):
        if purpose!='early' or not valid_facts(facts):return LightweightSelection(None,'unsupported_empirical_scope')
        if recall_threshold>self.artifact['protocol']['recall_threshold']:
            return LightweightSelection(None,'stricter_recall_than_empirical_fit')
        for rule in self.artifact['rules']:
            if (not in_scope(facts,rule['scope']) or not available.get(rule['path_id'],False)
                    or facts['semantic_signature'] not in rule['semantic_signatures']
                    or any(not rule['bounds'][k][0]<=facts[k]<=rule['bounds'][k][1] for k in FACT_FIELDS)
                    or rule['e2e_upper_ms']>min(latency_cap_ms,remaining_budget_ms)):
                continue
            return LightweightSelection(main_configuration(rule['path_id'],rule['params']),
                'empirical_'+rule['scope'],rule)
        return LightweightSelection(None,'outside_empirical_validated_scope')
