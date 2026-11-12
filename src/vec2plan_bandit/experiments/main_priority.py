"""Train-selected ordered FAISS/PG certificates; dev only accepts or rejects.

This opt-in changes only the main selector. No test observations are accepted.
FAISS parameters and PG corpus bounds use declared train-only candidate grids.
Below-k queries are reported separately and supply no quality support.
"""
from collections import defaultdict
import math
import statistics

from vec2plan_bandit.policies.main_lightweight import (
    FACT_FIELDS, TREE_FEATURES, VERSION, PG_NATIVE, main_configuration, valid_facts,
)


def fit_priority(rows, *, train_ids, dev_ids, binding, recall_threshold=.8,
                 probability_min=.8, latency_cap_ms=5000., routing_allowance_ms=.5,
                 min_relative_gain=.05, confidence_z=1.6448536269514722):
    if not 0 < probability_min <= 1 or not 0 < recall_threshold <= 1:
        raise ValueError('ordered routing needs probability in (0,1] and valid Recall')
    roles = {'train':set(train_ids),'dev':set(dev_ids)}
    if roles['train'] & roles['dev']:
        raise ValueError('overlapping calibration splits')
    records = defaultdict(list); facts = {}; groups = {}; group_roles = {}; seen = set()
    for row in rows:
        role,qid = row['split'],row['query_id']
        if role not in roles or qid not in roles[role] or row.get('measurement_scope') != 'e2e':
            raise ValueError('only paired train/dev E2E observations are accepted')
        group=row['semantic_group']
        if group in group_roles and group_roles[group]!=role:
            raise ValueError('semantic group crosses splits')
        group_roles[group]=role
        f=row.get('routing_facts')
        if not valid_facts(f):
            continue
        if qid in facts and (facts[qid]!=f or groups[qid]!=group):
            raise ValueError('inconsistent query facts or semantic group')
        facts[qid]=f;groups[qid]=group
        if row['mode'] not in ('main','candidate'):
            raise ValueError('unknown calibration mode')
        cid='__main__' if row['mode']=='main' else row['config_id']
        if cid!='__main__' and main_configuration(row['path_id'],row['requested_params']).config_id!=cid:
            raise ValueError('candidate configuration mismatch')
        key=(role,qid,cid,row['repeat'])
        if key in seen:raise ValueError('duplicate calibration observation')
        seen.add(key);records[(role,qid,cid)].append(row)
    if probability_min == 1:
        # An all-success Wilson lower bound n/(n+z^2) is strictly below 1
        # for every finite n. Keep the requested threshold and return no gate,
        # rather than rounding it down or dividing by zero below.
        return dict(version=VERSION, binding=binding, tree_features=list(TREE_FEATURES),
            nodes=[{'leaf':0}], leaves={}, pg_fallbacks={}, priority_rules=[],
            rejected=[dict(path=path, reason='probability_one_has_no_finite_certificate')
                      for path in ('FAISS', PG_NATIVE)],
            fit_protocol=dict(parameter_selection='train', enable_or_reject='dev',
                test_labels_used=False, recall_threshold=recall_threshold, probability_min=1.,
                routing_allowance_ms=routing_allowance_ms, latency_cap_ms=latency_cap_ms,
                minimum_independent_groups=None, confidence_z=confidence_z,
                order=['simple_unfiltered_FAISS','certified_PG','full_model'],
                quality_evidence='finite calibration cannot certify probability one', train_grid_audit=[]))
    # This is the all-success Wilson lower bound, solving n/(n+z^2)>=p.
    minimum=math.ceil(confidence_z**2*probability_min/(1-probability_min))
    faiss_actions=[main_configuration('FAISS',{'ef_search':ef,'candidate_multiplier':mult})
                   for ef,mult in [(100,4.),(200,8.),(400,16.)]]

    def certify(role, action, scope, upper):
        qids=[q for q in roles[role] if q in facts and facts[q]['corpus_size']<=upper
              and (scope!='simple_unfiltered' or facts[q].get('direct_unfiltered',False))]
        fast=[];main=[];excluded=[]
        for q in sorted(qids):
            a=records.get((role,q,action.config_id),[]);b=records.get((role,q,'__main__'),[])
            if not a or not b or {r['repeat'] for r in a}!={r['repeat'] for r in b}:
                return None,'missing_paired_measurements'
            if any(r.get('timed_out') or r.get('latency_censored') or r.get('error_type') for r in a+b):
                return None,'timeout_or_execution_error'
            if all(not r.get('eligible_for_primary_quality') for r in a+b):
                excluded.append(q);continue
            if not all(r.get('eligible_for_primary_quality') and r.get('result_full_k')
                       and r.get('execution_success',True) and not r.get('fallback_used')
                       and not r.get('router_fallback')
                       and r.get('recall_at_k') is not None and r['recall_at_k']>=recall_threshold
                       and isinstance(r.get('e2e_latency_ms'),(int,float))
                       and 0<=r['e2e_latency_ms']+routing_allowance_ms<latency_cap_ms for r in a):
                return None,'quality_or_budget_failure'
            fast.extend(a);main.extend(b)
        support=len({r['semantic_group'] for r in fast})
        if support<minimum:return None,'insufficient_independent_groups'
        def mean(rs):
            per_query=defaultdict(list);per_group=defaultdict(list)
            for r in rs:per_query[r['query_id']].append(r['e2e_latency_ms'])
            for q,values in per_query.items():per_group[groups[q]].append(statistics.median(values))
            return statistics.mean(statistics.mean(v) for v in per_group.values())
        if mean(fast)+routing_allowance_ms>=mean(main)*(1-min_relative_gain):
            return None,'no_e2e_gain'
        return dict(rows=fast,groups=support,mean_ms=mean(fast)+routing_allowance_ms,
                    upper_ms=max(r['e2e_latency_ms'] for r in fast)+routing_allowance_ms,
                    excluded_below_k=excluded),None

    rules=[];rejected=[];audit=[]
    for actions,scope,limits in [(faiss_actions,'simple_unfiltered',[2.**62]),
                                ([main_configuration(PG_NATIVE,{})],'corpus_bound',
                                 [32.,128.,512.,2048.,8192.,32768.,131072.,2.**62])]:
        choices=[]
        for action in actions:
            for upper in limits:
                receipt,error=certify('train',action,scope,upper)
                audit.append(dict(path=action.path_id,params=dict(action.params),corpus_upper=upper,train_rejection=error))
                if receipt is not None:choices.append((receipt['groups'],upper,receipt,action))
        if not choices:
            rejected.append(dict(path=actions[0].path_id,reason='no_train_certificate'));continue
        _,upper,train,action=max(choices,key=lambda v:(v[0],v[1],-v[2]['mean_ms']))
        dev,error=certify('dev',action,scope,upper)
        if error:
            rejected.append(dict(path=action.path_id,reason='dev_'+error,corpus_upper=upper));continue
        signatures={r['routing_facts']['semantic_signature'] for r in train['rows']} & {
            r['routing_facts']['semantic_signature'] for r in dev['rows']}
        if not signatures:
            rejected.append(dict(path=action.path_id,reason='no_shared_semantics'));continue
        all_rows=train['rows']+dev['rows'];fs=[r['routing_facts'] for r in all_rows]
        rule=dict(path_id=action.path_id,params=dict(action.params),config_id=action.config_id,
            scope=scope,corpus_upper=upper,acceptance='quality_and_e2e_gain',
            bounds={k:[min(f[k] for f in fs),max(f[k] for f in fs)] for k in FACT_FIELDS},
            semantic_signatures=sorted(signatures),quality_mean=1.,
            quality_lower_bound=min(n/(n+confidence_z**2) for n in (train['groups'],dev['groups'])),
            train_groups=train['groups'],dev_groups=dev['groups'],
            e2e_mean_ms=max(train['mean_ms'],dev['mean_ms']),
            e2e_upper_ms=max(train['upper_ms'],dev['upper_ms']),
            min_observed_recall=min(r['recall_at_k'] for r in all_rows),
            excluded_below_k={'train':train['excluded_below_k'],'dev':dev['excluded_below_k']})
        rules.append(rule)
    return dict(version=VERSION,binding=binding,tree_features=list(TREE_FEATURES),
        nodes=[{'leaf':0}],leaves={},pg_fallbacks={},priority_rules=rules,rejected=rejected,
        fit_protocol=dict(parameter_selection='train',enable_or_reject='dev',test_labels_used=False,
            recall_threshold=recall_threshold,probability_min=probability_min,
            routing_allowance_ms=routing_allowance_ms,latency_cap_ms=latency_cap_ms,
            min_relative_gain=min_relative_gain,minimum_independent_groups=minimum,
            confidence_z=confidence_z,order=['simple_unfiltered_FAISS','certified_PG','full_model'],
            quality_evidence='all eligible observations pass; independent-group Wilson lower bound',
            train_grid_audit=audit))
