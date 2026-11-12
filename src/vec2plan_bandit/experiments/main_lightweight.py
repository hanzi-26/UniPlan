"""Offline train-only shallow tree, followed by dev enable/reject certificates.

This fitter is explicit and never invoked by loading a model or a benchmark.
Inputs are fresh paired main/candidate E2E measurements, including failed and
censored observations. Test rows are rejected. No dataset names are features.
"""
from collections import Counter, defaultdict
import hashlib
import statistics

from vec2plan_bandit.policies.main_lightweight import (
    FACT_FIELDS, TREE_FEATURES, VERSION, feature_vector, finite, valid_facts,
    main_configuration,
)


def fit_lightweight(rows, *, train_ids, dev_ids, binding, recall_threshold=.8,
                    probability_min=.8,
                    min_relative_gain=.05, routing_allowance_ms=.5, max_depth=3,
                    min_train_groups=12, min_dev_groups=12, confidence_z=1.6448536269514722,
                    latency_cap_ms=5000., seed=42, cost_target='e2e', tail_quantile=1.,
                    train_gate_fraction=0., independent_pg_actions=False, tree_objective='cost',
                    candidate_availability=None, quality_evidence='all_pass',tail_protocol='point'):
    """Wilson bound is group-level empirical evidence, not a Recall guarantee."""
    if (not 0 < recall_threshold <= 1 or not 0 <= probability_min <= 1 or not 0 <= min_relative_gain < 1
            or not finite(routing_allowance_ms, 1e-9) or not finite(latency_cap_ms, 1e-9)
            or type(max_depth) is not int or not 1 <= max_depth <= 7
            or min_train_groups < 1 or min_dev_groups < 1 or not finite(confidence_z, 1e-9)):
        raise ValueError('invalid lightweight selector calibration settings')
    if cost_target not in ('e2e', 'selection_execution'):
        raise ValueError('unknown lightweight cost target')
    if tree_objective not in ('cost','quality_admission') or tree_objective!='cost' and not independent_pg_actions:
        raise ValueError('invalid PG tree objective')
    if quality_evidence not in ('all_pass', 'group_wilson'):
        raise ValueError('invalid PG quality evidence protocol')
    if quality_evidence == 'group_wilson' and not train_gate_fraction:
        raise ValueError('group Wilson admission requires reserved train groups')
    if tail_protocol not in ('point','paired_cluster_bootstrap'):
        raise ValueError('invalid PG tail protocol')
    if not finite(tail_quantile, .5) or tail_quantile > 1:
        raise ValueError('invalid lightweight cost tail quantile')
    if not finite(train_gate_fraction) or train_gate_fraction >= .5:
        raise ValueError('invalid held-out train gate fraction')
    roles = {'train': set(train_ids), 'dev': set(dev_ids)}
    if roles['train'] & roles['dev']:
        raise ValueError('overlapping calibration splits')
    if candidate_availability is not None:
        if set(candidate_availability) != roles['train'] | roles['dev']:
            raise ValueError('candidate availability must cover exactly train/dev queries')
        candidate_availability = {q:set(cs) for q,cs in candidate_availability.items()}
    records, facts, group_role, groups, seen = defaultdict(list), {}, {}, {}, set()
    unsupported_rows = 0
    for row in rows:
        role, qid = row['split'], row['query_id']
        if role not in roles or qid not in roles[role] or row.get('measurement_scope') != 'e2e':
            raise ValueError('fresh paired train/dev E2E observations required; test rows forbidden')
        group = row['semantic_group']
        if group in group_role and group_role[group] != role:
            raise ValueError('semantic group crosses calibration splits')
        if qid in groups and groups[qid] != group:
            raise ValueError('query semantic group changed')
        group_role[group], groups[qid] = role, group
        if row.get('routing_facts') is None:
            unsupported_rows += 1
            continue  # Explicit abstention; count it in the fit receipt.
        if not valid_facts(row['routing_facts']):
            raise ValueError('complete raw corpus/query routing facts are required')
        if qid in facts and facts[qid] != row['routing_facts']:
            raise ValueError('routing facts changed for a query')
        facts[qid] = row['routing_facts']
        mode = row['mode']
        if mode not in ('main', 'candidate'):
            raise ValueError('unknown calibration mode')
        cid = '__main__'
        if mode == 'candidate':
            action = main_configuration(row['path_id'], row['requested_params'])
            if action.config_id != row['config_id']:
                raise ValueError('candidate configuration mismatch')
            cid = action.config_id
            if candidate_availability is not None and cid not in candidate_availability[qid]:
                raise ValueError('measurement contradicts physical candidate availability')
        identity = (role, qid, cid, row['repeat'])
        if identity in seen:
            raise ValueError('duplicate calibration request')
        seen.add(identity)
        records[(role, qid, cid)].append(row)

    def observed_cost(row):
        value = row.get('e2e_latency_ms')
        if row.get('timed_out') or row.get('latency_censored'):
            bound = row.get('latency_lower_bound_ms', latency_cap_ms)
            if not finite(bound, 1e-9):
                raise ValueError('invalid censored latency lower bound')
            return bound  # A lower bound for comparison, never a completed latency label.
        if not finite(value):
            raise ValueError('invalid E2E observation')
        return value

    def cost(row):
        observed = observed_cost(row)
        if cost_target == 'e2e' or row.get('timed_out') or row.get('latency_censored'):
            return observed
        spans = row.get('spans_ms', {})
        # Embedding is a common prefix. Its independent service jitter cannot
        # teach a PG-versus-full-selector label. Still retain the complete E2E
        # measurement for admission, deadlines, reporting and confidence checks.
        if not all(finite(spans.get(k)) for k in ('policy_selection', 'execution')):
            raise ValueError('selection/execution spans required for suffix-cost fitting')
        suffix = spans['policy_selection'] + spans['execution']
        if suffix > observed + 1e-6:
            raise ValueError('suffix spans exceed complete request duration')
        return suffix

    def good(row):
        return (not row.get('timed_out') and not row.get('latency_censored')
                and not row.get('error_type') and row.get('execution_success', True)
                and not row.get('fallback_used') and not row.get('router_fallback')
                and row.get('eligible_for_primary_quality') and row.get('result_full_k')
                and finite(row.get('recall_at_k')) and recall_threshold <= row['recall_at_k'] <= 1
                and observed_cost(row) + routing_allowance_ms < latency_cap_ms)

    candidates = sorted({key[2] for key in records if key[2] != '__main__'})
    actions = {cid: main_configuration(r['path_id'], r['requested_params'])
               for (_, _, cid), values in records.items() if cid != '__main__' for r in values[:1]}
    if independent_pg_actions and any(a.path_id not in ('PRE', 'POST') for a in actions.values()):
        raise ValueError('independent PG admission supports only PRE and POST')
    train_groups = sorted({groups[q] for q in facts if q in roles['train']},
        key=lambda g: hashlib.sha256(f'{seed}:pg-gate:{g}'.encode()).hexdigest())
    heldout_groups = set(train_groups[:max(1, round(len(train_groups)*train_gate_fraction))]) if train_gate_fraction else set()
    gate_ids = {q for q in roles['train'] if groups.get(q) in heldout_groups}
    fit_ids = roles['train']-gate_ids
    training = sorted({qid for role,qid,cid in records if role == 'train' and cid == '__main__' and qid in fit_ids})
    if not training:
        raise ValueError('no train main-policy measurements')
    labels = []
    for qid in training:
        main = records[('train',qid,'__main__')]
        choices = []
        quality_choices=[]
        for cid in candidates:
            fast = records.get(('train',qid,cid), [])
            if ({r['repeat'] for r in fast} != {r['repeat'] for r in main}
                    or not fast or not all(good(r) for r in fast)):
                continue
            value = statistics.median(cost(r) for r in fast) + routing_allowance_ms
            quality_choices.append(cid)
            if value < statistics.median(cost(r) for r in main) * (1-min_relative_gain):
                choices.append((value,cid))
        labels.append(('__admit__' if quality_choices else '__full__') if tree_objective=='quality_admission'
                      else min(choices)[1] if choices else '__full__')
    from sklearn.tree import DecisionTreeClassifier
    counts = Counter(groups[qid] for qid in training)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_train_groups,
                                   random_state=seed)
    model.fit([feature_vector(facts[qid]) for qid in training], labels,
              sample_weight=[1/counts[groups[qid]] for qid in training])
    tree, nodes = model.tree_, []
    for index in range(tree.node_count):
        if tree.children_left[index] < 0:
            nodes.append({'leaf':index})
        else:
            nodes.append(dict(feature=int(tree.feature[index]), threshold=float(tree.threshold[index]),
                              left=int(tree.children_left[index]), right=int(tree.children_right[index])))
    assignments = {qid:int(model.apply([feature_vector(f)])[0]) for qid,f in facts.items()}

    def independent_mean(rs, measure=cost):
        per_query = defaultdict(list)
        for r in rs:
            per_query[r['query_id']].append(measure(r))
        by_group = defaultdict(list)
        for qid, values in per_query.items():
            by_group[groups[qid]].append(statistics.median(values))
        return statistics.mean(statistics.mean(v) for v in by_group.values())

    def group_bound(number):
        # All observations must pass; repeated requests do not increase support.
        return number / (number + confidence_z**2)

    def quality_statistics(rs):
        by_group = defaultdict(list)
        for row in rs:
            by_group[row['semantic_group']].append(good(row))
        n = len(by_group)
        passed = sum(all(v) for v in by_group.values())
        if not n:
            return 0., 0.
        fraction, z2 = passed/n, confidence_z**2
        lower = (fraction + z2/(2*n) - confidence_z *
                 (fraction*(1-fraction)/n + z2/(4*n*n))**.5)/(1+z2/n)
        return fraction, max(0., lower)

    def quality_eligible(rs):
        if quality_evidence == 'all_pass':
            return bool(rs) and all(good(r) for r in rs)
        # Keep execution errors/timeouts as stability vetoes. Recall misses
        # remain failures in the group-level quality estimate, not discarded.
        return (bool(rs) and all(not r.get('timed_out') and not r.get('latency_censored')
                and not r.get('error_type') and not r.get('fallback_used')
                and r.get('execution_success', True) and r.get('eligible_for_primary_quality')
                and finite(r.get('recall_at_k')) and observed_cost(r)+routing_allowance_ms<latency_cap_ms
                for r in rs) and
                independent_mean(rs, lambda r:r['recall_at_k']) >= recall_threshold and
                quality_statistics(rs)[0] >= probability_min)

    def certify(leaf, cid, *, require_gain):
        paired, support, signatures, failure, tail_checks = {}, {}, {}, None, {}
        partitions = [('train', fit_ids, min_train_groups), ('dev', roles['dev'], min_dev_groups)]
        if train_gate_fraction:
            partitions.insert(1, ('train_gate', gate_ids, 1))
        for role, ids, minimum in partitions:
            qids = [q for q in facts if q in ids and assignments[q] == leaf
                    and (candidate_availability is None or cid in candidate_availability[q])]
            fast, main = [], []
            for qid in qids:
                record_role = 'train' if role == 'train_gate' else role
                a, b = records.get((record_role,qid,cid),[]), records.get((record_role,qid,'__main__'),[])
                if not a or not b or {r['repeat'] for r in a}!={r['repeat'] for r in b}:
                    failure = role+'_missing_paired_measurements'; break
                if any(r.get('error_type') and not r.get('timed_out') and not r.get('latency_censored') for r in b):
                    failure = role+'_reference_execution_error'; break
                fast.extend(a); main.extend(b)
            if failure:
                break
            support[role] = len({r['semantic_group'] for r in fast})
            if support[role] < minimum or not quality_eligible(fast):
                failure = role+'_quality_or_independent_support'; break
            if require_gain and independent_mean(fast)+routing_allowance_ms >= independent_mean(main)*(1-min_relative_gain):
                failure = role+'_no_e2e_gain'; break
            if (require_gain and cost_target != 'e2e' and
                    independent_mean(fast, observed_cost)+routing_allowance_ms >=
                    independent_mean(main, observed_cost)*(1-min_relative_gain)):
                failure = role+'_no_observed_e2e_gain'; break
            # Default retains the historical worst-case gate. An explicit P95
            # experiment may optimize that measured tail instead; every sample
            # still passes the full E2E deadline and all quality checks above.
            import numpy as np
            if require_gain:
                if tail_protocol=='paired_cluster_bootstrap':
                    from .paired_tail import paired_tail_evidence
                    check=paired_tail_evidence(fast,main,measure=cost,allowance_ms=routing_allowance_ms,
                        quantile=tail_quantile,seed=seed)
                    tail_checks[role]=check
                    if check['detected_tail_regression']:
                        failure=role+'_detected_tail_regression';break
                    if check['mean_difference_upper_ms']>=0:
                        failure=role+'_uncertain_mean_gain';break
                elif (np.quantile([cost(r) for r in fast], tail_quantile)+routing_allowance_ms >
                        np.quantile([cost(r) for r in main], tail_quantile)):
                    failure = role+'_tail_regression'; break
            paired[role] = fast
            signatures[role] = {facts[r['query_id']]['semantic_signature'] for r in fast}
        if failure:
            return None, failure
        shared_signatures = set.intersection(*signatures.values())
        if not shared_signatures:
            return None, 'no_shared_validated_semantics'
        # The reserved train groups never select parameters, labels or the tree.
        # They may supply independent quality support alongside dev. The default
        # protocol keeps its historical per-split minimum bound unchanged.
        evidence_rows = paired['train_gate']+paired['dev'] if train_gate_fraction else []
        quality_mean = 1.
        if quality_evidence == 'group_wilson':
            quality_mean, quality_bound = quality_statistics(evidence_rows)
        else:
            quality_bound = (group_bound(support['train_gate']+support['dev']) if train_gate_fraction else
                             min(group_bound(n) for n in support.values()))
        if quality_bound < probability_min:
            return None, 'insufficient_group_probability_evidence'
        fast = [row for values in paired.values() for row in values]
        qids = {r['query_id'] for r in fast}
        first = fast[0]
        certificate = dict(path_id=first['path_id'],params=first['requested_params'],config_id=cid,
            bounds={k:[min(facts[q][k] for q in qids),max(facts[q][k] for q in qids)] for k in FACT_FIELDS},
            semantic_signatures=sorted(shared_signatures), quality_mean=quality_mean,
            quality_lower_bound=quality_bound,
            train_groups=support['train'],dev_groups=support['dev'],
            e2e_mean_ms=independent_mean(fast, observed_cost)+routing_allowance_ms,
            e2e_upper_ms=max(observed_cost(r) for r in fast)+routing_allowance_ms,
            min_observed_recall=min(r['recall_at_k'] for r in fast),
            acceptance='quality_and_e2e_gain' if require_gain else 'quality_and_budget_only')
        if tail_checks:certificate['paired_tail_checks']=tail_checks
        if train_gate_fraction:
            certificate.update(train_gate_groups=support['train_gate'],
                independent_quality_groups=support['train_gate']+support['dev'],
                    quality_evidence='held-out train semantic groups plus dev; no fit groups counted',
                    quality_protocol=quality_evidence)
        return certificate, None

    leaves, pg_fallbacks, rejected, pg_action_leaves, planned_actions = {}, {}, [], {}, {}
    for node in nodes:
        if 'leaf' not in node:
            continue
        leaf = node['leaf']
        if independent_pg_actions:
            # Choose parameters and cost order using fit-train only. Every PG
            # action receives its own quality certificate; rejecting POST must
            # not erase a separately validated PRE in the same tree region.
            ranked = []
            qids = [q for q in training if assignments[q] == leaf]
            for cid in candidates:
                fast, main = [], []
                for qid in qids:
                    if candidate_availability is not None and cid not in candidate_availability[qid]:
                        continue  # Runtime independently abstains if this exact PRE plan is unavailable.
                    a, b = records.get(('train', qid, cid), []), records[('train', qid, '__main__')]
                    if not a or {r['repeat'] for r in a} != {r['repeat'] for r in b}:
                        break
                    fast.extend(a); main.extend(b)
                else:
                    if (quality_eligible(fast) and
                            independent_mean(fast)+routing_allowance_ms < independent_mean(main)*(1-min_relative_gain)):
                        ranked.append((independent_mean(fast)+routing_allowance_ms, cid))
            # At most one train-selected configuration per PG path. Dev and
            # reserved groups can only veto these frozen choices, not replace
            # their parameters or change the train cost ranking.
            selected_paths, choices = set(), []
            for train_cost, cid in sorted(ranked):
                path = actions[cid].path_id
                if path not in selected_paths:
                    choices.append((train_cost, cid)); selected_paths.add(path)
            planned_actions[str(leaf)] = [cid for _, cid in choices]
            accepted = []
            for train_cost, cid in choices:
                certificate, failure = certify(leaf, cid, require_gain=True)
                if certificate is None:
                    rejected.append(dict(leaf=leaf, config_id=cid, reason=failure))
                else:
                    certificate['train_selection_execution_ms'] = train_cost
                    accepted.append(certificate)
            if accepted:
                pg_action_leaves[str(leaf)] = accepted
            continue
        cid = str(model.classes_[tree.value[leaf][0].argmax()])
        if cid == '__full__':
            rejected.append(dict(leaf=leaf,reason='train_prefers_full_policy'))
        else:
            certificate, failure = certify(leaf, cid, require_gain=True)
            if certificate is not None:
                leaves[str(leaf)] = certificate
            else:
                rejected.append(dict(leaf=leaf,reason=failure))
        # The train-selected partition stays frozen. Dev can only accept/reject
        # this predeclared native action, never tune its parameters or the tree.
        if 'native-pg' in candidates:
            certificate, failure = certify(leaf, 'native-pg', require_gain=False)
            if certificate is not None:
                pg_fallbacks[str(leaf)] = certificate
            else:
                rejected.append(dict(leaf=leaf,purpose='pg_fallback',reason=failure))
    result = dict(version=VERSION,binding=binding,tree_features=list(TREE_FEATURES),nodes=nodes,
        leaves=leaves,pg_fallbacks=pg_fallbacks,rejected=rejected,fit_protocol=dict(parameter_selection='train',enable_or_reject='dev',
            test_labels_used=False,recall_threshold=recall_threshold,min_relative_gain=min_relative_gain,
            probability_min=probability_min,unsupported_rows=unsupported_rows,
            routing_allowance_ms=routing_allowance_ms,latency_cap_ms=latency_cap_ms,
            min_train_groups=min_train_groups,min_dev_groups=min_dev_groups,confidence_z=confidence_z,
            cost_target=cost_target,tail_quantile=tail_quantile,
            train_gate_fraction=train_gate_fraction,
            tree_objective=tree_objective,
            train_gate_groups=sorted(heldout_groups),
            train_gate_query_ids=sorted(gate_ids),tree_fit_query_ids=training,
            quality_evidence='all paired observations pass; group Wilson bound is empirical, not a guarantee',
            physical_availability_conditioned=candidate_availability is not None,
            quality_protocol=quality_evidence,
            tail_protocol=tail_protocol,
            pg_fallback='certified quality/budget; only if full main selector has no feasible candidate',
            censored_main_cost='deadline lower bound; no exact completion latency inferred',
            censored_candidates='retained and reject their selected leaf'))
    if independent_pg_actions:
        result['pg_action_leaves'] = pg_action_leaves
        result['fit_protocol'].update(pg_action_selection='independent_quality_then_train_cost',
            planned_pg_actions=planned_actions, pg_cost_order='fit-train only; dev may veto each action')
    return result
