"""Train-selected shallow routing across competing fixed execution actions.

Finite train/dev evidence is an empirical fallback, never a probability-one
certificate. Dataset names, query identities and test labels are not features.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from .main_lightweight import (
    FACT_FIELDS, TREE_FEATURES, PG_NATIVE, LightweightSelection, feature_vector,
    main_configuration, valid_facts,
)
from vec2plan_bandit.paths.registry import pre_configuration_key

VERSION = 'main-competitive-early-v2'
LEGACY_VERSION = 'main-competitive-early-v1'


def candidate_paths(binding):
    paths = ('FAISS', 'POST', 'PRE', PG_NATIVE)
    if binding.get('main_exact_pg', {}).get('enabled') is True:
        paths += ('PG_EXACT_SQL',)
    return paths


def validate_nodes(nodes):
    visited = set()
    def visit(i):
        if type(i) is not int or not 0 <= i < len(nodes) or i in visited:
            raise ValueError('Invalid competitive routing tree')
        visited.add(i)
        node = nodes[i]
        if 'leaf' in node:
            if node['leaf'] != i:
                raise ValueError('Invalid leaf identity')
        else:
            if (type(node.get('feature')) is not int or not 0 <= node['feature'] < len(TREE_FEATURES)
                    or not math.isfinite(node['threshold'])):
                raise ValueError('Invalid tree split')
            visit(node['left']); visit(node['right'])
    if not nodes or len(nodes) > 63:
        raise ValueError('Competitive routing tree must be shallow')
    visit(0)
    if len(visited) != len(nodes):
        raise ValueError('Unreachable competitive routing nodes')


def leaf_index(nodes, facts, *, allow_underfilled_estimate=False):
    x, i = feature_vector(facts, allow_underfilled_estimate=allow_underfilled_estimate), 0
    while 'leaf' not in nodes[i]:
        node = nodes[i]
        i = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
    return nodes[i]['leaf']


def action_supported(action, facts):
    return action.path_id != 'FAISS' or (
        facts['vector_blocks'] == 1 and facts.get('direct_unfiltered') is True)


def fit_competitive_early(rows, *, train_ids, dev_ids, binding,
                          recall_threshold=1., max_depth=3,
                          min_train_groups=5, min_dev_groups=2,
                          routing_allowance_ms=.25, min_relative_gain=.02,
                          train_regret_fraction=0., train_regret_ms=0.,
                          allow_underfilled_estimate=False, partition_nodes=None,
                          validate_below_k_exact=False):
    """Train determines the partition and winning action; dev only rejects it."""
    if (not 0 < recall_threshold <= 1 or type(max_depth) is not int
            or not 1 <= max_depth <= 5 or min_train_groups < 1 or min_dev_groups < 1
            or not 0 < routing_allowance_ms <= 5 or not 0 <= min_relative_gain < 1
            or not 0 <= train_regret_fraction <= 1 or not 0 <= train_regret_ms < 5000
            or bool(train_regret_fraction) != bool(train_regret_ms)
            or type(allow_underfilled_estimate) is not bool or type(validate_below_k_exact) is not bool):
        raise ValueError('Invalid competitive routing settings')
    roles = {'train': set(train_ids), 'dev': set(dev_ids)}
    if roles['train'] & roles['dev']:
        raise ValueError('Overlapping train/dev identities')
    records, facts, groups, group_role, actions = defaultdict(list), {}, {}, {}, {}
    seen = set()
    for row in rows:
        role, qid = row['split'], row['query_id']
        if role not in roles or qid not in roles[role] or row.get('measurement_scope') != 'e2e':
            raise ValueError('Only paired train/dev E2E measurements are accepted')
        group = row['semantic_group']
        if group in group_role and group_role[group] != role:
            raise ValueError('Semantic group crosses train/dev')
        group_role[group] = role
        f = row.get('routing_facts')
        if not valid_facts(f, allow_underfilled_estimate=allow_underfilled_estimate):
            continue
        if qid in facts and (facts[qid] != f or groups[qid] != group):
            raise ValueError('Inconsistent routing facts or semantic groups')
        facts[qid], groups[qid] = f, group
        if row['mode'] == 'main':
            cid = '__main__'
        elif row['mode'] == 'candidate':
            action = main_configuration(row['path_id'], row['requested_params'])
            cid = action.config_id
            if cid != row['config_id'] or action.path_id not in candidate_paths(binding):
                raise ValueError('Invalid competitive candidate identity')
            actions[cid] = action
        else:
            raise ValueError('Unknown calibration mode')
        key = (role, qid, cid, row['repeat'])
        if key in seen:
            raise ValueError('Duplicate calibration observation')
        seen.add(key)
        cost = row.get('e2e_latency_ms')
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError('Invalid E2E latency')
        records[(role, qid, cid)].append(row)

    def good(row):
        execution_ok = (row.get('execution_success', True)
            and not any(row.get(k) for k in ('timed_out', 'latency_censored', 'error_type',
                                             'fallback_used', 'router_fallback'))
            and row['e2e_latency_ms'] + routing_allowance_ms < 5000.)
        # Below-k truth is excluded from quality evidence, not from latency.
        below_k_ok = row.get('below_k_exact_match') is True if validate_below_k_exact else True
        quality_ok = (
            row.get('result_full_k') and row.get('recall_at_k') is not None
            and recall_threshold <= row['recall_at_k'] <= 1.
        ) if row.get('eligible_for_primary_quality') else below_k_ok
        return execution_ok and quality_ok

    def independent_mean(rs):
        per_query, per_group = defaultdict(list), defaultdict(list)
        for row in rs:
            per_query[row['query_id']].append(row['e2e_latency_ms'])
        for qid, values in per_query.items():
            per_group[groups[qid]].append(statistics.median(values))
        return statistics.mean(statistics.mean(v) for v in per_group.values())

    def cost_allowed(role, mean, reference):
        if role == 'train' and train_regret_fraction:
            return mean <= reference + min(reference * train_regret_fraction, train_regret_ms)
        return mean < reference * (1 - min_relative_gain)

    def measure(role, qids, cid):
        selected, reference = [], []
        for qid in qids:
            a = records.get((role, qid, cid), [])
            b = records.get((role, qid, '__main__'), [])
            if not a or not b or {r['repeat'] for r in a} != {r['repeat'] for r in b}:
                return None, 'missing_paired_measurements'
            if not action_supported(actions[cid], facts[qid]):
                return None, 'unsupported_action_scope'
            if not all(good(r) for r in a):
                return None, 'quality_or_execution_failure'
            selected.extend(a)
            reference.extend(b)
        primary_support = len({r['semantic_group'] for r in selected if r.get('eligible_for_primary_quality')})
        support = len({r['semantic_group'] for r in selected if
                       r.get('eligible_for_primary_quality') or validate_below_k_exact})
        if support < (min_train_groups if role == 'train' else min_dev_groups):
            return None, 'insufficient_independent_quality_groups'
        mean = independent_mean(selected) + routing_allowance_ms
        if not cost_allowed(role, mean, independent_mean(reference)):
            return None, 'no_gain_over_full_policy'
        return dict(rows=selected, groups=support, primary_groups=primary_support, mean_ms=mean,
                    upper_ms=max(r['e2e_latency_ms'] for r in selected) + routing_allowance_ms), None

    training = sorted(q for q in roles['train'] if q in facts and records.get(('train', q, '__main__')))
    if not training or not actions:
        raise ValueError('Missing train observations or competing actions')
    labels = []
    for qid in training:
        choices = []
        reference = records[('train', qid, '__main__')]
        repeats = {r['repeat'] for r in reference}
        for cid, action in actions.items():
            values = records.get(('train', qid, cid), [])
            if (not action_supported(action, facts[qid]) or not values
                    or {r['repeat'] for r in values} != repeats or not all(good(r) for r in values)
                    or not any(r.get('eligible_for_primary_quality') or validate_below_k_exact for r in values)):
                continue
            mean = statistics.median(r['e2e_latency_ms'] for r in values) + routing_allowance_ms
            if cost_allowed('train', mean, statistics.median(r['e2e_latency_ms'] for r in reference)):
                choices.append((mean, cid))
        labels.append(min(choices)[1] if choices else '__full__')
    if partition_nodes is None:
        from sklearn.tree import DecisionTreeClassifier
        counts = Counter(groups[q] for q in training)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_train_groups, random_state=42)
        model.fit([feature_vector(facts[q], allow_underfilled_estimate=allow_underfilled_estimate) for q in training], labels,
                  sample_weight=[1 / counts[groups[q]] for q in training])
        tree = model.tree_
        nodes = [dict(leaf=i) if tree.children_left[i] < 0 else dict(feature=int(tree.feature[i]),
            threshold=float(tree.threshold[i]), left=int(tree.children_left[i]), right=int(tree.children_right[i]))
            for i in range(tree.node_count)]
    else:
        nodes = copy.deepcopy(partition_nodes)
    validate_nodes(nodes)
    assignments = {q: leaf_index(nodes, f, allow_underfilled_estimate=allow_underfilled_estimate)
                   for q, f in facts.items()}
    leaves, rejected, choices_audit = {}, [], []
    for node in nodes:
        if 'leaf' not in node:
            continue
        leaf = node['leaf']
        qids = {role: sorted(q for q in ids if assignments.get(q) == leaf) for role, ids in roles.items()}
        choices = []
        for cid in sorted(actions):
            result, reason = measure('train', qids['train'], cid)
            choices_audit.append(dict(leaf=leaf, config_id=cid, train_rejection=reason,
                                     train_mean_ms=result['mean_ms'] if result else None))
            if result:
                choices.append((result['mean_ms'], cid, result))
        if not choices:
            rejected.append(dict(leaf=leaf, reason='no_qualified_train_action'))
            continue
        _, cid, train = min(choices, key=lambda v: (v[0], v[1]))
        dev, reason = measure('dev', qids['dev'], cid)
        if reason:
            rejected.append(dict(leaf=leaf, config_id=cid, reason='dev_' + reason))
            continue
        signatures = ({facts[q]['semantic_signature'] for q in qids['train']}
                      & {facts[q]['semantic_signature'] for q in qids['dev']})
        if not signatures:
            rejected.append(dict(leaf=leaf, config_id=cid, reason='no_shared_semantics'))
            continue
        all_rows = train['rows'] + dev['rows']
        fs = [r['routing_facts'] for r in all_rows]
        primary_rows = [r for r in all_rows if r.get('eligible_for_primary_quality')]
        leaves[str(leaf)] = dict(path_id=actions[cid].path_id, params=dict(actions[cid].params),
            config_id=cid, acceptance='empirical_best_effort', quality_mean=1.,
            quality_lower_bound=min(n / (n + 1.6448536269514722**2)
                                    for n in [train['primary_groups'], dev['primary_groups']]),
            train_groups=train['groups'], dev_groups=dev['groups'],
            primary_train_groups=train['primary_groups'], primary_dev_groups=dev['primary_groups'],
            eligible_observation_count=len(primary_rows),
            result_agreement_lower_bound=min(n / (n + 1.6448536269514722**2)
                                             for n in [train['groups'], dev['groups']]),
            min_observed_recall=min(r['recall_at_k'] for r in primary_rows) if primary_rows else None,
            semantic_signatures=sorted(signatures),
            bounds={k: [min(f[k] for f in fs), max(f[k] for f in fs)] for k in FACT_FIELDS},
            e2e_mean_ms=max(train['mean_ms'], dev['mean_ms']),
            e2e_upper_ms=max(train['upper_ms'], dev['upper_ms']))
    return dict(version=VERSION, binding=binding, tree_features=list(TREE_FEATURES), nodes=nodes,
        leaves=leaves, rejected=rejected, train_choices=choices_audit,
        protocol=dict(parameter_selection='train', enable_or_reject='dev', test_labels_used=False,
            recall_threshold=recall_threshold, max_depth=max_depth, min_train_groups=min_train_groups,
            min_dev_groups=min_dev_groups, routing_allowance_ms=routing_allowance_ms,
            min_relative_gain=min_relative_gain, train_regret_fraction=train_regret_fraction,
            train_regret_ms=train_regret_ms, allow_underfilled_estimate=allow_underfilled_estimate,
            validate_below_k_exact=validate_below_k_exact,
            partition_selection='train' if partition_nodes is None else 'reused_train_only_partition',
            partition_sha256=hashlib.sha256(json.dumps(nodes, sort_keys=True).encode()).hexdigest(),
            evidence='empirical best effort; no probability-one certificate'))


class CompetitiveEarlySelector:
    compact_early_features = True
    inference_engine = VERSION

    def __init__(self, artifact, *, binding):
        if (artifact.get('version') not in (VERSION, LEGACY_VERSION) or artifact.get('binding') != binding
                or artifact.get('tree_features') != list(TREE_FEATURES)):
            raise ValueError('Competitive routing provenance/schema mismatch')
        protocol = artifact['protocol']
        if (protocol.get('parameter_selection') != 'train' or protocol.get('enable_or_reject') != 'dev'
                or protocol.get('test_labels_used') is not False):
            raise ValueError('Competitive routing needs train/dev evidence only')
        self.artifact = artifact
        self.allow_underfilled_estimate = protocol.get('allow_underfilled_estimate', False)
        if type(self.allow_underfilled_estimate) is not bool:
            raise ValueError('Invalid competitive facts validation settings')
        self.overhead_budget_ms = .5
        nodes = artifact['nodes']
        validate_nodes(nodes)
        self.configurations = {}
        for key, leaf in artifact['leaves'].items():
            if key not in {str(n['leaf']) for n in nodes if 'leaf' in n}:
                raise ValueError('Unreachable competitive routing leaf')
            config = main_configuration(leaf['path_id'], leaf['params'])
            recall_valid = (leaf['min_observed_recall'] is not None
                            and leaf['min_observed_recall'] >= protocol['recall_threshold'])
            if (protocol.get('validate_below_k_exact', False)
                    and leaf.get('eligible_observation_count') == 0 and leaf['min_observed_recall'] is None
                    and leaf['quality_lower_bound'] == 0):
                recall_valid = True
            if (config.path_id not in candidate_paths(binding) or config.config_id != leaf['config_id']
                    or leaf.get('acceptance') != 'empirical_best_effort' or leaf['quality_mean'] != 1.
                    or not 0 <= leaf['quality_lower_bound'] < 1
                    or not recall_valid
                    or leaf['train_groups'] < protocol['min_train_groups']
                    or leaf['dev_groups'] < protocol['min_dev_groups'] or not leaf['semantic_signatures']
                    or not 0 <= leaf['e2e_mean_ms'] <= leaf['e2e_upper_ms'] < 5000.):
                raise ValueError('Invalid competitive routing evidence')
            for k in FACT_FIELDS:
                lo, hi = leaf['bounds'][k]
                if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo <= hi:
                    raise ValueError('Invalid competitive routing support bounds')
            self.configurations[key] = config

    @classmethod
    def load(cls, path, *, binding, sha256):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError('Competitive routing artifact changed')
        return cls(json.loads(payload), binding=binding)

    def propose(self, facts, *, available, available_pre_configurations,
                recall_threshold, probability_min, latency_cap_ms, remaining_budget_ms, purpose='early'):
        if purpose != 'early' or not valid_facts(facts, allow_underfilled_estimate=self.allow_underfilled_estimate):
            return LightweightSelection(None, 'unsupported_competitive_scope')
        if recall_threshold > self.artifact['protocol']['recall_threshold']:
            return LightweightSelection(None, 'stricter_recall_than_competitive_fit')
        key = str(leaf_index(self.artifact['nodes'], facts,
                            allow_underfilled_estimate=self.allow_underfilled_estimate))
        leaf = self.artifact['leaves'].get(key)
        if leaf is None:
            return LightweightSelection(None, 'competitive_leaf_not_validated')
        config = self.configurations[key]
        if (not action_supported(config, facts) or not available.get(config.path_id, False)
                or facts['semantic_signature'] not in leaf['semantic_signatures']
                or any(not leaf['bounds'][k][0] <= facts[k] <= leaf['bounds'][k][1] for k in FACT_FIELDS)
                or leaf['e2e_upper_ms'] > min(latency_cap_ms, remaining_budget_ms)):
            return LightweightSelection(None, 'outside_competitive_validated_scope')
        if config.path_id == 'PRE' and (available_pre_configurations is None
                or pre_configuration_key(config.params) not in available_pre_configurations):
            return LightweightSelection(None, 'competitive_pre_plan_not_verified')
        return LightweightSelection(config, 'competitive_train_selected_leaf', leaf)
