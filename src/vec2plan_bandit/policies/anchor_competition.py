"""Train-selected actions compete against quality-qualified fixed anchors."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .competitive_early import leaf_index, validate_nodes
from .main_lightweight import FACT_FIELDS, TREE_FEATURES, LightweightSelection, feature_vector, valid_facts
from .scoped_early import ScopedEarlySelector
from vec2plan_bandit.execution.main_sql_candidates import SQL_PATHS, configuration

VERSION = 'main-anchor-competition-v1'


def fit_anchor_competition(rows, *, truth_by_query, train_ids, dev_ids, binding, fallback,
                           max_depth=5, min_train_groups=2, min_dev_groups=1, routing_allowance_ms=.15,
                           recall_threshold=1.):
    if not isinstance(recall_threshold, (int, float)) or isinstance(recall_threshold, bool) or not 0 < recall_threshold <= 1:
        raise ValueError('Invalid recall threshold')
    roles = {'train': set(train_ids), 'dev': set(dev_ids)}
    if roles['train'] & roles['dev'] or set(truth_by_query) - (roles['train'] | roles['dev']):
        raise ValueError('Only disjoint train/dev roles are allowed')
    records, facts, groups, group_roles, actions, seen = defaultdict(list), {}, {}, {}, {}, set()
    for row in rows:
        role, qid = row['split'], row['query_id']
        if (role not in roles or qid not in roles[role] or qid not in truth_by_query
                or row.get('measurement_scope') != 'e2e' or row['mode'] not in ('candidate', 'anchor')):
            raise ValueError('Unexpected calibration observation')
        f, group = row['routing_facts'], row['semantic_group']
        if not valid_facts(f, allow_underfilled_estimate=True):
            raise ValueError('Invalid calibration facts')
        if qid in facts and (facts[qid] != f or groups[qid] != group):
            raise ValueError('Inconsistent query metadata')
        if group in group_roles and group_roles[group] != role:
            raise ValueError('Semantic group crosses roles')
        if bool(row['eligible_for_primary_quality']) != (len(truth_by_query[qid]) >= f['k']):
            raise ValueError('Quality denominator changed')
        facts[qid], groups[qid], group_roles[group] = f, group, role
        if row['mode'] == 'candidate':
            action = configuration(row['path_id'], row['requested_params'])
            if action.config_id != row['config_id']:
                raise ValueError('Candidate identity mismatch')
            key = action.config_id; actions[key] = action
        else:
            if row['path_id'] not in ('POST', 'PG_NATIVE_SQL'):
                raise ValueError('Unexpected fixed anchor')
            key = 'anchor:' + row['path_id']
        identity = (role, qid, key, row['repeat'])
        if identity in seen:
            raise ValueError('Duplicate observation')
        seen.add(identity)
        if not isinstance(row['e2e_latency_ms'], (int, float)) or not math.isfinite(row['e2e_latency_ms']) or row['e2e_latency_ms'] < 0:
            raise ValueError('Invalid latency')
        records[(role, qid, key)].append(row)

    def good(r):
        agreement = Counter(map(tuple, r['row_ids'])) == Counter(map(tuple, truth_by_query[r['query_id']]))
        quality = (r.get('result_full_k') and r.get('recall_at_k') is not None
                   and recall_threshold <= r['recall_at_k'] <= 1.) if r['eligible_for_primary_quality'] else agreement
        if recall_threshold == 1.:
            quality = quality and agreement  # Preserve the existing strict mode.
        return (r.get('execution_success', True) and r['e2e_latency_ms'] + routing_allowance_ms < 5000
            and not any(r.get(k) for k in ('error_type', 'timed_out', 'latency_censored', 'fallback_used'))
            and quality)

    def measure(role, qids, key):
        values = [records.get((role, qid, key), []) for qid in qids]
        if not qids or any(not rs or not all(good(r) for r in rs) for rs in values):
            return None
        per_group = defaultdict(list)
        for qid, rs in zip(qids, values):
            per_group[groups[qid]].append(statistics.median(r['e2e_latency_ms'] for r in rs))
        observations = [r for rs in values for r in rs]
        return dict(mean=statistics.mean(statistics.mean(v) for v in per_group.values()),
            p95=float(np.percentile([r['e2e_latency_ms'] for r in observations], 95)),
            upper=max(r['e2e_latency_ms'] for r in observations), rows=observations,
            groups=len(per_group), primary_groups=len({r['semantic_group'] for r in observations if r['eligible_for_primary_quality']}))

    training = sorted(q for q in roles['train'] if q in facts)
    if not training or not actions:
        raise ValueError('Missing training candidates')
    expected_keys = set(actions) | {'anchor:POST', 'anchor:PG_NATIVE_SQL'}
    for role, qid in [(role, qid) for role in roles for qid in roles[role] if qid in facts]:
        repeats = {key: {r['repeat'] for r in records.get((role, qid, key), [])} for key in expected_keys}
        if not all(repeats.values()) or len({tuple(sorted(v)) for v in repeats.values()}) != 1:
            raise ValueError('Candidate/anchor repetitions are not paired')
    labels = []
    for qid in training:
        choices = [(result['mean'] + .1 * result['p95'], key) for key in actions
                   if (result := measure('train', [qid], key)) is not None]
        labels.append(min(choices)[1] if choices else '__fallback__')
    from sklearn.tree import DecisionTreeClassifier
    counts = Counter(groups[q] for q in training)
    tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_train_groups, random_state=42)
    tree_model.fit([feature_vector(facts[q], allow_underfilled_estimate=True) for q in training], labels,
                   sample_weight=[1 / counts[groups[q]] for q in training])
    tree = tree_model.tree_
    nodes = [dict(leaf=i) if tree.children_left[i] < 0 else dict(feature=int(tree.feature[i]),
        threshold=float(tree.threshold[i]), left=int(tree.children_left[i]), right=int(tree.children_right[i]))
        for i in range(tree.node_count)]
    assignments = {q: leaf_index(nodes, f, allow_underfilled_estimate=True) for q, f in facts.items()}
    leaves, rejected = {}, []
    for node in nodes:
        if 'leaf' not in node: continue
        leaf = node['leaf']
        ids = {role: sorted(q for q in roles[role] if assignments.get(q) == leaf) for role in roles}
        choices = []
        for key in sorted(actions):
            result = measure('train', ids['train'], key)
            if result is not None and result['groups'] >= min_train_groups:
                choices.append((result['mean'] + .1 * result['p95'], key, result))
        if not choices:
            rejected.append(dict(leaf=leaf, reason='no_quality_qualified_train_candidate')); continue
        _, key, train = min(choices, key=lambda v: (v[0], v[1]))
        dev = measure('dev', ids['dev'], key)
        if dev is None or dev['groups'] < min_dev_groups:
            rejected.append(dict(leaf=leaf, reason='dev_quality_or_support')); continue
        comparisons, acceptable = {}, True
        for role, candidate in [('train', train), ('dev', dev)]:
            anchors = {a: result for a in ('POST', 'PG_NATIVE_SQL')
                       if (result := measure(role, ids[role], 'anchor:' + a)) is not None}
            comparisons[role] = {a: dict(mean_ms=r['mean'], p95_ms=r['p95']) for a, r in anchors.items()}
            if anchors and candidate['mean'] + routing_allowance_ms >= min(r['mean'] for r in anchors.values()):
                acceptable = False
        if not acceptable:
            rejected.append(dict(leaf=leaf, reason='does_not_beat_qualified_anchor', comparisons=comparisons)); continue
        signatures = {facts[q]['semantic_signature'] for q in ids['train']} & {facts[q]['semantic_signature'] for q in ids['dev']}
        if not signatures:
            rejected.append(dict(leaf=leaf, reason='no_shared_semantics')); continue
        rs = train['rows'] + dev['rows']; fs = [r['routing_facts'] for r in rs]
        n = min(train['primary_groups'], dev['primary_groups'])
        leaves[str(leaf)] = dict(path_id=actions[key].path_id, params=actions[key].params,
            config_id=key, acceptance='empirical_best_effort', quality_mean=1.,
            quality_lower_bound=n / (n + 1.6448536269514722**2),
            train_groups=train['groups'], dev_groups=dev['groups'],
            semantic_signatures=sorted(signatures),
            bounds={k: [min(f[k] for f in fs), max(f[k] for f in fs)] for k in FACT_FIELDS},
            e2e_mean_ms=max(train['mean'], dev['mean']) + routing_allowance_ms,
            e2e_upper_ms=max(train['upper'], dev['upper']) + routing_allowance_ms,
            anchor_comparisons=comparisons)
    return dict(version=VERSION, binding=binding, nodes=nodes, tree_features=list(TREE_FEATURES),
        leaves=leaves, rejected=rejected, fallback=fallback,
        protocol=dict(parameter_selection='train', enable_or_reject='dev', test_labels_used=False,
            recall_threshold=recall_threshold,
            truth_used_online=False, allow_underfilled_estimate=True, min_train_groups=min_train_groups,
            min_dev_groups=min_dev_groups, max_depth=max_depth, routing_allowance_ms=routing_allowance_ms,
            objective='quality-qualified candidate vs fastest quality-qualified POST/PG anchor',
            empirical_not_probability_certificate=True))


class AnchorCompetitionSelector:
    compact_early_features = True
    inference_engine = VERSION
    overhead_budget_ms = .5

    def __init__(self, artifact, *, binding):
        if (artifact.get('version') != VERSION or artifact.get('binding') != binding
                or artifact.get('tree_features') != list(TREE_FEATURES)
                or binding.get('main_sql_candidates', {}).get('enabled') is not True
                or artifact['protocol'].get('parameter_selection') != 'train'
                or artifact['protocol'].get('enable_or_reject') != 'dev'
                or artifact['protocol'].get('test_labels_used') is not False
                or artifact['protocol'].get('truth_used_online') is not False):
            raise ValueError('Anchor competition provenance/protocol mismatch')
        threshold = artifact['protocol'].get('recall_threshold', 1.)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError('Invalid routing Recall evidence threshold')
        self.artifact = artifact
        validate_nodes(artifact['nodes'])
        self.fallback = ScopedEarlySelector(artifact['fallback'], binding=binding)
        self.configurations = {}
        for key, leaf in artifact['leaves'].items():
            config = configuration(leaf['path_id'], leaf['params'])
            if (key not in {str(n['leaf']) for n in artifact['nodes'] if 'leaf' in n}
                    or config.config_id != leaf['config_id'] or leaf['acceptance'] != 'empirical_best_effort'
                    or leaf['quality_mean'] != 1. or not 0 <= leaf['quality_lower_bound'] < 1
                    or leaf['train_groups'] < artifact['protocol']['min_train_groups']
                    or leaf['dev_groups'] < artifact['protocol']['min_dev_groups']
                    or not 0 <= leaf['e2e_mean_ms'] <= leaf['e2e_upper_ms'] < 5000):
                raise ValueError('Invalid candidate evidence')
            for k in FACT_FIELDS:
                lo, hi = leaf['bounds'][k]
                if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo <= hi:
                    raise ValueError('Invalid metadata bounds')
            self.configurations[key] = config

    @classmethod
    def load(cls, path, *, binding, sha256):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError('Anchor competition artifact changed')
        return cls(json.loads(data), binding=binding)

    def propose(self, facts, **options):
        if (valid_facts(facts, allow_underfilled_estimate=True) and options.get('purpose', 'early') == 'early'
                and options['recall_threshold'] <= self.artifact['protocol'].get('recall_threshold', 1.)):
            key = str(leaf_index(self.artifact['nodes'], facts, allow_underfilled_estimate=True))
            leaf = self.artifact['leaves'].get(key)
            if leaf is not None:
                config = self.configurations[key]
                if (options['available'].get(config.path_id, False)
                        and facts['semantic_signature'] in leaf['semantic_signatures']
                        and all(leaf['bounds'][k][0] <= facts[k] <= leaf['bounds'][k][1] for k in FACT_FIELDS)
                        and leaf['e2e_upper_ms'] <= min(options['latency_cap_ms'], options['remaining_budget_ms'])):
                    return LightweightSelection(config, 'anchor_competition_leaf_' + key, leaf)
        return self.fallback.propose(facts, **options)
