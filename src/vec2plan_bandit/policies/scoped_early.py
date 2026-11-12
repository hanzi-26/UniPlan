"""Capability-scoped empirical routing with exact-result checks for short truth.

Only offline train/dev data contains result identities. Online routing uses
query facts, available paths and verified plans; it never reads truth or IDs.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .competitive_early import CompetitiveEarlySelector, fit_competitive_early, leaf_index
from .main_lightweight import LightweightSelection, valid_facts

VERSION = 'main-scoped-early-v1'
COARSE_SCOPE = 'supported_sql'


def capability_scope(facts):
    if not valid_facts(facts, allow_underfilled_estimate=True):
        return None
    try:
        shape = json.loads(facts['semantic_signature'])
    except (ValueError, TypeError):
        return None
    if (not isinstance(shape, list) or len(shape) != 5
            or shape[:2] != [facts['join_count'], facts['filter_count']]
            or not isinstance(shape[2], list) or len(shape[2]) != 9
            or type(shape[3]) is not bool or not isinstance(shape[4], list) or len(shape[4]) != 6
            or not all(type(v) is bool for v in shape[2] + shape[4])):
        return None
    return ('simple_unfiltered' if facts.get('direct_unfiltered') is True
            and facts['vector_blocks'] == 1 else 'relational_sql')


def scoped_facts(facts, scope):
    return {**facts, 'semantic_signature': scope}


def intersection_key(scope, fine_leaf, coarse_leaf):
    return json.dumps([scope, fine_leaf, coarse_leaf], separators=(',', ':'))


def fit_scoped_early(rows, *, truth_by_query, train_ids, dev_ids, binding, coarse_partition_nodes,
                     min_train_groups=2, min_dev_groups=1, max_depth=4):
    roles = {'train': set(train_ids), 'dev': set(dev_ids)}
    if roles['train'] & roles['dev'] or set(truth_by_query) - (roles['train'] | roles['dev']):
        raise ValueError('Only train/dev truth and disjoint roles are allowed')
    records, per_scope = [], defaultdict(list)
    audit = dict(below_k_rows=0, below_k_exact_matches=0, primary_quality_rows=0, unsupported_rows=0)
    for original in rows:
        role, qid = original['split'], original['query_id']
        if role not in roles or qid not in roles[role] or original.get('measurement_scope') != 'e2e':
            raise ValueError('Only train/dev E2E observations are accepted')
        if qid not in truth_by_query:
            raise ValueError('Missing offline truth for a calibration query')
        row = deepcopy(original)
        facts = row.get('routing_facts')
        scope = capability_scope(facts)
        if scope is None:
            audit['unsupported_rows'] += 1
            continue
        truth = truth_by_query[qid]
        eligible = len(truth) >= facts['k']
        if bool(row.get('eligible_for_primary_quality')) != eligible:
            raise ValueError('Calibration eligibility and prepared truth disagree')
        if eligible:
            audit['primary_quality_rows'] += 1
        else:
            if 'row_ids' not in row:
                raise ValueError('Below-k validation needs the actual returned identities')
            row['below_k_exact_match'] = Counter(map(tuple, row['row_ids'])) == Counter(map(tuple, truth))
            audit['below_k_rows'] += 1
            audit['below_k_exact_matches'] += int(row['below_k_exact_match'])
        row['routing_facts'] = scoped_facts(facts, scope)
        per_scope[scope].append(row)
        coarse = deepcopy(row)
        coarse['routing_facts'] = scoped_facts(facts, COARSE_SCOPE)
        records.append(coarse)
    options = dict(train_ids=train_ids, dev_ids=dev_ids, binding=binding, recall_threshold=1.,
        min_train_groups=min_train_groups, min_dev_groups=min_dev_groups,
        min_relative_gain=0., train_regret_fraction=.1, train_regret_ms=2.,
        allow_underfilled_estimate=True, validate_below_k_exact=True)
    trees = {}
    for scope, values in sorted(per_scope.items()):
        if not any(r['split'] == 'train' and r['mode'] == 'main' for r in values):
            continue
        trees[scope] = fit_competitive_early(values, max_depth=max_depth, **options)
    coarse = fit_competitive_early(records, partition_nodes=coarse_partition_nodes, **options)
    intersections, subsets = {}, defaultdict(list)
    for scope, values in per_scope.items():
        if scope not in trees:
            continue
        for row in values:
            f = row['routing_facts']
            key = intersection_key(scope,
                leaf_index(trees[scope]['nodes'], f, allow_underfilled_estimate=True),
                leaf_index(coarse['nodes'], f, allow_underfilled_estimate=True))
            copied = deepcopy(row); copied['routing_facts'] = scoped_facts(f, key)
            subsets[key].append(copied)
    for key, values in sorted(subsets.items()):
        if any(r['split'] == 'train' and r['mode'] == 'main' for r in values):
            intersections[key] = fit_competitive_early(values, partition_nodes=[{'leaf': 0}], **options)
    return dict(version=VERSION, binding=binding, scopes=trees, coarse=coarse,
        intersections=intersections, calibration_audit=audit,
        protocol=dict(parameter_selection='train', enable_or_reject='dev', test_labels_used=False,
            truth_used_online=False, recall_threshold=1., allow_underfilled_estimate=True,
            validate_below_k_exact=True, primary_metric_eligibility_unchanged=True,
            max_depth=max_depth, min_train_groups=min_train_groups, min_dev_groups=min_dev_groups,
            semantic_matching='validated_query_capability_and_numeric_support',
            action_comparison='matched fine/coarse train intersections; dev accepts or rejects',
            evidence='empirical only; zero primary groups implies quality lower bound zero'))


class ScopedEarlySelector:
    compact_early_features = True
    inference_engine = VERSION
    overhead_budget_ms = .5

    def __init__(self, artifact, *, binding):
        if artifact.get('version') != VERSION or artifact.get('binding') != binding:
            raise ValueError('Scoped routing provenance/schema mismatch')
        p = artifact['protocol']
        if (p.get('parameter_selection') != 'train' or p.get('enable_or_reject') != 'dev'
                or p.get('test_labels_used') is not False or p.get('truth_used_online') is not False
                or p.get('validate_below_k_exact') is not True or p.get('allow_underfilled_estimate') is not True
                or p.get('recall_threshold') != 1.):
            raise ValueError('Invalid scoped routing evidence protocol')
        if set(artifact['scopes']) - {'simple_unfiltered', 'relational_sql'}:
            raise ValueError('Unknown capability scope')
        self.artifact = artifact
        self.scopes = {key: CompetitiveEarlySelector(tree, binding=binding)
                       for key, tree in artifact['scopes'].items()}
        self.coarse = CompetitiveEarlySelector(artifact['coarse'], binding=binding)
        self.intersections = {key: CompetitiveEarlySelector(tree, binding=binding)
                              for key, tree in artifact['intersections'].items()}
        for key in self.intersections:
            try:
                scope, fine_leaf, coarse_leaf = json.loads(key)
                valid_key = (scope in self.scopes and type(fine_leaf) is int and type(coarse_leaf) is int
                    and fine_leaf in {n['leaf'] for n in artifact['scopes'][scope]['nodes'] if 'leaf' in n}
                    and coarse_leaf in {n['leaf'] for n in artifact['coarse']['nodes'] if 'leaf' in n})
            except (ValueError, TypeError):
                valid_key = False
            if not valid_key:
                raise ValueError('Invalid matched capability intersection')
        for scope, selector in [*self.scopes.items(), (COARSE_SCOPE, self.coarse), *self.intersections.items()]:
            if not selector.artifact['protocol'].get('validate_below_k_exact'):
                raise ValueError('Scoped children need below-k exact validation')
            if any(leaf['semantic_signatures'] != [scope] for leaf in selector.artifact['leaves'].values()):
                raise ValueError('Child semantic capability differs from its scope')

    @classmethod
    def load(cls, path, *, binding, sha256):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError('Scoped routing artifact changed')
        return cls(json.loads(payload), binding=binding)

    def propose(self, facts, *, available, available_pre_configurations,
                recall_threshold, probability_min, latency_cap_ms, remaining_budget_ms, purpose='early'):
        scope = capability_scope(facts)
        if purpose != 'early' or scope is None:
            return LightweightSelection(None, 'unsupported_scoped_query_capability')
        options = dict(available=available, available_pre_configurations=available_pre_configurations,
            recall_threshold=recall_threshold, probability_min=probability_min,
            latency_cap_ms=latency_cap_ms, remaining_budget_ms=remaining_budget_ms, purpose=purpose)
        selectors = []
        if scope in self.scopes:
            key = intersection_key(scope,
                leaf_index(self.scopes[scope].artifact['nodes'], facts, allow_underfilled_estimate=True),
                leaf_index(self.coarse.artifact['nodes'], facts, allow_underfilled_estimate=True))
            if key in self.intersections:
                selectors.append((key, self.intersections[key]))
        selectors.append((COARSE_SCOPE, self.coarse))
        for key, selector in selectors:
            selection = selector.propose(scoped_facts(facts, key), **options)
            if selection.config is not None:
                return LightweightSelection(selection.config, 'scoped_' + key, selection.evidence)
        return LightweightSelection(None, 'scoped_and_coarse_evidence_unavailable')
