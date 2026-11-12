"""Replace a validated PRE route only after fresh paired exact-PG evidence."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .competitive_early import CompetitiveEarlySelector, fit_competitive_early
from .main_lightweight import LightweightSelection
from .scoped_early import ScopedEarlySelector
from vec2plan_bandit.execution.main_exact_pg import PG_EXACT

VERSION = 'main-exact-optimizer-v1'


def rebind_scoped(artifact, binding):
    """Transfer frozen rules explicitly; source deltas are audited by the queue.

    No old latency is used as a measurement of the new exact executor.
    """
    result = deepcopy(artifact)
    result['binding'] = binding
    for tree in [*result['scopes'].values(), result['coarse'], *result['intersections'].values()]:
        tree['binding'] = binding
    ScopedEarlySelector(result, binding=binding)
    return result


def fit_exact_optimizer(rows, *, fallback, binding, truth_by_query, train_ids, dev_ids):
    roles = {'train': set(train_ids), 'dev': set(dev_ids)}
    if roles['train'] & roles['dev'] or set(truth_by_query) - (roles['train'] | roles['dev']):
        raise ValueError('Exact optimizer accepts train/dev only')
    per_route = defaultdict(list)
    for original in rows:
        role, qid = original['split'], original['query_id']
        if role not in roles or qid not in roles[role] or qid not in truth_by_query:
            raise ValueError('Unexpected calibration identity or truth role')
        row = deepcopy(original)
        if row['mode'] == 'candidate':
            if row['path_id'] != PG_EXACT or row.get('no_ann_plan_verified') is not True:
                raise ValueError('Only verified exact candidates may enter this supplement')
            agreement = Counter(map(tuple, row['row_ids'])) == Counter(map(tuple, truth_by_query[qid]))
            row['below_k_exact_match'] = agreement
            row['execution_success'] = agreement and row.get('execution_success', True)
        elif row['mode'] != 'main':
            raise ValueError('Unexpected reference observation mode')
        if row.get('reference_proposal_path') != 'PRE':
            raise ValueError('The frozen reference proposal must be PRE')
        if bool(row['eligible_for_primary_quality']) != (len(truth_by_query[qid]) >= row['routing_facts']['k']):
            raise ValueError('Prepared quality eligibility changed')
        route = row['reference_route']
        row['routing_facts']['semantic_signature'] = route
        per_route[route].append(row)
    rules = {route: fit_competitive_early(values, train_ids=train_ids, dev_ids=dev_ids,
        binding=binding, partition_nodes=[{'leaf': 0}], min_train_groups=2, min_dev_groups=1,
        routing_allowance_ms=.25, min_relative_gain=.02, allow_underfilled_estimate=True,
        validate_below_k_exact=True) for route, values in sorted(per_route.items())}
    return dict(version=VERSION, binding=binding, fallback=fallback, rules=rules,
        protocol=dict(parameter_selection='train', enable_or_reject='dev', test_labels_used=False,
            truth_used_online=False, allow_underfilled_estimate=True,
            scope='only the frozen selector PRE proposals', empirical_not_probability_certificate=True))


class ExactOptimizerSelector:
    compact_early_features = True
    inference_engine = VERSION
    overhead_budget_ms = .5

    def __init__(self, artifact, *, binding):
        if (artifact.get('version') != VERSION or artifact.get('binding') != binding
                or binding.get('main_exact_pg', {}).get('enabled') is not True
                or artifact['protocol'].get('test_labels_used') is not False
                or artifact['protocol'].get('truth_used_online') is not False):
            raise ValueError('Exact optimizer provenance/protocol mismatch')
        self.artifact = artifact
        self.fallback = ScopedEarlySelector(artifact['fallback'], binding=binding)
        self.rules = {key: CompetitiveEarlySelector(tree, binding=binding)
                      for key, tree in artifact['rules'].items()}
        for key, rule in self.rules.items():
            if any(leaf['path_id'] != PG_EXACT or leaf['semantic_signatures'] != [key]
                   for leaf in rule.artifact['leaves'].values()):
                raise ValueError('Unexpected exact optimizer replacement action')

    @classmethod
    def load(cls, path, *, binding, sha256):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError('Exact optimizer artifact changed')
        return cls(json.loads(data), binding=binding)

    def propose(self, facts, **options):
        prior = self.fallback.propose(facts, **options)
        if prior.config is not None and prior.config.path_id == 'PRE' and prior.reason in self.rules:
            selection = self.rules[prior.reason].propose(
                {**facts, 'semantic_signature': prior.reason}, **options)
            if selection.config is not None:
                return LightweightSelection(selection.config, 'exact_optimizer_' + prior.reason, selection.evidence)
        return prior
