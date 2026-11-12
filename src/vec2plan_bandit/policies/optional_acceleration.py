"""Dataset-independent, empirically validated acceleration before a main policy.

Descriptors contain structural and numeric facts, never dataset/column names.
Train chooses one action per bucket. Dev can only enable or reject that action.
All costs used for the gate are measured E2E costs, including routing overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from numbers import Real
from pathlib import Path
import statistics

from vec2plan_bandit.paths.registry import make_execution_config

VERSION = 'optional-acceleration-v1'
FIELDS = ('n', 'm', 'k', 'selectivity')


@lru_cache(maxsize=16)
def _load(path, binding_json):
    return OptionalAccelerationRouter.load(path, binding=json.loads(binding_json))


def configured_acceleration(config):
    """One opt-in configuration contract, shared by all runtime adapters.

    Load once per process. The caller must record and verify expected corpus,
    split, encoder, indexes and frozen-main identity in the binding, as it does
    for model checkpoints. Changing an artifact requires a new frozen run.
    """
    options = config.get('optional_acceleration', {})
    if not options.get('enabled', False):
        return None
    return _load(str(Path(options['artifact']).resolve()),
                 json.dumps(options['binding'], sort_keys=True))


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def describe_query(query, stats, *, corpus_size=None):
    """Normalize facts from any SQL adapter; estimated counts may be fractional.

    sqlglot is loaded only when acceleration is used. Unsupported SQL or missing
    statistics disable the fast route rather than guessing a dataset rule.
    """
    stats = stats or {}
    n = stats.get('corpus_size', corpus_size)
    m, s, k = stats.get('candidate_count'), stats.get('estimated_selectivity'), query.k
    blocks = getattr(query, 'blocks', None)
    count = len(blocks) if blocks is not None else int(bool(getattr(query, 'embedding_column', '')))
    if (not count or not all(_number(v) for v in (n, m, s, k)) or
            n <= 0 or not 0 <= m <= n or not 0 <= s <= 1 or k < 1 or int(k) != k or m < k):
        return None
    structure = _sql_structure(query.source_sql)
    if structure is None:
        return None
    key = [count, *structure,
           int(math.log10(n)), 'zero' if s == 0 else int(math.floor(math.log10(s))),
           int(math.floor(math.log2(k)))]
    return {'key': json.dumps(key, separators=(',', ':')), 'n': float(n), 'm': float(m),
            'k': int(k), 'selectivity': float(s)}


@lru_cache(maxsize=4096)
def _sql_structure(source_sql):
    """Cache parsing only; numeric statistics and routing decisions stay live."""
    from sqlglot import exp
    from sqlglot.errors import SqlglotError
    from vec2plan_bandit.queries.lightweight import routing_select
    try:
        tree = routing_select(source_sql)
    except (SqlglotError, ValueError):
        return None
    if not isinstance(tree, exp.Select):
        return None
    wheres = list(tree.find_all(exp.Where))
    families = sorted({name for name, kinds in (
        ('text', (exp.Like, exp.ILike)),
        ('range', (exp.Between, exp.GT, exp.GTE, exp.LT, exp.LTE)),
        ('membership', (exp.In,)),
        ('comparison', (exp.EQ, exp.NEQ, exp.Is)),
        ('boolean', (exp.Or, exp.Not)),
    ) if any(w.find(t) for w in wheres for t in kinds)})
    # Preserve every filter family, rather than making dataset-specific priority
    # decisions when a query contains more than one kind of predicate.
    if wheres and not families:
        families = ['other']
    return (len(list(tree.find_all(exp.Join))), bool(tree.find(exp.CTE)),
            bool(tree.find(exp.Group) or tree.find(exp.Having) or tree.find(exp.AggFunc)),
            bool(tree.find(exp.Window)), bool(tree.find(exp.Offset)),
            any(t.args.get('distinct') for t in tree.find_all(exp.Select)), families)


def _valid_description(desc):
    return (isinstance(desc, dict) and isinstance(desc.get('key'), str) and
            all(_number(desc.get(f)) for f in FIELDS) and desc['n'] > 0 and
            1 <= desc['k'] <= desc['m'] <= desc['n'] and int(desc['k']) == desc['k'] and
            0 <= desc['selectivity'] <= 1)


def _validate_action(row):
    if row['path_id'] == 'PG_NATIVE_SQL':
        if row['config_id'] != 'native-pg' or row['requested_params']:
            raise ValueError('Invalid native PG action')
    else:
        config = make_execution_config(row['path_id'], row['requested_params'])
        if config.config_id != row['config_id']:
            raise ValueError('Configuration identity mismatch')


@dataclass(frozen=True)
class AccelerationDecision:
    rule: dict | None
    reason: str


class OptionalAccelerationRouter:
    def __init__(self, artifact, *, binding):
        if artifact.get('version') != VERSION or artifact.get('binding') != binding:
            raise ValueError('Acceleration artifact version/corpus/split/encoder mismatch')
        self.artifact = artifact
        self.rules = {}
        if artifact.get('fit_protocol', {}).get('budget_faiss') and any(
                r['path_id'] != 'FAISS' for r in artifact['rules']):
            raise ValueError('Budget shortcut only accepts FAISS actions')
        for rule in artifact['rules']:
            _validate_action({**rule, 'requested_params': rule['params']})
            if rule['key'] in self.rules:
                raise ValueError('Duplicate acceleration bucket')
            if any(not _number(rule.get(f + suffix)) for f in FIELDS for suffix in ('_min', '_max')):
                raise ValueError('Invalid acceleration bounds')
            if any(rule[f + '_min'] > rule[f + '_max'] for f in FIELDS):
                raise ValueError('Reversed acceleration bounds')
            self.rules[rule['key']] = rule

    @classmethod
    def load(cls, path, *, binding):
        return cls(json.loads(Path(path).read_text()), binding=binding)

    def select(self, description, *, available_paths, available_configurations=None,
               recall_threshold=None, latency_cap_ms=None):
        if not _valid_description(description):
            return AccelerationDecision(None, 'unsupported_or_missing_statistics')
        rule = self.rules.get(description['key'])
        if rule is None:
            return AccelerationDecision(None, 'no_validated_acceleration')
        if recall_threshold is not None and self.artifact['fit_protocol']['recall_threshold'] < recall_threshold:
            return AccelerationDecision(None, 'quality_target_mismatch')
        if latency_cap_ms is not None and rule['calibrated_e2e_max_ms'] >= latency_cap_ms:
            return AccelerationDecision(None, 'insufficient_latency_budget')
        if not available_paths.get(rule['path_id'], False):
            return AccelerationDecision(None, 'fast_path_unavailable')
        if available_configurations is not None and rule['config_id'] not in available_configurations:
            return AccelerationDecision(None, 'fast_configuration_unavailable')
        if not all(rule[f + '_min'] <= description[f] <= rule[f + '_max'] for f in FIELDS):
            return AccelerationDecision(None, 'outside_calibrated_range')
        return AccelerationDecision(rule, 'validated_quality_and_e2e_gain')

    def select_budget_faiss(self, description, *, remaining_ms, threshold_ms=100., **kwargs):
        if not _number(threshold_ms) or threshold_ms <= 0:
            raise ValueError('Invalid budget shortcut threshold')
        if not _number(remaining_ms) or not 0 < remaining_ms < threshold_ms:
            return AccelerationDecision(None, 'outside_budget_shortcut_window')
        if not self.artifact['fit_protocol'].get('budget_faiss'):
            return AccelerationDecision(None, 'no_budget_faiss_validation')
        decision = self.select(description, latency_cap_ms=remaining_ms, **kwargs)
        if decision.rule is None:
            return decision
        return AccelerationDecision(decision.rule, 'budget_faiss_quality_validated')


def fit_acceleration(rows, *, binding, train_ids, dev_ids, recall_threshold=.8,
                     latency_cap_ms=10000, min_relative_gain=.05,
                     min_train_groups=2, min_dev_groups=1, budget_faiss=False):
    """Fit a single optional fast action against measured frozen-main requests.

    Rows: split/query_id/semantic_group/repeat, mode=main|candidate,
    routing_description, e2e_latency_ms, quality fields; candidates additionally
    identify a path/config/params and measured routing_overhead_ms. Candidate E2E
    must include fresh embedding, SQL/result composition and postprocessing.
    The explicit overhead field measures description + lookup, and is added once.
    Repeats never count as independent semantic groups. Test rows are forbidden.
    budget_faiss=True restricts actions to FAISS and requires quality/support,
    without requiring a gain against a main request that may miss its deadline.
    """
    train_ids, dev_ids = set(train_ids), set(dev_ids)
    if train_ids & dev_ids:
        raise ValueError('Train/dev overlap')
    if (not 0 < recall_threshold <= 1 or not _number(latency_cap_ms) or latency_cap_ms <= 0 or
            not 0 <= min_relative_gain < 1 or min_train_groups < 1 or min_dev_groups < 1):
        raise ValueError('Invalid calibration thresholds')
    buckets, seen, group_roles, descriptions = {}, set(), {}, {}
    for row in rows:
        role, qid, mode = row['split'], row['query_id'], row['mode']
        if role not in ('train', 'dev') or qid not in (train_ids if role == 'train' else dev_ids):
            raise ValueError('Held-out or incorrectly labelled calibration row')
        if mode not in ('main', 'candidate'):
            raise ValueError('Unknown calibration mode')
        semantic = row['semantic_group']
        if semantic in group_roles and group_roles[semantic] != role:
            raise ValueError('Semantic group crosses splits')
        group_roles[semantic] = role
        desc = row['routing_description']
        if not _valid_description(desc):
            raise ValueError('Invalid calibration description')
        if qid in descriptions and descriptions[qid] != (semantic, desc):
            raise ValueError('Query description/group changed during calibration')
        descriptions[qid] = (semantic, desc)
        if mode == 'candidate':
            _validate_action(row)
            if budget_faiss and row['path_id'] != 'FAISS':
                raise ValueError('Budget calibration only accepts FAISS')
            if not _number(row.get('routing_overhead_ms')) or row['routing_overhead_ms'] < 0:
                raise ValueError('Measured routing overhead is required')
        if row.get('measurement_scope') != 'e2e':
            raise ValueError('Fresh end-to-end measurements are required, not execution-only labels')
        identity = (role, qid, mode, row.get('config_id') if mode == 'candidate' else None, row['repeat'])
        if identity in seen:
            raise ValueError('Duplicate calibration observation')
        seen.add(identity)
        buckets.setdefault(desc['key'], []).append(row)

    def cost(r):
        overhead = (r['routing_overhead_ms'] if r['mode'] == 'candidate' and
                    not r.get('e2e_includes_routing', False) else 0.)
        return r['e2e_latency_ms'] + overhead

    def usable(rs):
        return bool(rs) and all(_number(r.get('e2e_latency_ms')) and
            r['e2e_latency_ms'] >= 0 and not r.get('timed_out') and not r.get('error_type') for r in rs)

    def quality(rs):
        return usable(rs) and all(r.get('eligible_for_primary_quality') and r.get('result_full_k') and
            r.get('execution_success', True) and _number(r.get('recall_at_k')) and
            recall_threshold <= r['recall_at_k'] <= 1 and cost(r) < latency_cap_ms for r in rs)

    def independent_cost(rs):
        groups = {}
        for qid in sorted({r['query_id'] for r in rs}):
            qr = [r for r in rs if r['query_id'] == qid]
            groups.setdefault(qr[0]['semantic_group'], []).append(statistics.median(cost(r) for r in qr))
        return statistics.mean(statistics.mean(values) for values in groups.values())

    def supported(fast, main, min_groups):
        identities = lambda rs: {(r['query_id'], r['repeat']) for r in rs}
        if (not quality(fast) or (not budget_faiss and not usable(main)) or identities(fast) != identities(main) or
                len({r['semantic_group'] for r in fast}) < min_groups):
            return False
        return budget_faiss or independent_cost(fast) < independent_cost(main) * (1 - min_relative_gain)

    rules, rejected = [], []
    for key, bucket in sorted(buckets.items()):
        main = {role: [r for r in bucket if r['mode'] == 'main' and r['split'] == role]
                for role in ('train', 'dev')}
        candidates = [r for r in bucket if r['mode'] == 'candidate']
        choices = []
        for cid in sorted({r['config_id'] for r in candidates}):
            tr = [r for r in candidates if r['config_id'] == cid and r['split'] == 'train']
            if supported(tr, main['train'], min_train_groups):
                choices.append((independent_cost(tr), cid, tr))
        if not choices:
            rejected.append({'key': key, 'reason': 'train_quality_or_support' if budget_faiss else 'train_quality_support_or_e2e_gain'})
            continue
        _, cid, tr = min(choices, key=lambda v: (v[0], v[1]))
        dv = [r for r in candidates if r['config_id'] == cid and r['split'] == 'dev']
        if not supported(dv, main['dev'], min_dev_groups):
            rejected.append({'key': key, 'reason': 'dev_quality_or_support' if budget_faiss else 'dev_quality_support_or_e2e_gain'})
            continue
        rule = {'key': key, 'path_id': tr[0]['path_id'], 'config_id': cid,
                'params': tr[0]['requested_params'], 'train_groups': len({r['semantic_group'] for r in tr}),
                'dev_groups': len({r['semantic_group'] for r in dv}),
                'dev_min_recall': min(r['recall_at_k'] for r in dv),
                'calibrated_e2e_max_ms': max(cost(r) for r in tr + dv),
                'train_relative_gain': 1 - independent_cost(tr) / max(independent_cost(main['train']), 1e-9),
                'dev_relative_gain': 1 - independent_cost(dv) / max(independent_cost(main['dev']), 1e-9)}
        for field in FIELDS:
            rule[field + '_min'] = min(r['routing_description'][field] for r in tr + dv)
            rule[field + '_max'] = max(r['routing_description'][field] for r in tr + dv)
        rules.append(rule)
    return {'version': VERSION, 'binding': binding, 'rules': rules, 'rejected': rejected,
            'fit_protocol': {'parameter_selection': 'train', 'enable_or_reject': 'dev',
                'recall_threshold': recall_threshold, 'latency_cap_ms': latency_cap_ms,
                'budget_faiss': budget_faiss, 'require_e2e_gain': not budget_faiss,
                'min_relative_gain': min_relative_gain, 'min_train_groups': min_train_groups,
                'min_dev_groups': min_dev_groups, 'objective': 'mean of semantic-group E2E medians',
                'quality_evidence': 'all observations pass; empirical, not a probability guarantee',
                'test_labels_used': False}}
