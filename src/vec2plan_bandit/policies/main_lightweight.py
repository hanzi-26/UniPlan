"""Main-only, calibrated shallow selector; uncertainty falls back to MaxUCB.

The artifact contains a numeric tree and empirical leaf certificates, not a
dataset-name rule. No estimator, SQL execution, model fitting or result cache
is used online. Corpus-size bounds are mandatory even for exact PRE.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import struct

from vec2plan_bandit.paths.registry import PATH_IDS, make_execution_config, pre_configuration_key

VERSION = 'main-lightweight-selector-v2'
PG_NATIVE = 'PG_NATIVE_SQL'


@dataclass(frozen=True)
class MainNativePGConfig:
    """Main-only action: never extend the shared four-path model/registry."""
    path_id: str = PG_NATIVE
    config_id: str = 'native-pg'
    action_id: None = None

    @property
    def params(self):
        return {}


def main_configuration(path_id, params):
    if path_id in ('FAISS_SQL', 'POST_SQL', 'PG_SQL', 'EXACT_SQL'):
        from vec2plan_bandit.execution.main_sql_candidates import configuration
        return configuration(path_id, params)
    if path_id == 'PG_EXACT_SQL':
        from vec2plan_bandit.execution.main_exact_pg import MainExactPGConfig
        if params:
            raise ValueError('exact PG has no path-specific parameters')
        return MainExactPGConfig()
    if path_id == PG_NATIVE:
        if params:
            raise ValueError('native PG has no path-specific parameters')
        return MainNativePGConfig()
    return make_execution_config(path_id, params)


FACT_FIELDS = ('corpus_size', 'candidate_count', 'k', 'selectivity',
               'embedding_dimension', 'vector_blocks', 'join_count', 'filter_count')
TREE_FEATURES = ('log1p_corpus_size', 'log1p_candidate_count', 'log1p_k', 'selectivity',
                 'embedding_dimension', 'vector_blocks', 'join_count', 'filter_count')


def finite(value, minimum=0.):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum)


@lru_cache(maxsize=2048)
def _structure(source_sql):
    from sqlglot import exp
    from sqlglot.errors import SqlglotError
    from vec2plan_bandit.queries.lightweight import routing_select
    try:
        tree = routing_select(source_sql)
    except (ValueError, TypeError, SqlglotError):
        return None
    joins = len(list(tree.find_all(exp.Join)))
    filters = len(list(tree.find_all(exp.Where)))
    complicated = [bool(tree.find(t)) for t in
                   (exp.CTE, exp.Subquery, exp.AggFunc, exp.Group, exp.Having,
                    exp.Window, exp.Offset, exp.Unnest, exp.Lateral)]
    distinct = any(s.args.get('distinct') for s in tree.find_all(exp.Select))
    families = [bool(tree.find(t)) for t in
                (exp.Like, exp.ILike, exp.Between, exp.In, exp.Or, exp.Not)]
    signature = json.dumps([joins, filters, complicated, bool(distinct), families], separators=(',', ':'))
    # Only remove join-setting alternatives when there is nowhere to join.
    # CTE/subquery/function cases retain the original candidates.
    plain = (joins == 0 and not any(complicated) and not distinct
             and len(list(tree.find_all(exp.Table))) == 1
             and not any(not (isinstance(f,exp.Anonymous) and f.name.lower()=='lembed')
                         for f in tree.find_all(exp.Func)))
    return joins, filters, signature, plain


def routing_facts(query, statistics, *, embedding_dimension, corpus_size=None,
                  allow_underfilled_estimate=False):
    """Use pre-execution metadata only; missing/unsupported facts fail closed."""
    statistics = statistics or {}
    shape = _structure(query.source_sql)
    if shape is None:
        return None
    blocks = getattr(query, 'blocks', None)
    count = len(blocks) if blocks is not None else int(bool(getattr(query, 'embedding_column', None)))
    n = statistics.get('corpus_size', corpus_size)
    if n is None and blocks:
        sizes = [getattr(b, 'asset', {}).get('estimated_points') for b in blocks]
        if all(finite(v, 1) for v in sizes):
            n = sum(sizes)
    from vec2plan_bandit.queries.lightweight import retrieval_semantics
    direct_unfiltered=retrieval_semantics(query.source_sql).direct_unfiltered and count == 1
    if not direct_unfiltered and count == 1 and shape[0] == shape[1] == 0:
        from vec2plan_bandit.queries.arxiv_sql_adapter import unfiltered_retrieval_specification
        direct_unfiltered=unfiltered_retrieval_specification(query)
    facts = dict(corpus_size=n, candidate_count=statistics.get('candidate_count'), k=query.k,
                 selectivity=statistics.get('estimated_selectivity'), embedding_dimension=embedding_dimension,
                 vector_blocks=count, join_count=shape[0], filter_count=shape[1],
                 semantic_signature=shape[2], pre_join_settings_irrelevant=shape[3] and count == 1,
                 direct_unfiltered=direct_unfiltered)
    return facts if valid_facts(facts, allow_underfilled_estimate=allow_underfilled_estimate) else None


def valid_facts(facts, *, allow_underfilled_estimate=False):
    if not isinstance(facts, dict) or not all(finite(facts.get(k)) for k in FACT_FIELDS):
        return False
    return (facts['corpus_size'] >= facts['candidate_count'] >= 0 and facts['k'] >= 1
            and (allow_underfilled_estimate or facts['candidate_count'] >= facts['k'])
            and 0 <= facts['selectivity'] <= 1 and facts['embedding_dimension'] >= 1
            and facts['vector_blocks'] >= 1
            and all(int(facts[k]) == facts[k] for k in
                    ('k', 'embedding_dimension', 'vector_blocks', 'join_count', 'filter_count'))
            and isinstance(facts.get('semantic_signature'), str))


def feature_vector(facts, *, allow_underfilled_estimate=False):
    if not valid_facts(facts, allow_underfilled_estimate=allow_underfilled_estimate):
        raise ValueError('incomplete main-selector routing facts')
    values = [math.log1p(facts[k]) if k in ('corpus_size', 'candidate_count', 'k') else float(facts[k])
              for k in FACT_FIELDS]
    # sklearn's training/apply boundary uses float32 even with float64 inputs.
    return [struct.unpack('f', struct.pack('f', v))[0] for v in values]


def pg_semantic_family(query, facts, *, allow_complex=False):
    """Label-free execution families; unsupported complex shapes abstain.

    Exact counts/filter selectivity remain tree inputs and certificate bounds.
    Table names, query IDs and result labels never define a family.
    """
    if facts is None:
        return None
    from sqlglot import exp
    from vec2plan_bandit.queries.lightweight import routing_select
    result=dict(facts,pg_family_supported=False)
    tree=routing_select(query.source_sql)
    if allow_complex:
        # Preserve the full, label-free AST family rather than treating every
        # CTE/aggregate/multi-vector query as the same simple inner join.
        if (any(tree.find(t) for t in (exp.Window,exp.Offset,exp.Union)) or
                any(s.args.get('distinct') for s in tree.find_all(exp.Select)) or
                any(j.args.get('side') or j.args.get('kind','').upper() not in ('','INNER')
                    for j in tree.find_all(exp.Join))):
            return result
        shape=_structure(query.source_sql)
        if shape is None:return result
        result.update(pg_family_supported=True,semantic_signature='relational-v2:'+shape[2])
        return result
    if (facts['vector_blocks']!=1 or any(tree.find(t) for t in
            (exp.CTE,exp.Subquery,exp.AggFunc,exp.Group,exp.Having,exp.Window,exp.Offset,exp.Union))
            or tree.args.get('distinct')):
        return result
    source=tree.args.get('from_');joins=tree.args.get('joins',[])
    if not source or not isinstance(source.this,exp.Table) or any(j.args.get('side') or
            j.args.get('kind','').upper() not in ('','INNER') or j.args.get('on') is None
            or not isinstance(j.this,exp.Table) for j in joins):
        return result
    result.update(pg_family_supported=True,semantic_signature=
        'single-vector-inner-join-v1' if joins else 'single-vector-scalar-v1')
    return result


@dataclass(frozen=True)
class LightweightSelection:
    config: object | None
    reason: str
    evidence: dict | None = None


class MainLightweightSelector:
    def __init__(self, artifact, *, binding):
        if (artifact.get('version') != VERSION or artifact.get('binding') != binding
                or artifact.get('tree_features') != list(TREE_FEATURES)):
            raise ValueError('main lightweight selector provenance/schema mismatch')
        protocol = artifact['fit_protocol']
        if protocol.get('parameter_selection') != 'train' or protocol.get('enable_or_reject') != 'dev' or protocol.get('test_labels_used') is not False:
            raise ValueError('main lightweight selector must be trained on train and accepted on dev')
        if not finite(protocol.get('recall_threshold'),1e-9) or protocol['recall_threshold'] > 1:
            raise ValueError('invalid calibrated Recall threshold')
        self.artifact = artifact
        self.compact_early_features=protocol.get('compact_early_features',False)
        if type(self.compact_early_features) is not bool:
            raise ValueError('invalid compact early feature protocol')
        children=artifact.get('pg_action_trees',[])
        if not isinstance(children,list) or len(children)>16 or any(c.get('pg_action_trees') for c in children):
            raise ValueError('invalid independent PG trees')
        self.pg_action_trees=[MainLightweightSelector(c,binding=binding) for c in children]
        if children:
            if artifact['leaves'] or artifact.get('pg_action_leaves'):
                raise ValueError('independent PG tree container cannot have extra leaf routes')
            for child in children:
                if child.get('priority_rules') or child.get('pg_fallbacks') or child['leaves'] or any(
                        c['path_id'] not in ('PRE','POST') for cs in child.get('pg_action_leaves',{}).values() for c in cs):
                    raise ValueError('PG trees may only select PRE/POST')
        self.overhead_budget_ms = protocol['routing_allowance_ms']
        if not finite(self.overhead_budget_ms, 1e-9):
            raise ValueError('invalid lightweight routing allowance')
        nodes = artifact['nodes']
        if not nodes or len(nodes) > 255:
            raise ValueError('lightweight tree must have 1..255 nodes')
        visiting, visited = set(), set()
        def validate(index):
            if type(index) is not int or not 0 <= index < len(nodes) or index in visiting:
                raise ValueError('invalid or cyclic lightweight tree')
            if index in visited:
                raise ValueError('lightweight tree has shared descendants')
            visiting.add(index)
            node = nodes[index]
            if 'leaf' in node:
                if type(node['leaf']) is not int or node['leaf'] < 0:
                    raise ValueError('invalid leaf identity')
            else:
                if type(node.get('feature')) is not int or not 0 <= node['feature'] < len(TREE_FEATURES) or not finite(node.get('threshold'), -math.inf):
                    raise ValueError('invalid tree feature/threshold')
                validate(node['left']); validate(node['right'])
            visiting.remove(index); visited.add(index)
        validate(0)
        if len(visited) != len(nodes):
            raise ValueError('unreachable lightweight nodes')
        leaf_ids = [str(node['leaf']) for node in nodes if 'leaf' in node]
        if len(leaf_ids)!=len(set(leaf_ids)) or set(artifact['leaves'])-set(leaf_ids):
            raise ValueError('duplicate or unreachable leaf certificates')
        fallback_leaves = artifact.get('pg_fallbacks', {})
        if set(fallback_leaves)-set(leaf_ids):
            raise ValueError('unreachable PG fallback certificate')
        pg_actions = artifact.get('pg_action_leaves', {})
        if 'pg_action_leaves' in artifact:
            if (set(pg_actions)-set(leaf_ids) or artifact['leaves'] or
                    protocol.get('pg_action_selection') != 'independent_quality_then_train_cost'):
                raise ValueError('invalid independent PG admission schema')
            for choices in pg_actions.values():
                if (not isinstance(choices, list) or not 1 <= len(choices) <= 2 or
                        any(c.get('path_id') not in ('PRE', 'POST') or
                            not finite(c.get('train_selection_execution_ms')) for c in choices) or
                        len({c['path_id'] for c in choices}) != len(choices) or
                        choices != sorted(choices, key=lambda c: (c['train_selection_execution_ms'], c['config_id']))):
                    raise ValueError('PG actions require independent certificates and train-only cost order')
        priority_rules = artifact.get('priority_rules', [])
        if not isinstance(priority_rules, list) or len(priority_rules) > 2:
            raise ValueError('invalid ordered main rules')
        if len({r['path_id'] for r in priority_rules}) != len(priority_rules):
            raise ValueError('duplicate priority rules')
        if priority_rules and [r['path_id'] for r in priority_rules] != sorted(
                [r['path_id'] for r in priority_rules], key=lambda p: {'FAISS':0,PG_NATIVE:1}.get(p,2)):
            raise ValueError('priority rules must put FAISS before native PG')
        for rule in priority_rules:
            if (rule['path_id'] not in ('FAISS', PG_NATIVE)
                    or rule.get('scope') != ('simple_unfiltered' if rule['path_id']=='FAISS' else 'corpus_bound')
                    or not finite(rule.get('corpus_upper'), 1)):
                raise ValueError('invalid priority rule scope')
        for leaf in [*artifact['leaves'].values(), *fallback_leaves.values(), *priority_rules,
                     *(c for choices in pg_actions.values() for c in choices)]:
            config = main_configuration(leaf['path_id'], leaf['params'])
            if config.config_id != leaf['config_id'] or leaf['path_id'] not in (*PATH_IDS, PG_NATIVE):
                raise ValueError('invalid leaf execution configuration')
            for field in FACT_FIELDS:
                bounds = leaf['bounds'].get(field, [])
                if len(bounds) != 2 or not all(finite(v) for v in bounds) or bounds[0] > bounds[1]:
                    raise ValueError('missing finite leaf support bounds')
            if (not finite(leaf.get('quality_lower_bound')) or not 0 <= leaf['quality_lower_bound'] <= 1
                    or not finite(leaf.get('e2e_upper_ms'), 1e-9)
                    or not finite(leaf.get('e2e_mean_ms'))
                    or not isinstance(leaf.get('semantic_signatures'),list) or not leaf['semantic_signatures']
                    or not all(isinstance(s,str) for s in leaf['semantic_signatures'])
                    or not finite(leaf.get('quality_mean')) or not leaf['quality_lower_bound'] <= leaf['quality_mean'] <= 1
                    or leaf['e2e_mean_ms'] > leaf['e2e_upper_ms']
                    or not all(type(leaf.get(k)) is int and leaf[k] >= 1 for k in ('train_groups','dev_groups'))):
                raise ValueError('invalid empirical leaf certificate')
        if any(leaf['path_id'] != PG_NATIVE for leaf in fallback_leaves.values()):
            raise ValueError('only native PG may use a PG fallback certificate')
        if any(leaf.get('acceptance') != 'quality_and_budget_only' for leaf in fallback_leaves.values()):
            raise ValueError('PG fallback needs its own quality/budget certificate')
        if any(leaf['path_id'] == PG_NATIVE and leaf.get('acceptance') != 'quality_and_e2e_gain'
               for leaf in [*artifact['leaves'].values(), *priority_rules]):
            raise ValueError('early PG selection requires measured E2E gain')

    @classmethod
    def load(cls, path, *, binding):
        return cls(json.loads(Path(path).read_text()), binding=binding)

    def propose(self, facts, *, available, available_pre_configurations,
                recall_threshold, probability_min, latency_cap_ms, remaining_budget_ms,
                purpose='early'):
        if purpose not in ('early', 'pg_fallback'):
            raise ValueError('unknown main selection purpose')
        def reject(reason):
            return LightweightSelection(None, reason)
        if not valid_facts(facts):
            return reject('missing_or_unsupported_facts')
        protocol = self.artifact['fit_protocol']
        if recall_threshold > protocol['recall_threshold']:
            return reject('stricter_recall_than_calibration')
        if purpose == 'early':
            for rule in self.artifact.get('priority_rules', []):
                if rule['scope'] == 'simple_unfiltered' and not facts.get('direct_unfiltered', False):
                    continue
                if facts['corpus_size'] > rule['corpus_upper']:
                    continue
                if (facts['semantic_signature'] not in rule['semantic_signatures']
                        or any(not rule['bounds'][k][0] <= facts[k] <= rule['bounds'][k][1] for k in FACT_FIELDS)
                        or rule['quality_lower_bound'] < probability_min
                        or rule['e2e_upper_ms'] > min(latency_cap_ms,remaining_budget_ms)
                        or not available.get(rule['path_id'], False)):
                    continue
                return LightweightSelection(main_configuration(rule['path_id'],rule['params']),
                    'priority_unfiltered_faiss' if rule['path_id']=='FAISS' else 'priority_native_pg',rule)
            if self.pg_action_trees:
                proposals=[child.propose(facts,available=available,
                    available_pre_configurations=available_pre_configurations,recall_threshold=recall_threshold,
                    probability_min=probability_min,latency_cap_ms=latency_cap_ms,
                    remaining_budget_ms=remaining_budget_ms) for child in self.pg_action_trees]
                valid=[v for v in proposals if v.config is not None]
                if valid:
                    return min(valid,key=lambda v:(v.evidence['train_selection_execution_ms'],v.config.config_id))
                return reject('independent_pg_trees_abstained')
        try:
            x = feature_vector(facts)
        except (OverflowError, ValueError):
            return reject('unrepresentable_facts')
        index = 0
        while 'leaf' not in self.artifact['nodes'][index]:
            node = self.artifact['nodes'][index]
            index = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
        leaf_id = str(self.artifact['nodes'][index]['leaf'])
        if purpose == 'early' and 'pg_action_leaves' in self.artifact:
            # Quality, scale, readiness and budget are checked independently.
            # The order was frozen on fit-train; no test costs/labels or online
            # exploratory SQL are used to pick between the two PG paths.
            reason = 'leaf_not_validated'
            for leaf in self.artifact['pg_action_leaves'].get(leaf_id, []):
                selected = self._propose_leaf(leaf, facts, available, available_pre_configurations,
                    probability_min, latency_cap_ms, remaining_budget_ms)
                if selected.config is not None:
                    return selected
                reason = selected.reason
            return reject(reason)
        certificates = self.artifact['leaves'] if purpose == 'early' else self.artifact.get('pg_fallbacks', {})
        leaf = certificates.get(leaf_id)
        if leaf is None:
            return reject('leaf_not_validated')
        return self._propose_leaf(leaf, facts, available, available_pre_configurations,
            probability_min, latency_cap_ms, remaining_budget_ms)

    @staticmethod
    def _propose_leaf(leaf, facts, available, available_pre_configurations,
                      probability_min, latency_cap_ms, remaining_budget_ms):
        def reject(reason):
            return LightweightSelection(None, reason)
        if facts['semantic_signature'] not in leaf['semantic_signatures']:
            return reject('unvalidated_query_structure')
        if any(not leaf['bounds'][k][0] <= facts[k] <= leaf['bounds'][k][1] for k in FACT_FIELDS):
            return reject('outside_calibrated_scale')
        if leaf['quality_lower_bound'] < probability_min:
            return reject('insufficient_quality_evidence')
        if leaf['e2e_upper_ms'] > min(latency_cap_ms, remaining_budget_ms):
            return reject('insufficient_remaining_budget')
        config = main_configuration(leaf['path_id'], leaf['params'])
        if not available.get(config.path_id, False):
            return reject('path_unavailable')
        # PRE must have current physical-plan evidence, even though it is exact.
        if config.path_id == 'PRE' and (available_pre_configurations is None or
                pre_configuration_key(config.params) not in available_pre_configurations):
            return reject('pre_plan_not_verified')
        return LightweightSelection(config, 'validated_leaf', leaf)
