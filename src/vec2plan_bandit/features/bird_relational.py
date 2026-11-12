"""Optional label-free relational plan features, shared by all BIRD selectors."""
import math

import numpy as np

from vec2plan_bandit.bird.experiment import BirdFeatures

PLAN_FEATURES = ('pre_plan_rows', 'join_intermediate_rows', 'join_fanout', 'aggregate_rows', 'pre_plan_cost')
PROFILE_FEATURES=('best_exact_plan_cost','scan_plan_cost','optimizer_plan_cost','hash_join_plan_cost','merge_join_plan_cost','nested_loop_plan_cost')


def plan_statistics(payload):
    from vec2plan_bandit.execution.plan_semantics import plan_nodes
    nodes = list(plan_nodes(payload))
    joins = [n for n in nodes if n.get('Node Type') in ('Hash Join','Merge Join','Nested Loop')]
    fanout = [n.get('Plan Rows',0)/max(1,max((c.get('Plan Rows',0) for c in n.get('Plans',[])),default=1)) for n in joins]
    return dict(pre_plan_rows=max((n.get('Plan Rows',0) for n in nodes),default=0),
        join_intermediate_rows=max((n.get('Plan Rows',0) for n in joins),default=0),
        join_fanout=max(fanout,default=0),
        aggregate_rows=max((n.get('Plan Rows',0) for n in nodes if 'Aggregate' in n.get('Node Type','')),default=0),
        pre_plan_cost=max((n.get('Total Cost',0) for n in nodes),default=0))


def plan_profile_statistics(plans):
    from vec2plan_bandit.execution.plan_semantics import plan_nodes
    values={key:max((n.get('Total Cost',0) for n in plan_nodes(payload)),default=0)
            for key,payload in plans.items()}
    def best(keys):
        return min((values[k] for k in keys if k in values),default=0)
    return dict(best_exact_plan_cost=best(values),scan_plan_cost=best(k for k in values if k.startswith('scan:')),
        optimizer_plan_cost=best(k for k in values if k.startswith('optimizer:')),
        **{f'{join}_plan_cost':best(k for k in values if k.endswith(':'+strategy))
            for join,strategy in [('hash_join','hash'),('merge_join','merge'),('nested_loop','nested_loop')]})


class BirdRelationalFeatures(BirdFeatures):
    @property
    def base_feature_names(self):
        return super().base_feature_names + [f'log1p_{k}' for k in PLAN_FEATURES]

    def _base_vector(self, query, statistics):
        original = super()._base_vector(query, statistics)
        stats = (statistics or {}).get(query.query_id,{})
        values = [float(stats.get(k,0)) for k in PLAN_FEATURES]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid label-free relational plan statistics')
        return np.concatenate((original,np.log1p(values)))


class BirdRelationalPlanFeatures(BirdRelationalFeatures):
    @property
    def base_feature_names(self):
        return super().base_feature_names + [f'log1p_{k}' for k in PROFILE_FEATURES]

    def _base_vector(self,query,statistics):
        original=super()._base_vector(query,statistics)
        stats=(statistics or {}).get(query.query_id,{})
        values=np.asarray([stats.get(k,0.) for k in PROFILE_FEATURES],dtype=float)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('invalid exact-plan profile statistics')
        return np.concatenate((original,np.log1p(values)))
