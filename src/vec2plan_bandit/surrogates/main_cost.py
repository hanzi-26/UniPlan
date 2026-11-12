"""Main-only quality masking, completed execution cost and timeout-risk head.

Timeouts train the risk head, never a fictitious 5s completion-time target.
The raw scale/relational facts are pre-execution inputs, not query identities.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .catboost_ensemble import (CatBoostEnsembleSurrogate, ConstrainedCatBoostSurrogate,
                               ConstrainedPrediction, Prediction)

FACTS = ('corpus_size','candidate_count','k','selectivity','embedding_dimension',
         'vector_blocks','join_count','filter_count')


def fact_vector(facts):
    facts = facts or {}
    result = [float(bool(facts))]
    for name in FACTS:
        value = facts.get(name,0.)
        if value is None or not np.isfinite(value) or value < 0:
            value = 0.
        result.append(float(value) if name=='selectivity' else math.log1p(float(value)))
    # Explicitly expose the smaller allowed/excluded set and relational work.
    n,m = facts.get('corpus_size',0.) or 0.,facts.get('candidate_count',0.) or 0.
    result.append(math.log1p(max(0.,min(m,n-m))))
    result.append(math.log1p(max(0.,m)*(1+max(0.,facts.get('join_count',0.) or 0.))))
    return result


def feedback_fields(row):
    spans = row.get('spans_ms',{}) or {}
    execution = row.get('execution_ms',spans.get('execution',spans.get('execution_ms')))
    if execution is None:
        execution = float('nan')
    stages=row.get('execution_stages_ms',row.get('execution_details',{}).get('execution_stages_ms',{})) or {}
    stage_costs=[sum(stages.get(k,0.) for k in names) for names in
        [('filter_prepare_ms','candidate_filter_ms'),('ann_search_ms','pg_execute_ms'),
         ('id_mapping_ms','result_assembly_ms'),('sql_compile_ms','execution_unattributed_ms')]]
    return dict(quality_eligible=bool(row.get('eligible_for_primary_quality',row.get('corpus_has_k',True))),
        censored=bool(row.get('timed_out') or row.get('latency_censored')),
        execution_ms=float(execution),routing_facts=row.get('main_routing_facts'),
        execution_error=bool(row.get('error_type')),stage_costs=stage_costs,stage_costs_known=bool(stages))


class MainCostSurrogate:
    VERSION = 'main-quality-execution-timeout-v1'
    latency_target = 'execution_ms'

    def __init__(self, *, training_devices=(), **options):
        self.base = ConstrainedCatBoostSurrogate(**options)
        risk = dict(options)
        risk.update(ensemble_size=2,iterations=min(int(options.get('iterations',300)),120),depth=4,
                    seed=int(options.get('seed',42))+20000)
        self.timeout_model = CatBoostEnsembleSurrogate(**risk)
        self.training_devices = tuple(str(d) for d in training_devices)
        for model in (self.base.quality_model,self.base.latency_model,self.timeout_model):
            if self.training_devices:
                model.training_devices = self.training_devices
        self.base_feature_count = None
        self.training_audit = {}
        self.stage_tree = None
        self.fused_inference = None

    @property
    def feature_count_(self):
        return self.base_feature_count

    @property
    def quality_model(self): return self.base.quality_model
    @property
    def latency_model(self): return self.base.latency_model
    @property
    def fast_inference(self): return self.fused_inference or getattr(self.base,'fast_inference',None)
    @property
    def use_fast_inference(self):return getattr(self.base,'use_fast_inference',True)
    @use_fast_inference.setter
    def use_fast_inference(self,value):self.base.use_fast_inference=value
    @property
    def use_shared_pool(self):return getattr(self.base,'use_shared_pool',True)
    @use_shared_pool.setter
    def use_shared_pool(self,value):self.base.use_shared_pool=value

    def _augment(self,x,facts):
        x = np.atleast_2d(np.asarray(x,dtype=float))
        return np.column_stack((x,np.asarray([fact_vector(f) for f in facts],dtype=float)))

    def fit_feedback(self,features,quality,latencies,metadata):
        x = self._augment(features,[r.get('routing_facts') for r in metadata])
        quality = np.asarray(quality,dtype=float)
        eligible = np.asarray([r.get('quality_eligible',True) for r in metadata],dtype=bool)
        censored = np.asarray([r.get('censored',False) for r in metadata],dtype=bool)
        execution = np.asarray([r.get('execution_ms',float('nan')) for r in metadata],dtype=float)
        errors = np.asarray([r.get('execution_error',False) for r in metadata],dtype=bool)
        completed = ~censored & ~errors & np.isfinite(execution) & (execution>=0)
        if eligible.sum()<2 or completed.sum()<2:
            raise ValueError('main cost model needs evaluable quality and completed execution labels')
        def fresh(old):
            model=CatBoostEnsembleSurrogate(ensemble_size=old.ensemble_size,iterations=old.iterations,
                depth=old.depth,learning_rate=old.learning_rate,seed=old.seed,
                uncertainty_floor=old.uncertainty_floor,estimator_factory=old._estimator_factory,
                task_type=old.task_type,devices=old.devices)
            if self.training_devices:model.training_devices=self.training_devices
            return model
        quality_model=fresh(self.base.quality_model).fit(x[eligible],quality[eligible])
        latency_model=fresh(self.base.latency_model).fit(x[completed],np.log1p(execution[completed]))
        timeout_model=fresh(self.timeout_model).fit(x,censored.astype(float))
        stage_tree=None
        known=completed & np.asarray([r.get('stage_costs_known',False) for r in metadata],dtype=bool)
        if known.sum()>=2:
            from sklearn.tree import DecisionTreeRegressor
            target=np.asarray([r.get('stage_costs',[0.]*4) for r in metadata],dtype=float)
            tree=DecisionTreeRegressor(max_depth=5,min_samples_leaf=min(10,max(1,int(known.sum())//8)),random_state=self.base.seed)
            tree.fit(x[known],target[known]);t=tree.tree_
            stage_tree=dict(left=t.children_left.tolist(),right=t.children_right.tolist(),
                features=t.feature.tolist(),threshold=t.threshold.tolist(),values=t.value[:,:,0].tolist())
        self.base.quality_model=quality_model;self.base.latency_model=latency_model
        self.timeout_model=timeout_model;self.base.fast_inference=None;self.stage_tree=stage_tree
        self.fused_inference = None
        self.base_feature_count = np.asarray(features).shape[1]
        self.training_audit = dict(observations=len(x),quality_rows=int(eligible.sum()),
            masked_quality_rows=int((~eligible).sum()),completed_execution_rows=int(completed.sum()),
            censored_rows=int(censored.sum()),latency_target=self.latency_target,
            censored_treatment='timeout risk labels; not exact completion targets',
            training_devices=list(self.training_devices),test_labels_used=False)
        return self

    def predict_main(self,features,facts):
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        # These pre-execution facts are shared by every candidate of a request.
        # Compute their features once, preserving the original feature schema.
        x = np.column_stack((matrix, np.broadcast_to(fact_vector(facts), (len(matrix), len(FACTS)+3))))
        prediction, risk = self._predict_heads(x)
        stages=None
        if self.stage_tree:
            tree=self.stage_tree;indices=np.zeros(len(x),dtype=int)
            left=np.asarray(tree['left']);right=np.asarray(tree['right']);cols=np.asarray(tree['features'])
            bounds=np.asarray(tree['threshold'])
            for _ in range(6):
                active=left[indices]>=0
                if not active.any():break
                rows=np.flatnonzero(active);nodes=indices[rows]
                indices[rows]=np.where(x[rows,cols[nodes]].astype(np.float32)<=bounds[nodes],left[nodes],right[nodes])
            stages=np.asarray(tree['values'])[indices]
        return SimpleNamespace(quality=prediction.quality,log_latency=prediction.log_latency,execution_stages=stages,
            timeout_probability=SimpleNamespace(mean=np.clip(risk.mean,0.,1.),uncertainty=risk.uncertainty))

    def _predict_heads(self, x):
        if self.fused_inference is None or not self.use_fast_inference:
            return self.base.predict(x), self.timeout_model.predict(x)
        values = self.fused_inference.predict(x)
        q, c = len(self.quality_model.models), len(self.latency_model.models)
        def aggregate(rows, model):
            return Prediction(rows.mean(axis=0),
                np.maximum(rows.std(axis=0, ddof=1), model.uncertainty_floor))
        quality = aggregate(values[:q], self.quality_model)
        latency = aggregate(values[q:q+c], self.latency_model)
        risk = aggregate(values[q+c:], self.timeout_model)
        return ConstrainedPrediction(
            Prediction(np.clip(quality.mean, 0., 1.), quality.uncertainty),
            Prediction(np.maximum(latency.mean, 0.), latency.uncertainty)), risk

    def prepare_fused_inference(self, directory):
        """Opt-in: evaluate unchanged quality/cost/risk trees in one native call."""
        from .frozen_forest import FrozenForest
        native = FrozenForest(self.quality_model.models + self.latency_model.models + self.timeout_model.models,
                              self.base.feature_count_, directory)
        count = min(native.comparisons, 256)
        probes = np.random.default_rng(42).normal(size=(65 + 3*count, self.base.feature_count_))
        probes[0] = 0.
        for row, test in enumerate(np.linspace(0, native.comparisons-1, count, dtype=int)):
            feature, border = native.arrays[0][test], native.arrays[1][test]
            probes[65+3*row:68+3*row, feature] = [np.nextafter(border, np.float32(-np.inf)), border,
                                                              np.nextafter(border, np.float32(np.inf))]
        previous, flag = self.fused_inference, self.use_fast_inference
        try:
            self.use_fast_inference = False
            reference, reference_risk = self._predict_heads(probes)
            self.fused_inference, self.use_fast_inference = native, True
            actual, actual_risk = self._predict_heads(probes)
            difference = 0.
            for a, b in ((reference.quality, actual.quality), (reference.log_latency, actual.log_latency),
                         (reference_risk, actual_risk)):
                for statistic in ('mean', 'uncertainty'):
                    first, second = getattr(a, statistic), getattr(b, statistic)
                    difference = max(difference, float(np.max(np.abs(first-second))))
                    if not np.allclose(first, second, rtol=1e-12, atol=1e-12):
                        raise ValueError('fused main inference failed reference-equivalence guard')
        except BaseException:
            self.fused_inference = previous
            raise
        finally:
            self.use_fast_inference = flag
        native.audit.update(heads=['quality', 'execution_cost', 'timeout_risk'],
            main_cost_training=self.training_audit,
            startup_validation=dict(probes=len(probes), max_absolute_error=difference,
                                    labels_used=False, tolerance=1e-12))
        (Path(directory)/'manifest.json').write_text(json.dumps(native.audit, indent=2)+'\n')
        return native.audit

    def prepare_fast_inference(self,directory):
        self.fused_inference = None
        audit = self.base.prepare_fast_inference(directory)
        audit['main_cost_training'] = self.training_audit
        (Path(directory)/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
        return audit

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        self.base.save(directory/'base')
        self.timeout_model.save(directory/'timeout')
        (directory/'main-cost.json').write_text(json.dumps(dict(version=self.VERSION,
            base_feature_count=self.base_feature_count,training_audit=self.training_audit,stage_tree=self.stage_tree),indent=2)+'\n')

    @classmethod
    def load(cls,directory):
        directory=Path(directory);meta=json.loads((directory/'main-cost.json').read_text())
        if meta['version']!=cls.VERSION: raise ValueError('unknown main cost model')
        result=cls()
        result.base=ConstrainedCatBoostSurrogate.load(directory/'base')
        result.timeout_model=CatBoostEnsembleSurrogate.load(directory/'timeout')
        result.base_feature_count=meta['base_feature_count'];result.training_audit=meta['training_audit']
        result.stage_tree=meta.get('stage_tree')
        result.training_devices=tuple(str(d) for d in result.training_audit.get('training_devices',()))
        return result
