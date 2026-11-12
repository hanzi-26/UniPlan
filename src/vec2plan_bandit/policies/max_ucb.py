from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import math
from typing import Any, Mapping, Protocol
import uuid
import time

import numpy as np

from vec2plan_bandit.features.action_features import joint_action_features, joint_action_feature_matrix
from vec2plan_bandit.paths.registry import (
    PATH_IDS,
    PRE,
    PRE_JOIN_STRATEGIES,
    REGISTRY,
    ExecutionConfig,
    ParameterSpec,
    make_execution_config,
    pre_configuration_key,
)


class ConstrainedSurrogate(Protocol):
    def predict(self, features: np.ndarray) -> Any: ...


@dataclass(frozen=True)
class MaxUCBDecision:
    decision_id: str
    config: ExecutionConfig
    score: float
    predicted_quality_probability: float
    quality_uncertainty: float
    predicted_e2e_latency_ms: float
    latency_log_uncertainty: float
    quality_lcb: float
    quality_ucb: float
    latency_lcb_ms: float
    latency_ucb_ms: float
    constraints_satisfied: bool
    feasible_candidate_count: int
    best_by_path: dict[str, str]
    evaluated_candidates: int
    round_index: int = 0
    path_counts: dict[str, int] = field(default_factory=dict)
    scores_by_path: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_seeds: dict[str, int] = field(default_factory=dict)
    selected_features: tuple[float, ...] = ()
    policy_version: str = "cash-maxucb-catboost-pre-join-v5"
    fallback_reason: str | None = None
    fallback_quality_tolerance: float = 0.0
    fallback_candidate_count: int = 0
    selection_mode: str = 'full_maxucb'
    selection_evidence: dict[str, Any] = field(default_factory=dict)


def _lhs(rows: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    if dimensions == 0:
        return np.empty((rows, 0), dtype=np.float64)
    result = np.empty((rows, dimensions), dtype=np.float64)
    for column in range(dimensions):
        result[:, column] = (rng.permutation(rows) + rng.random(rows)) / rows
    return result


@lru_cache(maxsize=1)
def _numeric_parameters() -> dict[str, tuple[ParameterSpec, ...]]:
    return {
        spec.path_id: tuple(p for p in spec.parameters if p.kind in {"real", "integer"})
        for spec in REGISTRY.specs
    }


def sample_path_configurations(
    path_id: str,
    count: int,
    *,
    seed: int,
    include_default: bool = True,
) -> tuple[ExecutionConfig, ...]:
    # PRE is independent of RNG. Numeric request/round seeds usually change;
    # caching those designs would add allocations without reuse.
    if path_id != PRE:
        return _sample_path_configurations(path_id, count, seed=seed, include_default=include_default)
    templates = _configuration_templates(path_id, 1, seed=0, include_default=include_default)
    return tuple(ExecutionConfig(cid, action, path, dict(params))
                 for cid, action, path, params in templates)


@lru_cache(maxsize=2)
def _configuration_templates(path_id, count, *, seed, include_default):
    configs = _sample_path_configurations(path_id, count, seed=seed, include_default=include_default)
    return tuple((c.config_id, c.action_id, c.path_id, tuple(c.params.items())) for c in configs)


def _sample_path_configurations(path_id, count, *, seed, include_default=True):
    """Sample a conditional path space without turning it into a fixed grid."""
    spec = REGISTRY.get(path_id)
    if path_id == PRE:
        access_params = (
            {"index_mode": "no_index"},
            *(
                {"index_mode": "index", "index_strategy": strategy}
                for strategy in ("btree", "hash", "bitmap")
            ),
        )
        return tuple(
            make_execution_config(
                PRE,
                {**params, "join_strategy": join_strategy},
            )
            for params in access_params
            for join_strategy in PRE_JOIN_STRATEGIES
        )
    parameters = _numeric_parameters()[path_id]
    if count <= 0 or not parameters:
        raise ValueError(f"positive sample count required for {path_id}")
    rng = np.random.default_rng(seed)
    design = _lhs(count, len(parameters), rng)
    configs: list[ExecutionConfig] = []
    if include_default:
        configs.append(make_execution_config(path_id))
    for row in design:
        values: dict[str, Any] = {}
        for unit, parameter in zip(row, parameters):
            assert parameter.lower is not None and parameter.upper is not None
            # Values remain real in the search space. Integer-only backend
            # parameters are rounded by ParameterSpec at configuration boundary.
            values[parameter.name] = parameter.lower + unit * (parameter.upper - parameter.lower)
        configs.append(make_execution_config(path_id, values))
    unique = {config.config_id: config for config in configs}
    return tuple(unique.values())


def initial_design(points_per_numeric_path: int = 8, *, seed: int = 42) -> tuple[ExecutionConfig, ...]:
    """CASH bootstrap: PRE access/join choices plus LHS points for numeric paths."""
    configs: list[ExecutionConfig] = []
    for action, path_id in enumerate(PATH_IDS):
        configs.extend(
            sample_path_configurations(
                path_id,
                points_per_numeric_path,
                seed=seed + 1009 * action,
                include_default=False,
            )
        )
    return tuple(configs)


@lru_cache(maxsize=64)
def _main_candidate_bank(path_id,count,seed):
    """Main-only frozen bank; no repeated RNG/config hashing per request."""
    return _sample_path_configurations(path_id,count,seed=seed)


class MaxUCB:
    """Quality-constrained CASH optimizer over path and continuous parameters.

    During training, optimistic quality and latency confidence bounds preserve
    MaxUCB exploration. During frozen evaluation, conservative bounds enforce
    the Recall/full-k and E2E deadline constraints before minimizing predicted
    latency. MaxUCB supplies the acquisition rule; CatBoost supplies mu/sigma.
    """

    VERSION = "cash-maxucb-catboost-pre-join-v5"
    LEGACY_VERSION = "cash-maxucb-catboost-pre-join-v4"
    SETTINGS = (
        "beta_quality", "beta_latency", "kappa_quality", "kappa_latency",
        "quality_probability_min", "gamma", "seed", "candidates_per_numeric_path",
        "fallback_quality_tolerance",
    )

    def __init__(
        self,
        surrogate: ConstrainedSurrogate,
        *,
        beta_quality: float = 1.0,
        beta_latency: float = 1.0,
        kappa_quality: float = 1.0,
        kappa_latency: float = 1.0,
        quality_probability_min: float = 0.8,
        gamma: float = 1.0,
        candidates_per_numeric_path: int = 128,
        seed: int = 42,
        fallback_quality_tolerance: float = 0.0,
    ):
        coefficients = (beta_quality, beta_latency, kappa_quality, kappa_latency, gamma)
        if (any(not math.isfinite(value) or value < 0 for value in coefficients)
                or not math.isfinite(quality_probability_min)
                or not 0 <= quality_probability_min <= 1
                or not math.isfinite(fallback_quality_tolerance)
                or not 0 <= fallback_quality_tolerance <= 1
                or candidates_per_numeric_path <= 0):
            raise ValueError("invalid MaxUCB configuration")
        self.surrogate = surrogate
        self.beta_quality = float(beta_quality)
        self.beta_latency = float(beta_latency)
        self.kappa_quality = float(kappa_quality)
        self.kappa_latency = float(kappa_latency)
        self.quality_probability_min = float(quality_probability_min)
        self.gamma = float(gamma)
        self.candidates_per_numeric_path = int(candidates_per_numeric_path)
        self.seed = int(seed)
        self.fallback_quality_tolerance = float(fallback_quality_tolerance)
        self.path_counts = {path: 0 for path in PATH_IDS}

    @property
    def round_index(self) -> int:
        """Number of completed labeled executions, including bootstrap rows."""
        return sum(self.path_counts.values())

    def update(self, path_id: str) -> None:
        """Count a completed execution (including a failed execution) once."""
        if path_id not in self.path_counts:
            raise ValueError(f"unknown feedback path: {path_id}")
        self.path_counts[path_id] += 1

    def path_bonus(self, path_id: str) -> float:
        return self.gamma * math.sqrt(math.log1p(self.round_index) / (self.path_counts[path_id] + 1))

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "beta_quality": self.beta_quality,
            "beta_latency": self.beta_latency,
            "kappa_quality": self.kappa_quality,
            "kappa_latency": self.kappa_latency,
            "quality_probability_min": self.quality_probability_min,
            "fallback_quality_tolerance": self.fallback_quality_tolerance,
            "gamma": self.gamma,
            "seed": self.seed, "candidates_per_numeric_path": self.candidates_per_numeric_path,
            "round_index": self.round_index, "path_counts": dict(self.path_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        state = dict(state)
        if state.get("version") == self.LEGACY_VERSION:
            state["version"] = self.VERSION
            state.setdefault("fallback_quality_tolerance", 0.0)
        expected = self.state_dict()
        for key in ("version", *self.SETTINGS):
            if state.get(key) != expected[key]:
                raise ValueError(f"policy checkpoint mismatch: {key}")
        counts = state.get("path_counts", {})
        if set(counts) != set(PATH_IDS) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("invalid path counts")
        if state.get("round_index") != sum(counts.values()):
            raise ValueError("inconsistent round index")
        self.path_counts = dict(counts)

    def _seed_for(self, request_key: str, path_id: str) -> int:
        digest = hashlib.sha256(
            f"{self.seed}:{self.round_index}:{request_key}:{path_id}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def select(
        self,
        query_context: np.ndarray,
        *,
        request_key: str,
        latency_cap_ms: float,
        remaining_budget_ms: float,
        available: Mapping[str, bool] | None = None,
        available_pre_configurations: frozenset[str] | None = None,
        exploration: bool = True,
        path_priors: Mapping[str, float] | None = None,
        path_prior_weight: float = 0.0,
        request_facts: Mapping[str, Any] | None = None,
        request_facts_ms: float = 0.0,
        native_pg_available: bool = False,
        exact_pg_available: bool = False,
        sql_candidates_available: bool = False,
        parameter_profile_keys: tuple[str, ...] = (),
        _full_search: bool = False,
        _skip_lightweight: bool = False,
    ) -> MaxUCBDecision:
        self.last_timings={}
        stage_start=time.perf_counter_ns()
        selection_start = stage_start
        initial_remaining_budget_ms = remaining_budget_ms
        readiness = {path_id: True for path_id in PATH_IDS}
        if available is not None:
            unknown = set(available) - set(PATH_IDS)
            if unknown:
                raise ValueError(f"unknown paths in availability: {sorted(unknown)}")
            readiness.update({key: bool(value) for key, value in available.items()})
        from vec2plan_bandit.execution.timing import record_request_audit
        lightweight = getattr(self, 'main_lightweight', None)
        self.last_selection_evidence = {'mode':'full_maxucb','reason':'lightweight_disabled'}
        if request_facts is not None:
            record_request_audit(main_routing_facts=dict(request_facts))
        # Exploration and path-prior experiments retain the complete algorithm.
        if (lightweight is not None and not exploration and path_priors is None and not _skip_lightweight
                and path_prior_weight == 0. and getattr(self,'use_main_lightweight',True)):
            from vec2plan_bandit.policies.main_lightweight import PG_NATIVE
            gate_start = time.perf_counter_ns()
            proposal = lightweight.propose(request_facts,available={**readiness,
                **{p: sql_candidates_available and getattr(self, 'main_sql_candidates_early', False)
                   for p in ('FAISS_SQL', 'POST_SQL', 'PG_SQL', 'EXACT_SQL')},
                PG_NATIVE: native_pg_available and getattr(self,'main_native_pg_early',False),
                'PG_EXACT_SQL': exact_pg_available and getattr(self,'main_exact_pg_early',False)},
                available_pre_configurations=available_pre_configurations,
                recall_threshold=self.main_recall_threshold,probability_min=self.quality_probability_min,
                latency_cap_ms=latency_cap_ms,remaining_budget_ms=remaining_budget_ms)
            elapsed = (time.perf_counter_ns()-gate_start)/1e6
            reason = proposal.reason
            if proposal.config is not None:
                selected = (() if (proposal.config.path_id == PG_NATIVE
                    or getattr(lightweight, 'compact_early_features', False)) else
                    joint_action_features(query_context,proposal.config,
                        latency_cap_ms=latency_cap_ms,remaining_budget_ms=remaining_budget_ms))
                elapsed = (time.perf_counter_ns()-gate_start)/1e6
                if not math.isfinite(request_facts_ms) or request_facts_ms < 0:
                    raise ValueError('invalid main routing facts duration')
                if elapsed + request_facts_ms > lightweight.overhead_budget_ms:
                    reason = 'routing_allowance_exceeded'
                elif proposal.evidence['e2e_upper_ms'] > remaining_budget_ms-elapsed:
                    reason = 'budget_consumed_during_routing'
                else:
                    evidence = dict(proposal.evidence)
                    self.last_timings = {'lightweight_lookup_ms':elapsed,
                                         'routing_facts_ms':request_facts_ms}
                    self.last_selection_evidence = dict(mode='lightweight',reason=reason,
                        quality_evidence='paired train/dev semantic groups; empirical',
                        train_groups=evidence['train_groups'],dev_groups=evidence['dev_groups'])
                    record_request_audit(policy_stages_ms=dict(self.last_timings),
                        main_selection=dict(self.last_selection_evidence),main_inference_engine='main-lightweight-tree-v1')
                    mean, upper = evidence['e2e_mean_ms'], evidence['e2e_upper_ms']
                    quality = evidence['quality_mean']; lower = evidence['quality_lower_bound']
                    empirical = evidence.get('acceptance') == 'empirical_best_effort'
                    constraint_ok = lower >= self.quality_probability_min if empirical else True
                    if empirical:
                        self.last_selection_evidence.update(mode='empirical_early_fallback',
                            acceptance='empirical_best_effort',quality_lower_bound=lower,
                            probability_min=self.quality_probability_min,constraints_satisfied=constraint_ok,
                            fallback_reason='empirical_early_best_effort')
                        record_request_audit(main_selection=dict(self.last_selection_evidence),
                            main_inference_engine=getattr(lightweight, 'inference_engine', 'main-empirical-early-v1'))
                    return MaxUCBDecision(decision_id=str(uuid.uuid4()),config=proposal.config,
                        score=-math.log1p(mean),predicted_quality_probability=quality,
                        quality_uncertainty=quality-lower,predicted_e2e_latency_ms=mean,
                        latency_log_uncertainty=0.,quality_lcb=lower,quality_ucb=quality,
                        latency_lcb_ms=mean,latency_ucb_ms=upper,constraints_satisfied=constraint_ok,
                        feasible_candidate_count=int(constraint_ok),best_by_path={proposal.config.path_id:proposal.config.config_id},
                        evaluated_candidates=1,round_index=self.round_index,path_counts=dict(self.path_counts),
                        selected_features=tuple(float(v) for v in selected),
                        selection_mode='empirical_early_fallback' if empirical else 'lightweight',
                        fallback_reason='empirical_early_best_effort' if empirical else None,
                        selection_evidence=dict(self.last_selection_evidence))
            elapsed = (time.perf_counter_ns()-gate_start)/1e6
            remaining_budget_ms = max(0.,remaining_budget_ms-elapsed)
            self.last_timings['lightweight_lookup_ms'] = elapsed
            self.last_selection_evidence['reason'] = reason
        record_request_audit(main_selection=dict(self.last_selection_evidence))
        stage_start=time.perf_counter_ns()
        candidates: list[ExecutionConfig] = []
        search = getattr(self,'main_search',{}) if not exploration and path_priors is None else {}
        staged = bool(search.get('enabled'))
        coarse = staged and not _full_search
        full_candidate_count = 0
        configuration_bank = getattr(self,'main_configuration_bank',None)
        for path_id in PATH_IDS:
            if readiness[path_id]:
                if configuration_bank is not None:
                    sampled=configuration_bank[path_id]
                    full_path_count=len(sampled)
                elif staged:
                    sampled=_main_candidate_bank(path_id,self.candidates_per_numeric_path,
                        self.seed+1009*PATH_IDS.index(path_id))
                    full_path_count = len(sampled)
                    if coarse and path_id!=PRE:
                        sampled=sampled[:min(len(sampled),int(search.get('initial_candidates',32))+1)]
                else:
                    sampled = sample_path_configurations(
                        path_id,
                        self.candidates_per_numeric_path,
                        seed=self._seed_for(request_key, path_id),
                    )
                    full_path_count = len(sampled)
                if path_id == PRE and available_pre_configurations is not None:
                    from vec2plan_bandit.paths.registry import include_optimizer_pre
                    if configuration_bank is None:
                        sampled = include_optimizer_pre(sampled, available_pre_configurations)
                    sampled = tuple(
                        config for config in sampled
                        if pre_configuration_key(config.params) in available_pre_configurations
                    )
                if (path_id == PRE and not exploration and getattr(self,'deduplicate_pre_joins',False)
                        and request_facts and request_facts.get('pre_join_settings_irrelevant') is True):
                    access_key = lambda c: tuple(sorted((k,v) for k,v in c.params.items() if k!='join_strategy'))
                    canonical = {access_key(c) for c in sampled if c.params['join_strategy']=='all'}
                    sampled = tuple(c for c in sampled if c.params['join_strategy']=='all' or access_key(c) not in canonical)
                full_candidate_count += len(sampled) if path_id == PRE else full_path_count
                candidates.extend(sampled)
        if configuration_bank is not None:
            coarse=False  # All calibrated configurations are cheap to enumerate.
        tuned=getattr(self,'main_parameter_profiles',None)
        profile_evidence=None
        if tuned is not None and not exploration:
            from .parameter_profiles import admitted_candidates
            profiled,profile_evidence=admitted_candidates(tuned,parameter_profile_keys)
            candidates=[c for c in profiled if readiness[c.path_id]
                and (c.path_id!=PRE or available_pre_configurations is None or pre_configuration_key(c.params) in available_pre_configurations)]
            coarse=False
        if not candidates:
            raise RuntimeError("no operational path is available")
        if not math.isfinite(path_prior_weight) or path_prior_weight < 0:
            raise ValueError("path_prior_weight must be finite and nonnegative")
        priors = {path_id: 1.0 for path_id in PATH_IDS}
        if path_priors is not None:
            unknown = set(path_priors) - set(PATH_IDS)
            if unknown:
                raise ValueError(f"unknown paths in path_priors: {sorted(unknown)}")
            supplied = {
                path: float(path_priors.get(path, 0.0))
                for path in PATH_IDS if readiness[path]
            }
            if any(not math.isfinite(value) or value < 0 for value in supplied.values()):
                raise ValueError("path priors must be finite and nonnegative")
            total = sum(supplied.values())
            if total <= 0:
                raise ValueError("path priors must assign positive mass to an available path")
            priors.update({path: value / total for path, value in supplied.items()})
        self.last_timings['candidate_generation_ms']=(time.perf_counter_ns()-stage_start)/1e6
        stage_start=time.perf_counter_ns()
        if getattr(self, 'use_batched_action_features', True):
            features = joint_action_feature_matrix(query_context, candidates,
                latency_cap_ms=latency_cap_ms, remaining_budget_ms=remaining_budget_ms)
        else:
            features = np.vstack([
                joint_action_features(
                    query_context,
                    config,
                    latency_cap_ms=latency_cap_ms,
                    remaining_budget_ms=remaining_budget_ms,
                )
                for config in candidates
            ])
        self.last_timings['feature_matrix_ms']=(time.perf_counter_ns()-stage_start)/1e6
        stage_start=time.perf_counter_ns()
        predictor=getattr(self.surrogate,'predict_main',None)
        prediction = predictor(features,request_facts) if predictor is not None else self.surrogate.predict(features)
        self.last_timings['model_prediction_ms']=(time.perf_counter_ns()-stage_start)/1e6
        stage_start=time.perf_counter_ns()
        quality_means = np.asarray(prediction.quality.mean, dtype=np.float64)
        quality_sigma = np.asarray(prediction.quality.uncertainty, dtype=np.float64)
        latency_means = np.asarray(prediction.log_latency.mean, dtype=np.float64)
        latency_sigma = np.asarray(prediction.log_latency.uncertainty, dtype=np.float64)
        arrays = (quality_means, quality_sigma, latency_means, latency_sigma)
        if any(values.shape != (len(candidates),) for values in arrays):
            raise ValueError("surrogate output shape mismatch")
        if (any(not np.all(np.isfinite(values)) for values in arrays)
                or np.any(quality_sigma < 0) or np.any(latency_sigma < 0)):
            raise ValueError("surrogate must return finite means and nonnegative uncertainty")

        quality_ucb = np.clip(
            quality_means + (self.beta_quality * quality_sigma if exploration else 0.0),
            0.0, 1.0,
        )
        quality_lcb = np.clip(
            quality_means - (self.kappa_quality * quality_sigma if not exploration else 0.0),
            0.0, 1.0,
        )
        latency_lcb_log = np.maximum(
            0.0,
            latency_means - (self.beta_latency * latency_sigma if exploration else 0.0),
        )
        latency_ucb_log = np.maximum(
            0.0,
            latency_means + (self.kappa_latency * latency_sigma if not exploration else 0.0),
        )
        predicted_latency_ms = np.expm1(np.maximum(latency_means, 0.0))
        latency_lcb_ms = np.expm1(latency_lcb_log)
        latency_ucb_ms = np.expm1(latency_ucb_log)
        correction=getattr(self,'main_execution_cost_factors',{})
        if correction:
            factors=np.asarray([correction.get(c.config_id,1.) for c in candidates])
            if not np.all(np.isfinite(factors)) or np.any(factors<1.):
                raise ValueError('invalid trained execution cost correction')
            predicted_latency_ms*=factors;latency_lcb_ms*=factors;latency_ucb_ms*=factors
            latency_means=np.log1p(predicted_latency_ms)
            latency_lcb_log=np.log1p(latency_lcb_ms);latency_ucb_log=np.log1p(latency_ucb_ms)
        execution_cost_mode = getattr(self.surrogate,'latency_target',None)=='execution_ms'
        elapsed_selection = (time.perf_counter_ns()-selection_start)/1e6
        prefix_ms = max(0.,latency_cap_ms-initial_remaining_budget_ms)+elapsed_selection
        timeout_prediction=getattr(prediction,'timeout_probability',None)
        timeout_ucb = (np.clip(timeout_prediction.mean+self.kappa_quality*timeout_prediction.uncertainty,0.,1.)
                       if timeout_prediction is not None else np.zeros(len(candidates)))
        if execution_cost_mode:
            predicted_latency_ms += prefix_ms
            latency_lcb_ms += prefix_ms
            latency_ucb_ms += prefix_ms
        path_bonuses = np.asarray([self.path_bonus(c.path_id) if exploration else 0.0 for c in candidates])
        prior_adjustments = np.asarray([
            path_prior_weight * math.log(max(priors[c.path_id], 1e-12))
            for c in candidates
        ])
        if exploration:
            feasible = (quality_ucb >= self.quality_probability_min) & (latency_lcb_ms <= latency_cap_ms)
            scores = -latency_lcb_log + path_bonuses + prior_adjustments
        else:
            feasible = (quality_lcb >= self.quality_probability_min) & (latency_ucb_ms <= latency_cap_ms)
            if timeout_prediction is not None:
                feasible &= timeout_ucb <= 1.-self.quality_probability_min
            scores = -latency_means + prior_adjustments

        any_feasible = bool(np.any(feasible))
        fallback_reason = None
        fallback_pool = np.zeros(len(candidates), dtype=bool)
        recovery = getattr(self, 'main_constraint_recovery', None)
        if not any_feasible and not exploration and recovery is not None:
            from .constraint_recovery import recovery_pool
            fallback_pool, evidence = recovery_pool(candidates, quality_means, quality_sigma,
                latency_ucb_ms, timeout_ucb, probability_min=self.quality_probability_min,
                latency_cap_ms=latency_cap_ms, options=recovery)
            fallback_reason = evidence['reason']
            self.last_selection_evidence.update(constraint_recovery=evidence)
        # Model probability bounds remain untouched. This explicit experimental
        # objective ranks only configurations admitted by train/dev quality
        # evidence; it must never be reported as a probability-one certificate.
        if (fallback_reason is None and not any_feasible and not exploration and profile_evidence is not None
                and profile_evidence['evidence']=='zero_observed_train_dev_failures'
                and getattr(self,'main_profile_cost_ranking',False)):
            safe=latency_ucb_ms<=latency_cap_ms
            fallback_pool=safe if np.any(safe) else np.ones(len(candidates),dtype=bool)
            fallback_reason='admitted_profiles_latency_ucb'
            self.last_selection_evidence.update(parameter_admission=profile_evidence,
                objective='latency_ucb_after_admission',predicted_constraints_satisfied=False)
        # Keep the historical rule for exploration and explicitly configured
        # controls. Frozen deployment uses a quality band so regression noise
        # cannot make a seconds-long exact PRE beat an equivalent fast plan.
        # At probability=1 there is no permitted quality risk budget. A fast
        # candidate within a tolerance of the best estimate is still weaker;
        # retain quality-first ranking when the prediction is infeasible.
        if (fallback_reason is None and not any_feasible and not exploration and self.quality_probability_min < 1.
                and self.fallback_quality_tolerance > 0):
            time_safe = latency_ucb_ms <= latency_cap_ms
            if timeout_prediction is not None:
                time_safe &= timeout_ucb <= 1.-self.quality_probability_min
            pool = time_safe if np.any(time_safe) else np.ones(len(candidates), dtype=bool)
            best_quality = np.max(quality_lcb[pool])
            fallback_pool = pool & (quality_lcb >= best_quality - self.fallback_quality_tolerance)
            fallback_reason = "quality_band_within_deadline" if np.any(time_safe) else "no_predicted_deadline_candidate"

        def rank(index: int) -> tuple[float, ...]:
            # If no candidate passes both constraints, preserve service by
            # choosing the strongest conservative/optimistic quality estimate,
            # then the lower latency bound. This fallback is logged explicitly.
            if any_feasible:
                return (float(feasible[index]), float(scores[index]), candidates[index].config_id)
            if fallback_reason is not None:
                return (
                    float(fallback_pool[index]),
                    -float(latency_ucb_log[index]),
                    float(prior_adjustments[index]),
                    float(quality_lcb[index]),
                    candidates[index].config_id,
                )
            quality_bound = quality_ucb[index] if exploration else quality_lcb[index]
            latency_bound = latency_lcb_log[index] if exploration else latency_ucb_log[index]
            return (
                float(quality_bound),
                float(prior_adjustments[index]),
                -float(latency_bound),
                candidates[index].config_id,
            )

        best_by_path: dict[str, int] = {}
        for index, config in enumerate(candidates):
            old = best_by_path.get(config.path_id)
            if old is None or rank(index) > rank(old):
                best_by_path[config.path_id] = index
        winner = max(range(len(candidates)), key=rank)
        self.last_timings['constraint_ranking_ms']=(time.perf_counter_ns()-stage_start)/1e6
        from vec2plan_bandit.execution.timing import record_request_audit
        record_request_audit(policy_stages_ms=dict(self.last_timings),
            main_inference_engine=('main-frozen-forest-v1' if
                getattr(self.surrogate, 'fast_inference', None) is not None and
                getattr(self.surrogate, 'use_fast_inference', True) else 'catboost-reference'))
        decision = MaxUCBDecision(
            decision_id=str(uuid.uuid4()),
            config=candidates[winner],
            score=float(scores[winner]),
            predicted_quality_probability=float(quality_means[winner]),
            quality_uncertainty=float(quality_sigma[winner]),
            predicted_e2e_latency_ms=float(predicted_latency_ms[winner]),
            latency_log_uncertainty=float(latency_sigma[winner]),
            quality_lcb=float(quality_lcb[winner]),
            quality_ucb=float(quality_ucb[winner]),
            latency_lcb_ms=float(latency_lcb_ms[winner]),
            latency_ucb_ms=float(latency_ucb_ms[winner]),
            constraints_satisfied=bool(feasible[winner]),
            feasible_candidate_count=int(np.sum(feasible)),
            best_by_path={path: candidates[index].config_id for path, index in best_by_path.items()},
            evaluated_candidates=len(candidates),
            round_index=self.round_index,
            path_counts=dict(self.path_counts),
            candidate_seeds={p: (self.seed+1009*PATH_IDS.index(p) if staged else self._seed_for(request_key,p)) for p in best_by_path},
            selected_features=tuple(float(v) for v in features[winner]),
            fallback_reason=fallback_reason or (
                ("no_feasible_candidate_at_probability_one" if self.quality_probability_min == 1.
                 and not exploration else "legacy_quality_first") if not any_feasible else None),
            fallback_quality_tolerance=self.fallback_quality_tolerance,
            fallback_candidate_count=int(np.sum(fallback_pool)),
            selection_evidence=dict(self.last_selection_evidence),
            scores_by_path={p: {
                "config_id": candidates[i].config_id, "params": dict(candidates[i].params),
                "quality_mean": float(quality_means[i]),
                "quality_sigma": float(quality_sigma[i]),
                "quality_lcb": float(quality_lcb[i]),
                "quality_ucb": float(quality_ucb[i]),
                "log_latency_mean": float(latency_means[i]),
                "log_latency_sigma": float(latency_sigma[i]),
                "predicted_e2e_latency_ms": float(predicted_latency_ms[i]),
                "latency_lcb_ms": float(latency_lcb_ms[i]),
                "latency_ucb_ms": float(latency_ucb_ms[i]),
                "quality_exploration_bonus": float(self.beta_quality * quality_sigma[i]) if exploration else 0.0,
                "latency_exploration_bonus": float(self.beta_latency * latency_sigma[i]) if exploration else 0.0,
                "path_bonus": float(path_bonuses[i]),
                "path_prior": float(priors[p]),
                "path_prior_adjustment": float(prior_adjustments[i]),
                "constraints_satisfied": bool(feasible[i]),
                "timeout_probability_ucb":float(timeout_ucb[i]),
                "latency_target":"execution_plus_measured_prefix" if execution_cost_mode else "e2e",
                "predicted_execution_stages_ms": (getattr(prediction,'execution_stages',None)[i].tolist()
                    if getattr(prediction,'execution_stages',None) is not None else None),
                "score": float(scores[i]),
            } for p, i in best_by_path.items()},
        )
        if coarse:
            expand = (not decision.constraints_satisfied or
                decision.quality_lcb < min(1.,self.quality_probability_min+float(search.get('quality_margin',.03))))
            controller = getattr(self, 'main_search_budget', None)
            if expand and controller is not None and path_prior_weight == 0.:
                spent = (time.perf_counter_ns()-selection_start)/1e6
                incumbent_execution = (max(0.,decision.predicted_e2e_latency_ms-prefix_ms)
                    if execution_cost_mode else decision.predicted_e2e_latency_ms)
                incumbent_upper = (max(0.,decision.latency_ucb_ms-prefix_ms)
                    if execution_cost_mode else decision.latency_ucb_ms)
                control = controller.expansion(coarse_candidates=len(candidates),
                    full_candidates=full_candidate_count, coarse_compute_ms=spent,
                    incumbent_execution_ms=incumbent_execution,
                    incumbent_upper_ms=incumbent_upper,
                    remaining_ms=max(0.,initial_remaining_budget_ms-spent))
                record_request_audit(main_search_budget=control)
                expand = control['expand']
                if not expand:
                    decision = replace(decision, selection_evidence={**decision.selection_evidence,
                        'search_stop':control, 'constraints_satisfied':decision.constraints_satisfied,
                        'fallback_reason':decision.fallback_reason})
                    record_request_audit(main_search=dict(stage='budget_stopped',
                        evaluated_candidates=len(candidates), reason=control['reason']))
            if expand:
                spent=(time.perf_counter_ns()-selection_start)/1e6
                first_timings=dict(self.last_timings)
                early_gate=dict(self.last_selection_evidence)
                decision=self.select(query_context,request_key=request_key,latency_cap_ms=latency_cap_ms,
                    remaining_budget_ms=max(0.,initial_remaining_budget_ms-spent),available=available,
                    available_pre_configurations=available_pre_configurations,exploration=exploration,
                    path_priors=path_priors,path_prior_weight=path_prior_weight,request_facts=request_facts,
                    native_pg_available=native_pg_available,parameter_profile_keys=parameter_profile_keys,
                    _full_search=True,_skip_lightweight=True)
                self.last_timings={k:first_timings.get(k,0.)+self.last_timings.get(k,0.)
                    for k in set(first_timings)|set(self.last_timings)}
                record_request_audit(policy_stages_ms=dict(self.last_timings),
                    main_search=dict(stage='expanded',coarse_candidates=len(candidates),full_candidates=decision.evaluated_candidates))
                evidence=dict(decision.selection_evidence or {})
                evidence['early_gate']=early_gate
                if evidence.get('reason')=='lightweight_disabled':
                    evidence['reason']=early_gate.get('reason','lightweight_disabled')
                return replace(decision,evaluated_candidates=decision.evaluated_candidates+len(candidates),selection_evidence=evidence)
            if 'search_stop' not in decision.selection_evidence:
                record_request_audit(main_search=dict(stage='coarse_accepted',evaluated_candidates=len(candidates)))
        # This is a decision-time fallback, never a second execution with a new
        # deadline. Unknown scale, quality or remaining budget retains the old
        # four-path decision. Baselines and exploration never enter this gate.
        if (not any_feasible and lightweight is not None and not exploration
                and path_priors is None and path_prior_weight == 0.
                and native_pg_available and getattr(self,'main_native_pg_fallback',False)
                and getattr(self,'use_main_lightweight',True)):
            from vec2plan_bandit.policies.main_lightweight import PG_NATIVE
            fallback_start = time.perf_counter_ns()
            # remaining_budget_ms was already reduced by the early lookup.
            # Subtract the whole selection once from its original budget.
            remaining = max(0., initial_remaining_budget_ms -
                (fallback_start-selection_start)/1e6)
            proposal = lightweight.propose(request_facts,available={PG_NATIVE: True},
                available_pre_configurations=available_pre_configurations,
                recall_threshold=self.main_recall_threshold,probability_min=self.quality_probability_min,
                latency_cap_ms=latency_cap_ms,remaining_budget_ms=remaining,purpose='pg_fallback')
            elapsed = (time.perf_counter_ns()-fallback_start)/1e6
            self.last_timings['pg_fallback_lookup_ms'] = elapsed
            accepted = (proposal.config is not None and elapsed <= lightweight.overhead_budget_ms
                and proposal.evidence['e2e_upper_ms'] <= remaining-elapsed)
            evidence = dict(mode='certified_pg_fallback' if accepted else 'full_maxucb',
                reason=('certified_native_pg' if accepted else
                    'fallback_routing_or_budget_exceeded' if proposal.config is not None else proposal.reason),
                original_config_id=decision.config.config_id,
                original_quality_lcb=decision.quality_lcb,remaining_budget_ms=remaining-elapsed)
            self.last_selection_evidence = evidence
            record_request_audit(main_selection=evidence,policy_stages_ms=dict(self.last_timings))
            if accepted:
                certificate = proposal.evidence
                evidence.update(train_groups=certificate['train_groups'],dev_groups=certificate['dev_groups'],
                    quality_evidence='paired train/dev semantic groups; empirical')
                quality, lower = certificate['quality_mean'], certificate['quality_lower_bound']
                mean, upper = certificate['e2e_mean_ms'], certificate['e2e_upper_ms']
                decision = replace(decision, config=proposal.config,score=-math.log1p(mean),
                    predicted_quality_probability=quality,quality_uncertainty=quality-lower,
                    quality_lcb=lower,quality_ucb=quality,predicted_e2e_latency_ms=mean,
                    latency_log_uncertainty=0.,latency_lcb_ms=mean,latency_ucb_ms=upper,
                    constraints_satisfied=True,feasible_candidate_count=1,
                    evaluated_candidates=decision.evaluated_candidates+1,
                    best_by_path={**decision.best_by_path,PG_NATIVE:'native-pg'},selected_features=(),
                    fallback_reason='certified_native_pg',fallback_candidate_count=1,
                    selection_mode='certified_pg_fallback',selection_evidence=evidence)
            else:
                decision = replace(decision,selection_evidence=evidence)
        return decision
