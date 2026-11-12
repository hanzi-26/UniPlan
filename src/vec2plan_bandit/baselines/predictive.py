from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vec2plan_bandit.baselines.base import BaselineRequest
from vec2plan_bandit.features.action_features import joint_action_features, joint_action_feature_matrix
from vec2plan_bandit.paths.registry import (
    ExecutionConfig, PATH_IDS, PRE, pre_configuration_key,
)
from vec2plan_bandit.policies.max_ucb import sample_path_configurations


EstimatorFactory = Callable[[], Any]


class PredictiveRewardBaseline:
    """Mean-only reward predictor used by the MLP and LightGBM baselines."""

    def __init__(
        self,
        name: str,
        estimator_factory: EstimatorFactory,
        *,
        candidates_per_numeric_path: int = 128,
        seed: int = 42,
        candidate_bank=None,
    ):
        self.name = name
        self._factory = estimator_factory
        self.candidates_per_numeric_path = int(candidates_per_numeric_path)
        self.seed = int(seed)
        self.model: Any | None = None
        from vec2plan_bandit.paths.configuration_bank import load_bank
        self.configuration_bank=load_bank(candidate_bank)

    def fit(self, observations: Sequence[Mapping[str, Any]]) -> "PredictiveRewardBaseline":
        if len(observations) < 2:
            raise ValueError(f"{self.name} requires at least two training observations")
        features = np.vstack([np.asarray(row["features"], dtype=np.float64) for row in observations])
        rewards = np.asarray([float(row["reward"]) for row in observations])
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(rewards)):
            raise ValueError("baseline training data contains non-finite values")
        self.model = self._factory()
        self.model.fit(features, rewards)
        return self

    def select(self, request: BaselineRequest) -> ExecutionConfig:
        if self.model is None:
            raise RuntimeError(f"{self.name} baseline is not fit")
        candidates: list[ExecutionConfig] = []
        for offset, path_id in enumerate(PATH_IDS):
            if request.available_paths.get(path_id, False):
                sampled = self.configuration_bank[path_id] if self.configuration_bank is not None else sample_path_configurations(
                    path_id,
                    self.candidates_per_numeric_path,
                    seed=self.seed + 1009 * offset,
                )
                if path_id == PRE and request.available_pre_configurations is not None:
                    from vec2plan_bandit.paths.registry import include_optimizer_pre
                    if self.configuration_bank is None:
                        sampled = include_optimizer_pre(sampled, request.available_pre_configurations)
                    sampled = tuple(
                        config for config in sampled
                        if pre_configuration_key(config.params)
                        in request.available_pre_configurations
                    )
                candidates.extend(sampled)
        if not candidates:
            raise RuntimeError("no operational path is available")
        features = joint_action_feature_matrix(request.query_context, candidates,
            latency_cap_ms=request.latency_cap_ms, remaining_budget_ms=request.remaining_budget_ms)
        predictions = np.asarray(self.model.predict(features), dtype=np.float64).reshape(-1)
        winner = max(range(len(candidates)), key=lambda index: (predictions[index], candidates[index].config_id))
        return candidates[winner]


def mlp_factory(config: Mapping[str, Any]) -> PredictiveRewardBaseline:
    def estimator() -> Any:
        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError as error:
            raise RuntimeError("MLP baseline requires scikit-learn") from error
        return MLPRegressor(
            hidden_layer_sizes=tuple(config.get("hidden_layer_sizes", [64, 32])),
            alpha=float(config.get("alpha", 1e-4)),
            learning_rate_init=float(config.get("learning_rate_init", 1e-3)),
            max_iter=int(config.get("max_iter", 500)),
            early_stopping=True,
            random_state=int(config.get("seed", 42)),
        )

    return PredictiveRewardBaseline(
        "mlp",
        estimator,
        candidates_per_numeric_path=int(config.get("candidates_per_numeric_path", 128)),
        seed=int(config.get("seed", 42)),
        candidate_bank=config.get('candidate_bank'),
    )


def lightgbm_factory(config: Mapping[str, Any]) -> PredictiveRewardBaseline:
    def estimator() -> Any:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise RuntimeError("LightGBM baseline requires the lightgbm package") from error
        return LGBMRegressor(
            n_estimators=int(config.get("n_estimators", 300)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            num_leaves=int(config.get("num_leaves", 31)),
            max_depth=int(config.get("max_depth", -1)),
            random_state=int(config.get("seed", 42)),
            n_jobs=1,
            verbosity=-1,
        )

    return PredictiveRewardBaseline(
        "lightgbm",
        estimator,
        candidates_per_numeric_path=int(config.get("candidates_per_numeric_path", 128)),
        seed=int(config.get("seed", 42)),
        candidate_bank=config.get('candidate_bank'),
    )
