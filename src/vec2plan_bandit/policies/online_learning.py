"""Feedback-driven CatBoost refits with deterministic replay and checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from vec2plan_bandit.policies.max_ucb import MaxUCB
from vec2plan_bandit.surrogates.catboost_ensemble import (
    CatBoostEnsembleSurrogate,
    ConstrainedCatBoostSurrogate,
)


class OnlineLearner:
    VERSION = "catboost-constrained-maxucb-online-v2"

    def __init__(self, policy: MaxUCB, observations: Sequence[Mapping[str, Any]], *, refit_every: int = 8):
        if type(refit_every) is not int or refit_every < 1:
            raise ValueError("refit_every must be a positive integer")
        if not observations:
            raise ValueError("bootstrap observations are required")
        self.policy = policy
        self.refit_every = refit_every
        self.features, self.quality_targets = CatBoostEnsembleSurrogate._training_arrays(
            np.vstack([row["features"] for row in observations]),
            np.asarray([row["quality_target"] for row in observations]),
        )
        self.e2e_latencies_ms = np.asarray(
            [row["e2e_latency_ms"] for row in observations], dtype=np.float64
        )
        from vec2plan_bandit.surrogates.main_cost import feedback_fields
        self.feedback_metadata = [row.get('feedback_metadata',feedback_fields(row)) for row in observations]
        if (len(self.e2e_latencies_ms) != len(self.features)
                or not np.all(np.isfinite(self.e2e_latencies_ms))
                or np.any(self.e2e_latencies_ms < 0)):
            raise ValueError("bootstrap E2E latencies must be finite and nonnegative")
        if policy.round_index != len(observations):
            raise ValueError("bootstrap counts must match observed executions")
        self.feedback_ids: list[str] = []
        self._seen: set[str] = set()
        self.refit_count = 0

    def observe(
        self,
        *,
        observation_id: str,
        path_id: str,
        features: Sequence[float],
        quality_target: float,
        e2e_latency_ms: float,
        feedback_metadata: Mapping[str,Any] | None = None,
    ) -> bool:
        """Append one quality/latency label; return False for replayed feedback."""
        if not observation_id:
            raise ValueError("feedback requires a stable observation_id")
        if observation_id in self._seen:
            return False
        if path_id not in self.policy.path_counts:
            raise ValueError("unknown feedback path")
        feature_row = np.asarray(features, dtype=float)
        if feature_row.shape != (self.features.shape[1],) or not np.all(np.isfinite(feature_row)):
            raise ValueError("invalid feedback features")
        if not np.isfinite(quality_target) or quality_target not in (0, 1):
            raise ValueError("quality_target must be binary")
        if not np.isfinite(e2e_latency_ms) or e2e_latency_ms < 0:
            raise ValueError("e2e_latency_ms must be finite and nonnegative")
        new_features = np.vstack([self.features, feature_row])
        new_quality = np.append(self.quality_targets, quality_target)
        new_latencies = np.append(self.e2e_latencies_ms, e2e_latency_ms)
        new_metadata = self.feedback_metadata+[dict(feedback_metadata or {})]
        refit = (len(self.feedback_ids) + 1) % self.refit_every == 0
        # fit() builds all models before replacing the ensemble. If a refit
        # fails, counters/buffer remain unchanged and the journal can be replayed.
        if refit:
            self._fit(new_features,new_quality,new_latencies,new_metadata)
        self.features = new_features
        self.quality_targets = new_quality
        self.e2e_latencies_ms = new_latencies
        self.feedback_metadata = new_metadata
        self.policy.update(path_id)
        self.feedback_ids.append(observation_id)
        self._seen.add(observation_id)
        self.refit_count += int(refit)
        return True

    def _fit(self,features,quality,latencies,metadata):
        fitter=getattr(self.policy.surrogate,'fit_feedback',None)
        if fitter is not None:
            fitter(features,quality,latencies,metadata)
        else:
            self.policy.surrogate.fit(features,quality,latencies)

    def refit(self):
        self._fit(self.features,self.quality_targets,self.e2e_latencies_ms,self.feedback_metadata)

    def save(self, directory: Path) -> None:
        """Publish a complete generation atomically; interrupted saves keep the old one."""
        directory.mkdir(parents=True, exist_ok=True)
        generation = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=directory))
        self.policy.surrogate.save(generation / "surrogate")
        np.savez_compressed(
            generation / "observations.npz",
            features=self.features,
            quality_targets=self.quality_targets,
            e2e_latencies_ms=self.e2e_latencies_ms,
        )
        state = {
            "version": self.VERSION, "policy": self.policy.state_dict(),
            "refit_every": self.refit_every, "refit_count": self.refit_count,
            "feedback_ids": self.feedback_ids,
            "feedback_metadata": self.feedback_metadata,
        }
        (generation / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary = directory / "current.tmp"
        temporary.write_text(json.dumps({"generation": generation.name}) + "\n", encoding="utf-8")
        os.replace(temporary, directory / "current.json")

    @classmethod
    def load(cls, directory: Path) -> "OnlineLearner":
        name = json.loads((directory / "current.json").read_text())["generation"]
        if Path(name).name != name or not name.startswith("checkpoint-"):
            raise ValueError("invalid checkpoint generation")
        generation = directory / name
        state = json.loads((generation / "state.json").read_text())
        if state["version"] != cls.VERSION:
            raise ValueError("unsupported learner checkpoint")
        surrogate = ConstrainedCatBoostSurrogate.load(generation / "surrogate")
        config = state["policy"]
        policy = MaxUCB(surrogate, **{key: config[key] for key in MaxUCB.SETTINGS if key in config})
        policy.load_state_dict(config)
        with np.load(generation / "observations.npz", allow_pickle=False) as data:
            rows = [
                {"features": x, "quality_target": q, "e2e_latency_ms": latency}
                for x, q, latency in zip(
                    data["features"], data["quality_targets"], data["e2e_latencies_ms"]
                )
            ]
        learner = cls(policy, rows, refit_every=state["refit_every"])
        if 'feedback_metadata' in state:
            if len(state['feedback_metadata'])!=len(learner.features):
                raise ValueError('feedback metadata length mismatch')
            learner.feedback_metadata=state['feedback_metadata']
        learner.feedback_ids = list(state["feedback_ids"])
        learner._seen = set(learner.feedback_ids)
        learner.refit_count = int(state["refit_count"])
        if len(learner._seen) != len(learner.feedback_ids) or learner.refit_count != len(learner.feedback_ids) // learner.refit_every:
            raise ValueError("inconsistent feedback checkpoint")
        if surrogate.feature_count_ != learner.features.shape[1]:
            raise ValueError("checkpoint feature dimension mismatch")
        return learner
