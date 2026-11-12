from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vec2plan_bandit.baselines.base import BaselineRequest
from vec2plan_bandit.features.action_features import joint_action_features, joint_action_feature_matrix
from vec2plan_bandit.paths.registry import (
    ExecutionConfig, PATH_IDS, REGISTRY, pre_configuration_key,
)
from vec2plan_bandit.policies.max_ucb import sample_path_configurations
from vec2plan_bandit.surrogates.catboost_ensemble import CatBoostEnsembleSurrogate


PATH_DESCRIPTIONS = {
    "PRE": "PG exact filter then cosine top-k.",
    "POST": "PG HNSW then filter.",
    "FAISS": "Local FAISS HNSW then filter.",
    "ACORN": "Qdrant filtered HNSW traversal.",
}
JEV_PROMPT_VERSION = "jev-path-routing-v3-compact"
JEV_FAILURE_POLICIES = ("fail_closed", "no_prior")


@dataclass(frozen=True)
class JevPathDecision:
    chosen_path: str
    probabilities: dict[str, float]
    confidence: float | None
    request_id: str | None
    latency_ms: float
    prompt_version: str = JEV_PROMPT_VERSION
    state_sha256: str | None = None
    source: str = "live"
    effective_timeout_seconds: float | None = None
    resolved_model: str | None = None


class JevRoutingError(RuntimeError):
    """A Jev failure carrying enough structured data for formal audit logs."""

    def __init__(self, message: str, *, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = dict(audit)


class JevPathRouter:
    """One typed Jev judgment per query, restricted to operational paths."""

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        base_url: str | None = None,
        timeout_seconds: float = 1.0,
        recall_threshold: float = 0.9,
        max_retries: int = 0,
        path_prior_weight: float = 0.25,
        failure_policy: str = "fail_closed",
        prompt_version: str = JEV_PROMPT_VERSION,
        replay_path: str | Path | None = None,
        record_path: str | Path | None = None,
        charge_replay_latency: bool = True,
        client: Any | None = None,
    ):
        if not model or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("invalid Jev model or timeout")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("Jev max_retries must be a nonnegative integer")
        if not math.isfinite(path_prior_weight) or path_prior_weight < 0:
            raise ValueError("Jev path_prior_weight must be finite and nonnegative")
        if failure_policy not in JEV_FAILURE_POLICIES:
            raise ValueError(f"Jev failure_policy must be one of {JEV_FAILURE_POLICIES}")
        if prompt_version != JEV_PROMPT_VERSION:
            raise ValueError(f"unsupported Jev prompt version: {prompt_version}")
        self.model = model
        # The SDK appends /v1/systemone itself. Accept provider URLs that
        # already end in /v1 without producing /v1/v1/systemone.
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url and self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.timeout_seconds = float(timeout_seconds)
        self.recall_threshold = float(recall_threshold)
        self.max_retries = max_retries
        self.path_prior_weight = float(path_prior_weight)
        self.failure_policy = failure_policy
        self.prompt_version = prompt_version
        self.replay_path = Path(replay_path).expanduser() if replay_path else None
        self.record_path = Path(record_path).expanduser() if record_path else None
        if self.replay_path is not None and self.record_path is not None:
            if self.replay_path.resolve() == self.record_path.resolve():
                raise ValueError("Jev replay_path and record_path must differ")
        self.charge_replay_latency = bool(charge_replay_latency)
        self._client = client
        self._owns_client = client is None
        self._client_timeout: float | None = None
        self._replay = self._load_replay(self.replay_path)
        self._recorded_success_states: set[str] = set()
        if self.record_path is not None and self.record_path.is_file():
            for line in self.record_path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    if row.get("probabilities") and row.get("state_sha256"):
                        self._recorded_success_states.add(str(row["state_sha256"]))

    @staticmethod
    def _load_replay(path: Path | None) -> dict[str, dict[str, Any]]:
        if path is None:
            return {}
        if not path.is_file():
            raise FileNotFoundError(f"Jev replay journal not found: {path}")
        rows: dict[str, dict[str, Any]] = {}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("state_sha256", ""))
            if not key:
                raise ValueError(f"invalid Jev replay row {number}: {path}")
            if not row.get("probabilities"):
                continue
            if key in rows:
                if any(rows[key].get(field) != row.get(field) for field in
                       ("chosen_path", "probabilities", "prompt_version", "model")):
                    raise ValueError(f"conflicting Jev replay row {number}: {path}")
                continue
            rows[key] = row
        return rows

    def _get_client(self, effective_timeout: float) -> Any:
        # Reuse TCP/TLS connections despite changing request budgets.
        # Per-call SDK timeout/retry options enforce the current budget.
        if self._client is None:
            try:
                from typesafe_sdk import RetryPolicy, TypeSafeClient
            except ImportError as error:
                raise RuntimeError(
                    "Jev routing requires typesafe-sdk; install vec2plan-bandit[jev]"
                ) from error
            kwargs: dict[str, Any] = {
                "model": self.model,
                # RetryPolicy bounds the complete retry loop; the client
                # timeout also bounds each individual HTTP attempt.
                "timeout": effective_timeout,
                "retry": RetryPolicy(
                    max_retries=self.max_retries,
                    backoff_max=0.2,
                    timeout=effective_timeout,
                ),
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = TypeSafeClient(**kwargs)
            self._client_timeout = effective_timeout
        return self._client

    def _state(self, request: BaselineRequest) -> dict[str, Any]:
        query = request.query
        return {
            "semantic_text": query.semantic_text,
            "embedding_column": query.embedding_column,
            "top_k": query.k,
            "predicates": [dict(value) for value in query.predicates],
            "estimated_selectivity": (request.query_statistics or {}).get("estimated_selectivity"),
            "candidate_count": (request.query_statistics or {}).get("candidate_count"),
            "latency_cap_ms": request.latency_cap_ms,
            "recall_threshold": self.recall_threshold,
            "available_paths": [
                path for path in PATH_IDS if request.available_paths.get(path, False)
            ],
            "available_pre_configurations": sorted(
                request.available_pre_configurations or ()
            ),
            "prompt_version": JEV_PROMPT_VERSION,
        }

    @staticmethod
    def _state_sha256(state: Mapping[str, Any]) -> str:
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _record(self, row: Mapping[str, Any]) -> None:
        if self.record_path is None:
            return
        key = str(row.get("state_sha256", ""))
        successful = bool(row.get("probabilities"))
        if successful and key in self._recorded_success_states:
            return
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        with self.record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if successful and key:
            self._recorded_success_states.add(key)

    def _replayed_decision(
        self,
        row: Mapping[str, Any],
        *,
        available: list[str],
        state_sha256: str,
        effective_timeout: float,
    ) -> JevPathDecision:
        probabilities = {
            str(path): float(value)
            for path, value in dict(row["probabilities"]).items()
        }
        chosen = str(row["chosen_path"])
        if chosen not in available or set(probabilities) != set(available):
            raise ValueError("Jev replay choices disagree with current operational paths")
        original_latency = float(row.get("latency_ms", 0.0))
        if self.charge_replay_latency and original_latency > 0:
            time.sleep(min(original_latency / 1000.0, effective_timeout))
        return JevPathDecision(
            chosen,
            probabilities,
            float(row["confidence"]) if row.get("confidence") is not None else None,
            str(row["request_id"]) if row.get("request_id") else None,
            original_latency if self.charge_replay_latency else 0.0,
            prompt_version=self.prompt_version,
            state_sha256=state_sha256,
            source="replay",
            effective_timeout_seconds=effective_timeout,
            resolved_model=row.get("resolved_model", row.get("model")),
        )

    def route(self, request: BaselineRequest) -> JevPathDecision:
        available = [path for path in PATH_IDS if request.available_paths.get(path, False)]
        if not available:
            raise RuntimeError("no operational path is available")
        effective_timeout = min(
            self.timeout_seconds,
            max(0.0, float(request.remaining_budget_ms)) / 1000.0,
        )
        if effective_timeout <= 0:
            raise JevRoutingError(
                "Jev has no remaining request budget",
                audit={
                    "status": "error", "error_type": "JevBudgetExhausted",
                    "error_message": "no remaining request budget", "latency_ms": 0.0,
                    "model": self.model, "prompt_version": self.prompt_version,
                    "failure_policy": self.failure_policy,
                },
            )
        state = self._state(request)
        state_sha256 = self._state_sha256(state)
        if self.replay_path is not None:
            started = time.perf_counter_ns()
            try:
                if state_sha256 not in self._replay:
                    raise KeyError(f"no Jev replay row for state {state_sha256}")
                return self._replayed_decision(
                    self._replay[state_sha256], available=available,
                    state_sha256=state_sha256, effective_timeout=effective_timeout,
                )
            except Exception as error:
                elapsed = (time.perf_counter_ns() - started) / 1_000_000
                audit = {
                    "status": "error", "error_type": "JevReplayMiss" if isinstance(error, KeyError) else type(error).__name__,
                    "error_message": str(error)[:500], "latency_ms": elapsed,
                    "model": self.model, "prompt_version": self.prompt_version,
                    "state_sha256": state_sha256, "source": "replay",
                    "effective_timeout_seconds": effective_timeout,
                    "failure_policy": self.failure_policy,
                }
                raise JevRoutingError(f"Jev replay failed: {error}", audit=audit) from error
        criteria = {path: PATH_DESCRIPTIONS[path] for path in available}
        started = time.perf_counter_ns()
        try:
            # Injected clients are deliberately SDK-independent (tests and
            # offline adapters consume the plain schema).  The owned live
            # client receives the SDK's typed Choice object.
            try:
                if not self._owns_client:
                    raise ImportError
                from typesafe_sdk import Choice
            except ImportError:
                question: Any = {
                    "type": "choice",
                    "instructions": f"Minimize E2E latency subject to Recall>={self.recall_threshold:g} and full top-k. Use only this pre-execution state.",
                    "criteria": criteria,
                }
            else:
                question = Choice(
                    instructions=f"Minimize E2E latency subject to Recall>={self.recall_threshold:g} and full top-k. Use only this pre-execution state.",
                    criteria=criteria,
                )
            options: dict[str, Any] = {}
            if self._owns_client:
                from typesafe_sdk import RetryPolicy
                options.update(timeout=effective_timeout, retry=RetryPolicy(
                    max_retries=self.max_retries, backoff_max=0.2,
                    timeout=effective_timeout,
                ))
            response = self._get_client(effective_timeout).system_one(
                state=state, questions={"path": question}, **options
            )
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            answers = getattr(response, "answers", None)
            answer = answers["path"] if answers is not None else response.choices["path"]
            chosen = str(answer.choice).upper()
            raw = {str(key).upper(): float(value) for key, value in answer.probabilities.items()}
            if chosen not in available or set(raw) != set(available):
                raise ValueError("Jev returned paths outside the supplied operational choices")
            if any(not math.isfinite(value) or value < 0 for value in raw.values()):
                raise ValueError("Jev returned invalid path probabilities")
            total = sum(raw.values())
            if total <= 0:
                raise ValueError("Jev returned an empty probability distribution")
            probabilities = {path: raw[path] / total for path in available}
            # SDK request_id is a property that raises when a proxy omits
            # x-typesafe-request-id. Missing audit metadata must not discard
            # an otherwise schema-validated prediction. Never fabricate IDs.
            http_response = getattr(response, "raw_http_response", None)
            if http_response is not None:
                request_id = (http_response.headers.get("x-typesafe-request-id")
                              or http_response.headers.get("x-request-id"))
            else:
                request_id = getattr(response, "request_id", None)
            decision = JevPathDecision(
                chosen_path=chosen,
                probabilities=probabilities,
                confidence=float(answer.confidence) if getattr(answer, "confidence", None) is not None else None,
                request_id=str(request_id) if request_id else None,
                latency_ms=elapsed,
                prompt_version=self.prompt_version,
                state_sha256=state_sha256,
                source="live",
                effective_timeout_seconds=effective_timeout,
                resolved_model=getattr(response, "model", None),
            )
            self._record({
                "state_sha256": state_sha256,
                "prompt_version": self.prompt_version,
                "model": self.model,
                "resolved_model": decision.resolved_model,
                "request_id_missing": decision.request_id is None,
                "chosen_path": decision.chosen_path,
                "probabilities": decision.probabilities,
                "confidence": decision.confidence,
                "request_id": decision.request_id,
                "latency_ms": decision.latency_ms,
                "effective_timeout_seconds": effective_timeout,
                "available_paths": available,
                "available_pre_configurations": state["available_pre_configurations"],
            })
            return decision
        except JevRoutingError:
            raise
        except Exception as error:
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            audit = {
                "status": "error", "error_type": type(error).__name__,
                "error_message": str(error)[:500], "latency_ms": elapsed,
                "model": self.model, "prompt_version": self.prompt_version,
                "state_sha256": state_sha256, "source": "live",
                "effective_timeout_seconds": effective_timeout,
                "configured_max_retries": self.max_retries,
                "failure_policy": self.failure_policy,
            }
            self._record(audit)
            raise JevRoutingError(f"Jev routing failed: {error}", audit=audit) from error

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


def jev_router_from_config(config: Mapping[str, Any]) -> JevPathRouter | None:
    if not config.get("enabled", False):
        return None
    return JevPathRouter(
        model=str(config.get("model", "jev-latest")),
        recall_threshold=float(config.get("recall_threshold", 0.9)),
        base_url=str(config["base_url"]) if config.get("base_url") else None,
        timeout_seconds=float(config.get("timeout_seconds", 1.0)),
        max_retries=int(config.get("max_retries", 0)),
        path_prior_weight=float(config.get("path_prior_weight", 0.25)),
        failure_policy=str(config.get("failure_policy", "fail_closed")),
        prompt_version=str(config.get("prompt_version", JEV_PROMPT_VERSION)),
        replay_path=config.get("replay_path"),
        record_path=config.get("record_path"),
        charge_replay_latency=bool(config.get("charge_replay_latency", True)),
    )


def _path_features(context: np.ndarray, path_id: str) -> np.ndarray:
    one_hot = np.zeros(len(PATH_IDS), dtype=np.float64)
    one_hot[REGISTRY.get(path_id).action_id] = 1.0
    return np.concatenate((np.asarray(context, dtype=np.float64).reshape(-1), one_hot))


class MaxUCBBayesianOptimizationBaseline:
    """Two-stage baseline: path-level MaxUCB, then path-local GP-BO."""

    name = "maxucb-bo"

    def __init__(
        self,
        *,
        beta: float = 1.0,
        gamma: float = 1.0,
        bo_kappa: float = 1.0,
        candidates_per_numeric_path: int = 128,
        max_gp_points_per_path: int = 512,
        seed: int = 42,
        path_surrogate: Any | None = None,
        gp_factory: Callable[[], Any] | None = None,
        exploration: bool = False,
        candidate_bank=None,
    ):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.bo_kappa = float(bo_kappa)
        self.candidates_per_numeric_path = int(candidates_per_numeric_path)
        self.max_gp_points_per_path = int(max_gp_points_per_path)
        if self.max_gp_points_per_path < 2:
            raise ValueError("max_gp_points_per_path must be at least two")
        self.seed = int(seed)
        self.exploration = bool(exploration)
        self.path_surrogate = path_surrogate or CatBoostEnsembleSurrogate(seed=self.seed)
        self._gp_factory = gp_factory
        self.parameter_models: dict[str, Any] = {}
        self.path_counts = {path: 0 for path in PATH_IDS}
        from vec2plan_bandit.paths.configuration_bank import load_bank
        self.configuration_bank=load_bank(candidate_bank)

    def _new_gp(self) -> Any:
        if self._gp_factory is not None:
            return self._gp_factory()
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        except ImportError as error:
            raise RuntimeError("MaxUCB+BO baseline requires scikit-learn") from error
        return GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-3),
            alpha=1e-6,
            normalize_y=True,
            random_state=self.seed,
            n_restarts_optimizer=0,
        )

    def fit(self, observations: Sequence[Mapping[str, Any]]) -> "MaxUCBBayesianOptimizationBaseline":
        if len(observations) < 2:
            raise ValueError("MaxUCB+BO requires at least two training observations")
        best: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
        by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in observations:
            path_id = str(row["path_id"])
            context = np.asarray(row["query_context"], dtype=np.float64)
            reward = float(row["reward"])
            key = (str(row["query_id"]), path_id)
            if key not in best or reward > best[key][1]:
                best[key] = (context, reward)
            by_path[path_id].append(row)
        path_x = np.vstack([_path_features(context, path) for (_query, path), (context, _reward) in best.items()])
        path_y = np.asarray([reward for _key, (_context, reward) in best.items()])
        self.path_surrogate.fit(path_x, path_y)
        self.path_counts = {path: len(by_path[path]) for path in PATH_IDS}
        self.parameter_models = {}
        for path_id, rows in by_path.items():
            if len(rows) < 2:
                continue
            # Exact GP fitting grows cubically with the number of labels. Draw
            # a fixed, seeded subset for each path so large shared training
            # journals remain feasible and repeated runs use the same labels.
            if len(rows) > self.max_gp_points_per_path:
                rng = np.random.default_rng(self.seed + REGISTRY.get(path_id).action_id)
                indices = np.sort(rng.choice(len(rows), self.max_gp_points_per_path, replace=False))
                rows = [rows[int(index)] for index in indices]
            model = self._new_gp()
            model.fit(
                np.vstack([np.asarray(row["features"], dtype=np.float64) for row in rows]),
                np.asarray([float(row["reward"]) for row in rows]),
            )
            self.parameter_models[path_id] = model
        return self

    def _select_path(self, request: BaselineRequest) -> str:
        available = [path for path in PATH_IDS if request.available_paths.get(path, False)]
        if not available:
            raise RuntimeError("no operational path is available")
        path_prediction = self.path_surrogate.predict(
            np.vstack([_path_features(request.query_context, path) for path in available])
        )
        scores = np.asarray(path_prediction.mean) + (self.beta * np.asarray(path_prediction.uncertainty) if self.exploration else 0)
        if self.exploration:
            total = sum(self.path_counts.values())
            scores += self.gamma * np.sqrt(np.log1p(total) / np.asarray([self.path_counts[p] + 1 for p in available]))
        return available[max(range(len(available)), key=lambda index: (scores[index], available[index]))]

    def _select_configuration(self, request: BaselineRequest, chosen_path: str) -> ExecutionConfig:
        try:
            model = self.parameter_models[chosen_path]
        except KeyError as error:
            raise RuntimeError(f"no BO model was fit for selected path {chosen_path}") from error
        candidates = self.configuration_bank[chosen_path] if self.configuration_bank is not None else sample_path_configurations(
            chosen_path,
            self.candidates_per_numeric_path,
            seed=self.seed + REGISTRY.get(chosen_path).action_id,
        )
        if chosen_path == "PRE" and request.available_pre_configurations is not None:
            from vec2plan_bandit.paths.registry import include_optimizer_pre
            if self.configuration_bank is None:
                candidates = include_optimizer_pre(candidates, request.available_pre_configurations)
            candidates = tuple(
                config for config in candidates
                if pre_configuration_key(config.params)
                in request.available_pre_configurations
            )
        if not candidates:
            raise RuntimeError(f"no realized configuration for selected path {chosen_path}")
        features = joint_action_feature_matrix(request.query_context, candidates,
            latency_cap_ms=request.latency_cap_ms, remaining_budget_ms=request.remaining_budget_ms)
        mean, std = model.predict(features, return_std=True)
        acquisition = np.asarray(mean) + (self.bo_kappa * np.asarray(std) if self.exploration else 0)
        winner = max(range(len(candidates)), key=lambda index: (acquisition[index], candidates[index].config_id))
        return candidates[winner]

    def select(self, request: BaselineRequest) -> ExecutionConfig:
        return self._select_configuration(request, self._select_path(request))


class JevRoutingBaseline(MaxUCBBayesianOptimizationBaseline):
    """Jev chooses the path once; the shared-label GP chooses its parameters."""

    name = "jev-router"

    def __init__(self, *, router: JevPathRouter | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.router = router or JevPathRouter()
        self.last_selection_metadata: dict[str, Any] = {}

    def _select_path(self, request: BaselineRequest) -> str:
        try:
            decision = self.router.route(request)
        except JevRoutingError as error:
            self.last_selection_metadata = {"jev_error": dict(error.audit)}
            raise
        self.last_selection_metadata = {
            "jev_chosen_path": decision.chosen_path,
            "jev_path_probabilities": decision.probabilities,
            "jev_confidence": decision.confidence,
            "jev_request_id": decision.request_id,
            "jev_latency_ms": decision.latency_ms,
            "jev_model": self.router.model,
            "jev_resolved_model": decision.resolved_model,
            "jev_request_id_missing": decision.request_id is None,
            "jev_prompt_version": decision.prompt_version,
            "jev_state_sha256": decision.state_sha256,
            "jev_source": decision.source,
            "jev_effective_timeout_seconds": decision.effective_timeout_seconds,
        }
        return decision.chosen_path

    def close(self) -> None:
        self.router.close()


def maxucb_bo_factory(config: Mapping[str, Any]) -> MaxUCBBayesianOptimizationBaseline:
    return MaxUCBBayesianOptimizationBaseline(
        beta=float(config.get("beta", 1.0)),
        gamma=float(config.get("gamma", 1.0)),
        bo_kappa=float(config.get("bo_kappa", 1.0)),
        candidates_per_numeric_path=int(config.get("candidates_per_numeric_path", 128)),
        max_gp_points_per_path=int(config.get("max_gp_points_per_path", 512)),
        seed=int(config.get("seed", 42)),
        candidate_bank=config.get('candidate_bank'),
    )


def jev_router_factory(config: Mapping[str, Any]) -> JevRoutingBaseline:
    return JevRoutingBaseline(
        router=JevPathRouter(
            model=str(config.get("model", "jev-latest")),
            recall_threshold=float(config.get("recall_threshold", 0.9)),
            base_url=str(config["base_url"]) if config.get("base_url") else None,
            timeout_seconds=float(config.get("timeout_seconds", 1.0)),
            max_retries=int(config.get("max_retries", 0)),
            path_prior_weight=float(config.get("path_prior_weight", 0.25)),
            failure_policy=str(config.get("failure_policy", "fail_closed")),
            prompt_version=str(config.get("prompt_version", JEV_PROMPT_VERSION)),
            replay_path=config.get("replay_path"),
            record_path=config.get("record_path"),
            charge_replay_latency=bool(config.get("charge_replay_latency", True)),
        ),
        candidates_per_numeric_path=int(config.get("candidates_per_numeric_path", 128)),
        max_gp_points_per_path=int(config.get("max_gp_points_per_path", 512)),
        seed=int(config.get("seed", 42)),
    )
