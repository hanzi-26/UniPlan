from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _shared_prediction_pool(matrix, models):
    # Custom estimator factories keep their existing ndarray interface.
    if not models or not all(type(m).__module__.startswith('catboost') for m in models):
        return None
    from catboost import CatBoost, Pool
    if not all(isinstance(m, CatBoost) for m in models):
        return None
    return Pool(np.ascontiguousarray(matrix, dtype=np.float32), thread_count=1)


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class ConstrainedPrediction:
    """Predictions consumed by quality-constrained MaxUCB.

    Quality is the probability-like regression target for satisfying the
    Recall/full-k gate. Latency is modeled in log1p milliseconds so the long
    tail does not dominate CatBoost's squared-error objective.
    """

    quality: Prediction
    log_latency: Prediction


class CatBoostEnsembleSurrogate:
    """Small-data reward surrogate with ensemble epistemic uncertainty.

    CatBoost predicts reward.  MaxUCB does not invent uncertainty itself: the
    standard deviation across independently bootstrapped CatBoost models is
    the sigma consumed by the UCB acquisition function.
    """

    VERSION = "catboost-bootstrap-ensemble-v1"

    def __init__(
        self,
        *,
        ensemble_size: int = 8,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 42,
        uncertainty_floor: float = 1e-6,
        estimator_factory: Callable[[int], Any] | None = None,
        task_type: str = "CPU",
        devices: str | None = None,
    ):
        if ensemble_size < 2 or iterations <= 0 or depth <= 0:
            raise ValueError("invalid CatBoost ensemble configuration")
        if not np.isfinite(learning_rate) or not np.isfinite(uncertainty_floor) or learning_rate <= 0 or uncertainty_floor < 0:
            raise ValueError("invalid learning rate or uncertainty floor")
        self.ensemble_size = int(ensemble_size)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.uncertainty_floor = float(uncertainty_floor)
        self._estimator_factory = estimator_factory
        if task_type not in {"CPU", "GPU"}:
            raise ValueError("task_type must be CPU or GPU")
        self.task_type = task_type
        self.devices = devices
        self.models: list[Any] = []
        self.feature_count_: int | None = None

    def _factory(self, seed: int) -> Any:
        if self._estimator_factory is not None:
            return self._estimator_factory(seed)
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise RuntimeError(
                "CatBoost is required for CASH+MaxUCB; install the optional "
                "dependency 'catboost'"
            ) from error
        hardware = {"task_type": self.task_type}
        if self.task_type == "GPU":
            hardware.update(devices=self.devices or "0", gpu_ram_part=0.2)
        return CatBoostRegressor(
            loss_function="RMSE",
            allow_const_label=True,
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            **hardware,
        )

    @staticmethod
    def _training_arrays(features: np.ndarray, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if matrix.ndim != 2 or len(matrix) != len(target) or len(target) < 2:
            raise ValueError("training data must contain at least two aligned rows")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
            raise ValueError("training data contains non-finite values")
        return matrix, target

    def fit(self, features: np.ndarray, rewards: np.ndarray) -> "CatBoostEnsembleSurrogate":
        matrix, target = self._training_arrays(features, rewards)
        if getattr(self,'training_devices',()) and self._estimator_factory is None:
            from .parallel_fit import fit_gpu_ensemble
            return fit_gpu_ensemble(self,matrix,target)
        rng = np.random.default_rng(self.seed)
        models = []
        for offset in range(self.ensemble_size):
            indices = rng.integers(0, len(target), size=len(target))
            model = self._factory(self.seed + offset)
            model.fit(matrix[indices], target[indices])
            models.append(model)
        self.models = models
        self.feature_count_ = int(matrix.shape[1])
        return self

    def predict(self, features: np.ndarray, *, _pool=None) -> Prediction:
        if not self.models or self.feature_count_ is None:
            raise RuntimeError("CatBoost surrogate is not fit")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count_:
            raise ValueError("prediction feature dimension mismatch")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("prediction features contain non-finite values")
        pool = _pool
        if pool is None and getattr(self, 'use_shared_pool', True):
            pool = _shared_prediction_pool(matrix, self.models)
        return self._predict_validated(matrix, pool)

    def _predict_validated(self, matrix, pool):
        """Evaluate an already validated batch, shared by both constrained heads."""
        predictions = np.vstack([
            np.asarray(model.predict(pool, thread_count=1) if pool is not None else model.predict(matrix),
                       dtype=np.float64).reshape(-1) for model in self.models])
        if predictions.shape != (len(self.models), len(matrix)) or not np.all(np.isfinite(predictions)):
            raise ValueError("invalid ensemble predictions")
        mean = predictions.mean(axis=0)
        uncertainty = np.maximum(predictions.std(axis=0, ddof=1), self.uncertainty_floor)
        return Prediction(mean=mean, uncertainty=uncertainty)

    def save(self, directory: Path) -> None:
        if not self.models or self.feature_count_ is None:
            raise RuntimeError("CatBoost surrogate is not fit")
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": self.VERSION,
            "ensemble_size": self.ensemble_size,
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "uncertainty_floor": self.uncertainty_floor,
            "feature_count": self.feature_count_,
            "task_type": self.task_type,
            "devices": self.devices,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for index, model in enumerate(self.models):
            model.save_model(str(directory / f"model_{index:02d}.cbm"))

    @classmethod
    def load(cls, directory: Path) -> "CatBoostEnsembleSurrogate":
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("version") != cls.VERSION:
            raise ValueError(f"unsupported surrogate version: {metadata.get('version')}")
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("loading the surrogate requires catboost") from error
        surrogate = cls(
            ensemble_size=int(metadata["ensemble_size"]),
            iterations=int(metadata["iterations"]),
            depth=int(metadata["depth"]),
            learning_rate=float(metadata["learning_rate"]),
            seed=int(metadata["seed"]),
            uncertainty_floor=float(metadata["uncertainty_floor"]),
            task_type=metadata.get("task_type", "CPU"),
            devices=metadata.get("devices"),
        )
        surrogate.feature_count_ = int(metadata["feature_count"])
        for index in range(surrogate.ensemble_size):
            model = CatBoostRegressor()
            model.load_model(str(directory / f"model_{index:02d}.cbm"))
            surrogate.models.append(model)
        return surrogate


class ConstrainedCatBoostSurrogate:
    """Two bootstrap ensembles: quality feasibility and E2E log latency."""

    VERSION = "catboost-quality-loglatency-ensemble-v1"

    def __init__(
        self,
        *,
        ensemble_size: int = 8,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 42,
        uncertainty_floor: float = 1e-6,
        estimator_factory: Callable[[int], Any] | None = None,
        task_type: str = "CPU",
        devices: str | None = None,
    ):
        shared = dict(
            ensemble_size=ensemble_size,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            uncertainty_floor=uncertainty_floor,
            estimator_factory=estimator_factory,
            task_type=task_type,
            devices=devices,
        )
        self.quality_model = CatBoostEnsembleSurrogate(seed=seed, **shared)
        # A disjoint deterministic bootstrap stream prevents the two reported
        # uncertainty estimates from being artifacts of identical resamples.
        self.latency_model = CatBoostEnsembleSurrogate(seed=seed + 10_000, **shared)
        self.seed = int(seed)

    @property
    def feature_count_(self) -> int | None:
        if self.quality_model.feature_count_ != self.latency_model.feature_count_:
            return None
        return self.quality_model.feature_count_

    def fit(
        self,
        features: np.ndarray,
        quality_targets: np.ndarray,
        e2e_latency_ms: np.ndarray,
    ) -> "ConstrainedCatBoostSurrogate":
        self.fast_inference = None
        matrix, quality = CatBoostEnsembleSurrogate._training_arrays(features, quality_targets)
        latency = np.asarray(e2e_latency_ms, dtype=np.float64).reshape(-1)
        if len(latency) != len(matrix) or not np.all(np.isfinite(latency)) or np.any(latency < 0):
            raise ValueError("E2E latency targets must be finite and nonnegative")
        if np.any((quality < 0) | (quality > 1)):
            raise ValueError("quality targets must be in [0, 1]")
        self.quality_model.fit(matrix, quality)
        self.latency_model.fit(matrix, np.log1p(latency))
        return self

    def predict(self, features: np.ndarray) -> ConstrainedPrediction:
        fast = getattr(self, 'fast_inference', None)
        if fast is not None and getattr(self, 'use_fast_inference', True):
            values = fast.predict(features)
            size = len(self.quality_model.models)
            def aggregate(rows, model):
                return Prediction(rows.mean(axis=0),
                    np.maximum(rows.std(axis=0, ddof=1), model.uncertainty_floor))
            quality = aggregate(values[:size], self.quality_model)
            latency = aggregate(values[size:], self.latency_model)
            return ConstrainedPrediction(
                Prediction(np.clip(quality.mean, 0., 1.), quality.uncertainty),
                Prediction(np.maximum(latency.mean, 0.), latency.uncertainty))
        pool = None
        optimized = getattr(self, 'use_shared_pool', True)
        self.quality_model.use_shared_pool = optimized
        self.latency_model.use_shared_pool = optimized
        if optimized:
            matrix = np.asarray(features, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.ndim != 2 or matrix.shape[1] != self.feature_count_ or not np.all(np.isfinite(matrix)):
                raise ValueError('invalid prediction feature matrix')
            pool = _shared_prediction_pool(matrix, self.quality_model.models + self.latency_model.models)
        if optimized:
            quality = self.quality_model._predict_validated(matrix, pool)
            latency = self.latency_model._predict_validated(matrix, pool)
        else:
            quality = self.quality_model.predict(features)
            latency = self.latency_model.predict(features)
        return ConstrainedPrediction(
            quality=Prediction(
                mean=np.clip(quality.mean, 0.0, 1.0),
                uncertainty=quality.uncertainty,
            ),
            log_latency=Prediction(
                mean=np.maximum(latency.mean, 0.0),
                uncertainty=latency.uncertainty,
            ),
        )

    def prepare_fast_inference(self, directory):
        from .frozen_forest import FrozenForest
        native = FrozenForest(self.quality_model.models + self.latency_model.models,
                              self.feature_count_, directory)
        # Label-free startup guard: random inputs, zeros and values on either
        # side of sampled split thresholds. A mismatch aborts before evaluation.
        count = min(native.comparisons, 256)
        probes = np.random.default_rng(42).normal(size=(65 + 2*count, self.feature_count_))
        probes[0] = 0.
        for row, test in enumerate(np.linspace(0, native.comparisons-1, count, dtype=int)):
            feature, border = native.arrays[0][test], native.arrays[1][test]
            probes[65+2*row, feature] = np.nextafter(border, np.float32(-np.inf))
            probes[66+2*row, feature] = np.nextafter(border, np.float32(np.inf))
        previous, flag = getattr(self, 'fast_inference', None), getattr(self, 'use_fast_inference', True)
        try:
            self.use_fast_inference = False
            reference = self.predict(probes)
            self.fast_inference, self.use_fast_inference = native, True
            actual = self.predict(probes)
            difference = 0.
            for head in ('quality', 'log_latency'):
                for statistic in ('mean', 'uncertainty'):
                    a, b = (getattr(getattr(p, head), statistic) for p in (reference, actual))
                    difference = max(difference, float(np.max(np.abs(a-b))))
                    if not np.allclose(a, b, rtol=1e-12, atol=1e-12):
                        raise ValueError('native main inference failed reference-equivalence guard')
        except BaseException:
            self.fast_inference = previous
            raise
        finally:
            self.use_fast_inference = flag
        native.audit['startup_validation'] = dict(probes=len(probes), max_absolute_error=difference,
                                                 labels_used=False, tolerance=1e-12)
        (Path(directory)/'manifest.json').write_text(json.dumps(native.audit, indent=2)+'\n')
        return native.audit

    def save(self, directory: Path) -> None:
        if self.feature_count_ is None:
            raise RuntimeError("constrained CatBoost surrogate is not fit")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(
            json.dumps({"version": self.VERSION, "seed": self.seed}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.quality_model.save(directory / "quality")
        self.latency_model.save(directory / "log_latency")

    @classmethod
    def load(cls, directory: Path) -> "ConstrainedCatBoostSurrogate":
        if (Path(directory)/'main-cost.json').exists():
            from .main_cost import MainCostSurrogate
            return MainCostSurrogate.load(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("version") != cls.VERSION:
            raise ValueError(f"unsupported constrained surrogate version: {metadata.get('version')}")
        instance = cls(seed=int(metadata["seed"]))
        instance.quality_model = CatBoostEnsembleSurrogate.load(directory / "quality")
        instance.latency_model = CatBoostEnsembleSurrogate.load(directory / "log_latency")
        if instance.feature_count_ is None:
            raise ValueError("quality and latency feature dimensions disagree")
        return instance
