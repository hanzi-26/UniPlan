from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrainOnlyPCA:
    n_components: int = 8
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None
    fit_split: str | None = None

    def fit(self, matrix: np.ndarray, split: str = "train") -> "TrainOnlyPCA":
        if split != "train":
            raise ValueError("PCA may only be fit on train")
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)) or values.shape[0] < 2:
            raise ValueError("invalid PCA training matrix")
        n = min(self.n_components, values.shape[0] - 1, values.shape[1])
        self.mean_ = values.mean(axis=0)
        _, singular_values, vt = np.linalg.svd(values - self.mean_, full_matrices=False)
        self.components_ = vt[:n]
        squared = singular_values ** 2
        denominator = float(squared.sum())
        self.explained_variance_ratio_ = (
            squared[:n] / denominator if denominator > 0 else np.zeros(n, dtype=np.float64)
        )
        self.fit_split = split
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA is not fit")
        return (np.asarray(matrix, dtype=np.float64) - self.mean_) @ self.components_.T

    def save(self, path: Path) -> None:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA is not fit")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean_,
            components=self.components_,
            explained_variance_ratio=self.explained_variance_ratio_,
            metadata=json.dumps({"n_components": self.n_components, "fit_split": self.fit_split}),
        )
