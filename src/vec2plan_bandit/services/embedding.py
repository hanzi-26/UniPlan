from __future__ import annotations

import time
from typing import Any

import numpy as np
import requests


class EmbeddingServiceClient:
    """Read-only adapter for the existing ``Embedding_Service`` HTTP API."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        dimension: int = 384,
        normalize: bool = True,
        timeout_seconds: float = 60.0,
        session: Any = requests,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.dimension = int(dimension)
        self.normalize = bool(normalize)
        self.timeout_seconds = float(timeout_seconds)
        self._session = session

    def embed(self, text: str) -> tuple[np.ndarray, float]:
        started = time.perf_counter_ns()
        response = self._session.post(
            self.url,
            json={"model": self.model, "texts": [text]},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get('model', self.model) != self.model:
            raise ValueError('embedding service returned a different model')
        matrix = np.asarray(payload["embeddings"], dtype=np.float32)
        if matrix.shape != (1, self.dimension) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid embedding shape/content: {matrix.shape}")
        vector = matrix[0]
        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("zero embedding")
            vector = vector / norm
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        return np.ascontiguousarray(vector, dtype=np.float32), latency_ms

    def health(self) -> dict[str, Any]:
        endpoint = self.url.rsplit("/", 1)[0] + "/health"
        response = self._session.get(endpoint, timeout=min(self.timeout_seconds, 5.0))
        response.raise_for_status()
        return dict(response.json())
