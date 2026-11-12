from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


RESULT_SCHEMA_VERSION = "vec2plan-live-result-v1"
REQUIRED_FIELDS = {
    "method_id", "query_id", "repeat", "k", "e2e_latency_ms",
    "deadline_pass", "article_ids", "reward", "recorded_at",
}


def result_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["method_id"]), str(row["query_id"]), int(row["repeat"])


def validate_result(row: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"result row is missing fields: {sorted(missing)}")
    if int(row["repeat"]) <= 0 or int(row["k"]) <= 0:
        raise ValueError("result repeat and k must be positive")
    if not math.isfinite(float(row["e2e_latency_ms"])) or float(row["e2e_latency_ms"]) < 0:
        raise ValueError("result latency must be non-negative")


class JsonlResultStore:
    """Append-only result store with deterministic resume keys."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed = {
            result_key(row) for row in self.read()
        } if self.path.exists() else set()

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def contains(self, method_id: str, query_id: str, repeat: int) -> bool:
        return (method_id, query_id, repeat) in self.completed

    def append(self, row: Mapping[str, Any]) -> None:
        payload = {"result_schema_version": RESULT_SCHEMA_VERSION, **dict(row)}
        validate_result(payload)
        key = result_key(payload)
        if key in self.completed:
            raise ValueError(f"duplicate result key: {key}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed.add(key)
