"""Check configured runtime services without requiring historical manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from vec2plan_bandit.experiments.collector import MatrixCollector


def run_preflight(config: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    worker = MatrixCollector(work_dir / "runtime", config, read_only=True)
    try:
        try:
            vector, latency = worker.embed("vector search service check")
            embedding = {"ready": len(vector) == config["embedding"]["dimension"], "probe_ms": latency}
        except Exception as error:
            embedding = {"ready": False, "error": str(error)}
        paths = worker.path_router.available(probe=True)
        required = config.get("require_paths", [])
        ready = embedding["ready"] and all(paths.get(path, False) for path in required)
        return {"embedding_service": embedding, "paths": paths,
                "required_paths": required, "formal_ready": bool(ready)}
    finally:
        worker.close()
