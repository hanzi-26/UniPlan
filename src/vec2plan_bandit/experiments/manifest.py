from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "psycopg", "faiss-cpu", "qdrant-client", "catboost", "scikit-learn", "lightgbm"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return None


class RunManifest:
    def __init__(self, path: Path, *, project_root: Path, phase: str, seed: int, inputs: Iterable[Path]):
        self.path = path
        previous = json.loads(path.read_text()) if path.is_file() else None
        self.payload: dict[str, Any] = {
            "manifest_version": "vec2plan-run-manifest-v1",
            "status": "RUNNING",
            "phase": phase,
            "seed": int(seed),
            "started_at": _now(),
            "git_commit": _git_commit(project_root),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "packages": _package_versions(),
            "inputs": {str(item): _sha256(item) for item in inputs if item is not None},
            "previous_attempts": ([*previous.get("previous_attempts", []),
                                   {k: v for k, v in previous.items() if k != "previous_attempts"}]
                                  if previous else []),
        }
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def complete(self, **details: Any) -> None:
        self.payload.update(details, status="COMPLETE", completed_at=_now())
        self._write()

    def fail(self, error: BaseException) -> None:
        self.payload.update(
            status="FAILED", failed_at=_now(),
            error_type=type(error).__name__, error_message=str(error)[:1000],
        )
        self._write()
