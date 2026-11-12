"""Shared immutable run contracts and model bundles for frozen experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from vec2plan_bandit.assets.indexes import sha256
from vec2plan_bandit.features.action_features import ACTION_FEATURE_VERSION
from vec2plan_bandit.features.feature_pipeline import FeaturePipeline
from vec2plan_bandit.feedback.reward import REWARD_VERSION, reward_version
from vec2plan_bandit.policies.max_ucb import MaxUCB
from vec2plan_bandit.surrogates.catboost_ensemble import ConstrainedCatBoostSurrogate

VERSION = "frozen-constrained-e2e-pre-strategies-v3"


def bind_run(args: Any, config: dict[str, Any], phase: str) -> dict[str, Any]:
    inputs = {}
    for name in ("query_source", "split_json", "training_matrix", "ground_truth_jsonl",
                 "feature_embeddings", "query_statistics", "baseline_config"):
        path = getattr(args, name, None)
        if path is not None:
            inputs[name] = sha256(Path(path))
    root = Path(__file__).resolve().parents[3]
    for name, value in (
        ("snapshot_manifest", Path(config.get("metadata", {}).get("snapshot_dir", "data/metadata")) / "manifest.json"),
        ("faiss_manifest", Path(config.get("assets", {}).get("faiss_dir", "data/indexes/faiss")) / "manifest.json"),
        ("qdrant_receipt", Path(config.get("assets", {}).get("qdrant_receipt", "data/indexes/qdrant-import.json"))),
        ("extra_metadata_manifest", Path(config.get("metadata", {}).get("extra_manifest", "data/metadata/manifest.json"))),
        ("qdrant_extra_receipt", Path(config.get("assets", {}).get("qdrant_extra_receipt", "data/indexes/qdrant-extra-v1.json"))),
    ):
        if phase == "prepare" and name not in {"snapshot_manifest", "extra_metadata_manifest"}:
            continue
        path = value if value.is_absolute() else root / value
        if path.is_file():
            inputs[name] = sha256(path)
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        inputs["checkpoint"] = {str(p.relative_to(checkpoint)): sha256(p)
                                for p in sorted(checkpoint.rglob("*")) if p.is_file()}
    # Bind the source, not just the data/config; an implementation change requires
    # a new result directory even if a developer forgot to bump a version.
    package = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*.py"))}
    payload = {"version": VERSION, "phase": phase, "config": config, "inputs": inputs,
               "source": source, "reward": reward_version(float(config.get("reward", {}).get("recall_threshold", 0.9))), "features": ACTION_FEATURE_VERSION,
               "roles": args.roles, "repeats": args.repeats, "cap_ms": args.latency_cap_ms,
               "seed": getattr(args, "seed", 42)}
    payload["training_policy"] = getattr(args, "training_policy", "cash-maxucb")
    baseline_path = getattr(args, "baseline_config", None)
    if baseline_path:
        for spec in json.loads(Path(baseline_path).read_text()).get("baselines", []):
            if spec.get("training_matrix"):
                inputs[f"method_training:{spec['id']}"] = sha256(Path(spec["training_matrix"]))
            if spec.get("name") == "frozen-policy":
                bundle = Path(spec["config"]["checkpoint"])
                inputs[f"method_checkpoint:{spec['id']}"] = {
                    str(p.relative_to(bundle)): sha256(p) for p in sorted(bundle.rglob("*")) if p.is_file()
                }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("run contract changed; use a new output directory")
    else:
        if any(args.output_dir.glob("*live.jsonl")):
            raise ValueError("unversioned result journal exists; use a new output directory")
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2)
    return payload


def save_bundle(directory: Path, pipeline: FeaturePipeline, policy: MaxUCB, metadata: dict[str, Any]) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"frozen model bundle already exists: {directory}")
    staging = Path(tempfile.mkdtemp(prefix="frozen-", dir=directory.parent))
    pipeline.save(staging / "features.npz")
    policy.surrogate.save(staging / "surrogate")
    payload = {"version": VERSION, "action_features": ACTION_FEATURE_VERSION, "reward": reward_version(float(metadata.get("recall_threshold", 0.9))),
               "policy": policy.state_dict(), **metadata}
    payload["files"] = {str(p.relative_to(staging)): sha256(p) for p in staging.rglob("*") if p.is_file()}
    (staging / "bundle.json").write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(staging, directory)


def load_bundle(directory: Path, *, cap_ms: float, train_ids: list[str], expected: dict[str, Any] | None = None,
                expected_policy: dict[str, Any] | None = None):
    metadata = json.loads((directory / "bundle.json").read_text())
    metadata.setdefault("recall_threshold", 0.9)
    if (metadata["version"] != VERSION or metadata["reward"] != reward_version(float(metadata.get("recall_threshold", 0.9)))
            or metadata["action_features"] != ACTION_FEATURE_VERSION or metadata["cap_ms"] != cap_ms):
        raise ValueError("frozen bundle protocol mismatch")
    if expected is not None and any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen bundle dataset/embedding provenance mismatch")
    for name, digest in metadata["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError(f"frozen bundle checksum mismatch: {name}")
    pipeline = FeaturePipeline.load(directory / "features.npz")
    if set(pipeline.fit_query_ids) != set(train_ids):
        raise ValueError("frozen bundle was trained on a different query split")
    surrogate = ConstrainedCatBoostSurrogate.load(directory / "surrogate")
    state = metadata["policy"]
    policy = MaxUCB(surrogate, **{key: state[key] for key in MaxUCB.SETTINGS if key in state})
    policy.load_state_dict(state)
    if expected_policy is not None:
        configured = MaxUCB(surrogate, **{key: expected_policy[key] for key in MaxUCB.SETTINGS if key in expected_policy})
        for key in MaxUCB.SETTINGS:
            if policy.state_dict()[key] != configured.state_dict()[key]:
                raise ValueError(f"frozen bundle policy/config mismatch: {key}; create a new bundle")
    return pipeline, policy, surrogate
