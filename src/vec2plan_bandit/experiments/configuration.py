from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from vec2plan_bandit.baselines import builtin_baselines
from vec2plan_bandit.baselines.maxucb_bo import JEV_FAILURE_POLICIES, JEV_PROMPT_VERSION
from vec2plan_bandit.evaluation.quality import RECALL_PASS_THRESHOLD, required_hits
from vec2plan_bandit.paths.registry import PATH_IDS, REGISTRY, make_execution_config


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return payload


def _validate_jev_settings(jev: Mapping[str, Any], *, enabled: bool) -> None:
    model = str(jev.get("model", "jev-latest"))
    if not model:
        raise ValueError("jev.model must be nonempty")
    if enabled and "latest" in model.lower() and not jev.get("allow_unpinned_model", False):
        raise ValueError("enabled Jev experiments require a pinned model, not *-latest")
    timeout = float(jev.get("timeout_seconds", 1.0))
    prior_weight = float(jev.get("path_prior_weight", 0.25))
    retries = jev.get("max_retries", 0)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("jev.timeout_seconds must be finite and positive")
    if not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("jev.path_prior_weight must be finite and nonnegative")
    if type(retries) is not int or retries < 0:
        raise ValueError("jev.max_retries must be a nonnegative integer")
    if str(jev.get("failure_policy", "fail_closed")) not in JEV_FAILURE_POLICIES:
        raise ValueError(f"jev.failure_policy must be one of {JEV_FAILURE_POLICIES}")
    if str(jev.get("prompt_version", JEV_PROMPT_VERSION)) != JEV_PROMPT_VERSION:
        raise ValueError(f"jev.prompt_version must be {JEV_PROMPT_VERSION}")
    if jev.get("base_url") and not str(jev["base_url"]).startswith(("http://", "https://")):
        raise ValueError("jev.base_url must be HTTP(S)")
    if type(jev.get("charge_replay_latency", True)) is not bool:
        raise ValueError("jev.charge_replay_latency must be boolean")
    replay, record = jev.get("replay_path"), jev.get("record_path")
    if replay and record and Path(str(replay)).expanduser().resolve() == Path(str(record)).expanduser().resolve():
        raise ValueError("jev replay_path and record_path must differ")


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    reward = config.get("reward", {})
    threshold = float(reward.get("recall_threshold", -1))
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("reward.recall_threshold must be within (0, 1]")
    quality = config.get("quality", {})
    allowed_k = list(range(1, 11))
    if quality.get("allowed_k") != allowed_k or quality.get("required_hits") != {
        str(k): required_hits(k, recall_threshold=threshold) for k in allowed_k
    }:
        raise ValueError("quality.allowed_k/required_hits must match Recall@k protocol")
    embedding = config.get("embedding")
    if not isinstance(embedding, Mapping):
        raise ValueError("experiment config requires embedding")
    if not str(embedding.get("url", "")).startswith(("http://", "https://")):
        raise ValueError("embedding.url must be HTTP(S)")
    if not str(embedding.get("model", "")) or int(embedding.get("dimension", 0)) <= 0:
        raise ValueError("embedding model and positive dimension are required")
    measurement = config.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ValueError("experiment config requires measurement")
    for name in ("warmup_runs", "latency_runs", "statement_timeout_ms"):
        if int(measurement.get(name, 0)) < (0 if name == "warmup_runs" else 1):
            raise ValueError(f"measurement.{name} must be positive")
    design = config.get("initial_design", {})
    if int(design.get("points_per_numeric_path", 0)) <= 0:
        raise ValueError("initial_design.points_per_numeric_path must be positive")
    actions = config.get("actions", {})
    configured = []
    for action_id in range(len(PATH_IDS)):
        row = actions.get(str(action_id))
        if not isinstance(row, Mapping):
            raise ValueError(f"missing actions.{action_id}")
        path_id = str(row.get("path_id"))
        if path_id != PATH_IDS[action_id]:
            raise ValueError(f"actions.{action_id} must be {PATH_IDS[action_id]}")
        supplied = make_execution_config(path_id, row.get("params"))
        if supplied.params != make_execution_config(path_id).params:
            raise ValueError(f"actions.{action_id} defaults disagree with registry; change the registry or a baseline's params")
        configured.append(path_id)
    if tuple(configured) != PATH_IDS:
        raise ValueError("experiment must configure exactly the four stable paths")
    # Registry is the single source of truth. Mirrored documentation in JSON
    # must match, never silently accept a bound the optimizer will ignore.
    for path, parameters in config.get("parameter_spaces", {}).items():
        spec = REGISTRY.get(path)
        for name, declared in parameters.items():
            parameter = next((p for p in spec.parameters if p.name == name), None)
            if parameter is None:
                raise ValueError(f"unknown parameter space: {path}.{name}")
            if isinstance(declared, Mapping):
                expected_kind = "real_interval" if parameter.kind == "real" else "integer_interval"
                if declared != {"type": expected_kind, "min": parameter.lower, "max": parameter.upper}:
                    raise ValueError(f"parameter_spaces disagrees with registry: {path}.{name}")
            elif parameter.kind == "categorical" and tuple(declared) != parameter.choices:
                if (path == 'PRE' and name == 'index_strategy' and tuple(declared) == ('btree','hash','bitmap')
                        and not config.get('relational_pre',{}).get('enabled',False)):
                    continue  # Legacy availability masks never enable the new mixed-access action.
                raise ValueError(f"parameter choices disagree: {path}.{name}")
    fixed = config.get("backend_fixed_settings", {})
    expected_fixed = {"POST": {"iterative_scan": "strict_order", "scan_mem_multiplier": 2.0, "enable_seqscan": False},
                      "ACORN": {"enable": True, "max_selectivity": 1.0, "exact": False}}
    if any(value != expected_fixed.get(path) for path, value in fixed.items()):
        raise ValueError("backend_fixed_settings disagrees with executor implementation")
    acorn = config.get("acorn", {})
    if acorn.get("max_selectivity", 1.0) != 1.0 or acorn.get("exact_fallback", False):
        raise ValueError("ACORN requires max_selectivity=1 and no exact fallback")
    for name in ("beta_quality", "beta_latency", "kappa_quality", "kappa_latency", "gamma"):
        value = float(config.get("max_ucb", {}).get(name, 1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"max_ucb.{name} must be finite and nonnegative")
    threshold = float(config.get("max_ucb", {}).get("quality_probability_min", 0.8))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("max_ucb.quality_probability_min must be in [0, 1]")
    refit_every = config.get("online_learning", {}).get("refit_every", 8)
    if type(refit_every) is not int or refit_every < 1:
        raise ValueError("online_learning.refit_every must be a positive integer")
    jev = config.get("jev", {})
    if not isinstance(jev, Mapping) or type(jev.get("enabled", False)) is not bool:
        raise ValueError("jev must be an object with a boolean enabled field")
    _validate_jev_settings(jev, enabled=bool(jev.get("enabled", False)))


def checkpoint_only_baselines(config: Mapping[str, Any]) -> bool:
    """Only frozen-policy controls can skip shared training observations."""
    return bool(config.get("baselines")) and all(
        row["name"] == "frozen-policy" and not row.get("training_matrix")
        for row in config["baselines"]
    )


def validate_baseline_config(config: Mapping[str, Any]) -> None:
    rows = config.get("baselines")
    if not isinstance(rows, list) or not rows:
        raise ValueError("baseline config requires a non-empty baselines list")
    known = set(builtin_baselines().names()) | {"native-pg", "text2vectorsql", "frozen-policy"}
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every baseline entry must be an object")
        identifier = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip().lower()
        if not identifier or identifier in identifiers:
            raise ValueError(f"duplicate or empty baseline id: {identifier!r}")
        if name not in known:
            raise ValueError(f"unknown baseline {name!r}; available={sorted(known)}")
        if name == "jev-router":
            settings = row.get("config", {})
            if not isinstance(settings, Mapping):
                raise ValueError("jev-router config must be an object")
            _validate_jev_settings(settings, enabled=True)
        if name == "frozen-policy":
            settings = row.get("config", {})
            if not settings.get("checkpoint") or not isinstance(settings.get("max_ucb"), Mapping):
                raise ValueError("frozen-policy requires a checkpoint and explicit max_ucb settings")
        if name == "native-pg" and row.get("runner") != "external_native_pg":
            raise ValueError("native-pg must use external_native_pg runner")
        if name == "text2vectorsql":
            if row.get("runner") != "external_text2vectorsql":
                raise ValueError("text2vectorsql must use external_text2vectorsql runner")
            from vec2plan_bandit.baselines.text2vectorsql import Text2VectorSQLTranslator
            Text2VectorSQLTranslator(**dict(row.get("config", {})))
        identifiers.add(identifier)
