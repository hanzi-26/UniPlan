from __future__ import annotations

from typing import Any, Mapping

from vec2plan_bandit.baselines.fixed import FixedConfigurationBaseline
from vec2plan_bandit.paths.registry import ACORN, FAISS, POST, PRE


DEFAULTS: dict[str, dict[str, Any]] = {
    PRE: {"index_mode": "no_index", "join_strategy": "all"},
    POST: {"ef_search": 100, "max_scan_tuples": 100_000},
    FAISS: {"ef_search": 100, "candidate_multiplier": 4.0},
    ACORN: {"hnsw_ef": 100},
}


def _pure(path_id: str, config: Mapping[str, Any]) -> FixedConfigurationBaseline:
    params = dict(DEFAULTS[path_id])
    params.update(dict(config.get("params", {})))
    baseline = FixedConfigurationBaseline(path_id, params)
    baseline.name = f"pure-{path_id.lower()}"
    return baseline


def pure_pre_factory(config: Mapping[str, Any]) -> FixedConfigurationBaseline:
    return _pure(PRE, config)


def pure_post_factory(config: Mapping[str, Any]) -> FixedConfigurationBaseline:
    return _pure(POST, config)


def pure_faiss_factory(config: Mapping[str, Any]) -> FixedConfigurationBaseline:
    return _pure(FAISS, config)


def pure_acorn_factory(config: Mapping[str, Any]) -> FixedConfigurationBaseline:
    return _pure(ACORN, config)
