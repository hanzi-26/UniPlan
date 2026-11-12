from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Any, Iterable, Mapping

from vec2plan_bandit.queries.loader import QueryRecord


PRE = "PRE"
POST = "POST"
FAISS = "FAISS"
ACORN = "ACORN"
PATH_IDS = (PRE, POST, FAISS, ACORN)
PRE_ACCESS_STRATEGIES = ("scan", "btree", "hash", "bitmap")
PRE_JOIN_STRATEGIES = ("all", "hash", "merge", "nested_loop")


def pre_access_strategy(params: Mapping[str, Any]) -> str:
    """Return the physical-access label represented by a PRE configuration."""
    return (
        "scan"
        if params.get("index_mode", "no_index") == "no_index"
        else str(params.get("index_strategy", "btree"))
    )


def pre_configuration_key(params: Mapping[str, Any]) -> str:
    """Stable key used by EXPLAIN availability gates and policy filtering."""
    return f"{pre_access_strategy(params)}:{params.get('join_strategy', 'all')}"


@dataclass(frozen=True)
class ParameterSpec:
    """Schema for one path-local parameter.

    Numeric parameters describe a bounded interval. The policy may predict any
    value in that interval; integer-valued backends are rounded only at the
    execution boundary. Categorical parameters are reserved for genuine
    choices such as PRE index/no-index, not for discretising numeric tuning.
    """

    name: str
    kind: str
    default: Any
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()
    description: str = ""
    active_when: Mapping[str, Any] = field(default_factory=dict)

    def normalize(self, value: Any) -> Any:
        if self.kind == "categorical":
            if value not in self.choices:
                raise ValueError(
                    f"{self.name} must be one of {self.choices}, got {value!r}"
                )
            return value
        if self.kind == "string":
            if not isinstance(value, str) or not value:
                raise ValueError(f"{self.name} must be a non-empty string")
            return value
        if self.kind not in {"real", "integer"} or not isinstance(value, Real):
            raise ValueError(f"invalid {self.kind} value for {self.name}: {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{self.name} must be finite")
        if self.lower is not None and number < self.lower:
            raise ValueError(f"{self.name} is below {self.lower}: {number}")
        if self.upper is not None and number > self.upper:
            raise ValueError(f"{self.name} is above {self.upper}: {number}")
        return int(round(number)) if self.kind == "integer" else number


@dataclass(frozen=True)
class PathSpec:
    action_id: int
    path_id: str
    backend: str
    parameters: tuple[ParameterSpec, ...]
    description: str

    @property
    def params(self) -> dict[str, Any]:
        """Backward-compatible access to default parameters."""
        return {parameter.name: parameter.default for parameter in self.parameters}

    def normalize_params(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        supplied = dict(values or {})
        known = {parameter.name for parameter in self.parameters}
        unknown = set(supplied) - known
        if unknown:
            raise ValueError(f"unknown {self.path_id} parameters: {sorted(unknown)}")
        normalized: dict[str, Any] = {}
        for parameter in self.parameters:
            value = supplied.get(parameter.name, parameter.default)
            if parameter.active_when and any(
                normalized.get(name, supplied.get(name)) != expected
                for name, expected in parameter.active_when.items()
            ):
                continue
            normalized[parameter.name] = parameter.normalize(value)
        return normalized


# PRE access methods remain variants of one exact path; no vector index is used.
PATH_SPECS = (
    PathSpec(
        0,
        PRE,
        "postgresql",
        (
            ParameterSpec(
                "index_mode",
                "categorical",
                "no_index",
                choices=("index", "no_index"),
                description="Use or disable non-vector indexes while constructing the exact eligible set.",
            ),
            ParameterSpec(
                "index_strategy",
                "categorical",
                "btree",
                choices=("btree", "hash", "bitmap", "optimizer"),
                description="Requested ordinary PostgreSQL access strategy; plans must be audited.",
                active_when={"index_mode": "index"},
            ),
            ParameterSpec(
                "join_strategy",
                "categorical",
                "all",
                choices=PRE_JOIN_STRATEGIES,
                description=(
                    "PostgreSQL join family allowed inside PRE: normal optimizer choice, "
                    "hash join, merge join, or nested-loop join."
                ),
            ),
        ),
        "Exact pre-filter search; neither mode may use HNSW.",
    ),
    PathSpec(
        1,
        POST,
        "postgresql",
        (
            ParameterSpec("ef_search", "integer", 100, 40, 1000),
            ParameterSpec("max_scan_tuples", "integer", 100_000, 20_000, 3_000_000),
        ),
        "PostgreSQL pgvector HNSW search with metadata post-filtering.",
    ),
    PathSpec(
        2,
        FAISS,
        "faiss",
        (
            ParameterSpec("ef_search", "integer", 100, 40, 32000),
            ParameterSpec("candidate_multiplier", "real", 4.0, 1.0, 10000.0),
        ),
        "FAISS HNSW candidate generation followed by an exact eligibility filter.",
    ),
    PathSpec(
        3,
        ACORN,
        "qdrant",
        (ParameterSpec("hnsw_ef", "integer", 100, 40, 32000),),
        "Qdrant ACORN-1 filtered graph traversal; ACORN is forced on.",
    ),
)

# Historical code imports ACTIONS. Its contents now express the new four-path
# protocol rather than the retired brute-force arm.
ACTIONS = PATH_SPECS


class PathRegistry:
    def __init__(self, specs: Iterable[PathSpec] = PATH_SPECS):
        self._specs = tuple(specs)
        ids = [spec.action_id for spec in self._specs]
        names = [spec.path_id for spec in self._specs]
        if ids != list(range(len(self._specs))):
            raise ValueError("action IDs must be unique, contiguous, and zero-based")
        if len(names) != len(set(names)):
            raise ValueError("path IDs must be unique")
        self._by_id = {spec.action_id: spec for spec in self._specs}
        self._by_name = {spec.path_id: spec for spec in self._specs}

    @property
    def specs(self) -> tuple[PathSpec, ...]:
        return self._specs

    def get(self, path: int | str) -> PathSpec:
        try:
            return self._by_id[path] if isinstance(path, int) else self._by_name[path]
        except KeyError as error:
            raise ValueError(f"unknown path: {path!r}") from error

    def normalize(self, path: int | str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.get(path).normalize_params(params)


REGISTRY = PathRegistry()


@dataclass(frozen=True)
class ExecutionConfig:
    config_id: str
    action_id: int
    path_id: str
    params: dict[str, Any] = field(default_factory=dict)


def _config_id(path_id: str, params: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"path_id": path_id, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{path_id.lower()}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def make_execution_config(
    path: int | str, params: Mapping[str, Any] | None = None
) -> ExecutionConfig:
    spec = REGISTRY.get(path)
    values = spec.normalize_params(params)
    return ExecutionConfig(
        config_id=_config_id(spec.path_id, values),
        action_id=spec.action_id,
        path_id=spec.path_id,
        params=values,
    )


def default_execution_config(action_id: int) -> ExecutionConfig:
    return make_execution_config(action_id)


def include_optimizer_pre(configs, available):
    """Opt-in mixed-access PRE only when a fresh EXPLAIN receipt enables it.

    Legacy/default designs and frozen availability masks remain unchanged.
    Every learned selector receives the same new PRE alternatives.
    """
    result = {c.config_id: c for c in configs}
    for join in PRE_JOIN_STRATEGIES:
        if f'optimizer:{join}' in available:
            c = make_execution_config(PRE, dict(index_mode='index', index_strategy='optimizer', join_strategy=join))
            result[c.config_id] = c
    return tuple(result.values())


def legal_action_mask(
    query: QueryRecord,
    *,
    available: Mapping[str, bool] | None = None,
    faiss_abstract: bool = True,
    faiss_title: bool = True,
    acorn_available: bool = True,
) -> list[bool]:
    """Return availability without changing stable action numbering."""

    resolved = {path_id: True for path_id in PATH_IDS}
    if available is not None:
        unknown = set(available) - set(PATH_IDS)
        if unknown:
            raise ValueError(f"availability contains unknown paths: {sorted(unknown)}")
        resolved.update({name: bool(value) for name, value in available.items()})
    resolved[FAISS] &= (
        faiss_abstract if query.embedding_column == "abstract_embedding" else faiss_title
    )
    resolved[ACORN] &= bool(acorn_available)
    return [resolved[spec.path_id] for spec in PATH_SPECS]
