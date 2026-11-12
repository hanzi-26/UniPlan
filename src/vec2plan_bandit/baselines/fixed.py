from __future__ import annotations

from typing import Any, Mapping, Sequence

from vec2plan_bandit.baselines.base import BaselineRequest
from vec2plan_bandit.paths.registry import ExecutionConfig, make_execution_config


class FixedConfigurationBaseline:
    """Reproducible control that always requests one legal configuration."""

    requires_context = False

    def __init__(self, path_id: str, params: Mapping[str, Any] | None = None):
        self.config = make_execution_config(path_id, params)
        self.name = f"fixed:{self.config.path_id.lower()}"

    def fit(self, observations: Sequence[Mapping[str, Any]]) -> "FixedConfigurationBaseline":
        del observations
        return self

    def select(self, request: BaselineRequest) -> ExecutionConfig:
        if not request.available_paths.get(self.config.path_id, False):
            raise RuntimeError(f"fixed baseline path is unavailable: {self.config.path_id}")
        return self.config


def fixed_factory(config: Mapping[str, Any]) -> FixedConfigurationBaseline:
    try:
        path_id = str(config["path_id"])
    except KeyError as error:
        raise ValueError("fixed baseline requires path_id") from error
    return FixedConfigurationBaseline(path_id, config.get("params"))
