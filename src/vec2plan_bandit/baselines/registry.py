from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Mapping

from vec2plan_bandit.baselines.base import Baseline


BaselineFactory = Callable[[Mapping[str, Any]], Baseline]


class BaselineRegistry:
    """Local factories plus optional third-party entry-point discovery."""

    ENTRY_POINT_GROUP = "vec2plan_bandit.baselines"

    def __init__(self) -> None:
        self._factories: dict[str, BaselineFactory] = {}

    def register(self, name: str, factory: BaselineFactory, *, replace: bool = False) -> None:
        normalized = name.strip().lower()
        if not normalized or (normalized in self._factories and not replace):
            raise ValueError(f"invalid or duplicate baseline: {name!r}")
        self._factories[normalized] = factory

    def discover(self) -> None:
        for item in entry_points(group=self.ENTRY_POINT_GROUP):
            self.register(item.name, item.load(), replace=False)

    def create(self, name: str, config: Mapping[str, Any] | None = None) -> Baseline:
        try:
            baseline = self._factories[name.strip().lower()](dict(config or {}))
        except KeyError as error:
            raise ValueError(f"unknown baseline {name!r}; available={self.names()}") from error
        if not isinstance(baseline, Baseline):
            raise TypeError(f"baseline {name!r} does not implement the Baseline protocol")
        return baseline

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
