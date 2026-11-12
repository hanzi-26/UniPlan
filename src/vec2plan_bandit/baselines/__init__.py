"""Offline selector baselines."""
from .base import Baseline, BaselineRequest
from .fixed import FixedConfigurationBaseline, fixed_factory
from .dynamic_ann import dynamic_ann_factory
from .selectivity_switch import selectivity_switch_factory
from .maxucb_bo import (
    JevPathRouter,
    JevRoutingBaseline,
    MaxUCBBayesianOptimizationBaseline,
    jev_router_factory,
    maxucb_bo_factory,
)
from .predictive import PredictiveRewardBaseline, lightgbm_factory, mlp_factory
from .pure_paths import pure_acorn_factory, pure_faiss_factory, pure_post_factory, pure_pre_factory
from .registry import BaselineRegistry


def builtin_baselines() -> BaselineRegistry:
    registry = BaselineRegistry()
    registry.register("fixed", fixed_factory)
    registry.register("dynamic-ann", dynamic_ann_factory)
    registry.register("selectivity-switch", selectivity_switch_factory)
    registry.register("pure-pre", pure_pre_factory)
    registry.register("pure-post", pure_post_factory)
    registry.register("pure-faiss", pure_faiss_factory)
    registry.register("pure-acorn", pure_acorn_factory)
    registry.register("maxucb-bo", maxucb_bo_factory)
    registry.register("jev-router", jev_router_factory)
    registry.register("mlp", mlp_factory)
    registry.register("lightgbm", lightgbm_factory)
    return registry


__all__ = [
    "Baseline",
    "BaselineRequest",
    "BaselineRegistry",
    "FixedConfigurationBaseline",
    "MaxUCBBayesianOptimizationBaseline",
    "JevPathRouter",
    "JevRoutingBaseline",
    "PredictiveRewardBaseline",
    "builtin_baselines",
]
