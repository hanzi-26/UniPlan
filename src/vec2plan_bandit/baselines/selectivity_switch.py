"""Train-calibrated selectivity switch between exact PRE and HNSW POST.

This is an explicit heuristic control, not a claimed reproduction of a named
paper. The crossover is chosen from paired train measurements only.
"""
from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .base import BaselineRequest
from .pure_paths import DEFAULTS
from vec2plan_bandit.paths.registry import PRE, POST, make_execution_config, pre_configuration_key


class SelectivitySwitchBaseline:
    name = 'selectivity-switch'
    requires_context = False

    def __init__(self, config: Mapping[str, Any]):
        self.pre = make_execution_config(PRE, DEFAULTS[PRE])
        self.post = make_execution_config(POST, {**DEFAULTS[POST], **config.get('post_params', {})})
        self.threshold = None
        self.training_pairs = 0

    def fit(self, observations: Sequence[Mapping[str, Any]]):
        pairs = defaultdict(lambda: defaultdict(list))
        selectivities = {}
        for row in observations:
            path = row['path_id']
            if path not in {PRE, POST}:
                continue
            config = make_execution_config(path, row.get('requested_params'))
            wanted = self.pre if path == PRE else self.post
            if config.config_id != wanted.config_id:
                continue
            value = (row.get('query_statistics') or {}).get('estimated_selectivity')
            if value is None:
                continue
            value = float(value)
            reward = float(row['reward'])
            if not math.isfinite(value) or not 0 <= value <= 1 or not math.isfinite(reward):
                raise ValueError('Invalid Dynamic ANN training statistics or reward')
            qid = str(row['query_id'])
            if qid in selectivities and selectivities[qid] != value:
                raise ValueError('Inconsistent train selectivity')
            selectivities[qid] = value
            pairs[qid][path].append(reward)
        paired = [(selectivities[qid], float(np.mean(values[PRE])), float(np.mean(values[POST])))
                  for qid, values in pairs.items() if values[PRE] and values[POST]]
        if not paired:
            raise ValueError('Dynamic ANN needs paired train PRE/POST default measurements and selectivity estimates')
        # PRE for selective filters, POST for dense filters. Train reward
        # penalizes missing k, low recall, timeouts and execution failures.
        candidates = [-1.0, *sorted({value for value, _, _ in paired}), 1.0]
        self.threshold = max(candidates, key=lambda threshold: (
            sum(pre if value <= threshold else post for value, pre, post in paired), threshold))
        self.training_pairs = len(paired)
        return self

    def select(self, request: BaselineRequest):
        if self.threshold is None:
            raise RuntimeError('Dynamic ANN baseline is not fit')
        pre_available = request.available_paths.get(PRE, False)
        if request.available_pre_configurations is not None:
            pre_available = pre_available and pre_configuration_key(self.pre.params) in request.available_pre_configurations
        post_available = request.available_paths.get(POST, False)
        if not pre_available and not post_available:
            raise RuntimeError('Dynamic ANN has no operational PRE/POST path')
        if not pre_available:
            return self.post
        value = (request.query_statistics or {}).get('estimated_selectivity')
        if value is not None and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
            raise ValueError('Invalid Dynamic ANN request selectivity')
        # Unknown selectivity keeps the exact path rather than using labels
        # or timing both alternatives on a held-out query.
        return self.pre if value is None or not post_available or float(value) <= self.threshold else self.post


def selectivity_switch_factory(config):
    return SelectivitySwitchBaseline(config)
