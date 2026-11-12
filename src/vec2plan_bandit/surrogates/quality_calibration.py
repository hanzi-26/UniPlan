"""Opt-in, train-group-held-out calibration for the arXiv main policy only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vec2plan_bandit.features.action_features import ACTION_FEATURE_NAMES
from vec2plan_bandit.surrogates.catboost_ensemble import Prediction

VERSION = 'arxiv-heldout-quality-calibration-v1'
PATHS = ('PRE', 'POST', 'FAISS', 'ACORN')


def heldout_groups(groups, *, seed=42, fraction=.2):
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    if len(unique) < 5 or not 0 < fraction < 1:
        raise ValueError('Quality calibration requires at least five independent train groups')
    return set(unique[:max(1, int(round(len(unique) * fraction)))])


def action_paths(features):
    matrix = np.atleast_2d(features)
    one_hot = matrix[:, -len(ACTION_FEATURE_NAMES):-len(ACTION_FEATURE_NAMES)+4]
    if not np.allclose(one_hot.sum(axis=1), 1) or not np.all((one_hot == 0) | (one_hot == 1)):
        raise ValueError('Quality calibration action feature schema mismatch')
    return np.asarray(PATHS)[one_hot.argmax(axis=1)]


def fit_calibration(scores, targets, paths, groups, *, binding):
    from sklearn.isotonic import IsotonicRegression
    scores, targets = np.asarray(scores), np.asarray(targets)
    paths, groups = np.asarray(paths), np.asarray(groups)
    if not len(scores) or len({len(v) for v in (scores, targets, paths, groups)}) != 1:
        raise ValueError('Empty or unaligned calibration data')
    if not np.all(np.isfinite(scores)) or not np.all((targets >= 0) & (targets <= 1)):
        raise ValueError('Invalid calibration scores or labels')
    maps = {}
    for path in ('GLOBAL', *PATHS):
        selected = np.ones(len(scores), dtype=bool) if path == 'GLOBAL' else paths == path
        if not selected.any() or (path != 'GLOBAL' and len(set(groups[selected])) < 5):
            continue
        # Each semantic group has the same total weight, despite different
        # numbers of queries/configurations/repetitions in the training journal.
        counts = {g: int(np.sum(groups[selected] == g)) for g in set(groups[selected])}
        weight = np.asarray([1 / counts[g] for g in groups[selected]])
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(
            scores[selected], targets[selected], sample_weight=weight)
        maps[path] = dict(x=model.X_thresholds_.tolist(), y=model.y_thresholds_.tolist(),
                          rows=int(selected.sum()), groups=len(counts))
    return dict(version=VERSION, binding=binding, maps=maps,
                labels_role='train_heldout_groups', dev_test_labels_used=False,
                evidence='empirical probability calibration, not a coverage guarantee')


class CalibratedMainSurrogate:
    def __init__(self, inner, artifact):
        if artifact.get('version') != VERSION or not artifact.get('maps', {}).get('GLOBAL'):
            raise ValueError('Invalid main quality calibration artifact')
        self.inner, self.artifact = inner, artifact

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def predict_main(self, features, facts):
        prediction = self.inner.predict_main(features, facts)
        mean, sigma = prediction.quality.mean, prediction.quality.uncertainty
        calibrated, uncertainty = np.empty_like(mean), np.empty_like(sigma)
        paths = action_paths(features)
        for path in PATHS:
            selected = paths == path
            curve = self.artifact['maps'].get(path, self.artifact['maps']['GLOBAL'])
            def apply(values):
                return np.interp(np.clip(values, 0, 1), curve['x'], curve['y'])
            calibrated[selected] = apply(mean[selected])
            lower = apply(mean[selected] - sigma[selected])
            uncertainty[selected] = np.maximum(sigma[selected], calibrated[selected] - lower)
        return SimpleNamespace(quality=Prediction(calibrated, uncertainty),
            log_latency=prediction.log_latency, timeout_probability=prediction.timeout_probability,
            execution_stages=prediction.execution_stages)


def attach_quality_calibration(policy, options, config):
    if not options.get('enabled', False):
        return
    if not config.get('snapshot_id', '').startswith('vectorsqlbench-arxiv-'):
        raise ValueError('This quality calibration opt-in is restricted to arXiv')
    artifact = json.loads(Path(options['artifact']).read_text())
    if hashlib.sha256(Path(options['artifact']).read_bytes()).hexdigest() != options['sha256']:
        raise ValueError('Quality calibration artifact hash changed')
    # Bind the calibration to the exact quality/latency forest being evaluated.
    model = Path(options['checkpoint'])
    for name, digest in artifact['binding']['checkpoint_files'].items():
        path = (model / name).resolve()
        if not path.is_relative_to(model.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Quality calibration checkpoint changed')
    if policy.surrogate.base_feature_count != artifact['binding']['base_feature_count']:
        raise ValueError('Quality calibration feature count changed')
    policy.surrogate = CalibratedMainSurrogate(policy.surrogate, artifact)
