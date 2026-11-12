"""Dataset-independent opt-in for train-group-held-out quality calibration.

The old arXiv-only entry point is unchanged. Only a main policy explicitly
configured with this new artifact is wrapped; cost/risk outputs pass through.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .catboost_ensemble import Prediction
from .quality_calibration import action_paths, heldout_groups, fit_calibration, PATHS

VERSION = 'main-train-group-quality-calibration-v1'


def fit_group_calibration(scores, targets, paths, groups, *, binding):
    result = fit_calibration(scores, targets, paths, groups, binding=binding)
    result['version'] = VERSION
    return result


class GroupCalibratedMainSurrogate:
    def __init__(self, inner, artifact):
        if (artifact.get('version') != VERSION or not artifact.get('maps', {}).get('GLOBAL')
                or artifact.get('labels_role') != 'train_heldout_groups'
                or artifact.get('dev_test_labels_used') is not False):
            raise ValueError('Invalid group quality calibration artifact')
        self.inner, self.artifact = inner, artifact

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def use_fast_inference(self):
        return self.inner.use_fast_inference

    @use_fast_inference.setter
    def use_fast_inference(self, value):
        self.inner.use_fast_inference = value

    @property
    def use_shared_pool(self):
        return self.inner.use_shared_pool

    @use_shared_pool.setter
    def use_shared_pool(self, value):
        self.inner.use_shared_pool = value

    def predict_main(self, features, facts):
        original = self.inner.predict_main(features, facts)
        paths = action_paths(features)
        mean, sigma = original.quality.mean, original.quality.uncertainty
        calibrated, uncertainty = np.empty_like(mean), np.empty_like(sigma)
        for path in PATHS:
            selected = paths == path
            curve = self.artifact['maps'].get(path, self.artifact['maps']['GLOBAL'])
            def apply(values):
                return np.interp(np.clip(values,0,1),curve['x'],curve['y'])
            calibrated[selected] = apply(mean[selected])
            # Keep the old spread as a floor; this is an empirical conservative
            # score, not a confidence interval with a claimed coverage level.
            uncertainty[selected] = np.maximum(sigma[selected],
                calibrated[selected]-apply(mean[selected]-sigma[selected]))
        return SimpleNamespace(quality=Prediction(calibrated,uncertainty),
            log_latency=original.log_latency, timeout_probability=original.timeout_probability,
            execution_stages=original.execution_stages)


def attach_group_quality_calibration(policy, options, config, checkpoint):
    if not options.get('enabled',False):
        return
    if config.get('main_quality_calibration',{}).get('enabled',False):
        raise ValueError('Cannot apply both legacy and group quality calibration')
    if checkpoint is None:
        raise ValueError('Group calibration requires the loaded frozen checkpoint')
    artifact_path = Path(options['artifact'])
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != options['sha256']:
        raise ValueError('Group calibration artifact hash changed')
    artifact = json.loads(artifact_path.read_text())
    binding = artifact['binding']
    if binding['recall_threshold'] != config['reward']['recall_threshold']:
        raise ValueError('Group calibration Recall mismatch')
    if binding['base_feature_count'] != policy.surrogate.base_feature_count:
        raise ValueError('Group calibration feature dimension mismatch')
    checkpoint = Path(checkpoint).resolve()
    actual = {str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in checkpoint.rglob('*') if p.is_file()}
    if actual != binding['checkpoint_files']:
        raise ValueError('Group calibration checkpoint changed')
    feature_path = checkpoint.parent / 'features.npz'
    if hashlib.sha256(feature_path.read_bytes()).hexdigest() != binding['features_sha256']:
        raise ValueError('Group calibration feature pipeline changed')
    if isinstance(policy.surrogate,GroupCalibratedMainSurrogate):
        raise ValueError('Group calibration already attached')
    policy.surrogate = GroupCalibratedMainSurrogate(policy.surrogate,artifact)
