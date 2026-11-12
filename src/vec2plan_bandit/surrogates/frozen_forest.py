"""Opt-in main-policy inference; shared backends/baselines never call this.

Export the existing numeric symmetric trees without retraining, share identical
float32 comparisons across all heads, and evaluate the full ensemble in one
single-thread native call. Startup cost and model/kernel hashes are recorded.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np


class FrozenForest:
    def __init__(self, models, feature_count, directory):
        started = time.perf_counter()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_count = feature_count
        self.model_count = len(models)
        tests, splits, depths, leaves = {}, [], [], []
        split_starts, leaf_starts, model_starts = [], [], [0]
        scales, biases, model_hashes = [], [], []
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            for index, model in enumerate(models):
                if not type(model).__module__.startswith('catboost'):
                    raise ValueError('native inference requires real CatBoost models')
                path = Path(tmp) / f'{index}.json'
                model.save_model(str(path), format='json')
                model_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                data = json.loads(path.read_text())
                info = data['features_info']
                if any(info.get(k) for k in ('categorical_features', 'text_features', 'embedding_features', 'ctrs')):
                    raise ValueError('native inference supports numeric symmetric trees only')
                mapping = {f['feature_index']: f['flat_feature_index'] for f in info['float_features']}
                trees = data.get('oblivious_trees')
                if trees is None:
                    raise ValueError('native inference requires symmetric trees')
                for tree in trees:
                    tree_splits = tree.get('splits') or []
                    if len(tree_splits) > 16 or len(tree['leaf_values']) != 2 ** len(tree_splits):
                        raise ValueError('unsupported tree depth or output dimension')
                    split_starts.append(len(splits))
                    leaf_starts.append(len(leaves))
                    depths.append(len(tree_splits))
                    leaves.extend(tree['leaf_values'])
                    for test in tree_splits:
                        if test['split_type'] != 'FloatFeature':
                            raise ValueError('unsupported categorical tree split')
                        feature = mapping[test['float_feature_index']]
                        if not 0 <= feature < feature_count:
                            raise ValueError('model split outside feature schema')
                        border = np.float32(test['border'])
                        key = (feature, border.tobytes())
                        if key not in tests:
                            tests[key] = len(tests)
                        splits.append(tests[key])
                scale, bias = data['scale_and_bias']
                if len(bias) != 1:
                    raise ValueError('expected scalar CatBoost head')
                scales.append(scale)
                biases.append(bias[0])
                model_starts.append(len(depths))
        features = [key[0] for key in tests]
        borders = [np.frombuffer(key[1], dtype=np.float32)[0] for key in tests]
        self.arrays = [np.ascontiguousarray(a, dtype=d) for a, d in (
            (features, np.int32), (borders, np.float32), (model_starts, np.int32),
            (split_starts, np.int32), (leaf_starts, np.int32), (depths, np.int32),
            (splits, np.int32), (leaves, np.float64), (scales, np.float64), (biases, np.float64))]
        self.comparisons = len(tests)
        source = Path(__file__).with_suffix('.cpp')
        compiler = shutil.which('g++')
        if not compiler:
            raise RuntimeError('g++ is required for opt-in main-policy native inference')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        library = directory / f'forest-{digest[:16]}.so'
        compile_started = time.perf_counter()
        if not library.exists():
            temporary = library.with_suffix('.tmp.so')
            subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                            '-ffp-contract=off', str(source), '-o', str(temporary)],
                           check=True, capture_output=True, text=True)
            temporary.replace(library)
        self.library = ctypes.CDLL(str(library.resolve()))
        self.function = self.library.predict_frozen_forest
        self.function.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 11
        self.function.restype = ctypes.c_int
        self.audit = dict(engine='main-frozen-forest-v1', threads=1, models=len(models),
            trees=len(depths), unique_comparisons=len(tests), model_json_sha256=model_hashes,
            kernel_sha256=digest, library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
            compile_or_load_ms=(time.perf_counter()-compile_started)*1000,
            startup_ms=(time.perf_counter()-started)*1000,
            weights_unchanged=True, startup_excluded_from_request_latency=True)
        (directory / 'manifest.json').write_text(json.dumps(self.audit, indent=2) + '\n')

    def predict(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.feature_count or not np.all(np.isfinite(x)):
            raise ValueError('invalid native prediction feature matrix')
        with np.errstate(over='ignore'):
            column_major = np.ascontiguousarray(x.T, dtype=np.float32)
        if not np.all(np.isfinite(column_major)):
            raise ValueError('prediction features overflow float32')
        output = np.empty((self.model_count, len(x)), dtype=np.float64)
        result = self.function(column_major.ctypes.data, len(x), self.feature_count, self.model_count,
            self.comparisons, *(a.ctypes.data for a in self.arrays), output.ctypes.data)
        if result or not np.all(np.isfinite(output)):
            raise RuntimeError('native forest prediction failed')
        return output
