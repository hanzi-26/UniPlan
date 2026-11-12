"""Dynamic ANN cost model (adapted): paper Section 4.1 over shared PRE/POST.

Reference: https://doi.org/10.1038/s41598-026-58877-1.
Does not reproduce SwitchOp, runtime re-estimation or connectivity profiles.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from .pure_paths import DEFAULTS
from vec2plan_bandit.paths.registry import PRE, POST, make_execution_config, pre_configuration_key

VERSION = 'dynamicann-cost-model-adapted-v1'
COEFFICIENTS = ('alpha', 'beta', 'gamma', 'scan_io_per_row', 'ann_overhead', 'scan_overhead')
PAPER_COEFFICIENTS = dict(zip(COEFFICIENTS, (1.05, .42, .3, .1, 0., 0.)))


def finite(value, name, low=0., high=math.inf):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'missing/invalid Dynamic ANN {name}') from error
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'invalid Dynamic ANN {name}')
    return value


def cost_terms(stats, k, density_mode, fixed_density):
    ann, scan, used = np.zeros(6), np.zeros(6), []
    for block in stats.get('dynamicann_blocks') or [stats]:
        n = finite(block.get('total_rows', block.get('corpus_size')), 'corpus size', 1)
        d = finite(block.get('embedding_dimension'), 'dimension', 1)
        s = finite(block.get('estimated_selectivity'), 'selectivity', 0, 1)
        rho = finite(block.get('density'), 'measured density', 0, 1) if density_mode == 'measured' else fixed_density
        bk = finite(block.get('k', k), 'k', 1)
        kp = bk * (1.5 - rho)  # Abstract and Section 4.1 convention.
        ann[:3] += (math.log2(max(n, 2)), kp * d, kp * (1 - s))
        scan[1:4] += (s * n * d, s * n, n)
        used.append(dict(total_rows=n, embedding_dimension=d, estimated_selectivity=s,
                         density=rho, k=bk, expected_candidates=kp,
                         density_source=block.get('density_source','provided_statistic')
                            if density_mode == 'measured' else 'fixed_ablation'))
    ann[4], scan[5] = 1., 1.  # One shared-backend fixed cost per request.
    return ann, scan, used


class DynamicANNBaseline:
    name = 'dynamic-ann'
    display_name = 'Dynamic ANN cost model (adapted)'
    requires_context = False

    def __init__(self, config):
        self.config = dict(config)
        unknown=set(config)-{'coefficient_mode','density_mode','fixed_density','tolerance_factor',
            'validation_fraction','seed','post_params','density_artifact','calibration_artifact'}
        if unknown:
            raise ValueError(f'unknown Dynamic ANN options: {sorted(unknown)}')
        self.coefficient_mode = config.get('coefficient_mode', 'calibrated')
        self.density_mode = config.get('density_mode', 'measured')
        if self.coefficient_mode not in {'paper', 'calibrated'} or self.density_mode not in {'fixed', 'measured'}:
            raise ValueError('invalid Dynamic ANN coefficient/density mode')
        self.fixed_density = finite(config.get('fixed_density', .5), 'fixed_density', 0, 1)
        self.tolerance = finite(config.get('tolerance_factor', .9), 'tolerance', 1e-12, 1)
        self.validation_fraction = finite(config.get('validation_fraction', .2), 'validation_fraction', .01, .5)
        self.seed = config.get('seed', 42)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Dynamic ANN seed must be a nonnegative integer')
        self.pre = make_execution_config(PRE, DEFAULTS[PRE])
        self.post = make_execution_config(POST, {**DEFAULTS[POST], **config.get('post_params', {})})
        self.coefficients = dict(PAPER_COEFFICIENTS)
        self.fit_audit, self.last_selection_metadata = {}, {}
        self._fit, self._catalog = self.coefficient_mode == 'paper', None

    def signature(self):
        path = self.config.get('density_artifact')
        return dict(version=VERSION, pre_config_id=self.pre.config_id, post_config_id=self.post.config_id,
            density_mode=self.density_mode, fixed_density=self.fixed_density, tolerance_factor=self.tolerance,
            density_artifact_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)

    def _paired_rows(self, observations):
        paired = defaultdict(lambda: defaultdict(list))
        query_groups = {}
        for row in observations:
            if row.get('split') != 'train':
                raise ValueError('Dynamic ANN fitting requires explicitly marked train rows only')
            group = str(row.get('group_id', row['query_id']))
            if str(row['query_id']) in query_groups and query_groups[str(row['query_id'])] != group:
                raise ValueError('inconsistent Dynamic ANN semantic group')
            query_groups[str(row['query_id'])] = group
            if row.get('path_id') not in {PRE, POST}:
                continue
            wanted = self.pre if row['path_id'] == PRE else self.post
            if make_execution_config(row['path_id'], row.get('requested_params')).config_id != wanted.config_id:
                continue
            if (row.get('timed_out') or row.get('latency_censored') or row.get('error_type')
                    or row.get('execution_success') is False or row.get('fallback_used')):
                continue
            finite(row.get('execution_ms'), 'measured execution_ms')
            cost_terms(row.get('query_statistics') or {}, row['k'], self.density_mode, self.fixed_density)
            paired[str(row['query_id'])][row['path_id']].append(row)
        return [r for paths in paired.values() if paths[PRE] and paths[POST]
                for rows in paths.values() for r in rows]

    def fit(self, observations):
        started = time.perf_counter()
        if self.coefficient_mode == 'paper':
            self.fit_audit = dict(mode='paper', coefficient_units='reported_paper_cost_units',
                coefficients=self.coefficients, signature=self.signature(), fitted_observations=0)
            return self
        path = self.config.get('calibration_artifact')
        if path:
            data = json.loads(Path(path).read_text())
            if (data.get('signature') != self.signature() or data.get('coefficient_units') != 'milliseconds'
                    or data.get('training_only') is not True
                    or set(data.get('calibration_groups', [])) & set(data.get('validation_groups', []))):
                raise ValueError('Dynamic ANN calibration artifact/backend/density mismatch')
            self.coefficients = {k: finite(data['coefficients'][k], k) for k in COEFFICIENTS}
            self.fit_audit, self._fit = data, True
            return self
        observations = list(observations)
        rows = self._paired_rows(observations)
        groups = sorted({str(r.get('group_id', r['query_id'])) for r in rows})
        if len(groups) < 5:
            raise ValueError('Dynamic ANN needs at least five paired train groups for fit/validation')
        np.random.default_rng(self.seed).shuffle(groups)
        held = set(groups[:max(1, math.ceil(len(groups) * self.validation_fraction))])
        train = [r for r in rows if str(r.get('group_id', r['query_id'])) not in held]
        validation = [r for r in rows if str(r.get('group_id', r['query_id'])) in held]
        def terms(r):
            return cost_terms(r['query_statistics'], r['k'], self.density_mode, self.fixed_density)[r['path_id'] == PRE]
        x = np.array([terms(r) for r in train]); y = np.array([r['execution_ms'] for r in train])
        scale = np.maximum(np.linalg.norm(x, axis=0), 1e-12)
        from scipy.optimize import nnls
        beta, residual = nnls(x / scale, y, maxiter=10000)
        beta /= scale
        self.coefficients, self._fit = dict(zip(COEFFICIENTS, map(float, beta))), True
        errors = [abs(float(terms(r) @ beta) - r['execution_ms']) for r in validation]
        self.fit_audit = dict(mode='calibrated', coefficient_units='milliseconds', signature=self.signature(),
            coefficients=self.coefficients, seed=self.seed, training_only=True,
            target='execution_ms: filtering, search, ID mapping and result assembly; common embedding excluded',
            paired_train_queries=len({r['query_id'] for r in rows}),
            input_observations=len(observations), unused_or_failed_observations=len(observations)-len(rows),
            measurement_sha256=hashlib.sha256(json.dumps([
                [r['query_id'],r.get('group_id'),r['path_id'],r['requested_params'],r['k'],
                 r['query_statistics'],r['execution_ms']] for r in rows],sort_keys=True).encode()).hexdigest(),
            calibration_groups=sorted(set(groups) - held), validation_groups=sorted(held),
            fitted_observations=len(train), validation_observations=len(validation),
            design_rank=int(np.linalg.matrix_rank(x / scale)), coefficient_count=6,
            residual_l2_ms=float(residual), validation_mae_ms=float(np.mean(errors)),
            fit_time_ms=(time.perf_counter() - started) * 1000,
            limitations=['adapted PRE/POST backends', 'no runtime switching',
                         'normalized variance density proxy', 'no connectivity profile'])
        return self

    def request_statistics(self, worker, query, vectors, statistics):
        from .dynamic_ann_statistics import request_statistics
        return request_statistics(self, worker, query, vectors, statistics)

    def select(self, request):
        if not self._fit:
            raise RuntimeError('Dynamic ANN cost model must be calibrated/frozen before evaluation')
        if request.remaining_budget_ms <= 0:
            raise TimeoutError('Dynamic ANN request budget exhausted')
        ann, scan, blocks = cost_terms(request.query_statistics or {}, request.query.k,
                                      self.density_mode, self.fixed_density)
        beta = np.array([self.coefficients[k] for k in COEFFICIENTS])
        ac, sc = float(ann @ beta), float(scan @ beta)
        pre = request.available_paths.get(PRE, False)
        if request.available_pre_configurations is not None:
            pre &= pre_configuration_key(self.pre.params) in request.available_pre_configurations
        post = request.available_paths.get(POST, False)
        if not pre and not post:
            raise RuntimeError('Dynamic ANN has no operational PRE/POST path')
        if not pre:
            chosen, reason = self.post, 'pre_unavailable'
        elif not post:
            chosen, reason = self.pre, 'post_unavailable'
        elif sc < self.tolerance * ac:
            chosen, reason = self.pre, 'scan_cost_below_tolerance_band'
        else:
            chosen = self.post
            reason = 'ann_cost_below_tolerance_band' if ac < self.tolerance * sc else 'within_band_prefer_ann'
        self.last_selection_metadata = {'dynamicann': dict(version=VERSION, display_name=self.display_name,
            coefficient_mode=self.coefficient_mode, density_mode=self.density_mode,
            coefficient_units='milliseconds' if self.coefficient_mode == 'calibrated' else 'reported_paper_cost_units',
            ann_cost=ac, scan_cost=sc, blocks=blocks, reason=reason,
            selected_path=chosen.path_id, selected_config_id=chosen.config_id, runtime_switching=False)}
        return chosen

    def save(self, path):
        if not self._fit:
            raise RuntimeError('cannot save an unfit Dynamic ANN model')
        if not self.fit_audit:
            self.fit([])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.fit_audit, indent=2) + '\n')


def dynamic_ann_factory(config):
    return DynamicANNBaseline(config)
