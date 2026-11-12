"""Explicit best-effort recovery when predicted constraints are infeasible.

This never certifies probability one or changes a Recall/probability threshold.
Only already operational configurations of the existing paths are considered.
"""
import numpy as np


def recovery_pool(candidates, quality_mean, quality_sigma, latency_upper, timeout_upper,
                  *, probability_min, latency_cap_ms, options):
    z = options['quality_overlap_sigma']
    if not np.isfinite(z) or not 0 <= z <= 4:
        raise ValueError('invalid calibrated quality overlap')
    exact = bool(options.get('exact_recall', False))
    allowed = np.asarray([c.path_id == 'PRE' if exact else True for c in candidates])
    if not allowed.any():
        raise ValueError('exact recovery requires an operational PRE configuration')
    timely = allowed & (latency_upper <= latency_cap_ms)
    pool = timely if timely.any() else allowed
    # A timeout model supplies a ranking, not an unearned certainty certificate.
    # Select the lowest-risk stratum before cost if nothing is predicted timely.
    if not timely.any():
        pool &= timeout_upper <= np.min(timeout_upper[pool]) + 1e-12
    if not exact:
        lower = np.clip(quality_mean - quality_sigma, 0., 1.)
        upper = np.clip(quality_mean + z * quality_sigma, 0., 1.)
        pool &= upper >= np.max(lower[pool])
    return pool, dict(reason='exact_contract_cost_recovery' if exact else 'calibrated_quality_overlap_recovery',
        predicted_constraints_satisfied=False, probability_threshold=probability_min,
        probability_one_certified=False, quality_overlap_sigma=z,
        retrieval_semantics='exact conditional on successful PRE completion' if exact else 'empirical prediction only',
        predicted_deadline_candidate=bool(timely.any()), candidates=int(pool.sum()))
