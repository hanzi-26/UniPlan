"""Deterministic stratification of indivisible semantic groups by routing facts.

Only query structure and label-free corpus statistics are allowed as strata.
No recall, latency, selected paths or quality outcomes enter this algorithm.
"""
from __future__ import annotations

import numpy as np

ROLES = ('train', 'dev', 'test')


def validate_grouped_split(split, groups):
    if set(split) != set(ROLES):
        raise ValueError('Split must contain train/dev/test')
    expected = {qid for group in groups for qid in group}
    flat = [qid for role in ROLES for qid in split[role]]
    if len(flat) != len(set(flat)) or set(flat) != expected:
        raise ValueError('Split has duplicates, missing or unexpected queries')
    owner = {qid: role for role in ROLES for qid in split[role]}
    if any(len({owner[qid] for qid in group}) != 1 for group in groups):
        raise ValueError('Semantic group crosses split roles')


def stratified_group_split(groups, labels, *, seed=42, restarts=16, steps=2000):
    """Preserve 60/20/20 group allocation and maximize feasible bucket coverage.

    A bucket needs >=4 independent groups for >=2 train, >=1 dev, >=1 test.
    Random swaps preserve role sizes, including the sizes of unequal groups.
    Optimize this coverage, then balance the remaining bucket distributions.
    The search budget and seed are fixed before running any quality measurements.
    """
    groups = [sorted(g) for g in groups]
    if not groups or set(labels) != {q for g in groups for q in g}:
        raise ValueError('A stratum is required for every grouped query')
    keys = sorted(set(labels.values()))
    index = {key: i for i, key in enumerate(keys)}
    matrix = np.zeros((len(groups), len(keys)), dtype=np.int64)
    for i, group in enumerate(groups):
        for qid in group:
            matrix[i, index[labels[qid]]] = 1
    presence = (matrix > 0).astype(np.int64)
    supported = presence.sum(axis=0) >= 4
    sizes = np.array([len(g) for g in groups])
    proportions = np.array([.6, .2, .2])
    targets = sizes.sum() * proportions
    counts = np.zeros(3)
    assignment = np.zeros(len(groups), dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(len(groups)), key=lambda i: -sizes[i])
    for i in order:
        role = int(np.argmax((targets - counts) / targets))
        assignment[i] = role
        counts[role] += sizes[i]
    expected = matrix.sum(axis=0)[None, :] * proportions[:, None]

    def score(a):
        membership = (a[None, :] == np.arange(3)[:, None]).astype(np.int64)
        support = membership @ presence
        covered = (support[0] >= 2) & (support[1] >= 1) & (support[2] >= 1) & supported
        deficits = (np.maximum(0, 2 - support[0, supported]).sum() +
                    np.maximum(0, 1 - support[1:, supported]).sum())
        observed = membership @ matrix
        imbalance = np.sum((observed - expected) ** 2 / (expected + 1.)) / len(keys)
        return 100. * covered.sum() - 10. * deficits - imbalance

    best = assignment.copy()
    best_score = score(best)
    for restart in range(restarts):
        current = assignment.copy() if restart == 0 else rng.permutation(best)
        # Random permutations are safe only for equal-size groups; otherwise
        # retain the original size-preserving assignment and randomize via swaps.
        if not np.all(sizes == sizes[0]):
            current = best.copy()
        current_score = score(current)
        for step in range(steps):
            i, j = rng.integers(0, len(groups), size=2)
            if current[i] == current[j] or sizes[i] != sizes[j]:
                continue
            current[i], current[j] = current[j], current[i]
            candidate = score(current)
            temperature = max(.05, 4. * (1. - step / steps))
            if candidate >= current_score or rng.random() < np.exp(max(-700., (candidate-current_score)/temperature)):
                current_score = candidate
                if candidate > best_score:
                    best, best_score = current.copy(), candidate
            else:
                current[i], current[j] = current[j], current[i]
    split = {role: sorted(q for i,g in enumerate(groups) if best[i] == r for q in g)
             for r,role in enumerate(ROLES)}
    validate_grouped_split(split, groups)
    support = (best[None, :] == np.arange(3)[:, None]).astype(np.int64) @ presence
    audit = {'algorithm': 'semantic-group-routing-stratification-v1', 'seed': seed,
        'restarts': restarts, 'steps': steps, 'label_fields': 'query structure and label-free corpus statistics',
        'quality_or_latency_labels_used': False, 'group_count': len(groups),
        'split_counts': {r: len(split[r]) for r in ROLES},
        'strata': len(keys), 'strata_with_at_least_four_groups': int(supported.sum()),
        'strata_with_two_train_one_dev_one_test': int(((support[0]>=2)&(support[1]>=1)&(support[2]>=1)&supported).sum()),
        'bucket_group_counts': {key:{role:int(support[r,index[key]]) for r,role in enumerate(ROLES)} for key in keys}}
    return split, audit
