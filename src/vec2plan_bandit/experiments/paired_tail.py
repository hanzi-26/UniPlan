"""Cluster bootstrap for paired routing admission; semantic groups are units."""
from collections import defaultdict

import numpy as np


def paired_tail_evidence(fast, main, *, measure, allowance_ms, quantile=.95,
                         repetitions=499, seed=42):
    if repetitions<99 or not .5<=quantile<=1 or allowance_ms<0:
        raise ValueError('invalid paired tail settings')
    def group(rows):
        result=defaultdict(dict)
        for r in rows:
            key=(r['query_id'],r['repeat'])
            if key in result[r['semantic_group']]:raise ValueError('duplicate paired observation')
            result[r['semantic_group']][key]=float(measure(r))
        return result
    a,b=group(fast),group(main)
    if not a or set(a)!=set(b) or any(set(a[g])!=set(b[g]) for g in a):
        raise ValueError('tail bootstrap requires paired semantic groups and repeats')
    names=sorted(a);indices=[];av=[];bv=[]
    for g in names:
        start=len(av)
        for key in sorted(a[g]):av.append(a[g][key]+allowance_ms);bv.append(b[g][key])
        indices.append(np.arange(start,len(av)))
    av,bv=np.asarray(av),np.asarray(bv)
    if not np.isfinite(av).all() or not np.isfinite(bv).all():raise ValueError('nonfinite paired cost')
    rng=np.random.default_rng(seed);tails=[];means=[]
    for _ in range(repetitions):
        selected=np.concatenate([indices[i] for i in rng.integers(len(names),size=len(names))])
        tails.append(float(np.quantile(av[selected],quantile)-np.quantile(bv[selected],quantile)))
        means.append(float(np.mean(av[selected]-bv[selected])))
    lower,upper=np.quantile(tails,[.05,.95])
    return dict(protocol='paired_semantic_cluster_bootstrap',independent_groups=len(names),
        repetitions=repetitions,tail_quantile=quantile,allowance_ms=allowance_ms,
        observed_tail_difference_ms=float(np.quantile(av,quantile)-np.quantile(bv,quantile)),
        tail_difference_lower_ms=float(lower),tail_difference_upper_ms=float(upper),
        mean_difference_upper_ms=float(np.quantile(means,.95)),
        detected_tail_regression=bool(lower>0),proven_tail_improvement=bool(upper<0),
        interpretation='absence of detected regression is not proof of tail improvement')
