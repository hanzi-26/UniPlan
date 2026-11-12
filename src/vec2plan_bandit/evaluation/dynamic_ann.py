"""Offline route diagnostics. Held-out measurements never enter baseline.select."""
from collections import defaultdict
import math
import numpy as np


def selectivity_band(s):
    return 'unknown' if s is None else ('<=0.01' if s <= .01 else '<=0.1' if s <= .1 else '<=0.5' if s <= .5 else '>0.5')


def route_diagnostics(measurements, predictions, *, recall_threshold=.8, budget_ms=5000.):
    paired = defaultdict(dict)
    for row in measurements:
        key = (row.get('split'), row['query_id'], row.get('repeat', 1))
        paired[key][row['path_id']] = row
    records, incomplete = [], 0
    for prediction in predictions:
        key = (prediction.get('split'), prediction['query_id'], prediction.get('repeat', 1))
        rows = paired.get(key, {})
        if not {'PRE', 'POST'} <= rows.keys():
            incomplete += 1
            continue
        expected = prediction.get('config_ids', {})
        if expected and any(rows[p].get('config_id') != expected[p] for p in ('PRE', 'POST')):
            incomplete += 1
            continue
        def valid(row):
            return (row.get('eligible_for_primary_quality', False) and row.get('result_full_k', False)
                and row.get('recall_at_k') is not None and row['recall_at_k'] >= recall_threshold
                and not row.get('timed_out') and not row.get('error_type')
                and not row.get('latency_censored') and not row.get('fallback_used')
                and row.get('e2e_latency_ms', math.inf) <= budget_ms)
        eligible = {p: r for p, r in rows.items() if valid(r)}
        selected = prediction['selected_path']; chosen = rows[selected]
        record = dict(query_id=key[1], split=key[0], repeat=key[2], selected_path=selected,
            selectivity_band=selectivity_band(prediction.get('estimated_selectivity')),
            selected_quality_pass=valid(chosen), oracle_path=None, route_match=None,
            eligible_for_primary_quality=bool(chosen.get('eligible_for_primary_quality', False)),
            execution_regret_ms=None, execution_slowdown=None)
        if eligible:
            fastest = min(eligible, key=lambda p: eligible[p]['execution_ms'])
            best = eligible[fastest]['execution_ms']
            record.update(oracle_path=fastest, route_match=selected == fastest)
            if valid(chosen):
                record.update(execution_regret_ms=max(0., chosen['execution_ms'] - best),
                    execution_slowdown=chosen['execution_ms'] / max(best, 1e-9))
        records.append(record)
    def aggregate(rows):
        matches = [r['route_match'] for r in rows if r['route_match'] is not None]
        regrets = [r['execution_regret_ms'] for r in rows if r['execution_regret_ms'] is not None]
        quality = [r['selected_quality_pass'] for r in rows if r['eligible_for_primary_quality']]
        return dict(requests=len(rows), primary_quality_requests=len(quality),
            quality_pass_rate=float(np.mean(quality)) if quality else None,
            excluded_insufficient_truth=len(rows)-len(quality),
            no_quality_qualified_control=sum(r['oracle_path'] is None for r in rows),
            oracle_eligible_requests=len(matches), route_match_rate=float(np.mean(matches)) if matches else None,
            mean_execution_regret_ms=float(np.mean(regrets)) if regrets else None,
            p95_execution_regret_ms=float(np.percentile(regrets,95)) if regrets else None,
            confusion_matrix={f'{actual}->{predicted}':sum(r['oracle_path']==actual and
                r['selected_path']==predicted for r in rows) for actual in ('PRE','POST') for predicted in ('PRE','POST')},
            selected_paths={p:sum(r['selected_path']==p for r in rows) for p in ('PRE','POST')})
    return dict(scope='offline diagnostics; fastest measured quality-qualified PRE/POST control; not a deployable oracle',
        recall_threshold=recall_threshold, budget_ms=budget_ms, incomplete_control_pairs=incomplete,
        overall=aggregate(records), by_selectivity={b:aggregate([r for r in records if r['selectivity_band']==b])
            for b in sorted({r['selectivity_band'] for r in records})}, records=records)
