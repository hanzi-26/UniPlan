"""Fit a universal optional-acceleration artifact from fresh E2E observations.

This entry point accepts the same measurement schema for every SQL adapter.
It deliberately rejects execution-only Wiki calibration and historical rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vec2plan_bandit.policies.optional_acceleration import fit_acceleration


def calibration_row(row, *, query, split, repeat, mode, description,
                    routing_overhead_ms=None, e2e_includes_routing=False, routing_facts=None):
    """Normalize a scored live request without using query IDs as features.

    Both modes must use fresh embeddings and the original full result semantics.
    If measuring through accelerated_request, pass router_ms as overhead and
    e2e_includes_routing=True. For fixed requests, separately measure descriptor,
    lookup and applicability costs and pass e2e_includes_routing=False.
    """
    fields = ('path_id', 'config_id', 'requested_params', 'e2e_latency_ms',
              'eligible_for_primary_quality', 'result_full_k', 'recall_at_k',
              'execution_success', 'error_type', 'error_message', 'timed_out', 'latency_censored',
              'latency_lower_bound_ms', 'fallback_used', 'router_fallback',
              'fast_error_type', 'fast_error_message', 'router_reason')
    result = {f: row[f] for f in fields if f in row}
    result.update(query_id=query.query_id, semantic_group=query.group_id, split=split,
                  repeat=repeat, mode=mode, routing_description=description, measurement_scope='e2e')
    if routing_facts is not None:
        result['routing_facts'] = routing_facts
    if mode == 'candidate':
        result.update(routing_overhead_ms=routing_overhead_ms, e2e_includes_routing=e2e_includes_routing)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True,
                        help='Expected corpus, splits, encoder, index and frozen-main provenance JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recall-threshold', type=float, default=.8)
    parser.add_argument('--latency-cap-ms', type=float, default=10000.)
    parser.add_argument('--min-relative-gain', type=float, default=.05)
    parser.add_argument('--min-train-groups', type=int, default=2)
    parser.add_argument('--min-dev-groups', type=int, default=1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.measurements.read_text().splitlines() if line.strip()]
    split = json.loads(args.splits.read_text())
    artifact = fit_acceleration(rows, binding=json.loads(args.binding.read_text()),
        train_ids=split['train'], dev_ids=split['dev'], recall_threshold=args.recall_threshold,
        latency_cap_ms=args.latency_cap_ms, min_relative_gain=args.min_relative_gain,
        min_train_groups=args.min_train_groups, min_dev_groups=args.min_dev_groups)
    artifact['calibration_inputs'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (args.measurements, args.splits, args.binding)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to replace an artifact that might be used by a frozen evaluation.
    with args.output.open('x') as handle:
        handle.write(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'rules': len(artifact['rules']),
                      'rejected': len(artifact['rejected'])}))


if __name__ == '__main__':
    main()
