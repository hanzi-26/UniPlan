"""Independent adapted Dynamic ANN preparation, calibration and diagnostics.

Does not resume the grid or overwrite any existing experiment output. Collection
is explicit; fitting never consumes dev/test rows. No original-system claim.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from vec2plan_bandit.baselines.base import BaselineRequest
from vec2plan_bandit.baselines.dynamic_ann import DynamicANNBaseline
from vec2plan_bandit.baselines.dynamic_ann_statistics import (
    VERSION, DENSITY_DEFINITION, index_inputs, profile_index)
from vec2plan_bandit.evaluation.dynamic_ann import route_diagnostics
from vec2plan_bandit.execution.timing import run_with_deadline, RequestTimer, evaluation_deadline_ns

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def load_backend(args, dataset, config):
    if args.kind == 'arxiv':
        from vec2plan_bandit.queries.loader import QueryRecord
        from vec2plan_bandit.experiments.collector import MatrixCollector
        queries = []
        for row in load_rows(Path(dataset['queries'])):
            row['predicates'], row['candidate_sql'] = tuple(row['predicates']), tuple(row['candidate_sql'])
            queries.append(QueryRecord(**row))
        worker = MatrixCollector(args.output/'runtime', config, read_only=True)
    elif args.kind == 'wiki':
        from vec2plan_bandit.queries.wikipedia_sql_adapter import load_queries
        from vec2plan_bandit.execution.wikipedia_runtime import WikiRuntime
        queries = load_queries(Path(dataset['source']), config['embedding']['model'])
        worker = WikiRuntime(config)
    else:
        from vec2plan_bandit.spider.adapter import Query as SpiderQuery
        from vec2plan_bandit.spider.runtime import Runtime as SpiderRuntime
        assets = json.loads(Path(dataset['catalog']).read_text())['assets']
        metadata = {r['db_id']: r for r in json.loads(Path(dataset['metadata']).read_text())}
        for db, columns in json.loads(Path(dataset['schema']).read_text()).items():
            metadata[db]['_server_columns'] = columns
        assets = [{**a, 'primary_key': ['rowid']} for a in assets]
        queries = []
        for i, row in enumerate(json.loads(Path(dataset['source']).read_text()), 1):
            qid = f'S{i:03d}'
            if qid in dataset.get('allowed_missing_queries', []):
                continue
            q = SpiderQuery(i, row, metadata[row['db_id']], assets)
            q.deterministic_limit_ordering = config.get('runtime',{}).get('deterministic_limit_ordering',False)
            q.query_id = qid
            queries.append(q)
        worker = SpiderRuntime(config)
    if hasattr(worker, 'execute') and args.kind != 'arxiv':
        from vec2plan_bandit.execution.detailed_timing import instrument_runtime
        instrument_runtime(worker)
    return queries, worker


def profile(args, dataset, config):
    started = time.perf_counter()
    import faiss
    faiss.omp_set_num_threads(config.get('runtime', {}).get('cpu_threads', 4))
    queries, worker = load_backend(args, dataset, config)
    assets = {}; seen = set()
    path = args.output/'density.json'
    if path.exists():
        raise ValueError('density artifact already exists; use a new output directory')
    try:
        for q in queries:
            keys = [b.asset_key for b in q.blocks] if hasattr(q, 'blocks') else [q.embedding_column]
            if not set(keys) - seen:
                continue
            if hasattr(worker, 'prepare_query'):
                worker.prepare_query(q)
            for key, index, index_path, _ in index_inputs(worker, q):
                if key not in seen:
                    assets[key] = profile_index(index, index_path, sample_size=args.sample_size,
                        clusters=args.clusters, seed=args.seed)
                    seen.add(key)
                    print(f'profiled {len(assets)} assets', flush=True)
    finally:
        worker.close()
    write(path, dict(version=VERSION, density_definition=DENSITY_DEFINITION, assets=assets,
        query_labels_used=False, source='uniform corpus samples; nearest centroid at request time',
        profile_wall_time_ms=(time.perf_counter()-started)*1000, resources=args.runtime_resources))


def prepared(args, dataset):
    from vec2plan_bandit.experiments.ground_truth_identity import truth_by_query
    folder = args.prepared_dir or Path(dataset['output_dir'])/'prepare'
    stats = json.loads((folder/'statistics.json').read_text()) if (folder/'statistics.json').exists() else None
    if (folder/'prepared.jsonl').exists():
        return truth_by_query(load_rows(folder/'prepared.jsonl'),statistics=stats)
    if (folder/'ground_truth.jsonl').exists() and (folder/'statistics.json').exists():
        stats = json.loads((folder/'statistics.json').read_text())
        return truth_by_query(load_rows(folder/'ground_truth.jsonl'),statistics=stats)
    raise ValueError('prepared ground truth/statistics missing; supply --prepared-dir')


def collect(args, dataset, config):
    started = time.perf_counter()
    queries, worker = load_backend(args, dataset, config)
    split = json.loads(Path(dataset['splits']).read_text())
    truth = prepared(args, dataset)
    source_hash = hashlib.sha256(Path(dataset.get('source', dataset.get('queries'))).read_bytes()).hexdigest()
    split_hash = hashlib.sha256(Path(dataset['splits']).read_bytes()).hexdigest()
    contract = dict(source_sha256=source_hash, splits_sha256=split_hash, role=args.role,
        repeats=args.repeats, recall=args.recall, budget_ms=args.budget_ms, config=config,
        density_sha256=hashlib.sha256((args.output/'density.json').read_bytes()).hexdigest(),
        max_groups=args.max_groups, seed=args.seed, warmup_runs=args.warmup_runs,
        post_params=args.post_params,
        resources=args.runtime_resources,
        input='fresh embedding for each measured path request')
    contract_path = args.output/f'{args.role}-collection-contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Dynamic ANN collection contract changed')
    write(contract_path, contract)
    probe = DynamicANNBaseline(dict(coefficient_mode='paper', density_mode='measured',
        post_params=args.post_params, density_artifact=str(args.output/'density.json'))).fit([])
    selected = [q for q in queries if q.query_id in split[args.role] and
                (not hasattr(q, 'blocks') or q.blocks)]
    groups = sorted({q.group_id for q in selected})
    random.Random(args.seed).shuffle(groups)
    groups = set(groups[:args.max_groups] if args.max_groups else groups)
    selected = [q for q in selected if q.group_id in groups]
    path = args.output/f'{args.role}-measurements.jsonl'
    done = {(r['query_id'], r['path_id'], r['repeat']) for r in load_rows(path)}
    resumed_requests = len(done)
    tasks = [(q, action, repeat) for repeat in range(1, args.repeats + 1)
             for q in selected for action in (probe.pre, probe.post)]
    random.Random(args.seed).shuffle(tasks)
    try:
        preload = getattr(worker, 'preload_statistics', None)
        statistics_preload_ms = preload() if preload is not None else 0.
        write(args.output/f'{args.role}-runtime-startup.json', dict(
            statistics_preload_ms=statistics_preload_ms,
            query_independent_corpus_reservoir_preloaded=preload is not None,
            query_labels_used=False, per_query_statistics_remain_timed=True))
        for q, action, repeat in tasks:
            if (q.query_id, action.path_id, repeat) in done:
                continue
            if q.query_id not in truth:
                raise ValueError(f'missing prepared truth: {q.query_id}')
            if truth[q.query_id].get('source_hash', q.source_hash) != q.source_hash:
                raise ValueError('prepared ground truth belongs to a different query source')
            if hasattr(worker, 'prepare_query'):
                worker.prepare_query(q)
            timer = RequestTimer(args.budget_ms)
            audit = {}
            def call():
                with timer.span('embedding'):
                    vectors = worker.vectors(q) if hasattr(worker, 'vectors') else [worker.embed(q.semantic_text)[0]]
                with timer.span('statistics'):
                    statistics = (worker.estimate_statistics(q) if args.kind == 'arxiv' else
                                  truth[q.query_id]['statistics'])
                    audit['query_statistics'] = probe.request_statistics(worker, q, vectors, statistics)
                with timer.span('execution'):
                    if args.kind == 'arxiv':
                        from vec2plan_bandit.experiments.collector import _vector_literal
                        compiled = worker.compile(q)
                        ids, _, details = worker.execute(action.action_id, q, compiled, vectors[0],
                            _vector_literal(vectors[0]), action,
                            deadline_ns=evaluation_deadline_ns())
                        result = dict(row_ids=[[int(i)] for i in ids], execution_details=details)
                    else:
                        result = worker.execute(q, vectors, action)
                return result
            cancel = getattr(worker, 'cancel_pending', getattr(worker, 'cancel_pending_request', None))
            def guarded():
                try:
                    return call()
                except Exception as error:
                    import requests
                    if isinstance(error, requests.RequestException):
                        raise
                    return dict(row_ids=[], error_type=type(error).__name__, error_message=str(error))
            def measure():
                value = run_with_deadline(guarded, q, args.budget_ms, cancel)
                finish = getattr(worker, 'finish_cancelled_request', None)
                if value.get('timed_out') and finish is not None:
                    finish()
                return value
            for _ in range(args.warmup_runs):
                timer, audit = RequestTimer(args.budget_ms), {}
                measure()
            timer, audit = RequestTimer(args.budget_ms), {}
            row = measure()
            if row.get('timed_out'):
                row['row_ids'] = []
            expected = Counter(map(tuple, truth[q.query_id]['topk_row_ids']))
            actual = Counter(map(tuple, row.get('row_ids', [])))
            count = sum(expected.values()); recall = sum((expected & actual).values()) / count if count else None
            row.update(audit, query_id=q.query_id, group_id=q.group_id, k=q.k,
                path_id=action.path_id, config_id=action.config_id, requested_params=dict(action.params),
                split=args.role, repeat=repeat, execution_ms=timer.spans_ms.get('execution', 0.),
                recall_at_k=recall, result_full_k=len(row.get('row_ids', [])) >= q.k,
                eligible_for_primary_quality=count >= q.k, execution_success=not row.get('error_type') and not row.get('timed_out'))
            with path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            done.add((q.query_id, action.path_id, repeat))
            write(args.output/'status.json', dict(stage='collect', role=args.role, completed=len(done), total=len(tasks)))
    finally:
        worker.close()
    write(args.output/f'{args.role}-collection-cost.json', dict(
        collection_wall_time_ms=(time.perf_counter()-started)*1000,
        measured_requests_this_invocation=len(done)-resumed_requests,
        warmup_requests_this_invocation=(len(done)-resumed_requests)*args.warmup_runs,
        planned_measured_requests=len(tasks), resumed_measured_requests=resumed_requests,
        statistics_preload_ms=statistics_preload_ms,
        includes='runtime initialization, fresh embeddings, statistics, warmups and paired execution'))


def variants(args):
    for coefficients in ('paper', 'calibrated'):
        for density in ('fixed', 'measured'):
            name = f'dynamicann_{coefficients}_{density}'
            options = dict(coefficient_mode=coefficients, density_mode=density, seed=args.seed,
                           post_params=args.post_params, density_artifact=str((args.output/'density.json').resolve()))
            yield name, options


def fit(args):
    rows = load_rows(args.output/'train-measurements.jsonl')
    baselines = []
    for name, options in variants(args):
        baseline = DynamicANNBaseline(options).fit(rows)
        path = args.output/(name+'.json')
        baseline.save(path)
        if options['coefficient_mode'] == 'calibrated':
            options['calibration_artifact'] = str(path.resolve())
        baselines.append(dict(id=name, name='dynamic-ann', config=options))
    write(args.output/'baselines.json', dict(baselines=baselines,
        label='Dynamic ANN cost model (adapted); 2x2 coefficient/density ablation'))
    write(args.output/'controls.json', dict(baselines=[
        dict(id='dynamicann_control_pre',name='pure-pre',config={'params':dict(baseline.pre.params)}),
        dict(id='dynamicann_control_post',name='pure-post',config={'params':dict(baseline.post.params)})]))


def diagnose(args):
    rows = load_rows(args.output/f'{args.role}-measurements.jsonl')
    reports = {}
    specs = json.loads((args.output/'baselines.json').read_text())['baselines']
    # Every variant is evaluated on the same groups held out by the calibrated
    # model, rather than allowing the paper/default controls to see easier rows.
    reference = json.loads((args.output/'dynamicann_calibrated_measured.json').read_text())
    held = set(reference['validation_groups'])
    rows = [r for r in rows if args.role != 'train' or r['group_id'] in held]
    one = {(r['query_id'], r['repeat']): r for r in rows}
    for spec in specs:
        baseline = DynamicANNBaseline(spec['config']).fit([])
        predictions = []
        for row in one.values():
            if 'query_statistics' not in row:
                continue
            query = type('Query', (), dict(k=row['k']))()
            chosen = baseline.select(BaselineRequest(query, np.empty(0), {'PRE':True,'POST':True},
                args.budget_ms, args.budget_ms, query_statistics=row['query_statistics']))
            blocks = row['query_statistics']['dynamicann_blocks']
            predictions.append(dict(query_id=row['query_id'], repeat=row['repeat'], split=row['split'],
                selected_path=chosen.path_id, config_ids={'PRE':baseline.pre.config_id,'POST':baseline.post.config_id},
                estimated_selectivity=float(np.mean([b['estimated_selectivity'] for b in blocks])),
                metadata=baseline.last_selection_metadata))
        reports[spec['id']] = route_diagnostics(rows, predictions,
            recall_threshold=args.recall, budget_ms=args.budget_ms)
    write(args.output/f'{args.role}-diagnostics.json', reports)


def prepare_dynamicann(dataset, dataset_path, status):
    """Resume the training-only Dynamic ANN calibration for one workload."""
    setup = dataset.get('dynamicann_setup')
    if not setup:
        return
    config = json.loads(Path(dataset['experiment_config']).read_text())
    recall = config['reward']['recall_threshold']
    budget_ms = config['reward']['latency_cap_ms']
    output = Path(setup['output'])
    output.mkdir(parents=True, exist_ok=True)
    for stage in ('profile', 'collect', 'fit', 'diagnose'):
        marker = output / f'{stage}.complete.json'
        if marker.exists():
            continue
        status('dynamicann_' + stage)
        command = [sys.executable, '-u', '-m', 'experiments.dynamic_ann',
                   stage, '--dataset', str(dataset_path), '--kind', setup['kind'],
                   '--output', str(output), '--resources', setup['resources'],
                   '--recall', str(recall), '--budget-ms', str(budget_ms), '--repeats', '3']
        subprocess.run(command, cwd=ROOT, check=True)
        marker.write_text(json.dumps({'complete': True, 'at_unix': time.time()}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('profile','collect','fit','diagnose','all'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--kind', choices=('arxiv','wiki','spider'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--role', choices=('train','dev','test'), default='train')
    parser.add_argument('--max-groups', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--sample-size', type=int, default=2048)
    parser.add_argument('--clusters', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recall', type=float, default=.8)
    parser.add_argument('--budget-ms', type=float, default=5000.)
    parser.add_argument('--post-params', default='{}', help='JSON overrides within shared POST parameter bounds')
    parser.add_argument('--resources', type=Path,
        help='reuse client_cpus from the shared resource contract, if present')
    args = parser.parse_args()
    try:
        args.post_params=json.loads(args.post_params)
        if not isinstance(args.post_params,dict):raise ValueError('must be an object')
        DynamicANNBaseline({'post_params':args.post_params})
    except (ValueError,TypeError) as error:
        parser.error(f'invalid --post-params: {error}')
    if args.repeats < 1 or args.warmup_runs < 0 or args.max_groups < 0 or not 0 < args.recall <= 1 or args.budget_ms <= 0:
        parser.error('invalid repeat/group/recall/budget settings')
    if args.stage == 'all' and args.role != 'train':
        parser.error('all includes calibration; use collect/diagnose for dev/test')
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.dataset.read_text())
    config = json.loads(Path(dataset['experiment_config']).read_text())
    resource_contract = args.resources is not None and args.resources.is_file()
    if resource_contract:
        if not hasattr(os, 'sched_getaffinity'):
            parser.error('a CPU-affinity resource contract requires Linux')
        resource = json.loads(args.resources.read_text())
        wanted = set(resource['client_cpus'])
        if not wanted <= set(os.sched_getaffinity(0)):
            parser.error('resource contract CPUs unavailable on this host')
        os.sched_setaffinity(0, wanted)
    args.runtime_resources = dict(client_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        cpu_threads=config.get('runtime',{}).get('cpu_threads',4),
        resource_contract=str(args.resources) if resource_contract else None)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.runtime_resources['cpu_threads']):
        for stage in ('profile','collect','fit','diagnose') if args.stage == 'all' else [args.stage]:
            if stage == 'profile': profile(args,dataset,config)
            elif stage == 'collect': collect(args,dataset,config)
            elif stage == 'fit': fit(args)
            else: diagnose(args)
    write(args.output/'status.json', dict(stage=args.stage+'_complete', role=args.role))


if __name__ == '__main__':
    main()
