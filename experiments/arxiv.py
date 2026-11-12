"""arXiv preparation, training, calibration, and benchmark workflow."""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
from vec2plan_bandit.experiments import run as r
from vec2plan_bandit.experiments.wikipedia import write, read_rows, append
from vec2plan_bandit.experiments.optional_acceleration import calibration_row
from vec2plan_bandit.experiments.workflow import _outcome
from vec2plan_bandit.execution.timing import run_with_deadline
from vec2plan_bandit.paths.registry import make_execution_config
from vec2plan_bandit.policies.optional_acceleration import OptionalAccelerationRouter, VERSION, FIELDS, describe_query, fit_acceleration



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actions(config=None):
    from vec2plan_bandit.experiments.wiki_optional_acceleration import candidate_actions
    from vec2plan_bandit.policies.main_lightweight import MainNativePGConfig
    paths = (config or {}).get('acceleration_calibration',{}).get('candidate_paths',['FAISS'])
    if not paths or set(paths)-{'PRE','POST','FAISS','ACORN','PG_NATIVE_SQL'}:
        raise ValueError('invalid main acceleration candidate paths')
    return [action if action is not None else MainNativePGConfig() for action in candidate_actions()
            if (action.path_id if action is not None else 'PG_NATIVE_SQL') in paths]


def calibration_router(action, description, binding, recall_threshold=.8):
    rule = {'key': description['key'], 'path_id': action.path_id, 'config_id': action.config_id,
            'params': dict(action.params), 'calibrated_e2e_max_ms': 0.}
    for field in FIELDS:
        rule[field+'_min'] = rule[field+'_max'] = description[field]
    return OptionalAccelerationRouter({'version': VERSION, 'binding': binding, 'rules': [rule], 'fit_protocol': {'recall_threshold': recall_threshold, 'scope':'temporary forced calibration action'}}, binding=binding)


def calibration_tasks(records, stats, split, asset_sizes, config, seed):
    """Plan train/dev work and make every skipped query visible before startup."""
    by_id = {q.query_id:q for q in records}
    tasks, coverage = [], {}
    candidates = actions(config)
    repeats = config['acceleration_calibration']['repeats']
    if type(repeats) is not int or repeats < 1:
        raise ValueError('calibration repeats must be a positive integer')
    for role in ('train','dev'):
        accepted, skipped = [], []
        for qid in split[role]:
            query = by_id[qid]
            desc = describe_query(query,stats.get(qid),corpus_size=asset_sizes[query.embedding_column])
            if desc is None:
                skipped.append(qid)
                continue
            accepted.append(qid)
            for repeat in range(1,repeats+1):
                tasks.append((role,qid,repeat,None,desc))
                tasks.extend((role,qid,repeat,a,desc) for a in candidates)
        coverage[role] = dict(total=len(split[role]),supported=len(accepted),skipped_query_ids=skipped)
    random.Random(seed).shuffle(tasks)
    return tasks, coverage


def calibrate(dataset, config, output, status):
    split = r._read_json(ROOT/dataset['splits'])
    records = r._queries(ROOT/dataset['queries'])
    by_id = {q.query_id:q for q in records}
    folder = Path(dataset['checkpoint'])
    pipeline, policy, _ = r.load_bundle(folder, cap_ms=config['training_latency_cap_ms'],
        train_ids=split['train'], expected={'snapshot_id': config['snapshot_id'],
            'embedding': config['embedding'], 'recall_threshold': config['reward']['recall_threshold'],
            'training_source_hashes': {q.query_id:q.source_hash for q in records if q.query_id in split['train']}})
    policy.quality_probability_min = config['max_ucb']['quality_probability_min']
    from vec2plan_bandit.experiments.main_inference import configure_main_inference
    configure_main_inference(policy, config, output, checkpoint=folder,calibrating=True)
    frozen = copy.deepcopy(policy.state_dict())
    stats = r._read_json(output/'prepare/statistics.json')
    truth = r._ground_truth(output/'prepare/ground_truth.jsonl')
    import numpy as np
    asset_sizes = {}
    for column in {q.embedding_column for q in records}:
        path = Path(config['faiss_assets'][column]['article_ids']).expanduser()
        if not path.is_absolute():
            path = ROOT/path
        values = np.load(path,mmap_mode='r',allow_pickle=False)
        if values.ndim != 1:
            raise ValueError('calibration requires a one-dimensional logical article-ID map')
        asset_sizes[column] = len(values)
    metadata = Path(config['metadata']['snapshot_dir'])
    binding = {'queries':sha(ROOT/dataset['queries']), 'splits':sha(ROOT/dataset['splits']),
        'encoder':config['embedding'], 'snapshot_id':config['snapshot_id'],
        'metadata_manifest':sha(metadata/'metadata-manifest.json'),
        'acorn_receipt':sha(metadata/'acorn-enrichment.json'),
        'article_ids':{column:sha(ROOT/config['faiss_assets'][column]['article_ids']) for column in asset_sizes},
        'frozen_bundle':sha(folder/'bundle.json'), 'config':sha(ROOT/dataset['experiment_config'])}
    options = config['acceleration_calibration']
    contract = {'binding':binding, 'options':options, 'actions':[dict(path_id=a.path_id,params=dict(a.params)) for a in actions(config)],
                'test_used_for_calibration':False}
    contract_path = output/'calibration/contract.json'
    if contract_path.exists() and r._read_json(contract_path) != contract:
        raise ValueError('Calibration inputs changed; use a new output directory')
    write(contract_path, contract)
    tasks,coverage = calibration_tasks(records,stats,split,asset_sizes,config,dataset['seed'])
    write(output/'calibration/coverage.json',coverage)
    if any(not coverage[role]['supported'] for role in ('train','dev')):
        raise ValueError('Calibration has no supported train/dev queries; inspect calibration/coverage.json and SQL/statistics adapters')
    journal=output/'calibration/measurements.jsonl'
    done={(v['split'],v['query_id'],v['repeat'],v.get('config_id')) for v in read_rows(journal)}
    worker=r._collector(output/'calibration',config)
    cap=config['reward']['latency_cap_ms']
    try:
        for column in asset_sizes:
            worker.preload_faiss(column)
        startup_ms = worker.preload_statistics()
        write(output/'calibration/runtime-startup.json',{'statistics_preload_ms':startup_ms})
        for role,qid,repeat,action,desc in tasks:
            cid=action.config_id if action else None
            if (role,qid,repeat,cid) in done: continue
            status('calibrating',completed=len(done),total=len(tasks),split=role,query_id=qid)
            query=by_id[qid]
            router=calibration_router(action,desc,binding,config['reward']['recall_threshold']) if action else None
            def measure():
                row=run_with_deadline(lambda: _outcome(lambda: r.run_adaptive_request(
                    query=query,collector=worker,feature_pipeline=pipeline,policy=policy,
                    latency_cap_ms=cap,statistics=stats,exploration=False,acceleration_router=router),query,cap),
                    query,cap,worker.cancel_pending_request)
                if row.get('timed_out'): worker.finish_cancelled_request()
                row=r._adjudicate(row,truth,cap,config['reward']['recall_threshold'])
                row['execution_success']=not row.get('error_type') and not row.get('timed_out')
                return row
            for _ in range(config['measurement']['warmup_runs']): measure()
            row=measure()
            facts = None
            if config.get('main_lightweight_selector',{}).get('collect_facts',False):
                from vec2plan_bandit.policies.main_lightweight import routing_facts
                facts = routing_facts(query,stats.get(qid),embedding_dimension=config['embedding']['dimension'],
                                      corpus_size=desc['n'])
            observation=calibration_row(row,query=query,split=role,repeat=repeat,
                mode='candidate' if action else 'main',description=desc,
                routing_overhead_ms=row.get('router_ms',0.),e2e_includes_routing=True,routing_facts=facts)
            if action:
                observation.update(config_id=cid,path_id=action.path_id,requested_params=dict(action.params),
                    execution_success=bool(row.get('router_fast_accepted')),
                    router_fallback=row.get('router_fallback',False))
            else: observation.pop('config_id',None)
            append(journal,observation)
            done.add((role,qid,repeat,cid))
        if policy.state_dict()!=frozen: raise ValueError('Calibration mutated frozen policy')
        artifact=fit_acceleration(read_rows(journal),binding=binding,train_ids=split['train'],dev_ids=split['dev'],
            recall_threshold=config['reward']['recall_threshold'],latency_cap_ms=cap,**{k:options[k] for k in ('min_relative_gain','min_train_groups','min_dev_groups')})
        write(output/'router.json',artifact)
        observations = read_rows(journal)
        write(output/'calibration/summary.json',{'measurements':len(done),'rules':len(artifact['rules']),
            'request_errors':sum(bool(v.get('error_type')) for v in observations),
            'candidate_execution_failures':sum(v['mode']=='candidate' and not v.get('execution_success') for v in observations),
            'coverage':coverage,
            'test_used_for_calibration':False,'quality_evidence':'empirical; not a probability guarantee'})
        from vec2plan_bandit.experiments.main_inference import fit_auto_main_selector
        fit_auto_main_selector(config,output,split)
        evaluation=copy.deepcopy(config)
        evaluation['optional_acceleration']={'enabled':True,'artifact':str(output/'router.json'),
            'binding':binding,'corpus_size_by_column':asset_sizes}
        if config.get('main_selector_calibration',{}).get('enabled'):
            evaluation['optional_acceleration']['enabled']=False
        write(output/'evaluation-config.json',evaluation)
    finally: worker.close()


def report(output):
    rows=[]
    for role in ('dev','test'):
        summary=r._read_json(output/role/'benchmark_summary.json')
        measured=read_rows(output/role/'benchmark_live.jsonl')
        for method,metrics in summary.items():
            selected=[v for v in measured if v['method_id']==method]
            recalls=[v['recall_at_k'] for v in selected if v.get('recall_at_k') is not None]
            metrics.update(mean_recall_at_k=sum(recalls)/len(recalls) if recalls else None,
                error_count=sum(bool(v.get('error_type')) for v in selected),
                timeout_count=sum(bool(v.get('timed_out')) for v in selected),
                full_k_rate=sum(bool(v.get('result_full_k')) for v in selected)/len(selected),
                router_hit_rate=sum(bool(v.get('router_hit')) for v in selected)/len(selected))
        rows.extend({'split':role,'method':method,**metrics} for method,metrics in summary.items())
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'comparison.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    write(output/'results.json',rows)
    cfg=r._read_json(output/'evaluation-config.json')
    lines=['# arXiv optimized experiment', '',
        f"Recall={cfg['reward']['recall_threshold']}; main predicted quality probability={cfg['max_ucb']['quality_probability_min']}; E2E budget={cfg['reward']['latency_cap_ms']}ms.",
        'Fixed train/dev/test=565/185/200. Five repeats and one warmup, fresh embeddings.',
        'Train chooses FAISS parameters; dev enables/rejects with quality and >5% gain checks. Test is frozen.',
        'Dev is calibration data. Fast-route quality evidence is empirical, not a probability guarantee.', '',
        '| Split | Method | Quality pass | Joint success | Mean E2E ms |','|---|---|---:|---:|---:|']
    if cfg.get('threshold_sweep',{}).get('predictor_weights_reused'):
        lines[2:2]=[f"Frozen predictor trained at probability threshold {cfg['threshold_sweep']['source_training_probability']}; this is a deployment-threshold sweep, not probability-specific retraining. Main train/dev routing calibration and all baseline dev/test benchmarks are measured afresh. Provenance: model-reuse.json and threshold-sweep.json.", '']
    for v in rows:
        lines.append(f"| {v['split']} | {v['method']} | {v.get('quality_pass_rate')} | {v.get('joint_success_rate')} | {v.get('e2e_latency_mean_ms')} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def stage_command(dataset, stage, override=None):
    config_path = ROOT / (override or dataset["experiment_config"])
    config = json.loads(config_path.read_text())
    output = ROOT / dataset["output_dir"]
    benchmark = stage in ("dev", "test")
    roles = [stage] if benchmark else ["train"] if stage in ("collect", "train") else ["train", "dev", "test"]
    cap = config["reward"]["latency_cap_ms"]
    if stage in ("collect", "train"):
        cap = config.get("training_latency_cap_ms", cap)
    command = [
        sys.executable, "-u", "-m", "vec2plan_bandit.experiments.run",
        "benchmark" if benchmark else stage,
        "--config", str(config_path),
        "--query-source", str(ROOT / dataset["queries"]),
        "--split-json", str(ROOT / dataset["splits"]),
        "--output-dir", str(output / stage),
        "--seed", str(dataset["seed"]),
        "--roles", *roles,
        "--repeats", str(dataset["repeats"] if benchmark else 1),
        "--latency-cap-ms", str(cap),
    ]
    prepared = output / "prepare"
    if stage in ("collect", "train") or benchmark:
        command += ["--ground-truth-jsonl", str(prepared / "ground_truth.jsonl")]
    if stage == "train" or benchmark:
        training = output / ("train/shared_training.jsonl" if benchmark else "collect/configuration_matrix.jsonl")
        command += [
            "--training-matrix", str(training),
            "--feature-embeddings", str(prepared / "query_embeddings.npz"),
            "--query-statistics", str(prepared / "statistics.json"),
        ]
    if benchmark:
        command += ["--baseline-config", str(ROOT / dataset["baselines"]),
                    "--checkpoint", str(ROOT / dataset["checkpoint"])]
    return output / stage, command


def main():
    # Local services must bypass the workstation's outbound HTTP proxy.
    for key in ('NO_PROXY', 'no_proxy'):
        os.environ[key]=','.join(filter(None,[os.environ.get(key,''),'127.0.0.1','localhost','::1']))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','preflight','all'],default='all',nargs='?')
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--train-config',type=Path)
    args=parser.parse_args()
    dataset_path=args.dataset
    dataset=r._read_json(dataset_path)
    train_config=args.train_config or ROOT/dataset['experiment_config']
    config=r._read_json(ROOT/dataset['experiment_config'])
    # Catch runner declarations before preparation/training/calibration spend time.
    r.validate_baseline_config(r._read_json(ROOT/dataset['baselines']))
    output=ROOT/dataset['output_dir'];output.mkdir(parents=True,exist_ok=True)
    def status(stage,**fields):
        write(output/'status.json',{'stage':stage,'pid':os.getpid(),'updated_at_unix':time.time(),**fields})
    with (output/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            source=ROOT/'src/vec2plan_bandit'
            contract={'dataset':dataset,'config':config,'train_config':r._read_json(train_config),
                'queries':sha(ROOT/dataset['queries']),'splits':sha(ROOT/dataset['splits']),
                'runner':sha(__file__),'source':{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}}
            path=output/'run_contract.json'
            if path.exists() and r._read_json(path)!=contract: raise ValueError('Experiment contract changed')
            write(path,contract)
            stages=['plan','preflight','prepare','collect','train','dynamicann_setup','calibrate','dev','test','report'] if args.stage=='all' else [args.stage]
            for stage in stages:
                marker=output/(stage+'.stage-complete.json')
                if marker.exists():continue
                status(stage)
                if stage=='dynamicann_setup':
                    from experiments.dynamic_ann import prepare_dynamicann
                    prepare_dynamicann(dataset,dataset_path,status)
                elif stage=='calibrate':calibrate(dataset,config,output,status)
                elif stage=='report':report(output)
                else:
                    cfg=output/'evaluation-config.json' if stage in ('dev','test') else train_config
                    destination, command = stage_command(dataset, stage, cfg)
                    destination.mkdir(parents=True, exist_ok=True)
                    log_path = destination / 'phase.log'
                    print(f'arXiv: {stage}. Log: {log_path}', flush=True)
                    with log_path.open('a') as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                write(marker,{'stage':stage,'completed_at_unix':time.time()})
            status('complete' if args.stage=='all' else args.stage+'_complete')
        except Exception as error:
            status('failed',error=str(error));raise


if __name__=='__main__': main()
