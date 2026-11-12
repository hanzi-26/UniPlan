"""Shared main experiment workflow for Wikipedia and Spider."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,time,types
from pathlib import Path
import numpy as np
from vec2plan_bandit.experiments.wikipedia import ROOT,write,read_rows
from vec2plan_bandit.bird.shared_experiment import score,SharedExperiment
from vec2plan_bandit.experiments.wiki_optional_acceleration import WikiOptionalExperiment,sha


def report(run):
    rows=[]
    for role in ('dev','test'):
        path=run.output/role/'summary.json'
        if path.exists():
            for method,s in json.loads(path.read_text()).items():
                rows.append({'split':role,'method_id':method,**{k:v for k,v in s.items() if not isinstance(v,(dict,list))}})
    if rows:
        with (run.output/'comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
    options=run.config['acceleration_calibration']
    text=f"# Unified experiment\n\nRecall={run.config['reward']['recall_threshold']}; main predicted probability threshold={run.config['max_ucb']['quality_probability_min']}; E2E budget=5000ms.\n\nAll methods share query data, quality target, warmup/repeats and client CPU limits. Main and applicable baselines share numerical feature batching/Pool optimization. FAISS candidates are identical across datasets; train/dev validate all quality observations and >{options['min_relative_gain']:.0%} E2E gain.\n\nExisting test splits retained by user instruction. Probability is a main-policy decision threshold, not an established statistical guarantee. Shared PostgreSQL/Qdrant services are not claimed to be exclusively pinned. Their service overhead remains in E2E. PRE actions are available only after physical-plan verification; unsupported accesses are explicitly masked.\n"
    if run.config.get('runtime',{}).get('main_fast_inference',False):
        text += '\nMain-only inference revision: all existing baseline decisions and shared execution/feature/query code are retained. Only the main frozen ensemble opts into native batched tree evaluation. All model weights, candidates, uncertainty calculations and quality gates remain unchanged. Startup compilation/load and label-free equivalence checks are recorded in main-inference/manifest.json; request prediction remains inside E2E latency. Shared backend improvements are not claimed as main-only contributions.\n'
    if run.config.get('main_training',{}).get('enabled',False):
        text = f'# Main cost optimization experiment\n\nRecall={run.config["reward"]["recall_threshold"]}; predicted quality probability threshold={run.config["max_ucb"]["quality_probability_min"]}; hard E2E budget=5000ms.\n\n'
        if run.config.get('threshold_sweep',{}).get('predictor_weights_reused'):
            sweep=run.config['threshold_sweep']
            text += f"The main predictor reuses frozen weights trained at probability threshold {sweep['source_training_probability']}. This is a deployment-threshold sweep, not probability-specific retraining. Main train/dev routing calibration and all baseline dev/test benchmarks are measured afresh. See model-reuse.json and threshold-sweep.json for provenance. "
        else:
            text += 'The main method is retrained with eligible quality labels, completed execution-time labels, and a separate timeout head. '
        text += 'It uses staged candidate evaluation and bounded adaptive FAISS fill. Optional PG routing requires frozen train/dev certificates. Native PG and the other eight baselines are measured with the same queries, quality target, resource limits, warmups, repeats and timing boundary. Baseline decision algorithms are unchanged. Shared ordinal-ID validation and FAISS candidate reranking correctness fixes apply equally to all methods.\n\n'
        text += 'Only offline main ensemble fitting uses GPU parallelism; CPU inference and formal serial evaluation do not overlap with training. Existing splits are retained. The probability threshold is a policy setting, not an established statistical guarantee. Startup compilation and equivalence checks are recorded separately; request prediction remains inside E2E timing. PostgreSQL and Qdrant service work remains in E2E.\n'
    if run.config.get('pg_execution_contract'):
        text += f'\nShared PostgreSQL execution correction: {run.config["pg_execution_contract"]}. PRE/POST execute one whole-query result SELECT without eligibility-ID roundtrips or a second assembly SQL. This backend applies equally to all methods using PRE/POST; their decision algorithms remain unchanged. '
        if run.config.get('threshold_sweep',{}).get('predictor_weights_reused'):
            text += 'Frozen model weights and preparation are reused as documented in model-reuse.json; routing calibration, Dynamic ANN train-only calibration and held-out timings are freshly measured.\n'
        elif (run.output/'dynamicann-calibration/reuse.json').exists():
            text += 'Main bootstrap costs, main training, routing calibration and held-out benchmarks are freshly measured. Probability-independent fixed-path Dynamic ANN calibration is reused after input/resource/index validation; see dynamicann-calibration/reuse.json.\n'
        else:
            text += 'Costs, main training, Dynamic ANN train-only calibration and held-out benchmarks are freshly collected for this experiment.\n'
    if run.config.get('main_selector_calibration',{}).get('strategy')=='faiss_then_pg':
        text += ('\nMain-only ordered routing: semantically proven simple unfiltered single-vector FAISS, '
                 'then certified native PG, then the frozen full model. FAISS parameters and PG corpus '
                 'bounds are selected on train; dev only accepts or rejects. Every eligible paired '
                 'quality observation must pass Recall. Independent semantic groups, rather than '
                 'repeated requests, determine the empirical Wilson evidence threshold. Below-k '
                 'requests provide no quality support. No test labels select or enable rules. '
                 'See main-selector.json for actual enabled rules, scope bounds and exclusions.\n')
    if run.config.get('pg_execution_revision'):
        text += ('\nPlanner repair: view-aware indexes on physical filter/join columns and catalog-proven '
                 'CTID mapping access. PRE retains exact semantics and every original parameter. POST '
                 'retains HNSW controls but allows optimizer scan choice; not every POST plan uses HNSW. '
                 'Shared corrected assets and execution settings apply to all methods. Selected PG '
                 'configurations receive EXPLAIN-only audit after all timed evaluation.\n')
    (run.output/'report.md').write_text(text)
    write(run.output/'results.json',rows)


def make_run(label,path):
    if label=='wiki':run=WikiOptionalExperiment(path)
    elif label=='spider':
        from vec2plan_bandit.spider.experiment import Experiment
        run=Experiment(path)
    else:raise ValueError(label)
    run.report=types.MethodType(report,run)
    run.benchmark=types.MethodType(WikiOptionalExperiment.benchmark,run)
    return run


def install_pre(run):
    """Separate write connection: online connections remain read-only."""
    import psycopg
    from vec2plan_bandit.assets.bird import pg_parameters
    from vec2plan_bandit.execution.pre_access import install_scalar_indexes
    marker=run.output/'setup/scalar-indexes.json'
    view_aware=run.config.get('runtime',{}).get('view_aware_pre',False)
    if marker.exists():
        saved=json.loads(marker.read_text())
        if not view_aware:return
        if saved.get('version')=='view-aware-pre-v2' and any(r['status']=='installed' for r in saved['indexes']):return
        write(marker.with_name('scalar-indexes.invalidated.json'),saved)
    worker=run.runtime();groups={}
    if hasattr(run,'corpus_sizes') and not hasattr(run.routable[0],'db_id'):
        params=pg_parameters(Path(run.config['database']['config_path']));params['dbname']=run.config['database']['name']
        groups[params['dbname']]={b.table for q in run.routable for b in q.blocks}
    else:
        params=pg_parameters(Path(run.config['database']['config_path']))
        for q in run.routable:groups.setdefault(q.db_id,set()).update(q.columns)
    receipts=[]
    selected_columns=None
    if view_aware:
        from vec2plan_bandit.execution.relation_lineage import filter_join_columns
        selected_columns=filter_join_columns(run.routable)
    for db,tables in groups.items():
        actual=run.config['database'].get('prefix','')+db if run.config['database'].get('prefix') and not db.startswith(run.config['database']['prefix']) else db
        def progress(**fields):write(run.output/'status.json',{'stage':'scalar_indexes_running','database':actual,**fields})
        with psycopg.connect(**{**params,'dbname':actual},autocommit=True) as conn:
            schema=run.config['database'].get('search_path_schema','public')
            columns={t:selected_columns.get((db,t),set()) for t in tables} if selected_columns is not None else None
            wanted={t for t in tables if columns is None or columns[t]}
            receipts.extend({'database':actual,**r} for r in install_scalar_indexes(conn,wanted,schema=schema,progress=progress,resolve_views=view_aware,column_filter=columns))
    if view_aware and not any(r['status']=='installed' for r in receipts):
        raise ValueError('View-aware PRE installed no usable scalar indexes')
    write(marker,{'version':'view-aware-pre-v2' if view_aware else 'legacy',
                  'scope':'offline scalar indexes on physical relations; bitmap reuses ordinary index access','indexes':receipts})


def refresh_prepare(run):
    path=run.output/'prepare/prepared.jsonl';rows=read_rows(path);worker=run.runtime();rebuilt=[]
    for i,row in enumerate(rows):
        q=run.by_id[row['query_id']]
        if q.blocks:
            if hasattr(worker,'prepare_query'):worker.prepare_query(q)
            vectors=[np.asarray(v,dtype=np.float32) for v in row['vectors']]
            try:plans=worker.pre_availability(q,vectors)
            except Exception as error:raise ValueError(f'{q.query_id}: PRE validation failed: {error}') from error
            row['available_pre']=list(plans);row['pre_plans']=plans
            if run.config.get('runtime',{}).get('view_aware_pre',False):
                from vec2plan_bandit.execution.relational_pg import post_plan_evidence
                row['post_plans']=[worker.explain_post(q,vectors)]
                row['post_plan_evidence']=post_plan_evidence(worker,row['post_plans'])
                row['pre_relation_lineage']={str(k):v for k,v in getattr(worker,'_pre_lineage',{}).items() if k[0]==q.db_id}
                row['pg_planner_revision']='view-aware-pre-optimizer-post-v2'
        rebuilt.append(row)
        write(run.output/'status.json',{'stage':'pre_plan_validation','completed':i+1,'total':len(rows),'query_id':q.query_id})
    if rows:path.write_text(''.join(json.dumps(r)+'\n' for r in rebuilt));run.prepared={r['query_id']:r for r in rebuilt}
    if run.config.get('runtime',{}).get('view_aware_pre',False):
        indexed=sum(any(not k.startswith('scan:') for k in r['available_pre']) for r in rebuilt)
        write(run.output/'prepare/planfix-validation.json',{'queries':len(rebuilt),'indexed_pre_queries':indexed,
              'post_planner_mode':run.config.get('relational_pg',{}).get('post_planner_mode','legacy')})
        if not indexed:raise ValueError('View-aware PRE still has no indexed candidates; refuse training')
    write(run.output/'prepare/pre-access-rejections.json',[{'query_id':k[0],'configuration':k[1],'reason':v} for k,v in getattr(worker,'pre_rejections',{}).items()])


def copy_verified(source,target,names):
    copied={}
    for name in names:
        a=source/name;b=target/name
        if not a.exists():continue
        b.parent.mkdir(parents=True,exist_ok=True)
        if not b.exists():
            if a.is_dir():shutil.copytree(a,b)
            else:shutil.copy2(a,b)
        for p in b.rglob('*') if b.is_dir() else [b]:
            if p.is_file():copied[str(p.relative_to(target))]=sha(p)
    return copied


def relabel_bootstrap(run,source):
    target=run.output/'collect/configuration_matrix.jsonl'
    if target.exists():return
    rows=read_rows(source/'collect/configuration_matrix.jsonl');truth={q:r['topk_row_ids'] for q,r in run.prepared.items()}
    for row in rows:
        if row['query_id'] not in run.split['train']:raise ValueError('Held-out bootstrap label')
        score(row,run.by_id[row['query_id']],truth[row['query_id']],run.config)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(run.output/'bootstrap-relabel.json',{'source':str(source),'source_sha256':sha(source/'collect/configuration_matrix.jsonl'),'recall':run.config['reward']['recall_threshold'],'held_out_labels_used':False,'fresh_online_training_required':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--kind',choices=['wiki','spider'],required=True);parser.add_argument('--bootstrap-source',type=Path);parser.add_argument('--model-source',type=Path);parser.add_argument('--prepare-source',type=Path)
    args=parser.parse_args();dataset=json.loads(args.dataset.read_text());output=Path(dataset['output_dir']);output.mkdir(parents=True,exist_ok=True)
    # Copies precede construction so adapters restore preparation consistently.
    if args.prepare_source:copy_verified(args.prepare_source,output,['prepare'])
    if args.bootstrap_source:copy_verified(args.bootstrap_source,output,['prepare','setup'])
    if args.model_source:
        copied=copy_verified(args.model_source,output,['prepare','setup','train'])
        statepath=output/'train/frozen_model/policy.json';state=json.loads(statepath.read_text());original_probability=state['quality_probability_min'];state['quality_probability_min']=json.loads(Path(dataset['experiment_config']).read_text())['max_ucb']['quality_probability_min'];write(statepath,state)
        write(output/'model-reuse.json',{'source':str(args.model_source),'copied_sha256':copied,'original_training_probability':original_probability,'evaluation_probability':state['quality_probability_min'],'mutation':'policy threshold only; weights, feature pipeline and recall labels unchanged'})
    run=make_run(args.kind,args.dataset);stage_name='initializing'
    try:
        if not args.model_source and not (output/'train.complete.json').exists():
            install_pre(run)
            if not args.bootstrap_source:
                stage_name='prepare';write(output/'status.json',{'stage':'prepare_running'});run.prepare();refresh_prepare(run)
            if args.bootstrap_source:relabel_bootstrap(run,args.bootstrap_source)
            else:
                stage_name='collect';write(output/'status.json',{'stage':'collect_running'});run.collect()
            stage_name='train';write(output/'status.json',{'stage':'train_running'})
            try:run.train()
            finally:
                from vec2plan_bandit.surrogates.parallel_fit import shutdown_training_workers
                shutdown_training_workers()
            write(output/'train.complete.json',{'complete':True,'model_policy_sha256':sha(output/'train/frozen_model/policy.json')})
        from experiments.dynamic_ann import prepare_dynamicann
        stage_name='dynamicann_setup'
        prepare_dynamicann(run.dataset,args.dataset,
            lambda stage: write(output/'status.json',{'stage':stage+'_running'}))
        for stage_name in ('calibrate','dev','test','report'):
            marker=output/f'{stage_name}.complete.json'
            if marker.exists():continue
            write(output/'status.json',{'stage':stage_name+'_running'})
            if stage_name in ('dev','test'):run.benchmark(stage_name)
            elif stage_name=='report':run.report()
            else:run.calibrate()
            write(marker,{'complete':True,'at_unix':time.time()})
        if run.config.get('runtime',{}).get('view_aware_pre',False):
            stage_name='selected_plan_audit';write(output/'status.json',{'stage':stage_name})
            from vec2plan_bandit.experiments.selected_pg_audit import audit
            audit(run)
        write(output/'status.json',{'stage':'complete','recall':run.config['reward']['recall_threshold'],'probability':run.config['max_ucb']['quality_probability_min'],'budget_ms':5000})
    except BaseException as e:
        write(output/'status.json',{'stage':'failed','phase':stage_name,'error_type':type(e).__name__,'error':str(e)});raise
    finally:
        if run.worker is not None:run.worker.close()

if __name__=='__main__':main()
