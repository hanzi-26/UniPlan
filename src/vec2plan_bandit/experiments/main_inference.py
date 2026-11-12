"""Explicit main-policy-only opt-ins; no baseline objects enter these helpers."""
import hashlib
import json
from pathlib import Path
import time


def configure_main_inference(policy, config, output, checkpoint=None,calibrating=False):
    profiles=config.get('main_parameter_tuning',{})
    if profiles.get('enabled',False):
        from vec2plan_bandit.policies.parameter_profiles import load
        policy.main_parameter_profiles=load(profiles)
        objective=profiles.get('objective','quality_first')
        if objective not in ('quality_first','latency_ucb_after_admission'):
            raise ValueError('Unknown admitted-profile ranking objective')
        policy.main_profile_cost_ranking=objective=='latency_ucb_after_admission'
    result = None
    from vec2plan_bandit.paths.configuration_bank import load_bank
    policy.main_configuration_bank=load_bank(config.get('main_candidate_bank'))
    if policy.main_configuration_bank is not None and profiles.get('enabled',False):
        raise ValueError('parameter profiles cannot replace a calibrated candidate bank')
    policy.main_pg_semantic_families=config.get('runtime',{}).get('pg_semantic_families',False)
    policy.main_pg_complex_families=config.get('runtime',{}).get('pg_complex_families',False)
    recovery = config.get('main_constraint_recovery', {})
    policy.main_constraint_recovery = None
    policy.main_execution_cost_factors={}
    if recovery.get('enabled', False):
        artifact = Path(recovery['artifact'])
        raw = artifact.read_bytes()
        if hashlib.sha256(raw).hexdigest() != recovery['sha256']:
            raise ValueError('constraint recovery calibration changed')
        calibrated = json.loads(raw)
        if calibrated.get('candidate_bank_sha256') is not None and calibrated['candidate_bank_sha256'] != config.get('main_candidate_bank',{}).get('sha256'):
            raise ValueError('recovery calibration belongs to a different candidate bank')
        if (calibrated.get('version') != 'main-constraint-recovery-v1' or
                calibrated.get('test_labels_used') is not False or
                calibrated.get('parameter_selection') != 'train' or
                calibrated.get('enable_or_reject') != 'dev' or
                calibrated.get('recall_threshold') != config['reward']['recall_threshold']):
            raise ValueError('invalid train/dev constraint recovery artifact')
        if checkpoint is None or calibrated.get('surrogate_sha256') != {
                str(p.relative_to(Path(checkpoint)/'surrogate')):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(checkpoint)/'surrogate').rglob('*') if p.is_file()}:
            raise ValueError('recovery calibration belongs to different model weights')
        if calibrated['accepted']:
            policy.main_constraint_recovery = dict(quality_overlap_sigma=calibrated['quality_overlap_sigma'],
                exact_recall=config['reward']['recall_threshold'] == 1.)
        if calibrated.get('execution_cost_factors'):
            if calibrated.get('cost_calibration_test_labels_used') is not False:
                raise ValueError('invalid execution cost calibration provenance')
            policy.main_execution_cost_factors=calibrated['execution_cost_factors']
    policy.main_search=dict(config.get('main_search',{}))
    if policy.main_search:
        if (set(policy.main_search)-{'enabled','initial_candidates','quality_margin'}
                or type(policy.main_search.get('enabled',False)) is not bool
                or type(policy.main_search.get('initial_candidates',32)) is not int
                or not 1<=policy.main_search.get('initial_candidates',32)<=policy.candidates_per_numeric_path
                or not 0<=policy.main_search.get('quality_margin',.03)<=1):
            raise ValueError('invalid main-only staged search settings')
    policy.main_search_budget = None
    if 'main_search_budget' in config:
        from vec2plan_bandit.policies.search_budget import SearchBudget
        policy.main_search_budget = SearchBudget.from_config(config['main_search_budget'])
        if policy.main_search_budget is not None and not policy.main_search.get('enabled'):
            raise ValueError('search budget requires enabled staged candidate search')
    if config.get('runtime', {}).get('main_fused_inference', False):
        result = policy.surrogate.prepare_fused_inference(output / 'main-inference')
    elif config.get('runtime', {}).get('main_fast_inference', False):
        result = policy.surrogate.prepare_fast_inference(output / 'main-inference')
    calibration = config.get('main_quality_calibration', {})
    if calibration.get('enabled', False):
        from vec2plan_bandit.surrogates.quality_calibration import attach_quality_calibration
        attach_quality_calibration(policy, calibration, config)
    group_calibration = config.get('main_group_quality_calibration', {})
    if group_calibration.get('enabled', False):
        from vec2plan_bandit.surrogates.group_quality_calibration import attach_group_quality_calibration
        attach_group_quality_calibration(policy, group_calibration, config, checkpoint)
    options = dict(config.get('main_lightweight_selector', {}))
    empirical = config.get('main_empirical_early', {})
    competitive = config.get('main_competitive_early', {})
    policy.main_sql_candidates_early = (competitive.get('enabled', False)
                                        and competitive.get('strategy') == 'anchor_competition')
    policy.main_exact_pg_early = (competitive.get('enabled', False)
                                  and competitive.get('strategy') == 'exact_optimizer')
    if competitive.get('enabled', False):
        if competitive.get('allow_empirical_fallback') is not True:
            raise ValueError('Competitive early routing needs explicit best-effort opt-in')
        if competitive.get('strategy', 'competitive') not in ('competitive', 'scoped', 'exact_optimizer', 'anchor_competition'):
            raise ValueError('Unknown competitive early routing strategy')
        if empirical.get('enabled', False):
            raise ValueError('Choose one empirical early routing mechanism')
        if type(competitive.get('allow_underfilled_estimate', False)) is not bool:
            raise ValueError('allow_underfilled_estimate must be boolean')
        policy.main_allow_underfilled_estimate = competitive.get('allow_underfilled_estimate', False)
        options['collect_facts'] = True
    if empirical.get('enabled', False):
        if empirical.get('allow_empirical_fallback') is not True:
            raise ValueError('Empirical early routing needs explicit best-effort opt-in')
        options['collect_facts'] = True
    auto=config.get('main_selector_calibration',{})
    if auto.get('enabled') and not calibrating and Path(auto['artifact']).is_file():
        options.update(enabled=True,artifact=auto['artifact'],native_pg_early=True,native_pg_fallback=True)
    unknown = set(options)-{'enabled','artifact','collect_facts','deduplicate_pre_joins',
                            'native_pg_early','native_pg_fallback'}
    if unknown:
        raise ValueError(f'unknown main lightweight options: {sorted(unknown)}')
    if not options:
        return result
    policy.main_lightweight = None
    policy.main_recall_threshold = config['reward']['recall_threshold']
    policy.main_embedding_dimension = config['embedding']['dimension']
    policy.deduplicate_pre_joins = options.get('deduplicate_pre_joins', False)
    policy.main_native_pg_early = options.get('native_pg_early',False)
    policy.main_native_pg_fallback = options.get('native_pg_fallback',False)
    policy.collect_main_routing_facts = (options.get('collect_facts',False)
        or options.get('enabled',False) or policy.deduplicate_pre_joins)
    for name in ('enabled','collect_facts','deduplicate_pre_joins','native_pg_early','native_pg_fallback'):
        if name in options and type(options[name]) is not bool:
            raise ValueError(f'main_lightweight_selector.{name} must be boolean')
    if (policy.main_native_pg_early or policy.main_native_pg_fallback) and not options.get('enabled',False):
        raise ValueError('native PG routing requires an enabled, calibrated main selector')
    if not policy.collect_main_routing_facts:
        return result
    if checkpoint is None:
        raise ValueError('main lightweight configuration requires a frozen checkpoint')
    checkpoint = Path(checkpoint)
    files = {str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(checkpoint.rglob('*')) if p.is_file()}
    if not files:
        raise ValueError('empty main lightweight checkpoint binding')
    external_features = checkpoint.parent/'features.npz'
    if external_features.exists():
        files['../features.npz'] = hashlib.sha256(external_features.read_bytes()).hexdigest()
    receipt_paths = [config.get('assets',{}).get('wiki_receipt')]
    for asset in config.get('faiss_assets',{}).values():
        receipt = asset.get('id_mapping_receipt') or str(Path(asset['article_ids']).parent/'id-mapping.json')
        receipt_paths.extend((receipt,str(Path(asset['index']).parent/'manifest.json')))
    receipt_paths.extend(config.get('acorn',{}).get('identity_receipts',{}).values())
    for directory,name in [(config.get('ann_execution',{}).get('snapshot_dir'),'manifest.json'),
                           (config.get('metadata',{}).get('snapshot_dir'),'metadata-manifest.json')]:
        if directory:
            receipt_paths.append(str(Path(directory)/name))
    receipts = {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in receipt_paths if p and Path(p).is_file()}
    asset_settings = {k:config.get(k) for k in ('assets','faiss_assets','ann_execution','acorn','database','relational_pg')}
    # The native SQL renderer and resource/timing protocol are part of a PG
    # certificate. Changing the execution code invalidates previous evidence.
    source_root = Path(__file__).resolve().parents[1]
    execution_files = ('policies/main_lightweight.py','policies/max_ucb.py','experiments/main_priority.py',
        'experiments/main_lightweight.py','bird/shared_experiment.py','bird/runtime.py',
        'execution/adaptive_runtime.py','execution/wikipedia_runtime.py',
        'execution/article_identity.py','execution/acorn_path.py','execution/faiss_path.py',
        'baselines/native_pg.py','experiments/collector.py',
        'queries/lightweight.py','queries/arxiv_sql_adapter.py','policies/optional_acceleration.py',
        'execution/relational_pg.py','execution/pre_access.py','execution/relation_lineage.py','bird/adapter.py')
    execution_hashes = {name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                        for name in execution_files}
    if config.get('main_candidate_bank',{}).get('enabled',False):
        asset_settings['main_candidate_bank']=config['main_candidate_bank']
        execution_hashes['paths/configuration_bank.py']=hashlib.sha256(
            (source_root/'paths/configuration_bank.py').read_bytes()).hexdigest()
    if policy.main_pg_semantic_families:
        asset_settings['pg_semantic_families']=True
        asset_settings['pg_complex_families']=policy.main_pg_complex_families
    if config.get('runtime',{}).get('reuse_main_routing_facts',False):
        asset_settings['reuse_main_routing_facts']=True
    if config.get('raw_duplicate_identities',{}).get('enabled',False):
        duplicate_options=config['raw_duplicate_identities']
        asset_settings['raw_duplicate_identities']=duplicate_options
        if hashlib.sha256((Path(duplicate_options['directory'])/'manifest.json').read_bytes()).hexdigest()!=duplicate_options['sha256']:
            raise ValueError('Duplicate metadata receipt changed')
        name='bird/duplicate_vectors.py'
        execution_hashes[name]=hashlib.sha256((source_root/name).read_bytes()).hexdigest()
        name='experiments/faiss_rule.py'
        execution_hashes[name]=hashlib.sha256((source_root/name).read_bytes()).hexdigest()
    if recovery.get('enabled',False):
        asset_settings['main_constraint_recovery'] = recovery
        for name in ('policies/constraint_recovery.py',):
            execution_hashes[name] = hashlib.sha256((source_root/name).read_bytes()).hexdigest()
    if config.get('runtime',{}).get('main_fused_inference',False):
        asset_settings['main_fused_inference'] = True
        for name in ('surrogates/main_cost.py','surrogates/frozen_forest.py','surrogates/frozen_forest.cpp'):
            execution_hashes[name] = hashlib.sha256((source_root/name).read_bytes()).hexdigest()
    if config.get('relational_pre',{}).get('enabled',False):
        asset_settings['relational_pre']=config['relational_pre']
        for name in ('execution/relational_pre.py','execution/relational_pre_assets.py','paths/registry.py'):
            execution_hashes[name]=hashlib.sha256((source_root/name).read_bytes()).hexdigest()
        if config['relational_pre'].get('exact_join_topk',False):
            name='execution/relational_topk.py'
            execution_hashes[name]=hashlib.sha256((source_root/name).read_bytes()).hexdigest()
        if config['relational_pre'].get('bounded_key_join',False):
            name='execution/relational_key_join.py'
            execution_hashes[name]=hashlib.sha256((source_root/name).read_bytes()).hexdigest()
    if config.get('runtime',{}).get('batch_local_settings',False):
        asset_settings['batch_local_settings']=True
        execution_hashes['execution/sql_batch.py']=hashlib.sha256(
            (source_root/'execution/sql_batch.py').read_bytes()).hexdigest()
    binding = dict(checkpoint_files_sha256=files, snapshot_id=config.get('snapshot_id'),
                   embedding=config['embedding'], policy=policy.state_dict(),snapshot_receipts_sha256=receipts,
                   asset_settings_sha256=hashlib.sha256(json.dumps(asset_settings,sort_keys=True).encode()).hexdigest(),
                   execution_files_sha256=execution_hashes,
                   resources={k:config.get('runtime',{}).get(k) for k in
                       ('cpu_threads','inference_threads','unified_resources','detailed_timing')})
    if policy.main_search_budget is not None:
        binding['search_settings'] = dict(search=policy.main_search,
            budget=config['main_search_budget'], controller_sha256=hashlib.sha256(
                (source_root/'policies/search_budget.py').read_bytes()).hexdigest())
    if group_calibration.get('enabled', False):
        binding['group_quality_calibration'] = dict(artifact_sha256=group_calibration['sha256'],
            implementation_sha256=hashlib.sha256(
                (source_root/'surrogates/group_quality_calibration.py').read_bytes()).hexdigest())
    if empirical.get('enabled', False):
        binding['empirical_early_implementation_sha256'] = hashlib.sha256(
            (source_root/'policies/empirical_early.py').read_bytes()).hexdigest()
    if competitive.get('enabled', False):
        binding['competitive_early_implementation_sha256'] = hashlib.sha256(
            (source_root/'policies/competitive_early.py').read_bytes()).hexdigest()
        if competitive.get('allow_underfilled_estimate', False):
            binding['competitive_routing_options'] = dict(allow_underfilled_estimate=True)
        if competitive.get('strategy') in ('scoped', 'exact_optimizer', 'anchor_competition'):
            binding['scoped_early_implementation_sha256'] = hashlib.sha256(
                (source_root/'policies/scoped_early.py').read_bytes()).hexdigest()
        if policy.main_exact_pg_early:
            binding['main_exact_pg'] = dict(enabled=True, implementation_sha256={
                name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in
                ('execution/main_exact_pg.py', 'policies/exact_optimizer.py', 'queries/wikipedia_sql_adapter.py')})
        if policy.main_sql_candidates_early:
            binding['main_sql_candidates'] = dict(enabled=True, implementation_sha256={
                name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in
                ('execution/main_sql_candidates.py', 'policies/anchor_competition.py',
                 'queries/wikipedia_sql_adapter.py', 'execution/simple_ann.py')})
            if competitive.get('execution_adapter') == 'spider':
                binding['main_sql_candidates']['adapter_sha256'] = {
                    name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in
                    ('spider/main_sql.py', 'spider/runtime.py', 'spider/adapter.py', 'bird/covering.py')}
                mapping = competitive.get('identity_mapping')
                if not mapping or not Path(mapping).is_file():
                    raise ValueError('Spider main SQL needs a frozen full-corpus point/CTID mapping')
                binding['main_sql_candidates']['identity_mapping_sha256'] = hashlib.sha256(
                    Path(mapping).read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    (output/'main-lightweight-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    if options.get('enabled',False):
        if not options.get('artifact'):
            raise ValueError('main lightweight selector must be calibrated before enabling')
        from vec2plan_bandit.policies.main_lightweight import MainLightweightSelector
        policy.main_lightweight = MainLightweightSelector.load(options['artifact'],binding=binding)
        if policy.main_configuration_bank is not None:
            ids={c.config_id for cs in policy.main_configuration_bank.values() for c in cs}
            def certificates(artifact):
                yield from artifact['leaves'].values()
                yield from artifact.get('priority_rules',[])
                yield from artifact.get('pg_fallbacks',{}).values()
                for values in artifact.get('pg_action_leaves',{}).values():yield from values
                for child in artifact.get('pg_action_trees',[]):yield from certificates(child)
            if any(v['config_id'] not in ids for v in certificates(policy.main_lightweight.artifact)):
                raise ValueError('early selector escaped the measured candidate bank')
    if empirical.get('enabled', False) and not calibrating:
        from vec2plan_bandit.policies.empirical_early import EmpiricalEarlySelector
        policy.main_lightweight = EmpiricalEarlySelector.load(empirical['artifact'],binding=binding,
            sha256=empirical['sha256'])
        policy.main_native_pg_early = True
        policy.main_native_pg_fallback = False
    if competitive.get('enabled', False) and not calibrating:
        if competitive.get('strategy') == 'anchor_competition':
            from vec2plan_bandit.policies.anchor_competition import AnchorCompetitionSelector as Selector
        elif competitive.get('strategy') == 'exact_optimizer':
            from vec2plan_bandit.policies.exact_optimizer import ExactOptimizerSelector as Selector
        elif competitive.get('strategy') == 'scoped':
            from vec2plan_bandit.policies.scoped_early import ScopedEarlySelector as Selector
        else:
            from vec2plan_bandit.policies.competitive_early import CompetitiveEarlySelector as Selector
        policy.main_lightweight = Selector.load(competitive['artifact'], binding=binding,
            sha256=competitive['sha256'])
        if (policy.main_lightweight.artifact['protocol'].get('allow_underfilled_estimate', False)
                != policy.main_allow_underfilled_estimate):
            raise ValueError('Competitive artifact/request facts settings differ')
        policy.main_native_pg_early = True
        policy.main_native_pg_fallback = False
        policy.main_compact_early = True
    return result


def fit_auto_main_selector(config,output,split):
    """Freeze train/dev evidence before any test evaluation; reject safely."""
    options=config.get('main_selector_calibration',{})
    if not options.get('enabled'):return
    from vec2plan_bandit.experiments.main_lightweight import fit_lightweight
    from vec2plan_bandit.policies.main_lightweight import MainLightweightSelector
    output=Path(output)
    binding=json.loads((output/'main-lightweight-binding.json').read_text())
    journal=output/'calibration/measurements.jsonl'
    if not journal.is_file():
        raise ValueError('Main selector calibration measurements are missing; run calibration and inspect its query coverage')
    rows=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Main selector calibration measurements are empty; inspect SQL/statistics adapters and query coverage')
    if options.get('strategy') == 'faiss_then_pg':
        from vec2plan_bandit.experiments.main_priority import fit_priority
        artifact=fit_priority(rows,train_ids=split['train'],dev_ids=split['dev'],binding=binding,
            recall_threshold=config['reward']['recall_threshold'],
            probability_min=config['max_ucb']['quality_probability_min'],
            latency_cap_ms=config['reward']['latency_cap_ms'])
    else:
        artifact=fit_lightweight(rows,train_ids=split['train'],dev_ids=split['dev'],binding=binding,
            recall_threshold=config['reward']['recall_threshold'],probability_min=config['max_ucb']['quality_probability_min'],
            latency_cap_ms=config['reward']['latency_cap_ms'],
            min_train_groups=options.get('min_train_groups',12),min_dev_groups=options.get('min_dev_groups',12))
    MainLightweightSelector(artifact,binding=binding)
    target=Path(options['artifact']);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and json.loads(target.read_text())!=artifact:
        raise ValueError('main selector calibration changed; use a new experiment directory')
    target.write_text(json.dumps(artifact,indent=2)+'\n')
    (output/'main-selector-summary.json').write_text(json.dumps(dict(
        early_leaves=len(artifact['leaves']),pg_fallback_leaves=len(artifact['pg_fallbacks']),
        priority_rules=[dict(path_id=r['path_id'],scope=r['scope'],train_groups=r['train_groups'],
            dev_groups=r['dev_groups'],quality_lower_bound=r['quality_lower_bound'])
            for r in artifact.get('priority_rules',[])],
        rejected=artifact['rejected'],test_labels_used=False),indent=2)+'\n')


def main_request_options(policy, query, statistics, *, corpus_size=None):
    """Called inside the main request timer; baseline branches never call it."""
    if not getattr(policy,'collect_main_routing_facts',False):
        return {}
    from vec2plan_bandit.policies.main_lightweight import routing_facts
    start = time.perf_counter_ns()
    facts = routing_facts(query,statistics,
        embedding_dimension=policy.main_embedding_dimension,corpus_size=corpus_size,
        allow_underfilled_estimate=getattr(policy,'main_allow_underfilled_estimate',False))
    if getattr(policy,'main_pg_semantic_families',False):
        from vec2plan_bandit.policies.main_lightweight import pg_semantic_family
        facts=pg_semantic_family(query,facts,allow_complex=getattr(policy,'main_pg_complex_families',False))
    return {'request_facts':facts,'request_facts_ms':(time.perf_counter_ns()-start)/1e6}


def runtime_routing_facts(query,statistics,worker,config,corpus_sizes=None):
    from vec2plan_bandit.policies.main_lightweight import routing_facts
    n=None
    blocks=getattr(query,'blocks',())
    sizes=corpus_sizes or {}
    if blocks:
        known=[sizes.get(b.asset_key,getattr(worker,'assets',{}).get(b.asset_key,{}).get('points',
                    getattr(b,'asset',{}).get('estimated_points'))) for b in blocks]
        if all(v is not None for v in known): n=sum(known)
    else:
        mapping=getattr(worker,'article_id_mappings',{}).get(query.embedding_column)
        if mapping is not None:n=len(mapping.values)
    competitive = config.get('main_competitive_early', {})
    facts=routing_facts(query,statistics,embedding_dimension=config.get('embedding',{}).get('dimension',384),corpus_size=n,
        allow_underfilled_estimate=bool(competitive.get('enabled',False)
                                       and competitive.get('allow_underfilled_estimate',False)))
    if config.get('runtime',{}).get('pg_semantic_families',False):
        from vec2plan_bandit.policies.main_lightweight import pg_semantic_family
        facts=pg_semantic_family(query,facts,allow_complex=config.get('runtime',{}).get('pg_complex_families',False))
    item=getattr(query,'item',None)
    if facts is not None and isinstance(item,dict) and 'scalar_filters' in item:
        # Predicate families, never values or query IDs, distinguish YFCC
        # equality, ranges and tags for train-calibrated parameter admission.
        facts['yfcc_filter_signature']=','.join(sorted(
            str(f['field'])+':'+str(f['operator']) for f in item['scalar_filters'])) or 'none'
    return facts
