"""Main-only bounded FAISS fill; embedding and deadline belong to the caller."""
import math
from vec2plan_bandit.paths.registry import REGISTRY,make_execution_config


def execute_with_fill(execute,config,*,k,remaining_ms,facts=None,enabled=False,id_field='row_ids',allowed_configurations=None):
    row=execute(config)
    attempts=[dict(config_id=config.config_id,params=dict(config.params),count=len(row.get(id_field,[])))]
    # Only a proved single-table retrieval may expand this search. Complex
    # nested LIMIT/JOIN semantics and unknown cardinalities keep the old call.
    can_expand=(enabled and config.path_id=='FAISS' and facts
        and facts.get('pre_join_settings_irrelevant') is True and facts.get('candidate_count',0)>=k)
    selected=config
    for _ in range(2 if can_expand else 0):
        if len(row.get(id_field,[]))>=k or row.get('error_type') or row.get('timed_out') or remaining_ms()<=1.:
            break
        if allowed_configurations is not None:
            choices=[c for c in allowed_configurations if c.path_id=='FAISS'
                and c.params['ef_search']>=selected.params['ef_search']
                and c.params['candidate_multiplier']>selected.params['candidate_multiplier']]
            if not choices:break
            selected=min(choices,key=lambda c:(c.params['candidate_multiplier'],c.params['ef_search']))
        else:
            limits={p.name:p for p in REGISTRY.get('FAISS').parameters}
            params=dict(selected.params)
            params['candidate_multiplier']=min(limits['candidate_multiplier'].upper,params['candidate_multiplier']*2)
            params['ef_search']=min(limits['ef_search'].upper,max(params['ef_search']+1,math.ceil(params['ef_search']*1.5)))
            selected=make_execution_config('FAISS',params)
        if selected.config_id==attempts[-1]['config_id']:break
        row=execute(selected)
        attempts.append(dict(config_id=selected.config_id,params=dict(selected.params),count=len(row.get(id_field,[]))))
    if enabled and config.path_id=='FAISS':
        row.update(main_fill_attempts=attempts,main_effective_config_id=selected.config_id,
                   main_effective_params=dict(selected.params),main_adaptive_fill=len(attempts)>1)
    return row
