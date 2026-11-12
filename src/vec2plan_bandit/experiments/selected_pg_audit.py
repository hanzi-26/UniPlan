"""Post-benchmark EXPLAIN of selected PG configurations, never ANALYZE."""
import json
from pathlib import Path

from vec2plan_bandit.execution.relational_pg import configured_post_settings
from vec2plan_bandit.execution.plan_semantics import plan_nodes


def audit(run):
    worker=run.runtime()
    observations={}
    for line in (run.output/'test/benchmark_live.jsonl').open():
        row=json.loads(line)
        if row['method_id'] in (run.dataset['main_method'],'pg_native'):
            key=(row['query_id'],row['method_id'],row.get('config_id'))
            observations.setdefault(key,row)
    records=[]
    for row in observations.values():
        if row['path_id'] not in ('PRE','POST','PG_NATIVE_SQL'):
            continue
        q=run.by_id[row['query_id']]
        vectors=run.prepared[q.query_id]['vectors']
        settings=(worker.pre_settings(row['requested_params']) if row['path_id']=='PRE' else
                  configured_post_settings(worker,row['requested_params']) if row['path_id']=='POST' else [])
        item={k:row[k] for k in ('query_id','method_id','path_id','requested_params','config_id')}
        try:
            worker.open(q.db_id)
            sql,binds=q.render(vectors,exact=row['path_id']=='PRE')
            plan=worker.sql('EXPLAIN (FORMAT JSON) '+sql,binds,[*settings,'SET LOCAL statement_timeout=2000'])[0][0]
            nodes=plan_nodes(plan)
            item.update(plan=plan,node_types=[n['Node Type'] for n in nodes],
                indexes=sorted({n['Index Name'] for n in nodes if n.get('Index Name')}))
        except Exception as error:
            item.update(error_type=type(error).__name__,error=str(error))
        records.append(item)
    target=run.output/'diagnostics/selected-pg-plans.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(dict(scope='After all dev/test timings; EXPLAIN only with selected params, prepared vectors; not contemporaneous executed plans or server timings',records=records),indent=2)+'\n')
    return target
