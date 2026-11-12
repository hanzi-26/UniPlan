"""An opt-in finite candidate domain shared by training and all selectors."""
import hashlib
import json
from pathlib import Path

from .registry import PATH_IDS, make_execution_config


def load_bank(options):
    if not options or not options.get('enabled', False):
        return None
    raw=Path(options['artifact']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=options['sha256']:
        raise ValueError('candidate bank changed')
    data=json.loads(raw)
    if data.get('version')!='finite-four-path-bank-v1' or data.get('test_labels_used') is not False:
        raise ValueError('invalid candidate bank provenance')
    result={p:[] for p in PATH_IDS};seen=set()
    for entry in data['configurations']:
        c=make_execution_config(entry['path_id'],entry['params'])
        if c.config_id in seen:
            raise ValueError('duplicate candidate configuration')
        seen.add(c.config_id);result[c.path_id].append(c)
    if not all(result.values()):
        raise ValueError('candidate bank must retain all four paths')
    return {p:tuple(cs) for p,cs in result.items()}


def standard_bank():
    """Declared resource envelope; no query names, outcomes or fitted labels."""
    configs=[]
    for join in ('all','hash','merge','nested_loop'):
        configs.append(make_execution_config('PRE',dict(index_mode='no_index',join_strategy=join)))
        for access in ('optimizer','btree','hash','bitmap'):
            configs.append(make_execution_config('PRE',dict(index_mode='index',index_strategy=access,join_strategy=join)))
    configs += [make_execution_config('POST',dict(ef_search=e,max_scan_tuples=n)) for e,n in
        ((100,100000),(200,500000),(400,1000000),(800,3000000),(1000,3000000))]
    configs += [make_execution_config('FAISS',dict(ef_search=e,candidate_multiplier=m)) for e,m in
        ((100,4.),(200,8.),(400,16.),(800,32.),(1000,64.))]
    configs += [make_execution_config('ACORN',dict(hnsw_ef=e)) for e in (100,200,400,800,1000)]
    return dict(version='finite-four-path-bank-v1',test_labels_used=False,
        source='predeclared configuration envelope, shared by all four selectors',
        configurations=[dict(path_id=c.path_id,params=dict(c.params)) for c in configs])
