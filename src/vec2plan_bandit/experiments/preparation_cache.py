"""Reuse verified offline truth/embeddings; never reuse training timing labels."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_receipt(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def verified_cache(config, queries, *, query_source, split_json, package_root=None):
    options=config.get('offline_preparation_cache')
    if not options:return {},None
    source=Path(options['source_output']);folder=source/'prepare'
    manifest=json.loads((folder/'run_manifest.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    if manifest.get('status')!='COMPLETE' or not (source/'prepare.stage-complete.json').exists():
        raise ValueError('Preparation cache is not complete')
    if (protocol['inputs']['query_source']!=sha(query_source)
            or protocol['inputs']['split_json']!=sha(split_json)):
        raise ValueError('Preparation cache query data or split changed')
    for filename,digest in manifest['inputs'].items():
        if sha(filename)!=digest:raise ValueError('Preparation source input changed')
    for key in ('snapshot_id','distance_metric','embedding','database','metadata','faiss_assets','acorn'):
        if protocol['config'].get(key)!=config.get(key):
            raise ValueError(f'Preparation snapshot/encoder/mapping mismatch: {key}')
    root=Path(package_root) if package_root else Path(__file__).resolve().parents[1]
    required={'paths/sql.py','experiments/collector.py','queries/loader.py','execution/article_identity.py'}
    if not required.issubset(protocol['source']):
        raise ValueError('Preparation cache lacks semantic code provenance')
    checked={}
    for name,digest in protocol['source'].items():
        if (name.startswith('paths/') or name.startswith('execution/') or name.startswith('assets/')
                or name in ('experiments/collector.py','queries/loader.py')):
            if sha(root/name)!=digest:raise ValueError(f'Preparation semantics changed: {name}')
            checked[name]=digest
    by_id={q.query_id:q for q in queries};rows={}
    for line in (folder/'prepared.jsonl').open():
        row=json.loads(line);qid=row['query_id']
        if qid in rows:raise ValueError('Duplicate preparation cache query')
        if qid not in by_id:raise ValueError('Unexpected preparation cache query')
        q=by_id[qid];v=np.asarray(row['embedding'],dtype=np.float32)
        if (row['source_hash']!=q.source_hash or row.get('tie_rule')!='cosine_distance_then_article_id'
                or v.shape!=(config['embedding']['dimension'],) or not np.isfinite(v).all()
                or (config['embedding'].get('normalized') and not np.isclose(np.linalg.norm(v),1.,atol=1e-3))
                or len(row['topk_ids'])>q.k
                or any(type(i) is not int for i in row['topk_ids'])
                or len(set(row['topk_ids']))!=len(row['topk_ids'])):
            raise ValueError(f'Invalid preparation cache identity/vector: {qid}')
        rows[qid]=row
    if set(rows)!=set(by_id):raise ValueError('Incomplete preparation cache query coverage')
    compared=0
    if options.get('recent_preparation'):
        for line in Path(options['recent_preparation']).open():
            row=json.loads(line);qid=row['query_id']
            if qid not in rows:raise ValueError('Unexpected recent preparation query')
            if any(row[k]!=rows[qid][k] for k in ('source_hash','embedding','topk_ids','statistics','tie_rule')):
                raise ValueError(f'Recent preparation disagrees with cache: {qid}')
            compared+=1
    receipt=dict(source_output=str(source),queries=len(rows),recent_rows_compared=compared,
        source_prepared_sha256=sha(folder/'prepared.jsonl'),source_manifest_sha256=sha(folder/'run_manifest.json'),
        checked_semantics_sha256=checked,scope='offline truth, vectors and statistics only',
        timing_labels_reused=False,models_reused=False,plan_policy='refresh_each_query',
        plans_refreshed=False,live_train_probes=[])
    return rows,receipt
