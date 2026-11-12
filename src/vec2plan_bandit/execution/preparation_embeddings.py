"""Validated embeddings for offline preparation only, never timed requests."""
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION='preparation-embeddings-v1'


def load_preparation_embeddings(path, *, embedding, source, queries):
    payload=json.loads(Path(path).read_text())
    if (payload.get('version')!=VERSION or payload.get('embedding')!=embedding
            or payload.get('source_sha256')!=hashlib.sha256(Path(source).read_bytes()).hexdigest()):
        raise ValueError('Preparation embedding source/model changed')
    by_id={q.query_id:q for q in queries}
    if set(payload['queries'])!=set(by_id):
        raise ValueError('Preparation embedding query coverage mismatch')
    result={}
    for qid,q in by_id.items():
        item=payload['queries'][qid]
        if item['source_hash']!=q.source_hash:
            raise ValueError('Preparation embedding SQL changed')
        vectors=np.asarray(item['vectors'],dtype=np.float32)
        if (vectors.shape!=(len(q.blocks),embedding['dimension']) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-3)):
            raise ValueError('Invalid preparation embedding dimensions/content')
        result[qid]=list(vectors)
    return result
