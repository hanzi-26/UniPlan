#!/usr/bin/env python3
"""Reuse existing graphs by deriving exact point-ordinal to article-ID arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vec2plan_bandit.execution.article_identity import ArticleIDMapping, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-store', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    storage = args.index_store.resolve()
    destination = args.output.resolve()
    asset = next(a for a in json.loads((storage / 'catalog.json').read_text())['assets']
                 if a['column'] == 'abstract_embedding')
    snapshot = storage / 'snapshots' / asset['asset_id']
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    fingerprint = hashlib.sha256((snapshot / 'manifest.json').read_bytes()).hexdigest()
    faiss = json.loads((storage / 'faiss' / asset['asset_id'] / 'manifest.json').read_text())
    acorn = json.loads((storage / 'acorn-progress' / (asset['asset_id'] + '.json')).read_text())
    if faiss['snapshot_manifest_sha256'] != fingerprint or acorn['snapshot_manifest_sha256'] != fingerprint:
        raise ValueError('FAISS and ACORN do not share the source snapshot')
    if (faiss['points'] != manifest['points'] or acorn['expected'] != manifest['points']
            or acorn['uploaded'] != manifest['points'] or acorn['status'] != 'complete'):
        raise ValueError('FAISS/ACORN snapshot is incomplete or has a different point count')
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'article_ids.npy'
    receipt = destination / 'id-mapping.json'
    if target.exists():
        if not receipt.exists() or json.loads(receipt.read_text())['snapshot_manifest_sha256'] != fingerprint:
            raise ValueError('Existing ID mapping belongs to another snapshot')
        mapping = ArticleIDMapping(target,receipt=receipt,
            faiss_manifest=storage/'faiss'/asset['asset_id']/'manifest.json',
            acorn_receipt=storage/'acorn-progress'/(asset['asset_id']+'.json'),
            collection=acorn['collection'],require_proof=True)
        if len(mapping.values) != manifest['points']:
            raise ValueError('Existing article-ID mapping has a different point count')
        print('Mapping already complete')
        return
    temporary = target.with_suffix('.npy.tmp')
    if file_sha256(snapshot/'rows.sqlite') != manifest['files']['rows.sqlite']:
        raise ValueError('Source ordinal-to-primary-key file differs from its snapshot manifest')
    count = manifest['points']
    values = np.lib.format.open_memmap(temporary, mode='w+', shape=(count,), dtype=np.int64)
    seen = 0
    with sqlite3.connect((snapshot / 'rows.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
        for point_id, encoded in conn.execute('SELECT point_id,key_json FROM rows ORDER BY point_id'):
            if point_id != seen:
                raise ValueError('Source point ordinals are not contiguous')
            identity = json.loads(encoded)
            logical_id = identity['id']
            if logical_id is None or str(int(logical_id)) != str(logical_id):
                raise ValueError('Article primary key must be an exact integer')
            values[point_id] = int(logical_id)
            seen += 1
    if seen != count or len(np.unique(values)) != count:
        raise ValueError('Article ID mapping is incomplete or contains duplicates')
    values.flush()
    del values
    digest = hashlib.sha256()
    with temporary.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    result = {'points': count, 'snapshot_manifest_sha256': fingerprint,
        'article_ids_sha256': digest.hexdigest(), 'source_mapping': str(snapshot / 'rows.sqlite'),
        'source_mapping_sha256': manifest['files']['rows.sqlite'],
        'semantics': 'FAISS / ACORN point ordinal maps to articles.id', 'unique': True}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps({'points': count, 'destination': str(target), 'unique': True}))


if __name__ == '__main__':
    main()
