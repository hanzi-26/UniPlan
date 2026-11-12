"""Local ordinal -> logical article IDs shared by FAISS and ACORN.

The array belongs to the immutable source snapshot. No SQL/payload fetch is
needed to materialize an ANN result. Identity errors fail, never become recall
labels for an apparently successful query.
"""
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def integer_id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError('article/point ID must be an integer, never a float or UUID')
    return int(value)


class ArticleIDMapping:
    def __init__(self, path, *, receipt=None, faiss_manifest=None, acorn_receipt=None,
                 collection=None, require_proof=False):
        self.path = Path(path)
        self.receipt = Path(receipt) if receipt else None
        self.faiss_manifest = Path(faiss_manifest) if faiss_manifest else None
        self.acorn_receipt = Path(acorn_receipt) if acorn_receipt else None
        self.collection = collection
        self.require_proof = require_proof
        self._values = None
        self._inverse_order = None

    @property
    def values(self):
        if self._values is None:
            values = np.load(self.path, mmap_mode='r', allow_pickle=False)
            if (values.ndim != 1 or values.dtype != np.int64
                    or len(np.unique(values)) != len(values)):
                raise ValueError('article-ID mapping requires a unique one-dimensional int64 array')
            if self.require_proof and not all((self.receipt, self.faiss_manifest, self.acorn_receipt)):
                raise ValueError('ordinal ID mapping requires mapping, FAISS and ACORN snapshot receipts')
            if self.receipt:
                proof = json.loads(self.receipt.read_text())
                fingerprint = proof.get('snapshot_manifest_sha256')
                if (not fingerprint or proof.get('points') != len(values) or proof.get('unique') is not True
                        or proof.get('article_ids_sha256') != file_sha256(self.path)):
                    raise ValueError('article-ID mapping receipt/hash mismatch')
                for path in (self.faiss_manifest, self.acorn_receipt):
                    if path:
                        asset = json.loads(path.read_text())
                        if asset.get('snapshot_manifest_sha256') != fingerprint:
                            raise ValueError('FAISS/ACORN ID mapping source snapshot mismatch')
                        size = asset.get('points', asset.get('expected'))
                        if size is not None and size != len(values):
                            raise ValueError('ANN snapshot/mapping point-count mismatch')
                        if path == self.acorn_receipt:
                            if (asset.get('status') != 'complete' or asset.get('collection') != self.collection
                                    or asset.get('uploaded') != len(values)):
                                raise ValueError('ACORN collection does not match the ordinal mapping receipt')
                        elif asset.get('smoke_test_passed') is not True:
                            raise ValueError('FAISS source graph has no completed validation receipt')
            elif self.faiss_manifest or self.acorn_receipt:
                raise ValueError('snapshot verification requires an article-ID mapping receipt')
            self._values = values
        return self._values

    def decode(self, point_ids):
        values = self.values
        positions = [integer_id(v) for v in point_ids]
        if any(v < 0 or v >= len(values) for v in positions):
            raise ValueError('ANN point ordinal is outside the article-ID mapping')
        return [int(values[v]) for v in positions]

    def encode(self, article_ids):
        """Map exact logical-ID membership to point ordinals, including gaps.

        Only the immutable snapshot inverse is cached; query-specific member
        sets are computed afresh. Publish the inverse after sorting completes
        so interruption cannot leave a partially built mapping.
        """
        values = self.values
        if self._inverse_order is None:
            order = np.argsort(values)
            self._inverse_order = order
        order = self._inverse_order
        ids = np.fromiter((integer_id(v) for v in article_ids),dtype=np.int64)
        positions = np.searchsorted(values[order],ids)
        if np.any(positions >= len(values)) or np.any(values[order[positions]] != ids):
            raise ValueError('logical article ID is outside the ANN snapshot mapping')
        return order[positions].tolist()


def materialize_article_ids(ids, k):
    """Same stable identity assembly for both ANN executors; keep search order."""
    output, seen = [], set()
    for value in ids:
        value = integer_id(value)
        if value in seen:
            raise ValueError('ANN returned duplicate logical article IDs')
        seen.add(value)
        output.append(value)
    return output[:k]
