"""Exact merge of existing ANN shard results; no extra vector search."""
import numpy as np


def merge_shards(parts, count, metric):
    """Match sorting by (distance, corpus row ID), including cutoff ties."""
    if count <= 0 or not parts:
        return []
    if metric not in ('l2', 'cosine'):
        raise ValueError('Unknown shard metric')
    score_parts, id_parts = [], []
    for scores, ids, start in parts:
        scores, ids = np.asarray(scores), np.asarray(ids)
        valid = ids >= 0
        if valid.any():
            score_parts.append(scores[valid] if metric == 'l2' else -scores[valid])
            id_parts.append(ids[valid].astype('int64') + start)
    if not id_parts:
        return []
    distance, ids = np.concatenate(score_parts), np.concatenate(id_parts)
    if len(ids) > count:
        threshold = np.partition(distance, count-1)[count-1]
        better = np.flatnonzero(distance < threshold)
        tied = np.flatnonzero(distance == threshold)
        needed = count-len(better)
        if len(tied) > needed:
            tied = tied[np.argpartition(ids[tied], needed-1)[:needed]]
        selected = np.concatenate((better, tied))
        distance, ids = distance[selected], ids[selected]
    return ids[np.lexsort((ids, distance))[:count]].tolist()
