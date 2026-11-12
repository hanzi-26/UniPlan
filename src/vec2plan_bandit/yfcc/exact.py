"""Bounded-memory exact filtered L2 search over the original uint8 corpus.

Every eligible vector is evaluated; no ANN candidate or prepared answer is used.
Successful completion certifies exact top-k with corpus-row-ID tie breaking.
"""
import numpy as np


def filtered_l2(raw, scalars, query, vector, *, check, filter_batch=65536, vector_batch=2048):
    if raw.dtype != np.dtype('uint8'):
        raise ValueError('Exact YFCC L2 requires the original uint8 corpus')
    if filter_batch < 1 or vector_batch < 1 or query.k < 1:
        raise ValueError('Invalid exact-search batch size or k')
    vector = np.asarray(vector, dtype='float64')
    if vector.shape != (raw.shape[1],) or not np.all(np.isfinite(vector)):
        raise ValueError('Invalid exact-search vector')
    # Raw YFCC queries are uint8 reference vectors, represented exactly in float32.
    if np.any(vector != np.floor(vector)) or np.any((vector < 0) | (vector > 255)):
        raise ValueError('Exact YFCC L2 requires a raw reference vector')
    equal, ranges, ptr, tags = scalars
    if len(equal) != len(raw) or len(ranges) != len(raw) or len(ptr) != len(raw) + 1:
        raise ValueError('Metadata/corpus length mismatch')
    best_ids = np.empty(0, dtype='int64')
    best_distances = np.empty(0, dtype='float64')
    eligible = 0
    for start in range(0, len(raw), filter_batch):
        check()
        end = min(start + filter_batch, len(raw))
        keep = np.ones(end - start, dtype=bool)
        for predicate in query.item['scalar_filters']:
            check()
            field, op, value = predicate['field'], predicate['operator'], predicate['value']
            if predicate.get('logic') != 'and':
                raise ValueError('Only AND filters supported')
            if field == 'equal' and op == '==':
                keep &= equal[start:end] == value
            elif field == 'range' and op in ('>', '>=', '<', '<='):
                compare = {'>': np.greater, '>=': np.greater_equal,
                           '<': np.less, '<=': np.less_equal}[op]
                keep &= compare(ranges[start:end].astype('float64'), value)
            elif field == 'tags' and op == 'contains':
                begin, finish = int(ptr[start]), int(ptr[end])
                positions = np.flatnonzero(tags[begin:finish] == int(value)) + begin
                matched = np.searchsorted(ptr[start:end + 1], positions, side='right') - 1
                tag_mask = np.zeros(end - start, dtype=bool)
                tag_mask[matched] = True
                keep &= tag_mask
            else:
                raise ValueError(f'Unsupported filter {field} {op}')
        ids = np.flatnonzero(keep) + start
        eligible += len(ids)
        for offset in range(0, len(ids), vector_batch):
            check()
            batch_ids = ids[offset:offset + vector_batch]
            # Integer inputs bound squared distances below 2**53, so float64
            # subtraction and accumulation are exact for the 1920D corpus.
            values = raw[batch_ids].astype('float64')
            values -= vector
            distances = np.einsum('ij,ij->i', values, values)
            if len(distances) > query.k:
                cutoff = np.partition(distances, query.k - 1)[query.k - 1]
                selected = distances <= cutoff  # Retain cutoff ties before ID ordering.
                batch_ids, distances = batch_ids[selected], distances[selected]
            merged_ids = np.concatenate((best_ids, batch_ids))
            merged_distances = np.concatenate((best_distances, distances))
            order = np.lexsort((merged_ids, merged_distances))[:query.k]
            best_ids, best_distances = merged_ids[order], merged_distances[order]
    check()  # Never certify an incomplete/deadline-exceeded scan.
    return dict(row_ids=[[int(i)] for i in best_ids],
                exact_certificate=dict(complete=True, metric='squared_l2',
                    corpus_rows=len(raw), eligible_rows=eligible, evaluated_vectors=eligible,
                    returned_rows=len(best_ids), tie_rule='distance_asc_then_row_id_asc',
                    ground_truth_or_query_result_cache_used=False))
