"""Normalize prepared truth without confusing logical IDs and ANN ordinals."""
from vec2plan_bandit.execution.article_identity import integer_id


def projection_rows_equal(query, projected, reference):
    """Audit identities without inventing order for an unordered SELECT.

    Outer ORDER BY requires identical row order. An inner CTE's ORDER BY only
    determines its own top-k; it does not order the outer join result. In that
    case compare multisets, preserving every duplicate and identity component.
    """
    from collections import Counter
    from vec2plan_bandit.queries.lightweight import parsed_select
    left = [tuple(row) for row in projected]
    right = [tuple(row) for row in reference]
    if parsed_select(query.source_sql).args.get('order') is not None:
        return left == right
    return Counter(left) == Counter(right)


def normalized_truth(row):
    result = dict(row)
    joined = row.get('topk_row_ids')
    scalar = row.get('topk_ids')
    if joined is None and scalar is None:
        raise ValueError(f'{row.get("query_id")}: prepared ground truth has no result identities')
    if scalar is not None:
        if not isinstance(scalar, list):
            raise ValueError('arXiv topk_ids must contain logical integer article IDs')
        converted = [[integer_id(value)] for value in scalar]
        if joined is not None and joined != converted:
            raise ValueError('conflicting scalar and joined-row ground truth identities')
        joined = converted
    if not isinstance(joined, list) or any(not isinstance(v,(list,tuple)) or not v for v in joined):
        raise ValueError('joined ground truth must be a list of nonempty identity rows')
    result['topk_row_ids'] = [list(values) for values in joined]
    return result


def truth_by_query(rows, statistics=None):
    result = {}
    for raw in rows:
        row = normalized_truth(raw)
        qid = row['query_id']
        if qid in result:
            raise ValueError(f'duplicate ground truth query ID: {qid}')
        if 'statistics' not in row and statistics is not None:
            row['statistics'] = statistics[qid]
        result[qid] = row
    return result
