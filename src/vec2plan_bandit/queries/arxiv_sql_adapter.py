"""Strict adapter for the four supplied arXiv MATCH/lembed query files.

The input SQL is a query specification, never SQL executed against PostgreSQL.
In particular, ``a.k`` is a top-k annotation and OR filter alternatives are
grouped as one metadata predicate. Unknown syntax fails closed.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Any

from vec2plan_bandit.queries.loader import QueryRecord


MATCH_RE = re.compile(
    r"\bWHERE\s+a\.(?P<column>abstract_embedding|title_embedding)\s+MATCH\s+"
    r"lembed\(\s*'(?P<model>[^']+)'\s*,\s*"
    r"(?P<text>\"(?:\\.|[^\"\\])*\"|'(?:''|[^'])*')\s*\)",
    re.IGNORECASE | re.DOTALL,
)
K_RE = re.compile(r"\bAND\s+a\.k\s*=\s*(?P<k>\d+)\s*;?\s*$", re.IGNORECASE)
DATE = r"\d{4}-\d{2}-\d{2}"
FIELD = r"a\.(license|comments|journal_ref|report_no)"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _predicate(fragment: str) -> tuple[dict[str, Any], ...]:
    fragment = fragment.strip()
    if not fragment:
        return ()
    prefix = re.match(r"^AND\s+", fragment, re.IGNORECASE)
    if prefix is None:
        raise ValueError(f"unsupported filter prefix: {fragment[:100]!r}")
    expression = fragment[prefix.end():].strip()

    match = re.fullmatch(r"(s|au)\.name\s*(=|LIKE)\s*'(.*)'", expression, re.IGNORECASE | re.DOTALL)
    if match:
        owner = "submitter" if match.group(1).lower() == "s" else "author"
        value = match.group(3).replace("''", "'")
        kind = f"{owner}_{'like' if match.group(2).upper() == 'LIKE' else 'eq'}"
        return ({"type": kind, "values": [value]},)

    if re.fullmatch(r"a\.license\s*=\s*'[^']*'(?:\s+OR\s+a\.license\s*=\s*'[^']*')+", expression, re.IGNORECASE):
        values = re.findall(r"a\.license\s*=\s*'([^']*)'", expression, re.IGNORECASE)
        return ({"type": "license_in", "values": list(dict.fromkeys(values))},)

    match = re.fullmatch(rf"{FIELD}\s*(=|LIKE)\s*'(.*)'", expression, re.IGNORECASE | re.DOTALL)
    if match:
        field = match.group(1).lower()
        kind = f"{field}_{'like' if match.group(2).upper() == 'LIKE' else 'eq'}"
        return ({"type": kind, "values": [match.group(3).replace("''", "'")]},)

    for pattern, kind in (
        (rf"a\.update_date\s+BETWEEN\s+'({DATE})'\s+AND\s+'({DATE})'", "update_date_between"),
        (rf"a\.update_date\s*<=\s*'({DATE})'\s+OR\s+a\.update_date\s*>=\s*'({DATE})'", "update_date_outside"),
        (rf"a\.update_date\s*(=|<=|>=)\s*'({DATE})'", None),
    ):
        match = re.fullmatch(pattern, expression, re.IGNORECASE | re.DOTALL)
        if match:
            if kind is None:
                kind = {"=": "update_date_eq", "<=": "update_date_le", ">=": "update_date_ge"}[match.group(1)]
                values = [match.group(2)]
            else:
                values = list(match.groups())
            return ({"type": kind, "values": values},)
    raise ValueError(f"unsupported filter expression: {expression[:200]!r}")


def parse_record(item: dict[str, Any], *, bucket: str, row_number: int) -> tuple[QueryRecord, dict[str, Any]]:
    sql = item.get("sql")
    if not isinstance(sql, str):
        raise ValueError("SQL must be a string")
    matches = list(MATCH_RE.finditer(sql))
    k_match = K_RE.search(sql)
    if len(matches) != 1 or k_match is None or matches[0].end() > k_match.start():
        raise ValueError("expected exactly one MATCH/lembed expression and one trailing a.k annotation")
    match = matches[0]
    if match.group("model") != "all-MiniLM-L6-v2":
        raise ValueError(f"unexpected embedding model: {match.group('model')}")
    original_k = int(k_match.group("k"))
    if not 1 <= original_k <= 10:
        raise ValueError("source k outside 1..10")
    predicates = _predicate(sql[match.end():k_match.start()])
    if bucket == "unfiltered" and predicates:
        raise ValueError("unfiltered source contains a metadata filter")
    quoted_text = match.group("text")
    # Dataset specifications use JSON-style double quotes; released models
    # normally emit SQL single quotes. Preserve either generated string.
    semantic = (json.loads(quoted_text) if quoted_text.startswith('"')
                else quoted_text[1:-1].replace("''", "'"))
    query_id = f"{bucket[0].upper()}{row_number:03d}" if bucket != "unfiltered" else f"U{row_number:03d}"
    # High and medium have distinct first letters. The group key prevents a
    # repeated semantic query from appearing on both sides of a train/test split.
    canonical = " ".join(semantic.casefold().split())
    group_id = "G" + _digest(canonical.encode("utf-8"))[:16]
    provenance = {
        "query_id": query_id,
        "bucket": bucket,
        "source_row": row_number,
        "source_sha256": _digest(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")),
        "original_k": original_k,
        "effective_k": original_k,
        "predicates": list(predicates),
    }
    record = QueryRecord(
        query_id=query_id,
        group_id=group_id,
        question=str(item["question"]),
        semantic_text=semantic,
        embedding_column=match.group("column").lower(),
        k=provenance["effective_k"],
        predicates=predicates,
        source_sql=sql,
        cosine_sql=sql,
        candidate_sql=(),
        source_record_number=row_number,
        source_hash=provenance["source_sha256"],
    )
    return record, provenance


def structural_sql(source_sql: str) -> str:
    """Expose relational structure of a validated MATCH query specification.

    This is only for routing facts, never execution or retrieval proofs. MATCH
    and a.k are annotations in this format. Preserve joins and group the scalar
    filter, including OR alternatives, just as the execution adapter does.
    Validation also handles JSON-quoted embedding text without rewriting text
    inside SQL literals.
    """
    parse_record({'sql': source_sql, 'question': ''}, bucket='structure', row_number=1)
    match = MATCH_RE.search(source_sql)
    end = K_RE.search(source_sql)
    fragment = source_sql[match.end():end.start()].strip()
    prefix = source_sql[:match.start()].rstrip()
    if fragment:
        scalar_filter = re.sub(r'^AND\s+', '', fragment, count=1, flags=re.IGNORECASE)
        return f'{prefix} WHERE ({scalar_filter})'
    return prefix


@lru_cache(maxsize=2048)
def _unfiltered_specification(source_sql):
    """Validate a typed single-relation retrieval specification, not SQL output.

    MATCH and a.k belong to the existing specification adapter. This proof is
    only for main routing; it does not authorize result projection rewrites.
    """
    from sqlglot import exp
    from vec2plan_bandit.queries.lightweight import parsed_select
    record,_=parse_record({'sql':source_sql,'question':''},bucket='unfiltered',row_number=1)
    tree=parsed_select(structural_sql(source_sql))
    tables=list(tree.find_all(exp.Table))
    if (len(tables)!=1 or tables[0].name!='articles' or tables[0].alias_or_name!='a'
            or len(list(tree.find_all(exp.Select)))!=1 or tree.args.get('distinct')
            or any(tree.find(kind) for kind in (exp.Join,exp.Where,exp.CTE,exp.Subquery,
                exp.Group,exp.Having,exp.AggFunc,exp.Window,exp.Offset,exp.Unnest,exp.Lateral,exp.Func))
            or any(tree.args.get(key) for key in ('order','limit','qualify'))):
        return None
    return record.embedding_column,record.k,record.semantic_text


def unfiltered_retrieval_specification(query):
    """Only a matching validated QueryRecord may enable the main FAISS gate."""
    from sqlglot.errors import SqlglotError
    if not isinstance(query,QueryRecord) or query.predicates:
        return False
    try:
        return _unfiltered_specification(query.source_sql)==(
            query.embedding_column,query.k,query.semantic_text)
    except (ValueError,TypeError,SqlglotError):
        return False
