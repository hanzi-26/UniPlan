from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEMBED_RE = re.compile(r"lembed\(\s*'(?P<model>[^']+)'\s*,\s*'(?P<text>(?:''|[^'])*)'\s*\)", re.I)
EMBEDDING_COLUMN_RE = re.compile(r'"?(abstract_embedding|title_embedding)"?\s*<->', re.I)
LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.I)
NAME_EQ_RE = re.compile(r'"?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)"?\."?name"?\s*=\s*\'(?P<value>(?:\'\'|[^\'])*)\'', re.I)
NAME_IN_RE = re.compile(r'"?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)"?\."?name"?\s+IN\s*\((?P<values>[^)]*)\)', re.I)
NAME_LIKE_RE = re.compile(r'"?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)"?\."?name"?\s+LIKE\s+\'(?P<value>[^\']+)\'', re.I)
CODE_EQ_RE = re.compile(r'"?[A-Za-z_][A-Za-z0-9_]*"?\."?code"?\s*=\s*\'(?P<value>[^\']+)\'', re.I)
CODE_IN_RE = re.compile(r'"?[A-Za-z_][A-Za-z0-9_]*"?\."?code"?\s+IN\s*\((?P<values>[^)]*)\)', re.I)
UPDATE_DATE_RANGE_RE = re.compile(
    r'"?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)"?\."?update_date"?\s*>=\s*\'(?P<start>[^\']+)\'\s*'
    r'AND\s*"?(?P=alias)"?\."?update_date"?\s*<\s*\'(?P<end>[^\']+)\'',
    re.I,
)
JOIN_RE = re.compile(r'JOIN\s+"?(?P<table>submitters|authors)"?\s+(?:AS\s+)?"?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)"?', re.I)
STRING_RE = re.compile(r"'((?:''|[^'])*)'")


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    group_id: str
    question: str
    semantic_text: str
    embedding_column: str
    k: int
    predicates: tuple[dict[str, Any], ...]
    source_sql: str
    cosine_sql: str
    candidate_sql: tuple[str, ...]
    source_record_number: int
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["predicates"] = list(self.predicates)
        value["candidate_sql"] = list(self.candidate_sql)
        return value


def _predicates(sql: str) -> tuple[dict[str, Any], ...]:
    aliases = {m.group("alias").lower(): m.group("table").lower() for m in JOIN_RE.finditer(sql)}
    values: list[dict[str, Any]] = []
    for match in NAME_EQ_RE.finditer(sql):
        table = aliases.get(match.group("alias").lower())
        if table:
            values.append({"type": table[:-1] + "_eq", "values": [match.group("value").replace("''", "'")]})
    for match in NAME_IN_RE.finditer(sql):
        table = aliases.get(match.group("alias").lower())
        if table:
            values.append({"type": table[:-1] + "_in", "values": [v.replace("''", "'") for v in STRING_RE.findall(match.group("values"))]})
    for match in NAME_LIKE_RE.finditer(sql):
        table = aliases.get(match.group("alias").lower())
        if table:
            values.append({"type": table[:-1] + "_like", "values": [match.group("value")]})
    for match in CODE_EQ_RE.finditer(sql):
        values.append({"type": "category_eq", "values": [match.group("value")]})
    for match in CODE_IN_RE.finditer(sql):
        values.append({"type": "category_in", "values": STRING_RE.findall(match.group("values"))})
    for match in UPDATE_DATE_RANGE_RE.finditer(sql):
        values.append({
            "type": "update_date_range",
            "values": [match.group("start"), match.group("end")],
        })
    if '"LatestVersions"' in sql and '"update_date" = "lv"."latest_version_date"' in sql:
        values.append({"type": "latest_update", "values": []})
    return tuple(values)


def load_source_queries(path: Path) -> list[QueryRecord]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("query source must be a non-empty JSON array")
    records: list[QueryRecord] = []
    for index, item in enumerate(payload, 1):
        sql = str(item.get("sql", ""))
        embeddings = list(LEMBED_RE.finditer(sql))
        columns = EMBEDDING_COLUMN_RE.findall(sql)
        limits = LIMIT_RE.findall(sql)
        candidates = item.get("sql_candidate")
        if len(embeddings) != 1 or len(columns) != 1 or not limits:
            raise ValueError(f"Q{index:03d}: expected one embedding expression/column and at least one LIMIT")
        if not isinstance(candidates, list) or len(candidates) != 5:
            raise ValueError(f"Q{index:03d}: expected exactly five candidate SQL statements")
        query_id = f"Q{index:03d}"
        records.append(QueryRecord(
            query_id=query_id,
            group_id=f"G{index:03d}",
            question=str(item["question"]),
            semantic_text=embeddings[0].group("text").replace("''", "'"),
            embedding_column=columns[0].lower(),
            k=int(limits[-1]),
            predicates=_predicates(sql),
            source_sql=sql,
            cosine_sql=sql.replace("<->", "<=>"),
            candidate_sql=tuple(str(value).replace("<->", "<=>") for value in candidates),
            source_record_number=index,
            source_hash=hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        ))
    return records


def split_queries(records: list[QueryRecord], seed: int = 42, sizes: tuple[int, int, int] = (39, 9, 9)) -> dict[str, list[str]]:
    if sum(sizes) != len(records):
        raise ValueError("split sizes must equal query count")
    groups = [record.group_id for record in records]
    if len(groups) != len(set(groups)):
        raise ValueError("original query groups must be unique before candidate expansion")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train, dev, test = sizes
    return {
        "train": [r.query_id for r in shuffled[:train]],
        "dev": [r.query_id for r in shuffled[train:train + dev]],
        "test": [r.query_id for r in shuffled[train + dev:train + dev + test]],
    }
