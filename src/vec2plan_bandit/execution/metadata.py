"""Query-independent local metadata snapshot; never calls PostgreSQL at serving time."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable


class VocabularyExpansionTooLarge(ValueError):
    """An exact ID filter is needed instead of a large keyword vocabulary."""


def sql_like(pattern: str) -> re.Pattern[str]:
    """Case-sensitive SQL LIKE with %, _, and PostgreSQL's default backslash escape."""
    parts = []
    escaped = False
    for char in pattern:
        if escaped:
            parts.append(re.escape(char))
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            parts.append(".*" if char == "%" else "." if char == "_" else re.escape(char))
    if escaped:
        raise ValueError("LIKE pattern must not end with an escape character")
    return re.compile("".join(parts), re.DOTALL)


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def matches(payload: dict[str, Any], predicates: Iterable[dict[str, Any]]) -> bool:
    fields = {"author": "author_names", "category": "category_codes", "submitter": "submitter_name",
              "license": "license", "comments": "comments", "journal_ref": "journal_ref", "report_no": "report_no"}
    for predicate in predicates:
        kind, values = predicate["type"], predicate["values"]
        if kind == "latest_update":
            ok = payload.get("is_latest_update") is True
        elif kind == "update_date_range":
            date = payload.get("update_date")
            ok = date is not None and timestamp(values[0]) <= timestamp(date) < timestamp(values[1])
        elif kind.startswith("update_date_"):
            date = payload.get("update_date")
            if date is None:
                ok = False
            else:
                day = timestamp(date).date()
                start = timestamp(values[0]).date()
                if kind == "update_date_eq":
                    ok = day == start
                elif kind == "update_date_le":
                    ok = day <= start
                elif kind == "update_date_ge":
                    ok = day >= start
                elif kind == "update_date_between":
                    ok = start <= day <= timestamp(values[1]).date()
                elif kind == "update_date_outside":
                    ok = day <= start or day >= timestamp(values[1]).date()
                else:
                    raise ValueError(f"unsupported metadata predicate: {kind}")
        else:
            prefix, operation = kind.rsplit("_", 1)
            if prefix not in fields or operation not in {"eq", "in", "like"}:
                raise ValueError(f"unsupported metadata predicate: {kind}")
            current = payload.get(fields[prefix])
            actual = current if isinstance(current, list) else [current]
            if operation == "like":
                pattern = sql_like(str(values[0]))
                ok = any(v is not None and pattern.fullmatch(str(v)) for v in actual)
            else:
                allowed = values[:1] if operation == "eq" else values
                ok = any(v is not None and v in allowed for v in actual)
        if not ok:
            return False
    return True


def build_metadata(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("CREATE TABLE authors (name TEXT PRIMARY KEY)")
        for row in rows:
            payload = {**row["payload"], "_vector_columns": list(row.get("vectors", row.get("vector_columns", [])))}
            db.execute("INSERT INTO articles VALUES (?, ?)", (int(row["article_id"]), json.dumps(payload)))
            db.executemany("INSERT OR IGNORE INTO authors VALUES (?)", [(v,) for v in payload.get("author_names", [])])
            count += 1
    return count


class LocalMetadata:
    def __init__(self, path: Path, extra_path: Path | None = None):
        self.path = path
        self.extra_path = extra_path
        self._db: sqlite3.Connection | None = None
        self._sample: list[dict[str, Any]] | None = None
        self._count = 0

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"local metadata snapshot is missing: {self.path}")
            self._db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            if self.extra_path is not None:
                if not self.extra_path.is_file():
                    raise FileNotFoundError(f"extra metadata sidecar is missing: {self.extra_path}")
                self._db.execute("ATTACH DATABASE ? AS extras", (self.extra_path.resolve().as_uri() + "?mode=ro",))
        return self._db

    def available(self, **kwargs: Any) -> bool:
        if not self.path.is_file():
            return False
        self.db.execute("SELECT id FROM articles LIMIT 1").fetchall()
        return True

    def author_values_like(self, pattern: str) -> list[str]:
        matcher = sql_like(pattern)
        # No query-specific precomputed eligibility or ground-truth information.
        return [name for (name,) in self.db.execute("SELECT name FROM authors ORDER BY name")
                if matcher.fullmatch(name)]

    def field_values_like(self, field: str, pattern: str, *, max_values: int = 8192) -> list[str]:
        if field not in {"comments", "journal_ref"}:
            raise ValueError(f"unsupported metadata vocabulary: {field}")
        if self.extra_path is None:
            raise ValueError("extra metadata sidecar is required for this LIKE filter")
        matcher = sql_like(pattern)
        values = []
        for (value,) in self.db.execute(
            "SELECT value FROM extras.metadata_values WHERE field=? ORDER BY value", (field,)):
            if matcher.fullmatch(value):
                values.append(value)
                if len(values) > max_values:
                    raise VocabularyExpansionTooLarge(f"{field} LIKE expansion exceeds {max_values} distinct values")
        return values

    def field_ids_like(self, field: str, pattern: str) -> Iterable[int]:
        """Exact LIKE membership from the same snapshot, computed per request.

        Large comment strings need not be sent back to Qdrant as MatchAny
        keywords. Return logical article IDs; the caller must map them to the
        collection's point IDs before constructing a HasId condition.
        """
        if field not in {'comments','journal_ref'}:
            raise ValueError(f'unsupported metadata vocabulary: {field}')
        if self.extra_path is None:
            raise ValueError('extra metadata sidecar is required for this LIKE filter')
        matcher = sql_like(pattern)
        for article_id,value in self.db.execute(f'SELECT id,{field} FROM extras.article_extra WHERE {field} IS NOT NULL'):
            if matcher.fullmatch(value):
                yield int(article_id)

    def _with_extra(self, article_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.extra_path is None:
            return payload
        row = self.db.execute("SELECT license,comments,journal_ref,report_no FROM extras.article_extra WHERE id=?",
                              (article_id,)).fetchone()
        if row is None:
            raise ValueError(f"extra metadata missing article {article_id}")
        return {**payload, **dict(zip(("license", "comments", "journal_ref", "report_no"), row))}

    def prepare_statistics(self) -> None:
        """Publish the corpus reservoir only after a complete, interruptible build."""
        if self._sample is None:
            import random
            rng = random.Random(42)
            sample = []
            count = 0
            for i, (article_id, encoded) in enumerate(self.db.execute("SELECT id,payload FROM articles ORDER BY id")):
                count = i + 1
                if i < 2048:
                    sample.append((article_id, encoded))
                else:
                    position = rng.randrange(i + 1)
                    if position < 2048:
                        sample[position] = (article_id, encoded)
            sample = [self._with_extra(article_id, json.loads(encoded))
                      for article_id, encoded in sample]
            self._count = count
            self._sample = sample

    def estimate(self, query: Any) -> dict[str, float]:
        """Deterministic query-independent reservoir; not exact query cardinality."""
        self.prepare_statistics()
        selected = [p for p in self._sample if query.embedding_column in p.get("_vector_columns", [])]
        selectivity = sum(matches(p, query.predicates) for p in selected) / max(1, len(selected))
        estimated_corpus = self._count * len(selected) / max(1, len(self._sample))
        return {"estimated_selectivity": selectivity, "candidate_count": selectivity * estimated_corpus,
                "corpus_size": estimated_corpus}

    def eligible_ids(self, query: Any, candidate_ids: list[int]) -> set[int]:
        result = set()
        for start in range(0, len(candidate_ids), 500):
            chunk = candidate_ids[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for article_id, encoded in self.db.execute(f"SELECT id,payload FROM articles WHERE id IN ({placeholders})", chunk):
                if matches(self._with_extra(article_id, json.loads(encoded)), query.predicates):
                    result.add(article_id)
        return result

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
