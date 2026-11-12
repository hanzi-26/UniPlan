"""Streaming binary-vector snapshot for the full migrated arXiv corpus."""
from __future__ import annotations
import json
from pathlib import Path
import struct
import time
import numpy as np
from vec2plan_bandit.assets.indexes import COLUMNS, fresh_directory, seal, rows
from vec2plan_bandit.experiments.collector import MatrixCollector
from vec2plan_bandit.execution.metadata import build_metadata, timestamp

EXPORT_SQL = """WITH author_sets AS (
 SELECT aa.article_id,array_agg(DISTINCT au.name ORDER BY au.name) FILTER(WHERE au.name IS NOT NULL) AS names
 FROM public.article_authors aa JOIN public.authors au ON au.id=aa.author_id GROUP BY aa.article_id
), category_sets AS (
 SELECT ac.article_id,array_agg(DISTINCT c.code ORDER BY c.code) FILTER(WHERE c.code IS NOT NULL) AS codes
 FROM public.article_categories ac JOIN public.categories c ON c.id=ac.category_id GROUP BY ac.article_id
), latest AS (SELECT article_id,max(created::timestamptz) AS created FROM public.versions GROUP BY article_id)
SELECT a.id::bigint,vector_send(a.abstract_embedding),vector_send(a.title_embedding),
 s.name,a.update_date::timestamptz,v.created,aa.names,ac.codes
 FROM public.articles a
 LEFT JOIN public.submitters s ON s.id=NULLIF(a.submitter_id,'')::bigint
 LEFT JOIN author_sets aa ON aa.article_id=a.id::bigint
 LEFT JOIN category_sets ac ON ac.article_id=a.id::bigint
 LEFT JOIN latest v ON v.article_id=a.id::bigint
 ORDER BY a.id::bigint"""


def decode_vector(value: bytes, dimension: int) -> np.ndarray:
    if value is None or len(value) != 4 + 4 * dimension or struct.unpack_from(">h", value)[0] != dimension:
        raise ValueError("missing or malformed pgvector binary value")
    vector = np.frombuffer(value, dtype=">f4", offset=4).astype(np.float32)
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 0:
        raise ValueError("nonfinite or zero vector in source data")
    return vector


def export_binary(config: dict, destination: Path) -> None:
    fresh_directory(destination)
    worker = MatrixCollector(destination / "runtime", config, read_only=True)
    started = time.monotonic()
    dimension = int(config["embedding"]["dimension"])
    compare_dates = config.get("metadata", {}).get("latest_update_comparison", "timestamp") == "date"
    try:
        connection = worker.connection
        with connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute("SET LOCAL statement_timeout=0")
            cursor.execute("SET LOCAL work_mem='64MB'")
            cursor.execute("SET LOCAL timezone='UTC'")
            cursor.execute("SELECT pg_current_snapshot()::text,datcollate FROM pg_database WHERE datname=current_database()")
            snapshot, collation = cursor.fetchone()
            cursor.execute("SELECT count(*),count(abstract_embedding),count(title_embedding) FROM public.articles")
            count, abstracts, titles = cursor.fetchone()
            if count == 0 or count != abstracts or count != titles:
                raise ValueError("binary snapshot currently requires both vectors on every article")
        ids = np.lib.format.open_memmap(destination / "article_ids.npy", mode="w+", dtype=np.int64, shape=(count,))
        vectors = {column: np.lib.format.open_memmap(destination / f"{column}.npy", mode="w+", dtype=np.float32,
                                                   shape=(count, dimension)) for column in COLUMNS}
        print(json.dumps({"stage": "export", "total_articles": count, "binary_vectors_gb": count * dimension * 8 / 1e9}), flush=True)
        processed = 0
        with connection.cursor(name="snapshot_articles") as cursor, (destination / "articles.jsonl").open("x") as handle:
            cursor.itersize = 1024
            cursor.execute(EXPORT_SQL)
            for article_id, abstract, title, submitter, date, latest, authors, categories in cursor:
                if processed and article_id <= ids[processed - 1]:
                    raise ValueError("article IDs must be strictly increasing")
                ids[processed] = article_id
                for column, value in zip(COLUMNS, (abstract, title)):
                    vectors[column][processed] = decode_vector(value, dimension)
                latest_match = bool(date and latest and (date.date() == latest.date() if compare_dates else date == latest))
                payload = {"submitter_name": submitter, "author_names": authors or [], "category_codes": categories or [],
                           "update_date": timestamp(date.isoformat()).isoformat() if date else None,
                           "latest_version_date": timestamp(latest.isoformat()).isoformat() if latest else None,
                           "is_latest_update": latest_match}
                handle.write(json.dumps({"article_id": article_id, "payload": payload, "vector_columns": list(COLUMNS)},
                                        ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += 1
                if processed % 100000 == 0:
                    print(json.dumps({"stage": "export", "articles": processed, "total": count,
                                      "elapsed_s": round(time.monotonic() - started)}), flush=True)
        if processed != count:
            raise ValueError("snapshot count mismatch")
        ids.flush()
        for vector in vectors.values():
            vector.flush()
        del ids, vectors
    finally:
        worker.close()
    print(json.dumps({"stage": "local_metadata", "articles": count}), flush=True)
    build_metadata(destination / "metadata.sqlite", rows(destination / "articles.jsonl"))
    seal(destination, {"snapshot_id": config["snapshot_id"], "postgres_snapshot": snapshot,
        "postgres_collation": collation, "articles": count, "embedding": config["embedding"],
        "embedding_model_provenance": "configuration_only; original model requires confirmation",
        "vector_storage": "npy_float32", "vector_columns": list(COLUMNS), "latest_update_comparison": "date" if compare_dates else "timestamp",
        "source_layout": "arxiv_public_migrated", "export_elapsed_s": round(time.monotonic() - started)})
    print(json.dumps({"stage": "snapshot_complete", "articles": count, "elapsed_s": round(time.monotonic() - started)}), flush=True)
