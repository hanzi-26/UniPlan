"""Build a vector-free metadata sidecar for the four arXiv query files.

The base vector snapshot and FAISS indexes remain unchanged. The sidecar is
usable only when its manifest is bound to the base snapshot manifest and the
entire ordered article-ID sequence has been checked during export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from vec2plan_bandit.assets.indexes import sha256
from vec2plan_bandit.experiments.collector import MatrixCollector


VERSION = "arxiv-extra-metadata-v1"


def build(config: dict, output: Path) -> dict:
    snapshot = Path(config["metadata"]["snapshot_dir"])
    if not snapshot.is_absolute():
        snapshot = Path(__file__).resolve().parents[3] / snapshot
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("metadata sidecar or its manifest already exists")
    ids = np.load(snapshot / "article_ids.npy", mmap_mode="r")
    worker = MatrixCollector(output.parent / "extra-metadata-runtime", config, read_only=True)
    started = time.monotonic()
    count = 0
    try:
        with worker.connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute("SET LOCAL statement_timeout=0")
            cursor.execute("SELECT pg_current_snapshot()::text")
            pg_snapshot = cursor.fetchone()[0]
        with sqlite3.connect(output) as db:
            db.execute("CREATE TABLE article_extra (id INTEGER PRIMARY KEY, license TEXT, comments TEXT, journal_ref TEXT, report_no TEXT)")
            db.execute("CREATE TABLE metadata_values (field TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(field,value))")
            with worker.connection.cursor(name="extra_metadata_articles") as cursor:
                cursor.itersize = 2048
                cursor.execute("SELECT id::bigint,license,comments,journal_ref,report_no "
                               "FROM public.articles ORDER BY id::bigint")
                for article_id, license_value, comments, journal_ref, report_no in cursor:
                    if count >= len(ids) or int(ids[count]) != int(article_id):
                        raise ValueError(f"source/article-ID mismatch at row {count}")
                    db.execute("INSERT INTO article_extra VALUES (?,?,?,?,?)",
                               (article_id, license_value, comments, journal_ref, report_no))
                    if comments is not None:
                        db.execute("INSERT OR IGNORE INTO metadata_values VALUES (?,?)", ("comments", comments))
                    if journal_ref is not None:
                        db.execute("INSERT OR IGNORE INTO metadata_values VALUES (?,?)", ("journal_ref", journal_ref))
                    count += 1
                    if count % 100000 == 0:
                        print(json.dumps({"stage": "extra_metadata", "rows": count,
                                          "elapsed_s": round(time.monotonic() - started)}), flush=True)
            if count != len(ids):
                raise ValueError(f"source/article-ID count mismatch: {count} != {len(ids)}")
            db.execute("CREATE INDEX metadata_values_field ON metadata_values(field)")
            db.execute("ANALYZE")
        receipt = {"version": VERSION, "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
                   "postgres_snapshot": pg_snapshot, "articles": count,
                   "file_sha256": sha256(output), "fields": ["license", "comments", "journal_ref", "report_no"],
                   "elapsed_s": round(time.monotonic() - started)}
        manifest_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        worker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(json.loads(args.config.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
