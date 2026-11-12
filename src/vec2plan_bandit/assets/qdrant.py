"""Resumable payload-only enrichment of existing ACORN collections.

Vectors and point IDs are not changed. The receipt is written only after every
article has been updated in both collections and payload indexes are present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time

from vec2plan_bandit.assets.indexes import sha256


FIELDS = ("license", "comments", "journal_ref", "report_no")
VERSION = "qdrant-extra-payload-v1"


def enrich(config: dict, *, progress_path: Path, receipt_path: Path, batch_size: int = 512) -> None:
    from qdrant_client import QdrantClient, models
    if batch_size < 1 or batch_size > 2048:
        raise ValueError("batch_size must be in 1..2048")
    root = Path(__file__).resolve().parents[3]
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path
    metadata = resolve(config["metadata"]["extra_path"])
    manifest = resolve(config["metadata"]["extra_manifest"])
    meta = json.loads(manifest.read_text())
    if meta["file_sha256"] != sha256(metadata):
        raise ValueError("extra metadata checksum mismatch")
    source_hash = sha256(manifest)
    collections = config["acorn"]["collections"]
    if receipt_path.exists():
        raise FileExistsError(f"Qdrant extra-payload receipt already exists: {receipt_path}")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": VERSION, "extra_metadata_sha256": source_hash,
             "collections": collections, "last_id": {column: 0 for column in collections}}
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())
        if any(saved.get(key) != state[key] for key in ("version", "extra_metadata_sha256", "collections")):
            raise ValueError("Qdrant enrichment progress belongs to different input data")
        state = saved
    client = QdrantClient(url=config["acorn"]["url"], timeout=600,
                          prefer_grpc=config["acorn"].get("prefer_grpc", False),
                          grpc_port=config["acorn"].get("grpc_port", 6334))
    started = time.monotonic()
    try:
        with sqlite3.connect(metadata.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for column, name in collections.items():
                info = client.get_collection(name)
                if info.points_count != meta["articles"]:
                    raise ValueError(f"point count mismatch before enrichment: {name}")
                while True:
                    rows = db.execute(
                        "SELECT id,license,comments,journal_ref,report_no FROM article_extra "
                        "WHERE id>? ORDER BY id LIMIT ?", (state["last_id"][column], batch_size)).fetchall()
                    if not rows:
                        break
                    operations = []
                    for article_id, *values in rows:
                        payload = {field: value for field, value in zip(FIELDS, values) if value is not None}
                        operations.append(models.SetPayloadOperation(set_payload=models.SetPayload(
                            payload=payload, points=[int(article_id)])))
                    for attempt in range(5):
                        try:
                            client.batch_update_points(name, update_operations=operations, wait=True)
                            break
                        except Exception:
                            if attempt == 4:
                                raise
                            time.sleep(min(30, 2 ** attempt))
                    state["last_id"][column] = int(rows[-1][0])
                    temporary = progress_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state, indent=2) + "\n")
                    temporary.replace(progress_path)
                    if state["last_id"][column] % 100000 < batch_size:
                        print(json.dumps({"stage": "qdrant_extra_payload", "column": column,
                                          "last_id": state["last_id"][column],
                                          "elapsed_s": round(time.monotonic() - started)}), flush=True)
                for field in FIELDS:
                    info = client.get_collection(name)
                    if field not in info.payload_schema:
                        client.create_payload_index(name, field_name=field,
                            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, on_disk=True),
                            wait=True)
                if client.count(name, exact=True).count != meta["articles"]:
                    raise ValueError(f"point count changed during enrichment: {name}")
                print(json.dumps({"stage": "qdrant_extra_column_complete", "column": column,
                                  "points": meta["articles"], "elapsed_s": round(time.monotonic() - started)}), flush=True)
        receipt = {"version": VERSION, "extra_metadata_sha256": source_hash,
                   "collections": collections, "counts": {column: meta["articles"] for column in collections},
                   "fields": list(FIELDS), "payload_indexes_on_disk": True,
                   "elapsed_s": round(time.monotonic() - started)}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"stage": "qdrant_extra_complete", "elapsed_s": receipt["elapsed_s"]}), flush=True)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", type=Path, default=Path("outputs/qdrant-extra-progress.json"))
    parser.add_argument("--receipt", type=Path, default=Path("data/indexes/qdrant-extra-v1.json"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    enrich(json.loads(args.config.read_text()), progress_path=args.progress,
           receipt_path=args.receipt, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
