"""Reproducible, non-overwriting snapshot export and index construction commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vec2plan_bandit.execution.metadata import build_metadata, timestamp
from vec2plan_bandit.assets.build_resources import build_threads

COLUMNS = ("abstract_embedding", "title_embedding")
VERSION = "article-id-snapshot-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def fresh_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def seal(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = {"version": VERSION, **metadata,
               "files": {str(p.relative_to(directory)): sha256(p)
                         for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"}}
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def verify(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "manifest.json").read_text())
    if payload.get("version") != VERSION:
        raise ValueError("unsupported asset manifest")
    for name, expected in payload["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"asset checksum mismatch: {name}")
    return payload


def export_snapshot(config: dict[str, Any], destination: Path) -> None:
    """One repeatable-read PG transaction; vectors and metadata share a snapshot."""
    from vec2plan_bandit.experiments.collector import MatrixCollector
    if config.get("database", {}).get("layout") == "arxiv_public_migrated":
        from vec2plan_bandit.assets.postgres import export_binary
        return export_binary(config, destination)
    fresh_directory(destination)
    collector = MatrixCollector(destination / "runtime", config, read_only=True)
    sql = """SELECT a.id, a.abstract_embedding::text, a.title_embedding::text,
      s.name, a.update_date,
      COALESCE(a.update_date = (SELECT max(v.created) FROM arxiv_raw.versions v WHERE v.article_id=a.id), false),
      ARRAY(SELECT DISTINCT au.name FROM arxiv_raw.article_authors aa JOIN arxiv_raw.authors au ON au.id=aa.author_id WHERE aa.article_id=a.id AND au.name IS NOT NULL ORDER BY au.name),
      ARRAY(SELECT DISTINCT c.code FROM arxiv_raw.article_categories ac JOIN arxiv_raw.categories c ON c.id=ac.category_id WHERE ac.article_id=a.id AND c.code IS NOT NULL ORDER BY c.code)
      FROM arxiv_exp.articles a LEFT JOIN arxiv_raw.submitters s ON s.id=a.submitter_id ORDER BY a.id"""
    count = 0
    try:
        collector.connection.rollback()
        with collector.connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute("SELECT datcollate FROM pg_database WHERE datname=current_database()")
            collation = cursor.fetchone()[0]
            cursor.execute("SELECT pg_current_snapshot()::text")
            snapshot = cursor.fetchone()[0]
        with collector.connection.cursor(name="export_articles") as cursor, (destination / "articles.jsonl").open("x") as handle:
            cursor.execute(sql)
            for article_id, abstract, title, submitter, date, latest, authors, categories in cursor:
                payload = {"submitter_name": submitter, "author_names": authors,
                           "category_codes": categories, "is_latest_update": latest,
                           "update_date": timestamp(date.isoformat()).isoformat() if date else None}
                vectors = {column: json.loads(value) for column, value in zip(COLUMNS, (abstract, title)) if value is not None}
                handle.write(json.dumps({"article_id": int(article_id), "payload": payload, "vectors": vectors}) + "\n")
                count += 1
    finally:
        collector.close()
    build_metadata(destination / "metadata.sqlite", rows(destination / "articles.jsonl"))
    seal(destination, {"snapshot_id": config["snapshot_id"], "postgres_snapshot": snapshot,
                       "postgres_collation": collation, "articles": count, "embedding": config["embedding"],
                       "predicate_contract": "case-sensitive deterministic SQL LIKE; naive dates interpreted as UTC"})


def vector_rows(snapshot: Path, column: str):
    manifest = json.loads((snapshot / "manifest.json").read_text())
    binary = manifest.get("vector_storage") == "npy_float32"
    vectors = np.load(snapshot / f"{column}.npy", mmap_mode="r") if binary else None
    ids = np.load(snapshot / "article_ids.npy", mmap_mode="r") if binary else None
    for index, row in enumerate(rows(snapshot / "articles.jsonl")):
        if binary:
            if int(ids[index]) != int(row["article_id"]):
                raise ValueError("snapshot vector/metadata ID alignment mismatch")
            yield row, vectors[index]
        elif column in row["vectors"]:
            yield row, row["vectors"][column]


def build_faiss(snapshot: Path, destination: Path, *, m: int = 32, ef_construction: int = 200, threads: int = 1) -> None:
    import faiss
    manifest = verify(snapshot)
    fresh_directory(destination)
    dimension = int(manifest["embedding"]["dimension"])
    if threads < 1:
        raise ValueError("positive thread count required")
    faiss.omp_set_num_threads(threads)
    counts = {}
    for column in COLUMNS:
        folder = destination / column
        folder.mkdir()
        index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        ids = []
        batch = []
        started = time.monotonic()
        last_reported = 0
        for row, vector in vector_rows(snapshot, column):
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (dimension,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                raise ValueError(f"invalid vector: {row['article_id']} {column}")
            ids.append(int(row["article_id"]))
            batch.append(vector / np.linalg.norm(vector))
            if len(batch) == 4096:
                index.add(np.asarray(batch))
                batch = []
                if len(ids) - last_reported >= 100000:
                    print(json.dumps({"stage": "faiss", "column": column, "indexed": len(ids),
                                      "elapsed_s": round(time.monotonic() - started)}), flush=True)
                    last_reported = len(ids)
        if batch:
            index.add(np.asarray(batch))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"empty or duplicate IDs in {column}")
        faiss.write_index(index, str(folder / "index.faiss"))
        np.save(folder / "article_ids.npy", np.asarray(ids, dtype=np.int64))
        counts[column] = len(ids)
        print(json.dumps({"stage": "faiss_column_complete", "column": column, "indexed": len(ids),
                          "elapsed_s": round(time.monotonic() - started)}), flush=True)
        del index, ids
    seal(destination, {"snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
                       "dimension": dimension, "distance": "normalized_inner_product", "counts": counts,
                       "m": m, "ef_construction": ef_construction, "build_threads": threads,
                       "faiss_version": faiss.__version__})


def import_qdrant(snapshot: Path, config: dict[str, Any], receipt: Path) -> None:
    from qdrant_client import QdrantClient, models
    manifest = verify(snapshot)
    if receipt.exists():
        raise FileExistsError(receipt)
    client = QdrantClient(url=config["acorn"]["url"], timeout=600,
                         prefer_grpc=config["acorn"].get("prefer_grpc", False),
                         grpc_port=config["acorn"].get("grpc_port", 6334))
    counts = {}
    collections = config["acorn"]["collections"]
    try:
        for name in collections.values():
            if client.collection_exists(name):
                raise FileExistsError(f"Qdrant collection already exists; use a new versioned name: {name}")
        for column, name in collections.items():
            if column not in COLUMNS:
                raise ValueError(f"unsupported vector column: {column}")
            client.create_collection(name, vectors_config=models.VectorParams(
                size=int(manifest["embedding"]["dimension"]), distance=models.Distance.COSINE, on_disk=True),
                on_disk_payload=True,
                hnsw_config=models.HnswConfigDiff(m=0, ef_construct=200, full_scan_threshold=0,
                                                on_disk=True, max_indexing_threads=int(
                                                    config["acorn"].get("max_indexing_threads", 2))),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,
                    max_optimization_threads=int(config["acorn"].get("max_optimization_threads", 3)),
                    default_segment_number=int(config["acorn"].get("target_segments", 12))))
            for field in ("submitter_name", "author_names", "category_codes"):
                client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD, wait=True)
            client.create_payload_index(name, "update_date", models.PayloadSchemaType.DATETIME, wait=True)
            client.create_payload_index(name, "is_latest_update", models.PayloadSchemaType.BOOL, wait=True)
            batch = []
            count = 0
            started = time.monotonic()
            last_reported = 0
            for row, vector in vector_rows(snapshot, column):
                values = np.asarray(vector, dtype=np.float32)
                if values.shape != (int(manifest["embedding"]["dimension"]),) or not np.all(np.isfinite(values)) or np.linalg.norm(values) == 0:
                    raise ValueError("invalid Qdrant vector")
                payload = {k: v for k, v in row["payload"].items() if v is not None}
                batch.append(models.PointStruct(id=int(row["article_id"]), vector=values.tolist(), payload=payload))
                count += 1
                if len(batch) == 1024:
                    client.upsert(name, batch, wait=True)
                    batch = []
                    if count - last_reported >= 100000:
                        print(json.dumps({"stage": "qdrant_upload", "column": column, "points": count,
                                          "elapsed_s": round(time.monotonic() - started)}), flush=True)
                        last_reported = count
            if batch:
                client.upsert(name, batch, wait=True)
            if client.count(name, exact=True).count != count:
                raise ValueError("Qdrant import count mismatch")
            counts[column] = count
            client.update_collection(name, hnsw_config=models.HnswConfigDiff(
                                         m=32, max_indexing_threads=int(
                                             config["acorn"].get("max_indexing_threads", 2))),
                                     optimizers_config=models.OptimizersConfigDiff(
                                         indexing_threshold=int(config["acorn"].get("indexing_threshold_kb", 1)),
                                         max_optimization_threads=int(
                                             config["acorn"].get("max_optimization_threads", 3)),
                                         default_segment_number=int(config["acorn"].get("target_segments", 12))))
            print(json.dumps({"stage": "qdrant_upload_complete", "column": column, "points": count,
                              "graph_build": "asynchronous", "elapsed_s": round(time.monotonic() - started)}), flush=True)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"version": VERSION, "collections": collections, "counts": counts,
            "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
            "on_disk_vectors": True, "on_disk_hnsw": True, "hnsw_m": 32,
            "ef_construct": 200, "full_scan_threshold": 0}, indent=2) + "\n")
    finally:
        client.close()


def finalize_qdrant(config: dict[str, Any], receipt: Path, *, timeout: float = 7200) -> None:
    """Wait for all uploaded vectors to be graph-indexed before declaring ready."""
    from qdrant_client import QdrantClient
    payload = json.loads(receipt.read_text())
    if payload["collections"] != config["acorn"]["collections"]:
        raise ValueError("refusing to finalize collections outside the import receipt")
    client = QdrantClient(url=config["acorn"]["url"], timeout=120)
    started = time.monotonic()
    try:
        for name in payload["collections"].values():
            info = client.get_collection(name)
            if info.config.hnsw_config.m != 32:
                raise ValueError(f"collection {name} is not configured with M=32")
            if info.config.hnsw_config.full_scan_threshold != 0:
                raise ValueError(f"collection {name} is not configured with full_scan_threshold=0")
            if info.config.hnsw_config.max_indexing_threads != int(config["acorn"].get("max_indexing_threads", 2)):
                raise ValueError(f"collection {name} has the wrong HNSW indexing thread count")
            if info.config.optimizer_config.indexing_threshold != int(config["acorn"].get("indexing_threshold_kb", 1)):
                raise ValueError(f"collection {name} has the wrong indexing threshold")
            if info.config.optimizer_config.default_segment_number != int(config["acorn"].get("target_segments", 12)):
                raise ValueError(f"collection {name} has the wrong segment target")
            if info.config.optimizer_config.max_optimization_threads != int(config["acorn"].get("max_optimization_threads", 3)):
                raise ValueError(f"collection {name} has the wrong optimizer concurrency")
        while True:
            state = {}
            for column, name in payload["collections"].items():
                info = client.get_collection(name)
                state[column] = {"status": str(info.status), "indexed": info.indexed_vectors_count,
                                 "expected": payload["counts"][column], "optimizer": str(info.optimizer_status)}
                if str(info.status).lower() == "red":
                    raise RuntimeError(f"Qdrant optimizer failed: {state[column]}")
            print(json.dumps({"stage": "qdrant_graph_wait", "elapsed_s": round(time.monotonic() - started),
                              "collections": state}), flush=True)
            if all(s["status"].lower() == "green" and s["indexed"] >= s["expected"] for s in state.values()):
                for column, name in payload["collections"].items():
                    if client.count(name, exact=True).count != payload["counts"][column]:
                        raise ValueError("Qdrant count changed during indexing")
                payload.update(graph_build_complete=True, graph_completed_at=datetime.now(timezone.utc).isoformat(),
                               indexing_threshold=int(config["acorn"].get("indexing_threshold_kb", 1)),
                               target_segments=int(config["acorn"].get("target_segments", 12)), graph_state=state)
                payload["max_indexing_threads"] = int(config["acorn"].get("max_indexing_threads", 2))
                payload["max_optimization_threads"] = int(config["acorn"].get("max_optimization_threads", 3))
                temporary = receipt.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n")
                temporary.replace(receipt)
                return
            if time.monotonic() - started > timeout:
                raise TimeoutError("Qdrant graph build not finished; receipt is not marked complete")
            time.sleep(30)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "faiss", "qdrant", "finalize-qdrant", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=Path("data/snapshot"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", default='1', help="FAISS build threads: positive integer or auto (up to 32)")
    parser.add_argument("--qdrant-indexing-threads", type=int,
                        help="Per-index Qdrant build threads; pass the same override to finalize-qdrant")
    parser.add_argument("--qdrant-optimization-threads", type=int,
                        help="Qdrant optimizer concurrency; extra concurrency also uses more memory")
    parser.add_argument("--timeout", type=float, default=21600,
                        help="seconds to wait for asynchronous Qdrant graph optimization")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.snapshot), indent=2))
    elif args.command == "export":
        export_snapshot(json.loads(args.config.read_text()), args.snapshot)
    elif args.output is None:
        parser.error("--output is required for index construction/import receipt")
    elif args.command == "faiss":
        build_faiss(args.snapshot, args.output, threads=build_threads(args.threads))
    elif args.command == "qdrant":
        import_qdrant(args.snapshot, _qdrant_build_config(args), args.output)
    else:
        finalize_qdrant(_qdrant_build_config(args), args.output, timeout=args.timeout)


def _qdrant_build_config(args):
    config = json.loads(args.config.read_text())
    for key, value in [('max_indexing_threads', args.qdrant_indexing_threads),
                       ('max_optimization_threads', args.qdrant_optimization_threads)]:
        if value is not None:
            if value < 1:
                raise ValueError(f'{key} must be positive')
            config.setdefault('acorn', {})[key] = value
    return config


if __name__ == "__main__":
    main()
