"""Build query-scoped FAISS assets and an ACORN/Qdrant plan for BIRD-Vec.

The public BIRD-Vec PostgreSQL release is split across one database per BIRD
database.  This module deliberately treats every (database, table, vector
column) tuple as a separate ANN asset.  Point IDs are local, stable snapshot
ordinals; ``rows.sqlite`` maps them back to logical primary keys (or CTIDs for
tables whose public metadata does not declare a key).
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import time
from typing import Any, Iterable

import numpy as np
import psycopg
from psycopg import sql
import yaml

from vec2plan_bandit.services.database_layout import connection_parameters


CATALOG_VERSION = "bird-ann-catalog-v1"
FAISS_VERSION = "bird-faiss-hnsw-v1"
ACORN_PLAN_VERSION = "bird-qdrant-acorn-plan-v1"
KEYWORDS = {
    "ON", "WHERE", "ORDER", "GROUP", "LIMIT", "JOIN", "LEFT", "RIGHT",
    "INNER", "OUTER", "FULL", "CROSS", "UNION", "HAVING", "OFFSET",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pg_parameters(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = connection_parameters(config["database_connections"]["postgresql"])
    values["dbname"] = values.pop("database", values.pop("dbname", "postgres"))
    return values


def primary_keys(metadata: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for record in metadata:
        for declaration in record.get("primary_keys", []):
            indexes = declaration if isinstance(declaration, list) else [declaration]
            table: str | None = None
            columns: list[str] = []
            for index in indexes:
                table_index, column = record["column_names_original"][int(index)]
                current = str(record["table_names_original"][int(table_index)])
                if table is not None and current.lower() != table.lower():
                    raise ValueError(f"cross-table primary key in {record['db_id']}: {declaration}")
                table = current
                columns.append(str(column))
            if table is not None:
                result[(str(record["db_id"]).lower(), table.lower())] = columns
    return result


def query_vector_references(
    queries: list[dict[str, Any]],
    catalog: dict[str, dict[str, set[str]]],
) -> tuple[dict[tuple[str, str, str], list[int]], list[dict[str, Any]]]:
    """Resolve vector references, conservatively retaining ambiguous tables."""
    assets: dict[tuple[str, str, str], set[int]] = {}
    unresolved: list[dict[str, Any]] = []
    table_pattern = re.compile(
        r'(?is)\b(?:FROM|JOIN)\s+(?:"([^"]+)"|([A-Za-z_][\w]*))'
        r'(?:\s+(?:AS\s+)?(?:"([^"]+)"|([A-Za-z_][\w]*)))?'
    )
    vector_pattern = re.compile(
        r'(?i)(?:(?:"([^"]+)"|([A-Za-z_][\w]*))\s*\.\s*)?'
        r'(?:"([^"]+_embedding)"|([A-Za-z_][\w]*_embedding))'
    )
    for query_index, query in enumerate(queries):
        database = str(query["db_id"])
        database_catalog = catalog.get(database.lower(), {})
        aliases: dict[str, str] = {}
        for match in table_pattern.finditer(str(query["sql"])):
            table = next(value for value in match.groups()[:2] if value)
            alias = next((value for value in match.groups()[2:] if value), None)
            aliases[table.lower()] = table
            if alias and alias.upper() not in KEYWORDS:
                aliases[alias.lower()] = table
        referenced_tables = {value.lower() for value in aliases.values()}
        for match in vector_pattern.finditer(str(query["sql"])):
            qualifier = next((value for value in match.groups()[:2] if value), None)
            column = next(value for value in match.groups()[2:] if value).lower()
            matches: list[str] = []
            if qualifier:
                table = aliases.get(qualifier.lower(), qualifier)
                matches = [
                    name for name, columns in database_catalog.items()
                    if name.lower() == table.lower() and column in columns
                ]
            if not matches:
                matches = [
                    name for name, columns in database_catalog.items()
                    if name.lower() in referenced_tables and column in columns
                ]
            if not matches:
                matches = [name for name, columns in database_catalog.items() if column in columns]
            if not matches:
                unresolved.append({
                    "query_index": query_index,
                    "db_id": database,
                    "qualifier": qualifier,
                    "column": column,
                })
            for table in matches:
                assets.setdefault((database, table, column), set()).add(query_index)
    return ({key: sorted(indexes) for key, indexes in assets.items()}, unresolved)


def asset_id(database: str, table: str, column: str) -> str:
    raw = f"{database}\0{table}\0{column}"
    readable = re.sub(r"[^a-z0-9]+", "-", f"{database}-{table}-{column}".lower()).strip("-")
    return f"{readable[:72]}-{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"hex": bytes(value).hex()}
    return value


def _catalog_database(
    base: dict[str, Any], database: str
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    params = dict(base)
    params["dbname"] = database
    with psycopg.connect(**params) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT row_counts FROM public._vectorsqlbench_import WHERE db_id=%s AND status='complete'",
            (database,),
        )
        marker = cursor.fetchone()
        if marker is None:
            raise RuntimeError(f"BIRD database is not complete: {database}")
        row_counts = marker[0]
        cursor.execute(
            "SELECT c.relname,a.attname,format_type(a.atttypid,a.atttypmod),t.typname "
            "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_type t ON t.oid=a.atttypid "
            "WHERE n.nspname='public' AND c.relkind='r' AND c.relname<>'_vectorsqlbench_import' "
            "AND a.attnum>0 AND NOT a.attisdropped ORDER BY c.relname,a.attnum"
        )
        tables: dict[str, dict[str, Any]] = {}
        vector_lookup: dict[str, set[str]] = {}
        for table, column, formatted_type, base_type in cursor.fetchall():
            entry = tables.setdefault(
                str(table),
                {"rows": int(row_counts[str(table)]), "vectors": [], "scalars": []},
            )
            if base_type == "vector":
                dimension_match = re.search(r"\((\d+)\)", str(formatted_type))
                if dimension_match is None:
                    raise ValueError(f"unbounded vector dimension: {database}.{table}.{column}")
                entry["vectors"].append({"name": str(column), "dimension": int(dimension_match.group(1))})
                vector_lookup.setdefault(str(table), set()).add(str(column).lower())
            else:
                entry["scalars"].append({"name": str(column), "type": str(formatted_type)})
        return tables, vector_lookup


def build_catalog(
    *, metadata_path: Path, queries_path: Path, database_config: Path, output: Path
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    base = pg_parameters(database_config)
    tables_by_database: dict[str, dict[str, dict[str, Any]]] = {}
    vectors_by_database: dict[str, dict[str, set[str]]] = {}
    for record in sorted(metadata, key=lambda value: str(value["db_id"])):
        database = str(record["db_id"])
        tables, vectors = _catalog_database(base, database)
        tables_by_database[database.lower()] = tables
        vectors_by_database[database.lower()] = vectors
    references, unresolved = query_vector_references(queries, vectors_by_database)
    if unresolved:
        raise ValueError(f"unresolved vector references: {unresolved[:5]}")
    declared_keys = primary_keys(metadata)
    assets: list[dict[str, Any]] = []
    for (database, table, lowered_column), query_indexes in sorted(references.items()):
        table_entry = tables_by_database[database.lower()][table]
        vector = next(value for value in table_entry["vectors"] if value["name"].lower() == lowered_column)
        points = int(table_entry["rows"])
        dimension = int(vector["dimension"])
        estimated_flat_bytes = points * (dimension * 4 + 32 * 2 * 4 + 32)
        assets.append({
            "asset_id": asset_id(database, table, str(vector["name"])),
            "db_id": database,
            "table": table,
            "column": str(vector["name"]),
            "dimension": dimension,
            "estimated_points": points,
            "primary_key": declared_keys.get((database.lower(), table.lower()), []),
            "row_identity": "metadata_primary_key" if declared_keys.get((database.lower(), table.lower())) else "postgres_ctid_snapshot",
            "scalar_columns": table_entry["scalars"],
            "query_indexes": query_indexes,
            "estimated_raw_vector_bytes": points * dimension * 4,
            "estimated_faiss_hnsw_flat_bytes": estimated_flat_bytes,
        })
    payload = {
        "version": CATALOG_VERSION,
        "scope": "VectorSQLBench PostgreSQL BIRD candidate_sql.json",
        "database_count": len({asset["db_id"] for asset in assets}),
        "query_count": len(queries),
        "asset_count": len(assets),
        "metadata_sha256": sha256(metadata_path),
        "queries_sha256": sha256(queries_path),
        "totals": {
            "estimated_points": sum(asset["estimated_points"] for asset in assets),
            "estimated_raw_vector_bytes": sum(asset["estimated_raw_vector_bytes"] for asset in assets),
            "estimated_faiss_hnsw_flat_bytes": sum(asset["estimated_faiss_hnsw_flat_bytes"] for asset in assets),
        },
        "assets": assets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def decode_vector(value: bytes | memoryview, dimension: int) -> np.ndarray:
    raw = bytes(value)
    if len(raw) != 4 + 4 * dimension or struct.unpack_from(">h", raw)[0] != dimension:
        raise ValueError("missing or malformed pgvector binary value")
    vector = np.frombuffer(raw, dtype=">f4", offset=4).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 0:
        raise ValueError("nonfinite or zero source vector")
    return vector / norm


def _selected_assets(
    catalog: dict[str, Any], asset_names: list[str], max_points: int | None
) -> list[dict[str, Any]]:
    known = {asset["asset_id"]: asset for asset in catalog["assets"]}
    if asset_names:
        missing = sorted(set(asset_names) - set(known))
        if missing:
            raise ValueError(f"unknown asset IDs: {missing}")
        selected = [known[name] for name in asset_names]
    else:
        selected = list(catalog["assets"])
    if max_points is not None:
        selected = [asset for asset in selected if int(asset["estimated_points"]) <= max_points]
    return selected


def _build_one_faiss(
    asset: dict[str, Any], *, base: dict[str, Any], output: Path, m: int,
    ef_construction: int, threads: int, minimum_free_gib: float,
) -> dict[str, Any]:
    import faiss

    destination = output / str(asset["asset_id"])
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    partial = output / f".{asset['asset_id']}.partial"
    if partial.exists():
        raise FileExistsError(f"partial asset requires inspection before retry: {partial}")
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    expected = int(asset["estimated_faiss_hnsw_flat_bytes"])
    reserve = int(minimum_free_gib * 2**30)
    if free - expected < reserve:
        raise OSError(
            f"insufficient disk for {asset['asset_id']}: free={free / 2**30:.1f} GiB, "
            f"estimated={expected / 2**30:.1f} GiB, reserve={minimum_free_gib:.1f} GiB"
        )
    partial.mkdir()
    params = dict(base)
    params["dbname"] = str(asset["db_id"])
    dimension = int(asset["dimension"])
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    keys = list(asset["primary_key"])
    identity_columns = [sql.Identifier(value) for value in keys]
    selected = [sql.SQL("vector_send({})").format(sql.Identifier(str(asset["column"])))]
    if keys:
        selected.extend(identity_columns)
        ordering = sql.SQL(",").join(
            sql.SQL("{} NULLS LAST").format(value) for value in identity_columns
        ) + sql.SQL(",ctid")
    else:
        selected.append(sql.SQL("ctid::text"))
        ordering = sql.SQL("ctid")
    statement = sql.SQL("SELECT {} FROM {} WHERE {} IS NOT NULL ORDER BY {}").format(
        sql.SQL(",").join(selected),
        sql.Identifier(str(asset["table"])),
        sql.Identifier(str(asset["column"])),
        ordering,
    )
    mapping = sqlite3.connect(partial / "rows.sqlite")
    mapping.execute("CREATE TABLE rows(point_id INTEGER PRIMARY KEY, key_json TEXT NOT NULL)")
    mapping.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    started = time.monotonic()
    count = 0
    batch_vectors: list[np.ndarray] = []
    batch_keys: list[tuple[int, str]] = []
    try:
        with psycopg.connect(**params) as connection:
            connection.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            connection.execute("SET LOCAL statement_timeout=0")
            with connection.cursor(name=f"bird_faiss_{hashlib.sha1(str(asset['asset_id']).encode()).hexdigest()[:12]}") as cursor:
                cursor.itersize = 2048
                cursor.execute(statement)
                for row in cursor:
                    batch_vectors.append(decode_vector(row[0], dimension))
                    values = row[1:]
                    key = ({name: _json_value(value) for name, value in zip(keys, values)}
                           if keys else {"ctid": str(values[0])})
                    batch_keys.append((count, json.dumps(key, ensure_ascii=False, separators=(",", ":"))))
                    count += 1
                    if len(batch_vectors) == 4096:
                        index.add(np.asarray(batch_vectors, dtype=np.float32))
                        mapping.executemany("INSERT INTO rows VALUES (?,?)", batch_keys)
                        mapping.commit()
                        batch_vectors.clear()
                        batch_keys.clear()
                        if count % 100_000 < 4096:
                            print(json.dumps({"stage": "bird_faiss", "asset_id": asset["asset_id"],
                                "points": count, "elapsed_s": round(time.monotonic() - started)}), flush=True)
                if batch_vectors:
                    index.add(np.asarray(batch_vectors, dtype=np.float32))
                    mapping.executemany("INSERT INTO rows VALUES (?,?)", batch_keys)
                    mapping.commit()
        if count == 0 or index.ntotal != count:
            raise ValueError(f"empty or inconsistent FAISS asset: {asset['asset_id']}")
        faiss.write_index(index, str(partial / "index.faiss"))
        np.save(partial / "point_ids.npy", np.arange(count, dtype=np.int64))
        mapping.execute("INSERT INTO metadata VALUES ('row_identity',?)", (str(asset["row_identity"]),))
        mapping.commit()
    except Exception:
        mapping.close()
        raise
    mapping.close()
    manifest = {
        "version": FAISS_VERSION,
        "asset": asset,
        "points": count,
        "distance": "normalized_inner_product",
        "index_type": "IndexHNSWFlat",
        "m": m,
        "ef_construction": ef_construction,
        "threads": threads,
        "faiss_version": faiss.__version__,
        "elapsed_s": round(time.monotonic() - started),
        "files": {},
    }
    for name in ("index.faiss", "point_ids.npy", "rows.sqlite"):
        manifest["files"][name] = sha256(partial / name)
    (partial / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(destination)
    print(json.dumps({"stage": "bird_faiss_complete", "asset_id": asset["asset_id"],
                      "points": count, "elapsed_s": manifest["elapsed_s"]}), flush=True)
    return manifest


def build_faiss(
    *, catalog_path: Path, database_config: Path, output: Path,
    asset_names: list[str], max_points: int | None, m: int,
    ef_construction: int, threads: int, minimum_free_gib: float,
) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("version") != CATALOG_VERSION:
        raise ValueError("unsupported BIRD ANN catalog")
    if threads < 1 or m < 2 or ef_construction < 1:
        raise ValueError("invalid FAISS construction parameters")
    selected = _selected_assets(catalog, asset_names, max_points)
    base = pg_parameters(database_config)
    completed = []
    for asset in selected:
        manifest = _build_one_faiss(
            asset, base=base, output=output, m=m, ef_construction=ef_construction,
            threads=threads, minimum_free_gib=minimum_free_gib,
        )
        completed.append({"asset_id": asset["asset_id"], "points": manifest["points"]})
    receipt = {
        "version": FAISS_VERSION,
        "catalog_sha256": sha256(catalog_path),
        "selected_assets": len(selected),
        "completed": completed,
    }
    (output / "build-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_acorn_plan(*, catalog_path: Path, output: Path) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("version") != CATALOG_VERSION:
        raise ValueError("unsupported BIRD ANN catalog")
    collections = []
    for asset in catalog["assets"]:
        collection = f"bird_{asset['asset_id'].replace('-', '_')}_acorn_v1"
        collections.append({
            "asset_id": asset["asset_id"],
            "collection": collection[:200],
            "dimension": asset["dimension"],
            "distance": "Cosine",
            "point_id": "zero-based stable snapshot ordinal",
            "row_mapping": f"data/indexes/bird-faiss/{asset['asset_id']}/rows.sqlite",
            "payload_columns": asset["scalar_columns"],
            "recommended_payload_indexes": [],
            "hnsw": {"m": 32, "ef_construct": 200, "on_disk": True},
            "acorn_search": {"enable": True, "max_selectivity": 1.0},
            "estimated_points": asset["estimated_points"],
            "query_indexes": asset["query_indexes"],
        })
    plan = {
        "version": ACORN_PLAN_VERSION,
        "minimum_qdrant_version": "1.16.0",
        "catalog_sha256": sha256(catalog_path),
        "collection_count": len(collections),
        "collections": collections,
        "status": "prepared_not_uploaded",
        "note": "Payload indexes must be chosen from translated BIRD predicates before upload; indexing every text field is intentionally avoided.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog")
    catalog.add_argument("--metadata", type=Path, default=root / "history/data/VectorSQLBench/sqlite/results/bird/find_semantic_tables.json")
    catalog.add_argument("--queries", type=Path, default=root / "history/data/VectorSQLBench/postgresql/results/bird/candidate_sql.json")
    catalog.add_argument("--database-config", type=Path, default=root / "configs/database.yaml")
    catalog.add_argument("--output", type=Path, default=root / "outputs/bird-ann/catalog.json")

    faiss_parser = commands.add_parser("faiss")
    faiss_parser.add_argument("--catalog", type=Path, default=root / "outputs/bird-ann/catalog.json")
    faiss_parser.add_argument("--database-config", type=Path, default=root / "configs/database.yaml")
    faiss_parser.add_argument("--output", type=Path, default=root / "data/indexes/bird-faiss")
    faiss_parser.add_argument("--asset", action="append", default=[])
    faiss_parser.add_argument("--max-points", type=int)
    faiss_parser.add_argument("--m", type=int, default=32)
    faiss_parser.add_argument("--ef-construction", type=int, default=200)
    faiss_parser.add_argument("--threads", default=str(max(1, min(6, os.cpu_count() or 1))),
                              help="FAISS build threads: positive integer or auto (up to 32)")
    faiss_parser.add_argument("--minimum-free-gib", type=float, default=20.0)

    acorn = commands.add_parser("acorn-plan")
    acorn.add_argument("--catalog", type=Path, default=root / "outputs/bird-ann/catalog.json")
    acorn.add_argument("--output", type=Path, default=root / "outputs/bird-ann/acorn-plan.json")

    args = parser.parse_args()
    if args.command == "catalog":
        payload = build_catalog(
            metadata_path=args.metadata, queries_path=args.queries,
            database_config=args.database_config, output=args.output,
        )
        print(json.dumps({"stage": "bird_catalog_complete", **payload["totals"],
                          "assets": payload["asset_count"]}), flush=True)
    elif args.command == "faiss":
        from vec2plan_bandit.assets.build_resources import build_threads
        build_faiss(
            catalog_path=args.catalog, database_config=args.database_config,
            output=args.output, asset_names=args.asset, max_points=args.max_points,
            m=args.m, ef_construction=args.ef_construction, threads=build_threads(args.threads),
            minimum_free_gib=args.minimum_free_gib,
        )
    else:
        plan = prepare_acorn_plan(catalog_path=args.catalog, output=args.output)
        print(json.dumps({"stage": "bird_acorn_plan_complete",
                          "collections": plan["collection_count"]}), flush=True)


if __name__ == "__main__":
    main()
