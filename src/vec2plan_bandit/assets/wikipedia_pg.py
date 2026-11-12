"""Import the official Wikipedia-Vec SQLite release into the local pgvector instance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import requests

import numpy as np
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
import sqlite_vec
import yaml

from vec2plan_bandit.services.database_layout import connection_parameters


def download(path: Path, relative: str, expected_size: int):
    if path.is_file() and path.stat().st_size == expected_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    for attempt in range(10):
        endpoint = 'https://huggingface.co' if attempt % 2 == 0 else 'https://hf-mirror.com'
        url = f'{endpoint}/datasets/VectorSQL/VectorSQLBench/resolve/main/{relative}'
        print(f'download attempt {attempt + 1}: {relative}', flush=True)
        try:
            with requests.get(url, headers={'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            if temporary.stat().st_size != expected_size:
                raise ValueError('download size does not match official file listing')
            temporary.replace(path)
            return
        except (requests.RequestException, ValueError) as error:
            print(f'download retry: {error}', flush=True)
            time.sleep(5)
    raise RuntimeError(f'download failed: {relative}')


def schema_from_queries(path: Path):
    records = json.loads(path.read_text())
    schemas = {record['schema'] for record in records}
    if len(schemas) != 1:
        raise ValueError('expected one canonical Wikipedia PostgreSQL schema')
    result = {}
    for table, body in re.findall(r'CREATE TABLE (\w+)\s*\((.*?)\);', schemas.pop(), re.S):
        columns = []
        for fragment in body.split(','):
            name, kind = fragment.strip().split()
            if kind not in {'bigint', 'text', 'vector'}:
                raise ValueError(f'unsupported type: {kind}')
            columns.append((name, kind))
        result[table] = columns
    if len(result) != 5 or sum(t == 'vector' for cols in result.values() for _, t in cols) != 6:
        raise ValueError('unexpected logical Wikipedia schema')
    return result


def convert(value, kind):
    if value is None:
        return None
    if kind == 'vector':
        vector = np.frombuffer(value, dtype='<f4')
        if vector.size != 512 or not np.all(np.isfinite(vector)):
            raise ValueError('invalid 512-dimensional CLIP vector')
        return vector
    if kind == 'bigint':
        return int(value) if value != '' else None
    return str(value)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    source_relative = 'sqlite/results/wikipedia_multimodal/vector_databases/wikipedia_multimodal/wikipedia_multimodal_final.sqlite'
    schema_relative = 'postgresql/results/wikipedia_multimodal/candidate_sql.json'
    parser.add_argument('--source', type=Path, default=root / 'data/VectorSQLBench' / source_relative)
    parser.add_argument('--schema', type=Path, default=root / 'data/VectorSQLBench' / schema_relative)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--config', type=Path, default=root / 'configs/database.yaml')
    parser.add_argument('--database', default='wikipedia_multimodal')
    parser.add_argument('--report', type=Path, default=root / 'outputs/wikipedia-pg-import/completed.json')
    args = parser.parse_args()
    if args.download:
        download(args.schema, schema_relative, 290053)
        download(args.source, source_relative, 55062528)
    schema = schema_from_queries(args.schema)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    config = yaml.safe_load(args.config.read_text())
    base = connection_parameters(config['database_connections']['postgresql'])
    base['dbname'] = base.pop('database', base.pop('dbname', 'postgres'))
    name = args.database
    with psycopg.connect(**base, autocommit=True) as admin:
        exists = admin.execute('SELECT 1 FROM pg_database WHERE datname=%s', (name,)).fetchone()
        if not exists:
            admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    target = {**base, 'dbname': name}
    source = sqlite3.connect(f'file:{args.source.resolve()}?mode=ro&immutable=1', uri=True)
    source.enable_load_extension(True)
    sqlite_vec.load(source)
    counts = {}
    vector_counts = {}
    with source, psycopg.connect(**target) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.execute('CREATE TABLE IF NOT EXISTS _wikipedia_vec_import (source_sha256 text PRIMARY KEY, status text, report jsonb)')
        previous = conn.execute('SELECT source_sha256,status,report FROM _wikipedia_vec_import').fetchall()
        if previous:
            if len(previous) == 1 and previous[0][0] == source_hash and previous[0][1] == 'complete':
                report = previous[0][2]
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + '\n')
                print('Wikipedia-Vec already complete', flush=True)
                return
            raise RuntimeError('existing import marker does not match; refusing overwrite')
        existing = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename<>'_wikipedia_vec_import'").fetchall()
        if existing:
            raise RuntimeError(f'refusing to overwrite existing tables: {existing}')
        register_vector(conn)
        conn.execute("INSERT INTO _wikipedia_vec_import VALUES (%s,'importing',NULL)", (source_hash,))
        for table, columns in schema.items():
            declarations = [sql.SQL('{} {}').format(sql.Identifier(col), sql.SQL('vector(512)' if kind == 'vector' else kind)) for col, kind in columns]
            conn.execute(sql.SQL('CREATE TABLE {} ({})').format(sql.Identifier(table), sql.SQL(',').join(declarations)))
            quoted = lambda value: '"' + value.replace('"', '""') + '"'
            source_sql = 'SELECT ' + ','.join(quoted(c) for c, _ in columns) + ' FROM ' + quoted(table)
            expected = source.execute('SELECT count(*) FROM ' + quoted(table)).fetchone()[0]
            rows = 0
            vectors = {col: 0 for col, kind in columns if kind == 'vector'}
            with conn.cursor().copy(sql.SQL('COPY {} ({}) FROM STDIN BINARY').format(sql.Identifier(table), sql.SQL(',').join(sql.Identifier(c) for c, _ in columns))) as copy:
                copy.set_types([{'bigint': 'int8', 'text': 'text', 'vector': 'vector'}[kind] for _, kind in columns])
                for row in source.execute(source_sql):
                    copy.write_row(tuple(convert(value, kind) for value, (_, kind) in zip(row, columns)))
                    rows += 1
                    for i, (col, kind) in enumerate(columns):
                        if kind == 'vector' and row[i] is not None:
                            vectors[col] += 1
            if rows != expected:
                raise ValueError(f'{table} source row count mismatch')
            counts[table] = rows
            vector_counts[table] = vectors
            print(f'[{table}] copied {rows:,} rows', flush=True)
        conn.execute("SET maintenance_work_mem='512MB'")
        indexes = []
        conn.execute('ANALYZE')
        for table, expected in counts.items():
            actual = conn.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]
            if actual != expected:
                raise ValueError(f'{table} PostgreSQL count mismatch')
        report = {'status': 'complete', 'database': name, 'source_sha256': source_hash,
                  'source': str(args.source.resolve()), 'tables': len(schema),
                  'columns': sum(len(cols) for cols in schema.values()), 'vector_columns': 6,
                  'dimension': 512, 'embedding_model': 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
                  'row_counts': counts, 'total_rows': sum(counts.values()),
                  'vector_counts': vector_counts, 'hnsw_indexes': indexes}
        conn.execute("UPDATE _wikipedia_vec_import SET status='complete',report=%s WHERE source_sha256=%s", (json.dumps(report), source_hash))
        conn.commit()
    source.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
