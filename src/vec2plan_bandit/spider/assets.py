"""Import released Spider SQLite data into isolated PostgreSQL databases."""
import hashlib
import json
from pathlib import Path
import sqlite3
import re

import numpy as np
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
import sqlite_vec

from vec2plan_bandit.assets.bird import pg_parameters
from vec2plan_bandit.bird.shared_experiment import write


def source_connection(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def schema(path):
    with source_connection(path) as conn:
        names = [r[1] for r in conn.execute('PRAGMA table_list')
                 if r[0] == 'main' and r[2] in ('table', 'virtual') and not r[1].startswith('sqlite_')]
        virtuals=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE sql LIKE '%USING vec0%'")]
        names=[n for n in names if not any(re.fullmatch(re.escape(v)+r'_(?:info|chunks|rowids|vector_chunks\d+|metadatachunks\d+|metadatatext\d+|auxiliary)',n) for v in virtuals)]
        result = {}
        for table in names:
            ddl=conn.execute("SELECT sql FROM sqlite_master WHERE name=?",(table,)).fetchone()[0]
            if re.search(r'USING\s+vec0',ddl,re.I):
                body=re.search(r'USING\s+vec0\s*\((.*)\)\s*$',ddl,re.I|re.S)[1]
                columns=[]
                for fragment in body.split(','):
                    m=re.fullmatch(r'\s*\+?(?:"([^"]+)"|([\w]+))\s+(float\[\d+\]|integer|text|float)(?:\s+PRIMARY\s+KEY)?\s*',fragment,re.I)
                    if not m:raise ValueError('Unsupported vec0 schema: '+fragment)
                    kind=m[3].lower()
                    columns.append((m[1] or m[2], 'vector(384)' if kind=='float[384]' else {'integer':'bigint','float':'double precision','text':'text'}[kind]))
                result[table]=columns
                continue
            columns = []
            for _, name, kind, *_ in conn.execute('PRAGMA table_info(' + '"'+table.replace('"','""')+'")'):
                lower = kind.lower()
                pg = ('vector(384)' if lower == 'blob' else 'bigint' if 'int' in lower else
                      'double precision' if any(k in lower for k in ('real','float','double','numeric','decimal')) else 'text')
                # sqlite-vec declares vectors as BLOB; other blobs are not embeddings.
                if pg == 'vector(384)' and not name.endswith('_embedding'):
                    pg = 'bytea'
                columns.append((name, pg))
            result[table] = columns
        return result


def import_databases(dataset, config):
    root = Path(dataset['asset_source'])
    output = Path(dataset['output_dir'])
    rows = json.loads(Path(dataset['source']).read_text())
    assets = json.loads(Path(dataset['catalog']).read_text())['assets']
    base = pg_parameters(Path(config['database']['config_path']))
    all_columns = {}
    for db in sorted({r['db_id'] for r in rows}):
        path = root/'vector_databases'/db/(db+'.sqlite')
        definition = schema(path)
        all_columns[db] = {t:[c for c,_ in cols]+([] if 'row_id' in [c for c,_ in cols] else ['row_id']) for t,cols in definition.items()}
        name = config['database']['prefix'] + db
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with psycopg.connect(**base, autocommit=True) as admin:
            if not admin.execute('SELECT 1 FROM pg_database WHERE datname=%s',(name,)).fetchone():
                admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
        with psycopg.connect(**{**base,'dbname':name}) as conn:
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            conn.execute('CREATE TABLE IF NOT EXISTS _spider_import (source_sha256 text PRIMARY KEY, status text)')
            marker = conn.execute('SELECT source_sha256,status FROM _spider_import').fetchall()
            if marker:
                if marker != [(digest,'complete')]:
                    raise ValueError('Spider import source changed: '+db)
                existing_tables={r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")}
                if set(definition)<=existing_tables:
                    ensure_row_id(conn,definition)
                    print('Import already complete',db,flush=True)
                    continue
                print('Completing missing source tables',db,sorted(set(definition)-existing_tables),flush=True)
            if not marker and conn.execute("SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename <> '_spider_import'").fetchone():
                raise ValueError('Refusing overwrite of existing database: '+name)
            register_vector(conn)
            counts = {}
            with source_connection(path) as source:
                for table, columns in definition.items():
                    if marker and table in existing_tables:
                        counts[table]=conn.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]
                        continue
                    fields = [('rowid','bigint'),*columns]
                    declarations = [sql.SQL('{} {}').format(sql.Identifier(c),sql.SQL(k)) for c,k in fields]
                    conn.execute(sql.SQL('CREATE TABLE {} ({})').format(sql.Identifier(table),sql.SQL(',').join(declarations)))
                    stmt = sql.SQL('SELECT {} FROM {} ORDER BY rowid').format(sql.SQL(',').join(sql.Identifier(c) for c,_ in fields),sql.Identifier(table)).as_string(conn)
                    count = 0
                    with conn.cursor().copy(sql.SQL('COPY {} ({}) FROM STDIN BINARY').format(sql.Identifier(table),sql.SQL(',').join(sql.Identifier(c) for c,_ in fields))) as copy:
                        copy.set_types([{'bigint':'int8','double precision':'float8','vector(384)':'vector','text':'text','bytea':'bytea'}[k] for _,k in fields])
                        for row in source.execute(stmt):
                            values = []
                            for value,(_,kind) in zip(row,fields):
                                if value is None: values.append(None)
                                elif kind=='vector(384)':
                                    vec=np.frombuffer(value,dtype='<f4').copy()
                                    if vec.shape!=(384,) or not np.all(np.isfinite(vec)):raise ValueError('Invalid vector '+db+'.'+table)
                                    values.append(vec)
                                elif kind=='bigint': values.append(int(value) if value!='' else None)
                                elif kind=='double precision': values.append(float(value) if value!='' else None)
                                elif kind=='text': values.append(str(value))
                                else: values.append(value)
                            copy.write_row(values);count+=1
                    counts[table]=count
                    conn.execute(sql.SQL('CREATE UNIQUE INDEX ON {} (rowid)').format(sql.Identifier(table)))
            for a in [a for a in assets if a['db_id']==db]:
                idx='bird_cos_'+hashlib.sha256(a['asset_id'].encode()).hexdigest()[:20]
                conn.execute(sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw ({} vector_cosine_ops) WITH (m=32, ef_construction=200)').format(sql.Identifier(idx),sql.Identifier(a['table']),sql.Identifier(a['column'])))
            ensure_row_id(conn,definition)
            conn.execute('ANALYZE')
            conn.execute("INSERT INTO _spider_import VALUES (%s,'complete') ON CONFLICT (source_sha256) DO NOTHING",(digest,))
            write(output/'assets/imports'/(db+'.json'),{'database':name,'source':str(path),'source_sha256':digest,'row_counts':counts})
            print('Imported',db,sum(counts.values()),'rows',flush=True)
    write(Path(dataset['schema']),all_columns)
    # Retain the released graph and point ordering; adapt only identity metadata.
    for a in assets:
        folder=Path(config['assets']['faiss_dir'])/a['asset_id'];folder.mkdir(parents=True,exist_ok=True)
        released=root/'indexes/faiss'/a['asset_id']
        manifest=json.loads((released/'manifest.json').read_text())
        for filename in ('index.faiss','point_ids.npy'):
            target=folder/filename
            if not target.exists():target.symlink_to(released/filename)
        mapping=folder/'rows.sqlite'
        if not mapping.exists():
            with sqlite3.connect(Path(manifest['row_mapping']).resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as src,sqlite3.connect(mapping) as dst:
                dst.execute('CREATE TABLE rows(point_id INTEGER PRIMARY KEY,key_json TEXT NOT NULL)')
                dst.executemany('INSERT INTO rows VALUES (?,?)',((i,json.dumps({'rowid':rowid})) for i,rowid in src.execute('SELECT point_id,source_rowid FROM rows')))
        write(folder/'manifest.json',manifest)
    write(output/'assets/completed.json',{'databases':len(all_columns),'assets':len(assets),'source':'released SQLite vectors; no corpus re-encoding'})


def ensure_row_id(conn, definition):
    # Extended uploaded SQL names the physical SQLite rowid as row_id.
    for table, columns in definition.items():
        if 'row_id' not in [c for c,_ in columns]:
            conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS row_id bigint GENERATED ALWAYS AS (rowid) STORED').format(sql.Identifier(table)))
