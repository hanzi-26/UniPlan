"""ID-only BIRD execution over a frozen, pruned relational sidecar.

The sidecar contains CTIDs and only scalar columns needed by filtering, joins,
grouping and ordering. ANN point IDs are mapped to snapshot CTIDs. No PostgreSQL
query is issued by this module during online retrieval.
"""
from __future__ import annotations

import hashlib
import json
import locale
from pathlib import Path
import sqlite3
import time

import numpy as np
from sqlglot import exp, parse_one
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from .adapter import quote, parse, relations

VERSION = 'bird-covering-identities-v1'
CTID = '__bird_ctid'


def map_table(asset_id):
    return '__ann_' + hashlib.sha256(asset_id.encode()).hexdigest()[:24]


def present_column(column):
    return '__vec_' + hashlib.sha256(column.encode()).hexdigest()[:16]


def _identity(expression):
    tree = parse_one(expression, read='postgres')
    def convert(node):
        for key, value in list(node.args.items()):
            if isinstance(value, exp.Expression): node.set(key, convert(value))
            elif isinstance(value, list): node.set(key, [convert(v) if isinstance(v, exp.Expression) else v for v in value])
        if isinstance(node, exp.Array):
            return exp.Anonymous(this='bird_ids', expressions=node.expressions)
        if isinstance(node, exp.DPipe):
            return exp.Anonymous(this='bird_concat', expressions=[node.this, node.expression])
        return node
    return convert(tree)


def _base_tree(query):
    tree = query.tree(True)
    # Retain only identity and SELECT aliases actually required by root ORDER BY.
    order = tree.args.get('order')
    aliases = {c.name for c in order.find_all(exp.Column) if not c.table} if order else set()
    tree.set('expressions', [e for e in tree.expressions
        if e.alias_or_name == '__bird_ids' or e.alias_or_name in aliases])
    for node in list(tree.find_all(exp.Var)):
        if not node.this.startswith('%(v_'):
            node.replace(_identity(node.this))
    distances = [v.parent for v in tree.find_all(exp.Var) if v.this.startswith('%(v_')]
    for block, distance in zip(query.blocks, distances):
        distance.replace(exp.Anonymous(this='bird_distance', expressions=[
            exp.Literal.number(block.number), exp.column(CTID, table=block.alias, quoted=True)]))
    for column in tree.find_all(exp.Column):
        if column.name == 'ctid': column.set('this', exp.to_identifier(CTID, quoted=True))
    return tree


def reduce_grouped_joins(tree):
    """Semijoin aggregate CTEs to ANN candidate keys before aggregating.

    In B127, computing RatingAnalysis for every user scans 15M ratings despite
    VectorSearch containing just five lists. The same rewrite helps B207.
    Apply only INNER equijoins on projected GROUP BY columns and peers already
    restricted by ANN candidates; preserve all rows within each needed group.
    """
    ctes={cte.alias_or_name:cte.this for cte in tree.find_all(exp.CTE)}
    for scope in list(tree.find_all(exp.Select)):
        sources={table.alias_or_name:table for table in relations(scope)}
        for join in scope.args.get('joins',[]):
            if join.args.get('side') or join.args.get('kind') not in (None,'INNER'):continue
            condition=join.args.get('on')
            if not isinstance(condition,exp.EQ):continue
            left,right=condition.this,condition.expression
            if not isinstance(left,exp.Column) or not isinstance(right,exp.Column):continue
            for grouped,peer in [(left,right),(right,left)]:
                table=sources.get(grouped.table);peer_table=sources.get(peer.table)
                if table is None or peer_table is None:continue
                aggregate=ctes.get(table.name)
                if aggregate is None or not (aggregate.args.get('group') or aggregate.args.get('distinct')):continue
                projected=next((e.this if isinstance(e,exp.Alias) else e for e in aggregate.expressions if e.alias_or_name==grouped.name),None)
                if not isinstance(projected,exp.Column):continue
                if aggregate.args.get('group') and projected not in aggregate.args['group'].expressions:continue
                peer_cte=ctes.get(peer_table.name)
                # Avoid introducing recursive dependencies.
                if peer_cte is not None and any(t.name==table.name for t in peer_cte.find_all(exp.Table)):continue
                where=scope.args.get('where')
                predicates=[]
                if where:
                    for term in where.this.flatten() if isinstance(where.this,exp.And) else [where.this]:
                        columns=[c for c in term.find_all(exp.Column) if c.find_ancestor(exp.Subquery) is None]
                        if columns and all(c.table==peer.table for c in columns):predicates.append(term.copy())
                restricted=(peer_cte is not None and any(t.name.startswith('__candidates_') for t in peer_cte.find_all(exp.Table))) or any(
                    any(t.name.startswith(('__candidates_','__probe_')) for t in pred.find_all(exp.Table)) for pred in predicates)
                if not restricted:continue
                needed=exp.select(peer.copy()).from_(peer_table.copy())
                for predicate in predicates:needed.where(predicate,append=True,copy=False)
                aggregate.where(exp.In(this=projected.copy(),query=exp.Subquery(this=needed)),append=True,copy=False)
    return tree


def local_tree(query, *, candidates=False, eligibility=None):
    tree = _base_tree(query)
    functions = [f for f in tree.find_all(exp.Anonymous) if f.name.lower() == 'bird_distance']
    by_number = {int(f.expressions[0].this): f for f in functions}
    if eligibility is not None:
        block = query.blocks[eligibility]
        source = by_number[eligibility].find_ancestor(exp.Select).copy()
        if not source.args.get('where') and not source.args.get('joins'):
            return None
        source.set('expressions', [exp.column(CTID, table=block.alias, quoted=True)])
        source.set('distinct', exp.Distinct())
        for key in ('order', 'limit', 'offset'): source.set(key, None)
        if tree.args.get('with_'): source.set('with_', tree.args['with_'].copy())
        tree = source
    elif candidates:
        for block in query.blocks:
            function = by_number[block.number]
            select = function.find_ancestor(exp.Select)
            predicate = exp.In(this=exp.column(CTID, table=block.alias, quoted=True),
                query=exp.Subquery(this=exp.select(CTID).from_(f'__candidates_{block.number}')))
            select.where(predicate, append=True, copy=False)
    schema = {table: {column: 'TEXT' for column in columns if not column.endswith('_embedding')}
              for table, columns in query.columns.items()}
    for columns in schema.values(): columns[CTID] = 'TEXT'
    for block in query.blocks: schema[f'__candidates_{block.number}'] = {CTID: 'TEXT'}
    tree = qualify(tree, dialect='postgres', schema=schema,
                   validate_qualify_columns=False, allow_partial_qualification=True)
    tree = eliminate_ctes(pushdown_projections(tree))
    if eligibility is not None:
        # Eligibility asks only whether a point has a matching relation row.
        # Once unused SUM/AVG projections are pruned, DISTINCT projected group
        # keys is sufficient; do not aggregate every donation just to test a JOIN.
        for cte in tree.find_all(exp.CTE):
            select=cte.this
            if not isinstance(select,exp.Select) or not select.args.get('group') or select.args.get('having'):continue
            if any(a.find_ancestor(exp.Select) is select for a in select.find_all(exp.AggFunc)):continue
            select.set('group',None);select.set('distinct',exp.Distinct())
    tree = reduce_grouped_joins(tree)
    # PG compares identity arrays element-by-element using the database locale.
    # JSON byte ordering would change ties such as (1000,1) versus (100,1).
    for ordered in tree.find_all(exp.Ordered):
        identity = any(c.name == '__bird_ids' for c in ordered.this.find_all(exp.Column)) or any(
            f.name.lower() in {'bird_ids','bird_concat'} for f in ordered.this.find_all(exp.Anonymous))
        ordered.set('this', exp.Collate(this=ordered.this, expression=exp.Var(this='BIRD_IDS' if identity else 'BIRD_TEXT')))
    return tree


def requirements(queries):
    """Resolve physical scalar-column dependencies, after projection pruning."""
    result = {}
    for query in queries:
        if not query.blocks: continue
        trees = [local_tree(query, candidates=True)]
        trees += [t for b in query.blocks if (t := local_tree(query, eligibility=b.number)) is not None]
        for tree in trees:
            for scope in traverse_scope(tree):
                for column in scope.columns:
                    source = scope.sources.get(column.table)
                    if source is None and column.table:
                        # local_tree is SQLite SQL: quoted aliases are still
                        # case-insensitive. sqlglot can normalize a table alias
                        # while retaining the original case on a column ref.
                        matches = [value for alias, value in scope.sources.items()
                                   if alias.casefold() == column.table.casefold()]
                        if len(matches) == 1:
                            source = matches[0]
                    if isinstance(source, exp.Table) and source.name in query.columns:
                        result.setdefault(query.db_id, {}).setdefault(source.name, set()).add(column.name)
                for source in scope.sources.values():
                    if isinstance(source, exp.Table) and source.name in query.columns:
                        result.setdefault(query.db_id, {}).setdefault(source.name, set()).add(CTID)
        for block in query.blocks:
            result.setdefault(query.db_id, {}).setdefault(block.table, set()).update(block.asset['primary_key'])
    return {db: {t: sorted(cols - {CTID}) for t, cols in tables.items()} for db, tables in result.items()}


def connect_snapshot(path, collation='en_US.utf8'):
    locale.setlocale(locale.LC_COLLATE, collation)
    def text_compare(a,b):
        result=locale.strcoll(a,b)
        return result or ((a>b)-(a<b))  # deterministic PG collation byte tie-break
    def ids_compare(a,b):
        left,right=json.loads(a),json.loads(b)
        for x,y in zip(left,right):
            if x is None or y is None:
                result=(x is None)-(y is None)
            else:result=text_compare(x,y)
            if result:return result
        return (len(left)>len(right))-(len(left)<len(right))
    connection = sqlite3.connect(':memory:', uri=True, check_same_thread=False)
    connection.create_collation('BIRD_TEXT',text_compare)
    connection.create_collation('BIRD_IDS',ids_compare)
    connection.execute('ATTACH DATABASE ? AS snapshot', (path.resolve().as_uri() + '?mode=ro',))
    connection.execute('PRAGMA case_sensitive_like=ON')
    connection.execute('PRAGMA temp_store=MEMORY')
    connection.execute('PRAGMA cache_size=-65536')
    connection.create_function('bird_ids', -1,
        lambda *args: json.dumps(list(args), ensure_ascii=False, separators=(',', ':')))
    connection.create_function('bird_concat', -1,
        lambda *args: json.dumps([v for a in args if a is not None for v in json.loads(a)],
                                ensure_ascii=False, separators=(',', ':')))
    return connection


class CoveringRuntime:
    def __init__(self, directory, *, source_sha256, catalog_sha256, schema_sha256=None):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        if (manifest['version'] != VERSION or manifest['status'] != 'complete'
            or manifest['source_sha256'] != source_sha256 or manifest['catalog_sha256'] != catalog_sha256):
            raise ValueError('BIRD covering snapshot is incomplete or belongs to different inputs')
        if schema_sha256 is not None and manifest.get('schema_sha256') != schema_sha256:
            raise ValueError('BIRD covering snapshot belongs to a different database schema')
        self.manifest = manifest
        self.connection = None
        self.db = None
        self.plans = {}

    def bind(self, query, vectors, assets, *, simple=False):
        if self.db != query.db_id:
            if self.connection: self.connection.close()
            self.connection = connect_snapshot(self.directory / f'{query.db_id}.sqlite',
                self.manifest.get('collations',{}).get(query.db_id,'en_US.utf8'))
            self.db = query.db_id
        self.query, self.vectors, self.assets = query, vectors, assets
        self._ctids, self._point_ids = {}, {}
        self.scores = [{} for _ in query.blocks]
        self.excluded = set()
        self.connection.create_function('bird_distance', 2, self.distance)
        for block in (() if simple else query.blocks):
            self.connection.execute(f'CREATE TEMP TABLE IF NOT EXISTS __candidates_{block.number} ({quote(CTID)} TEXT PRIMARY KEY)')
            self.connection.execute(f'DELETE FROM __candidates_{block.number}')
        from vec2plan_bandit.execution.timing import evaluation_deadline_ns
        def expired():
            deadline = evaluation_deadline_ns()
            return int(deadline is not None and time.perf_counter_ns() >= deadline)
        self.connection.set_progress_handler(expired, 2000)

    def ctid(self, block, point_id):
        return self.ctids(block, [point_id])[0]

    def ctids(self, block, point_ids):
        from vec2plan_bandit.execution.identity import fetch_point_values
        missing = [int(i) for i in point_ids if (block.asset_key, int(i)) not in self._ctids]
        if missing:
            values = fetch_point_values(self.connection, map_table(block.asset_key), CTID, missing)
            if len(values) != len(set(missing)):
                raise ValueError(f'Unknown ANN point ID in {block.asset_key}')
            self._ctids.update({(block.asset_key, int(i)): ctid for i, ctid in values.items()})
            self._point_ids.update({(block.asset_key, ctid): int(i) for i, ctid in values.items()})
        return [self._ctids[block.asset_key, int(i)] for i in point_ids]

    def distance(self, number, ctid):
        number = int(number)
        if ctid not in self.scores[number]:
            block = self.query.blocks[number]
            point_id = self._point_ids.get((block.asset_key, ctid))
            if point_id is None:
                row = self.connection.execute(f'SELECT point_id FROM {quote(map_table(block.asset_key))} WHERE {quote(CTID)}=?', (ctid,)).fetchone()
                if row is None: return None
                point_id = row[0]
            # HNSWFlat holds the normalized cosine vectors locally. Use the same
            # local reranker for FAISS and ACORN; never fetch vectors from PG.
            source = np.asarray(self.assets[block.asset_key]['index'].reconstruct(point_id), dtype=np.float64)
            vector = np.asarray(self.vectors[number], dtype=np.float64)
            self.scores[number][ctid] = float(1 - np.dot(source, vector) / (np.linalg.norm(source) * np.linalg.norm(vector)))
        return self.scores[number][ctid]

    def eligible_ids(self, block):
        key = (self.query.query_id, block.number)
        if key not in self.plans:
            tree = local_tree(self.query, eligibility=block.number)
            self.plans[key] = tree.sql(dialect='sqlite') if tree is not None else None
        statement = self.plans[key]
        if statement is None: return None
        hints=self.manifest.get('eligibility_estimates',{}).get(self.query.query_id)
        if hints is not None:
            exclude=hints[block.number]>.5
            operator='NOT IN' if exclude else 'IN'
            sql=f'SELECT point_id FROM {quote(map_table(block.asset_key))} WHERE {quote(CTID)} {operator} ({statement})'
            ids=[row[0] for row in self.connection.execute(sql)]
            if exclude:
                if not ids:return None
                self.excluded.add(block.number)
            return ids
        sql = f'SELECT point_id FROM {quote(map_table(block.asset_key))} WHERE {quote(CTID)} IN ({statement})'
        count = self.connection.execute(f'SELECT count(*) FROM ({sql})').fetchone()[0]
        if count == self.assets[block.asset_key]['index'].ntotal: return None
        if count > self.assets[block.asset_key]['index'].ntotal / 2:
            self.excluded.add(block.number)
            return [row[0] for row in self.connection.execute(
                f'SELECT point_id FROM {quote(map_table(block.asset_key))} WHERE {quote(CTID)} NOT IN ({statement})')]
        return [row[0] for row in self.connection.execute(sql)]

    def filter_ids(self, block, ids):
        """Filter FAISS's shortlist, rather than enumerate all eligible rows."""
        key = (self.query.query_id, block.number, 'probe')
        if key not in self.plans:
            tree = local_tree(self.query, eligibility=block.number)
            if tree is None: self.plans[key] = None
            else:
                tree.where(exp.In(this=exp.column(CTID,table=block.alias,quoted=True),
                    query=exp.Subquery(this=exp.select(CTID).from_(f'__probe_{block.number}'))),append=True,copy=False)
                tree=reduce_grouped_joins(tree)
                self.plans[key] = tree.sql(dialect='sqlite')
        statement = self.plans[key]
        if statement is None: return set(ids)
        name = f'__probe_{block.number}'
        self.connection.execute(f'CREATE TEMP TABLE IF NOT EXISTS {name} ({quote(CTID)} TEXT PRIMARY KEY, point_id INTEGER)')
        self.connection.execute(f'DELETE FROM {name}')
        ctids = self.ctids(block, ids)
        self.connection.executemany(f'INSERT INTO {name} VALUES (?,?)', zip(ctids, map(int, ids)))
        return {row[0] for row in self.connection.execute(f'SELECT point_id FROM {name} WHERE {quote(CTID)} IN ({statement})')}

    def assemble(self, point_ids):
        source=parse(self.query.source_sql)
        if len(self.query.blocks)==1 and not any(source.args.get(k) for k in ('with_','joins','group','having','distinct','offset')):
            distance=next(f.parent for f in source.find_all(exp.Anonymous) if f.name.lower()=='lembed')
            order=source.args.get('order')
            alias=distance.parent.alias if isinstance(distance.parent,exp.Alias) else None
            if (order and len(order.expressions)==1 and not order.expressions[0].args.get('desc')
                and (order.expressions[0].this==distance or
                     isinstance(order.expressions[0].this,exp.Column) and order.expressions[0].this.name==alias)):
                ctids=self.ctids(self.query.blocks[0],point_ids[0])
                ctids.sort(key=lambda c:(self.distance(0,c),locale.strxfrm(c),c))
                return [[ctid] for ctid in ctids[:self.query.k]]
        for block, ids in zip(self.query.blocks, point_ids):
            self.connection.executemany(f'INSERT OR IGNORE INTO __candidates_{block.number} VALUES (?)',
                ((ctid,) for ctid in self.ctids(block, ids)))
        key = (self.query.query_id, 'results')
        if key not in self.plans:
            self.plans[key] = local_tree(self.query, candidates=True).sql(dialect='sqlite')
        return [json.loads(row[0]) for row in self.connection.execute(self.plans[key])]

    def interrupt(self):
        if self.connection: self.connection.interrupt()

    def close(self):
        if self.connection: self.connection.close()
