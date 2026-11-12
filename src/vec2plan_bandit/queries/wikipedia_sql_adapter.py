"""Wikipedia SQL/row identity adapter. Source SQL is data, never instructions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re

import sqlglot
from sqlglot import exp

from vec2plan_bandit.services.vector_contract import MODEL
from vec2plan_bandit.queries.cosine_sql import DIALECT
from vec2plan_bandit.queries.lightweight import cached_plan, parsed_select
KEYS = {'Articles': ('article_id',), 'Headings': ('heading_id',),
        'Images': ('image_id',), 'Paragraphs': ('paragraph_id',),
        'Image_Headings': ('image_id', 'heading_id')}
COLUMNS = {'raw_html_embedding': 'Articles', 'raw_wikitext_embedding': 'Articles',
           'heading_text_embedding': 'Headings', 'description_embedding': 'Images',
           'caption_embedding': 'Images', 'text_embedding': 'Paragraphs'}

TABLE_COLUMNS = {
    'Articles': ('article_id','wiki_id','title','url','raw_html','raw_wikitext','raw_html_embedding','raw_wikitext_embedding'),
    'Headings': ('heading_id','heading_text','parent_heading_id','heading_text_embedding'),
    'Images': ('image_id','article_id','filename','image_title','parsed_title','url','is_icon','on_commons','description','caption','description_embedding','caption_embedding'),
    'Paragraphs': ('paragraph_id','article_id','paragraph_index','text','text_embedding'),
    'Image_Headings': ('image_id','heading_id'),
}


def quote(value):
    return '"' + value.replace('"', '""') + '"'


def relations(select):
    result = [select.args['from_'].this]
    result += [j.this for j in select.args.get('joins', [])]
    if any(not isinstance(t, exp.Table) for t in result):
        raise ValueError('unsupported derived FROM relation')
    return result


def identity(select):
    parts = []
    for table in relations(select):
        alias = quote(table.alias_or_name)
        if table.name in KEYS:
            parts.append('ARRAY[' + ','.join(f'{alias}.{quote(k)}' for k in KEYS[table.name]) + ']::bigint[]')
        else:
            parts.append(f'{alias}."__wiki_ids"')
    return ' || '.join(parts)


def parse(sql):
    tree = parsed_select(sql)
    if any(tree.find(t) for t in (exp.Insert, exp.Update, exp.Delete, exp.Command, exp.Into)):
        raise ValueError('read-only SELECT required')
    for function in tree.find_all(exp.Anonymous):
        if function.name.lower() != 'lembed':
            raise ValueError('unsupported SQL function')
    return tree


@dataclass(frozen=True)
class VectorBlock:
    number: int
    table: str
    alias: str
    column: str
    text: str
    k: int

    @property
    def asset_key(self):
        return f'{self.table}.{self.column}'


@dataclass
class WikiQuery:
    query_id: str
    group_id: str
    question: str
    source_sql: str
    source_hash: str
    blocks: tuple[VectorBlock, ...]
    k: int

    @property
    def semantic_text(self):
        return ' '.join(b.text for b in self.blocks)

    @property
    def embedding_column(self):
        return self.blocks[0].column if self.blocks else ''

    @property
    def predicates(self):
        return ()

    def tree(self, *, ids_only=True, stable_ties=True):
        tree = parse(self.source_sql)
        # CTE projections retain source columns, with an additional composite
        # identity. IN subqueries retain their one-column projection unchanged.
        selects = [cte.this for cte in tree.find_all(exp.CTE)] + [tree]
        ctes = {cte.alias_or_name: cte.this for cte in tree.find_all(exp.CTE)}
        def original_columns(name, seen=()):
            if name in TABLE_COLUMNS:
                return TABLE_COLUMNS[name]
            if name in seen or name not in ctes:
                raise ValueError('unsupported CTE wildcard lineage')
            return tuple(e.alias_or_name for e in expanded(ctes[name], (*seen, name)))
        def expanded(select, seen=()):
            expressions = []
            for e in select.expressions:
                wildcard = isinstance(e, exp.Star) or isinstance(e, exp.Column) and isinstance(e.this, exp.Star)
                if not wildcard:
                    expressions.append(e.copy())
                    continue
                targets = [t for t in relations(select) if not isinstance(e, exp.Column) or t.alias_or_name == e.table]
                for t in targets:
                    expressions.extend(exp.column(c, table=t.alias_or_name, quoted=True)
                                       for c in original_columns(t.name, seen))
            return expressions
        # Expand source wildcards before injecting identities so a filtered CTE's
        # hidden identity cannot be duplicated by a downstream SELECT *.
        projections = {id(s): expanded(s) for s in selects}
        for select in selects:
            select.set('expressions', projections[id(select)])
            expression = exp.Var(this=identity(select))
            if select is tree and ids_only:
                distances = [e for e in select.expressions if e.find(exp.Distance)]
                select.set('expressions', [exp.alias_(expression, '__wiki_ids', quoted=True), *distances])
            else:
                select.append('expressions', exp.alias_(expression, '__wiki_ids', quoted=True))
        functions = list(tree.find_all(exp.Anonymous))
        for number, function in enumerate(functions):
            select = function.find_ancestor(exp.Select)
            function.replace(exp.Var(this=f'%(wiki_v_{number})s::vector'))
            if stable_ties and select.args.get('order'):
                select.args['order'].append('expressions', exp.Ordered(this=exp.Var(this=identity(select))))
        return tree

    def render(self, vectors, *, candidates=None, exact=False, ids_only=True):
        statement = cached_plan(self, ('render', candidates is not None, exact, ids_only),
            lambda: self._render_template(candidates is not None, exact, ids_only))
        parameters = {f'wiki_v_{i}': '[' + ','.join(str(float(v)) for v in vector) + ']'
                      for i, vector in enumerate(vectors)}
        if candidates is not None:
            parameters.update({f'wiki_ids_{b.number}': list(candidates[b.number]) for b in self.blocks})
        return statement, parameters

    def _render_template(self, with_candidates, exact, ids_only):
        tree = self.tree(ids_only=ids_only)
        distances = list(tree.find_all(exp.Distance))
        for block, distance in zip(self.blocks, distances):
            select = distance.find_ancestor(exp.Select)
            if with_candidates:
                key = f'wiki_ids_{block.number}'
                predicate = exp.Var(this=f'{quote(block.alias)}.{quote(KEYS[block.table][0])} = ANY(%({key})s::bigint[])')
                select.where(predicate, append=True, copy=False)
            if exact:
                distance.replace(exp.Paren(this=exp.Add(this=exp.Paren(this=distance.copy()), expression=exp.Literal.number('0.0'))))
        return tree.sql(dialect=DIALECT)

    def eligible_sql(self, number, vectors):
        statement = cached_plan(self, ('eligible', number), lambda: self._eligible_template(number))
        parameters = {f'wiki_v_{i}': '[' + ','.join(str(float(v)) for v in vector) + ']'
                      for i, vector in enumerate(vectors)}
        return statement, parameters

    def _eligible_template(self, number):
        tree = self.tree(ids_only=False)
        distance = list(tree.find_all(exp.Distance))[number]
        select = distance.find_ancestor(exp.Select).copy()
        block = self.blocks[number]
        select.set('expressions', [exp.Var(this=f'{quote(block.alias)}.{quote(KEYS[block.table][0])}')])
        select.set('distinct', exp.Distinct())
        for field in ('order', 'limit', 'offset'):
            select.set(field, None)
        if tree.args.get('with_') and select.args.get('with_') is None:
            select.set('with_', tree.args['with_'].copy())
        return select.sql(dialect=DIALECT)


def load_queries(path: Path, embedding_model=MODEL):
    records = json.loads(path.read_text())
    queries = []
    for n, record in enumerate(records, 1):
        tree = parse(record['sql'])
        blocks = []
        for i, function in enumerate(tree.find_all(exp.Anonymous)):
            if len(function.expressions) != 2 or any(not isinstance(v, exp.Literal) or not v.is_string for v in function.expressions):
                raise ValueError(f'W{n:03d}: literal model/text required')
            model, text = [v.this for v in function.expressions]
            if model != embedding_model or not isinstance(function.parent, exp.Distance):
                raise ValueError(f'W{n:03d}: configured MiniLM cosine expression required')
            column = function.parent.this
            if not isinstance(column, exp.Column) or column.name not in COLUMNS:
                raise ValueError(f'W{n:03d}: unsupported vector column')
            select = function.find_ancestor(exp.Select)
            table = COLUMNS[column.name]
            ctes = {c.alias_or_name: c.this for c in tree.find_all(exp.CTE)}
            def base_name(relation, seen=()):
                if relation.name in KEYS:
                    return relation.name
                if relation.name in seen or relation.name not in ctes:
                    return None
                sources = relations(ctes[relation.name])
                return base_name(sources[0], (*seen, relation.name)) if len(sources) == 1 else None
            targets = [t for t in relations(select) if base_name(t) == table and (not column.table or t.alias_or_name == column.table)]
            if len(targets) != 1 or select.args.get('limit') is None:
                raise ValueError(f'W{n:03d}: ambiguous vector relation or missing LIMIT')
            k = int(select.args['limit'].expression.this)
            if not 1 <= k <= 10:
                raise ValueError('source k outside 1..10')
            blocks.append(VectorBlock(i, table, targets[0].alias_or_name, column.name, text, k))
        semantics = sorted(' '.join(b.text.casefold().split()) for b in blocks)
        group = hashlib.sha256(json.dumps(semantics).encode()).hexdigest()[:16]
        root_limit = tree.args.get('limit')
        k = int(root_limit.expression.this) if root_limit else max((b.k for b in blocks), default=0)
        queries.append(WikiQuery(f'W{n:03d}', group, str(record.get('question', '')), tree.sql(dialect=DIALECT),
                         hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), tuple(blocks), k))
    return queries


def semantic_query_groups(queries):
    # Shared semantic text (including multi-vector overlaps) cannot cross roles.
    groups = []
    for query in queries:
        terms = {' '.join(b.text.casefold().split()) for b in query.blocks}
        overlap = [g for g in groups if g[0] & terms]
        ids = [query.query_id]
        for group in overlap:
            terms |= group[0]
            ids += group[1]
            groups.remove(group)
        groups.append((terms, ids))
    return [ids for _, ids in groups]


def grouped_split(queries, seed=42):
    groups = semantic_query_groups(queries)
    random.Random(seed).shuffle(groups)
    roles = {'train': [], 'dev': [], 'test': []}
    targets = {'train': .6 * len(queries), 'dev': .2 * len(queries), 'test': .2 * len(queries)}
    for ids in sorted(groups, key=lambda g: -len(g)):
        role = max(roles, key=lambda r: (targets[r] - len(roles[r])) / targets[r])
        roles[role].extend(ids)
    return roles
