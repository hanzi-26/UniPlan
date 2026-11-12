"""Query-independent scalar columns for exact CPU membership and scoring.

Built in RAM from the existing immutable SQLite snapshot, without changing any
vector index. No query IDs, ground truth, eligibility masks or results are cached.
"""
from datetime import date
from array import array
import json

import numpy as np

from .metadata import timestamp, sql_like

FIELDS = ('license', 'comments', 'journal_ref', 'report_no')


def like_values(values, pattern):
    """Fast common LIKE forms, retaining the exact existing SQL LIKE grammar."""
    # Only literal text between optional end wildcards takes the substring
    # shortcut. Escapes, internal %, and _ retain the general regex renderer.
    literal = pattern.strip('%')
    if literal and not any(c in literal for c in ('%', '_', '\\')):
        if pattern == '%'+literal+'%': return [v is not None and literal in v for v in values]
        if pattern == literal+'%': return [v is not None and v.startswith(literal) for v in values]
        if pattern == '%'+literal: return [v is not None and v.endswith(literal) for v in values]
        if pattern == literal: return [v == literal for v in values]
    matcher = sql_like(pattern)
    return [v is not None and matcher.fullmatch(v) is not None for v in values]


class ExactMetadata:
    def __init__(self, metadata, mapping):
        self.mapping = mapping
        n = len(mapping.values)
        self.days = np.zeros(n, dtype=np.int32)
        self.latest = np.zeros(n, dtype=bool)
        self.codes = {f:np.zeros(n,dtype=np.int32) for f in FIELDS}
        self.vocab = {f:[None] for f in FIELDS}
        self.lookup = {f:{} for f in FIELDS}
        self.submitter_codes=np.zeros(n,dtype=np.int32)
        self.submitter_vocab=[None];self.submitter_lookup={}
        author_codes=array('i');author_owners=array('i')
        self.author_vocab=[None];self.author_lookup={}
        if metadata.extra_path is None:
            raise ValueError('Exact scalar metadata requires the verified extra sidecar')
        values = mapping.values
        order = np.argsort(values)
        sorted_ids = values[order]
        sql = ('SELECT a.id,json_extract(a.payload,\'$.update_date\'),'
               'json_extract(a.payload,\'$.is_latest_update\'),'
               'e.license,e.comments,e.journal_ref,e.report_no,'
               'json_extract(a.payload,\'$.author_names\'),'
               'json_extract(a.payload,\'$.submitter_name\'),e.id '
               'FROM articles a LEFT JOIN extras.article_extra e ON e.id=a.id ORDER BY a.id')
        cursor = metadata.db.execute(sql)
        seen = np.zeros(n,dtype=bool)
        while True:
            rows = cursor.fetchmany(32768)
            if not rows: break
            ids = np.asarray([r[0] for r in rows],dtype=np.int64)
            found = np.searchsorted(sorted_ids,ids)
            valid = found < n
            valid[valid] &= sorted_ids[found[valid]] == ids[valid]
            for row, ordinal in zip((rows[i] for i in np.flatnonzero(valid)), order[found[valid]]):
                if row[9] is None: raise ValueError('Exact metadata sidecar is missing a vector row')
                seen[ordinal] = True
                if row[1] is not None: self.days[ordinal] = timestamp(row[1]).date().toordinal()
                self.latest[ordinal] = row[2] == 1
                for field,value in zip(FIELDS,row[3:]):
                    if value is None: continue
                    lookup = self.lookup[field]
                    code = lookup.get(value)
                    if code is None:
                        code = len(self.vocab[field]);lookup[value]=code;self.vocab[field].append(value)
                    self.codes[field][ordinal] = code
                submitter=row[8]
                if submitter is not None:
                    code=self.submitter_lookup.get(submitter)
                    if code is None:
                        code=len(self.submitter_vocab);self.submitter_lookup[submitter]=code;self.submitter_vocab.append(submitter)
                    self.submitter_codes[ordinal]=code
                for name in json.loads(row[7] or '[]'):
                    code=self.author_lookup.get(name)
                    if code is None:
                        code=len(self.author_vocab);self.author_lookup[name]=code;self.author_vocab.append(name)
                    author_codes.append(code);author_owners.append(int(ordinal))
        if not np.all(seen): raise ValueError('Scalar metadata does not cover the full vector snapshot')
        self.author_codes=np.frombuffer(author_codes,dtype=np.int32)
        self.author_owners=np.frombuffer(author_owners,dtype=np.int32)

    def eligible(self, query):
        """Complete logical membership; unsupported relational filters return None."""
        mask = None
        for predicate in query.predicates:
            kind,values = predicate['type'],predicate['values']
            if kind.startswith('update_date_'):
                operation=kind[len('update_date_'):]
                start=date.fromisoformat(values[0][:10]).toordinal()
                d=self.days
                if operation=='eq': current=d==start
                elif operation=='le': current=(d>0)&(d<=start)
                elif operation=='ge': current=d>=start
                elif operation in ('between','outside'):
                    end=date.fromisoformat(values[1][:10]).toordinal()
                    current=((d>=start)&(d<=end) if operation=='between' else ((d>0)&(d<=start))|(d>=end))
                elif operation=='range':
                    # Timestamp ranges can have a non-midnight bound. The day
                    # representation cannot prove their exact membership.
                    return None
                else:return None
            elif kind=='latest_update': current=self.latest
            else:
                field,_,operation=kind.rpartition('_')
                if field in ('author','submitter'):
                    vocab=getattr(self,field+'_vocab');lookup=getattr(self,field+'_lookup')
                    allowed=np.zeros(len(vocab),dtype=bool)
                    if operation in ('eq','in'):
                        for value in (values[:1] if operation=='eq' else values):
                            code=lookup.get(value)
                            if code is not None:allowed[code]=True
                    elif operation=='like':allowed=np.asarray(like_values(vocab,str(values[0])),dtype=bool)
                    else:return None
                    if field=='submitter':current=allowed[self.submitter_codes]
                    else:
                        current=np.zeros(len(self.days),dtype=bool)
                        current[self.author_owners[allowed[self.author_codes]]]=True
                    mask=current.copy() if mask is None else mask & current
                    continue
                if field not in FIELDS:return None
                codes=self.codes[field]
                if operation in ('eq','in'):
                    allowed=[self.lookup[field].get(v,-1) for v in (values[:1] if operation=='eq' else values)]
                    current=np.isin(codes,allowed)
                elif operation=='like':
                    matching=np.asarray(like_values(self.vocab[field],str(values[0])),dtype=bool)
                    current=matching[codes]
                else:return None
            mask=current.copy() if mask is None else mask & current
        return np.flatnonzero(mask) if mask is not None else np.arange(len(self.days),dtype=np.int64)
