"""Shared SQL adapter with scalar-aggregate result identity support."""
from sqlglot import exp
from vec2plan_bandit.bird.adapter import Query as SharedQuery, parse

class Query(SharedQuery):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        tree=parse(self.source_sql)
        if not tree.args.get('group') and any(e.find(exp.AggFunc) for e in tree.expressions):
            self.k=1  # A scalar aggregation returns one output row, regardless of inner LIMIT.
    def identity(self,select):
        if not select.args.get('group') and any(e.find(exp.AggFunc) for e in select.expressions):
            # Compare aggregate values, never assign every possible aggregate the same ID.
            exprs=[e.this if isinstance(e,exp.Alias) else e for e in select.expressions]
            return 'ARRAY['+','.join('('+e.sql(dialect='postgres')+')::text' for e in exprs)+']'
        return super().identity(select)
