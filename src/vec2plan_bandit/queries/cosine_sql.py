"""pgvector cosine operators, without rewriting operators inside SQL strings.

SQLGlot interprets <=> as null-safe equality in its PostgreSQL dialect. This
dialect retains the vector-distance AST used by our lineage adapters and always
emits cosine distance. Legacy <-> input is migrated when rendered.
"""
from sqlglot.dialects.postgres import Postgres
from sqlglot.tokens import TokenType


class CosinePostgres(Postgres):
    class Tokenizer(Postgres.Tokenizer):
        KEYWORDS = {**Postgres.Tokenizer.KEYWORDS, '<=>': TokenType.LR_ARROW}

    class Generator(Postgres.Generator):
        def distance_sql(self, expression):
            return self.binary(expression, '<=>')


DIALECT = 'cosinepostgres'
