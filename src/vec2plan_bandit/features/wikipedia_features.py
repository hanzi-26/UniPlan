"""Train-only 512D feature preprocessing, preserving shared action feature offsets."""
import math
import re
import numpy as np
from vec2plan_bandit.features.feature_pipeline import FeaturePipeline, TOKEN_RE, PREDICATE_TYPES
from vec2plan_bandit.features.projector import TrainOnlyPCA
from vec2plan_bandit.queries.wikipedia_sql_adapter import COLUMNS, parse
from sqlglot import exp
from vec2plan_bandit.queries.lightweight import sql_static_facts


class WikiFeaturePipeline(FeaturePipeline):
    feature_version = 'wiki-pre-execution-v1'

    @property
    def base_feature_names(self):
        return super().base_feature_names + ['vector_block_count', *['wiki_' + c for c in COLUMNS]]

    def _base_vector(self, query, statistics):
        where_count, join_count, literal_count, has_between, has_order = sql_static_facts(query.source_sql)
        stats = (statistics or {}).get(query.query_id, {})
        count, selectivity = stats.get('candidate_count'), stats.get('estimated_selectivity')
        values = [math.log1p(query.k), 0., math.log1p(len(query.question)),
                  math.log1p(len(query.semantic_text)), math.log1p(len(TOKEN_RE.findall(query.semantic_text))),
                  float(where_count), float(literal_count),
                  float(has_between), float(has_order),
                  float(selectivity or 0), float(selectivity is not None),
                  math.log1p(float(count or 0)), float(count is not None), float(not where_count),
                  math.log1p(join_count), float(bool(join_count)),
                  *[0.] * len(PREDICATE_TYPES), float(len(query.blocks)),
                  *[float(any(b.column == c for b in query.blocks)) for c in COLUMNS]]
        return np.asarray(values)

    def fit(self, records, train_query_ids, *, statistics=None, embeddings=None, split='train'):
        # Reuse the shared train split validation and standardizer. Only the
        # dataset-specific encoder dimension and schema indicators differ.
        super().fit(records, train_query_ids, statistics=statistics, embeddings=None, split=split)
        self.feature_version = 'wiki-pre-execution-v1'
        if embeddings is not None:
            matrix = np.vstack([embeddings[q] for q in self.fit_query_ids])
            if matrix.shape[1] not in (384, 512) or not np.all(np.isfinite(matrix)):
                raise ValueError('finite 384/512-dimensional Wiki embeddings required')
            self.feature_version += f'-{matrix.shape[1]}d'
            self.pca_ = TrainOnlyPCA(self.pca_dimensions).fit(matrix)
            projection = self.pca_.transform(matrix)
            self.mean_ = np.concatenate([self.mean_, projection.mean(axis=0)])
            self.scale_ = np.concatenate([self.scale_, np.where(projection.std(axis=0) > 1e-12, projection.std(axis=0), 1.)])
            self.embedding_enabled_ = True
        return self
