from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vec2plan_bandit.features.projector import TrainOnlyPCA
from vec2plan_bandit.queries.loader import QueryRecord


PREDICATE_TYPES = (
    "submitter_eq",
    "submitter_in",
    "author_eq",
    "author_in",
    "author_like",
    "category_eq",
    "category_in",
    "latest_update",
    "license_eq",
    "license_in",
    "comments_eq",
    "comments_like",
    "journal_ref_eq",
    "journal_ref_like",
    "report_no_eq",
    "update_date_eq",
    "update_date_le",
    "update_date_ge",
    "update_date_between",
    "update_date_outside",
)
SUPPORTED_PREDICATE_TYPES = PREDICATE_TYPES + ("update_date_range",)
FORBIDDEN_RESULT_FIELDS = {
    "latency",
    "recall",
    "execution_success",
    "reward",
    "oracle_action",
    "oracle_reward",
    "failure_reason",
    "test_statistics",
}
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
RANGE_RE = re.compile(r"\bBETWEEN\b|>=|<=|(?<!<)<(?![=>-])|(?<![-=])>(?![=])", re.I)


def load_frozen_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load an explicitly aligned NPZ without generating any embeddings."""
    payload = np.load(path, allow_pickle=False)
    if set(("query_ids", "embeddings")) - set(payload.files):
        raise ValueError("embedding NPZ must contain query_ids and embeddings")
    ids = [str(value) for value in payload["query_ids"].tolist()]
    matrix = np.asarray(payload["embeddings"], dtype=np.float64)
    if matrix.ndim != 2 or len(ids) != len(matrix) or len(ids) != len(set(ids)):
        raise ValueError("invalid or duplicate embedding alignment")
    if matrix.shape[1] != 384 or not np.all(np.isfinite(matrix)):
        raise ValueError("expected finite 384-dimensional embeddings")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError("frozen query embeddings must be L2 normalized")
    return {query_id: matrix[index] for index, query_id in enumerate(ids)}


@dataclass
class FeaturePipeline:
    pca_dimensions: int = 8
    feature_version: str = "pre-execution-features-v3-join-shape"
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    pca_: TrainOnlyPCA | None = None
    fit_query_ids: tuple[str, ...] = ()
    embedding_enabled_: bool = False

    @property
    def base_feature_names(self) -> list[str]:
        names = [
            "log1p_k",
            "embedding_is_title",
            "log1p_question_chars",
            "log1p_semantic_chars",
            "log1p_semantic_tokens",
            "filter_count",
            "filter_value_count",
            "has_range_condition",
            "has_order_by",
            "estimated_selectivity",
            "estimated_selectivity_available",
            "log1p_candidate_count",
            "candidate_count_available",
            "filter_type_none",
            "log1p_relational_join_count",
            "has_relational_join",
        ]
        names.extend(f"filter_type_{kind}" for kind in PREDICATE_TYPES)
        return names

    @property
    def feature_names(self) -> list[str]:
        names = ["intercept"] + self.base_feature_names
        if self.embedding_enabled_ and self.pca_ and self.pca_.components_ is not None:
            names.extend(f"query_embedding_pca_{i}" for i in range(len(self.pca_.components_)))
        return names

    def _base_vector(
        self,
        query: QueryRecord,
        statistics: Mapping[str, Mapping[str, Any]] | None,
    ) -> np.ndarray:
        stats = dict((statistics or {}).get(query.query_id, {}))
        selectivity = stats.get("estimated_selectivity")
        candidate_count = stats.get("candidate_count")
        selectivity_known = selectivity is not None
        candidate_known = candidate_count is not None
        if selectivity_known:
            selectivity = float(selectivity)
            if not math.isfinite(selectivity) or not 0 <= selectivity <= 1:
                raise ValueError(f"invalid estimated_selectivity for {query.query_id}")
        if candidate_known:
            candidate_count = float(candidate_count)
            if not math.isfinite(candidate_count) or candidate_count < 0:
                raise ValueError(f"invalid candidate_count for {query.query_id}")
        predicate_types = [str(item["type"]) for item in query.predicates]
        unknown = set(predicate_types) - set(SUPPORTED_PREDICATE_TYPES)
        if unknown:
            raise ValueError(f"unknown predicate types for {query.query_id}: {sorted(unknown)}")
        value_count = sum(len(item.get("values", [])) for item in query.predicates)
        join_count = sum(
            2 if kind.startswith(("author_", "category_"))
            else 1 if kind.startswith("submitter_")
            else 0
            for kind in predicate_types
        )
        range_sql = query.source_sql.replace("<->", "").replace("<=>", "")
        values = [
            math.log1p(query.k),
            float(query.embedding_column == "title_embedding"),
            math.log1p(len(query.question)),
            math.log1p(len(query.semantic_text)),
            math.log1p(len(TOKEN_RE.findall(query.semantic_text))),
            float(len(query.predicates)),
            float(value_count),
            float(getattr(query, "has_range_condition", bool(RANGE_RE.search(range_sql)))),
            float(getattr(query, "has_explicit_order_by", bool(re.search(r"\bORDER\s+BY\b", query.source_sql, re.I)))),
            float(selectivity) if selectivity_known else 0.0,
            float(selectivity_known),
            math.log1p(float(candidate_count)) if candidate_known else 0.0,
            float(candidate_known),
            float(not predicate_types),
            math.log1p(join_count),
            float(join_count > 0),
        ]
        values.extend(float(kind in predicate_types) for kind in PREDICATE_TYPES)
        return np.asarray(values, dtype=np.float64)

    def fit(
        self,
        records: Sequence[QueryRecord],
        train_query_ids: Sequence[str],
        *,
        statistics: Mapping[str, Mapping[str, Any]] | None = None,
        embeddings: Mapping[str, np.ndarray] | None = None,
        split: str = "train",
    ) -> "FeaturePipeline":
        if split != "train":
            raise ValueError("feature pipeline may only be fit on train")
        leaked = set(self.base_feature_names) & FORBIDDEN_RESULT_FIELDS
        if leaked:
            raise ValueError(f"result leakage in feature schema: {sorted(leaked)}")
        by_id = {record.query_id: record for record in records}
        train_ids = tuple(str(value) for value in train_query_ids)
        if not train_ids or len(train_ids) != len(set(train_ids)):
            raise ValueError("train_query_ids must be non-empty and unique")
        if set(train_ids) - set(by_id):
            raise ValueError("train split references unknown query IDs")
        base = np.vstack([self._base_vector(by_id[query_id], statistics) for query_id in train_ids])
        combined = base
        self.embedding_enabled_ = embeddings is not None
        if embeddings is not None:
            missing = set(by_id) - set(embeddings)
            if missing:
                raise ValueError(f"frozen embeddings missing query IDs: {sorted(missing)}")
            train_embeddings = np.vstack(
                [np.asarray(embeddings[query_id], dtype=np.float64) for query_id in train_ids]
            )
            if train_embeddings.shape[1] != 384:
                raise ValueError("expected 384-dimensional query embeddings")
            norms = np.linalg.norm(train_embeddings, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
                raise ValueError("query embeddings must be L2 normalized")
            self.pca_ = TrainOnlyPCA(self.pca_dimensions).fit(train_embeddings, split="train")
            combined = np.hstack([base, self.pca_.transform(train_embeddings)])
        else:
            self.pca_ = None
        self.mean_ = combined.mean(axis=0)
        scale = combined.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        self.fit_query_ids = train_ids
        return self

    def transform_one(
        self,
        query: QueryRecord,
        *,
        statistics: Mapping[str, Mapping[str, Any]] | None = None,
        embeddings: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("feature pipeline is not fit")
        base = self._base_vector(query, statistics)
        combined = base
        if self.embedding_enabled_:
            if embeddings is None or query.query_id not in embeddings or self.pca_ is None:
                raise ValueError(f"missing frozen embedding for {query.query_id}")
            embedding = np.asarray(embeddings[query.query_id], dtype=np.float64).reshape(1, -1)
            combined = np.concatenate([base, self.pca_.transform(embedding)[0]])
        return np.concatenate([[1.0], (combined - self.mean_) / self.scale_])

    def transform(
        self,
        records: Sequence[QueryRecord],
        *,
        statistics: Mapping[str, Mapping[str, Any]] | None = None,
        embeddings: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        return {
            query.query_id: self.transform_one(
                query, statistics=statistics, embeddings=embeddings
            )
            for query in records
        }

    def audit(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "fit_split": "train",
            "fit_query_ids": list(self.fit_query_ids),
            "embedding_enabled": self.embedding_enabled_,
            "pca_dimensions": (
                int(len(self.pca_.components_))
                if self.pca_ is not None and self.pca_.components_ is not None
                else 0
            ),
            "standardizer_fit_query_count": len(self.fit_query_ids),
            "feature_names": self.feature_names,
            "unavailable_features_are_zero_with_availability_indicator": [
                "estimated_selectivity",
                "candidate_count",
            ],
            "forbidden_feedback_features": [
                "latency",
                "recall",
                "execution_success",
                "oracle_action",
                "reward",
            ],
        }

    def save(self, path: Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("feature pipeline is not fit")
        components = (
            self.pca_.components_
            if self.pca_ is not None and self.pca_.components_ is not None
            else np.empty((0, 384), dtype=np.float64)
        )
        pca_mean = (
            self.pca_.mean_
            if self.pca_ is not None and self.pca_.mean_ is not None
            else np.empty((0,), dtype=np.float64)
        )
        metadata = {
            "feature_version": self.feature_version,
            "pca_dimensions": self.pca_dimensions,
            "fit_query_ids": list(self.fit_query_ids),
            "embedding_enabled": self.embedding_enabled_,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean_,
            scale=self.scale_,
            pca_components=components,
            pca_mean=pca_mean,
            metadata=json.dumps(metadata),
        )

    @classmethod
    def load(cls, path: Path) -> "FeaturePipeline":
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"].item()))
        pipeline = cls(
            pca_dimensions=int(metadata["pca_dimensions"]),
            feature_version=str(metadata["feature_version"]),
        )
        pipeline.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        pipeline.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        pipeline.fit_query_ids = tuple(str(value) for value in metadata["fit_query_ids"])
        pipeline.embedding_enabled_ = bool(metadata["embedding_enabled"])
        components = np.asarray(payload["pca_components"], dtype=np.float64)
        if pipeline.embedding_enabled_:
            pca = TrainOnlyPCA(pipeline.pca_dimensions)
            pca.components_ = components
            pca.mean_ = np.asarray(payload["pca_mean"], dtype=np.float64)
            pca.fit_split = "train"
            pipeline.pca_ = pca
        return pipeline
