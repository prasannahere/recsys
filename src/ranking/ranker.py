"""LightGBM LambdaMART ranker (retriever-agnostic)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import polars as pl

from src.io.storage import open_binary

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


class LambdaMartRanker:
    """LightGBM LambdaMART ranker over retrieval candidates.

    Each retriever trains its own ranker with retriever-specific features.
    Training data is grouped by user for listwise LambdaMART optimization.
    """

    def __init__(
        self,
        feature_cols: list[str],
        params: dict[str, Any] | None = None,
        random_state: int = 42,
        retriever: str = "unknown",
    ) -> None:
        merged = {**DEFAULT_LGBM_PARAMS, **(params or {})}
        merged["random_state"] = random_state
        self.feature_cols = feature_cols
        self.params = merged
        self.retriever = retriever
        self.model = lgb.LGBMRanker(**merged)

    def fit(self, training_pairs: pl.DataFrame) -> None:
        df = training_pairs.sort("customer_id")
        X = df.select(self.feature_cols).to_numpy()
        y = df["label"].to_numpy()
        group = (
            df.group_by("customer_id", maintain_order=True)
            .agg(pl.len())
            .get_column("len")
            .to_numpy()
        )

        self.model.fit(X, y, group=group)
        logger.info(
            "%s LambdaMART ranker trained on %d pairs across %d users (%.1f%% positive)",
            self.retriever,
            len(y),
            len(group),
            100.0 * y.mean(),
        )

    def predict_scores(self, df: pl.DataFrame) -> pl.Series:
        X = df.select(self.feature_cols).to_numpy()
        return pl.Series("ranker_score", self.model.predict(X))

    def rerank_candidates(self, candidates_with_features: pl.DataFrame) -> pl.DataFrame:
        """Rescore candidates and assign new ranks by ranker_score."""
        scored = candidates_with_features.with_columns(
            self.predict_scores(candidates_with_features)
        )
        return (
            scored.sort(["customer_id", "ranker_score"], descending=[False, True])
            .with_columns(
                pl.col("ranker_score")
                .rank(method="ordinal", descending=True)
                .over("customer_id")
                .alias("rank")
            )
            .select(["customer_id", "article_id", "ranker_score", "rank"])
        )

    def save(self, path: str | Path) -> None:
        path_str = str(path)
        with open_binary(path_str, "wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_cols": self.feature_cols,
                    "params": self.params,
                    "retriever": self.retriever,
                },
                handle,
            )
        logger.info("Saved %s LambdaMART ranker to %s", self.retriever, path_str)

    @classmethod
    def load(cls, path: str | Path) -> LambdaMartRanker:
        with open_binary(path, "rb") as handle:
            data = pickle.load(handle)
        ranker = cls.__new__(cls)
        ranker.feature_cols = data["feature_cols"]
        ranker.params = data.get("params", DEFAULT_LGBM_PARAMS.copy())
        ranker.retriever = data.get("retriever", "unknown")
        ranker.model = data["model"]
        return ranker


# Backward-compatible alias for notebooks and older imports.
PointwiseRanker = LambdaMartRanker
