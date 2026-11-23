"""ALS candidate retrieval."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from src.data.mappings import IdMapping
from src.retrieval.als.model import ALSModel

if TYPE_CHECKING:
    from src.evaluation.progress import ProgressLogger

logger = logging.getLogger(__name__)

RETRIEVER_NAME = "als"

_CANDIDATE_SCHEMA = {
    "customer_id": pl.Utf8,
    "article_id": pl.Utf8,
    "retrieval_score": pl.Float64,
    "rank": pl.Int64,
    "retriever": pl.Utf8,
}


def _empty_candidates() -> pl.DataFrame:
    return pl.DataFrame(schema=_CANDIDATE_SCHEMA)


def generate_candidates(
    model: ALSModel,
    user_indices: list[int],
    user_map: IdMapping,
    item_map: IdMapping,
    k: int,
    filter_seen: bool = True,
    progress: ProgressLogger | None = None,
) -> pl.DataFrame:
    """Produce candidate table with the common retrieval schema."""
    batch = model.recommend_batch(user_indices, k, filter_seen, progress=progress)
    if not batch:
        return _empty_candidates()

    user_idx_col: list[int] = []
    item_idx_col: list[int] = []
    score_col: list[float] = []
    rank_col: list[int] = []

    for user_idx, (item_indices, scores) in batch.items():
        n = len(item_indices)
        if n == 0:
            continue
        user_idx_col.extend([user_idx] * n)
        item_idx_col.extend(int(i) for i in item_indices)
        score_col.extend(float(s) for s in scores)
        rank_col.extend(range(1, n + 1))

    if not user_idx_col:
        return _empty_candidates()

    raw = pl.DataFrame(
        {
            "user_idx": user_idx_col,
            "item_idx": item_idx_col,
            "retrieval_score": score_col,
            "rank": rank_col,
        }
    )
    return (
        raw.join(
            user_map.forward.rename({"idx": "user_idx", "original_id": "customer_id"}),
            on="user_idx",
            how="left",
        )
        .join(
            item_map.forward.rename({"idx": "item_idx", "original_id": "article_id"}),
            on="item_idx",
            how="left",
        )
        .filter(pl.col("customer_id").is_not_null() & pl.col("article_id").is_not_null())
        .with_columns(pl.lit(RETRIEVER_NAME).alias("retriever"))
        .select(list(_CANDIDATE_SCHEMA.keys()))
    )


def sanity_check_recommendations(
    model: ALSModel,
    user_map: IdMapping,
    item_map: IdMapping,
    sample_user_indices: list[int],
    k: int = 5,
    filter_seen: bool = True,
) -> None:
    """Log sample recommendations and verify scores are descending."""
    logger.info("--- Sample ALS recommendations ---")
    for user_idx in sample_user_indices:
        item_indices, scores = model.recommend(user_idx, k, filter_seen)
        customer_id = user_map.to_original(user_idx)
        logger.info("User %s (idx=%d):", customer_id[:16], user_idx)
        for rank, (item_idx, score) in enumerate(
            zip(item_indices, scores, strict=True), start=1
        ):
            article_id = item_map.to_original(int(item_idx))
            logger.info("  rank=%d article=%s score=%.4f", rank, article_id, score)

        if len(scores) > 1:
            assert np.all(scores[:-1] >= scores[1:]), (
                f"Scores not descending for user {user_idx}"
            )
