"""Ranking features for pure Two-Tower retrieval candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from src.data.mappings import IdMapping
from src.ranking.common import sample_labeled_pairs
from src.retrieval.two_tower.model import TwoTowerModel

if TYPE_CHECKING:
    from src.evaluation.progress import ProgressLogger

FEATURE_COLS = ["two_tower_score", "log_popularity", "log_user_activity"]
RETRIEVER_NAME = "two_tower"


def _popularity_frame(train_df: pl.DataFrame) -> pl.DataFrame:
    return (
        train_df.group_by("article_id")
        .agg(pl.len().alias("purchase_count"))
        .with_columns(pl.col("purchase_count").log1p().alias("log_popularity"))
        .select("article_id", "log_popularity")
    )


def _activity_frame(train_df: pl.DataFrame) -> pl.DataFrame:
    return (
        train_df.group_by("customer_id")
        .agg(pl.len().alias("activity_count"))
        .with_columns(pl.col("activity_count").log1p().alias("log_user_activity"))
        .select("customer_id", "log_user_activity")
    )


def batch_dot_scores(
    model: TwoTowerModel,
    user_indices: list[int] | np.ndarray,
    item_indices: list[int] | np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    if len(user_indices) == 0:
        return np.array([], dtype=np.float64)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)
        user_emb = model.encode_users(user_tensor)
        item_emb = model.encode_items(item_tensor)
        scores = (user_emb * item_emb).sum(dim=-1).cpu().numpy()
    return scores.astype(np.float64)


def _attach_two_tower_scores(
    frame: pl.DataFrame,
    model: TwoTowerModel,
    user_map: IdMapping,
    item_map: IdMapping,
    device: torch.device | None = None,
) -> pl.DataFrame:
    featured = (
        frame.join(
            user_map.forward.rename({"original_id": "customer_id", "idx": "user_idx"}),
            on="customer_id",
            how="inner",
        )
        .join(
            item_map.forward.rename({"original_id": "article_id", "idx": "item_idx"}),
            on="article_id",
            how="inner",
        )
    )
    if featured.height == 0:
        return featured

    user_idx = featured["user_idx"].to_numpy()
    item_idx = featured["item_idx"].to_numpy()
    scores = batch_dot_scores(model, user_idx, item_idx, device=device)
    return featured.with_columns(pl.Series("two_tower_score", scores))


def build_training_pairs(
    train_df: pl.DataFrame,
    model: TwoTowerModel,
    user_map: IdMapping,
    item_map: IdMapping,
    max_users: int | None = 50_000,
    negatives_per_positive: int = 5,
    random_state: int = 42,
    device: torch.device | None = None,
) -> pl.DataFrame:
    pairs = sample_labeled_pairs(
        train_df,
        user_map,
        item_map,
        max_users=max_users,
        negatives_per_positive=negatives_per_positive,
        random_state=random_state,
    )
    if not pairs:
        return pl.DataFrame(
            schema={
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
                "label": pl.Int64,
                "two_tower_score": pl.Float64,
                "log_popularity": pl.Float64,
                "log_user_activity": pl.Float64,
            }
        )

    pairs_df = pl.DataFrame(
        {
            "customer_id": [p[0] for p in pairs],
            "article_id": [p[1] for p in pairs],
            "label": [p[2] for p in pairs],
        }
    )
    scored = _attach_two_tower_scores(pairs_df, model, user_map, item_map, device=device)
    return (
        scored.join(_popularity_frame(train_df), on="article_id", how="left")
        .join(_activity_frame(train_df), on="customer_id", how="left")
        .with_columns(
            pl.col("log_popularity").fill_null(0.0),
            pl.col("log_user_activity").fill_null(0.0),
        )
        .select(
            "customer_id",
            "article_id",
            "label",
            "two_tower_score",
            "log_popularity",
            "log_user_activity",
        )
    )


def add_features_to_candidates(
    candidates: pl.DataFrame,
    model: TwoTowerModel,
    user_map: IdMapping,
    item_map: IdMapping,
    train_df: pl.DataFrame,
    device: torch.device | None = None,
    progress: ProgressLogger | None = None,
) -> pl.DataFrame:
    if candidates.height == 0:
        return pl.DataFrame(
            schema={
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
                "retrieval_score": pl.Float64,
                "retrieval_rank": pl.Int64,
                "two_tower_score": pl.Float64,
                "log_popularity": pl.Float64,
                "log_user_activity": pl.Float64,
            }
        )

    base_cols = [
        c
        for c in ("customer_id", "article_id", "retrieval_score", "rank")
        if c in candidates.columns
    ]
    featured = _attach_two_tower_scores(
        candidates.select(base_cols),
        model,
        user_map,
        item_map,
        device=device,
    )
    featured = (
        featured.join(_popularity_frame(train_df), on="article_id", how="left")
        .join(_activity_frame(train_df), on="customer_id", how="left")
        .with_columns(
            pl.col("log_popularity").fill_null(0.0),
            pl.col("log_user_activity").fill_null(0.0),
        )
    )
    if "rank" in featured.columns:
        featured = featured.rename({"rank": "retrieval_rank"})
    if progress is not None:
        progress.update(featured.height, force=True)
    return featured.select(
        "customer_id",
        "article_id",
        "retrieval_score",
        "retrieval_rank",
        "two_tower_score",
        "log_popularity",
        "log_user_activity",
    )
