"""Build implicit interaction weights and sparse CSR matrix for ALS."""

from __future__ import annotations

import logging
from datetime import date
from typing import Literal

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

WeightingMethod = Literal["binary", "count", "recency"]


def build_interactions(
    train_df: pl.DataFrame,
    method: WeightingMethod,
    reference_date: date,
    recency_lambda: float = 0.01,
) -> pl.DataFrame:
    """Aggregate training transactions into user-item interaction weights.

    Each row represents one (user, item) pair with an implicit feedback
    strength r_ui used later as the preference term in confidence C_ui = 1 + alpha * r_ui.

    Args:
        train_df: Training transactions with customer_id, article_id, t_dat.
        method: Weighting strategy — binary, count, or recency.
        reference_date: Last training day; used as anchor for recency decay.
        recency_lambda: Decay rate for recency weighting (exp(-lambda * age_days)).

    Returns:
        DataFrame with columns: user_id, item_id, interaction_weight.
    """
    df = train_df.rename({"customer_id": "user_id", "article_id": "item_id"})

    if method == "binary":
        interactions = (
            df.group_by(["user_id", "item_id"])
            .agg(pl.lit(1.0).alias("interaction_weight"))
        )
    elif method == "count":
        interactions = (
            df.group_by(["user_id", "item_id"])
            .agg(pl.len().cast(pl.Float64).alias("interaction_weight"))
        )
    elif method == "recency":
        # weight per transaction = exp(-lambda * age_days), summed per pair
        interactions = (
            df.with_columns(
                (
                    (pl.lit(reference_date) - pl.col("t_dat"))
                    .dt.total_days()
                    .alias("age_days")
                )
            )
            .with_columns(
                (-recency_lambda * pl.col("age_days"))
                .exp()
                .alias("row_weight")
            )
            .group_by(["user_id", "item_id"])
            .agg(pl.col("row_weight").sum().alias("interaction_weight"))
        )
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    _validate_interactions(interactions)
    _log_repeated_examples(train_df, interactions, method)
    logger.info(
        "Built interactions (method=%s): %d unique user-item pairs",
        method,
        interactions.height,
    )
    return interactions


def _validate_interactions(interactions: pl.DataFrame) -> None:
    weights = interactions["interaction_weight"].to_numpy()
    assert not np.any(np.isnan(weights)), "NaN weights found in interactions"
    assert np.all(weights > 0), "All interaction weights must be > 0"


def _log_repeated_examples(
    train_df: pl.DataFrame,
    interactions: pl.DataFrame,
    method: str,
) -> None:
    """Log a few user-item pairs that appear multiple times in raw transactions."""
    counts = (
        train_df.group_by(["customer_id", "article_id"])
        .agg(pl.len().alias("purchase_count"))
        .filter(pl.col("purchase_count") > 1)
        .sort("purchase_count", descending=True)
        .head(5)
    )
    if counts.height == 0:
        logger.info("No repeated user-item pairs in training data")
        return

    logger.info("Sample repeated user-item pairs (method=%s):", method)
    for row in counts.iter_rows(named=True):
        uid = row["customer_id"]
        iid = row["article_id"]
        raw_count = row["purchase_count"]
        weight_row = interactions.filter(
            (pl.col("user_id") == uid) & (pl.col("item_id") == iid)
        )
        weight = weight_row["interaction_weight"][0] if weight_row.height > 0 else None
        logger.info(
            "  user=%s item=%s raw_count=%d aggregated_weight=%.4f",
            uid[:16],
            iid,
            raw_count,
            weight,
        )


def build_csr_matrix(
    interactions: pl.DataFrame,
    num_users: int,
    num_items: int,
) -> csr_matrix:
    """Construct a sparse user x item CSR interaction matrix.

    Row index = user_idx, column index = item_idx, value = interaction_weight.
    This is the preference matrix R; implicit ALS converts it to confidence
    C_ui = 1 + alpha * R_ui internally during training.

    Args:
        interactions: DataFrame with user_idx, item_idx, interaction_weight.
        num_users: Number of training users (matrix rows).
        num_items: Number of training items (matrix columns).

    Returns:
        scipy.sparse.csr_matrix of shape (num_users, num_items).
    """
    user_idx = interactions["user_idx"].to_numpy()
    item_idx = interactions["item_idx"].to_numpy()
    weights = interactions["interaction_weight"].to_numpy()

    assert user_idx.min() >= 0, "Negative user_idx"
    assert item_idx.min() >= 0, "Negative item_idx"
    assert user_idx.max() < num_users, f"user_idx max {user_idx.max()} >= num_users {num_users}"
    assert item_idx.max() < num_items, f"item_idx max {item_idx.max()} >= num_items {num_items}"
    assert not np.any(np.isnan(weights)), "NaN in matrix weights"
    assert np.all(weights > 0), "Matrix weights must be > 0"

    matrix = csr_matrix(
        (weights, (user_idx, item_idx)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    nnz = matrix.nnz
    density = nnz / (num_users * num_items) if num_users * num_items > 0 else 0.0
    approx_bytes = nnz * (8 + 4 + 4)  # value + col indices + row pointers (rough)

    logger.info("CSR matrix shape: (%d, %d)", num_users, num_items)
    logger.info("Non-zero entries: %s", f"{nnz:,}")
    logger.info("Matrix density: %.6f%%", density * 100)
    logger.info("Approximate sparse memory: %.1f MB", approx_bytes / 1e6)

    return matrix
