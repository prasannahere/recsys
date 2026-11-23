"""Temporal train/validation split for recommender evaluation."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

logger = logging.getLogger(__name__)


def temporal_split(
    df: pl.DataFrame,
    validation_days: int,
) -> tuple[pl.DataFrame, pl.DataFrame, date]:
    """Split transactions by a temporal cutoff.

    Training interactions have dates strictly before the validation window.
    Validation interactions fall in the last ``validation_days`` calendar days
    (inclusive of the dataset max date).

    Args:
        df: Transaction DataFrame with t_dat, customer_id, article_id.
        validation_days: Length of the validation window in days.

    Returns:
        (train_df, val_df, reference_date) where reference_date is the last
        training day (val_start - 1 day), used for recency weighting.
    """
    if validation_days < 1:
        raise ValueError("validation_days must be >= 1")

    max_date: date = df["t_dat"].max()  # type: ignore[assignment]
    val_start = max_date - timedelta(days=validation_days - 1)
    reference_date = val_start - timedelta(days=1)

    train_df = df.filter(pl.col("t_dat") < val_start)
    val_df = df.filter(pl.col("t_dat") >= val_start)

    # Sanity: no temporal leakage
    if train_df.height > 0 and val_df.height > 0:
        train_max = train_df["t_dat"].max()
        val_min = val_df["t_dat"].min()
        assert train_max < val_min, (
            f"Temporal leakage: train max ({train_max}) >= val min ({val_min})"
        )

    train_users = set(train_df["customer_id"].unique().to_list())
    train_items = set(train_df["article_id"].unique().to_list())
    val_users = set(val_df["customer_id"].unique().to_list())
    val_items = set(val_df["article_id"].unique().to_list())

    val_users_unseen = val_users - train_users
    val_items_unseen = val_items - train_items

    logger.info("Temporal split (validation_days=%d)", validation_days)
    logger.info("Train date range: %s to %s", train_df["t_dat"].min(), train_df["t_dat"].max())
    logger.info("Validation date range: %s to %s", val_df["t_dat"].min(), val_df["t_dat"].max())
    logger.info("Train transactions: %s", f"{train_df.height:,}")
    logger.info("Validation transactions: %s", f"{val_df.height:,}")
    logger.info("Train users: %s | items: %s", f"{len(train_users):,}", f"{len(train_items):,}")
    logger.info(
        "Validation users: %s | items: %s",
        f"{len(val_users):,}",
        f"{len(val_items):,}",
    )
    logger.info(
        "Validation users unseen in training: %s (%.2f%%)",
        f"{len(val_users_unseen):,}",
        100.0 * len(val_users_unseen) / max(len(val_users), 1),
    )
    logger.info(
        "Validation items unseen in training: %s (%.2f%%)",
        f"{len(val_items_unseen):,}",
        100.0 * len(val_items_unseen) / max(len(val_items), 1),
    )
    logger.info("Recency reference date (last training day): %s", reference_date)

    return train_df, val_df, reference_date
