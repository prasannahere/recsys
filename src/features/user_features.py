"""User-side features for Hybrid retrieval."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.data.mappings import IdMapping

logger = logging.getLogger(__name__)


def build_user_purchase_histories(
    train_df: pl.DataFrame,
    user_map: IdMapping,
    item_map: IdMapping,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded recent-purchase histories per training user.

    Histories are built from training transactions only, ordered by most recent
    purchase first. Each history excludes the target positive item when used as
    context during training.

    Returns:
        history_items: shape (num_users, history_length), padded with 0.
        history_lengths: shape (num_users,), count of valid history items.
    """
    num_users = user_map.size
    history_items = np.zeros((num_users, history_length), dtype=np.int64)
    history_lengths = np.zeros(num_users, dtype=np.int64)

    user_histories = (
        train_df.sort("t_dat", descending=True)
        .group_by("customer_id")
        .agg(pl.col("article_id").alias("article_ids"))
    )

    for row in user_histories.iter_rows(named=True):
        customer_id = row["customer_id"]
        user_idx = user_map.to_idx(customer_id)
        if user_idx is None:
            continue

        item_indices: list[int] = []
        seen: set[int] = set()
        for article_id in row["article_ids"]:
            item_idx = item_map.to_idx(article_id)
            if item_idx is None or item_idx in seen:
                continue
            seen.add(item_idx)
            item_indices.append(item_idx)
            if len(item_indices) >= history_length:
                break

        n = len(item_indices)
        if n == 0:
            continue
        history_items[user_idx, :n] = item_indices
        history_lengths[user_idx] = n

    logger.info(
        "Built user purchase histories (max_len=%d) for %d users",
        history_length,
        int((history_lengths > 0).sum()),
    )
    return history_items, history_lengths


def history_without_item(
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    user_idx: int,
    exclude_item_idx: int,
    history_length: int,
) -> tuple[np.ndarray, int]:
    """Return a padded copy of user history with one item removed (for training)."""
    length = int(history_lengths[user_idx])
    if length == 0:
        return np.zeros(history_length, dtype=np.int64), 0

    items = history_items[user_idx, :length]
    mask = items != exclude_item_idx
    filtered = items[mask]
    if filtered.size == 0:
        return np.zeros(history_length, dtype=np.int64), 0

    out = np.zeros(history_length, dtype=np.int64)
    n = min(filtered.size, history_length)
    out[:n] = filtered[:n]
    return out, n
