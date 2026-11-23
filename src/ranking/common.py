"""Shared utilities for learning-to-rank feature engineering."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.mappings import IdMapping


def item_popularity(train_df: pl.DataFrame) -> dict[str, float]:
    """Global purchase count per article."""
    pop = train_df.group_by("article_id").agg(pl.len().alias("count"))
    return dict(zip(pop["article_id"], pop["count"], strict=True))


def user_activity(train_df: pl.DataFrame) -> dict[str, int]:
    """Purchase count per customer in training."""
    act = train_df.group_by("customer_id").agg(pl.len().alias("count"))
    return dict(zip(act["customer_id"], act["count"], strict=True))


def activity_features(
    customer_id: str,
    article_id: str,
    pop: dict[str, float],
    activity: dict[str, int],
) -> dict[str, float]:
    return {
        "log_popularity": float(np.log1p(pop.get(article_id, 0))),
        "log_user_activity": float(np.log1p(activity.get(customer_id, 0))),
    }


def sample_labeled_pairs(
    train_df: pl.DataFrame,
    user_map: IdMapping,
    item_map: IdMapping,
    max_users: int | None = 50_000,
    negatives_per_positive: int = 5,
    random_state: int = 42,
) -> list[tuple[str, str, int, int, int]]:
    """Return (customer_id, article_id, label, user_idx, item_idx) rows."""
    rng = np.random.default_rng(random_state)
    all_item_ids = item_map.forward["original_id"].to_list()
    user_items = (
        train_df.group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("items"))
    )

    user_rows = user_items.to_dicts()
    if max_users is not None and len(user_rows) > max_users:
        rng.shuffle(user_rows)
        user_rows = user_rows[:max_users]

    records: list[tuple[str, str, int, int, int]] = []
    for row in user_rows:
        customer_id = row["customer_id"]
        positives = list(row["items"])
        user_idx = user_map.to_idx(customer_id)
        if user_idx is None:
            continue

        pos_set = set(positives)
        for article_id in positives:
            item_idx = item_map.to_idx(article_id)
            if item_idx is None:
                continue
            records.append((customer_id, article_id, 1, user_idx, item_idx))

            n_neg = min(negatives_per_positive, len(all_item_ids) - len(pos_set))
            for _ in range(n_neg):
                neg_id = all_item_ids[int(rng.integers(0, len(all_item_ids)))]
                if neg_id in pos_set:
                    continue
                neg_idx = item_map.to_idx(neg_id)
                if neg_idx is None:
                    continue
                records.append((customer_id, neg_id, 0, user_idx, neg_idx))

    return records
