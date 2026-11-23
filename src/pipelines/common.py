"""Shared pipeline utilities."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.data.load import load_transactions
from src.data.mappings import IdMapping, build_mappings, save_mappings
from src.data.temporal_split import temporal_split
from src.retrieval.als.train import train_als_model as _train_als_model

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    import os

    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _mapping_interactions(train_df: pl.DataFrame) -> pl.DataFrame:
    """Minimal train-only user-item pairs for ID mapping construction."""
    return (
        train_df.rename({"customer_id": "user_id", "article_id": "item_id"})
        .select(["user_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1.0).alias("interaction_weight"))
    )


def prepare_training_data(
    config: dict[str, Any],
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    date,
    IdMapping,
    IdMapping,
    set[str],
    set[str],
]:
    """Retriever-agnostic training data preparation.

    Performs load, temporal split, and train-only ID mappings. Does not build
    a CSR matrix or retriever-specific features.

    Returns:
        train_df, val_df, reference_date, user_map, item_map,
        train_users, train_items
    """
    df = load_transactions(config["data"]["transactions_path"])
    train_df, val_df, reference_date = temporal_split(
        df, config["split"]["validation_days"]
    )

    user_map, item_map = build_mappings(_mapping_interactions(train_df))
    mappings_dir = config["artifacts"]["mappings_dir"]
    save_mappings(user_map, item_map, mappings_dir)

    train_users = set(train_df["customer_id"].unique().to_list())
    train_items = set(train_df["article_id"].unique().to_list())

    assert train_df["t_dat"].max() < val_df["t_dat"].min()

    val_in_train = train_df.filter(pl.col("t_dat") >= val_df["t_dat"].min())
    assert val_in_train.height == 0, "Validation rows leaked into training set"

    return (
        train_df,
        val_df,
        reference_date,
        user_map,
        item_map,
        train_users,
        train_items,
    )


def train_als_model(matrix: Any, config: dict[str, Any]) -> Any:
    """Train ALS and attach the user-item matrix for recommend()."""
    return _train_als_model(matrix, config)


def get_eval_user_indices(
    val_df: pl.DataFrame,
    user_map: IdMapping,
    train_users: set[str],
    max_users: int | None = None,
) -> list[int]:
    """Return user_idx for customers who purchased in the validation window.

    Only includes warm-start users (present in training). Cold-start validation
    customers are excluded because the model has no history for them.
    """
    val_users = set(val_df["customer_id"].unique().to_list())
    eval_users = sorted(val_users & train_users)
    if max_users is not None and max_users > 0:
        eval_users = eval_users[:max_users]
    idx_series = user_map.idx_series(pl.Series(eval_users))
    return [int(i) for i in idx_series.drop_nulls()]


def verify_filter_seen(
    candidates: pl.DataFrame,
    train_df: pl.DataFrame,
) -> None:
    """Assert no recommended item was already purchased in training."""
    if candidates.height == 0:
        return
    train_pairs = (
        train_df.select(["customer_id", "article_id"])
        .unique()
        .with_columns(pl.lit(True).alias("seen"))
    )
    merged = candidates.join(
        train_pairs, on=["customer_id", "article_id"], how="inner"
    )
    assert merged.height == 0, (
        f"filter_seen violation: {merged.height} recommended items were already seen"
    )


def verify_article_ids_in_mapping(
    candidates: pl.DataFrame,
    item_map: IdMapping,
) -> None:
    """Assert all recommended article IDs exist in the training item mapping."""
    known = set(item_map.forward["original_id"].to_list())
    rec_items = set(candidates["article_id"].unique().to_list())
    unknown = rec_items - known
    assert not unknown, f"Recommended unknown articles: {list(unknown)[:5]}"
