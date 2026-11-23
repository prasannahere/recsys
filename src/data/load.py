"""Load H&M transaction data efficiently with Polars."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.io.storage import exists, is_gcs_uri, open_binary

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["t_dat", "customer_id", "article_id"]

ARTICLE_COLUMNS = [
    "article_id",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "department_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]


def load_transactions(path: str | Path) -> pl.DataFrame:
    """Load transactions CSV with only the columns needed for ALS.

    Args:
        path: Path to transactions_train.csv.

    Returns:
        Polars DataFrame with parsed date column, sorted by t_dat.
    """
    path_str = str(path)
    if not exists(path_str):
        raise FileNotFoundError(f"Transactions file not found: {path_str}")

    logger.info("Loading transactions from %s", path_str)

    if is_gcs_uri(path_str):
        with open_binary(path_str, "rb") as handle:
            source = pl.scan_csv(handle, dtypes={"article_id": pl.Utf8})
            df = (
                source.select(REQUIRED_COLUMNS)
                .with_columns(pl.col("t_dat").str.to_date("%Y-%m-%d"))
                .sort("t_dat")
                .collect()
            )
    else:
        df = (
            pl.scan_csv(path_str, dtypes={"article_id": pl.Utf8})
            .select(REQUIRED_COLUMNS)
            .with_columns(pl.col("t_dat").str.to_date("%Y-%m-%d"))
            .sort("t_dat")
            .collect()
        )

    n_rows = df.height
    n_users = df["customer_id"].n_unique()
    n_items = df["article_id"].n_unique()
    min_date = df["t_dat"].min()
    max_date = df["t_dat"].max()

    logger.info("Transactions loaded: %s rows", f"{n_rows:,}")
    logger.info("Unique users: %s", f"{n_users:,}")
    logger.info("Unique items: %s", f"{n_items:,}")
    logger.info("Date range: %s to %s", min_date, max_date)

    return df


def load_articles(path: str | Path) -> pl.DataFrame:
    """Load article metadata CSV with columns needed for Hybrid item features."""
    path_str = str(path)
    if not exists(path_str):
        raise FileNotFoundError(f"Articles file not found: {path_str}")

    logger.info("Loading articles from %s", path_str)
    if is_gcs_uri(path_str):
        with open_binary(path_str, "rb") as handle:
            source = pl.scan_csv(handle, dtypes={"article_id": pl.Utf8})
            df = source.select(ARTICLE_COLUMNS).collect()
    else:
        df = (
            pl.scan_csv(path_str, dtypes={"article_id": pl.Utf8})
            .select(ARTICLE_COLUMNS)
            .collect()
        )
    logger.info("Articles loaded: %s rows", f"{df.height:,}")
    return df
