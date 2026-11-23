"""Popularity-based candidate retrieval baseline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import polars as pl

from src.data.mappings import IdMapping

if TYPE_CHECKING:
    from src.evaluation.progress import ProgressLogger

logger = logging.getLogger(__name__)

RankBy = Literal["purchase_count", "unique_users"]
RETRIEVER_NAME = "popularity"


def build_popularity_ranking(
    train_df: pl.DataFrame,
    item_map: IdMapping,
    rank_by: RankBy = "purchase_count",
) -> pl.DataFrame:
    """Rank training items by global popularity."""
    if rank_by == "purchase_count":
        pop = (
            train_df.group_by("article_id")
            .agg(pl.len().alias("pop_score"))
        )
    elif rank_by == "unique_users":
        pop = (
            train_df.group_by("article_id")
            .agg(pl.col("customer_id").n_unique().alias("pop_score"))
        )
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")

    known_items = item_map.forward.rename({"original_id": "article_id"})
    pop = (
        pop.join(known_items, on="article_id", how="inner")
        .sort("pop_score", descending=True)
        .with_row_index("pop_rank")
        .with_columns((pl.col("pop_rank") + 1).alias("pop_rank"))
    )
    logger.info(
        "Built popularity ranking (%s): %d items",
        rank_by,
        pop.height,
    )
    return pop


def generate_popularity_candidates(
    eval_user_ids: list[str],
    popularity: pl.DataFrame,
    train_df: pl.DataFrame,
    k: int,
    filter_seen: bool = True,
    progress: ProgressLogger | None = None,
) -> pl.DataFrame:
    """Return Top-K popular items for each evaluation user."""
    pop_list = (
        popularity.sort("pop_rank")
        .select(["article_id", "pop_score", "pop_rank"])
        .to_dicts()
    )

    seen_lookup: dict[str, set[str]] = {}
    if filter_seen:
        for row in (
            train_df.group_by("customer_id")
            .agg(pl.col("article_id").alias("seen_items"))
            .iter_rows(named=True)
        ):
            seen_lookup[row["customer_id"]] = set(row["seen_items"])

    records: list[dict] = []
    for customer_id in eval_user_ids:
        seen = seen_lookup.get(customer_id, set()) if filter_seen else set()
        rank = 0
        for item in pop_list:
            if filter_seen and item["article_id"] in seen:
                continue
            rank += 1
            records.append(
                {
                    "customer_id": customer_id,
                    "article_id": item["article_id"],
                    "retrieval_score": float(item["pop_score"]),
                    "rank": rank,
                    "retriever": RETRIEVER_NAME,
                }
            )
            if rank >= k:
                break
        if progress is not None:
            progress.update(1)

    if not records:
        return pl.DataFrame(
            schema={
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
                "retrieval_score": pl.Float64,
                "rank": pl.Int64,
                "retriever": pl.Utf8,
            }
        )
    return pl.DataFrame(records)
