"""Pure two-tower candidate retrieval."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import torch

from src.data.mappings import IdMapping
from src.retrieval.hybrid.index import ItemVectorIndex
from src.retrieval.two_tower.model import TwoTowerModel

logger = logging.getLogger(__name__)

RETRIEVER_NAME = "two_tower"

_CANDIDATE_SCHEMA = {
    "customer_id": pl.Utf8,
    "article_id": pl.Utf8,
    "retrieval_score": pl.Float64,
    "rank": pl.Int64,
    "retriever": pl.Utf8,
}


def build_train_pairs(train_df: pl.DataFrame) -> pl.DataFrame:
    """Unique (customer_id, article_id) pairs from training purchases."""
    return train_df.select(["customer_id", "article_id"]).unique()


def _empty_candidates() -> pl.DataFrame:
    return pl.DataFrame(schema=_CANDIDATE_SCHEMA)


def generate_candidates(
    model: TwoTowerModel,
    index: ItemVectorIndex,
    user_indices: list[int],
    user_map: IdMapping,
    item_map: IdMapping,
    train_df: pl.DataFrame,
    k: int,
    filter_seen: bool = True,
    device: torch.device | None = None,
    search_multiplier: int = 5,
    train_pairs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Retrieve Top-K candidates via FAISS inner-product search."""
    if not user_indices:
        return _empty_candidates()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_idx_array = np.array(user_indices, dtype=np.int64)
    with torch.no_grad():
        user_tensor = torch.tensor(user_idx_array, dtype=torch.long, device=device)
        user_embeddings = model.encode_users(user_tensor).cpu().numpy()

    search_k = k * search_multiplier if filter_seen else k
    retrieved_indices, retrieved_scores = index.search(user_embeddings, search_k)

    _n_users, width = retrieved_indices.shape
    user_rep = np.repeat(user_idx_array, width)
    item_flat = retrieved_indices.reshape(-1).astype(np.int64)
    score_flat = retrieved_scores.reshape(-1).astype(np.float64)
    valid = item_flat >= 0

    if not valid.any():
        return _empty_candidates()

    raw = pl.DataFrame(
        {
            "user_idx": user_rep[valid],
            "item_idx": item_flat[valid],
            "retrieval_score": score_flat[valid],
        }
    )
    candidates = (
        raw.join(
            user_map.forward.rename({"idx": "user_idx", "original_id": "customer_id"}),
            on="user_idx",
            how="inner",
        )
        .join(
            item_map.forward.rename({"idx": "item_idx", "original_id": "article_id"}),
            on="item_idx",
            how="inner",
        )
    )

    if filter_seen:
        pairs = train_pairs if train_pairs is not None else build_train_pairs(train_df)
        candidates = candidates.join(
            pairs.with_columns(pl.lit(True).alias("_seen")),
            on=["customer_id", "article_id"],
            how="anti",
        )

    candidates = (
        candidates.sort(["user_idx", "retrieval_score"], descending=[False, True])
        .with_columns(
            pl.col("retrieval_score")
            .rank(method="ordinal", descending=True)
            .over("user_idx")
            .alias("rank")
        )
        .filter(pl.col("rank") <= k)
        .with_columns(pl.lit(RETRIEVER_NAME).alias("retriever"))
        .select(list(_CANDIDATE_SCHEMA.keys()))
    )
    return candidates
