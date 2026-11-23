"""Hybrid candidate retrieval."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
import torch

from src.data.mappings import IdMapping
from src.features.item_features import CATEGORICAL_ATTRIBUTES, ItemFeatureTable
from src.retrieval.hybrid.index import ItemVectorIndex
from src.retrieval.hybrid.model import HybridModel

logger = logging.getLogger(__name__)

RETRIEVER_NAME = "hybrid"


def _encode_users_batch(
    model: HybridModel,
    user_indices: np.ndarray,
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    item_features: ItemFeatureTable,
    device: torch.device,
) -> np.ndarray:
    feature_lookup = {
        int(item_idx): {
            name: int(item_features.categorical[name][pos])
            for name in CATEGORICAL_ATTRIBUTES
        }
        for pos, item_idx in enumerate(item_features.item_indices)
    }
    history_length = history_items.shape[1]
    batch_size = len(user_indices)

    history_feature_arrays = {
        name: np.zeros((batch_size, history_length), dtype=np.int64)
        for name in CATEGORICAL_ATTRIBUTES
    }
    for row, user_idx in enumerate(user_indices):
        history = history_items[int(user_idx)]
        for pos, item_idx in enumerate(history):
            item_idx = int(item_idx)
            if item_idx == 0:
                continue
            item_feats = feature_lookup.get(item_idx)
            if item_feats is None:
                continue
            for name in CATEGORICAL_ATTRIBUTES:
                history_feature_arrays[name][row, pos] = item_feats[name]

    with torch.no_grad():
        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=device)
        history_tensor = torch.tensor(
            history_items[user_indices], dtype=torch.long, device=device
        )
        length_tensor = torch.tensor(
            history_lengths[user_indices], dtype=torch.long, device=device
        )
        history_features = {
            name: torch.tensor(values, dtype=torch.long, device=device)
            for name, values in history_feature_arrays.items()
        }
        embeddings = model.encode_users(
            user_tensor,
            history_tensor,
            length_tensor,
            history_features,
        )
    return embeddings.cpu().numpy()


def generate_candidates(
    model: HybridModel,
    index: ItemVectorIndex,
    user_indices: list[int],
    user_map: IdMapping,
    item_map: IdMapping,
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    item_features: ItemFeatureTable,
    train_df: pl.DataFrame,
    k: int,
    filter_seen: bool = True,
    device: torch.device | None = None,
    search_multiplier: int = 5,
) -> pl.DataFrame:
    """Retrieve Top-K candidates via FAISS inner-product search."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seen_lookup: dict[str, set[str]] = {}
    if filter_seen:
        for row in (
            train_df.group_by("customer_id")
            .agg(pl.col("article_id").alias("seen_items"))
            .iter_rows(named=True)
        ):
            seen_lookup[row["customer_id"]] = set(row["seen_items"])

    user_idx_array = np.array(user_indices, dtype=np.int64)
    user_embeddings = _encode_users_batch(
        model,
        user_idx_array,
        history_items,
        history_lengths,
        item_features,
        device,
    )

    search_k = k * search_multiplier if filter_seen else k
    retrieved_indices, retrieved_scores = index.search(user_embeddings, search_k)

    records: list[dict[str, Any]] = []
    for row_idx, user_idx in enumerate(user_indices):
        customer_id = user_map.to_original(user_idx)
        seen = seen_lookup.get(customer_id, set()) if filter_seen else set()
        rank = 0
        for item_idx, score in zip(
            retrieved_indices[row_idx], retrieved_scores[row_idx], strict=True
        ):
            item_idx = int(item_idx)
            if item_idx < 0:
                continue
            article_id = item_map.to_original(item_idx)
            if filter_seen and article_id in seen:
                continue
            rank += 1
            records.append(
                {
                    "customer_id": customer_id,
                    "article_id": article_id,
                    "retrieval_score": float(score),
                    "rank": rank,
                    "retriever": RETRIEVER_NAME,
                }
            )
            if rank >= k:
                break

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
