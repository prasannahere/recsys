"""Ranking features for Hybrid retrieval candidates."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from src.data.mappings import IdMapping
from src.features.item_features import CATEGORICAL_ATTRIBUTES, ItemFeatureTable
from src.ranking.common import (
    activity_features,
    item_popularity,
    sample_labeled_pairs,
    user_activity,
)
from src.retrieval.hybrid.model import HybridModel

FEATURE_COLS = ["hybrid_score", "log_popularity", "log_user_activity"]
RETRIEVER_NAME = "hybrid"


def _item_feature_lookup(item_features: ItemFeatureTable) -> dict[int, dict[str, int]]:
    return {
        int(item_idx): {
            name: int(item_features.categorical[name][pos])
            for name in CATEGORICAL_ATTRIBUTES
        }
        for pos, item_idx in enumerate(item_features.item_indices)
    }


def _encode_users_for_indices(
    model: HybridModel,
    user_indices: np.ndarray,
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    item_features: ItemFeatureTable,
    device: torch.device,
) -> np.ndarray:
    feature_lookup = _item_feature_lookup(item_features)
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


def _encode_items_for_indices(
    model: HybridModel,
    item_indices: np.ndarray,
    item_features: ItemFeatureTable,
    device: torch.device,
) -> np.ndarray:
    feature_lookup = _item_feature_lookup(item_features)
    categorical = {
        name: np.zeros(len(item_indices), dtype=np.int64)
        for name in CATEGORICAL_ATTRIBUTES
    }
    for row, item_idx in enumerate(item_indices):
        item_feats = feature_lookup.get(int(item_idx), {})
        for name in CATEGORICAL_ATTRIBUTES:
            categorical[name][row] = item_feats.get(name, 0)

    with torch.no_grad():
        item_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)
        cat_tensors = {
            name: torch.tensor(values, dtype=torch.long, device=device)
            for name, values in categorical.items()
        }
        embeddings = model.encode_items(item_tensor, cat_tensors)
    return embeddings.cpu().numpy()


def batch_hybrid_scores(
    model: HybridModel,
    user_indices: list[int],
    item_indices: list[int],
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    item_features: ItemFeatureTable,
    device: torch.device | None = None,
) -> list[float]:
    if not user_indices:
        return []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_array = np.array(user_indices, dtype=np.int64)
    item_array = np.array(item_indices, dtype=np.int64)
    user_emb = _encode_users_for_indices(
        model, user_array, history_items, history_lengths, item_features, device
    )
    item_emb = _encode_items_for_indices(model, item_array, item_features, device)
    scores = np.sum(user_emb * item_emb, axis=1)
    return [float(s) for s in scores]


def _score_rows(
    pairs: list[tuple[str, str, int, int, int]],
    model: HybridModel,
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    item_features: ItemFeatureTable,
    pop: dict[str, float],
    activity: dict[str, int],
    device: torch.device | None = None,
) -> list[dict]:
    if not pairs:
        return []
    user_indices = [p[3] for p in pairs]
    item_indices = [p[4] for p in pairs]
    scores = batch_hybrid_scores(
        model,
        user_indices,
        item_indices,
        history_items,
        history_lengths,
        item_features,
        device=device,
    )
    records: list[dict] = []
    for (customer_id, article_id, label, _user_idx, _item_idx), score in zip(
        pairs, scores, strict=True
    ):
        records.append(
            {
                "customer_id": customer_id,
                "article_id": article_id,
                "label": label,
                "hybrid_score": score,
                **activity_features(customer_id, article_id, pop, activity),
            }
        )
    return records


def build_training_pairs(
    train_df: pl.DataFrame,
    model: HybridModel,
    user_map: IdMapping,
    item_map: IdMapping,
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    item_features: ItemFeatureTable,
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
    pop = item_popularity(train_df)
    activity = user_activity(train_df)
    return pl.DataFrame(
        _score_rows(
            pairs,
            model,
            history_items,
            history_lengths,
            item_features,
            pop,
            activity,
            device=device,
        )
    )


def add_features_to_candidates(
    candidates: pl.DataFrame,
    model: HybridModel,
    user_map: IdMapping,
    item_map: IdMapping,
    train_df: pl.DataFrame,
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    item_features: ItemFeatureTable,
    device: torch.device | None = None,
) -> pl.DataFrame:
    pop = item_popularity(train_df)
    activity = user_activity(train_df)

    user_indices: list[int] = []
    item_indices: list[int] = []
    meta: list[tuple[str, str, float | None, int | None]] = []
    for row in candidates.iter_rows(named=True):
        customer_id = row["customer_id"]
        article_id = row["article_id"]
        user_idx = user_map.to_idx(customer_id)
        item_idx = item_map.to_idx(article_id)
        if user_idx is None or item_idx is None:
            continue
        user_indices.append(user_idx)
        item_indices.append(item_idx)
        meta.append(
            (
                customer_id,
                article_id,
                row.get("retrieval_score"),
                row.get("rank"),
            )
        )

    if not meta:
        return pl.DataFrame(
            schema={
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
                "retrieval_score": pl.Float64,
                "retrieval_rank": pl.Int64,
                "hybrid_score": pl.Float64,
                "log_popularity": pl.Float64,
                "log_user_activity": pl.Float64,
            }
        )

    scores = batch_hybrid_scores(
        model,
        user_indices,
        item_indices,
        history_items,
        history_lengths,
        item_features,
        device=device,
    )
    rows = []
    for (customer_id, article_id, retrieval_score, retrieval_rank), score in zip(
        meta, scores, strict=True
    ):
        rows.append(
            {
                "customer_id": customer_id,
                "article_id": article_id,
                "retrieval_score": retrieval_score,
                "retrieval_rank": retrieval_rank,
                "hybrid_score": score,
                **activity_features(customer_id, article_id, pop, activity),
            }
        )
    return pl.DataFrame(rows)
