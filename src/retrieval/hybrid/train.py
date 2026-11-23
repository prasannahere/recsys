"""Hybrid model training."""

from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.data.load import load_articles
from src.data.mappings import IdMapping, apply_item_mapping, apply_user_mapping
from src.features.item_features import (
    CategoricalEncoders,
    ItemFeatureTable,
    build_item_feature_table,
    fit_categorical_encoders,
)
from src.features.user_features import build_user_purchase_histories
from src.io.storage import is_gcs_uri, join_uri, open_binary
from src.retrieval.hybrid.dataset import HybridDataset, collate_hybrid_batch
from src.retrieval.hybrid.index import ItemVectorIndex
from src.retrieval.hybrid.losses import in_batch_softmax_loss
from src.pipelines.gpu_env import resolve_torch_device
from src.retrieval.hybrid.model import HybridConfig, HybridModel

logger = logging.getLogger(__name__)


def build_training_pairs(
    train_df: pl.DataFrame,
    user_map: IdMapping,
    item_map: IdMapping,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique positive purchase pairs from training transactions only."""
    indexed = (
        apply_item_mapping(
            apply_user_mapping(train_df, user_map),
            item_map,
        )
        .filter(pl.col("user_idx").is_not_null() & pl.col("item_idx").is_not_null())
        .select(
            pl.col("user_idx").cast(pl.Int64),
            pl.col("item_idx").cast(pl.Int64),
        )
        .unique()
    )
    return (
        indexed["user_idx"].to_numpy(),
        indexed["item_idx"].to_numpy(),
    )


def _build_item_embeddings(
    model: HybridModel,
    item_features: ItemFeatureTable,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    item_indices = item_features.item_indices
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(item_indices), batch_size):
            batch_items = item_indices[start : start + batch_size]
            categorical = item_features.batch_rows(batch_items)
            tensors = {
                name: torch.tensor(values, dtype=torch.long, device=device)
                for name, values in categorical.items()
            }
            item_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)
            embeddings = model.encode_items(item_tensor, tensors)
            all_embeddings.append(embeddings.cpu().numpy())

    return item_indices, np.concatenate(all_embeddings, axis=0)


def train_hybrid_model(
    train_df: pl.DataFrame,
    user_map: IdMapping,
    item_map: IdMapping,
    config: dict[str, Any],
) -> tuple[HybridModel, CategoricalEncoders, ItemFeatureTable, np.ndarray, np.ndarray, ItemVectorIndex]:
    """Train Hybrid model and build a FAISS item index."""
    tt_cfg = HybridConfig(
        embedding_dim=config["hybrid"]["embedding_dim"],
        user_id_embedding_dim=config["hybrid"]["user_id_embedding_dim"],
        item_id_embedding_dim=config["hybrid"]["item_id_embedding_dim"],
        categorical_embedding_dim=config["hybrid"]["categorical_embedding_dim"],
        history_length=config["hybrid"]["history_length"],
        learning_rate=config["hybrid"]["learning_rate"],
        batch_size=config["hybrid"]["batch_size"],
        epochs=config["hybrid"]["epochs"],
        temperature=config["hybrid"]["temperature"],
        random_state=config["hybrid"]["random_state"],
    )

    torch.manual_seed(tt_cfg.random_state)
    np.random.seed(tt_cfg.random_state)

    articles_df = load_articles(config["data"]["articles_path"])
    train_items = set(train_df["article_id"].unique().to_list())
    encoders = fit_categorical_encoders(articles_df, train_items)
    item_features = build_item_feature_table(articles_df, item_map, encoders)

    history_items, history_lengths = build_user_purchase_histories(
        train_df,
        user_map,
        item_map,
        history_length=tt_cfg.history_length,
    )
    user_indices, item_indices = build_training_pairs(train_df, user_map, item_map)

    dataset = HybridDataset(
        user_indices=user_indices,
        item_indices=item_indices,
        history_items=history_items,
        history_lengths=history_lengths,
        item_features=item_features,
        history_length=tt_cfg.history_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=tt_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_hybrid_batch,
    )

    device = resolve_torch_device()
    model = HybridModel(
        num_users=user_map.size,
        num_items=item_map.size,
        vocab_sizes=encoders.vocab_sizes,
        config=tt_cfg,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tt_cfg.learning_rate)

    logger.info(
        "Training Hybrid on %d positive pairs (device=%s, epochs=%d)",
        len(dataset),
        device,
        tt_cfg.epochs,
    )
    model.train()
    for epoch in range(tt_cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            user_embeddings, item_embeddings = model(batch)
            loss = in_batch_softmax_loss(
                user_embeddings,
                item_embeddings,
                temperature=tt_cfg.temperature,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        logger.info(
            "Epoch %d/%d - loss=%.4f",
            epoch + 1,
            tt_cfg.epochs,
            epoch_loss / max(n_batches, 1),
        )

    indexed_items, item_embeddings = _build_item_embeddings(
        model, item_features, device
    )
    index = ItemVectorIndex(item_embeddings, indexed_items)
    return model, encoders, item_features, history_items, history_lengths, index


def _save_bytes(path: str | Path, data: bytes) -> None:
    path_str = str(path)
    if is_gcs_uri(path_str):
        with open_binary(path_str, "wb") as handle:
            handle.write(data)
        return
    path_obj = Path(path_str)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_bytes(data)


def _save_numpy(path: str | Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array)
    _save_bytes(path, buffer.getvalue())


def save_hybrid_artifacts(
    model_dir: str | Path,
    model: HybridModel,
    encoders: CategoricalEncoders,
    item_features: ItemFeatureTable,
    history_items: np.ndarray,
    history_lengths: np.ndarray,
    index: ItemVectorIndex,
) -> None:
    """Persist Hybrid model, encoders, histories, and FAISS index."""
    model_path = join_uri(model_dir, "hybrid_model.pt")
    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "vocab_sizes": encoders.vocab_sizes,
            "num_users": model.user_tower.user_id_embedding.num_embeddings,
            "num_items": model.item_tower.item_id_embedding.num_embeddings,
        },
        buffer,
    )
    _save_bytes(model_path, buffer.getvalue())

    encoders.save(join_uri(model_dir, "categorical_encoders.pkl"))
    features_buffer = io.BytesIO()
    pickle.dump(item_features, features_buffer)
    _save_bytes(join_uri(model_dir, "item_features.pkl"), features_buffer.getvalue())
    _save_numpy(join_uri(model_dir, "history_items.npy"), history_items)
    _save_numpy(join_uri(model_dir, "history_lengths.npy"), history_lengths)
    index.save(join_uri(model_dir, "item_index"))
    logger.info("Saved Hybrid artifacts to %s", model_dir)


def load_hybrid_artifacts(
    model_dir: str | Path,
    device: torch.device | None = None,
) -> tuple[
    HybridModel,
    CategoricalEncoders,
    ItemFeatureTable,
    np.ndarray,
    np.ndarray,
    ItemVectorIndex,
]:
    """Load Hybrid model and retrieval artifacts."""
    if device is None:
        device = resolve_torch_device()

    model_path = join_uri(model_dir, "hybrid_model.pt")
    if is_gcs_uri(model_path):
        with open_binary(model_path, "rb") as handle:
            checkpoint = torch.load(handle, map_location=device)
    else:
        checkpoint = torch.load(model_path, map_location=device)

    config: HybridConfig = checkpoint["config"]
    model = HybridModel(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        vocab_sizes=checkpoint["vocab_sizes"],
        config=config,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    encoders = CategoricalEncoders.load(join_uri(model_dir, "categorical_encoders.pkl"))
    features_path = join_uri(model_dir, "item_features.pkl")
    if is_gcs_uri(features_path):
        with open_binary(features_path, "rb") as handle:
            item_features = pickle.load(handle)
    else:
        with open(features_path, "rb") as f:
            item_features = pickle.load(f)

    history_items_path = join_uri(model_dir, "history_items.npy")
    history_lengths_path = join_uri(model_dir, "history_lengths.npy")
    if is_gcs_uri(history_items_path):
        with open_binary(history_items_path, "rb") as handle:
            history_items = np.load(handle)
        with open_binary(history_lengths_path, "rb") as handle:
            history_lengths = np.load(handle)
    else:
        history_items = np.load(history_items_path)
        history_lengths = np.load(history_lengths_path)

    index = ItemVectorIndex.load(join_uri(model_dir, "item_index"))
    return model, encoders, item_features, history_items, history_lengths, index
