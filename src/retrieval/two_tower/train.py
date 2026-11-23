"""Pure two-tower model training."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.data.mappings import IdMapping, apply_item_mapping, apply_user_mapping
from src.io.storage import is_gcs_uri, join_uri, open_binary
from src.retrieval.hybrid.index import ItemVectorIndex
from src.retrieval.hybrid.losses import in_batch_softmax_loss
from src.retrieval.two_tower.dataset import TwoTowerDataset, collate_two_tower_batch
from src.pipelines.gpu_env import resolve_torch_device
from src.retrieval.two_tower.model import TwoTowerConfig, TwoTowerModel

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
    model: TwoTowerModel,
    item_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(item_indices), batch_size):
            batch_items = item_indices[start : start + batch_size]
            item_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)
            embeddings = model.encode_items(item_tensor)
            all_embeddings.append(embeddings.cpu().numpy())
    return item_indices, np.concatenate(all_embeddings, axis=0)


def train_two_tower_model(
    train_df: pl.DataFrame,
    user_map: IdMapping,
    item_map: IdMapping,
    config: dict[str, Any],
) -> tuple[TwoTowerModel, ItemVectorIndex]:
    """Train pure two-tower model and build a FAISS item index."""
    tt_cfg = TwoTowerConfig(
        embedding_dim=config["two_tower"]["embedding_dim"],
        learning_rate=config["two_tower"]["learning_rate"],
        batch_size=config["two_tower"]["batch_size"],
        epochs=config["two_tower"]["epochs"],
        temperature=config["two_tower"]["temperature"],
        random_state=config["two_tower"]["random_state"],
    )

    torch.manual_seed(tt_cfg.random_state)
    np.random.seed(tt_cfg.random_state)

    user_indices, item_indices = build_training_pairs(train_df, user_map, item_map)
    dataset = TwoTowerDataset(user_indices, item_indices)
    loader = DataLoader(
        dataset,
        batch_size=tt_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_two_tower_batch,
    )

    device = resolve_torch_device()
    model = TwoTowerModel(
        num_users=user_map.size,
        num_items=item_map.size,
        config=tt_cfg,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tt_cfg.learning_rate)

    logger.info(
        "Training pure Two-Tower on %d positive pairs (device=%s, epochs=%d)",
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

    all_item_indices = item_map.forward["idx"].to_numpy().astype(np.int64)
    indexed_items, item_embeddings = _build_item_embeddings(model, all_item_indices, device)
    index = ItemVectorIndex(item_embeddings, indexed_items)
    return model, index


def _save_bytes(path: str | Path, data: bytes) -> None:
    path_str = str(path)
    if is_gcs_uri(path_str):
        with open_binary(path_str, "wb") as handle:
            handle.write(data)
        return
    path_obj = Path(path_str)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_bytes(data)


def save_two_tower_artifacts(
    model_dir: str | Path,
    model: TwoTowerModel,
    index: ItemVectorIndex,
) -> None:
    """Persist pure two-tower model and FAISS index."""
    model_path = join_uri(model_dir, "two_tower_model.pt")
    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "num_users": model.user_embedding.num_embeddings,
            "num_items": model.item_embedding.num_embeddings,
        },
        buffer,
    )
    _save_bytes(model_path, buffer.getvalue())
    index.save(join_uri(model_dir, "item_index"))
    logger.info("Saved Two-Tower artifacts to %s", model_dir)


def load_two_tower_artifacts(
    model_dir: str | Path,
    device: torch.device | None = None,
) -> tuple[TwoTowerModel, ItemVectorIndex]:
    """Load pure two-tower model and FAISS index."""
    if device is None:
        device = resolve_torch_device()

    model_path = join_uri(model_dir, "two_tower_model.pt")
    if is_gcs_uri(model_path):
        with open_binary(model_path, "rb") as handle:
            checkpoint = torch.load(handle, map_location=device)
    else:
        checkpoint = torch.load(model_path, map_location=device)

    config: TwoTowerConfig = checkpoint["config"]
    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        config=config,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    index = ItemVectorIndex.load(join_uri(model_dir, "item_index"))
    return model, index
