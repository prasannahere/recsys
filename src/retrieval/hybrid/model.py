"""Hybrid retrieval model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.features.item_features import CATEGORICAL_ATTRIBUTES
from src.retrieval.hybrid.item_tower import ItemTower
from src.retrieval.hybrid.user_tower import UserTower


@dataclass
class HybridConfig:
    embedding_dim: int = 128
    user_id_embedding_dim: int = 64
    item_id_embedding_dim: int = 64
    categorical_embedding_dim: int = 16
    history_length: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 2048
    epochs: int = 3
    temperature: float = 0.05
    random_state: int = 42


class HybridModel(nn.Module):
    """Dual-encoder retrieval model with shared item tower semantics."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        vocab_sizes: dict[str, int],
        config: HybridConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.item_tower = ItemTower(
            num_items=num_items,
            vocab_sizes=vocab_sizes,
            item_id_embedding_dim=config.item_id_embedding_dim,
            categorical_embedding_dim=config.categorical_embedding_dim,
            output_dim=config.embedding_dim,
        )
        self.user_tower = UserTower(
            num_users=num_users,
            item_tower=self.item_tower,
            user_id_embedding_dim=config.user_id_embedding_dim,
            output_dim=config.embedding_dim,
        )

    def encode_users(
        self,
        user_indices: torch.Tensor,
        history_items: torch.Tensor,
        history_lengths: torch.Tensor,
        history_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.user_tower(user_indices, history_items, history_lengths, history_features)

    def encode_items(
        self,
        item_indices: torch.Tensor,
        categorical_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.item_tower(item_indices, categorical_features)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pos_features = {
            name: batch[f"pos_{name}"] for name in CATEGORICAL_ATTRIBUTES
        }
        history_features = {
            name: batch[f"hist_{name}"] for name in CATEGORICAL_ATTRIBUTES
        }
        user_embeddings = self.encode_users(
            batch["user_idx"],
            batch["history_items"],
            batch["history_len"],
            history_features,
        )
        item_embeddings = self.encode_items(batch["item_idx"], pos_features)
        return user_embeddings, item_embeddings
