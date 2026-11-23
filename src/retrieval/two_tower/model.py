"""Pure ID-only two-tower retrieval model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwoTowerConfig:
    embedding_dim: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 2048
    epochs: int = 3
    temperature: float = 0.05
    random_state: int = 42


class TwoTowerModel(nn.Module):
    """Dual-encoder retrieval with user-ID and item-ID embeddings only."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        config: TwoTowerConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.user_embedding = nn.Embedding(num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, config.embedding_dim)

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_embedding(user_indices), p=2, dim=-1)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_embedding(item_indices), p=2, dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_users(batch["user_idx"]), self.encode_items(batch["item_idx"])
