"""User tower for Hybrid retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.item_features import CATEGORICAL_ATTRIBUTES
from src.retrieval.hybrid.item_tower import ItemTower


class UserTower(nn.Module):
    """Encode customer ID and recent purchase history into a normalized embedding."""

    def __init__(
        self,
        num_users: int,
        item_tower: ItemTower,
        user_id_embedding_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.item_tower = item_tower
        self.user_id_embedding = nn.Embedding(num_users, user_id_embedding_dim)
        self.projection = nn.Linear(user_id_embedding_dim + output_dim, output_dim)

    def _encode_history(
        self,
        history_items: torch.Tensor,
        history_lengths: torch.Tensor,
        history_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, history_length = history_items.shape
        flat_items = history_items.reshape(-1)
        flat_features = {
            name: values.reshape(-1) for name, values in history_features.items()
        }
        flat_embeddings = self.item_tower(flat_items, flat_features)
        history_embeddings = flat_embeddings.view(batch_size, history_length, -1)

        mask = (
            torch.arange(history_length, device=history_items.device)
            .unsqueeze(0)
            < history_lengths.unsqueeze(1)
        ).unsqueeze(-1)
        masked = history_embeddings * mask
        summed = masked.sum(dim=1)
        lengths = history_lengths.clamp(min=1).unsqueeze(-1).to(summed.dtype)
        return summed / lengths

    def forward(
        self,
        user_indices: torch.Tensor,
        history_items: torch.Tensor,
        history_lengths: torch.Tensor,
        history_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        user_embedding = self.user_id_embedding(user_indices)
        history_embedding = self._encode_history(
            history_items, history_lengths, history_features
        )
        hidden = torch.cat([user_embedding, history_embedding], dim=-1)
        return F.normalize(self.projection(hidden), p=2, dim=-1)

    @staticmethod
    def history_feature_keys() -> list[str]:
        return [f"hist_{name}" for name in CATEGORICAL_ATTRIBUTES]
