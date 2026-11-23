"""Item tower for Hybrid retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.item_features import CATEGORICAL_ATTRIBUTES


class ItemTower(nn.Module):
    """Encode article ID and categorical metadata into a normalized embedding."""

    def __init__(
        self,
        num_items: int,
        vocab_sizes: dict[str, int],
        item_id_embedding_dim: int,
        categorical_embedding_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.item_id_embedding = nn.Embedding(num_items, item_id_embedding_dim)
        self.categorical_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_sizes[name], categorical_embedding_dim)
                for name in CATEGORICAL_ATTRIBUTES
            }
        )
        input_dim = item_id_embedding_dim + len(CATEGORICAL_ATTRIBUTES) * categorical_embedding_dim
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        item_indices: torch.Tensor,
        categorical_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts = [self.item_id_embedding(item_indices)]
        for name in CATEGORICAL_ATTRIBUTES:
            parts.append(self.categorical_embeddings[name](categorical_features[name]))
        hidden = torch.cat(parts, dim=-1)
        return F.normalize(self.projection(hidden), p=2, dim=-1)
