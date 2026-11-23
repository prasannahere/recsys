"""Training dataset for pure two-tower retrieval."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    """Positive (user, item) purchase pairs."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
    ) -> None:
        self.user_indices = user_indices.astype(np.int64)
        self.item_indices = item_indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "user_idx": torch.tensor(int(self.user_indices[index]), dtype=torch.long),
            "item_idx": torch.tensor(int(self.item_indices[index]), dtype=torch.long),
        }


def collate_two_tower_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
