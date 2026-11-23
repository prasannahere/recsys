"""Hybrid training dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.item_features import CATEGORICAL_ATTRIBUTES, ItemFeatureTable
from src.features.user_features import history_without_item


class HybridDataset(Dataset):
    """Positive (user, item) purchase pairs with recent purchase histories."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        history_items: np.ndarray,
        history_lengths: np.ndarray,
        item_features: ItemFeatureTable,
        history_length: int,
    ) -> None:
        self.user_indices = user_indices.astype(np.int64)
        self.item_indices = item_indices.astype(np.int64)
        self.history_items = history_items
        self.history_lengths = history_lengths
        self.item_features = item_features
        self.history_length = history_length

        self._feature_lookup = {
            int(item_idx): {
                name: int(item_features.categorical[name][pos])
                for name in CATEGORICAL_ATTRIBUTES
            }
            for pos, item_idx in enumerate(item_features.item_indices)
        }

    def __len__(self) -> int:
        return len(self.user_indices)

    def _history_features(self, history: np.ndarray) -> dict[str, np.ndarray]:
        features = {
            name: np.zeros(self.history_length, dtype=np.int64)
            for name in CATEGORICAL_ATTRIBUTES
        }
        for pos, item_idx in enumerate(history):
            item_idx = int(item_idx)
            if item_idx == 0:
                continue
            item_feats = self._feature_lookup.get(item_idx)
            if item_feats is None:
                continue
            for name in CATEGORICAL_ATTRIBUTES:
                features[name][pos] = item_feats[name]
        return features

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_idx = int(self.user_indices[index])
        item_idx = int(self.item_indices[index])

        history, _hist_len = history_without_item(
            self.history_items,
            self.history_lengths,
            user_idx,
            item_idx,
            self.history_length,
        )
        pos_features = self.item_features.row(item_idx)
        history_features = self._history_features(history)

        sample = {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "item_idx": torch.tensor(item_idx, dtype=torch.long),
            "history_items": torch.tensor(history, dtype=torch.long),
            "history_len": torch.tensor(int((history > 0).sum()), dtype=torch.long),
        }
        for name in CATEGORICAL_ATTRIBUTES:
            sample[f"pos_{name}"] = torch.tensor(pos_features[name], dtype=torch.long)
            sample[f"hist_{name}"] = torch.tensor(
                history_features[name], dtype=torch.long
            )
        return sample


def collate_hybrid_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of dataset samples into a batch tensor dict."""
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
