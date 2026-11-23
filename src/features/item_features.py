"""Item-side features for Hybrid retrieval."""

from __future__ import annotations

import io
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from src.data.mappings import IdMapping
from src.io.storage import is_gcs_uri, open_binary

logger = logging.getLogger(__name__)

CATEGORICAL_ATTRIBUTES = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "department_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]


@dataclass
class CategoricalEncoders:
    """Train-fitted integer mappings for item categorical attributes."""

    vocabs: dict[str, dict[str, int]] = field(default_factory=dict)

    def encode(self, attribute: str, value: str) -> int:
        return self.vocabs[attribute].get(value, 0)

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {name: len(mapping) for name, mapping in self.vocabs.items()}

    def save(self, path: str | Path) -> None:
        path_str = str(path)
        if is_gcs_uri(path_str):
            buffer = io.BytesIO()
            pickle.dump(self.vocabs, buffer)
            with open_binary(path_str, "wb") as handle:
                handle.write(buffer.getvalue())
        else:
            path_obj = Path(path_str)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, "wb") as f:
                pickle.dump(self.vocabs, f)
        logger.info("Saved categorical encoders to %s", path_str)

    @classmethod
    def load(cls, path: str | Path) -> CategoricalEncoders:
        path_str = str(path)
        if is_gcs_uri(path_str):
            with open_binary(path_str, "rb") as handle:
                vocabs = pickle.load(handle)
        else:
            with open(path_str, "rb") as f:
                vocabs = pickle.load(f)
        return cls(vocabs=vocabs)


@dataclass
class ItemFeatureTable:
    """Dense item feature arrays indexed by item_idx."""

    item_indices: np.ndarray
    categorical: dict[str, np.ndarray]

    def row(self, item_idx: int) -> dict[str, int]:
        pos = int(np.searchsorted(self.item_indices, item_idx))
        if pos >= len(self.item_indices) or self.item_indices[pos] != item_idx:
            raise KeyError(f"Unknown item_idx: {item_idx}")
        return {name: int(values[pos]) for name, values in self.categorical.items()}

    def batch_rows(self, item_indices: np.ndarray) -> dict[str, np.ndarray]:
        lookup = {int(idx): i for i, idx in enumerate(self.item_indices)}
        positions = np.array([lookup[int(idx)] for idx in item_indices], dtype=np.int64)
        return {
            name: values[positions].astype(np.int64)
            for name, values in self.categorical.items()
        }


def fit_categorical_encoders(
    articles_df: pl.DataFrame,
    train_item_ids: set[str],
) -> CategoricalEncoders:
    """Fit integer mappings on training-known articles only."""
    train_articles = articles_df.filter(pl.col("article_id").is_in(list(train_item_ids)))
    vocabs: dict[str, dict[str, int]] = {}

    for attribute in CATEGORICAL_ATTRIBUTES:
        values = (
            train_articles.select(attribute)
            .unique()
            .sort(attribute)
            .get_column(attribute)
            .to_list()
        )
        mapping = {"<UNK>": 0}
        for i, value in enumerate(values, start=1):
            key = value if value is not None else "<UNK>"
            mapping[str(key)] = i
        vocabs[attribute] = mapping
        logger.info("Fitted %s vocabulary: %d categories", attribute, len(mapping))

    return CategoricalEncoders(vocabs=vocabs)


def build_item_feature_table(
    articles_df: pl.DataFrame,
    item_map: IdMapping,
    encoders: CategoricalEncoders,
) -> ItemFeatureTable:
    """Build dense categorical feature arrays for all training items."""
    known_items = item_map.forward.rename({"original_id": "article_id"})
    items = (
        articles_df.join(known_items, on="article_id", how="inner")
        .sort("idx")
    )

    item_indices = items["idx"].to_numpy().astype(np.int64)
    categorical: dict[str, np.ndarray] = {}

    for attribute in CATEGORICAL_ATTRIBUTES:
        encoded = [
            encoders.encode(attribute, str(value) if value is not None else "<UNK>")
            for value in items[attribute].to_list()
        ]
        categorical[attribute] = np.array(encoded, dtype=np.int64)

    logger.info("Built item feature table for %d training items", len(item_indices))
    return ItemFeatureTable(item_indices=item_indices, categorical=categorical)
