"""User and item ID mappings for ALS (built from training data only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src.io.storage import is_gcs_uri, write_parquet

logger = logging.getLogger(__name__)


@dataclass
class IdMapping:
    """Bidirectional mapping between external IDs and compact integer indices."""

    forward: pl.DataFrame  # original_id, idx
    name: str  # "user" or "item"
    _forward_cache: dict[str, int] | None = field(default=None, repr=False, compare=False)
    _reverse_cache: dict[int, str] | None = field(default=None, repr=False, compare=False)

    @property
    def size(self) -> int:
        return self.forward.height

    def _ensure_caches(self) -> None:
        if self._forward_cache is not None:
            return
        self._forward_cache = dict(
            zip(self.forward["original_id"], self.forward["idx"], strict=True)
        )
        self._reverse_cache = {
            int(idx): original for original, idx in self._forward_cache.items()
        }

    def to_idx(self, original_id: str) -> int | None:
        self._ensure_caches()
        assert self._forward_cache is not None
        idx = self._forward_cache.get(original_id)
        return int(idx) if idx is not None else None

    def to_original(self, idx: int) -> str:
        self._ensure_caches()
        assert self._reverse_cache is not None
        return self._reverse_cache[int(idx)]

    def idx_series(self, original_ids: pl.Series) -> pl.Series:
        """Map a series of original IDs to indices; unknown -> null."""
        lookup = self.forward.rename({"original_id": "lookup_id"})
        mapped = (
            pl.DataFrame({"lookup_id": original_ids})
            .join(lookup, on="lookup_id", how="left")
            .get_column("idx")
        )
        return mapped

    def save(self, path: str | Path) -> None:
        path_str = str(path)
        write_parquet(path_str, self.forward)
        logger.info("Saved %s mapping (%d entries) to %s", self.name, self.size, path_str)

    @classmethod
    def load(cls, path: str | Path, name: str) -> IdMapping:
        path_str = str(path)
        if is_gcs_uri(path_str):
            from src.io.storage import open_binary

            with open_binary(path_str, "rb") as handle:
                forward = pl.read_parquet(handle)
        else:
            forward = pl.read_parquet(path_str)
        return cls(forward=forward, name=name)


def build_user_mapping(interactions: pl.DataFrame) -> IdMapping:
    """Build customer_id -> user_idx from aggregated interactions."""
    users = (
        interactions.select("user_id")
        .unique()
        .sort("user_id")
        .with_row_index("idx")
        .rename({"user_id": "original_id"})
    )
    logger.info("Built user mapping: %d users", users.height)
    return IdMapping(forward=users, name="user")


def build_item_mapping(interactions: pl.DataFrame) -> IdMapping:
    """Build article_id -> item_idx from aggregated interactions."""
    items = (
        interactions.select("item_id")
        .unique()
        .sort("item_id")
        .with_row_index("idx")
        .rename({"item_id": "original_id"})
    )
    logger.info("Built item mapping: %d items", items.height)
    return IdMapping(forward=items, name="item")


def build_mappings(
    interactions: pl.DataFrame,
) -> tuple[IdMapping, IdMapping]:
    """Build user and item mappings from training interactions."""
    user_map = build_user_mapping(interactions)
    item_map = build_item_mapping(interactions)
    return user_map, item_map


def apply_user_mapping(df: pl.DataFrame, user_map: IdMapping) -> pl.DataFrame:
    """Add user_idx column; rows with unknown users get null user_idx."""
    lookup = user_map.forward.rename({"original_id": "customer_id"})
    return df.join(lookup, on="customer_id", how="left").rename({"idx": "user_idx"})


def apply_item_mapping(df: pl.DataFrame, item_map: IdMapping) -> pl.DataFrame:
    """Add item_idx column; rows with unknown items get null item_idx."""
    lookup = item_map.forward.rename({"original_id": "article_id"})
    return df.join(lookup, on="article_id", how="left").rename({"idx": "item_idx"})


def add_indices_to_interactions(
    interactions: pl.DataFrame,
    user_map: IdMapping,
    item_map: IdMapping,
) -> pl.DataFrame:
    """Map user_id/item_id columns to integer indices."""
    user_lookup = user_map.forward.rename({"original_id": "user_id", "idx": "user_idx"})
    item_lookup = item_map.forward.rename({"original_id": "item_id", "idx": "item_idx"})
    return (
        interactions.join(user_lookup, on="user_id", how="inner")
        .join(item_lookup, on="item_id", how="inner")
    )


def save_mappings(
    user_map: IdMapping,
    item_map: IdMapping,
    mappings_dir: str | Path,
) -> None:
    """Persist user and item mappings to parquet files."""
    from src.io.storage import join_uri

    user_map.save(join_uri(mappings_dir, "user_mapping.parquet"))
    item_map.save(join_uri(mappings_dir, "item_mapping.parquet"))
