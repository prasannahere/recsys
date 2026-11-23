"""ALS training utilities."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
from scipy.sparse import csr_matrix

from src.data.mappings import IdMapping, add_indices_to_interactions
from src.features.interactions import build_csr_matrix, build_interactions
from src.retrieval.als.model import ALSConfig, ALSModel


def prepare_als_matrix(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    reference_date: date,
    user_map: IdMapping,
    item_map: IdMapping,
    config: dict[str, Any],
) -> csr_matrix:
    """Build ALS user x item CSR matrix from training interactions."""
    weight_cfg = config["interaction_weighting"]
    interactions = build_interactions(
        train_df,
        method=weight_cfg["method"],
        reference_date=reference_date,
        recency_lambda=weight_cfg.get("lambda", 0.01),
    )
    indexed = add_indices_to_interactions(interactions, user_map, item_map)
    matrix = build_csr_matrix(indexed, user_map.size, item_map.size)

    assert matrix.shape == (user_map.size, item_map.size)
    assert train_df["t_dat"].max() < val_df["t_dat"].min()

    val_in_train = train_df.filter(pl.col("t_dat") >= val_df["t_dat"].min())
    assert val_in_train.height == 0, "Validation rows leaked into training set"

    return matrix


def train_als_model(matrix: csr_matrix, config: dict[str, Any]) -> ALSModel:
    """Train ALS and attach the user-item matrix for recommend()."""
    als_cfg = config["als"]
    model = ALSModel(
        ALSConfig(
            factors=als_cfg["factors"],
            regularization=als_cfg["regularization"],
            iterations=als_cfg["iterations"],
            alpha=als_cfg["alpha"],
            random_state=als_cfg["random_state"],
        )
    )
    model.fit(matrix)
    return model
