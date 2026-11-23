"""Backward-compatible re-exports for ALS ranking features."""

from src.ranking.als.features import (
    FEATURE_COLS,
    add_features_to_candidates,
    als_dot_score,
    build_training_pairs,
)
from src.ranking.common import item_popularity, user_activity

__all__ = [
    "FEATURE_COLS",
    "add_features_to_candidates",
    "als_dot_score",
    "build_training_pairs",
    "item_popularity",
    "user_activity",
]
