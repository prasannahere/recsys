"""Train retriever-specific LambdaMART rankers."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import polars as pl
import yaml
from scipy.sparse import csr_matrix

from src.data.mappings import IdMapping
from src.io.storage import artifact_exists, join_uri, write_text
from src.pipelines.common import prepare_training_data
from src.ranking.als import features as als_features
from src.ranking.hybrid import features as hybrid_features
from src.ranking.ranker import LambdaMartRanker
from src.ranking.two_tower import features as two_tower_features
from src.retrieval.als import ALSModel, prepare_als_matrix, train_als_model
from src.retrieval.hybrid.train import load_hybrid_artifacts, train_hybrid_model
from src.retrieval.two_tower.model import TwoTowerModel
from src.retrieval.two_tower.train import load_two_tower_artifacts, train_two_tower_model

logger = logging.getLogger(__name__)

RETRIEVERS = ("als", "two_tower", "hybrid")


def _ranking_cfg(config: dict[str, Any]) -> dict[str, Any]:
    if "ranking" not in config:
        config["ranking"] = {}
    return config["ranking"]


def _default_ranker_path(config: dict[str, Any], _retriever: str) -> str:
    return join_uri(config["artifacts"]["model_dir"], "ranker", "ranker.pkl")


def train_als_ranker(
    config: dict[str, Any],
    *,
    load_retrieval: bool = False,
    train_df: pl.DataFrame | None = None,
    val_df: pl.DataFrame | None = None,
    reference_date: date | None = None,
    user_map: IdMapping | None = None,
    item_map: IdMapping | None = None,
    matrix: csr_matrix | None = None,
    als_model: ALSModel | None = None,
) -> LambdaMartRanker:
    if train_df is None or val_df is None or reference_date is None:
        train_df, val_df, reference_date, user_map, item_map, _train_users, _train_items = (
            prepare_training_data(config)
        )
    else:
        assert user_map is not None and item_map is not None

    if matrix is None:
        matrix = prepare_als_matrix(
            train_df, val_df, reference_date, user_map, item_map, config
        )

    model_dir = config["artifacts"]["model_dir"]
    if als_model is not None:
        model = als_model
        model.attach_user_item_matrix(matrix)
    elif load_retrieval and artifact_exists(model_dir, "als_model.pkl"):
        logger.info("Loading saved ALS retrieval model from %s", model_dir)
        model = ALSModel.load(model_dir)
        model.attach_user_item_matrix(matrix)
    else:
        logger.info("Training ALS retrieval model for ranker features...")
        model = train_als_model(matrix, config)

    rank_cfg = _ranking_cfg(config)
    training_pairs = als_features.build_training_pairs(
        train_df,
        model,
        user_map,
        item_map,
        max_users=rank_cfg.get("max_train_users"),
        negatives_per_positive=rank_cfg.get("negatives_per_positive", 5),
        random_state=rank_cfg.get("random_state", 42),
    )
    ranker = LambdaMartRanker(
        feature_cols=als_features.FEATURE_COLS,
        params=rank_cfg.get("lgbm"),
        random_state=rank_cfg.get("random_state", 42),
        retriever=als_features.RETRIEVER_NAME,
    )
    ranker.fit(training_pairs)
    return ranker


def train_two_tower_ranker(
    config: dict[str, Any],
    *,
    load_retrieval: bool = False,
    train_df: pl.DataFrame | None = None,
    user_map: IdMapping | None = None,
    item_map: IdMapping | None = None,
    two_tower_model: TwoTowerModel | None = None,
) -> LambdaMartRanker:
    if train_df is None:
        train_df, _val_df, _reference_date, user_map, item_map, _train_users, _train_items = (
            prepare_training_data(config)
        )
    else:
        assert user_map is not None and item_map is not None

    model_dir = config["artifacts"]["model_dir"]
    if two_tower_model is not None:
        model = two_tower_model
    elif load_retrieval and artifact_exists(model_dir, "two_tower_model.pt"):
        logger.info("Loading saved Two-Tower retrieval model from %s", model_dir)
        model, _index = load_two_tower_artifacts(model_dir)
    else:
        logger.info("Training Two-Tower retrieval model for ranker features...")
        model, _index = train_two_tower_model(train_df, user_map, item_map, config)

    rank_cfg = _ranking_cfg(config)
    training_pairs = two_tower_features.build_training_pairs(
        train_df,
        model,
        user_map,
        item_map,
        max_users=rank_cfg.get("max_train_users"),
        negatives_per_positive=rank_cfg.get("negatives_per_positive", 5),
        random_state=rank_cfg.get("random_state", 42),
    )
    ranker = LambdaMartRanker(
        feature_cols=two_tower_features.FEATURE_COLS,
        params=rank_cfg.get("lgbm"),
        random_state=rank_cfg.get("random_state", 42),
        retriever=two_tower_features.RETRIEVER_NAME,
    )
    ranker.fit(training_pairs)
    return ranker


def train_hybrid_ranker(
    config: dict[str, Any],
    *,
    load_retrieval: bool = False,
) -> LambdaMartRanker:
    train_df, _val_df, _reference_date, user_map, item_map, _train_users, _train_items = (
        prepare_training_data(config)
    )
    model_dir = config["artifacts"]["model_dir"]
    if load_retrieval and artifact_exists(model_dir, "hybrid_model.pt"):
        logger.info("Loading saved Hybrid retrieval model from %s", model_dir)
        model, _enc, item_features, history_items, history_lengths, _index = (
            load_hybrid_artifacts(model_dir)
        )
    else:
        logger.info("Training Hybrid retrieval model for ranker features...")
        model, _enc, item_features, history_items, history_lengths, _index = (
            train_hybrid_model(train_df, user_map, item_map, config)
        )

    rank_cfg = _ranking_cfg(config)
    training_pairs = hybrid_features.build_training_pairs(
        train_df,
        model,
        user_map,
        item_map,
        history_items,
        history_lengths,
        item_features,
        max_users=rank_cfg.get("max_train_users"),
        negatives_per_positive=rank_cfg.get("negatives_per_positive", 5),
        random_state=rank_cfg.get("random_state", 42),
    )
    ranker = LambdaMartRanker(
        feature_cols=hybrid_features.FEATURE_COLS,
        params=rank_cfg.get("lgbm"),
        random_state=rank_cfg.get("random_state", 42),
        retriever=hybrid_features.RETRIEVER_NAME,
    )
    ranker.fit(training_pairs)
    return ranker


def train_ranker(
    retriever: str,
    config: dict[str, Any],
    *,
    load_retrieval: bool = False,
) -> LambdaMartRanker:
    if retriever == "als":
        return train_als_ranker(config, load_retrieval=load_retrieval)
    if retriever == "two_tower":
        return train_two_tower_ranker(config, load_retrieval=load_retrieval)
    if retriever == "hybrid":
        return train_hybrid_ranker(config, load_retrieval=load_retrieval)
    raise ValueError(f"Unknown retriever: {retriever}. Choose from {RETRIEVERS}")


def save_ranker(ranker: LambdaMartRanker, config: dict[str, Any]) -> str:
    rank_cfg = _ranking_cfg(config)
    ranker_path = rank_cfg.get("model_path") or _default_ranker_path(config, ranker.retriever)
    ranker.save(ranker_path)
    rank_cfg["model_path"] = ranker_path
    model_dir = config["artifacts"]["model_dir"]
    write_text(join_uri(model_dir, "ranker_config.yaml"), yaml.dump(rank_cfg, sort_keys=False))
    return ranker_path
