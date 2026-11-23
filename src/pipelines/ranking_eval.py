"""Shared helpers for retrieval + ranker evaluation."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from src.evaluation.ranking_metrics import MetricResults, format_metrics_table
from src.io.storage import join_uri
from src.pipelines.retrieval_eval import (
    EvalContext,
    als_candidates,
    hybrid_candidates,
    score_candidates,
    two_tower_candidates,
    validation_diagnostics,
)
from src.ranking.als import features as als_features
from src.ranking.hybrid import features as hybrid_features
from src.ranking.ranker import LambdaMartRanker
from src.ranking.train import save_ranker, train_ranker
from src.ranking.two_tower import features as two_tower_features
from src.retrieval.als import ALSModel, prepare_als_matrix
from src.retrieval.hybrid.train import load_hybrid_artifacts
from src.retrieval.two_tower.train import load_two_tower_artifacts

logger = logging.getLogger(__name__)

RETRIEVER_LABELS = {
    "als": ("ALS", "ALS Ranker"),
    "two_tower": ("Two-Tower", "Two-Tower Ranker"),
    "hybrid": ("Hybrid", "Hybrid Ranker"),
}


def ranker_path_for_config(config: dict[str, Any]) -> str:
    rank_cfg = config.get("ranking", {})
    if rank_cfg.get("model_path"):
        return rank_cfg["model_path"]
    return join_uri(config["artifacts"]["model_dir"], "ranker", "ranker.pkl")


def load_ranker(config: dict[str, Any]) -> LambdaMartRanker:
    path = ranker_path_for_config(config)
    logger.info("Loading ranker from %s", path)
    return LambdaMartRanker.load(path)


def get_ranker(
    retriever: str,
    config: dict[str, Any],
    *,
    load_ranker: bool,
    load_retrieval: bool,
) -> LambdaMartRanker:
    if load_ranker:
        return load_ranker(config)
    logger.info("Training %s LambdaMART ranker...", retriever)
    ranker = train_ranker(retriever, config, load_retrieval=load_retrieval)
    save_ranker(ranker, config)
    return ranker


def rerank_candidates(
    retriever: str,
    candidates: pl.DataFrame,
    ranker: LambdaMartRanker,
    ctx: EvalContext,
    config: dict[str, Any],
) -> pl.DataFrame:
    model_dir = config["artifacts"]["model_dir"]
    if retriever == "als":
        matrix = prepare_als_matrix(
            ctx.train_df,
            ctx.val_df,
            ctx.reference_date,
            ctx.user_map,
            ctx.item_map,
            config,
        )
        model = ALSModel.load(model_dir)
        model.attach_user_item_matrix(matrix)
        featured = als_features.add_features_to_candidates(
            candidates, model, ctx.user_map, ctx.item_map, ctx.train_df
        )
        return ranker.rerank_candidates(featured)

    if retriever == "two_tower":
        model, _index = load_two_tower_artifacts(model_dir)
        featured = two_tower_features.add_features_to_candidates(
            candidates, model, ctx.user_map, ctx.item_map, ctx.train_df
        )
        return ranker.rerank_candidates(featured)

    if retriever == "hybrid":
        model, _enc, item_features, history_items, history_lengths, _index = (
            load_hybrid_artifacts(model_dir)
        )
        featured = hybrid_features.add_features_to_candidates(
            candidates,
            model,
            ctx.user_map,
            ctx.item_map,
            ctx.train_df,
            history_items,
            history_lengths,
            item_features,
        )
        return ranker.rerank_candidates(featured)

    raise ValueError(f"Unknown retriever: {retriever}")


def evaluate_retrieval_and_ranker(
    retriever: str,
    ctx: EvalContext,
    config: dict[str, Any],
    *,
    load_retrieval: bool = False,
    load_ranker_flag: bool = False,
) -> dict[str, MetricResults]:
    if retriever == "als":
        candidates = als_candidates(ctx, config, load_saved=load_retrieval)
    elif retriever == "two_tower":
        candidates = two_tower_candidates(ctx, config, load_saved=load_retrieval)
    elif retriever == "hybrid":
        candidates = hybrid_candidates(ctx, config, load_saved=load_retrieval)
    else:
        raise ValueError(f"Unknown retriever: {retriever}")

    retrieval_label, ranker_label = RETRIEVER_LABELS[retriever]
    results = {retrieval_label: score_candidates(ctx, candidates)}

    ranker = get_ranker(
        retriever,
        config,
        load_ranker=load_ranker_flag,
        load_retrieval=True,
    )
    ranked = rerank_candidates(retriever, candidates, ranker, ctx, config)
    results[ranker_label] = score_candidates(ctx, ranked)
    return results


def print_retrieval_vs_ranker(
    retriever: str,
    results: dict[str, MetricResults],
    ctx: EvalContext,
) -> None:
    val_diagnostics = validation_diagnostics(ctx)
    retrieval_label, ranker_label = RETRIEVER_LABELS[retriever]
    print(f"\n=== {retrieval_label} Retrieval vs LambdaMART Ranker ===\n")
    print(format_metrics_table(results, ctx.ks))
    print("\n--- HitRate@K ---")
    for label in (retrieval_label, ranker_label):
        result = results[label]
        hit_12 = result.metrics.get("hit_rate@12", 0.0)
        hit_50 = result.metrics.get("hit_rate@50", 0.0)
        print(f"  {label:<18} HitRate@12={hit_12:.4f}  @50={hit_50:.4f}")
    print("\n--- Diagnostics ---")
    print(f"  Evaluable validation users: {val_diagnostics['pct_val_users_evaluable']:.1f}%")
    for label in (retrieval_label, ranker_label):
        coverage = results[label].diagnostics.get("candidate_coverage_pct", 0.0)
        print(f"  {label} candidate coverage (retrievable GT): {coverage:.1f}%")
