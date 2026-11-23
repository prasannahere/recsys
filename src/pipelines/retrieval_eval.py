"""Shared helpers for retrieval evaluation pipelines."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.data.mappings import IdMapping
from src.evaluation.ranking_metrics import (
    MetricResults,
    build_ground_truth,
    compute_validation_diagnostics,
    evaluate_candidates,
    format_metrics_table,
)
from src.io.storage import artifact_exists, join_uri, write_text
from src.pipelines.common import get_eval_user_indices
from src.retrieval.als import ALSModel, prepare_als_matrix, train_als_model
from src.retrieval.als.retrieve import generate_candidates as generate_als_candidates
from src.retrieval.hybrid.retrieve import generate_candidates as generate_hybrid_candidates
from src.retrieval.hybrid.train import (
    load_hybrid_artifacts,
    save_hybrid_artifacts,
    train_hybrid_model,
)
from src.retrieval.popularity import build_popularity_ranking, generate_popularity_candidates
from src.retrieval.two_tower.retrieve import generate_candidates as generate_two_tower_candidates
from src.retrieval.two_tower.train import (
    load_two_tower_artifacts,
    save_two_tower_artifacts,
    train_two_tower_model,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    train_df: pl.DataFrame
    val_df: pl.DataFrame
    reference_date: Any
    user_map: IdMapping
    item_map: IdMapping
    train_users: set[str]
    train_items: set[str]
    eval_user_indices: list[int]
    eval_user_ids: list[str]
    candidate_k: int
    filter_seen: bool
    ks: list[int]


def build_eval_context(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    reference_date: Any,
    user_map: IdMapping,
    item_map: IdMapping,
    train_users: set[str],
    train_items: set[str],
    config: dict[str, Any],
) -> EvalContext:
    retrieval_cfg = config["retrieval"]
    eval_user_indices = get_eval_user_indices(val_df, user_map, train_users)
    return EvalContext(
        train_df=train_df,
        val_df=val_df,
        reference_date=reference_date,
        user_map=user_map,
        item_map=item_map,
        train_users=train_users,
        train_items=train_items,
        eval_user_indices=eval_user_indices,
        eval_user_ids=[user_map.to_original(i) for i in eval_user_indices],
        candidate_k=retrieval_cfg["candidate_k"],
        filter_seen=retrieval_cfg["filter_seen"],
        ks=config["evaluation"]["ks"],
    )


def popularity_candidates(ctx: EvalContext, config: dict[str, Any]) -> pl.DataFrame:
    pop_ranking = build_popularity_ranking(
        ctx.train_df,
        ctx.item_map,
        rank_by=config.get("popularity", {}).get("rank_by", "purchase_count"),
    )
    return generate_popularity_candidates(
        ctx.eval_user_ids,
        pop_ranking,
        ctx.train_df,
        k=ctx.candidate_k,
        filter_seen=ctx.filter_seen,
    )


def als_candidates(
    ctx: EvalContext,
    als_config: dict[str, Any],
    *,
    load_saved: bool = False,
) -> pl.DataFrame:
    matrix = prepare_als_matrix(
        ctx.train_df,
        ctx.val_df,
        ctx.reference_date,
        ctx.user_map,
        ctx.item_map,
        als_config,
    )
    model_dir = als_config["artifacts"]["model_dir"]
    if load_saved and artifact_exists(model_dir, "als_model.pkl"):
        logger.info("Loading saved ALS model from %s", model_dir)
        model = ALSModel.load(model_dir)
        model.attach_user_item_matrix(matrix)
    else:
        logger.info("Training ALS model...")
        model = train_als_model(matrix, als_config)
        model.save(model_dir)

    logger.info("Generating ALS Top-%d candidates...", ctx.candidate_k)
    t0 = time.time()
    candidates = generate_als_candidates(
        model,
        ctx.eval_user_indices,
        ctx.user_map,
        ctx.item_map,
        k=ctx.candidate_k,
        filter_seen=ctx.filter_seen,
    )
    logger.info("ALS candidates: %d rows in %.1f s", candidates.height, time.time() - t0)
    return candidates


def two_tower_candidates(
    ctx: EvalContext,
    tt_config: dict[str, Any],
    *,
    load_saved: bool = False,
) -> pl.DataFrame:
    model_dir = tt_config["artifacts"]["model_dir"]
    if load_saved and artifact_exists(model_dir, "two_tower_model.pt"):
        logger.info("Loading saved Two-Tower model from %s", model_dir)
        model, index = load_two_tower_artifacts(model_dir)
    else:
        logger.info("Training pure Two-Tower model...")
        model, index = train_two_tower_model(
            ctx.train_df, ctx.user_map, ctx.item_map, tt_config
        )
        save_two_tower_artifacts(model_dir, model, index)
        write_text(join_uri(model_dir, "config.yaml"), yaml.dump(tt_config, sort_keys=False))

    logger.info("Generating Two-Tower Top-%d candidates...", ctx.candidate_k)
    t0 = time.time()
    candidates = generate_two_tower_candidates(
        model,
        index,
        ctx.eval_user_indices,
        ctx.user_map,
        ctx.item_map,
        ctx.train_df,
        k=ctx.candidate_k,
        filter_seen=ctx.filter_seen,
    )
    logger.info(
        "Two-Tower candidates: %d rows in %.1f s",
        candidates.height,
        time.time() - t0,
    )
    return candidates


def hybrid_candidates(
    ctx: EvalContext,
    hybrid_config: dict[str, Any],
    *,
    load_saved: bool = False,
) -> pl.DataFrame:
    model_dir = hybrid_config["artifacts"]["model_dir"]
    if load_saved and artifact_exists(model_dir, "hybrid_model.pt"):
        logger.info("Loading saved Hybrid model from %s", model_dir)
        model, _enc, item_features, history_items, history_lengths, index = (
            load_hybrid_artifacts(model_dir)
        )
    else:
        logger.info("Training Hybrid model...")
        (
            model,
            encoders,
            item_features,
            history_items,
            history_lengths,
            index,
        ) = train_hybrid_model(ctx.train_df, ctx.user_map, ctx.item_map, hybrid_config)
        save_hybrid_artifacts(
            model_dir,
            model,
            encoders,
            item_features,
            history_items,
            history_lengths,
            index,
        )
        write_text(join_uri(model_dir, "config.yaml"), yaml.dump(hybrid_config, sort_keys=False))

    logger.info("Generating Hybrid Top-%d candidates...", ctx.candidate_k)
    t0 = time.time()
    candidates = generate_hybrid_candidates(
        model,
        index,
        ctx.eval_user_indices,
        ctx.user_map,
        ctx.item_map,
        history_items,
        history_lengths,
        item_features,
        ctx.train_df,
        k=ctx.candidate_k,
        filter_seen=ctx.filter_seen,
    )
    logger.info("Hybrid candidates: %d rows in %.1f s", candidates.height, time.time() - t0)
    return candidates


def score_candidates(
    ctx: EvalContext,
    candidates: pl.DataFrame,
    all_gt: dict[str, set[str]] | None = None,
    retrievable_gt: dict[str, set[str]] | None = None,
) -> MetricResults:
    if all_gt is None or retrievable_gt is None:
        all_gt, retrievable_gt = build_ground_truth(
            ctx.val_df, ctx.train_users, ctx.train_items
        )
    return evaluate_candidates(
        candidates,
        all_gt,
        retrievable_gt,
        ctx.ks,
        score_col="retrieval_score",
    )


def print_single_results(
    model_name: str,
    results: MetricResults,
    val_diagnostics: dict[str, float],
    ks: list[int],
) -> None:
    print(f"\n=== {model_name} Retrieval Evaluation ===\n")
    table = format_metrics_table({model_name: results}, ks)
    print(table)
    print("\n--- HitRate@K ---")
    for k in ks:
        hit = results.metrics.get(f"hit_rate@{k}", 0.0)
        print(f"  HitRate@{k:<3} {hit:.4f}")
    print("\n--- Diagnostics ---")
    print(f"  Evaluable validation users: {val_diagnostics['pct_val_users_evaluable']:.1f}%")
    print(f"  Cold-start validation users: {val_diagnostics['pct_val_users_cold_start']:.1f}%")
    print(
        f"  Validation interactions with unseen items: "
        f"{val_diagnostics['pct_val_interactions_unseen_items']:.1f}%"
    )
    print(
        f"  Candidate coverage (retrievable GT): "
        f"{results.diagnostics.get('candidate_coverage_pct', 0):.1f}%"
    )


def print_comparison_results(
    results: dict[str, MetricResults],
    val_diagnostics: dict[str, float],
    ks: list[int],
) -> None:
    print("\n=== Candidate Retrieval Comparison ===\n")
    print(format_metrics_table(results, ks))
    print("\n--- HitRate@K ---")
    for model_name, result in results.items():
        hit_12 = result.metrics.get("hit_rate@12", 0.0)
        hit_50 = result.metrics.get("hit_rate@50", 0.0)
        hit_100 = result.metrics.get("hit_rate@100", 0.0)
        hit_200 = result.metrics.get("hit_rate@200", 0.0)
        print(
            f"  {model_name:<12} HitRate@12={hit_12:.4f}  "
            f"@50={hit_50:.4f}  @100={hit_100:.4f}  @200={hit_200:.4f}"
        )
    print("\n--- Diagnostics ---")
    print(f"  Evaluable validation users: {val_diagnostics['pct_val_users_evaluable']:.1f}%")
    print(f"  Cold-start validation users: {val_diagnostics['pct_val_users_cold_start']:.1f}%")
    print(
        f"  Validation interactions with unseen items: "
        f"{val_diagnostics['pct_val_interactions_unseen_items']:.1f}%"
    )
    for model_name, result in results.items():
        print(
            f"  {model_name} candidate coverage (retrievable GT): "
            f"{result.diagnostics.get('candidate_coverage_pct', 0):.1f}%"
        )


def validation_diagnostics(ctx: EvalContext) -> dict[str, float]:
    return compute_validation_diagnostics(ctx.val_df, ctx.train_users, ctx.train_items)
