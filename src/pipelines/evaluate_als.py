"""Evaluate ALS and popularity baselines with temporal offline metrics."""

from __future__ import annotations

import argparse
import logging
import time

from src.io.storage import artifact_exists
from src.evaluation.ranking_metrics import (
    build_ground_truth,
    compute_validation_diagnostics,
    evaluate_candidates,
    format_metrics_table,
)
from src.pipelines.common import (
    get_eval_user_indices,
    load_config,
    prepare_training_data,
    setup_logging,
    verify_article_ids_in_mapping,
    verify_filter_seen,
)
from src.io.storage import artifact_exists
from src.pipelines.ranking_eval import evaluate_retrieval_and_ranker, print_retrieval_vs_ranker
from src.pipelines.retrieval_eval import build_eval_context
from src.retrieval.als import ALSModel, prepare_als_matrix, train_als_model
from src.retrieval.als.retrieve import generate_candidates, sanity_check_recommendations
from src.retrieval.popularity import build_popularity_ranking, generate_popularity_candidates

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ALS candidate retrieval")
    parser.add_argument(
        "--config",
        default="configs/als.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load saved ALS model instead of retraining",
    )
    parser.add_argument(
        "--with-ranker",
        action="store_true",
        help="Also train/load LambdaMART ranker and compare retrieval vs reranked",
    )
    parser.add_argument(
        "--load-ranker",
        action="store_true",
        help="Load saved ranker (implies --with-ranker)",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    train_df, val_df, reference_date, user_map, item_map, train_users, train_items = (
        prepare_training_data(config)
    )

    if args.with_ranker or args.load_ranker:
        ctx = build_eval_context(
            train_df,
            val_df,
            reference_date,
            user_map,
            item_map,
            train_users,
            train_items,
            config,
        )
        logger.info("Evaluating %d validation users", len(ctx.eval_user_indices))
        results = evaluate_retrieval_and_ranker(
            "als",
            ctx,
            config,
            load_retrieval=args.load_model,
            load_ranker_flag=args.load_ranker,
        )
        print_retrieval_vs_ranker("als", results, ctx)
        return

    matrix = prepare_als_matrix(
        train_df, val_df, reference_date, user_map, item_map, config
    )

    retrieval_cfg = config["retrieval"]
    candidate_k = retrieval_cfg["candidate_k"]
    filter_seen = retrieval_cfg["filter_seen"]
    ks = config["evaluation"]["ks"]

    # Train or load ALS
    model_dir = config["artifacts"]["model_dir"]
    if args.load_model and artifact_exists(model_dir, "als_model.pkl"):
        logger.info("Loading saved ALS model from %s", model_dir)
        model = ALSModel.load(model_dir)
        model.attach_user_item_matrix(matrix)
    else:
        logger.info("Training ALS model...")
        model = train_als_model(matrix, config)
        model.save(model_dir)

    eval_user_indices = get_eval_user_indices(val_df, user_map, train_users)
    eval_user_ids = [user_map.to_original(i) for i in eval_user_indices]

    logger.info(
        "Evaluating %d validation users (present in training)",
        len(eval_user_indices),
    )

    # Sample sanity checks
    sample_indices = eval_user_indices[: min(5, len(eval_user_indices))]
    sanity_check_recommendations(
        model, user_map, item_map, sample_indices, k=5, filter_seen=filter_seen
    )

    # ALS candidates
    logger.info("Generating ALS Top-%d candidates...", candidate_k)
    t0 = time.time()
    als_candidates = generate_candidates(
        model,
        eval_user_indices,
        user_map,
        item_map,
        k=candidate_k,
        filter_seen=filter_seen,
    )
    logger.info(
        "ALS candidate generation: %d rows in %.1f s",
        als_candidates.height,
        time.time() - t0,
    )

    if filter_seen:
        verify_filter_seen(als_candidates, train_df)
    verify_article_ids_in_mapping(als_candidates, item_map)

    # Popularity baseline
    pop_ranking = build_popularity_ranking(
        train_df,
        item_map,
        rank_by=config.get("popularity", {}).get("rank_by", "purchase_count"),
    )
    pop_candidates = generate_popularity_candidates(
        eval_user_ids,
        pop_ranking,
        train_df,
        k=candidate_k,
        filter_seen=filter_seen,
    )

    # Ground truth
    all_gt, retrievable_gt = build_ground_truth(val_df, train_users, train_items)

    val_diagnostics = compute_validation_diagnostics(val_df, train_users, train_items)
    logger.info("--- Validation diagnostics ---")
    for key, val in val_diagnostics.items():
        logger.info("  %s: %.2f", key, val)

    # Evaluate both models
    als_results = evaluate_candidates(
        als_candidates, all_gt, retrievable_gt, ks, score_col="retrieval_score"
    )
    pop_results = evaluate_candidates(
        pop_candidates, all_gt, retrievable_gt, ks, score_col="retrieval_score"
    )

    # Print comparison table
    table = format_metrics_table(
        {"Popularity": pop_results, "ALS": als_results},
        ks,
    )
    print("\n=== Candidate Retrieval Evaluation ===\n")
    print(table)

    print("\n--- Recall variants (ALS, retrievable vs all GT) ---")
    for k in [50, 100, 200]:
        ret = als_results.metrics.get(f"recall_retrievable@{k}", 0)
        all_ = als_results.metrics.get(f"recall_all_gt@{k}", 0)
        print(f"  K={k:>3}: recall_retrievable={ret:.4f}  recall_all_gt={all_:.4f}")

    print("\n--- Diagnostics ---")
    print(f"  Evaluable validation users: {val_diagnostics['pct_val_users_evaluable']:.1f}%")
    print(f"  Cold-start validation users: {val_diagnostics['pct_val_users_cold_start']:.1f}%")
    print(
        f"  Validation interactions with unseen items: "
        f"{val_diagnostics['pct_val_interactions_unseen_items']:.1f}%"
    )
    print(
        f"  Avg ground-truth items per user: "
        f"{als_results.diagnostics.get('avg_gt_items_per_user', 0):.2f}"
    )
    print(
        f"  ALS candidate coverage (retrievable GT): "
        f"{als_results.diagnostics.get('candidate_coverage_pct', 0):.1f}%"
    )


if __name__ == "__main__":
    main()
