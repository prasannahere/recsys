"""Evaluate pure Two-Tower candidate retrieval."""

from __future__ import annotations

import argparse
import logging

from src.pipelines.common import (
    load_config,
    prepare_training_data,
    setup_logging,
    verify_article_ids_in_mapping,
    verify_filter_seen,
)
from src.pipelines.ranking_eval import evaluate_retrieval_and_ranker, print_retrieval_vs_ranker
from src.pipelines.retrieval_eval import (
    build_eval_context,
    print_single_results,
    score_candidates,
    two_tower_candidates,
    validation_diagnostics,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pure Two-Tower (ID-only) candidate retrieval"
    )
    parser.add_argument(
        "--config",
        default="configs/two_tower.yaml",
        help="Path to Two-Tower YAML configuration file",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load saved Two-Tower model instead of retraining",
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

    if args.with_ranker or args.load_ranker:
        results = evaluate_retrieval_and_ranker(
            "two_tower",
            ctx,
            config,
            load_retrieval=args.load_model,
            load_ranker_flag=args.load_ranker,
        )
        print_retrieval_vs_ranker("two_tower", results, ctx)
        return

    candidates = two_tower_candidates(ctx, config, load_saved=args.load_model)
    if ctx.filter_seen:
        verify_filter_seen(candidates, ctx.train_df)
    verify_article_ids_in_mapping(candidates, ctx.item_map)

    results = score_candidates(ctx, candidates)
    val_diagnostics = validation_diagnostics(ctx)
    print_single_results("Two-Tower", results, val_diagnostics, ctx.ks)


if __name__ == "__main__":
    main()
