"""Compare multiple retrievers under the same temporal validation split."""

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
from src.evaluation.ranking_metrics import build_ground_truth
from src.pipelines.retrieval_eval import (
    als_candidates,
    build_eval_context,
    hybrid_candidates,
    popularity_candidates,
    print_comparison_results,
    score_candidates,
    two_tower_candidates,
    validation_diagnostics,
)

logger = logging.getLogger(__name__)

RETRIEVER_CHOICES = ("popularity", "als", "two_tower", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate retrievers under a shared temporal split"
    )
    parser.add_argument(
        "--config",
        default="configs/als.yaml",
        help="Shared data split and evaluation settings",
    )
    parser.add_argument(
        "--retrievers",
        default="popularity,als,two_tower,hybrid",
        help=f"Comma-separated retrievers: {', '.join(RETRIEVER_CHOICES)}",
    )
    parser.add_argument("--als-config", default="configs/als.yaml")
    parser.add_argument("--two-tower-config", default="configs/two_tower.yaml")
    parser.add_argument("--hybrid-config", default="configs/hybrid.yaml")
    parser.add_argument("--load-als", action="store_true")
    parser.add_argument("--load-two-tower", action="store_true")
    parser.add_argument("--load-hybrid", action="store_true")
    args = parser.parse_args()

    setup_logging()
    data_config = load_config(args.config)
    selected = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    unknown = set(selected) - set(RETRIEVER_CHOICES)
    if unknown:
        raise ValueError(f"Unknown retrievers: {sorted(unknown)}")

    als_config = load_config(args.als_config)
    two_tower_config = load_config(args.two_tower_config)
    hybrid_config = load_config(args.hybrid_config)

    train_df, val_df, reference_date, user_map, item_map, train_users, train_items = (
        prepare_training_data(data_config)
    )
    ctx = build_eval_context(
        train_df,
        val_df,
        reference_date,
        user_map,
        item_map,
        train_users,
        train_items,
        data_config,
    )
    logger.info("Evaluating %d validation users", len(ctx.eval_user_indices))
    all_gt, retrievable_gt = build_ground_truth(val_df, train_users, train_items)

    generators = {
        "popularity": lambda: popularity_candidates(ctx, data_config),
        "als": lambda: als_candidates(ctx, als_config, load_saved=args.load_als),
        "two_tower": lambda: two_tower_candidates(
            ctx, two_tower_config, load_saved=args.load_two_tower
        ),
        "hybrid": lambda: hybrid_candidates(ctx, hybrid_config, load_saved=args.load_hybrid),
    }
    display_names = {
        "popularity": "Popularity",
        "als": "ALS",
        "two_tower": "Two-Tower",
        "hybrid": "Hybrid",
    }

    results = {}
    for name in selected:
        logger.info("--- %s ---", display_names[name])
        candidates = generators[name]()
        if ctx.filter_seen:
            verify_filter_seen(candidates, ctx.train_df)
        verify_article_ids_in_mapping(candidates, ctx.item_map)
        results[display_names[name]] = score_candidates(
            ctx, candidates, all_gt, retrievable_gt
        )

    val_diagnostics = validation_diagnostics(ctx)
    print_comparison_results(results, val_diagnostics, ctx.ks)


if __name__ == "__main__":
    main()
