#!/usr/bin/env python3
"""Container entrypoint: train ALS + ranker, evaluate, write artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from src.evaluation.progress import (
    ProgressLogger,
    evaluation_progress_settings,
    iter_chunks,
)
from src.evaluation.ranking_metrics import (
    MetricAccumulator,
    build_ground_truth,
    compute_validation_diagnostics,
    format_metrics_table,
)
from src.pipelines.job_env import apply_job_env_overrides
from src.pipelines.common import (
    get_eval_user_indices,
    load_config,
    prepare_training_data,
    setup_logging,
)
from src.pipelines.gpu_env import log_gpu_environment, skip_eval_from_env
from src.retrieval.als import prepare_als_matrix, train_als_model
from src.ranking.als.features import add_features_to_candidates
from src.ranking.train import save_ranker, train_als_ranker
from src.io.storage import join_uri, parent_uri, write_text
from src.retrieval.als.retrieve import generate_candidates
from src.retrieval.popularity import build_popularity_ranking, generate_popularity_candidates

logger = logging.getLogger(__name__)


def apply_env_overrides(config: dict) -> dict:
    return apply_job_env_overrides(config, retriever="als")


def save_metrics(path: str | Path, results: dict, diagnostics: dict) -> None:
    payload = {
        "results": {name: r.metrics for name, r in results.items()},
        "diagnostics": diagnostics,
    }
    write_text(path, json.dumps(payload, indent=2))
    logger.info("Metrics written to %s", path)


def _max_eval_users(config: dict) -> int | None:
    max_users = int(config.get("evaluation", {}).get("max_eval_users", 0))
    return max_users if max_users > 0 else None


def run(config_path: Path, skip_eval: bool = False) -> None:
    setup_logging()
    log_gpu_environment()
    config = apply_env_overrides(load_config(config_path))

    logger.info("Starting ALS training job")
    logger.info("Config: %s", config_path)
    logger.info("Transactions: %s", config["data"]["transactions_path"])
    logger.info("Output ALS: %s", config["artifacts"]["model_dir"])

    t0 = time.time()

    train_df, val_df, reference_date, user_map, item_map, train_users, train_items = (
        prepare_training_data(config)
    )
    matrix = prepare_als_matrix(
        train_df, val_df, reference_date, user_map, item_map, config
    )

    logger.info("Training ALS retrieval model...")
    als_model = train_als_model(matrix, config)
    als_dir = config["artifacts"]["model_dir"]
    als_model.save(als_dir)

    logger.info("Training LambdaMART ranker...")
    ranker = train_als_ranker(
        config,
        load_retrieval=True,
        train_df=train_df,
        val_df=val_df,
        reference_date=reference_date,
        user_map=user_map,
        item_map=item_map,
        matrix=matrix,
        als_model=als_model,
    )
    save_ranker(ranker, config)

    output_root = parent_uri(als_dir)
    import yaml

    write_text(join_uri(output_root, "config.yaml"), yaml.dump(config, sort_keys=False))

    if skip_eval:
        logger.info("Training complete (eval skipped) in %.1fs", time.time() - t0)
        return

    retrieval_cfg = config["retrieval"]
    candidate_k = retrieval_cfg["candidate_k"]
    filter_seen = retrieval_cfg["filter_seen"]
    ks = config["evaluation"]["ks"]
    log_interval_sec, user_chunk_size = evaluation_progress_settings(config)
    max_eval_users = _max_eval_users(config)

    # Eval users = customers who purchased during the validation window (last N days)
    # and also appear in training (warm-start users only).
    eval_user_indices = get_eval_user_indices(
        val_df, user_map, train_users, max_users=max_eval_users
    )
    eval_user_ids = [user_map.to_original(i) for i in eval_user_indices]
    n_eval_users = len(eval_user_indices)

    logger.info(
        "Starting evaluation for %d validation-window users (chunk_size=%d, "
        "candidate_k=%d, ks=%s, log_interval=%ds)",
        n_eval_users,
        user_chunk_size,
        candidate_k,
        ks,
        log_interval_sec,
    )

    all_gt, retrievable_gt = build_ground_truth(val_df, train_users, train_items)

    als_progress = ProgressLogger("als_candidates", n_eval_users, log_interval_sec)
    als_progress.start()
    als_metrics = MetricAccumulator(ks=ks)
    ranked_metrics = MetricAccumulator(ks=ks)

    for chunk_idx, user_chunk in enumerate(iter_chunks(eval_user_indices, user_chunk_size)):
        chunk_start = chunk_idx * user_chunk_size
        chunk_cands = generate_candidates(
            als_model,
            user_chunk,
            user_map,
            item_map,
            k=candidate_k,
            filter_seen=filter_seen,
            progress=als_progress,
        )

        n_rows = chunk_cands.height
        feature_progress = ProgressLogger(
            f"als_features_chunk_{chunk_idx + 1}",
            n_rows,
            log_interval_sec,
        )
        feature_progress.start()
        chunk_featured = add_features_to_candidates(
            chunk_cands,
            als_model,
            user_map,
            item_map,
            train_df,
            progress=feature_progress,
        )
        feature_progress.finish()
        ranked_chunk = ranker.rerank_candidates(chunk_featured)

        als_metrics.accumulate(chunk_cands, all_gt, retrievable_gt)
        ranked_metrics.accumulate(ranked_chunk, all_gt, retrievable_gt)

        logger.info(
            "Eval chunk users %d-%d complete (%d candidates)",
            chunk_start + 1,
            chunk_start + len(user_chunk),
            n_rows,
        )

    als_progress.finish()

    pop_ranking = build_popularity_ranking(
        train_df,
        item_map,
        rank_by=config.get("popularity", {}).get("rank_by", "purchase_count"),
    )
    pop_progress = ProgressLogger("popularity_candidates", n_eval_users, log_interval_sec)
    pop_progress.start()
    pop_metrics = MetricAccumulator(ks=ks)
    for user_chunk in iter_chunks(eval_user_ids, user_chunk_size):
        pop_chunk = generate_popularity_candidates(
            user_chunk,
            pop_ranking,
            train_df,
            k=candidate_k,
            filter_seen=filter_seen,
            progress=pop_progress,
        )
        pop_metrics.accumulate(pop_chunk, all_gt, retrievable_gt)
    pop_progress.finish()

    val_diagnostics = compute_validation_diagnostics(val_df, train_users, train_items)

    results = {
        "Popularity": pop_metrics.finalize(),
        "ALS": als_metrics.finalize(),
        "ALS Ranker": ranked_metrics.finalize(),
    }

    print("\n=== ALS Training Job Evaluation ===\n")
    print(format_metrics_table(results, ks))

    metrics_path = join_uri(output_root, "metrics.json")
    save_metrics(
        metrics_path,
        results,
        {
            **{k: float(v) for k, v in val_diagnostics.items()},
            **{k: float(v) for k, v in results["ALS Ranker"].diagnostics.items()},
        },
    )

    logger.info("Job complete in %.1fs", time.time() - t0)
    logger.info("Artifacts: %s", output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M ALS training job")
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_PATH", "configs/als.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train models only, skip evaluation",
    )
    args = parser.parse_args()

    try:
        run(Path(args.config), skip_eval=args.skip_eval or skip_eval_from_env())
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
