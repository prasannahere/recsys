#!/usr/bin/env python3
"""Container entrypoint: train Hybrid retrieval + ranker, evaluate, write artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

import polars as pl

from src.evaluation.progress import (
    ProgressLogger,
    evaluation_progress_settings,
    iter_chunks,
)
from src.evaluation.ranking_metrics import (
    build_ground_truth,
    compute_validation_diagnostics,
    evaluate_candidates,
    format_metrics_table,
)
from src.io.storage import join_uri, parent_uri, write_text
from src.pipelines.job_env import apply_job_env_overrides
from src.pipelines.common import (
    get_eval_user_indices,
    load_config,
    prepare_training_data,
    setup_logging,
    verify_article_ids_in_mapping,
    verify_filter_seen,
)
from src.pipelines.gpu_env import log_gpu_environment, skip_eval_from_env
from src.ranking.hybrid.features import add_features_to_candidates
from src.ranking.train import save_ranker, train_hybrid_ranker
from src.retrieval.hybrid.retrieve import generate_candidates as generate_hybrid_candidates
from src.retrieval.hybrid.train import (
    load_hybrid_artifacts,
    save_hybrid_artifacts,
    train_hybrid_model,
)

logger = logging.getLogger(__name__)


def apply_env_overrides(config: dict) -> dict:
    return apply_job_env_overrides(config, retriever="hybrid", require_articles=True)


def save_metrics(path: str | Path, results: dict, diagnostics: dict) -> None:
    payload = {
        "results": {name: r.metrics for name, r in results.items()},
        "diagnostics": diagnostics,
    }
    write_text(path, json.dumps(payload, indent=2))
    logger.info("Metrics written to %s", path)


def run(config_path: Path, skip_eval: bool = False) -> None:
    setup_logging()
    log_gpu_environment()
    config = apply_env_overrides(load_config(config_path))

    logger.info("Starting Hybrid training job")
    logger.info("Transactions: %s", config["data"]["transactions_path"])
    logger.info("Articles: %s", config["data"]["articles_path"])
    logger.info("Output Hybrid: %s", config["artifacts"]["model_dir"])

    t0 = time.time()

    train_df, val_df, reference_date, user_map, item_map, train_users, train_items = (
        prepare_training_data(config)
    )

    logger.info("Training Hybrid retrieval model...")
    model, encoders, item_features, history_items, history_lengths, index = (
        train_hybrid_model(train_df, user_map, item_map, config)
    )
    hybrid_dir = config["artifacts"]["model_dir"]
    save_hybrid_artifacts(
        hybrid_dir,
        model,
        encoders,
        item_features,
        history_items,
        history_lengths,
        index,
    )

    logger.info("Training Hybrid LambdaMART ranker...")
    ranker = train_hybrid_ranker(config, load_retrieval=True)
    save_ranker(ranker, config)

    output_root = parent_uri(hybrid_dir)
    write_text(join_uri(output_root, "config.yaml"), yaml.dump(config, sort_keys=False))

    if skip_eval:
        logger.info("Training complete (eval skipped) in %.1fs", time.time() - t0)
        return

    retrieval_cfg = config["retrieval"]
    candidate_k = retrieval_cfg["candidate_k"]
    filter_seen = retrieval_cfg["filter_seen"]
    ks = config["evaluation"]["ks"]
    log_interval_sec, user_chunk_size = evaluation_progress_settings(config)

    eval_user_indices = get_eval_user_indices(val_df, user_map, train_users)
    n_eval_users = len(eval_user_indices)
    logger.info(
        "Starting evaluation for %d users (chunk_size=%d, log_interval=%ds)",
        n_eval_users,
        user_chunk_size,
        log_interval_sec,
    )

    model, _enc, item_features, history_items, history_lengths, index = (
        load_hybrid_artifacts(hybrid_dir)
    )

    hybrid_frames: list[pl.DataFrame] = []
    ranked_frames: list[pl.DataFrame] = []
    hybrid_progress = ProgressLogger("hybrid_candidates", n_eval_users, log_interval_sec)
    hybrid_progress.start()

    for chunk_idx, user_chunk in enumerate(iter_chunks(eval_user_indices, user_chunk_size)):
        chunk_start = chunk_idx * user_chunk_size
        chunk_cands = generate_hybrid_candidates(
            model,
            index,
            user_chunk,
            user_map,
            item_map,
            history_items,
            history_lengths,
            item_features,
            train_df,
            k=candidate_k,
            filter_seen=filter_seen,
        )
        hybrid_frames.append(chunk_cands)
        hybrid_progress.update(len(user_chunk), force=True)

        featured = add_features_to_candidates(
            chunk_cands,
            model,
            user_map,
            item_map,
            train_df,
            history_items,
            history_lengths,
            item_features,
        )
        ranked_frames.append(ranker.rerank_candidates(featured))
        logger.info(
            "Eval chunk users %d-%d complete (%d candidates)",
            chunk_start + 1,
            chunk_start + len(user_chunk),
            chunk_cands.height,
        )

    hybrid_progress.finish()
    hybrid_candidates_df = pl.concat(hybrid_frames)
    ranked_candidates = pl.concat(ranked_frames)

    if filter_seen:
        verify_filter_seen(hybrid_candidates_df, train_df)
    verify_article_ids_in_mapping(hybrid_candidates_df, item_map)

    all_gt, retrievable_gt = build_ground_truth(val_df, train_users, train_items)
    val_diagnostics = compute_validation_diagnostics(val_df, train_users, train_items)

    results = {}
    for model_name, candidates, score_col in (
        ("Hybrid", hybrid_candidates_df, "retrieval_score"),
        ("Hybrid Ranker", ranked_candidates, "ranker_score"),
    ):
        model_progress = ProgressLogger(model_name, n_eval_users, log_interval_sec)
        model_progress.start()
        results[model_name] = evaluate_candidates(
            candidates,
            all_gt,
            retrievable_gt,
            ks,
            score_col=score_col,
            progress=model_progress,
        )
        model_progress.finish()

    print("\n=== Hybrid Training Job Evaluation ===\n")
    print(format_metrics_table(results, ks))

    save_metrics(
        join_uri(output_root, "metrics_hybrid.json"),
        results,
        {k: float(v) for k, v in val_diagnostics.items()},
    )

    logger.info("Job complete in %.1fs", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M Hybrid training job")
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_PATH", "configs/hybrid.yaml"),
        help="Path to Hybrid YAML config",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train Hybrid only, skip evaluation",
    )
    args = parser.parse_args()

    try:
        run(Path(args.config), skip_eval=args.skip_eval or skip_eval_from_env())
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
