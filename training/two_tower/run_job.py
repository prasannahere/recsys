#!/usr/bin/env python3
"""Container entrypoint: train Two-Tower retrieval + ranker, evaluate, write artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

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
from src.io.storage import join_uri, parent_uri, write_text
from src.pipelines.job_env import apply_job_env_overrides
from src.pipelines.common import (
    get_eval_user_indices,
    load_config,
    prepare_training_data,
    setup_logging,
)
from src.pipelines.gpu_env import log_gpu_environment, resolve_torch_device, skip_eval_from_env
from src.ranking.train import save_ranker, train_two_tower_ranker
from src.ranking.two_tower.features import add_features_to_candidates
from src.retrieval.two_tower.retrieve import (
    build_train_pairs,
    generate_candidates as generate_two_tower_candidates,
)
from src.retrieval.two_tower.train import save_two_tower_artifacts, train_two_tower_model

logger = logging.getLogger(__name__)


def apply_env_overrides(config: dict) -> dict:
    return apply_job_env_overrides(config, retriever="two_tower")


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

    logger.info("Starting Two-Tower training job")
    logger.info("Transactions: %s", config["data"]["transactions_path"])
    logger.info("Output Two-Tower: %s", config["artifacts"]["model_dir"])

    t0 = time.time()
    device = resolve_torch_device()

    train_df, val_df, _reference_date, user_map, item_map, train_users, train_items = (
        prepare_training_data(config)
    )

    logger.info("Training Two-Tower retrieval model...")
    tt_model, tt_index = train_two_tower_model(train_df, user_map, item_map, config)
    tt_dir = config["artifacts"]["model_dir"]
    save_two_tower_artifacts(tt_dir, tt_model, tt_index)

    logger.info("Training Two-Tower LambdaMART ranker...")
    ranker = train_two_tower_ranker(
        config,
        load_retrieval=True,
        train_df=train_df,
        user_map=user_map,
        item_map=item_map,
        two_tower_model=tt_model,
    )
    save_ranker(ranker, config)

    output_root = parent_uri(tt_dir)
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

    eval_user_indices = get_eval_user_indices(
        val_df, user_map, train_users, max_users=max_eval_users
    )
    n_eval_users = len(eval_user_indices)
    logger.info(
        "Starting evaluation for %d validation-window users (chunk_size=%d, "
        "candidate_k=%d, ks=%s, log_interval=%ds, device=%s)",
        n_eval_users,
        user_chunk_size,
        candidate_k,
        ks,
        log_interval_sec,
        device,
    )

    train_pairs = build_train_pairs(train_df) if filter_seen else None
    all_gt, retrievable_gt = build_ground_truth(val_df, train_users, train_items)

    tt_progress = ProgressLogger("two_tower_candidates", n_eval_users, log_interval_sec)
    tt_progress.start()
    tt_metrics = MetricAccumulator(ks=ks)
    ranked_metrics = MetricAccumulator(ks=ks)

    for chunk_idx, user_chunk in enumerate(iter_chunks(eval_user_indices, user_chunk_size)):
        chunk_start = chunk_idx * user_chunk_size
        chunk_cands = generate_two_tower_candidates(
            tt_model,
            tt_index,
            user_chunk,
            user_map,
            item_map,
            train_df,
            k=candidate_k,
            filter_seen=filter_seen,
            device=device,
            train_pairs=train_pairs,
        )
        tt_progress.update(len(user_chunk), force=True)

        n_rows = chunk_cands.height
        feature_progress = ProgressLogger(
            f"two_tower_features_chunk_{chunk_idx + 1}",
            n_rows,
            log_interval_sec,
        )
        feature_progress.start()
        featured = add_features_to_candidates(
            chunk_cands,
            tt_model,
            user_map,
            item_map,
            train_df,
            device=device,
            progress=feature_progress,
        )
        feature_progress.finish()
        ranked_chunk = ranker.rerank_candidates(featured)

        tt_metrics.accumulate(chunk_cands, all_gt, retrievable_gt)
        ranked_metrics.accumulate(ranked_chunk, all_gt, retrievable_gt)

        logger.info(
            "Eval chunk users %d-%d complete (%d candidates)",
            chunk_start + 1,
            chunk_start + len(user_chunk),
            n_rows,
        )

    tt_progress.finish()

    if filter_seen:
        logger.info("filter_seen enabled; per-chunk anti-join applied during retrieval")

    val_diagnostics = compute_validation_diagnostics(val_df, train_users, train_items)

    results = {
        "Two-Tower": tt_metrics.finalize(),
        "Two-Tower Ranker": ranked_metrics.finalize(),
    }

    print("\n=== Two-Tower Training Job Evaluation ===\n")
    print(format_metrics_table(results, ks))

    save_metrics(
        join_uri(output_root, "metrics_two_tower.json"),
        results,
        {k: float(v) for k, v in val_diagnostics.items()},
    )

    logger.info("Job complete in %.1fs", time.time() - t0)
    logger.info("Artifacts: %s", output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M Two-Tower training job")
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_PATH", "configs/two_tower.yaml"),
        help="Path to Two-Tower YAML config",
    )
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    try:
        run(Path(args.config), skip_eval=args.skip_eval or skip_eval_from_env())
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
