"""Train Hybrid retrieval model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.pipelines.common import load_config, prepare_training_data, setup_logging
from src.retrieval.hybrid.train import save_hybrid_artifacts, train_hybrid_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hybrid candidate retrieval model")
    parser.add_argument(
        "--config",
        default="configs/hybrid.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    train_df, _val_df, _reference_date, user_map, item_map, _train_users, _train_items = (
        prepare_training_data(config)
    )

    model, encoders, item_features, history_items, history_lengths, index = (
        train_hybrid_model(train_df, user_map, item_map, config)
    )

    model_dir = Path(config["artifacts"]["model_dir"])
    save_hybrid_artifacts(
        model_dir,
        model,
        encoders,
        item_features,
        history_items,
        history_lengths,
        index,
    )

    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info("Hybrid training complete. Artifacts saved to %s", model_dir)


if __name__ == "__main__":
    main()
