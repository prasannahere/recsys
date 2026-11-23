"""Train implicit ALS on H&M transactions and save artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.pipelines.common import load_config, prepare_training_data, setup_logging
from src.retrieval.als import prepare_als_matrix, train_als_model
from src.retrieval.als.retrieve import sanity_check_recommendations

logger = __import__("logging").getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ALS candidate retrieval model")
    parser.add_argument(
        "--config",
        default="configs/als.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    train_df, val_df, reference_date, user_map, item_map, _train_users, _train_items = (
        prepare_training_data(config)
    )
    matrix = prepare_als_matrix(
        train_df, val_df, reference_date, user_map, item_map, config
    )

    model = train_als_model(matrix, config)

    model_dir = Path(config["artifacts"]["model_dir"])
    model.save(model_dir)

    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    sample_indices = list(range(min(5, user_map.size)))
    sanity_check_recommendations(
        model, user_map, item_map, sample_indices, k=5,
        filter_seen=config["retrieval"]["filter_seen"],
    )

    logger.info("Training complete. Artifacts saved to %s", model_dir)


if __name__ == "__main__":
    main()
