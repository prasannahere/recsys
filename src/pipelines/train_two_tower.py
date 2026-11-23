"""Train pure two-tower (ID-only) retrieval model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.pipelines.common import load_config, prepare_training_data, setup_logging
from src.retrieval.two_tower.train import save_two_tower_artifacts, train_two_tower_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train pure two-tower (ID-only) candidate retrieval model"
    )
    parser.add_argument(
        "--config",
        default="configs/two_tower.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    train_df, _val_df, _reference_date, user_map, item_map, _train_users, _train_items = (
        prepare_training_data(config)
    )

    model, index = train_two_tower_model(train_df, user_map, item_map, config)

    model_dir = Path(config["artifacts"]["model_dir"])
    save_two_tower_artifacts(model_dir, model, index)

    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info("Two-Tower training complete. Artifacts saved to %s", model_dir)


if __name__ == "__main__":
    main()
