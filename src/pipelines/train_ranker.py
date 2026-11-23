"""Train a retriever-specific LightGBM LambdaMART ranker."""

from __future__ import annotations

import argparse
import logging

from src.pipelines.common import load_config, setup_logging
from src.ranking.train import RETRIEVERS, save_ranker, train_ranker

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a retriever-specific LambdaMART ranker"
    )
    parser.add_argument(
        "--retriever",
        required=True,
        choices=RETRIEVERS,
        help="Retriever whose candidates this ranker will rescore",
    )
    parser.add_argument(
        "--config",
        help="Path to retriever YAML config (defaults per retriever)",
    )
    parser.add_argument(
        "--load-retrieval",
        action="store_true",
        help="Load saved retrieval model instead of retraining",
    )
    args = parser.parse_args()

    default_configs = {
        "als": "configs/als.yaml",
        "two_tower": "configs/two_tower.yaml",
        "hybrid": "configs/hybrid.yaml",
    }
    config_path = args.config or default_configs[args.retriever]

    setup_logging()
    config = load_config(config_path)

    logger.info("Training %s LambdaMART ranker", args.retriever)
    ranker = train_ranker(
        args.retriever,
        config,
        load_retrieval=args.load_retrieval,
    )
    ranker_path = save_ranker(ranker, config)
    logger.info("Ranker saved to %s", ranker_path)


if __name__ == "__main__":
    main()
