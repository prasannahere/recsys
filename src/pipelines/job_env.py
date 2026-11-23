"""Shared env overrides for container training jobs."""

from __future__ import annotations

import os
from typing import Any

from src.io.storage import join_uri


def apply_job_env_overrides(
    config: dict[str, Any],
    *,
    retriever: str,
    require_articles: bool = False,
) -> dict[str, Any]:
    """Override config paths from env vars (local mounts or gs:// URIs)."""
    transactions_path = os.environ.get("TRANSACTIONS_PATH")
    articles_path = os.environ.get("ARTICLES_PATH")
    output_uri = os.environ.get("OUTPUT_URI")
    data_dir = os.environ.get("DATA_DIR")
    output_dir = os.environ.get("OUTPUT_DIR")

    if transactions_path:
        config["data"]["transactions_path"] = transactions_path
    elif data_dir:
        config["data"]["transactions_path"] = join_uri(
            data_dir, "raw", "transactions_train.csv"
        )

    if require_articles:
        if articles_path:
            config["data"]["articles_path"] = articles_path
        elif data_dir:
            config["data"]["articles_path"] = join_uri(data_dir, "raw", "articles.csv")

    config.setdefault("ranking", {})
    if output_uri:
        config["artifacts"]["model_dir"] = join_uri(output_uri, retriever)
        config["artifacts"]["mappings_dir"] = join_uri(output_uri, "mappings")
        config["ranking"]["model_path"] = join_uri(
            output_uri, retriever, "ranker", "ranker.pkl"
        )
    elif output_dir:
        config["artifacts"]["model_dir"] = join_uri(output_dir, retriever)
        config["artifacts"]["mappings_dir"] = join_uri(output_dir, "mappings")
        config["ranking"]["model_path"] = join_uri(
            output_dir, retriever, "ranker", "ranker.pkl"
        )

    return config
