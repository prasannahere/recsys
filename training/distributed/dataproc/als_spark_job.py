#!/usr/bin/env python3
"""Dataproc PySpark job: distributed ALS training + sharded evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
from pyspark import StorageLevel
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_common import (
    aggregate_metrics,
    build_ground_truth,
    build_id_mappings,
    build_ratings,
    eval_user_ids,
    join_uri,
    load_yaml_config,
    read_transactions,
    save_mappings,
    split_train_val,
    write_json_to_uri,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("als_spark_job")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed ALS on Dataproc")
    parser.add_argument(
        "--transactions-path",
        required=True,
        help="GCS or local path to transactions_train.csv",
    )
    parser.add_argument(
        "--output-uri",
        required=True,
        help="GCS or local output root for model, mappings, metrics",
    )
    parser.add_argument(
        "--config-path",
        default="configs/als.yaml",
        help="Path to als.yaml (local or gs://)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train only; skip distributed evaluation",
    )
    parser.add_argument(
        "--eval-partitions",
        type=int,
        default=15,
        help="Number of Spark partitions for distributed eval (default: 15)",
    )
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=0,
        help="Cap eval users (0 = all validation users in training)",
    )
    return parser.parse_args(argv)


def _train_als(ratings_df, als_cfg: dict[str, Any]):
    als = ALS(
        rank=int(als_cfg["factors"]),
        regParam=float(als_cfg["regularization"]),
        maxIter=int(als_cfg["iterations"]),
        alpha=float(als_cfg["alpha"]),
        implicitPrefs=True,
        coldStartStrategy="drop",
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        seed=int(als_cfg.get("random_state", 42)),
    )
    logger.info(
        "Fitting Spark MLlib ALS (rank=%s reg=%s maxIter=%s alpha=%s)",
        als.getRank(),
        als.getRegParam(),
        als.getMaxIter(),
        als.getAlpha(),
    )
    return als.fit(ratings_df)


def _oversampled_k(candidate_k: int, num_items: int, filter_seen: bool) -> int:
    """Request extra candidates so post-filter top-K still has enough items."""
    if not filter_seen:
        return min(candidate_k, num_items)
    return min(max(candidate_k * 5, candidate_k + 500), num_items)


def _distributed_eval(
    spark: SparkSession,
    model,
    ratings_df,
    user_map_df,
    item_map_df,
    val_df,
    train_df,
    config: dict[str, Any],
    eval_partitions: int,
    max_eval_users: int,
) -> dict[str, float]:
    retrieval_cfg = config["retrieval"]
    candidate_k = int(retrieval_cfg["candidate_k"])
    filter_seen = bool(retrieval_cfg.get("filter_seen", True))
    ks = [int(k) for k in config["evaluation"]["ks"]]

    user_ids = eval_user_ids(val_df, train_df, user_map_df)
    if max_eval_users > 0 and len(user_ids) > max_eval_users:
        rng = np.random.default_rng(int(config["als"].get("random_state", 42)))
        user_ids = sorted(rng.choice(user_ids, max_eval_users, replace=False).tolist())
    num_items = item_map_df.count()
    request_k = _oversampled_k(candidate_k, num_items, filter_seen)
    logger.info(
        "Distributed eval: %d users, %d partitions, candidate_k=%d, request_k=%d, filter_seen=%s",
        len(user_ids),
        eval_partitions,
        candidate_k,
        request_k,
        filter_seen,
    )
    if not user_ids:
        return {}

    train_items = set(
        row["article_id"] for row in train_df.select("article_id").distinct().collect()
    )
    all_gt, retrievable_gt = build_ground_truth(val_df, train_items)

    num_partitions = max(1, min(eval_partitions, len(user_ids)))
    eval_users_df = spark.createDataFrame(
        [(int(uid),) for uid in user_ids], ["userId"]
    ).repartition(num_partitions)

    recs_df = model.recommendForUserSubset(eval_users_df, request_k)
    candidates_df = (
        recs_df.select(
            "userId",
            F.explode("recommendations").alias("rec"),
        )
        .select(
            "userId",
            F.col("rec.itemId").cast("int").alias("itemId"),
            F.col("rec.rating").cast("double").alias("score"),
        )
    )

    if filter_seen:
        seen_pairs = ratings_df.select("userId", "itemId").distinct()
        candidates_df = candidates_df.join(seen_pairs, on=["userId", "itemId"], how="left_anti")

    rank_window = Window.partitionBy("userId").orderBy(F.desc("score"))
    ranked_df = (
        candidates_df.withColumn("rank", F.row_number().over(rank_window))
        .filter(F.col("rank") <= candidate_k)
    )

    user_lookup = user_map_df.select(
        F.col("idx").cast("int").alias("userId"),
        F.col("original_id").alias("customer_id"),
    )
    item_lookup = item_map_df.select(
        F.col("idx").cast("int").alias("itemId"),
        F.col("original_id").alias("article_id"),
    )
    final_df = (
        ranked_df.join(user_lookup, on="userId", how="inner")
        .join(item_lookup, on="itemId", how="inner")
        .select("customer_id", "article_id", "rank")
        .orderBy("customer_id", "rank")
    )

    rec_rows = final_df.collect()
    user_recs: dict[str, list[str]] = {}
    for row in rec_rows:
        user_recs.setdefault(row["customer_id"], []).append(row["article_id"])

    return aggregate_metrics(user_recs, all_gt, retrievable_gt, ks)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_yaml_config(args.config_path)

    spark = (
        SparkSession.builder.appName("recsys-als-dataproc")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    output_uri = args.output_uri.rstrip("/")
    model_dir = join_uri(output_uri, "als")
    mappings_dir = join_uri(output_uri, "mappings")
    spark_model_dir = join_uri(model_dir, "spark_als_model")

    logger.info("Reading transactions from %s", args.transactions_path)
    df = read_transactions(spark, args.transactions_path)

    validation_days = int(config["split"]["validation_days"])
    train_df, val_df, reference_date = split_train_val(df, validation_days)

    storage = StorageLevel.MEMORY_AND_DISK

    user_map_df, item_map_df = build_id_mappings(train_df)
    user_map_df = user_map_df.persist(storage)
    item_map_df = item_map_df.persist(storage)
    save_mappings(user_map_df, item_map_df, mappings_dir)
    n_users = user_map_df.count()
    n_items = item_map_df.count()
    logger.info("ID mappings materialized: %d users, %d items", n_users, n_items)

    shuffle_partitions = int(config.get("spark", {}).get("shuffle_partitions", 200))
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))

    ratings_df = build_ratings(
        train_df,
        user_map_df,
        item_map_df,
        reference_date,
        config["interaction_weighting"],
    )
    ratings_df = ratings_df.repartition(shuffle_partitions, "userId").persist(storage)
    n_ratings = ratings_df.count()
    logger.info("Ratings materialized: %d rows", n_ratings)

    model = _train_als(ratings_df, config["als"])
    model.write().overwrite().save(spark_model_dir)
    logger.info("Saved Spark ALS model to %s", spark_model_dir)

    if args.skip_eval:
        logger.info("Skipping evaluation (--skip-eval)")
        user_map_df.unpersist()
        item_map_df.unpersist()
        ratings_df.unpersist()
        spark.stop()
        return 0

    metrics = _distributed_eval(
        spark,
        model,
        ratings_df,
        user_map_df,
        item_map_df,
        val_df,
        train_df,
        config,
        eval_partitions=args.eval_partitions,
        max_eval_users=args.max_eval_users,
    )
    metrics_path = join_uri(output_uri, "metrics_spark_als.json")
    write_json_to_uri(metrics_path, {"results": {"Spark ALS": metrics}})
    logger.info("Metrics written to %s", metrics_path)
    logger.info("Spark ALS metrics: %s", metrics)

    user_map_df.unpersist()
    item_map_df.unpersist()
    ratings_df.unpersist()

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
