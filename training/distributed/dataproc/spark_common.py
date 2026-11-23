"""Shared Spark helpers for Dataproc ALS jobs."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from typing import Any

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, _, blob = uri[len("gs://") :].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load YAML config from a local path or gs:// URI (driver only)."""
    if path.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_name = _parse_gcs_uri(path)
        text = (
            storage.Client()
            .bucket(bucket_name)
            .blob(blob_name)
            .download_as_text()
        )
        return yaml.safe_load(text)
    with open(path) as handle:
        return yaml.safe_load(handle)


def write_json_to_uri(uri: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2)
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_name = _parse_gcs_uri(uri)
        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
            text, content_type="application/json"
        )
        return
    with open(uri, "w") as handle:
        handle.write(text)


def join_uri(base: str, *parts: str) -> str:
    return base.rstrip("/") + "/" + "/".join(parts)


def temporal_split_bounds(max_date: date, validation_days: int) -> tuple[date, date, date]:
    val_start = max_date - timedelta(days=validation_days - 1)
    reference_date = val_start - timedelta(days=1)
    return val_start, reference_date, max_date


def read_transactions(spark: SparkSession, path: str) -> DataFrame:
    df = (
        spark.read.option("header", True)
        .csv(path)
        .withColumn("t_dat", F.to_date("t_dat"))
        .select("t_dat", "customer_id", "article_id")
    )
    return df


def split_train_val(df: DataFrame, validation_days: int) -> tuple[DataFrame, DataFrame, date]:
    max_date_row = df.agg(F.max("t_dat").alias("max_date")).collect()[0]
    max_date: date = max_date_row["max_date"]
    val_start, reference_date, _ = temporal_split_bounds(max_date, validation_days)

    train_df = df.filter(F.col("t_dat") < F.lit(val_start))
    val_df = df.filter(F.col("t_dat") >= F.lit(val_start))
    logger.info(
        "Temporal split: train=%d rows val=%d rows reference_date=%s",
        train_df.count(),
        val_df.count(),
        reference_date,
    )
    return train_df, val_df, reference_date


def _distinct_indexed(df: DataFrame, column: str) -> DataFrame:
    """Map distinct string IDs to contiguous 0-based integers (distributed).

    Uses a distributed global sort (``orderBy``) followed by ``zipWithIndex``,
    which assigns partition-local indices and offsets them without collecting
    all IDs on a single executor.  Indices are deterministic (lexicographic
    on ``column``) and contiguous in ``[0, n)``, satisfying Spark ALS int ID
    requirements.  Avoids unpartitioned ``row_number()`` windows that force a
    single-partition shuffle.
    """
    return (
        df.select(column)
        .distinct()
        .orderBy(column)
        .rdd.map(lambda row: row[0])
        .zipWithIndex()
        .toDF(["original_id", "idx"])
    )


def build_id_mappings(train_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return user and item mapping DataFrames: original_id, idx (0-based)."""
    user_map = _distinct_indexed(train_df, "customer_id")
    item_map = _distinct_indexed(train_df, "article_id")
    return user_map, item_map


def build_ratings(
    train_df: DataFrame,
    user_map: DataFrame,
    item_map: DataFrame,
    reference_date: date,
    weight_cfg: dict[str, Any],
) -> DataFrame:
    """Build Spark MLlib ratings DataFrame: userId, itemId, rating."""
    method = weight_cfg["method"]
    train = train_df.withColumnRenamed("customer_id", "user_id").withColumnRenamed(
        "article_id", "item_id"
    )

    if method == "binary":
        interactions = train.groupBy("user_id", "item_id").agg(F.lit(1.0).alias("rating"))
    elif method == "count":
        interactions = train.groupBy("user_id", "item_id").agg(
            F.count(F.lit(1)).cast("double").alias("rating")
        )
    elif method == "recency":
        lam = float(weight_cfg.get("lambda", 0.01))
        interactions = (
            train.withColumn("age_days", F.datediff(F.lit(reference_date), F.col("t_dat")))
            .withColumn("row_weight", F.exp(F.lit(-lam) * F.col("age_days")))
            .groupBy("user_id", "item_id")
            .agg(F.sum("row_weight").alias("rating"))
        )
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    user_lookup = user_map.select(
        F.col("original_id").alias("user_id"),
        F.col("idx").alias("userId"),
    )
    item_lookup = item_map.select(
        F.col("original_id").alias("item_id"),
        F.col("idx").alias("itemId"),
    )
    return (
        interactions.join(user_lookup, on="user_id", how="inner")
        .join(item_lookup, on="item_id", how="inner")
        .select("userId", "itemId", "rating")
    )


def save_mappings(user_map: DataFrame, item_map: DataFrame, mappings_dir: str) -> None:
    user_map.write.mode("overwrite").parquet(join_uri(mappings_dir, "user_mapping.parquet"))
    item_map.write.mode("overwrite").parquet(join_uri(mappings_dir, "item_mapping.parquet"))


def eval_user_ids(
    val_df: DataFrame,
    train_df: DataFrame,
    user_map: DataFrame,
) -> list[int]:
    """Validation users present in training, as integer userId."""
    train_users = train_df.select("customer_id").distinct()
    val_users = val_df.select("customer_id").distinct()
    eval_users = val_users.join(train_users, on="customer_id", how="inner")
    lookup = user_map.select(
        F.col("original_id").alias("customer_id"),
        F.col("idx").alias("userId"),
    )
    rows = eval_users.join(lookup, on="customer_id", how="inner").select("userId").collect()
    return sorted(int(row["userId"]) for row in rows)


def build_ground_truth(val_df: DataFrame, train_items: set[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build per-user ground truth dicts keyed by customer_id."""
    all_gt: dict[str, set[str]] = {}
    retrievable_gt: dict[str, set[str]] = {}
    grouped = (
        val_df.groupBy("customer_id")
        .agg(F.collect_set("article_id").alias("articles"))
        .collect()
    )
    for row in grouped:
        user = row["customer_id"]
        articles = set(row["articles"])
        all_gt[user] = articles
        retrievable_gt[user] = articles & train_items
    return all_gt, retrievable_gt


def recall_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(recommended[:k])) / len(relevant)


def hit_rate_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    return 1.0 if relevant & set(recommended[:k]) else 0.0


def average_precision_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k)


def ndcg_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if not relevant:
        return 0.0

    def dcg(items: list[str]) -> float:
        score = 0.0
        for i, item in enumerate(items[:k], start=1):
            if item in relevant:
                score += 1.0 / math.log2(i + 1)
        return score

    actual = dcg(recommended)
    ideal = dcg(list(relevant)[:k])
    return actual / ideal if ideal > 0 else 0.0


def aggregate_metrics(
    user_recs: dict[str, list[str]],
    all_gt: dict[str, set[str]],
    retrievable_gt: dict[str, set[str]],
    ks: list[int],
) -> dict[str, float]:
    eval_users = [u for u in all_gt if u in user_recs]
    n_users = len(eval_users)
    if n_users == 0:
        return {}

    metrics: dict[str, float] = {}
    for k in ks:
        recall_all: list[float] = []
        recall_ret: list[float] = []
        hit: list[float] = []
        map_k: list[float] = []
        ndcg: list[float] = []
        for user in eval_users:
            recs = user_recs[user]
            gt_all = all_gt[user]
            gt_ret = retrievable_gt[user]
            recall_all.append(recall_at_k(gt_all, recs, k))
            recall_ret.append(recall_at_k(gt_ret, recs, k) if gt_ret else 0.0)
            hit.append(hit_rate_at_k(gt_all, recs, k))
            map_k.append(average_precision_at_k(gt_all, recs, k))
            ndcg.append(ndcg_at_k(gt_all, recs, k))
        metrics[f"recall_all_gt@{k}"] = sum(recall_all) / n_users
        metrics[f"recall_retrievable@{k}"] = sum(recall_ret) / n_users
        metrics[f"hit_rate@{k}"] = sum(hit) / n_users
        metrics[f"map@{k}"] = sum(map_k) / n_users
        metrics[f"ndcg@{k}"] = sum(ndcg) / n_users
    return metrics
