"""Offline ranking metrics for candidate retrieval evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from src.evaluation.progress import ProgressLogger

logger = logging.getLogger(__name__)


@dataclass
class MetricResults:
    """Macro-averaged metrics across evaluable users."""

    metrics: dict[str, float] = field(default_factory=dict)
    n_users: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)


def build_ground_truth(
    val_df: pl.DataFrame,
    train_users: set[str],
    train_items: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build per-user ground truth from validation purchases.

    Returns:
        all_gt: user -> all unique validation article_ids purchased.
        retrievable_gt: user -> validation articles that exist in training item set.
    """
    all_gt: dict[str, set[str]] = {}
    retrievable_gt: dict[str, set[str]] = {}

    for row in (
        val_df.group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("articles"))
        .iter_rows(named=True)
    ):
        user = row["customer_id"]
        if user not in train_users:
            continue
        articles = set(row["articles"])
        all_gt[user] = articles
        retrievable_gt[user] = articles & train_items

    return all_gt, retrievable_gt


def _recall_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    return len(relevant & top_k) / len(relevant)


def _hit_rate_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    top_k = set(recommended[:k])
    return 1.0 if len(relevant & top_k) > 0 else 0.0


def _average_precision_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant), k)


def _ndcg_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if not relevant:
        return 0.0

    def dcg(items: list[str]) -> float:
        score = 0.0
        for i, item in enumerate(items[:k], start=1):
            if item in relevant:
                score += 1.0 / __import__("math").log2(i + 1)
        return score

    actual = dcg(recommended)
    ideal = dcg(list(relevant)[:k])  # all relevant at top
    return actual / ideal if ideal > 0 else 0.0


def _build_user_recs(candidates: pl.DataFrame) -> dict[str, list[str]]:
    user_recs: dict[str, list[str]] = {}
    for row in (
        candidates.sort("rank")
        .group_by("customer_id")
        .agg(pl.col("article_id").alias("recs"))
        .iter_rows(named=True)
    ):
        user_recs[row["customer_id"]] = row["recs"]
    return user_recs


def _user_metric_values(
    recs: list[str],
    gt_all: set[str],
    gt_ret: set[str],
    ks: list[int],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for k in ks:
        values[f"recall_all_gt@{k}"] = _recall_at_k(gt_all, recs, k)
        values[f"recall_retrievable@{k}"] = (
            _recall_at_k(gt_ret, recs, k) if gt_ret else 0.0
        )
        values[f"hit_rate@{k}"] = _hit_rate_at_k(gt_all, recs, k)
        values[f"map@{k}"] = _average_precision_at_k(gt_all, recs, k)
        values[f"ndcg@{k}"] = _ndcg_at_k(gt_all, recs, k)
    return values


@dataclass
class MetricAccumulator:
    """Accumulates macro-averaged metric sums across evaluation chunks."""

    ks: list[int]
    n_users: int = 0
    metric_sums: dict[str, float] = field(default_factory=dict)
    gt_size_sum: float = 0.0
    ret_gt_size_sum: float = 0.0
    recommended_items: set[str] = field(default_factory=set)
    retrievable_items: set[str] = field(default_factory=set)

    def accumulate(
        self,
        candidates: pl.DataFrame,
        all_gt: dict[str, set[str]],
        retrievable_gt: dict[str, set[str]],
        progress: ProgressLogger | None = None,
    ) -> None:
        if candidates.height > 0:
            self.recommended_items.update(candidates["article_id"].unique().to_list())

        user_recs = _build_user_recs(candidates)
        eval_users = [u for u in all_gt if u in user_recs]
        if not eval_users:
            return

        for user in eval_users:
            recs = user_recs[user]
            gt_all = all_gt[user]
            gt_ret = retrievable_gt[user]
            self.retrievable_items.update(gt_ret)

            for key, value in _user_metric_values(recs, gt_all, gt_ret, self.ks).items():
                self.metric_sums[key] = self.metric_sums.get(key, 0.0) + value

            self.gt_size_sum += len(gt_all)
            self.ret_gt_size_sum += len(gt_ret)
            self.n_users += 1
            if progress is not None:
                progress.update(1)

    def finalize(self) -> MetricResults:
        if self.n_users == 0:
            return MetricResults()

        metrics = {key: total / self.n_users for key, total in self.metric_sums.items()}
        covered = self.retrievable_items & self.recommended_items
        diagnostics = {
            "n_eval_users": float(self.n_users),
            "avg_gt_items_per_user": self.gt_size_sum / self.n_users,
            "avg_retrievable_gt_per_user": self.ret_gt_size_sum / self.n_users,
            "candidate_coverage_pct": (
                100.0 * len(covered) / len(self.retrievable_items)
                if self.retrievable_items
                else 0.0
            ),
        }
        return MetricResults(metrics=metrics, n_users=self.n_users, diagnostics=diagnostics)


def evaluate_candidates(
    candidates: pl.DataFrame,
    all_gt: dict[str, set[str]],
    retrievable_gt: dict[str, set[str]],
    ks: list[int],
    score_col: str = "als_score",
    progress: ProgressLogger | None = None,
) -> MetricResults:
    """Evaluate retrieval candidates against validation ground truth.

    Args:
        candidates: DataFrame with customer_id, article_id, rank (and score col).
        all_gt: Full validation ground truth per user.
        retrievable_gt: Ground truth restricted to training item universe.
        ks: List of cutoff values.
        score_col: Unused but kept for API clarity; ranking uses rank column.

    Returns:
        MetricResults with macro-averaged metrics and diagnostics.
    """
    del score_col  # ranking determined by rank column

    accumulator = MetricAccumulator(ks=ks)
    accumulator.accumulate(candidates, all_gt, retrievable_gt, progress=progress)
    if accumulator.n_users == 0:
        logger.warning("No evaluable users found")
    return accumulator.finalize()


def compute_validation_diagnostics(
    val_df: pl.DataFrame,
    train_users: set[str],
    train_items: set[str],
) -> dict[str, float]:
    """Compute dataset-level validation diagnostics."""
    val_users = set(val_df["customer_id"].unique().to_list())
    val_user_count = len(val_users)
    evaluable_users = val_users & train_users
    cold_start_users = val_users - train_users

    unseen_item_rows = val_df.filter(~pl.col("article_id").is_in(list(train_items)))
    unseen_pct = 100.0 * unseen_item_rows.height / max(val_df.height, 1)

    return {
        "pct_val_users_evaluable": (
            100.0 * len(evaluable_users) / val_user_count if val_user_count else 0.0
        ),
        "pct_val_users_cold_start": (
            100.0 * len(cold_start_users) / val_user_count if val_user_count else 0.0
        ),
        "pct_val_interactions_unseen_items": unseen_pct,
        "n_cold_start_users": float(len(cold_start_users)),
    }


def format_metrics_table(results: dict[str, MetricResults], ks: list[int]) -> str:
    """Format a comparison table for retrieval models at the configured K values."""
    metric_specs: list[tuple[str, str]] = []
    for k in ks:
        metric_specs.append((f"Recall@{k}", f"recall_retrievable@{k}"))
        metric_specs.append((f"MAP@{k}", f"map@{k}"))
        metric_specs.append((f"NDCG@{k}", f"ndcg@{k}"))

    header_cols = [name for name, _ in metric_specs]
    metric_keys = [key for _, key in metric_specs]

    lines = ["Model       " + "  ".join(f"{h:>10}" for h in header_cols)]
    for model_name, result in results.items():
        vals = [f"{result.metrics.get(key, 0.0):>10.4f}" for key in metric_keys]
        lines.append(f"{model_name:<12}" + "  ".join(vals))

    return "\n".join(lines)
