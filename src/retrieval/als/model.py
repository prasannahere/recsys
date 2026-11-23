"""Implicit ALS model wrapper."""

from __future__ import annotations

import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.io.storage import join_uri, open_binary

if TYPE_CHECKING:
    from pathlib import Path
    from src.evaluation.progress import ProgressLogger

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


@dataclass
class ALSConfig:
    factors: int = 128
    regularization: float = 0.05
    iterations: int = 20
    alpha: float = 20.0
    random_state: int = 42


class ALSModel:
    """Wrapper around implicit ALS for training and Top-K candidate retrieval.

    The implicit library expects a user x item CSR matrix where values are
    preference strengths r_ui. During fit(), implicit converts these to
    confidence values: C_ui = 1 + alpha * r_ui (Hu-Koren-Volinsky formulation).

    Scoring at inference: score(u, i) = user_factors[u] · item_factors[i]
    """

    def __init__(self, config: ALSConfig) -> None:
        self.config = config
        self.model = AlternatingLeastSquares(
            factors=config.factors,
            regularization=config.regularization,
            iterations=config.iterations,
            alpha=config.alpha,
            random_state=config.random_state,
            use_native=True,
            use_cg=True,
        )
        self._user_item_matrix: csr_matrix | None = None

    @property
    def user_factors(self) -> np.ndarray:
        return self.model.user_factors

    @property
    def item_factors(self) -> np.ndarray:
        return self.model.item_factors

    def fit(self, user_item_matrix: csr_matrix) -> None:
        """Train ALS on a user x item CSR preference matrix."""
        self._user_item_matrix = user_item_matrix
        logger.info(
            "Training ALS (factors=%d, reg=%.3f, iter=%d, alpha=%.1f)",
            self.config.factors,
            self.config.regularization,
            self.config.iterations,
            self.config.alpha,
        )
        start = time.time()
        self.model = AlternatingLeastSquares(
            factors=self.config.factors,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            alpha=self.config.alpha,
            random_state=self.config.random_state,
            use_native=True,
            use_cg=True,
        )
        prev_threads = os.environ.get("OPENBLAS_NUM_THREADS")
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        try:
            self.model.fit(user_item_matrix, show_progress=True)
        finally:
            if prev_threads is None:
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = prev_threads
        elapsed = time.time() - start
        logger.info("ALS training completed in %.1f seconds", elapsed)

        assert self.model.user_factors.shape == (
            user_item_matrix.shape[0],
            self.config.factors,
        ), "User factor dimension mismatch"
        assert self.model.item_factors.shape == (
            user_item_matrix.shape[1],
            self.config.factors,
        ), "Item factor dimension mismatch"

    def attach_user_item_matrix(self, user_item_matrix: csr_matrix) -> None:
        """Attach training matrix required for recommend() after loading a saved model."""
        self._user_item_matrix = user_item_matrix

    def recommend(
        self,
        user_idx: int,
        k: int,
        filter_seen: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Top-K (item_indices, scores) for one user."""
        if self._user_item_matrix is None:
            raise RuntimeError("Model not fitted; call fit() first")

        user_items = self._user_item_matrix[user_idx]
        ids, scores = self.model.recommend(
            user_idx,
            user_items,
            N=k,
            filter_already_liked_items=filter_seen,
        )
        return ids, scores

    def recommend_batch(
        self,
        user_indices: list[int],
        k: int,
        filter_seen: bool = True,
        progress: ProgressLogger | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Generate Top-K recommendations for multiple users via implicit recommend()."""
        if self._user_item_matrix is None:
            raise RuntimeError("Model not fitted; call fit() first")

        if not user_indices:
            return {}

        def _recommend_one(user_idx: int) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
            return user_idx, self.recommend(user_idx, k, filter_seen)

        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        max_workers = min(8, len(user_indices), os.cpu_count() or 4)
        if max_workers <= 1:
            for user_idx in user_indices:
                _, rec = _recommend_one(user_idx)
                results[user_idx] = rec
                if progress is not None:
                    progress.update(1)
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for user_idx, rec in executor.map(_recommend_one, user_indices):
                results[user_idx] = rec
                if progress is not None:
                    progress.update(1)
        return results

    def save(self, model_dir: str | Path) -> None:
        """Persist model, config, and training matrix metadata."""
        model_path = join_uri(model_dir, "als_model.pkl")
        payload = {
            "model": self.model,
            "config": self.config,
            "matrix_shape": (
                self._user_item_matrix.shape
                if self._user_item_matrix is not None
                else None
            ),
        }
        with open_binary(model_path, "wb") as handle:
            pickle.dump(payload, handle)
        logger.info("Saved ALS model to %s", model_path)

    @classmethod
    def load(cls, model_dir: str | Path) -> ALSModel:
        """Load a previously saved ALS model."""
        model_path = join_uri(model_dir, "als_model.pkl")
        with open_binary(model_path, "rb") as handle:
            data = pickle.load(handle)

        instance = cls(config=data["config"])
        instance.model = data["model"]
        logger.info("Loaded ALS model from %s", model_path)
        return instance
