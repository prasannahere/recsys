"""FAISS index for Hybrid item retrieval."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import faiss
import numpy as np

from src.io.storage import is_gcs_uri, open_binary

logger = logging.getLogger(__name__)


class ItemVectorIndex:
    """Inner-product FAISS index over L2-normalized item embeddings."""

    def __init__(self, embeddings: np.ndarray, item_indices: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if len(item_indices) != embeddings.shape[0]:
            raise ValueError("item_indices length must match number of embeddings")

        self.item_indices = item_indices.astype(np.int64)
        vectors = embeddings.astype(np.float32)
        faiss.normalize_L2(vectors)

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        logger.info(
            "Built FAISS index with %d vectors (dim=%d)",
            vectors.shape[0],
            vectors.shape[1],
        )

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (item_indices, scores) for each query vector."""
        queries = query_vectors.astype(np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        faiss.normalize_L2(queries)
        scores, positions = self._index.search(queries, k)
        item_indices = self.item_indices[positions]
        return item_indices, scores

    def save(self, path: str | Path) -> None:
        path_str = str(path)
        faiss_path = f"{path_str}.faiss"
        indices_path = f"{path_str}.item_indices.npy"
        index_bytes = faiss.serialize_index(self._index)
        indices_buffer = io.BytesIO()
        np.save(indices_buffer, self.item_indices)

        if is_gcs_uri(path_str):
            with open_binary(faiss_path, "wb") as handle:
                handle.write(index_bytes)
            indices_buffer.seek(0)
            with open_binary(indices_path, "wb") as handle:
                handle.write(indices_buffer.read())
        else:
            path_obj = Path(path_str)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, faiss_path)
            np.save(indices_path, self.item_indices)
        logger.info("Saved FAISS index to %s", path_str)

    @classmethod
    def load(cls, path: str | Path) -> ItemVectorIndex:
        path_str = str(path)
        faiss_path = f"{path_str}.faiss"
        indices_path = f"{path_str}.item_indices.npy"
        if is_gcs_uri(path_str):
            with open_binary(faiss_path, "rb") as handle:
                index = faiss.deserialize_index(handle.read())
            with open_binary(indices_path, "rb") as handle:
                item_indices = np.load(handle)
        else:
            index = faiss.read_index(faiss_path)
            item_indices = np.load(indices_path)
        instance = cls.__new__(cls)
        instance._index = index
        instance.item_indices = item_indices.astype(np.int64)
        logger.info("Loaded FAISS index from %s", path_str)
        return instance
