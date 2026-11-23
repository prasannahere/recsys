"""Contrastive losses for Hybrid retrieval."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def in_batch_softmax_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Softmax contrastive loss with in-batch negatives."""
    logits = user_embeddings @ item_embeddings.T
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
