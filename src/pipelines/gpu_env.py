"""GPU environment helpers for Cloud Run GPU training jobs."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def log_gpu_environment() -> None:
    """Log runtime GPU/CUDA info (Cloud Run mounts drivers at /usr/local/nvidia)."""
    logger.info("LD_LIBRARY_PATH=%s", os.environ.get("LD_LIBRARY_PATH", ""))
    logger.info(
        "torch=%s cuda_available=%s device_count=%d",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            logger.info(
                "cuda:%d name=%s total_memory_gb=%.1f",
                index,
                props.name,
                props.total_memory / (1024**3),
            )
    else:
        logger.warning(
            "CUDA is not available; PyTorch will run on CPU. "
            "For Cloud Run GPU jobs, use a CUDA-enabled torch wheel in the image."
        )


def resolve_torch_device() -> torch.device:
    """Pick training device from TORCH_DEVICE or auto-detect CUDA."""
    preference = os.environ.get("TORCH_DEVICE", "auto").strip().lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("TORCH_DEVICE=cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def skip_eval_from_env() -> bool:
    """Return True when SKIP_EVAL is set (recommended for 1h Cloud Run GPU timeout)."""
    return os.environ.get("SKIP_EVAL", "").strip().lower() in {"1", "true", "yes"}
