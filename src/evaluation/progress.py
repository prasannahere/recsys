"""Progress logging for long-running evaluation steps."""

from __future__ import annotations

import logging
import time
from typing import Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_PROGRESS_LOG_INTERVAL_SEC = 1800
DEFAULT_USER_CHUNK_SIZE = 5000


def evaluation_progress_settings(config: dict) -> tuple[int, int]:
    """Return (log_interval_sec, user_chunk_size) from config."""
    eval_cfg = config.get("evaluation", {})
    interval = int(
        eval_cfg.get("progress_log_interval_sec", DEFAULT_PROGRESS_LOG_INTERVAL_SEC)
    )
    chunk_size = int(eval_cfg.get("user_chunk_size", DEFAULT_USER_CHUNK_SIZE))
    return interval, chunk_size


def iter_chunks(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


class ProgressLogger:
    """Emit eval progress on a time interval and at stage boundaries."""

    def __init__(
        self,
        stage: str,
        total: int,
        interval_sec: int = DEFAULT_PROGRESS_LOG_INTERVAL_SEC,
    ) -> None:
        self.stage = stage
        self.total = total
        self.interval_sec = interval_sec
        self.done = 0
        self._started = time.monotonic()
        self._last_log = self._started

    def _elapsed_min(self) -> float:
        return (time.monotonic() - self._started) / 60.0

    def log(self, message: str) -> None:
        logger.info(
            "Eval [%s] %s (elapsed %.1f min)",
            self.stage,
            message,
            self._elapsed_min(),
        )

    def start(self) -> None:
        self.log(f"starting ({self.total} items)")

    def update(self, n: int = 1, *, force: bool = False) -> None:
        self.done += n
        now = time.monotonic()
        if force or (now - self._last_log) >= self.interval_sec:
            pct = 100.0 * self.done / self.total if self.total else 100.0
            self.log(f"{self.done} / {self.total} ({pct:.1f}%)")
            self._last_log = now

    def finish(self) -> None:
        pct = 100.0 * self.done / self.total if self.total else 100.0
        self.log(f"complete {self.done} / {self.total} ({pct:.1f}%)")
