from __future__ import annotations

import logging
import time
from contextlib import ContextDecorator
from types import TracebackType

LATENCY: dict[str, list[float]] = {}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class Timer(ContextDecorator):
    def __init__(self, name: str, target: dict[str, list[float]] | None = None) -> None:
        self.name = name
        self.target = LATENCY if target is None else target
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.target.setdefault(self.name, []).append(self.elapsed_ms)
        return False
