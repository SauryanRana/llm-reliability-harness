from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed() -> Iterator[float]:
    start = time.perf_counter()
    yield start
    _ = time.perf_counter() - start
