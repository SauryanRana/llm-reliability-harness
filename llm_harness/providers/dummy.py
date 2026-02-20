from __future__ import annotations

import json
import random
import time
from copy import deepcopy

from .base import ProviderResult


class DummyProvider:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, item: dict, model: str) -> ProviderResult:
        del model
        expected = _coerce_expected(item.get("expected"), str(item.get("id", "unknown")))
        actual = deepcopy(expected)

        start = time.perf_counter()
        time.sleep(self._rng.uniform(0.005, 0.020))

        if self._rng.random() < 0.10:
            actual["priority"] = _flip_priority(str(actual.get("priority", "")))
        if self._rng.random() < 0.10:
            actual["category"] = "Network"

        latency_ms = (time.perf_counter() - start) * 1000.0
        return ProviderResult(
            actual=actual,
            latency_ms=latency_ms,
            usage=None,
            status="ok",
        )


def _coerce_expected(value: object, case_id: str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Case {case_id}: expected field is not valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Case {case_id}: expected field must be a JSON object")


def _flip_priority(priority: str) -> str:
    if priority == "P2":
        return "P3"
    if priority == "P3":
        return "P2"
    return "P2"
