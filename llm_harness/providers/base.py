from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ProviderResult:
    actual: dict | None
    latency_ms: float
    raw_text: str | None = None
    usage: dict[str, Any] | None = None
    status: str = "ok"
    error_type: str | None = None
    error_msg: str | None = None
    prompt_chars: int | None = None
    response_chars: int | None = None


class Provider(Protocol):
    def generate(self, item: dict, model: str) -> ProviderResult:
        """Return model output and provider metadata."""
        ...
