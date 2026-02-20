from __future__ import annotations

from .base import ProviderResult


class OpenAIProvider:
    def generate(self, item: dict, model: str) -> ProviderResult:
        del item, model
        raise NotImplementedError("OpenAI provider is not implemented yet")
