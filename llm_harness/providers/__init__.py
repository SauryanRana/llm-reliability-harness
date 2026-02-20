"""Model provider interfaces and implementations."""

from .base import Provider, ProviderResult
from .dummy import DummyProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider


def get_provider(
    name: str,
    *,
    base_url: str = "http://localhost:11434",
    timeout_seconds: int = 60,
    temperature: float = 0.0,
    num_predict: int = 320,
    num_ctx: int = 2048,
    json_mode: bool = False,
) -> Provider:
    if name == "dummy":
        return DummyProvider()
    if name == "ollama":
        return OllamaProvider(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            json_mode=json_mode,
        )
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name}")


__all__ = [
    "Provider",
    "ProviderResult",
    "DummyProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "get_provider",
]
