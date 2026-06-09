"""Factory for selecting an LLM provider."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from providers.base_provider import BaseProvider
from providers.ollama_provider import OllamaProvider, get_default_model, get_ollama_url


def get_llm_provider(
    provider_name: str | None = None,
    model: str | None = None,
) -> BaseProvider:
    """
    Return the configured LLM provider.

    Default: Ollama (local, no API key).
    Set LLM_PROVIDER=openai in .env to use OpenAI instead.
    """
    name = (provider_name or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()

    if name == "openai":
        from providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model) if model else OpenAIProvider()

    return OllamaProvider(
        model=model or get_default_model(),
        base_url=get_ollama_url(),
    )


def provider_label(provider: BaseProvider) -> str:
    """Human-readable label for status messages."""
    if isinstance(provider, OllamaProvider):
        return f"Ollama ({provider.model})"
    return "OpenAI"
