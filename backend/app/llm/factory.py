"""factory.py - picks the model client from the settings. New models (for example Gemini on Vertex AI) plug in here."""

from __future__ import annotations

from .anthropic_api import AnthropicClient
from .base import LLMClient, LLMError
from .ollama import OllamaClient


def create_llm(settings) -> LLMClient:
    backend = settings.llm_backend
    if backend == "auto":  # same rule as the original app: a key means hosted, no key means Ollama
        backend = "anthropic" if settings.anthropic_key else "ollama"
    if backend == "anthropic":
        if not settings.anthropic_key:
            raise LLMError("JSJ_LLM=anthropic needs the ANTHROPIC_API_KEY environment variable.")
        return AnthropicClient(settings.anthropic_key, settings.anthropic_model)
    if backend == "ollama":
        return OllamaClient(settings.ollama_url, settings.ollama_model)
    raise LLMError(f"Unknown JSJ_LLM value: {backend!r}. Use auto, anthropic or ollama.")
