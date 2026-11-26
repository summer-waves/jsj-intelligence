"""factory.py - picks the model client from the settings. New models plug in here."""

from __future__ import annotations

from .anthropic_api import AnthropicClient
from .base import LLMClient, LLMError
from .gemini import GeminiClient
from .ollama import OllamaClient


def create_llm(settings) -> LLMClient:
    backend = settings.llm_backend
    if backend == "auto":  # same rule as before: a key means hosted Claude, no key means Ollama. Gemini is never picked silently.
        backend = "anthropic" if settings.anthropic_key else "ollama"
    if backend == "anthropic":
        if not settings.anthropic_key:
            raise LLMError("JSJ_LLM=anthropic needs the ANTHROPIC_API_KEY environment variable.")
        return AnthropicClient(settings.anthropic_key, settings.anthropic_model)
    if backend == "gemini":
        return GeminiClient(settings.gcp_project, settings.gcp_location, settings.gemini_model)
    if backend == "ollama":
        return OllamaClient(settings.ollama_url, settings.ollama_model)
    raise LLMError(f"Unknown JSJ_LLM value: {backend!r}. Use auto, anthropic, gemini or ollama.")
