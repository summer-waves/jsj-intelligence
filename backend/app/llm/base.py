"""base.py - what every language-model client must provide."""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    """The model could not be reached or returned nothing usable."""


class LLMClient(Protocol):
    name: str    # a short label, for example "claude-haiku-5-5"
    where: str   # "hosted" or "local"

    def generate(self, system: str, prompt: str) -> str:
        """Send the rules (system) and the question with its sources (prompt); return the answer text."""
        ...
