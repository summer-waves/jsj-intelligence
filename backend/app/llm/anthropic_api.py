"""anthropic_api.py - Claude through the Anthropic API (the hosted model used by the live demo)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import LLMError


class AnthropicClient:
    where = "hosted"

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.name = model

    def generate(self, system, prompt):
        body = {
            "model": self.name,
            "max_tokens": 2000,  # room for the model's own reasoning plus a short answer
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                blocks = json.loads(resp.read().decode("utf-8"))["content"]
        except urllib.error.URLError as e:
            raise LLMError(f"Could not get an answer from the hosted model ({e}).") from e
        # The reply is a list of blocks. Some are the model's reasoning ("thinking"); only "text" blocks are the answer.
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        if not text:
            raise LLMError("The hosted model returned no answer text. Please try again.")
        return text
