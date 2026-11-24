"""ollama.py - a model running on your own computer through Ollama (used for the private library)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import LLMError


class OllamaClient:
    where = "local"

    def __init__(self, url, model):
        self.url = url
        self.name = model

    def generate(self, system, prompt):
        body = {
            "model": self.name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1, "num_ctx": 4096},
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()
        except urllib.error.URLError as e:
            raise LLMError(f"Could not reach Ollama at {self.url} ({e.reason}). "
                           f"Open the Ollama app (or run: ollama serve) and try again.") from e
