"""
gemini.py - Gemini on Google Cloud (Vertex AI, now called Agent Platform): the model for the live site.

There is no key anywhere. The google-genai library finds the login by itself:
  - on your computer: the login you made with `gcloud auth application-default login`
  - on Cloud Run: the service account attached to the service
"""

from __future__ import annotations

from .base import LLMError


def _make_client(project, location):
    """Create the Google client for Vertex AI. The library is imported here, so it is only needed when Gemini is used."""
    from google import genai

    try:
        return genai.Client(enterprise=True, project=project, location=location)
    except TypeError:  # older versions of the library name this switch "vertexai"
        return genai.Client(vertexai=True, project=project, location=location)


class GeminiClient:
    where = "hosted"

    def __init__(self, project, location, model, client=None, max_output_tokens=2000):
        if not project:
            raise LLMError("JSJ_LLM=gemini needs JSJ_GCP_PROJECT (your Google Cloud project ID, for example jsj-intelligence).")
        self.name = model
        self.project = project
        self.location = location
        self.max_output_tokens = max_output_tokens
        self.last_usage: dict = {}  # token counts of the latest call, kept so cost can be measured later
        self._client = client if client is not None else _make_client(project, location)

    def generate(self, system, prompt):
        # The settings go in as a plain dictionary; the library accepts that, so no extra import is needed here.
        config = {"system_instruction": system, "max_output_tokens": self.max_output_tokens}
        try:
            response = self._client.models.generate_content(model=self.name, contents=prompt, config=config)
            text = (response.text or "").strip()
        except Exception as e:  # the library raises its own error types; the app only needs to know it failed
            raise LLMError(f"Could not get an answer from Gemini ({type(e).__name__}: {e}).") from e

        usage = getattr(response, "usage_metadata", None)
        self.last_usage = {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        }
        if not text:  # for example when Google's own safety filter blocked the reply
            raise LLMError("Gemini returned no answer text. Please try again.")
        return text
