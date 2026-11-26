"""
smoke_gemini.py - one tiny call to Gemini on your Google Cloud project, to prove the setup works.

Run from the backend folder:
    python scripts/smoke_gemini.py
It sends one short question (about 20 tokens), so it costs a fraction of a cent.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # so "app" can be imported

os.environ.setdefault("JSJ_LLM", "gemini")

from app.config import Settings  # noqa: E402
from app.llm.base import LLMError  # noqa: E402
from app.llm.factory import create_llm  # noqa: E402

settings = Settings.from_env()
print(f"project={settings.gcp_project}  location={settings.gcp_location}  model={settings.gemini_model}")
try:
    llm = create_llm(settings)
    answer = llm.generate("You answer in one short sentence.", "Say hello and name the model family you are.")
except LLMError as e:
    print("FAILED:", e)
    sys.exit(1)
print("ANSWER:", answer)
print("TOKENS:", getattr(llm, "last_usage", {}))
