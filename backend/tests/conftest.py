import pytest

from app.config import Settings
from app.rag.retrieval import BM25Retriever, load_chunks
from app.safety.profiles import get_profile
from app.service import AssistantService


class FakeLLM:
    """A stand-in model that returns a fixed text, so tests never call a real model."""
    name = "fake-model"
    where = "local"

    def __init__(self, reply):
        self.reply = reply

    def generate(self, system, prompt):
        return self.reply


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("JSJ_LIBRARY_MODE", "demo")
    monkeypatch.setenv("JSJ_RETRIEVAL", "bm25")
    return Settings.from_env()


@pytest.fixture
def make_service(settings):
    chunks = load_chunks(settings.chunks_path)

    def _make(reply):
        return AssistantService(settings, BM25Retriever(chunks), "bm25", FakeLLM(reply),
                                get_profile("demo"), len(chunks))
    return _make
