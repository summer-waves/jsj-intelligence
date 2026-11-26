"""Tests for the Gemini connector. A fake Google client is used, so nothing here touches the cloud or costs money."""

import pytest

from app.config import Settings
from app.llm import factory
from app.llm.base import LLMError
from app.llm.gemini import GeminiClient


class FakeUsage:
    prompt_token_count = 1200
    candidates_token_count = 80
    thoughts_token_count = 0


class FakeResponse:
    def __init__(self, text, usage=FakeUsage()):
        self.text = text
        self.usage_metadata = usage


class FakeModels:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return self.reply


class FakeGoogleClient:
    def __init__(self, reply=None, error=None):
        self.models = FakeModels(reply, error)


def make(reply=None, error=None):
    fake = FakeGoogleClient(reply, error)
    return GeminiClient("jsj-intelligence", "global", "gemini-3.5-flash-lite", client=fake), fake


def test_returns_text_and_sends_rules_and_prompt():
    llm, fake = make(FakeResponse("  Yoga is generally safe [1].  "))
    assert llm.generate("RULES", "QUESTION") == "Yoga is generally safe [1]."
    call = fake.models.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite" and call["contents"] == "QUESTION"
    assert call["config"]["system_instruction"] == "RULES"


def test_records_token_usage():
    llm, _ = make(FakeResponse("ok [1]."))
    llm.generate("s", "p")
    assert llm.last_usage == {"input_tokens": 1200, "output_tokens": 80, "thinking_tokens": 0}


def test_usage_missing_is_not_an_error():
    llm, _ = make(FakeResponse("ok [1].", usage=None))
    assert llm.generate("s", "p") == "ok [1]."
    assert llm.last_usage["input_tokens"] is None


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_answer_is_an_error(empty):
    llm, _ = make(FakeResponse(empty))
    with pytest.raises(LLMError):
        llm.generate("s", "p")


def test_library_error_becomes_llm_error():
    llm, _ = make(error=RuntimeError("403 permission denied"))
    with pytest.raises(LLMError) as e:
        llm.generate("s", "p")
    assert "permission denied" in str(e.value)


def test_missing_project_is_a_clear_error():
    with pytest.raises(LLMError) as e:
        GeminiClient(None, "global", "gemini-3.5-flash-lite", client=FakeGoogleClient())
    assert "JSJ_GCP_PROJECT" in str(e.value)


def test_factory_builds_gemini_from_settings(monkeypatch):
    monkeypatch.setenv("JSJ_LLM", "gemini")
    monkeypatch.setenv("JSJ_GCP_PROJECT", "jsj-intelligence")
    monkeypatch.setenv("JSJ_GEMINI_MODEL", "gemini-3.8-flash")
    seen = {}

    class StubClient:
        def __init__(self, project, location, model):
            seen.update(project=project, location=location, model=model)

    monkeypatch.setattr(factory, "GeminiClient", StubClient)
    factory.create_llm(Settings.from_env())
    assert seen == {"project": "jsj-intelligence", "location": "global", "model": "gemini-3.8-flash"}


def test_auto_never_picks_gemini(monkeypatch):
    monkeypatch.setenv("JSJ_LLM", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("JSJ_GCP_PROJECT", "jsj-intelligence")
    assert type(factory.create_llm(Settings.from_env())).__name__ == "OllamaClient"


def test_unknown_backend_message_lists_gemini(monkeypatch):
    monkeypatch.setenv("JSJ_LLM", "nope")
    with pytest.raises(LLMError) as e:
        factory.create_llm(Settings.from_env())
    assert "gemini" in str(e.value)
