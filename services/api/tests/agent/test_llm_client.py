from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ethiclens_api.agent.llm_client import (
    FallbackLLMClient,
    LLMCallError,
    build_default_client,
    build_default_client_or_none,
    complete_json,
)


class _Answer(BaseModel):
    text: str


class _StubClient:
    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls = 0

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def test_complete_json_returns_parsed_schema_on_first_success() -> None:
    client = _StubClient(response=json.dumps({"text": "hello"}))
    result = complete_json(client, "system", "prompt", _Answer)
    assert result.text == "hello"
    assert client.calls == 1


def test_complete_json_retries_once_then_raises_llmcallerror() -> None:
    client = _StubClient(response="not valid json")
    with pytest.raises(LLMCallError):
        complete_json(client, "system", "prompt", _Answer)
    assert client.calls == 2  # the retry-once-then-raise design from the docstring


def test_complete_json_succeeds_on_the_retry() -> None:
    responses = iter(["not valid json", json.dumps({"text": "fixed"})])

    class _RetryClient:
        def complete(self, system: str, prompt: str) -> str:
            return next(responses)

    result = complete_json(_RetryClient(), "system", "prompt", _Answer)
    assert result.text == "fixed"


def test_complete_json_strips_markdown_code_fences() -> None:
    client = _StubClient(response=f"```json\n{json.dumps({'text': 'fenced'})}\n```")
    result = complete_json(client, "system", "prompt", _Answer)
    assert result.text == "fenced"


def test_fallback_client_tries_next_provider_on_failure() -> None:
    failing = _StubClient(error=RuntimeError("boom"))
    working = _StubClient(response="ok")
    client = FallbackLLMClient([failing, working])
    assert client.complete("s", "p") == "ok"
    assert failing.calls == 1
    assert working.calls == 1


def test_fallback_client_does_not_try_later_providers_on_success() -> None:
    first = _StubClient(response="first")
    second = _StubClient(response="second")
    client = FallbackLLMClient([first, second])
    assert client.complete("s", "p") == "first"
    assert second.calls == 0


def test_fallback_client_raises_with_every_providers_error_when_all_fail() -> None:
    """Regression test: the raised error must name every provider's failure, not just
    the last one — otherwise a dead primary-provider key is invisible, since the
    exception text would only ever mention whichever fallback failed last."""
    groq_like = _StubClient(error=RuntimeError("groq rate limited"))
    gemini_like = _StubClient(error=RuntimeError("gemini quota exceeded"))
    client = FallbackLLMClient([groq_like, gemini_like])
    with pytest.raises(LLMCallError) as exc_info:
        client.complete("s", "p")
    message = str(exc_info.value)
    assert "groq rate limited" in message
    assert "gemini quota exceeded" in message


def test_fallback_client_requires_at_least_one_client() -> None:
    with pytest.raises(ValueError):
        FallbackLLMClient([])


def test_fallback_client_logs_each_providers_failure(caplog) -> None:
    groq_like = _StubClient(error=RuntimeError("groq rate limited"))
    gemini_like = _StubClient(error=RuntimeError("gemini quota exceeded"))
    client = FallbackLLMClient([groq_like, gemini_like])
    with (
        caplog.at_level("WARNING", logger="ethiclens.agent.llm_client"),
        pytest.raises(LLMCallError),
    ):
        client.complete("s", "p")
    messages = [record.message for record in caplog.records]
    assert any("groq rate limited" in m for m in messages)
    assert any("gemini quota exceeded" in m for m in messages)


def test_fallback_client_logs_success(caplog) -> None:
    client = FallbackLLMClient([_StubClient(response="ok")])
    with caplog.at_level("INFO", logger="ethiclens.agent.llm_client"):
        client.complete("s", "p")
    assert any("succeeded" in record.message for record in caplog.records)


def test_build_default_client_or_none_without_any_key_returns_none() -> None:
    assert build_default_client_or_none(None, None) is None


def test_build_default_client_without_any_key_raises() -> None:
    with pytest.raises(ValueError):
        build_default_client(None, None)


def test_groq_client_passes_timeout_to_sdk() -> None:
    pytest.importorskip("groq")
    from ethiclens_api.agent.llm_client import GroqClient

    client = GroqClient("fake-key", timeout=5.0)
    assert client._client.timeout == 5.0


def test_gemini_client_passes_timeout_to_sdk() -> None:
    pytest.importorskip("google.genai")
    from ethiclens_api.agent.llm_client import GeminiClient

    client = GeminiClient("fake-key", timeout=5.0)
    assert client._client._api_client._http_options.timeout == 5000  # ms


def test_build_default_client_or_none_with_groq_key_only() -> None:
    pytest.importorskip("groq")
    client = build_default_client_or_none("fake-groq-key", None)
    assert client is not None


def test_groq_client_requests_native_json_mode() -> None:
    """Every caller goes through complete_json, which always wants a single JSON
    object back — native JSON mode avoids markdown-fence/prose parse failures that
    would otherwise burn a retry (a second call) against the daily quota."""
    pytest.importorskip("groq")
    from ethiclens_api.agent.llm_client import GroqClient

    client = GroqClient("fake-key")
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class _Choice:
                message = type("Msg", (), {"content": "{}"})()

            return type("Resp", (), {"choices": [_Choice()]})()

    client._client.chat.completions = _FakeCompletions()
    client.complete("system", "prompt")
    assert captured["response_format"] == {"type": "json_object"}


def test_gemini_client_requests_native_json_mode() -> None:
    pytest.importorskip("google.genai")
    from ethiclens_api.agent.llm_client import GeminiClient

    client = GeminiClient("fake-key")
    captured = {}

    def _fake_generate_content(**kwargs):
        captured.update(kwargs)
        return type("Resp", (), {"text": "{}"})()

    client._client.models.generate_content = _fake_generate_content
    client.complete("system", "prompt")
    assert captured["config"]["response_mime_type"] == "application/json"
