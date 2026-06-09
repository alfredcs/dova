"""Streaming error-handling tests for OpenAIProvider (Bedrock Mantle Responses API).

Verifies that server-side stream events are handled correctly:
- ``response.output_text.delta`` / ``response.refusal.delta`` → yielded as text
- ``error`` / ``response.failed`` → raised so LLMRouter can fall back
"""

from types import SimpleNamespace

import pytest

from dova.config.providers import (
    LLMRequest,
    ModelConfig,
    OpenAIProvider,
    ProviderConfig,
    TaskType,
)


def _make_provider() -> OpenAIProvider:
    config = ProviderConfig(
        name="openai",
        models={TaskType.CHAT: ModelConfig(model_id="openai.gpt-5.4", max_tokens=1024)},
    )
    return OpenAIProvider(
        config,
        api_key="sk-test",
        base_url="https://mantle.example/openai/v1",
        bearer_token="bearer-test",
    )


class _FakeResponses:
    """Stand-in for ``client.responses`` that replays canned stream events."""

    def __init__(self, events):
        self._events = events

    async def create(self, **_kwargs):
        async def _gen():
            for event in self._events:
                yield event

        return _gen()


def _install_stream(provider: OpenAIProvider, events) -> None:
    provider._client = SimpleNamespace(responses=_FakeResponses(events))


def _request() -> LLMRequest:
    return LLMRequest(
        task_type=TaskType.CHAT,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )


@pytest.mark.asyncio
async def test_stream_yields_text_and_refusal_deltas():
    provider = _make_provider()
    _install_stream(
        provider,
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="Hello"),
            SimpleNamespace(type="response.refusal.delta", delta=" (refused)"),
            SimpleNamespace(type="response.completed"),
        ],
    )

    chunks = [chunk async for chunk in provider.stream(_request())]

    assert chunks == ["Hello", " (refused)"]


@pytest.mark.asyncio
async def test_stream_raises_on_error_event():
    provider = _make_provider()
    _install_stream(
        provider,
        [
            SimpleNamespace(type="response.output_text.delta", delta="partial"),
            SimpleNamespace(type="error", code="rate_limit_exceeded", message="slow down"),
        ],
    )

    collected = []
    with pytest.raises(RuntimeError, match="slow down"):
        async for chunk in provider.stream(_request()):
            collected.append(chunk)

    # Partial text emitted before the error is still surfaced to the caller.
    assert collected == ["partial"]


@pytest.mark.asyncio
async def test_stream_raises_on_response_failed_event():
    provider = _make_provider()
    failed = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(error=SimpleNamespace(message="model exploded")),
    )
    _install_stream(provider, [failed])

    with pytest.raises(RuntimeError, match="model exploded"):
        async for _ in provider.stream(_request()):
            pass


@pytest.mark.asyncio
async def test_stream_response_failed_without_error_detail():
    provider = _make_provider()
    failed = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(error=None),
    )
    _install_stream(provider, [failed])

    with pytest.raises(RuntimeError, match="unknown error"):
        async for _ in provider.stream(_request()):
            pass
