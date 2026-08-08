"""A failed LLM call must still appear in the trace.

The spec says `steps[]` describes every LLM call the agent made, in order. A call that
was sent and then failed is still a call, so the step travels on the exception rather
than being lost with it.
"""

import httpx
import pytest

from lib import llm


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("LLMOD_API_KEY", "test-key")
    monkeypatch.setenv("LLMOD_API_BASE", "https://example.invalid/v1")


def test_missing_config_raises_without_a_step(monkeypatch):
    monkeypatch.delenv("LLMOD_API_KEY", raising=False)

    with pytest.raises(llm.LLMError) as caught:
        llm.call("Supervisor", "sys", "user")

    # No call was attempted, so there is nothing to put in the trace.
    assert caught.value.steps == []
    assert "not configured" in str(caught.value)


def test_transport_failure_carries_the_step(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)

    with pytest.raises(llm.LLMError) as caught:
        llm.call("FlightAgent", "sys prompt", "user prompt")

    (step,) = caught.value.steps
    assert step["module"] == "FlightAgent"
    assert step["prompt"] == {"system_prompt": "sys prompt", "user_prompt": "user prompt"}
    assert "connection refused" in step["response"]["error"]


def test_unparseable_json_carries_the_step(monkeypatch):
    def not_json(*args, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "sorry, here is some prose"}}]},
            request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
        )

    monkeypatch.setattr(httpx, "post", not_json)

    with pytest.raises(llm.LLMError) as caught:
        llm.call("DocumentationAgent", "sys", "user", expect_json=True)

    (step,) = caught.value.steps
    assert step["module"] == "DocumentationAgent"
    assert "expected JSON" in step["response"]["error"]


def test_successful_call_returns_text_and_a_step(monkeypatch):
    def ok(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
            request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
        )

    monkeypatch.setattr(httpx, "post", ok)
    before = dict(llm.usage)

    text, step = llm.call("Supervisor", "sys", "user")

    assert text == "hello"
    assert step["response"] == {"text": "hello"}
    assert llm.usage["calls"] == before["calls"] + 1
    assert llm.usage["prompt_tokens"] == before["prompt_tokens"] + 11
