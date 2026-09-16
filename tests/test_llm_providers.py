"""Tests for the OpenAI provider adapter (hands.agent.llm.OpenAIProvider)
and provider selection (hands.config.resolve_provider / model_name /
build_provider).

No network calls anywhere in this file. Every provider under test is either
constructed with a stub `client=` object (so `.complete()` never reaches
`openai`/`anthropic`) or exercised through its private translation
functions directly. `AnthropicProvider`/`OpenAIProvider` both construct
lazily (the SDK client is a `@property`, only touched on first `.complete`
call), so simply importing `hands.agent.llm` or instantiating either
provider is safe with zero API keys configured - a couple of the tests
below pin that down explicitly.

An autouse fixture scrubs ANTHROPIC_API_KEY / OPENAI_API_KEY /
HANDS_PROVIDER / HANDS_MODEL from the environment before every test in this
module, so results never depend on the developer's real shell environment
or a `.env` file (this file never calls `hands.config.load_env`).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from hands.agent.llm import (
    AnthropicProvider,
    FakeLLM,
    OpenAIProvider,
    _FakeTextBlock,
    _FakeToolUseBlock,
    _messages_to_responses_input,
    _response_to_blocks,
    _tools_to_responses,
)
from hands.agent.tools import TOOL_DEFS
from hands.config import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    build_provider,
    model_name,
    resolve_provider,
)

_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HANDS_PROVIDER", "HANDS_MODEL", "OPENAI_BASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------- stub helpers

class _StubResponses:
    """Stands in for `client.responses` - records the kwargs it was called
    with and returns a pre-built stub Response, so `.complete()` can be
    exercised end-to-end with no network access."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _StubClient:
    def __init__(self, response: Any) -> None:
        self.responses = _StubResponses(response)


def _stub_function_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)


def _stub_text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="output_text", text=text)


def _stub_refusal_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="refusal", refusal=text)


def _stub_message(parts: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(type="message", content=parts)


def _stub_response(
    output: list[Any],
    status: str = "completed",
    usage: Any = None,
    id_: str = "resp_1",
    model: str = "gpt-5.6",
    incomplete_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        output=output, status=status, usage=usage, id=id_, model=model, incomplete_details=incomplete_details
    )


# ------------------------------------------------------------ 1. tool schema

def test_tool_translation_is_flat_responses_shape() -> None:
    click_def = next(t for t in TOOL_DEFS if t["name"] == "click")
    [flat] = _tools_to_responses([click_def])

    assert flat["type"] == "function"
    assert flat["name"] == "click"
    assert flat["description"] == click_def["description"]
    assert flat["parameters"] == click_def["input_schema"]
    assert flat["strict"] == click_def["strict"]
    # NOT the older Chat Completions shape: no nested "function" key, and no
    # "input_schema" key surviving the translation.
    assert "function" not in flat
    assert "input_schema" not in flat


# --------------------------------------------------------- 2. message shapes

def test_message_translation_text_image_tool_use_tool_result() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello there"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
                },
            ],
        },
        {
            # The loop stores `resp.content_blocks` verbatim - real SDK
            # objects or our own duck-typed blocks, never plain dicts - so
            # an assistant turn is built from a block object, not a dict.
            "role": "assistant",
            "content": [_FakeToolUseBlock(id="call_1", name="click", input={"element": 3, "reason": "clicking"})],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok, url now https://x/", "is_error": False}
            ],
        },
    ]

    items = _messages_to_responses_input(messages)
    assert len(items) == 3

    user_item = items[0]
    assert user_item["role"] == "user"
    assert user_item["content"][0] == {"type": "input_text", "text": "hello there"}
    assert user_item["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,QUJD"}

    call_item = items[1]
    assert call_item["type"] == "function_call"
    assert call_item["call_id"] == "call_1"
    assert call_item["name"] == "click"
    assert json.loads(call_item["arguments"]) == {"element": 3, "reason": "clicking"}

    result_item = items[2]
    assert result_item["type"] == "function_call_output"
    assert result_item["call_id"] == "call_1"
    assert result_item["output"] == "ok, url now https://x/"


# --------------------------------------------------------- 3. error marking

def test_error_tool_result_is_marked_in_output_text() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_9", "content": "element 7 not found", "is_error": True}
            ],
        }
    ]
    [item] = _messages_to_responses_input(messages)
    assert item["type"] == "function_call_output"
    # The Responses wire format has no is_error flag on a function output,
    # so the failure has to be legible from the text itself.
    assert item["output"].startswith("ERROR:")
    assert "element 7 not found" in item["output"]


# ----------------------------------------------------- 4. response -> blocks

def test_response_translation_function_call_becomes_tool_use_block() -> None:
    resp = _stub_response(
        output=[_stub_function_call("call_42", "click", '{"element": 5, "reason": "go"}')],
        status="completed",
    )
    blocks, stop_reason = _response_to_blocks(resp)

    assert stop_reason == "tool_use"
    [block] = blocks
    assert block.type == "tool_use"
    assert block.id == "call_42"
    assert block.name == "click"
    assert isinstance(block.input, dict)  # parsed via json.loads, never a raw string
    assert block.input == {"element": 5, "reason": "go"}


# --------------------------------------------------------- 5. stop_reason map

def test_stop_reason_tool_use_when_function_calls_present() -> None:
    resp = _stub_response(
        output=[
            _stub_message([_stub_text_part("thinking out loud")]),
            _stub_function_call("call_1", "wait", '{"ms": 500, "reason": "let it settle"}'),
        ],
        status="completed",
    )
    _, stop_reason = _response_to_blocks(resp)
    assert stop_reason == "tool_use"


def test_stop_reason_end_turn_for_completed_text_only_response() -> None:
    resp = _stub_response(output=[_stub_message([_stub_text_part("all done")])], status="completed")
    blocks, stop_reason = _response_to_blocks(resp)
    assert stop_reason == "end_turn"
    [block] = blocks
    assert block.type == "text"
    assert block.text == "all done"


def test_stop_reason_max_tokens_for_incomplete_response() -> None:
    resp = _stub_response(
        output=[_stub_message([_stub_text_part("cut off mid-")])],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    _, stop_reason = _response_to_blocks(resp)
    assert stop_reason == "max_tokens"


def test_stop_reason_refusal() -> None:
    # OpenAI reports status == "completed" even for a refusal turn, so the
    # adapter has to inspect content, not just status, to catch this.
    resp = _stub_response(output=[_stub_message([_stub_refusal_part("cannot help with that")])], status="completed")
    _, stop_reason = _response_to_blocks(resp)
    assert stop_reason == "refusal"


# --------------------------------------------------- 6. resolve_provider precedence

def test_resolve_provider_explicit_beats_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    assert resolve_provider("anthropic") == "anthropic"


def test_resolve_provider_env_beats_key_autodetect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")
    monkeypatch.setenv("HANDS_PROVIDER", "anthropic")
    assert resolve_provider() == "anthropic"


def test_resolve_provider_autodetect_openai_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")
    assert resolve_provider() == "openai"


def test_resolve_provider_autodetect_prefers_anthropic_when_both_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")
    assert resolve_provider() == "anthropic"


def test_resolve_provider_defaults_to_anthropic_with_no_keys() -> None:
    assert resolve_provider() == "anthropic"


# ------------------------------------------------------- model_name defaults

def test_model_name_no_provider_preserves_old_default() -> None:
    assert model_name() == DEFAULT_MODEL


def test_model_name_openai_default() -> None:
    assert model_name(provider="openai") == DEFAULT_OPENAI_MODEL


def test_model_name_explicit_always_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_MODEL", "should-be-ignored")
    assert model_name("my-model", provider="openai") == "my-model"


# ------------------------------------------------------ 7. build_provider errors

def test_build_provider_openai_missing_key_names_the_env_var() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_provider("openai")
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_build_provider_anthropic_missing_key_names_the_env_var() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_provider("anthropic")
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_build_provider_fake_script_bypasses_key_requirement() -> None:
    provider = build_provider("openai", fake_script=[{"text": "hi"}])
    assert isinstance(provider, FakeLLM)


# --------------------------------------------------------- 8. round-trip

def test_round_trip_response_blocks_back_through_request_translator() -> None:
    resp = _stub_response(output=[_stub_function_call("call_7", "select_option", '{"element": 2, "option": "Checking", "reason": "pick account type"}')])
    blocks, stop_reason = _response_to_blocks(resp)
    assert stop_reason == "tool_use"

    # Feed the translated blocks back in, exactly as loop.py does:
    # messages.append({"role": "assistant", "content": resp.content_blocks})
    messages = [{"role": "assistant", "content": blocks}]
    items = _messages_to_responses_input(messages)

    [call_item] = items
    assert call_item["type"] == "function_call"
    assert call_item["call_id"] == "call_7"
    assert call_item["name"] == "select_option"
    assert json.loads(call_item["arguments"]) == {"element": 2, "option": "Checking", "reason": "pick account type"}


def test_round_trip_preserves_order_when_assistant_turn_mixes_text_and_tool_use() -> None:
    # A response with narration BEFORE a tool call: [text, tool_use]. The
    # request translator must not reorder these when the turn is replayed.
    resp = _stub_response(
        output=[
            _stub_message([_stub_text_part("I will click the submit button now.")]),
            _stub_function_call("call_8", "click", '{"element": 9, "reason": "submit"}'),
        ],
        status="completed",
    )
    blocks, stop_reason = _response_to_blocks(resp)
    assert stop_reason == "tool_use"
    assert [b.type for b in blocks] == ["text", "tool_use"]

    messages = [{"role": "assistant", "content": blocks}]
    items = _messages_to_responses_input(messages)

    assert len(items) == 2
    assert items[0] == {"role": "assistant", "content": "I will click the submit button now."}
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_8"


# ------------------------------------------------- lazy construction / stub client

def test_openai_provider_constructs_with_no_api_key_present() -> None:
    # Must not raise and must not touch the network - the client is a lazy
    # property, only built on first `.complete()` call.
    provider = OpenAIProvider(model="gpt-5.6")
    assert provider.model == "gpt-5.6"


def test_openai_provider_complete_end_to_end_with_stub_client() -> None:
    stub_response = _stub_response(
        output=[_stub_function_call("call_1", "click", '{"element": 1, "reason": "go"}')],
        status="completed",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        id_="resp_abc",
        model="gpt-5.6",
    )
    client = _StubClient(stub_response)
    provider = OpenAIProvider(model="gpt-5.6", client=client)

    messages = [{"role": "user", "content": [{"type": "text", "text": "GOAL: do the thing"}]}]
    resp = provider.complete(system="be helpful", messages=messages, tools=[TOOL_DEFS[0]])

    assert resp.stop_reason == "tool_use"
    assert resp.usage == {"input_tokens": 100, "output_tokens": 20}
    assert resp.message_id == "resp_abc"
    assert resp.model == "gpt-5.6"
    [block] = resp.content_blocks
    assert block.type == "tool_use"
    assert block.name == "click"

    # And the request side was translated too: instructions carries
    # `system`, tools are flat-shaped, input carries the user text.
    [call_kwargs] = client.responses.calls
    assert call_kwargs["instructions"] == "be helpful"
    assert call_kwargs["tools"][0]["type"] == "function"
    assert call_kwargs["input"][0]["role"] == "user"
