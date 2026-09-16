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
    _reasoning_item_to_param,
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


def _stub_reasoning(id_: str, encrypted_content: str = "opaque-blob", summary_text: str = "thinking...") -> SimpleNamespace:
    return SimpleNamespace(
        type="reasoning",
        id=id_,
        summary=[SimpleNamespace(type="summary_text", text=summary_text)],
        content=None,
        encrypted_content=encrypted_content,
        status="completed",
    )


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
    # `include=["reasoning.encrypted_content"]` is required to get
    # encrypted_content populated when context is managed manually/
    # statelessly, which is exactly what this adapter does.
    assert call_kwargs["include"] == ["reasoning.encrypted_content"]


# --------------------------------------------- reasoning-item replay (stateful)

def test_reasoning_item_replayed_immediately_before_its_function_call() -> None:
    stub_response = _stub_response(
        output=[
            _stub_reasoning("rs_1", encrypted_content="blob-1"),
            _stub_function_call("call_1", "click", '{"element": 1, "reason": "go"}'),
        ],
        status="completed",
    )
    client = _StubClient(stub_response)
    provider = OpenAIProvider(model="gpt-5.6", client=client)

    first_messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    resp = provider.complete(system="sys", messages=first_messages, tools=[])
    assert resp.stop_reason == "tool_use"
    [block] = resp.content_blocks
    assert block.type == "tool_use" and block.id == "call_1"

    # Next turn: replay the assistant turn (exactly as loop.py stores
    # `resp.content_blocks` verbatim) plus a tool_result, through the SAME
    # provider instance, and inspect what would be sent as `input`.
    next_messages = first_messages + [
        {"role": "assistant", "content": resp.content_blocks},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok", "is_error": False}],
        },
    ]
    items = provider._build_input_items(next_messages)

    call_idx = next(i for i, it in enumerate(items) if it.get("type") == "function_call")
    assert call_idx > 0
    reasoning_item = items[call_idx - 1]
    assert reasoning_item["type"] == "reasoning"
    assert reasoning_item["id"] == "rs_1"
    # encrypted_content preserved verbatim, not reconstructed/summarized.
    assert reasoning_item["encrypted_content"] == "blob-1"
    assert items[call_idx]["call_id"] == "call_1"


def test_reasoning_ordering_holds_for_two_function_calls_in_one_turn() -> None:
    stub_response = _stub_response(
        output=[
            _stub_reasoning("rs_1", encrypted_content="blob-1"),
            _stub_function_call("call_1", "click", '{"element": 1, "reason": "a"}'),
            _stub_reasoning("rs_2", encrypted_content="blob-2"),
            _stub_function_call("call_2", "click", '{"element": 2, "reason": "b"}'),
        ],
        status="completed",
    )
    client = _StubClient(stub_response)
    provider = OpenAIProvider(model="gpt-5.6", client=client)

    first_messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    resp = provider.complete(system="sys", messages=first_messages, tools=[])
    assert resp.stop_reason == "tool_use"
    assert len(resp.content_blocks) == 2

    next_messages = first_messages + [{"role": "assistant", "content": resp.content_blocks}]
    items = provider._build_input_items(next_messages)

    kinds = [(it.get("type"), it.get("id") or it.get("call_id")) for it in items if it.get("type") in ("reasoning", "function_call")]
    assert kinds == [
        ("reasoning", "rs_1"),
        ("function_call", "call_1"),
        ("reasoning", "rs_2"),
        ("function_call", "call_2"),
    ]


def test_no_reasoning_seen_matches_stateless_translation_exactly() -> None:
    # A provider instance that never recorded a reasoning item must behave
    # identically to the plain (non-reasoning) request translator - no
    # regression for models/providers that never emit reasoning items.
    provider = OpenAIProvider(model="gpt-5.6")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello there"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
            ],
        },
        {
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
    assert provider._build_input_items(messages) == _messages_to_responses_input(messages)


def test_reasoning_map_is_bounded_across_many_turns() -> None:
    provider = OpenAIProvider(model="gpt-5.6", max_reasoning_entries=5)
    for i in range(50):
        stub_response = _stub_response(
            output=[
                _stub_reasoning(f"rs_{i}", encrypted_content=f"blob-{i}"),
                _stub_function_call(f"call_{i}", "click", '{"element": 1, "reason": "x"}'),
            ],
            status="completed",
        )
        provider._client = _StubClient(stub_response)
        provider.complete(
            system="sys", messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}], tools=[]
        )

    assert len(provider._reasoning_by_call_id) <= 5
    # And it kept the most recent entries (FIFO eviction of the oldest),
    # not an arbitrary subset.
    assert "call_49" in provider._reasoning_by_call_id
    assert "call_0" not in provider._reasoning_by_call_id


def test_reasoning_content_field_survives_verbatim_alongside_encrypted_content() -> None:
    # Real GPT-5-class output can carry both `content` (reasoning_text
    # parts) and `encrypted_content` on the same item; both must round-trip
    # byte-for-byte, since this adapter never reconstructs or summarizes
    # opaque reasoning state.
    reasoning = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        summary=[SimpleNamespace(type="summary_text", text="summary text")],
        content=[SimpleNamespace(type="reasoning_text", text="the actual chain-of-thought text")],
        encrypted_content="blob-1",
        status="completed",
    )
    param = _reasoning_item_to_param(reasoning)

    assert param["type"] == "reasoning"
    assert param["id"] == "rs_1"
    assert param["summary"] == [{"type": "summary_text", "text": "summary text"}]
    assert param["content"] == [{"type": "reasoning_text", "text": "the actual chain-of-thought text"}]
    assert param["encrypted_content"] == "blob-1"
    assert param["status"] == "completed"


def test_reasoning_survives_a_message_between_it_and_the_function_call() -> None:
    # A `[reasoning, message(commentary), function_call]` turn is plausible
    # for a reasoning model. The failure this mechanism exists to prevent is
    # the API rejecting a function_call that arrives without its required
    # reasoning item, so intervening commentary must NOT discard the span.
    # Re-emitting it just before the call also preserves the order it had in
    # the original turn, so keeping it cannot itself reorder anything.
    stub_response = _stub_response(
        output=[
            _stub_reasoning("rs_1", encrypted_content="blob-1"),
            _stub_message([_stub_text_part("I will click the submit button now.")]),
            _stub_function_call("call_1", "click", '{"element": 1, "reason": "submit"}'),
        ],
        status="completed",
    )
    client = _StubClient(stub_response)
    provider = OpenAIProvider(model="gpt-5.6", client=client)

    first_messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    resp = provider.complete(system="sys", messages=first_messages, tools=[])
    assert resp.stop_reason == "tool_use"

    assert "call_1" in provider._reasoning_by_call_id

    next_messages = first_messages + [{"role": "assistant", "content": resp.content_blocks}]
    items = provider._build_input_items(next_messages)
    types = [it.get("type") for it in items]
    assert "reasoning" in types, "reasoning span must survive intervening commentary"
    # ...and must still sit immediately before the call it belongs to.
    assert types.index("reasoning") == types.index("function_call") - 1
    reasoning_item = items[types.index("reasoning")]
    assert reasoning_item.get("encrypted_content") == "blob-1"
