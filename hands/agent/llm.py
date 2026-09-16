"""LLM provider abstraction for the discovery loop.

`AnthropicProvider` talks to the real Claude API (never invoked by this
repo's test suite - no API key is configured here). `OpenAIProvider` talks
to the real OpenAI Responses API and is a second, drop-in implementation of
the same `LLMProvider` protocol. `FakeLLM` replays a scripted sequence of
tool calls / text turns, duck-typed to look like SDK content blocks
(`.type`, `.name`, `.input`, `.id`, `.text`), so the whole loop is
exercisable deterministically with no network access and no key.

Adapter boundary (deliberate design choice): `hands.agent.loop` owns all
agent semantics and is written entirely around Anthropic-shaped
`messages`/`tools` and duck-typed response blocks - it never branches on
which provider is in play. Each provider class is the ONLY place that knows
its own vendor's wire format: `OpenAIProvider` translates the Anthropic-
shaped conversation the loop maintains into OpenAI Responses API `input`
items on the way in, and translates `response.output` back into the same
duck-typed blocks `FakeLLM` already defines on the way out. This keeps the
loop provider-agnostic at zero cost to either provider - each adapter pays
the translation cost once, at its own boundary, instead of leaking vendor
shapes into the loop.

Known translation gap (OpenAI side only, documented rather than "solved"):
GPT-5-class models (the OpenAI default here) can emit opaque `reasoning`
items in `response.output` alongside `function_call` items, and the
Responses API expects those `reasoning` items to be replayed back on the
following turn's `input` when they preceded a still-pending function call.
`LLMResponse`/the Anthropic-shaped `messages` list the loop maintains have
no slot for opaque provider state (and `loop.py` is explicitly out of
scope to change), so `OpenAIProvider` does not attempt to thread reasoning
items through; it only forwards `function_call` and text/refusal content.
In practice this means a multi-tool-call OpenAI session may not get the
same "show your reasoning between calls" benefit a native Responses-API
loop would - the discovery loop still works (tool calls and results still
round-trip correctly), it just does not preserve that opaque state.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass
class LLMResponse:
    content_blocks: list[Any]
    stop_reason: str
    raw: Any
    model: str
    usage: dict[str, int]
    message_id: Optional[str]
    latency_ms: int
    stop_details: Any = None


class LLMProvider(Protocol):
    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


@dataclass
class _FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"
    text: str = ""


@dataclass
class _FakeTextBlock:
    text: str
    id: str = ""
    name: str = ""
    input: Optional[dict[str, Any]] = None
    type: str = "text"


class AnthropicProvider:
    """Real Claude API provider (anthropic>=1.6.0).

    `client` is constructed lazily on first use via `anthropic.Anthropic()`
    (reads ANTHROPIC_API_KEY from the environment) so this module stays
    importable even where the SDK cannot authenticate - the constructor
    itself never touches the network.
    """

    def __init__(self, model: str, max_tokens: int = 16000, client: Any = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # local import: only needed when a real call is made

            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        t0 = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        usage_obj = getattr(response, "usage", None)
        usage: dict[str, int] = {}
        if usage_obj is not None:
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                val = getattr(usage_obj, key, None)
                if val is not None:
                    usage[key] = val

        return LLMResponse(
            content_blocks=list(response.content),
            stop_reason=response.stop_reason,
            raw=response,
            model=getattr(response, "model", self.model),
            usage=usage,
            message_id=getattr(response, "id", None),
            latency_ms=latency_ms,
            stop_details=getattr(response, "stop_details", None),
        )


def _field(item: Any, key: str, default: Any = None) -> Any:
    """Read `key` off `item`, which may be a plain dict (every block the
    loop builds itself: user text/image turns, tool_result turns) or a
    duck-typed object with attributes (every assistant turn, since the loop
    stores `resp.content_blocks` - real SDK objects or our own
    `_FakeToolUseBlock`/`_FakeTextBlock` instances - verbatim). Both shapes
    have to be supported because `loop.py` mixes them across turns and is
    out of scope to change."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped `{"name","description","input_schema","strict"}` ->
    the FLAT Responses API tool shape `{"type":"function","name",
    "description","parameters","strict"}`. Note this is deliberately NOT
    nested under a `"function"` key - that is the older Chat Completions
    shape, which the Responses API does not accept.

    Not verified strict-clean: `done`'s `outputs` property is a free-form
    string-valued map (`"additionalProperties": {"type": "string"}`), and
    OpenAI strict-mode function calling requires every object schema
    (including nested ones) to set `additionalProperties: false`. We pass
    `strict` through verbatim rather than silently rewriting the schema, so
    behaviour matches whatever the tool def declares; a real OpenAI call
    with `done`'s schema under `strict: true` may be rejected by the API.
    This is a known limitation, not something this adapter can safely paper
    over without changing `tools.py`, which is out of scope here.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
                "strict": t.get("strict", False),
            }
        )
    return out


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped `messages` -> Responses API `input` items.

    - A user turn's `{"type":"text","text":...}` block -> `{"type":
      "input_text","text":...}`.
    - A user turn's `{"type":"image","source":{"type":"base64",
      "media_type":...,"data":...}}` block -> `{"type":"input_image",
      "image_url":"data:<media_type>;base64,<data>"}`.
    - An assistant turn's `tool_use` block -> a standalone `{"type":
      "function_call","call_id":...,"name":...,"arguments":<json string>}`
      item. Assistant text becomes an assistant message item.
    - A user turn's `tool_result` block -> a standalone `{"type":
      "function_call_output","call_id":<tool_use_id>,"output":<string>}`
      item. OpenAI has no `is_error` flag on a function output, so an error
      result is marked directly in the output text (prefixed `ERROR: `) -
      otherwise the model has no way to tell a failed action from a
      successful one.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            items.append({"role": role, "content": content})
            continue

        if role == "user":
            user_parts: list[dict[str, Any]] = []
            for block in content or []:
                btype = _field(block, "type")
                if btype == "tool_result":
                    result_content = _field(block, "content")
                    output_text = result_content if isinstance(result_content, str) else str(result_content)
                    if _field(block, "is_error", False):
                        output_text = f"ERROR: {output_text}"
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": _field(block, "tool_use_id"),
                            "output": output_text,
                        }
                    )
                elif btype == "text":
                    user_parts.append({"type": "input_text", "text": _field(block, "text", "")})
                elif btype == "image":
                    source = _field(block, "source") or {}
                    media_type = _field(source, "media_type", "image/png")
                    data = _field(source, "data", "")
                    user_parts.append(
                        {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}
                    )
            if user_parts:
                items.append({"role": "user", "content": user_parts})

        elif role == "assistant":
            # Appended in original block order (not accumulated-then-
            # flushed) so a turn that mixes narration and a tool call, e.g.
            # `[text_block, tool_use_block]`, keeps the model's text before
            # the function_call in the replayed `input` - reversing that
            # order would misrepresent what the model actually did.
            for block in content or []:
                btype = _field(block, "type")
                if btype == "tool_use":
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": _field(block, "id"),
                            "name": _field(block, "name"),
                            "arguments": json.dumps(_field(block, "input") or {}),
                        }
                    )
                elif btype == "text":
                    text = _field(block, "text", "")
                    if text:
                        # Plain string content, not a list of `output_text`
                        # parts: the Responses API's EasyInputMessageParam
                        # (what an "assistant"-role *input* item accepts)
                        # takes `content: str | list[input_text/input_image/
                        # input_file]` - `output_text` parts belong to
                        # ResponseOutputMessageParam (an *output* item,
                        # which also requires `id`/`status` we don't have
                        # here), not to something we're allowed to send back
                        # as input.
                        items.append({"role": "assistant", "content": text})

    return items


def _response_to_blocks(response: Any) -> tuple[list[Any], str]:
    """`response.output` -> (Anthropic-shaped duck-typed blocks, stop_reason).

    Reuses `_FakeToolUseBlock`/`_FakeTextBlock` (already defined above by
    `FakeLLM`) rather than inventing a parallel block hierarchy, so
    `loop.py`'s `getattr(b, "type"/"name"/"input"/"id"/"text", ...)` calls
    work identically regardless of which provider produced them.

    stop_reason mapping (the same vocabulary `loop.py` already switches on):
      - any `function_call` item present         -> "tool_use"
      - a `refusal` content part present          -> "refusal"
      - response.status == "completed"            -> "end_turn"
      - response.status == "incomplete" (length)   -> "max_tokens"
      - anything else                              -> pass the raw status
                                                       through (loop.py logs
                                                       it as an "unhandled_
                                                       stop_reason_<x>" clean
                                                       termination rather
                                                       than crashing)

    Refusal is checked BEFORE "completed": OpenAI reports `status ==
    "completed"` even when the model's turn produced a refusal (the request
    itself completed fine) - reading the spec's mapping rules as a strict
    top-to-bottom if/elif chain would make the refusal branch unreachable,
    since a refusal response is also a completed one. Inspecting the output
    content is the only reliable way to tell the two apart, so this adapter
    checks for refusal content before checking `status == "completed"`.
    """
    blocks: list[Any] = []
    has_function_call = False
    has_refusal = False

    for item in getattr(response, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "function_call":
            has_function_call = True
            arguments = getattr(item, "arguments", None) or "{}"
            try:
                parsed_input = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                parsed_input = {}
            blocks.append(
                _FakeToolUseBlock(
                    id=getattr(item, "call_id", "") or "",
                    name=getattr(item, "name", "") or "",
                    input=parsed_input,
                )
            )
        elif itype == "message":
            for part in getattr(item, "content", None) or []:
                ptype = getattr(part, "type", None)
                if ptype == "output_text":
                    blocks.append(_FakeTextBlock(text=getattr(part, "text", "") or ""))
                elif ptype == "refusal":
                    has_refusal = True
                    blocks.append(_FakeTextBlock(text=getattr(part, "refusal", "") or ""))

    status = getattr(response, "status", None)
    if has_function_call:
        stop_reason = "tool_use"
    elif has_refusal:
        stop_reason = "refusal"
    elif status == "completed":
        stop_reason = "end_turn"
    elif status == "incomplete":
        stop_reason = "max_tokens"
    else:
        stop_reason = status or "end_turn"

    return blocks, stop_reason


class OpenAIProvider:
    """Real OpenAI provider (openai>=3.14.0), talking to the Responses API.

    `client` is constructed lazily on first use via `openai.OpenAI()` (reads
    OPENAI_API_KEY from the environment, and OPENAI_BASE_URL when set) so
    this module stays importable, and this class constructs, with no API
    key present - the constructor itself never touches the network. See the
    module docstring for the request/response translation this class does
    to stay a drop-in `LLMProvider` for `loop.py`.
    """

    def __init__(self, model: str, max_output_tokens: int = 16000, client: Any = None) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # local import: only needed when a real call is made

            base_url = os.environ.get("OPENAI_BASE_URL")
            self._client = OpenAI(base_url=base_url) if base_url else OpenAI()
        return self._client

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        t0 = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=_messages_to_responses_input(messages),
            tools=_tools_to_responses(tools),
            max_output_tokens=self.max_output_tokens,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        content_blocks, stop_reason = _response_to_blocks(response)

        usage_obj = getattr(response, "usage", None)
        usage: dict[str, int] = {}
        if usage_obj is not None:
            input_tokens = getattr(usage_obj, "input_tokens", None)
            output_tokens = getattr(usage_obj, "output_tokens", None)
            if input_tokens is not None:
                usage["input_tokens"] = input_tokens
            if output_tokens is not None:
                usage["output_tokens"] = output_tokens

        return LLMResponse(
            content_blocks=content_blocks,
            stop_reason=stop_reason,
            raw=response,
            model=getattr(response, "model", self.model),
            usage=usage,
            message_id=getattr(response, "id", None),
            latency_ms=latency_ms,
            stop_details=getattr(response, "incomplete_details", None),
        )


class FakeLLM:
    """Scripted provider for tests.

    `script` is a list of turns. Each entry is either:
      {"tool": "click", "input": {...}}   -> one tool_use block, stop_reason "tool_use"
      {"text": "..."}                     -> one text block, stop_reason "end_turn"
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self._i = 0
        self.model = "fake-llm"

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        if self._i >= len(self.script):
            raise RuntimeError(
                f"FakeLLM script exhausted after {self._i} turn(s); the loop asked for another completion"
            )
        entry = self.script[self._i]
        self._i += 1
        turn_no = self._i

        if "tool" in entry:
            block: Any = _FakeToolUseBlock(
                id=f"fake-tool-{turn_no}", name=entry["tool"], input=dict(entry.get("input") or {})
            )
            stop_reason = "tool_use"
        else:
            block = _FakeTextBlock(text=entry.get("text", ""))
            stop_reason = "end_turn"

        return LLMResponse(
            content_blocks=[block],
            stop_reason=stop_reason,
            raw=entry,
            model=self.model,
            usage={"input_tokens": 0, "output_tokens": 0},
            message_id=f"fake-msg-{turn_no}",
            latency_ms=0,
            stop_details=None,
        )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "FakeLLM":
        """Rebuild a script from a saved transcript.jsonl (the shape written
        by `RunLog.transcript` in `hands/agent/loop.py`), so a recorded real
        run can drive a deterministic test."""
        script: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                calls = rec.get("tool_calls") or []
                if calls:
                    for c in calls:
                        script.append({"tool": c["name"], "input": c.get("input") or {}})
                elif rec.get("text"):
                    script.append({"text": rec["text"]})
        return cls(script)
