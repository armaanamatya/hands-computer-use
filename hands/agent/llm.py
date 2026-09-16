"""LLM provider abstraction for the discovery loop.

`AnthropicProvider` talks to the real Claude API (never invoked by this
repo's test suite - no API key is configured here). `FakeLLM` replays a
scripted sequence of tool calls / text turns, duck-typed to look like SDK
content blocks (`.type`, `.name`, `.input`, `.id`, `.text`), so the whole
loop is exercisable deterministically with no network access and no key.
"""
from __future__ import annotations

import json
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
