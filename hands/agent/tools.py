"""Tool schemas for the discovery loop, and a small dispatcher that maps a
validated tool call onto a Surface Action.

Every tool carries a REQUIRED `reason` string ("why you are doing this, in
one sentence"): the evidence log's answer to "what did the agent do and why"
depends on it being present on every step.
"""
from __future__ import annotations

from typing import Any, Optional

from hands.surface.base import Action

_REASON_PROP = {"type": "string", "description": "Why you are doing this, in one sentence."}

TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "click",
        "description": "Click an interactive element identified by its numbered index from the most recent observation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {
                    "type": "integer",
                    "description": "The [idx] number of the element to click, from the numbered element list in the last observation.",
                },
                "reason": _REASON_PROP,
            },
            "required": ["element", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "type_text",
        "description": (
            "Type text into a textbox element identified by its numbered index. "
            "The text may be a literal value or a {{placeholder}} naming an available run input."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {"type": "integer", "description": "The [idx] number of the textbox to type into."},
                "text": {
                    "type": "string",
                    "description": "Literal text, or a {{placeholder}} naming an available run input.",
                },
                "reason": _REASON_PROP,
            },
            "required": ["element", "text", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "select_option",
        "description": "Choose an option in a combobox (dropdown) element identified by its numbered index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {"type": "integer", "description": "The [idx] number of the combobox."},
                "option": {"type": "string", "description": "The option's visible label or value to select."},
                "reason": _REASON_PROP,
            },
            "required": ["element", "option", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "press_key",
        "description": (
            "Press a keyboard key, optionally focusing an element first. "
            "Pass element=null to send the key to whatever currently has focus."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {
                    "type": ["integer", "null"],
                    "description": "The [idx] number of the element to focus first, or null to use current focus.",
                },
                "key": {"type": "string", "description": "The key to press, e.g. 'Enter', 'Tab', 'Escape'."},
                "reason": _REASON_PROP,
            },
            "required": ["element", "key", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "navigate",
        "description": "Navigate the browser directly to a URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The absolute URL to navigate to."},
                "reason": _REASON_PROP,
            },
            "required": ["url", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "read_value",
        "description": "Record the visible text/value of an element as a named output of this run, to be returned by done().",
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {
                    "type": "integer",
                    "description": "The [idx] number of the element whose text/value to capture.",
                },
                "output_name": {"type": "string", "description": "The name to store this captured value under."},
                "reason": _REASON_PROP,
            },
            "required": ["element", "output_name", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "wait",
        "description": "Wait for a fixed duration before observing again, e.g. to let a slow page settle.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ms": {"type": "integer", "description": "Milliseconds to wait."},
                "reason": _REASON_PROP,
            },
            "required": ["ms", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "note_condition",
        "description": (
            "Flag a runtime condition observed on screen (a business outcome, a transient error, an "
            "interstitial, an auth prompt, a state that needs a human, or an application error) without "
            "acting on it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "A short machine-friendly code for the condition, e.g. 'not_found'."},
                "kind": {
                    "type": "string",
                    "enum": ["business", "transient", "interstitial", "auth", "requires_human", "app_error"],
                },
                "text_pattern": {
                    "type": "string",
                    "description": "A distinctive snippet of on-screen text that identifies this condition.",
                },
                "reason": _REASON_PROP,
            },
            "required": ["code", "kind", "text_pattern", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "request_human",
        "description": (
            "Stop and ask a human operator to take over the live session, because you are stuck, cannot find "
            "the target, are on an unexpected screen, or the task requires a step only a human may perform."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason_code": {
                    "type": "string",
                    "enum": ["STUCK_NO_PROGRESS", "TARGET_NOT_FOUND", "UNEXPECTED_SCREEN", "REQUIRES_HUMAN_STEP"],
                },
                "reason": _REASON_PROP,
            },
            "required": ["reason_code", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "done",
        "description": (
            "Declare the goal accomplished. Include every value the goal asked you to capture, keyed by name, "
            "using the values you recorded with read_value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                # An array of explicit name/value pairs rather than an
                # open-ended object. A map typed as
                # `additionalProperties: {type: string}` cannot satisfy
                # strict function calling, which requires
                # `additionalProperties: false` on every nested object, so
                # the map form is rejected outright by at least one
                # provider. The pair list carries the same information and
                # is strict-clean everywhere.
                "outputs": {
                    "type": "array",
                    "description": "Every value the goal asked you to capture, one entry per output.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The output name, exactly as you passed it to read_value.",
                            },
                            "value": {"type": "string", "description": "The captured value."},
                        },
                        "required": ["name", "value"],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string", "description": "A short summary of what was done and found."},
                "reason": _REASON_PROP,
            },
            "required": ["outputs", "summary", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

TOOL_NAMES: set[str] = {t["name"] for t in TOOL_DEFS}

# Tools that translate directly onto a Surface action. The remaining tools
# (read_value, note_condition, request_human, done) do not act on the
# surface and are handled by the loop itself.
TOOL_TO_ACTION_KIND: dict[str, str] = {
    "click": "click",
    "type_text": "type",
    "select_option": "select",
    "press_key": "press",
    "navigate": "navigate",
    "wait": "wait",
}


def build_action(name: str, tool_input: dict[str, Any], *, resolved_value: Optional[str] = None) -> Optional[Action]:
    """Translate a validated tool call into a Surface Action.

    `resolved_value` is the already-placeholder-substituted value for
    type_text; callers resolve `{{name}}` placeholders themselves before
    calling this, so a sensitive value never has to be threaded through
    `tool_input`.
    """
    reason = tool_input.get("reason", "")
    if name == "click":
        return Action(kind="click", element_idx=tool_input.get("element"), reason=reason)
    if name == "type_text":
        value = resolved_value if resolved_value is not None else tool_input.get("text")
        return Action(kind="type", element_idx=tool_input.get("element"), value=value, reason=reason)
    if name == "select_option":
        return Action(kind="select", element_idx=tool_input.get("element"), value=tool_input.get("option"), reason=reason)
    if name == "press_key":
        return Action(kind="press", element_idx=tool_input.get("element"), key=tool_input.get("key"), reason=reason)
    if name == "navigate":
        return Action(kind="navigate", url=tool_input.get("url"), reason=reason)
    if name == "wait":
        return Action(kind="wait", ms=tool_input.get("ms"), reason=reason)
    return None
