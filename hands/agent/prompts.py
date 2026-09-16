"""System prompt and observation rendering for the discovery loop.

NOTE (deliberate, read before editing): `SYSTEM_PROMPT` below must stay 100%
generic. It must never name the target application, its routes/URLs, its
field or screen names, or describe the flow needed to reach any goal.
Discovery is only meaningful evidence if the model finds the flow itself with
no hints baked into the prompt; any app-specific detail here would quietly
invalidate that evidence. If a goal is unreliable, fix the goal text, the
tool surface, or the element perception - never add a hint here.
"""
from __future__ import annotations

from typing import Any, Optional

from hands.surface.base import Element, Observation

PROMPT_VERSION = "1"

SYSTEM_PROMPT = """You operate a live, real web application on behalf of a human operator, working entirely from what you can currently see on screen plus a numbered list of interactive elements. You have never seen this application before and are given no map of it: you must explore, read what is on screen, and decide each next step yourself.

Each turn you are shown:
- The current page's URL and title.
- The list of frames on the page. A page may be built from more than one frame; each frame has its own URL and its own elements, and an element index from one frame cannot be used to act in another frame.
- A numbered list of the elements you can act on, one line per element, grouped by frame. Each line looks like `[idx] role "name" (extra detail)`. The idx is only valid for the CURRENT observation - once you act, the page can change and numbering is re-issued from scratch on the next observation, so never reuse an idx from a previous turn.
- Elements that come from a table row carry that row's position and column headers, so you can tell visually similar rows apart.
- A short excerpt of each frame's visible text, and, separately, a screenshot image of the current page. Use both: the screenshot shows layout, color, and things like open alerts that the text excerpt does not carry; the element list and text give you exact, reliable strings to act on.
- A note if a native browser dialog (alert/confirm/prompt) is currently open. While a dialog is open, other actions on the page are blocked; you can press a key or wait, and it will otherwise resolve on its own.

You act only through the provided tools, one deliberate step at a time. Every tool call requires a `reason` argument: a single sentence stating why you are taking that action right now. Write it as if someone auditing this run later will read only that sentence to understand the step - be specific about what you expect it to accomplish, not generic ("click the button" is not a reason; "open this record so I can read its balance" is).

Some tools accept a run input by name as a `{{placeholder}}` (for example `{{some_value}}`) instead of literal text. Placeholders refer to values configured for this run; some are sensitive and are deliberately never shown to you - you will only ever see the placeholder's name and a description of what it is for, never its value. Use a placeholder exactly as given, including the double braces, when you want that value typed. You may also type a literal string when the goal itself already tells you what belongs there.

Work step by step: observe, decide the single next action that makes progress toward the goal, act, then observe again before deciding the next one. Prefer the most direct, literal interpretation of an element's name and role over guessing. If more than one element looks plausible, prefer the one whose name and context most specifically match your intent, and use the screenshot to disambiguate visually similar candidates.

Call `request_human` and stop acting when you are genuinely stuck: you have made no real progress after several distinct attempts, the target you need does not appear to exist anywhere you can find, you have landed on a screen that does not make sense for the goal, or the task requires a step that only a human is permitted to perform (for example, one gated behind information or authority you do not have). Pick the reason code that best matches and explain briefly. Do not guess repeatedly hoping for a different result - ask for a human instead.

Call `done` only once the goal is fully satisfied. If the goal asked you to capture one or more values, you must have recorded each of them with `read_value` first, and you must include all of them, by the same name, in `done`'s `outputs`. Give a short, factual summary of what you did and what you found. Do not call `done` while a necessary step is still outstanding."""


def build_system_prompt(inputs_desc: Optional[dict[str, str]] = None) -> str:
    """Compose the system prompt for one run.

    `inputs_desc` maps each available run-input name to a short, value-free
    description. Values themselves are never included here - see the
    "sensitive" note in SYSTEM_PROMPT; this is what keeps a sensitive value
    out of the model's context entirely.
    """
    if not inputs_desc:
        return SYSTEM_PROMPT
    lines = [
        "",
        "Available run inputs (reference as {{name}} in a tool call; values are never shown to you here):",
    ]
    for name, desc in sorted(inputs_desc.items()):
        lines.append(f"  {{{{{name}}}}} - {desc}")
    return SYSTEM_PROMPT + "\n" + "\n".join(lines)


# --------------------------------------------------------------------- render

_MAX_CHARS = 6000
_TEXT_EXCERPT_CAP = 350
_TEXT_EXCERPT_CAP_TIGHT = 120

_INTERACTIVE_ROLES = {"link", "button", "textbox", "checkbox", "radio", "combobox"}


def _fmt_attrs(el: Element) -> str:
    a = el.attrs or {}
    bits: list[str] = []
    t = a.get("type")
    if t and t not in ("text",):
        bits.append(f"type={t}")
    val = a.get("value")
    if val:
        bits.append(f'value="{val}"')
    ph = a.get("placeholder")
    if ph:
        bits.append(f'placeholder="{ph}"')
    if "checked" in a:
        bits.append(f"checked={a['checked']}")
    opts = a.get("options")
    if opts:
        shown = opts[:8]
        more = "" if len(opts) <= 8 else f", +{len(opts) - 8} more"
        bits.append(f"options=[{', '.join(shown)}{more}]")
    href = a.get("href")
    if href:
        bits.append(f'href="{href[:60]}"')
    return "; ".join(bits)


def _fmt_row_ctx(row_ctx: Optional[dict[str, Any]]) -> str:
    if not row_ctx:
        return ""
    headers = row_ctx.get("headers") or []
    return f"row {row_ctx.get('row_index')} of table with headers {headers}"


def _element_line(el: Element) -> str:
    extra_parts = [p for p in (_fmt_attrs(el), _fmt_row_ctx(el.row_ctx)) if p]
    extra = f" ({'; '.join(extra_parts)})" if extra_parts else ""
    return f'[{el.idx}] {el.role} "{el.name}"{extra}'


def render_observation(obs: Observation, inputs_desc: Optional[dict[str, str]] = None) -> str:
    """Compact text rendering of an Observation, kept under ~_MAX_CHARS by
    truncating intelligently: interactive elements are never dropped; text
    excerpts are tightened first, then text-only elements are dropped from
    the frames that have the most of them, one at a time."""
    header: list[str] = [f"URL: {obs.url}", f"Title: {obs.title}"]
    if obs.dialog is not None:
        header.append(
            f'[DIALOG OPEN] {obs.dialog.type}: "{obs.dialog.message}" - other actions are blocked until it resolves.'
        )
    header.append("")
    header.append("Frames:")
    for f in obs.frames:
        label = f' "{f.name}"' if f.name else " (unnamed)"
        header.append(f"  [frame {f.index}]{label} {f.url}")

    by_frame: dict[int, list[Element]] = {}
    for el in obs.elements:
        by_frame.setdefault(el.frame_index, []).append(el)

    frame_interactive: dict[int, list[str]] = {}
    frame_textual: dict[int, list[str]] = {}
    for f in obs.frames:
        els = by_frame.get(f.index, [])
        frame_interactive[f.index] = [_element_line(e) for e in els if e.role in _INTERACTIVE_ROLES]
        frame_textual[f.index] = [_element_line(e) for e in els if e.role not in _INTERACTIVE_ROLES]

    def build(text_cap: int, textual_budget: dict[int, int]) -> str:
        parts = list(header)
        for f in obs.frames:
            parts.append("")
            parts.append(f"Elements in frame {f.index}:")
            keep_n = textual_budget.get(f.index, len(frame_textual[f.index]))
            kept_text = frame_textual[f.index][:keep_n]
            dropped = len(frame_textual[f.index]) - len(kept_text)
            all_el_lines = frame_interactive[f.index] + kept_text
            if not all_el_lines and not dropped:
                parts.append("  (none)")
            else:
                parts.extend(f"  {l}" for l in all_el_lines)
                if dropped:
                    parts.append(f"  ... +{dropped} more text-only elements omitted")
        parts.append("")
        parts.append("Visible text by frame:")
        for f in obs.frames:
            text = " ".join((obs.frame_texts.get(f.index) or "").split())
            if len(text) > text_cap:
                text = text[:text_cap].rstrip() + "…"
            parts.append(f"  Frame {f.index}: {text}" if text else f"  Frame {f.index}: (empty)")
        if inputs_desc:
            names = ", ".join(f"{{{{{n}}}}}" for n in sorted(inputs_desc))
            parts.append("")
            parts.append(f"Available input placeholders: {names}")
        return "\n".join(parts)

    full_budget = {f.index: len(frame_textual[f.index]) for f in obs.frames}
    rendered = build(_TEXT_EXCERPT_CAP, full_budget)
    if len(rendered) <= _MAX_CHARS:
        return rendered

    rendered = build(_TEXT_EXCERPT_CAP_TIGHT, full_budget)
    if len(rendered) <= _MAX_CHARS:
        return rendered

    budget = dict(full_budget)
    while len(rendered) > _MAX_CHARS and any(v > 0 for v in budget.values()):
        fi = max(budget, key=lambda k: budget[k])
        if budget[fi] <= 0:
            break
        budget[fi] -= 1
        rendered = build(_TEXT_EXCERPT_CAP_TIGHT, budget)

    if len(rendered) > _MAX_CHARS:
        rendered = rendered[: _MAX_CHARS - 20].rstrip() + "\n...[truncated]"
    return rendered
