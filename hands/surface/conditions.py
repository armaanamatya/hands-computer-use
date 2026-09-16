"""Evaluate the schema's closed `Condition` algebra against live run state.

Every condition boils down to one question the replay engine needs
answered without a model: "does this thing hold right now?" -- is this the
screen we expect, did this outcome fire, has this checkpoint been reached.
`evaluate()` answers that for one `Condition`, `evaluate_all()` for a list
(screen landmarks, capability.target.fingerprint), and `poll_until()` wraps
`evaluate()` in a re-observe-and-retry loop for the simple (non-outcome-
aware) waiting cases.

This module has no knowledge of steps, outcomes, or recovery -- that is
`hands.replay.engine`'s job. It only knows how to read a `Condition`
against a `Surface` + the most recent `Observation` + the outputs captured
so far.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from hands.schema.capability import Condition, FrameMatcher
from hands.surface import locator as locator_mod
from hands.surface.base import Observation


class _NoCoordinateFallback:
    """Minimal stand-in for `hands.safety.policy.Policy` used when a
    condition check has no effective Policy in scope (`ConditionContext.
    policy` is None). `locator.resolve()` only ever reads
    `.allow_coordinate_fallback` off whatever it's given, so this one
    attribute is all a condition check needs to safely refuse the bbox
    strategy rather than guessing permissively.
    """

    allow_coordinate_fallback = False


@dataclass
class ConditionContext:
    """Everything `evaluate()` needs besides the `Condition` itself.

    `default_frame` is the frame the ENCLOSING SCREEN selects -- the
    replay engine sets this to `capability.screens[step.screen].frame`
    while walking a step's checkpoint/wait/outcome conditions, or to a
    `Screen`'s own `frame` while checking that screen's own landmarks.
    See `_resolve_frame_info` below for exactly how this combines with a
    condition's own (optional) `frame`.

    `policy` is the run's effective Policy, needed only so that
    element_visible/element_absent/value_equals (which resolve a
    `TargetSpec`) can correctly refuse a `bbox` strategy when policy
    forbids coordinate fallback. It may be left `None` for a condition
    check made outside a full replay run (e.g. ad hoc); `_NoCoordinateFallback`
    then makes that refusal the safe default rather than an AttributeError.

    `step_id` is carried through purely for logging/attribution symmetry
    with `hands.surface.locator.ResolveContext`.
    """

    surface: Any
    observation: Observation
    outputs: dict[str, Any]
    default_frame: Optional[FrameMatcher] = None
    policy: Optional[Any] = None
    step_id: Optional[str] = None


def _pattern_search(pattern: str, text: str) -> bool:
    """`re.search`, never raising -- every *_pattern / *_match(es) field in
    this schema (`FrameMatcher.url_pattern`, `InputSpec.pattern`,
    `TableSelector.header_match`, `RowWhere.matches`, `ParseSpec.regex`) is
    documented or used elsewhere as a regex; `url_matches` / `title_matches`
    / `text_present` / `Screen.url_pattern` follow the same convention here
    for consistency -- a `re.search`, not a glob or a full-string match, so
    a pattern only needs to identify a distinguishing substring/anchor
    rather than describe the whole string. A malformed pattern is treated
    as simply not matching, not as a crash: a condition should never be
    able to take down a poll loop.
    """
    try:
        return re.search(pattern, text or "") is not None
    except re.error:
        return False


def _resolve_frame_info(ctx: ConditionContext, own_frame: Optional[FrameMatcher]) -> Optional[Any]:
    """Pick which `FrameInfo` (from `ctx.observation.frames`) a frame-
    scoped condition (`url_matches`/`title_matches`/`text_present`) reads
    against.

    Precedence -- this is the crux of the frame-scoping design note on
    `Screen` and on the `Condition` union in `hands/schema/capability.py`:
    a condition's OWN `frame`, if set, always wins. Otherwise fall back to
    `ctx.default_frame` -- the frame the ENCLOSING SCREEN selects. Only if
    NEITHER specifies one do we fall back to the top document (the frame
    with no parent).

    Why this order matters, concretely: the mock app is a frameset (`menu`
    + `work`); the TOP document's URL never changes across sign-on, member
    lookup, and member detail -- only the `work` frame's URL does. A
    condition that silently defaulted to the top document instead of the
    enclosing screen's frame would therefore see the exact same URL/title/
    text on every screen, making wrong-screen detection structurally
    unfireable no matter how careful the rest of the engine is. Falling
    back to `ctx.default_frame` before the top document is what keeps that
    detection possible for the common case (a screen's landmarks/url_
    pattern that don't bother repeating `frame=...` on every single
    condition, relying on the screen's own `frame` instead).
    """
    effective = own_frame or ctx.default_frame
    frames = ctx.observation.frames
    if not frames:
        return None

    def _top() -> Optional[Any]:
        for f in frames:
            if f.parent_index is None:
                return f
        return frames[0]

    if effective is None:
        return _top()
    if effective.url_pattern:
        for f in frames:
            if _pattern_search(effective.url_pattern, f.url):
                return f
        return None
    if effective.name:
        for f in frames:
            if f.name == effective.name:
                return f
        return None
    if effective.index is not None:
        for f in frames:
            if f.index == effective.index:
                return f
        return None
    return _top()


def evaluate(condition: Condition, ctx: ConditionContext) -> tuple[bool, str]:
    """Evaluate one `Condition`. Never raises: any internal failure (no
    matching frame, a target that won't resolve, a bad regex) comes back
    as `(False, reason)` instead of propagating, so one malformed or
    momentarily-mismatched condition inside an `all_of`/`any_of` -- or one
    checked mid-poll while the page is still loading -- never blows up the
    engine's poll loop. "Keep polling" depends on being able to call this
    on every tick without a try/except of its own.
    """
    try:
        return _evaluate(condition, ctx)
    except Exception as e:  # noqa: BLE001
        return False, f"condition raised {e.__class__.__name__}: {e}"


def _evaluate(condition: Condition, ctx: ConditionContext) -> tuple[bool, str]:
    kind = condition.type

    if kind == "dialog_open":
        # No DOM touch here, deliberately: a held (replay-mode) dialog
        # blocks the page's JS thread, so anything that would need
        # frame.evaluate() -- which is everything else in this function --
        # would hang for as long as the dialog stays open.
        # `surface.dialog_state()` is implemented without evaluate() for
        # exactly this reason (see hands/surface/web.py), so it's the one
        # safe call to make while a dialog might be open.
        dlg = ctx.surface.dialog_state()
        if dlg is None:
            return False, "no dialog is open"
        if condition.pattern and not _pattern_search(condition.pattern, dlg.message):
            return False, f"dialog open but message {dlg.message!r} does not match {condition.pattern!r}"
        return True, f"dialog open: {dlg.type} {dlg.message!r}"

    if kind == "url_matches":
        finfo = _resolve_frame_info(ctx, condition.frame)
        if finfo is None:
            return False, "no matching frame in the current observation"
        ok = _pattern_search(condition.pattern, finfo.url)
        return ok, f"frame {finfo.index} url {finfo.url!r} {'matches' if ok else 'does not match'} {condition.pattern!r}"

    if kind == "title_matches":
        finfo = _resolve_frame_info(ctx, condition.frame)
        if finfo is None:
            return False, "no matching frame in the current observation"
        title = ctx.observation.frame_titles.get(finfo.index, "")
        ok = _pattern_search(condition.pattern, title)
        return ok, f"frame {finfo.index} title {title!r} {'matches' if ok else 'does not match'} {condition.pattern!r}"

    if kind == "text_present":
        finfo = _resolve_frame_info(ctx, condition.frame)
        if finfo is None:
            return False, "no matching frame in the current observation"
        text = ctx.observation.frame_texts.get(finfo.index, "")
        found = _pattern_search(condition.pattern, text)
        ok = (not found) if condition.negate else found
        expect = "absent" if condition.negate else "present"
        actual = "present" if found else "absent"
        return ok, f"text matching {condition.pattern!r} is {actual} in frame {finfo.index} (expected {expect})"

    if kind in ("element_visible", "element_absent"):
        return _evaluate_element_presence(condition, ctx, absent=(kind == "element_absent"))

    if kind == "value_equals":
        return _evaluate_value_equals(condition, ctx)

    if kind == "output_matches":
        if condition.output not in ctx.outputs:
            return False, f"output {condition.output!r} has not been captured yet"
        value = ctx.outputs[condition.output]
        ok = _pattern_search(condition.pattern, str(value))
        return ok, f"output {condition.output!r}={value!r} {'matches' if ok else 'does not match'} {condition.pattern!r}"

    if kind == "all_of":
        results = [_evaluate(c, ctx) for c in condition.conditions]
        ok = all(r[0] for r in results)
        return ok, "all_of[" + "; ".join(r[1] for r in results) + "]"

    if kind == "any_of":
        results = [_evaluate(c, ctx) for c in condition.conditions]
        ok = any(r[0] for r in results)
        return ok, "any_of[" + "; ".join(r[1] for r in results) + "]"

    if kind == "not_":
        ok, why = _evaluate(condition.not_, ctx)
        return (not ok), f"not({why})"

    return False, f"unknown condition type {kind!r}"


def _evaluate_element_presence(condition: Any, ctx: ConditionContext, absent: bool) -> tuple[bool, str]:
    target = condition.target
    policy = ctx.policy or _NoCoordinateFallback()
    try:
        result = locator_mod.resolve(
            ctx.surface, target, policy, locator_mod.ResolveContext(step_id=ctx.step_id)
        )
    except locator_mod.TargetResolutionError:
        present = False
    else:
        present = True
        if result.locator is not None:
            try:
                present = result.locator.is_visible()
            except Exception:  # noqa: BLE001 - resolved to one match; treat as present
                present = True
    ok = (not present) if absent else present
    verb = "absent" if absent else "visible"
    actual = "absent" if not present else "present/visible"
    return ok, f"target {target.description!r} is {actual} (expected {verb})"


def _evaluate_value_equals(condition: Any, ctx: ConditionContext) -> tuple[bool, str]:
    target = condition.target
    policy = ctx.policy or _NoCoordinateFallback()
    try:
        result = locator_mod.resolve(
            ctx.surface, target, policy, locator_mod.ResolveContext(step_id=ctx.step_id)
        )
    except locator_mod.TargetResolutionError as e:
        return False, f"target {target.description!r} did not resolve: {e.message}"
    if result.locator is None:
        return False, "value_equals cannot be checked against a coordinate-fallback target"
    try:
        actual = result.locator.input_value()
    except Exception:  # noqa: BLE001 - not a form control; fall back to its text
        try:
            actual = result.locator.inner_text()
        except Exception:  # noqa: BLE001
            return False, "could not read a value from the resolved element"
    ok = actual == condition.value
    return ok, f"value {actual!r} {'equals' if ok else 'does not equal'} {condition.value!r}"


def evaluate_all(conditions: list[Condition], ctx: ConditionContext) -> tuple[bool, list[str]]:
    """AND all of `conditions` together. Used for a `Screen`'s `landmarks`
    and for `TargetApp.fingerprint` -- both "all of these must hold" lists.
    Returns `(all_true, [explanation, ...])`, one explanation per
    condition in order, so a caller can report exactly which landmark
    failed rather than only "landmarks didn't match".
    """
    results = [evaluate(c, ctx) for c in conditions]
    return all(r[0] for r in results), [r[1] for r in results]


def poll_until(
    condition: Condition,
    ctx: ConditionContext,
    timeout_ms: int,
    interval_ms: int = 250,
    refresh: Optional[Callable[[], ConditionContext]] = None,
) -> tuple[bool, str, ConditionContext]:
    """Poll `condition` against `ctx`, re-observing via `refresh()` between
    attempts, until it holds or `timeout_ms` elapses.

    `refresh`, when given, is a zero-arg callable returning a fresh
    `ConditionContext` (typically built around a new `surface.observe()`
    call) -- without it, this just checks `ctx` once (a re-observe-less
    poll is a single evaluation, not a retry loop against a stale
    snapshot). Returns `(ok, explanation, last_ctx)`, so the caller can
    inspect exactly what the final observation looked like whether or not
    the condition ever held.

    This is a general-purpose building block for the simple case: a single
    condition, no outcome interleaving. The replay engine's own per-step
    settle loop needs to interleave outcome checks with the checkpoint
    check on every tick (see `hands/replay/engine.py`'s phase-3 docstring),
    so it does not use this helper directly for that -- but idempotency-
    check-after-recovery and other single-condition waits do.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    cur = ctx
    while True:
        ok, why = evaluate(condition, cur)
        if ok:
            return True, why, cur
        remaining = deadline - time.monotonic()
        if remaining <= 0 or refresh is None:
            return False, why, cur
        time.sleep(min(interval_ms / 1000.0, remaining))
        cur = refresh()
