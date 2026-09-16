"""Deterministic element (re)location for replay.

`resolve()` is what lets a recorded `TargetSpec` find "the same" element
again on a later run of a legacy, id-less, ARIA-less page: it selects the
right frame, then tries each recorded `Strategy` in order, and only accepts
a strategy that matches EXACTLY ONE element. Zero matches means the
strategy simply didn't find anything on this run (try the next one); more
than one match means the strategy is ambiguous on this run even though it
presumably wasn't at discovery time (also try the next one, but remember
the ambiguity). Every attempt -- successful, empty, ambiguous, erroring, or
skipped -- is recorded as a `StrategyAttempt` so a replay result can show
*exactly* how (or why not) a target resolved, not just whether it did.

This module has no knowledge of `hands.replay.engine`'s step/outcome
machinery; it only knows how to turn a `TargetSpec` into a Playwright
`Locator` (or, for the `bbox` strategy, a page-coordinate point with no
`Locator` at all). That keeps it reusable from `conditions.py` (element_
visible/absent, value_equals) and `extract.py` (every extraction, including
`table_cell`'s use of `target` to find the table itself) without either of
those needing to know anything about frames, strategies, or Playwright.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from hands.schema.capability import FrameMatcher, TargetSpec
from hands.schema.result import StrategyAttempt

# perceive.js (hands/surface/perceive.js, `roleOf`) emits `"text"` as the
# role of a non-interactive innermost text block. Playwright's ARIA
# `get_by_role` has no "text" role -- it isn't one of the WAI-ARIA roles --
# so a Strategy recorded with role="text" is resolved via `get_by_text`
# instead. Every OTHER role perceive.js can emit (link, button, textbox,
# checkbox, radio, combobox, heading, columnheader, cell) is a real ARIA
# role that Playwright's `get_by_role` understands directly.
_ROLE_VIA_TEXT = frozenset({"text"})


class TargetResolutionError(Exception):
    """Raised when NO strategy (across the whole ordered list) resolved to
    exactly one element -- including the case where the frame itself could
    not be selected at all (frame selection failure is a resolution failure
    with a clear message, per the frame-matching rule below).

    `category` is one of:
      * "not_found"  -- at least one attempted (non-skipped) strategy came
        back with zero matches, or the frame could not be selected. This is
        the ordinary "the element genuinely isn't there this time" case.
      * "ambiguous"  -- every attempted (non-skipped) strategy matched MORE
        than one element and none matched zero. This is the "the page has
        several candidates and none of our recorded strategies pin down a
        single one" case -- a meaningfully different failure mode from
        not_found, because it usually means the recorded strategy needs a
        tighter discriminator, not that the element vanished.
    `attempts` carries every `StrategyAttempt` made before giving up, so the
    caller (the replay engine) can classify TARGET_NOT_FOUND vs
    TARGET_AMBIGUOUS and show the full resolution history either way.
    """

    def __init__(self, message: str, category: str, attempts: list[StrategyAttempt]) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.attempts = attempts


@dataclass
class ResolveContext:
    """Threaded through resolution purely for observability.

    `step_id` tags attempts/events with the step (or condition) they were
    made on behalf of, for anyone logging them (the replay engine does; a
    bare condition check from `conditions.py` typically leaves it `None`).
    `on_attempt`, when given, is invoked once per `StrategyAttempt` as it
    completes -- letting a caller stream resolution attempts to the run log
    in real time instead of only seeing them in bulk once `resolve()`
    returns (or raises).
    """

    step_id: Optional[str] = None
    on_attempt: Optional[Callable[[StrategyAttempt], None]] = None


@dataclass
class ResolveResult:
    """The outcome of a successful `resolve()` call.

    `locator` is a Playwright `Locator` resolved to exactly one element --
    EXCEPT when `strategy_used == "bbox"`, in which case there is no DOM
    element at all (`locator is None`) and the caller must act by page
    coordinates via `bbox_point` and `Surface.click_at`.

    `strategy_rank` is the 0-based index, within `target.strategies`, of
    the strategy that won. Rank 0 means the recorder's first choice still
    works; any rank > 0 is a drift signal (the page has moved on enough
    that a fallback was needed), and so is `coordinate_fallback` being True
    regardless of rank (bbox is inherently the least stable strategy kind).

    `element_name` is the accessible/semantic name the WINNING strategy's
    own params identify the element by (its `name`/`text`/`anchor_text`),
    when the strategy kind carries one. This is what the replay engine
    passes to `Policy.check_action(element_name=...)` for live risk
    classification -- deriving it here keeps knowledge of which param key
    means "name" for which strategy kind inside this module only.
    """

    locator: Any
    strategy_used: str
    strategy_rank: int
    attempts: list[StrategyAttempt]
    coordinate_fallback: bool
    frame: Any
    element_name: Optional[str] = None
    bbox_point: Optional[tuple[float, float]] = None


def _select_frame(surface: Any, matcher: FrameMatcher) -> tuple[Any, Any]:
    """Select the (playwright Frame, FrameInfo) `matcher` points at.

    Preference order, per `FrameMatcher`'s own docstring: `url_pattern`
    (regex against the frame's URL), then `name` (exact), then `index`
    (exact, positional -- brittle, last resort). If none of the three
    fields is set, default to the top/main frame (the one with no parent),
    since that is the only frame identifiable without any matcher at all.

    Raises `TargetResolutionError` (category "not_found", no attempts --
    there is nothing to attempt a strategy against without a frame) if no
    frame satisfies the matcher. This IS a resolution failure, not a
    separate error class: "the frame doesn't exist this run" is exactly as
    much a targeting failure as "the element doesn't exist this run".
    """
    frames = surface._frames()  # noqa: SLF001 - documented, intended seam (see hands/surface/web.py)

    if matcher.url_pattern:
        pattern = re.compile(matcher.url_pattern)
        for frame, info in frames:
            if pattern.search(info.url):
                return frame, info
        raise TargetResolutionError(
            f"no frame url matches {matcher.url_pattern!r}", "not_found", []
        )

    if matcher.name:
        for frame, info in frames:
            if info.name == matcher.name:
                return frame, info
        raise TargetResolutionError(f"no frame named {matcher.name!r}", "not_found", [])

    if matcher.index is not None:
        for frame, info in frames:
            if info.index == matcher.index:
                return frame, info
        raise TargetResolutionError(f"no frame at index {matcher.index}", "not_found", [])

    for frame, info in frames:
        if info.parent_index is None:
            return frame, info
    raise TargetResolutionError("no top frame is available", "not_found", [])


def _by_role(scope: Any, role: str, name: Optional[str], exact: bool) -> tuple[Any, Optional[str]]:
    """Resolve a role/name pair against `scope`, which may be a Playwright
    `Frame` (top-level resolution) or a `Locator` (row-scoped resolution
    from `_by_relative`, below) -- both expose the same `get_by_role` /
    `get_by_text` chaining API, so this helper works for either.
    """
    if role in _ROLE_VIA_TEXT:
        loc = scope.get_by_text(name, exact=exact) if name else scope.get_by_text("")
        return loc, name
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
        kwargs["exact"] = exact
    return scope.get_by_role(role, **kwargs), name


def _by_relative(frame: Any, params: dict[str, Any]) -> tuple[Optional[Any], Optional[str]]:
    """The legacy-table primitive: locate an element relative to an anchor
    cell's text, per the two supported `relation` values.

    `same_row`: find the `<tr>` containing a cell whose text matches
    `anchor_text` (via Playwright's `locator("tr").filter(has_text=...)`,
    a substring match), then within THAT row locate the target either by
    role/name (when `role` is given -- the common case, since most rows
    have exactly one control) or as the Nth cell under a header named
    `anchor_header` (computed from the table's own header row, so it does
    not depend on which row of the table this is -- exactly the primitive
    `extract.py`'s table_cell reads describe needing, and reusable here for
    click/type targets that are more naturally row-anchored than role-
    anchored).

    `within`: a looser relation for targets that are not laid out as a
    table row -- locate an element matching `anchor_text`, then search for
    role/name within its nearest enclosing table/form/div. This is a
    best-effort container heuristic (legacy markup rarely gives a cleaner
    scoping mechanism); `same_row` is preferred whenever the target really
    is a table row, which is the overwhelmingly common legacy-UI case this
    mock app was built to exercise.
    """
    anchor_text = params.get("anchor_text")
    if not anchor_text:
        raise ValueError("relative strategy requires anchor_text")
    relation = params.get("relation", "same_row")
    role = params.get("role")
    name = params.get("name")
    exact = bool(params.get("exact", False))

    if relation == "same_row":
        row = frame.locator("tr").filter(has_text=anchor_text)
        if role:
            loc, nm = _by_role(row, role, name, exact)
            return loc, nm or anchor_text
        anchor_header = params.get("anchor_header")
        if anchor_header:
            header_cells = frame.locator("tr").first.locator("th,td").all_inner_texts()
            col = next(
                (i for i, h in enumerate(header_cells) if anchor_header.strip() in h.strip()), None
            )
            if col is None:
                return None, None
            return row.locator("td,th").nth(col), anchor_header
        raise ValueError("relative/same_row requires either role or anchor_header")

    if relation == "within":
        anchor = frame.get_by_text(anchor_text, exact=exact)
        container = anchor.locator(
            "xpath=ancestor::*[self::table or self::form or self::div][1]"
        )
        if role:
            loc, nm = _by_role(container, role, name, exact)
            return loc, nm or anchor_text
        raise ValueError("relative/within requires role")

    raise ValueError(f"unknown relative relation {relation!r}")


def resolve(
    surface: Any,
    target: TargetSpec,
    policy: Any,
    run_ctx: Optional[ResolveContext] = None,
) -> ResolveResult:
    """Resolve `target` to exactly one element (or a coordinate point).

    `policy` needs only `.allow_coordinate_fallback` (duck-typed so
    `conditions.py`/`extract.py` can pass a minimal stand-in when no real
    effective `Policy` is in scope for a bare condition check).

    Frame selection happens once, up front (see `_select_frame`); a frame
    selection failure propagates as a `TargetResolutionError` immediately,
    since there is no frame to try any strategy against.

    Strategies are then tried IN ORDER. Every attempt -- match, no-match,
    ambiguous, error, or skip -- becomes a `StrategyAttempt`, appended to
    `attempts` and (if `run_ctx.on_attempt` is set) streamed to the caller
    as it happens. A strategy is accepted the moment it matches EXACTLY
    ONE element; that immediately returns, without trying any later
    (coarser) strategies -- they exist as fallbacks, not as agreement
    checks. If every strategy fails, `TargetResolutionError` is raised with
    the full attempt history and a category the engine maps straight onto
    TARGET_NOT_FOUND / TARGET_AMBIGUOUS.
    """
    ctx = run_ctx or ResolveContext()
    frame, _frame_info = _select_frame(surface, target.frame)

    attempts: list[StrategyAttempt] = []

    for rank, strategy in enumerate(target.strategies):
        t0 = time.perf_counter()
        match_count = 0
        error: Optional[str] = None
        loc: Any = None
        name: Optional[str] = None
        skipped = False

        try:
            if strategy.kind == "bbox":
                if not getattr(policy, "allow_coordinate_fallback", False):
                    skipped = True
                    error = "skipped: coordinate-fallback locators are not permitted by the effective policy"
                else:
                    # There is no DOM to count matches against; a bbox
                    # strategy "matches" trivially once policy allows it.
                    match_count = 1
            elif strategy.kind == "role":
                loc, name = _by_role(
                    frame,
                    strategy.params["role"],
                    strategy.params.get("name"),
                    bool(strategy.params.get("exact", False)),
                )
                match_count = loc.count()
            elif strategy.kind == "text":
                text = strategy.params["text"]
                loc = frame.get_by_text(text, exact=bool(strategy.params.get("exact", False)))
                name = text
                match_count = loc.count()
            elif strategy.kind == "relative":
                loc, name = _by_relative(frame, strategy.params)
                match_count = loc.count() if loc is not None else 0
            elif strategy.kind == "structural":
                loc = frame.locator(strategy.params["path"])
                match_count = loc.count()
            else:
                error = f"unknown strategy kind {strategy.kind!r}"
        except Exception as e:  # noqa: BLE001 - a strategy erroring is just a failed attempt
            error = f"{e.__class__.__name__}: {str(e).splitlines()[0][:200]}"
            match_count = 0

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        attempt = StrategyAttempt(
            strategy=strategy.kind,
            params=dict(strategy.params),
            match_count=match_count,
            elapsed_ms=elapsed_ms,
            error=error,
        )
        attempts.append(attempt)
        if ctx.on_attempt:
            ctx.on_attempt(attempt)

        if skipped:
            continue

        if match_count == 1:
            if strategy.kind == "bbox":
                b = strategy.params
                x = float(b["x"]) + float(b["w"]) / 2.0
                y = float(b["y"]) + float(b["h"]) / 2.0
                return ResolveResult(
                    locator=None, strategy_used="bbox", strategy_rank=rank, attempts=attempts,
                    coordinate_fallback=True, frame=frame, element_name=None, bbox_point=(x, y),
                )
            return ResolveResult(
                locator=loc, strategy_used=strategy.kind, strategy_rank=rank, attempts=attempts,
                coordinate_fallback=False, frame=frame, element_name=name,
            )

    # Nothing matched exactly once. Classify not_found vs ambiguous using
    # only the strategies that were actually attempted (skipped bbox
    # attempts don't count either way): ambiguous means EVERY attempted
    # strategy over-matched and none came back empty; anything else -- at
    # least one true zero-match, or no strategies attempted at all -- is
    # the more ordinary not_found.
    attempted = [a for a in attempts if not (a.error and a.error.startswith("skipped"))]
    category = "ambiguous" if attempted and all(a.match_count > 1 for a in attempted) else "not_found"
    raise TargetResolutionError(
        f"{target.description}: no strategy resolved to exactly one element", category, attempts
    )
