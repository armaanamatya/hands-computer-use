"""The recorder: turns a successful `DiscoveryTrace` into a saved, replayable
`Capability` artifact.

This is the hinge of the whole system -- "the model discovers, the artifact
becomes a reusable capability" -- so this module optimises for two things
above all else, in this order: (1) never emit an artifact the schema itself
considers incoherent (`Capability.validate_semantics()` must return `[]`,
enforced at the bottom of `record()`), and (2) never let a secret reach the
artifact (see `_bind_type_value`'s docstring).

Everything else -- which strategy order a target gets, how a screen's
landmarks are chosen, how a checkpoint is derived -- is a defensible,
documented inference over the trace, not a guess. Where the given API left a
real gap (see the module-level "Known compromises" note at the bottom),
this module adds an optional keyword argument rather than silently guessing.

## How a screen is identified

`hands.agent.loop.DiscoveryTrace.observations` are raw `Observation.to_dict()`
snapshots: a list of frames (top document + children) plus, per frame, its
own URL and visible text. In a frameset app (the mock bank this project
targets), the TOP document's URL never changes across login, lookup, and
member detail -- only one particular child frame's URL does; the other
child frame (menu/nav chrome) has a URL that never changes either. Screens
are therefore keyed on whichever child frame's URL PATH takes more than one
distinct value across the run: not the top document (`parent_index is
None`), and not a frame whose path is constant throughout (that is chrome).
See `_find_work_frame`.

## How a checkpoint is proven, not just asserted

A checkpoint is worthless if it happens to already be true before the step
runs -- it wouldn't prove the action ever occurred. Every checkpoint this
recorder emits is therefore linted with a PURE function over the recorded
before/after observation dicts (`_lint_eval` / `_checkpoint_lints_ok`): no
Surface, no browser, just re-deriving what the real
`hands.surface.conditions.evaluate()` would have said, restricted to the
three condition kinds this recorder ever emits as a checkpoint
(`url_matches`, `text_present`, `value_equals`). A checkpoint that lints
false-before/true-after survives; anything else is dropped, logged to
`pruning_log`, and the step's `wait` is reset to `WaitSpec()` (the default
`"load"` wait) -- NOT left pointing at the same unproven condition via
`wait.until`, which would let replay "verify" a click that never happened
just as surely as leaving `step.checkpoint` set would have.

## Known compromises (see module docstring footer)

Search this file for "COMPROMISE" for the handful of places where the given
`record()` signature does not expose information the algorithm in the task
description assumes the recorder has (the actual run input VALUES, for
example, or `Contract.side_effects`/`idempotent`). Each is patched with an
optional keyword argument with a derived default, never a silent guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

from hands.agent.loop import DiscoveryTrace, TraceStep
from hands.schema.capability import (
    AllOf,
    AnyOf,
    AppInfo,
    Approval,
    Capability,
    CapabilityPolicy,
    Checkpoint,
    Condition,
    Contract,
    Extraction,
    FrameMatcher,
    Handling,
    InputSpec,
    Not,
    Outcome,
    OutputMatches,
    OutputSpec,
    ParseSpec,
    Provenance,
    Risk,
    RowWhere,
    Screen,
    Step,
    Strategy,
    TableSelector,
    TargetApp,
    TargetSpec,
    TextPresent,
    TitleMatches,
    UrlMatches,
    ValueEquals,
    WaitSpec,
)

__all__ = ["record", "RecordResult", "RecorderError"]


class RecorderError(Exception):
    """Raised when a trace cannot be honestly turned into a capability --
    an undeclared input is referenced, a secret can't be bound to a
    placeholder, no work frame can be identified, or the assembled
    capability fails `validate_semantics()`. Never silently patched over.
    """


@dataclass
class RecordResult:
    capability: Capability
    raw_capability: Capability
    pruning_log: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_VIEWPORT: dict[str, int] = {"width": 1280, "height": 800}

# click->click, type_text->type, select_option->select, press_key->press,
# navigate->navigate, wait->wait, read_value->extract. note_condition,
# request_human, and done are deliberately absent: they become outcomes /
# success, never a Step (see _build_discovered_outcomes and _build_success).
_TOOL_KIND: dict[str, str] = {
    "click": "click",
    "type_text": "type",
    "select_option": "select",
    "press_key": "press",
    "navigate": "navigate",
    "wait": "wait",
    "read_value": "extract",
}

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
_REDACTED_MARKER_RE = re.compile(r"^\[redacted:(\w+)\]$")

# Element roles perceive.js assigns to plain form controls. Their `name` is
# usually just a copy of an adjacent label cell, so they add nothing new as
# landmark/checkpoint text candidates beyond what that label cell (a "cell"
# or "text" role element) already contributes; excluding them keeps
# candidate sets smaller without losing any distinguishing power.
_NON_LANDMARK_ROLES = frozenset({"textbox", "checkbox", "radio", "combobox"})


# ==========================================================================
# Small pure helpers: paths, ids, patterns
# ==========================================================================


def _path_of(url: str) -> str:
    return urlsplit(url).path or "/"


def _path_pattern(path: str) -> str:
    """A regex matching a URL whose path is exactly `path`, anchored on the
    path itself and tolerant of host/port (search, not fullmatch, is used
    everywhere this is consumed) and of a trailing query string.

    Anchoring at BOTH ends of the path (`^https?://[^/]+` on the left,
    `(?:$|\\?)` on the right) matters most for `path == "/"`: without the
    right-hand anchor, `"/"` would match as a bare substring of any deeper
    path (e.g. ".../subaccount/new") since every path contains at least one
    "/".
    """
    escaped = re.escape(path)
    return rf"^https?://[^/]+{escaped}(?:$|\?)"


def _screen_id_from_path(path: str) -> str:
    p = path.strip("/")
    if not p:
        return "root"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_").lower()
    return slug or "root"


def _frame_by_index(obs: dict[str, Any], index: int) -> Optional[dict[str, Any]]:
    for f in obs.get("frames") or []:
        if f.get("index") == index:
            return f
    return None


def _work_path(obs: dict[str, Any], work_index: int) -> Optional[str]:
    f = _frame_by_index(obs, work_index)
    return _path_of(f["url"]) if f else None


def _element_fragments(obs: dict[str, Any], frame_index: int) -> set[str]:
    """The set of distinct, non-empty visible-text fragments belonging to
    one frame of one observation.

    `frame_texts` (as actually produced by `hands/surface/web.py` +
    `perceive.js`) is a SINGLE whitespace-collapsed blob per frame
    (`norm(root.innerText)`), so there is no way to recover node-level
    "lines" from it after the fact. `elements[*].name` is that same content
    still segmented: each innermost text-bearing element is emitted with its
    own already-normalised text.

    But an element's name is NOT always drawn from the document text. An
    accessible name can come from an attribute instead -- a submit button is
    named by its `value`, a field by its `placeholder`, `title`, or the text
    of the label cell beside it. None of those appear in `innerText`, so a
    landmark built from such a name is a condition `text_present` can never
    satisfy: the screen check would fail on a screen that is in fact correct.

    That is not hypothetical. An earlier version of this function assumed
    every name was a substring of the frame text, and the resulting
    capability picked the submit buttons "Search" and "Clear" as landmarks
    for the lookup screen. Replay then rejected that screen as UNEXPECTED,
    which is how the bug was found.

    So candidates are verified against the frame's actual text rather than
    assumed to be in it.
    """
    blob = (obs.get("frame_texts") or {}).get(str(frame_index))
    if blob is None:
        blob = (obs.get("frame_texts") or {}).get(frame_index) or ""
    frags: set[str] = set()
    for e in obs.get("elements") or []:
        if e.get("frame_index") != frame_index:
            continue
        if e.get("role") in _NON_LANDMARK_ROLES:
            continue
        name = (e.get("name") or "").strip()
        # Only keep a fragment the evaluator will actually be able to find.
        if name and name in blob:
            frags.add(name)
    return frags


def _unique_frame_title(
    observations: list[dict[str, Any]], idxs: list[int], frame_index: int
) -> Optional[str]:
    """The work frame's document title, if every observation of this screen
    agrees on it. A title chosen by the application names the screen and
    does not vary with the record displayed on it, which makes it a far
    better screen landmark than any text picked out of the page body."""
    titles: set[str] = set()
    for i in idxs:
        ft = observations[i].get("frame_titles") or {}
        t = ft.get(str(frame_index))
        if t is None:
            t = ft.get(frame_index)
        t = (t or "").strip()
        if t:
            titles.add(t)
    if len(titles) == 1:
        return next(iter(titles))
    return None


def _find_same_element(obs: dict[str, Any], element_ref: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Find "the same" element in a different observation, purely from
    static data -- (frame_index, structural DOM path) is the closest thing
    to a stable identity a raw observation dict offers (see `Element` in
    `hands/surface/base.py`); it is what `structural` targets themselves
    rely on.
    """
    fi = element_ref.get("frame_index")
    structural = element_ref.get("structural")
    for e in obs.get("elements") or []:
        if e.get("frame_index") == fi and e.get("structural") == structural:
            return e
    return None


# ==========================================================================
# 1. Work frame identification + screens
# ==========================================================================


def _find_work_frame(observations: list[dict[str, Any]]) -> tuple[int, str]:
    """Identify the "work" frame: a CHILD frame (`parent_index is not
    None` -- excludes the top document) whose URL PATH takes more than one
    distinct value across the recorded observations (excludes chrome/nav
    frames, whose URL never changes).

    This is deliberately never the top document, even when candidates seem
    ambiguous: in a frameset the top document's URL never changes across
    screens, so keying screens off it would make wrong-screen detection
    structurally unfireable (every screen would look identical from that
    vantage point) -- see `Screen`'s and `Capability`'s own docstrings in
    `hands/schema/capability.py` for the same point made about replay.
    """
    urls_by_index: dict[int, set[str]] = {}
    parent_by_index: dict[int, Optional[int]] = {}
    name_by_index: dict[int, str] = {}
    for obs in observations:
        for f in obs.get("frames") or []:
            idx = f["index"]
            urls_by_index.setdefault(idx, set()).add(_path_of(f["url"]))
            parent_by_index[idx] = f.get("parent_index")
            if f.get("name"):
                name_by_index[idx] = f["name"]

    candidates = sorted(
        idx
        for idx, urls in urls_by_index.items()
        if parent_by_index.get(idx) is not None and len(urls) > 1
    )
    if not candidates:
        raise RecorderError(
            "could not identify a work frame: no child frame's URL path varies across "
            "the recorded observations. This recorder derives screens from a frameset "
            "layout (a menu/nav frame plus a work frame); a top-document-only run, or "
            "one where every frame's path is constant, gives it nothing to key screens on."
        )
    work_index = candidates[0]
    return work_index, name_by_index.get(work_index, "")


def _build_screens(
    observations: list[dict[str, Any]],
    work_index: int,
    work_name: str,
    volatile_values: Optional[set[str]] = None,
) -> tuple[dict[str, Screen], list[Optional[str]], dict[str, str], list[str]]:
    """Group observations into screens by the work frame's URL path, and
    derive each screen's landmarks by set difference: text present in
    EVERY observation of that screen and absent from every observation of
    every other screen.

    Returns (screens, obs_paths, path_to_id, warnings) where `obs_paths[i]`
    is the work-frame path observed at `observations[i]` (or `None` if that
    observation captured no work-frame data at all, e.g. a dialog was open)
    and `path_to_id` maps a path to the screen id it belongs to.
    """
    warnings: list[str] = []
    obs_paths: list[Optional[str]] = []
    for obs in observations:
        f = _frame_by_index(obs, work_index)
        obs_paths.append(_path_of(f["url"]) if f else None)

    # Carry the last known screen forward across an observation with no
    # work-frame data (e.g. a dialog was open) rather than dropping it --
    # such an observation still needs *some* screen for any step built
    # against it.
    last: Optional[str] = None
    for i, p in enumerate(obs_paths):
        if p is None:
            if last is not None:
                obs_paths[i] = last
                warnings.append(
                    f"observation {i} had no work-frame data (dialog open?); "
                    f"carried forward the previous screen {last!r}"
                )
        else:
            last = p

    path_to_id: dict[str, str] = {}
    obs_indices_by_path: dict[str, list[int]] = {}
    for i, p in enumerate(obs_paths):
        if p is None:
            continue
        if p not in path_to_id:
            path_to_id[p] = _screen_id_from_path(p)
        obs_indices_by_path.setdefault(p, []).append(i)

    frag_by_obs = [_element_fragments(obs, work_index) for obs in observations]
    volatile_values = volatile_values or set()

    screens: dict[str, Screen] = {}
    for path, sid in path_to_id.items():
        own_idxs = obs_indices_by_path[path]
        own_common = set.intersection(*(frag_by_obs[i] for i in own_idxs))
        other_idxs = [i for i in range(len(observations)) if i not in own_idxs]
        other_union: set[str] = set()
        for i in other_idxs:
            other_union |= frag_by_obs[i]
        distinctive = sorted(own_common - other_union, key=len)

        pattern = _path_pattern(path)
        frame_matcher = FrameMatcher(url_pattern=pattern)

        # A landmark has to identify the SCREEN, not the record that happened
        # to be on it during discovery. Text chosen purely by set difference
        # over a single run does the opposite: on a member detail screen the
        # most "distinctive" strings are the member's own name, number and
        # balances, so the capability would only ever recognise the screen
        # for the one member it was recorded against. That is the difference
        # between a recording and a capability, and it showed up as
        # UNEXPECTED_SCREEN the first time a recorded artifact was replayed
        # with a different member.
        #
        # The work frame's document title is preferred instead: it is chosen
        # by the application to name the screen, and it does not vary with
        # the record being viewed. Distinctive text is kept as a fallback,
        # with anything that matches a run input or a captured output
        # removed, since those are known to be per-record values.
        own_title = _unique_frame_title(observations, own_idxs, work_index)
        other_titles = {
            _unique_frame_title(observations, obs_indices_by_path[p2], work_index)
            for p2 in path_to_id
            if p2 != path
        }
        safe_text = [t for t in distinctive if t not in volatile_values]

        landmarks: list[Condition]
        if own_title and own_title not in other_titles:
            # The title alone, deliberately. Adding a page-text landmark on
            # top can only make the check more brittle, never more correct:
            # a single discovery run cannot tell text that identifies the
            # SCREEN from text that merely describes the RECORD that was on
            # it. Doing exactly that picked "LN" (the loan row of the member
            # used during discovery) as a member-screen landmark, so the
            # capability then rejected the member screen for every member
            # without a loan. A title is structural by construction: the
            # application chose it to name the screen.
            landmarks = [TitleMatches(pattern=re.escape(own_title), frame=frame_matcher)]
        elif safe_text:
            landmarks = [TextPresent(pattern=re.escape(t), frame=frame_matcher) for t in safe_text[:2]]
            warnings.append(
                f"screen {sid!r} has no distinctive work-frame title; using page text as a "
                "landmark, which is more likely to drift than a title"
            )
        else:
            titles = {observations[i].get("title", "") for i in own_idxs}
            title_text = next(iter(titles)) if len(titles) == 1 and next(iter(titles)) else path
            landmarks = [TitleMatches(pattern=re.escape(title_text), frame=frame_matcher)]
            warnings.append(
                f"screen {sid!r} has no work-frame text that is distinctive across the "
                "recorded run; falling back to a title_matches landmark"
            )

        screens[sid] = Screen(
            id=sid,
            description=f"Screen at work-frame path {path!r}",
            frame=frame_matcher,
            url_pattern=pattern,
            landmarks=landmarks,
        )

    return screens, obs_paths, path_to_id, warnings


def _build_fingerprint(observations: list[dict[str, Any]], work_index: int, work_name: str) -> list[Condition]:
    """App-level identification, as opposed to a `Screen`'s landmarks: text
    common to EVERY observation of the work frame regardless of which
    screen it's on (the app's own persistent chrome/branding) -- exactly
    the text a screen's own landmark computation excludes for being
    non-distinctive between screens.
    """
    frag_sets = [
        _element_fragments(obs, work_index)
        for obs in observations
        if _frame_by_index(obs, work_index) is not None
    ]
    if not frag_sets:
        return []
    common = set.intersection(*frag_sets)
    if not common:
        return []
    chosen = sorted(common, key=len)[0]
    frame = FrameMatcher(name=work_name) if work_name else FrameMatcher(index=work_index)
    return [TextPresent(pattern=re.escape(chosen), frame=frame)]


# ==========================================================================
# 2. Value binding for `type` steps
# ==========================================================================


def _bind_type_value(
    literal: str,
    inputs: dict[str, str],
    inputs_spec: dict[str, InputSpec],
    redactor: Any,
    step_label: str,
) -> tuple[str, Optional[str], bool]:
    """Reverse-match a typed literal against the run's inputs. Returns
    `(bound_value, warning_or_None, is_secret)`.

    Order, per the task spec:
      1. the model already used a `{{placeholder}}` -- keep it verbatim;
      2. the literal equals a known (non-sensitive) run input's actual
         value -- replace with `{{that_input_name}}`;
      3. the literal is a secret -- replace with `{{that_input_name}}` and
         NEVER let the raw value reach the return value. Two sub-cases:
         (a) the REALISTIC one: `hands.agent.loop.DiscoveryLoop` already
             redacts `tool_input` before it ever reaches a `TraceStep`
             (`tool_input=self.redactor.obj(tool_input)`), so by the time
             we see it, a secret is already the ASCII marker
             `"[redacted:name]"`, never the raw value -- detected here by
             pattern match;
         (b) a defensive fallback for a hand-built trace that was never
             passed through that redaction layer: ask `redactor.has_value`
             directly.
      4. otherwise: keep the literal, but say so in a warning (it is
         hard-coded to this one discovery run).

    Every branch that resolves to a placeholder RAISES `RecorderError`
    instead of emitting `{{name}}` for a name absent from `inputs_spec` --
    that would make `validate_semantics()` report an undeclared-input
    reference, i.e. emit a capability the schema itself considers
    incoherent.
    """
    literal = literal or ""

    if _PLACEHOLDER_RE.search(literal):
        names = _PLACEHOLDER_RE.findall(literal)
        for pname in names:
            if pname not in inputs_spec:
                raise RecorderError(f"{step_label}: value references undeclared input {{{{{pname}}}}}")
        sensitive = any(inputs_spec[n].sensitive for n in names)
        return literal, None, sensitive

    for iname, ival in inputs.items():
        if ival and literal == ival:
            if iname not in inputs_spec:
                raise RecorderError(
                    f"{step_label}: typed literal matches input {iname!r}, which is not in inputs_spec"
                )
            return f"{{{{{iname}}}}}", None, inputs_spec[iname].sensitive

    m = _REDACTED_MARKER_RE.match(literal)
    if m:
        name = m.group(1)
        if name not in inputs_spec:
            raise RecorderError(
                f"{step_label}: redacted value names input {name!r}, which is not in inputs_spec"
            )
        return f"{{{{{name}}}}}", None, True

    secret_name = redactor.has_value(literal)
    if secret_name is not None:
        if secret_name not in inputs_spec:
            raise RecorderError(
                f"{step_label}: secret value names input {secret_name!r}, which is not in inputs_spec"
            )
        return f"{{{{{secret_name}}}}}", None, True

    return literal, f"{step_label}: value {literal!r} is hard-coded (no matching input or placeholder)", False


def _is_secret_placeholder(value: Optional[str], inputs_spec: dict[str, InputSpec]) -> bool:
    if not value:
        return False
    names = _PLACEHOLDER_RE.findall(value)
    return any(inputs_spec[n].sensitive for n in names if n in inputs_spec)


# ==========================================================================
# 3. Targets
# ==========================================================================


def _pick_row_key(headers: list[str], cells: list[str], exclude: Optional[int]) -> tuple[int, str]:
    """Pick a column, other than `exclude`, to identify this row by. Prefers
    a header that looks like a type/kind discriminator (e.g. "Type" in an
    accounts table distinguishing SAV/CHK/LN rows), since that is the
    natural, human-legible way this table's rows are told apart; falls back
    to the first other column, then to column 0.
    """
    for i, h in enumerate(headers):
        if i == exclude:
            continue
        if "type" in h.lower():
            return i, (cells[i] if i < len(cells) else "")
    for i, h in enumerate(headers):
        if i != exclude and i < len(cells):
            return i, cells[i]
    return 0, (cells[0] if cells else "")


def _describe_element(el: dict[str, Any], screen_id: Optional[str] = None) -> str:
    """A one-line description of a target that a human reviewer can read.

    The artifact is meant to be reviewable, and "Element for trace step 3"
    tells a reviewer nothing. This says what the control actually is, the
    way a person would refer to it on screen.
    """
    role = (el.get("role") or "element").strip()
    name = (el.get("name") or "").strip()
    rc = el.get("row_ctx") or {}
    cells = list(rc.get("cells") or [])
    headers = list(rc.get("headers") or [])

    # For a data cell, name it by its COLUMN, not by the value that happened
    # to be in it during discovery. The target itself is resolved by matching
    # the column header and a distinguishing cell in the row, so describing
    # it by its recorded value would misrepresent the artifact to a reviewer
    # as being pinned to one record's data.
    label = ""
    if role == "cell" and name and headers and rc.get("row_index"):
        try:
            col = cells.index(name)
        except ValueError:
            col = -1
        if 0 <= col < len(headers) and headers[col]:
            label = f"the {headers[col]!r} cell"
    if not label:
        label = f"the {name!r} {role}" if name else f"the unnamed {role}"

    non_empty = [c for c in cells if c]
    if rc and rc.get("row_index") and non_empty and non_empty[0] != name:
        label += f" in the table row for {non_empty[0]!r}"
    if screen_id:
        label += f" on the {screen_id} screen"
    return label


def _build_target(
    el: dict[str, Any],
    policy: Any,
    description: str,
    warnings: list[str],
    step_label: str,
) -> TargetSpec:
    """Build an ordered, semantic-first `TargetSpec` for one perceived
    element. Order: role, text (links/buttons only), relative (row_ctx
    only), structural, bbox (only when policy allows coordinate fallback).
    """
    strategies: list[Strategy] = []
    role = el.get("role", "")
    name = (el.get("name") or "").strip()

    if name:
        strategies.append(
            Strategy(
                kind="role",
                params={"role": role, "name": name, "exact": True},
                rationale=(
                    f"the {role} named {name!r} gets its accessible name from the app's own "
                    "markup (an adjacent legacy-table label cell, or its own value/placeholder/"
                    "link text), not a generated id, so it keeps resolving as long as that "
                    "labelling text is unchanged"
                ),
            )
        )

    if role in ("link", "button") and name:
        strategies.append(
            Strategy(
                kind="text",
                params={"text": name, "exact": True},
                rationale=(
                    f"the visible {role} text {name!r} is authored page content; it survives a "
                    "markup refactor that drops or renames the element's own attributes, which "
                    "would break the role strategy above it"
                ),
            )
        )

    row_ctx = el.get("row_ctx")
    if row_ctx:
        headers = row_ctx.get("headers") or []
        cells = row_ctx.get("cells") or []
        value_col = cells.index(name) if name and name in cells else None
        key_col, key_val = _pick_row_key(headers, cells, value_col)
        if key_val:
            params: dict[str, Any] = {"anchor_text": key_val, "relation": "same_row"}
            if role and name:
                params["role"] = role
                params["name"] = name
            key_header = headers[key_col] if key_col < len(headers) else "a key column"
            strategies.append(
                Strategy(
                    kind="relative",
                    params=params,
                    rationale=(
                        f"anchored to the table row whose {key_header!r} cell reads {key_val!r}; "
                        "this survives the table being resorted or gaining/losing rows elsewhere, "
                        "unlike a positional row index"
                    ),
                )
            )

    structural = el.get("structural")
    if structural:
        strategies.append(
            Strategy(
                kind="structural",
                params={"path": structural},
                rationale=(
                    "a positional DOM path captured at discovery time; correct only until the "
                    "surrounding markup is edited, so it is ordered after every semantic strategy above"
                ),
            )
        )

    if getattr(policy, "allow_coordinate_fallback", False):
        pb = el.get("page_bbox") or {}
        strategies.append(
            Strategy(
                kind="bbox",
                params={
                    "x": pb.get("x", 0),
                    "y": pb.get("y", 0),
                    "w": pb.get("w", 0),
                    "h": pb.get("h", 0),
                    "viewport": _VIEWPORT,
                },
                rationale=(
                    "raw viewport coordinates from discovery time; the least stable strategy of "
                    "all (it survives no markup change whatsoever), kept only as a last resort "
                    "because policy explicitly allows coordinate fallback"
                ),
            )
        )
    else:
        warnings.append(
            f"{step_label}: coordinate-fallback strategy omitted "
            "(policy.allow_coordinate_fallback is False)"
        )

    if not strategies:
        raise RecorderError(f"{step_label}: could not build any target strategy for element {el!r}")

    frame_pattern = _path_pattern(_path_of(el.get("frame_url", "")))
    return TargetSpec(
        description=description,
        frame=FrameMatcher(url_pattern=frame_pattern),
        strategies=strategies,
        snapshot={"role": role, "name": name, "bbox": el.get("page_bbox"), "viewport": _VIEWPORT},
    )


def _build_table_extraction_target(el: dict[str, Any], step_label: str) -> TargetSpec:
    """The `target` for a `table_cell` extraction points at the TABLE
    itself, never at a specific cell -- resolving to a specific cell would
    silently reintroduce the row-order dependency `table_cell` exists to
    avoid (see `hands/surface/extract.py`'s module docstring). A `<table>`
    element carries no ARIA role at all (`perceive.js`'s SKIP set excludes
    `table`/`tr`/`tbody`/etc. from perception entirely), so structural
    (its DOM path) is the only strategy available to locate it; the actual
    value is then found by header+row match, not by this path.
    """
    row_ctx = el.get("row_ctx") or {}
    table_path = row_ctx.get("table_path")
    if not table_path:
        raise RecorderError(f"{step_label}: row_ctx has no table_path; cannot locate the enclosing table")
    frame_pattern = _path_pattern(_path_of(el.get("frame_url", "")))
    return TargetSpec(
        description="The table containing the extracted cell",
        frame=FrameMatcher(url_pattern=frame_pattern),
        strategies=[
            Strategy(
                kind="structural",
                params={"path": table_path},
                rationale=(
                    "the enclosing <table> has no accessible role or name (perceive.js never "
                    "perceives table/row elements themselves), so its DOM position is the only "
                    "way to locate it at all; table_cell then reads the actual value by header "
                    "and row match, not by this path, so the path only needs to find the table"
                ),
            )
        ],
        snapshot={"role": "table", "name": None, "bbox": None, "viewport": _VIEWPORT},
    )


# ==========================================================================
# 4. Checkpoints (build + pure lint)
# ==========================================================================


def _frame_entry(obs: dict[str, Any], matcher: Optional[FrameMatcher]) -> Optional[dict[str, Any]]:
    frames = obs.get("frames") or []
    if matcher is not None and matcher.url_pattern:
        for f in frames:
            if re.search(matcher.url_pattern, f.get("url", "")):
                return f
        return None
    if matcher is not None and matcher.name:
        for f in frames:
            if f.get("name") == matcher.name:
                return f
        return None
    if matcher is not None and matcher.index is not None:
        for f in frames:
            if f.get("index") == matcher.index:
                return f
        return None
    for f in frames:
        if f.get("parent_index") is None:
            return f
    return frames[0] if frames else None


def _lint_eval(
    condition: Condition,
    obs: dict[str, Any],
    element_ref: Optional[dict[str, Any]],
    default_frame: Optional[FrameMatcher] = None,
) -> Optional[bool]:
    """Pure evaluation of a `Condition` against one raw observation dict --
    no Surface, no browser. Supports exactly the condition kinds this
    recorder itself ever emits (`text_present`, `url_matches`,
    `title_matches`, `value_equals`) plus the boolean combinators wrapping
    them, since those are the only kinds a checkpoint or a discovered
    outcome is built from here.

    Returns `None`, never raises, for a condition kind this pure evaluator
    cannot check from a static observation dict (`dialog_open`,
    `element_visible`/`element_absent`, `output_matches` -- these need live
    browser/run state). Callers decide what `None` means for their own
    purposes: `_checkpoint_lints_ok` treats it as a lint failure (never
    trust an unverifiable checkpoint), `_outcome_survives_lint` treats it as
    "cannot disprove" (never silently drop an outcome we simply can't
    check).
    """
    kind = condition.type

    if kind == "text_present":
        f = _frame_entry(obs, condition.frame or default_frame)
        if f is None:
            return False
        text = (obs.get("frame_texts") or {}).get(str(f["index"]), "")
        found = re.search(condition.pattern, text) is not None
        return (not found) if condition.negate else found

    if kind == "url_matches":
        f = _frame_entry(obs, condition.frame or default_frame)
        if f is None:
            return False
        return re.search(condition.pattern, f.get("url", "")) is not None

    if kind == "title_matches":
        f = _frame_entry(obs, condition.frame or default_frame)
        if f is None:
            return False
        title = (obs.get("frame_titles") or {}).get(str(f["index"]), "")
        return re.search(condition.pattern, title) is not None

    if kind == "value_equals":
        if element_ref is None:
            return None
        el = _find_same_element(obs, element_ref)
        if el is None:
            return False
        val = (el.get("attrs") or {}).get("value", "")
        return val == condition.value

    if kind == "all_of":
        results = [_lint_eval(c, obs, element_ref, default_frame) for c in condition.conditions]
        if any(r is None for r in results):
            return None
        return all(results)

    if kind == "any_of":
        results = [_lint_eval(c, obs, element_ref, default_frame) for c in condition.conditions]
        if any(r is None for r in results):
            return None
        return any(results)

    if kind == "not_":
        r = _lint_eval(condition.not_, obs, element_ref, default_frame)
        return None if r is None else (not r)

    return None


def _checkpoint_lints_ok(
    condition: Condition,
    before_obs: dict[str, Any],
    after_obs: dict[str, Any],
    element_ref: Optional[dict[str, Any]],
) -> bool:
    """A checkpoint is proof that a step actually happened only if it is
    FALSE against the observation captured before the action and TRUE
    against the observation captured after it. A checkpoint that was
    already true before the action proves nothing about the action at all
    -- it would let replay "verify" a click that never happened. Fails
    closed (drops the checkpoint) when either observation can't be
    evaluated purely (`None`), since an unverifiable checkpoint is exactly
    as untrustworthy as a wrong one.
    """
    before = _lint_eval(condition, before_obs, element_ref)
    after = _lint_eval(condition, after_obs, element_ref)
    if before is None or after is None:
        return False
    return (before is False) and (after is True)


def _build_checkpoint(
    kind: str,
    is_sensitive: bool,
    before_obs: dict[str, Any],
    after_obs: Optional[dict[str, Any]],
    work_index: int,
    work_name: str,
    target: Optional[TargetSpec],
    literal_value: Optional[str],
    element_ref: Optional[dict[str, Any]],
    step_label: str,
    warnings: list[str],
) -> Optional[Checkpoint]:
    """Derive a checkpoint from what actually changed between the
    before/after observations, in order: work-frame URL path change, new
    distinctive work-frame text, then (only for `type`) the field's own
    value. Everything else -- including every `extract` step, which never
    mutates the page -- gets no checkpoint.

    COMPROMISE: for the `value_equals` fallback, the expected value is the
    literal actually typed at DISCOVERY time, not a placeholder -- the real
    replay evaluator (`hands/surface/conditions.py::_evaluate_value_equals`)
    does a raw string compare with no `{{placeholder}}` substitution, so a
    literal `{{name}}` stored here would never match a live field and would
    always fail. Pinning it to the discovery-time literal means the
    checkpoint won't generalise to a differently-parameterised replay input
    -- a real limitation, not an oversight; see the module footer.
    """
    if after_obs is None:
        warnings.append(f"{step_label}: no observation was captured after this action; no checkpoint")
        return None

    before_path = _work_path(before_obs, work_index)
    after_path = _work_path(after_obs, work_index)
    if after_path is not None and before_path != after_path:
        pattern = _path_pattern(after_path)
        frame = FrameMatcher(url_pattern=pattern)
        url_cond: Condition = UrlMatches(pattern=pattern, frame=frame)

        # A URL alone is too weak a checkpoint whenever a path serves more
        # than one outcome, which in a back-office app is most of them: a
        # member lookup that finds nothing, that hits a busy host, or that
        # 500s all still land on the member path. Accepting the step on the
        # URL then lets replay walk on and fail later with "unexpected
        # screen", which tells a caller only that something is wrong, never
        # which business outcome occurred -- and it means a declared outcome
        # like MEMBER_NOT_FOUND never gets the chance to fire, because the
        # checkpoint was already satisfied.
        #
        # So the checkpoint also asserts the destination's document title,
        # which is what distinguishes those pages from one another ("Member
        # Detail" vs "Member Inquiry" vs "System Busy" vs "Error"). It is the
        # same signal used to identify a screen, and for the same reason: the
        # application chose it to name the page, and it does not vary with
        # the record being displayed.
        after_title = _unique_frame_title([after_obs], [0], work_index)
        before_title = _unique_frame_title([before_obs], [0], work_index)
        if after_title and after_title != before_title:
            cond: Condition = AllOf(
                conditions=[
                    url_cond,
                    TitleMatches(pattern=re.escape(after_title), frame=frame),
                ]
            )
            return Checkpoint(
                condition=cond,
                description=f"work frame navigated to path {after_path!r} showing {after_title!r}",
            )
        return Checkpoint(condition=url_cond, description=f"work frame navigated to path {after_path!r}")

    before_frags = _element_fragments(before_obs, work_index)
    after_frags = _element_fragments(after_obs, work_index)
    new_text = sorted(after_frags - before_frags, key=len)
    if new_text:
        chosen = new_text[0]
        frame = FrameMatcher(name=work_name) if work_name else FrameMatcher(index=work_index)
        cond = TextPresent(pattern=re.escape(chosen), frame=frame)
        return Checkpoint(condition=cond, description=f"new text {chosen!r} appeared on the work frame")

    if kind == "type":
        if is_sensitive:
            warnings.append(
                f"{step_label}: no checkpoint recorded for this sensitive field -- comparing "
                "against the live value would mean storing (or matching against) the secret itself"
            )
            return None
        if literal_value is None or target is None or element_ref is None:
            warnings.append(f"{step_label}: no checkpoint (typed literal or target unavailable)")
            return None
        cond = ValueEquals(target=target, value=literal_value)
        return Checkpoint(
            condition=cond,
            description=(
                f"field holds the discovery-time value {literal_value!r} "
                "(pinned to the literal typed while recording; see recorder module docstring)"
            ),
        )

    warnings.append(f"{step_label}: nothing observably changed; no checkpoint recorded")
    return None


# ==========================================================================
# 5. Extraction
# ==========================================================================


def _sniff_parse(value: Optional[str]) -> ParseSpec:
    """A currency-looking string -> money, digits -> int, else string."""
    v = (value or "").strip()
    if re.fullmatch(r"-?\$?[\d,]+\.\d{2}", v):
        return ParseSpec(type="money", regex=r"-?[\d,]+\.\d+", group=0)
    if re.fullmatch(r"-?\d+", v):
        return ParseSpec(type="int")
    return ParseSpec(type="string")


def _build_extraction(
    el: dict[str, Any],
    policy: Any,
    output_name: str,
    raw_value: str,
    step_label: str,
    warnings: list[str],
) -> Extraction:
    row_ctx = el.get("row_ctx")
    parse = _sniff_parse(raw_value)

    if row_ctx:
        headers = row_ctx.get("headers") or []
        cells = row_ctx.get("cells") or []
        name = (el.get("name") or "").strip()
        value_col = cells.index(name) if name and name in cells else None
        if value_col is None or value_col >= len(headers):
            warnings.append(
                f"{step_label}: could not uniquely locate the extracted cell's own column in "
                "its row (its text does not appear exactly once among the row's cells); "
                "defaulting to column 0, which may be wrong"
            )
            value_col = 0
        header_text = headers[value_col] if value_col < len(headers) else ""
        key_col, key_val = _pick_row_key(headers, cells, value_col)
        target = _build_table_extraction_target(el, step_label)
        return Extraction(
            target=target,
            from_="table_cell",
            table=TableSelector(
                header_match=re.escape(header_text),
                row_where=RowWhere(
                    header=(headers[key_col] if key_col < len(headers) else ""),
                    matches=re.escape(key_val),
                ),
            ),
            parse=parse,
            output=output_name,
        )

    target = _build_target(el, policy, "the value read from " + _describe_element(el), warnings, step_label)
    return Extraction(target=target, from_="text", parse=parse, output=output_name)


# ==========================================================================
# 6. Outcomes
# ==========================================================================


def _stamp_default_frame(condition: Condition, frame_matcher: FrameMatcher) -> Condition:
    """Recursively fill in `.frame` on every `text_present`/`url_matches`/
    `title_matches` (the kinds `_resolve_frame_info` actually uses a
    condition's own `frame` for) that doesn't already carry one, with
    `frame_matcher`.

    This matters beyond linting: a library outcome's `detect` with no
    `frame` at all falls back, at REAL replay time, to the top document
    (see `_resolve_frame_info`'s docstring in `hands/surface/conditions.py`)
    -- which in a frameset has essentially no body text of its own, so such
    an outcome would never fire, ever, silently. The recorder is the one
    component that actually knows which frame is the work frame; leaving a
    library outcome's frame unset and hoping the replay-time default lines
    up is not an option here.
    """
    kind = condition.type
    if kind in ("text_present", "url_matches", "title_matches"):
        if condition.frame is None:
            return condition.model_copy(update={"frame": frame_matcher})
        return condition
    if kind in ("all_of", "any_of"):
        return condition.model_copy(
            update={"conditions": [_stamp_default_frame(c, frame_matcher) for c in condition.conditions]}
        )
    if kind == "not_":
        return condition.model_copy(update={"not_": _stamp_default_frame(condition.not_, frame_matcher)})
    return condition


def _build_discovered_outcomes(trace: DiscoveryTrace, work_name: str, work_index: int) -> list[Outcome]:
    frame = FrameMatcher(name=work_name) if work_name else FrameMatcher(index=work_index)
    outcomes: list[Outcome] = []
    seen: set[str] = set()
    for ts in trace.steps:
        if ts.tool != "note_condition":
            continue
        code = ts.tool_input.get("code")
        kind = ts.tool_input.get("kind")
        pattern = ts.tool_input.get("text_pattern")
        if not code or not kind or not pattern or code in seen:
            continue
        seen.add(code)
        outcomes.append(
            Outcome(
                code=code,
                kind=kind,
                origin="discovered",
                scope="global",
                detect=TextPresent(pattern=pattern, frame=frame),
                message=ts.reason or f"condition {code!r} observed during discovery",
                handling=Handling(action="stop"),
            )
        )
    return outcomes


def _ingest_library_outcomes(
    outcome_library: Optional[list[Any]], work_name: str, work_index: int
) -> list[Outcome]:
    frame = FrameMatcher(name=work_name) if work_name else FrameMatcher(index=work_index)
    result: list[Outcome] = []
    for item in outcome_library or []:
        o = item if isinstance(item, Outcome) else Outcome.model_validate(item)
        stamped = _stamp_default_frame(o.detect, frame)
        result.append(o.model_copy(update={"origin": "library", "detect": stamped}))
    return result


def _outcome_survives_lint(detect: Condition, observations: list[dict[str, Any]]) -> tuple[bool, bool]:
    """`(survives, fully_checked)`. `survives=False` means `detect` fired on
    at least one observation from the recorded run -- a loose pattern that
    also matches the success page would silently shadow that successful
    run's own outcome, so it must never reach the artifact.
    `fully_checked=False` flags that at least one observation used a
    condition kind this pure lint cannot evaluate (kept, not dropped: we
    never silently drop something we can't actually disprove).
    """
    fully_checked = True
    for obs in observations:
        r = _lint_eval(detect, obs, None)
        if r is None:
            fully_checked = False
            continue
        if r:
            return False, fully_checked
    return True, fully_checked


def _lint_outcomes(
    outcomes: list[Outcome],
    observations: list[dict[str, Any]],
    pruning_log: list[dict[str, Any]],
    warnings: list[str],
) -> list[Outcome]:
    survivors: list[Outcome] = []
    for o in outcomes:
        survives, fully_checked = _outcome_survives_lint(o.detect, observations)
        if not survives:
            pruning_log.append(
                {
                    "kind": "outcome",
                    "code": o.code,
                    "origin": o.origin,
                    "reason": "detect condition also matched an observation on the recorded success path",
                }
            )
            warnings.append(f"outcome {o.code!r} dropped: its detect condition fires on the success path too")
            continue
        if not fully_checked:
            warnings.append(
                f"outcome {o.code!r} kept, but its detect condition could not be fully verified "
                "against every observation (it uses a condition kind this pure lint can't check)"
            )
        survivors.append(o)
    return survivors


# ==========================================================================
# 7. Success
# ==========================================================================


def _build_success(
    final_obs: dict[str, Any],
    work_index: int,
    work_name: str,
    screens: dict[str, Screen],
    path_to_id: dict[str, str],
    captured_outputs: list[str],
) -> Checkpoint:
    path = _work_path(final_obs, work_index)
    screen = screens.get(path_to_id.get(path, "")) if path else None
    base_cond: Condition
    if screen is not None and screen.landmarks:
        base_cond = screen.landmarks[0]
    else:
        frags = sorted(_element_fragments(final_obs, work_index), key=len)
        frame = FrameMatcher(name=work_name) if work_name else FrameMatcher(index=work_index)
        base_cond = TextPresent(pattern=re.escape(frags[0]) if frags else r".+", frame=frame)

    if captured_outputs:
        conds: list[Condition] = [base_cond] + [
            OutputMatches(output=name, pattern=r".+") for name in captured_outputs
        ]
        condition: Condition = AllOf(conditions=conds)
        description = "final screen reached with expected content, and outputs captured"
    else:
        condition = base_cond
        description = "final screen reached with expected content"
    return Checkpoint(condition=condition, description=description)


# ==========================================================================
# Step-level helpers
# ==========================================================================


def _step_failed(ts: TraceStep) -> bool:
    ok = ts.action_result.get("ok", True) if ts.action_result else True
    return (ok is False) or (ts.error is not None)


def _policy_blocked(ts: TraceStep) -> bool:
    pd = ts.policy_decision
    return bool(pd) and pd.get("allowed") is False


def _derive_side_effects(curated_steps: list[Step]) -> str:
    if any(s.risk.level in ("risky", "irreversible") for s in curated_steps):
        return "modifies"
    return "reads"


def _derive_idempotent(side_effects: str) -> bool:
    return side_effects == "reads"


# ==========================================================================
# record()
# ==========================================================================


def record(
    trace: DiscoveryTrace,
    *,
    capability_id: str,
    name: str,
    description: str,
    inputs_spec: dict[str, InputSpec],
    outputs_spec: dict[str, OutputSpec],
    policy: Any,
    redactor: Any,
    outcome_library: Optional[list[Any]] = None,
    entry_url: str,
    app: AppInfo,
    discovery_run_id: str,
    evidence_ref: Optional[str] = None,
    # COMPROMISE: the run's actual input VALUES ("teller1", "12345", ...)
    # are needed to reverse-match a typed literal against a *non-sensitive*
    # input (step 2 of the task spec) -- `DiscoveryTrace` does not carry
    # `DiscoveryLoop.inputs` anywhere, and the documented signature has no
    # slot for it either. Exposed here as an optional keyword rather than
    # silently only ever matching via the redaction marker / secret path.
    inputs: Optional[dict[str, str]] = None,
    # COMPROMISE: `Contract.side_effects`/`idempotent` are required schema
    # fields with no derivation rule given and no slot in the documented
    # signature. Auto-derived from step risk when not given (see
    # _derive_side_effects/_derive_idempotent); pass explicitly to override.
    side_effects: Optional[str] = None,
    idempotent: Optional[bool] = None,
    # COMPROMISE: `Provenance.recorded_against` is a required field the
    # documented signature has no slot for either; defaulted to a fixed
    # viewport/browser pair (matching this project's own test fixtures).
    recorded_against: Optional[dict[str, Any]] = None,
) -> RecordResult:
    """Turn a successful `DiscoveryTrace` into a `Capability`.

    A "successful action" (the only kind that can end up in the curated
    capability) is one where the action reported ok (`action_result.ok is
    not False` AND `error is None` -- `read_value` never overwrites
    `action_result`, so its own `error` field is the only signal for it)
    AND it is on the path to `done`: since `trace` is one linear recorded
    sequence (no branching/backtracking model in this system) and `record`
    requires `trace.status == "success"`, every such step in `trace.steps`
    -- in order, up to and including whichever step precedes `done` -- IS,
    by construction, the path that led to `done`. There is nothing to
    branch away from within a single trace; a step is excluded from the
    curated capability only for failing or being policy-blocked, never for
    "having happened but not counting".
    """
    if trace.status != "success":
        raise RecorderError(
            f"record() only turns a successful discovery run into a capability artifact; "
            f"trace.status={trace.status!r}"
        )
    observations = trace.observations
    if not observations:
        raise RecorderError("trace has no observations to record from")

    warnings: list[str] = []
    pruning_log: list[dict[str, Any]] = []
    inputs = dict(inputs or {})

    work_index, work_name = _find_work_frame(observations)
    # Values known to vary per record (the run's own inputs and anything the
    # run captured as an output) must not become screen landmarks: they
    # identify the record, not the screen.
    volatile_values: set[str] = set()
    for _v in (inputs or {}).values():
        if _v:
            volatile_values.add(str(_v).strip())
    for _v in (trace.outputs or {}).values():
        if _v:
            volatile_values.add(str(_v).strip())
    screens, obs_paths, path_to_id, screen_warnings = _build_screens(
        observations, work_index, work_name, volatile_values
    )
    warnings.extend(screen_warnings)

    raw_steps: list[Step] = []
    curated_steps: list[Step] = []
    output_source_raw: dict[str, str] = {}
    output_source_curated: dict[str, str] = {}
    output_values: dict[str, Any] = dict(trace.outputs or {})

    raw_counter = 0
    curated_counter = 0

    for ts in trace.steps:
        kind = _TOOL_KIND.get(ts.tool)
        if kind is None:
            continue  # note_condition / request_human / done -- not steps

        step_label = f"trace step {ts.index} ({ts.tool})"
        path = obs_paths[ts.observation_index] if ts.observation_index < len(obs_paths) else None
        screen_id = path_to_id.get(path) if path is not None else None
        if screen_id is None:
            warnings.append(f"{step_label}: could not determine a screen for this step; dropping it entirely")
            continue

        el = ts.element
        target: Optional[TargetSpec] = None
        if el is not None:
            target = _build_target(el, policy, _describe_element(el, screen_id), warnings, step_label)

        value: Optional[str] = None
        is_sensitive = False
        literal_typed: Optional[str] = None
        if kind == "type":
            literal_raw = ts.tool_input.get("text", "")
            value, bind_warn, is_sensitive = _bind_type_value(literal_raw, inputs, inputs_spec, redactor, step_label)
            if bind_warn:
                warnings.append(bind_warn)
            literal_typed = None if is_sensitive else literal_raw
        elif kind == "select":
            value = ts.tool_input.get("option")
        elif kind == "press":
            value = ts.tool_input.get("key")
        elif kind == "navigate":
            value = ts.tool_input.get("url")

        element_name = el.get("name") if el else None
        element_attrs = el.get("attrs") if el else None
        level, reason = policy.classify_risk(kind, element_name, element_attrs)
        risk = Risk(level=level, reason=reason)

        before_obs = observations[ts.observation_index]
        after_obs = (
            observations[ts.observation_index + 1]
            if ts.observation_index + 1 < len(observations)
            else None
        )

        checkpoint: Optional[Checkpoint] = None
        if kind != "extract":
            checkpoint = _build_checkpoint(
                kind, is_sensitive, before_obs, after_obs, work_index, work_name,
                target, literal_typed, el, step_label, warnings,
            )
        else:
            warnings.append(f"{step_label}: extract steps get no checkpoint (nothing on the page changes)")

        extract: Optional[Extraction] = None
        output_name: Optional[str] = None
        if kind == "extract":
            output_name = ts.tool_input.get("output_name")
            if not output_name or el is None:
                warnings.append(f"{step_label}: read_value with no output_name/element; dropping")
                continue
            raw_value = output_values.get(output_name, "")
            extract = _build_extraction(el, policy, output_name, raw_value, step_label, warnings)

        intent = ((ts.reason or "").strip() or f"{kind} step").splitlines()[0][:160]

        raw_counter += 1
        raw_id = f"s{raw_counter}"
        raw_step = Step(
            id=raw_id, kind=kind, screen=screen_id, intent=intent, origin="agent",
            target=target, value=value, wait=WaitSpec(), checkpoint=checkpoint,
            extract=extract, risk=risk,
        )
        raw_steps.append(raw_step)
        if output_name:
            output_source_raw[output_name] = raw_id

        failed = _step_failed(ts)
        blocked = _policy_blocked(ts)
        if failed or blocked:
            pruning_log.append(
                {
                    "kind": "step",
                    "trace_step_index": ts.index,
                    "tool": ts.tool,
                    "reason": (
                        f"blocked by policy: {ts.error or 'denied'}"
                        if blocked
                        else f"action failed: {ts.error or 'unknown error'}"
                    ),
                }
            )
            continue

        curated_checkpoint = checkpoint
        curated_wait = WaitSpec()
        if checkpoint is not None:
            if _checkpoint_lints_ok(checkpoint.condition, before_obs, after_obs, el):
                curated_wait = WaitSpec(until=checkpoint.condition, timeout_ms=15000)
            else:
                pruning_log.append(
                    {
                        "kind": "checkpoint",
                        "trace_step_index": ts.index,
                        "reason": "checkpoint condition was already true before the action executed; proves nothing",
                    }
                )
                warnings.append(f"{step_label}: checkpoint dropped (already true before the action)")
                curated_checkpoint = None
                curated_wait = WaitSpec()

        curated_counter += 1
        curated_id = f"s{curated_counter}"
        curated_step = raw_step.model_copy(
            update={"id": curated_id, "checkpoint": curated_checkpoint, "wait": curated_wait}
        )
        curated_steps.append(curated_step)
        if output_name:
            output_source_curated[output_name] = curated_id

    # -------------------------------------------------------------- outcomes
    discovered = _build_discovered_outcomes(trace, work_name, work_index)
    library = _ingest_library_outcomes(outcome_library, work_name, work_index)
    raw_outcomes = discovered + library
    curated_outcomes = _lint_outcomes(raw_outcomes, observations, pruning_log, warnings)

    # -------------------------------------------------------- outputs/contract
    curated_outputs = {
        oname: ospec.model_copy(update={"source_step": output_source_curated.get(oname)})
        for oname, ospec in outputs_spec.items()
    }
    raw_outputs = {
        oname: ospec.model_copy(update={"source_step": output_source_raw.get(oname)})
        for oname, ospec in outputs_spec.items()
    }

    resolved_side_effects = side_effects or _derive_side_effects(curated_steps)
    resolved_idempotent = idempotent if idempotent is not None else _derive_idempotent(resolved_side_effects)

    curated_contract = Contract(
        inputs=inputs_spec,
        outputs=curated_outputs,
        outcome_codes=[o.code for o in curated_outcomes],
        side_effects=resolved_side_effects,
        idempotent=resolved_idempotent,
    )
    raw_contract = Contract(
        inputs=inputs_spec,
        outputs=raw_outputs,
        outcome_codes=[o.code for o in raw_outcomes],
        side_effects=resolved_side_effects,
        idempotent=resolved_idempotent,
    )

    # ------------------------------------------------------------------ success
    captured_names = [name_ for name_, spec in curated_outputs.items() if spec.source_step]
    success = _build_success(observations[-1], work_index, work_name, screens, path_to_id, captured_names)

    # --------------------------------------------------------------- target app
    fingerprint = _build_fingerprint(observations, work_index, work_name)
    target_app = TargetApp(app=app, entry_url=entry_url, fingerprint=fingerprint)

    # ---------------------------------------------------------- capability policy
    #
    # Two details here are load-bearing, and both were found by replaying a
    # recorded artifact rather than by reading the code:
    #
    # 1. The domain must be the HOSTNAME, not netloc. Policy.check_url
    #    compares against urlsplit().hostname, which is port-free, so a
    #    "127.0.0.1:5050" entry can never match anything. Because the
    #    effective policy is the INTERSECTION of the operator policy and
    #    this one, a host:port entry silently empties the domain set and
    #    blocks every action in the run.
    # 2. `navigate` must always be permitted, even when no navigate STEP was
    #    recorded. Replay always navigates to the entry URL before the first
    #    step, so deriving the action list purely from recorded step kinds
    #    locks the capability out of its own starting page.
    domain = urlsplit(entry_url).hostname or ""
    used_actions = {s.kind for s in curated_steps}
    used_actions.add("navigate")
    cap_policy = CapabilityPolicy(
        allowed_domains=[domain] if domain else [],
        allowed_actions=sorted(used_actions),
        allow_coordinate_fallback=bool(getattr(policy, "allow_coordinate_fallback", False)),
    )

    # ------------------------------------------------------------------ provenance
    resolved_recorded_against = recorded_against or {"viewport": _VIEWPORT, "browser": "chromium"}
    provenance = Provenance(
        source_goal=trace.goal,
        discovery_run_id=discovery_run_id,
        model=trace.model or "unknown",
        prompt_version=trace.prompt_version,
        recorded_against=resolved_recorded_against,
        evidence_ref=evidence_ref,
        raw_artifact_ref="capability.raw.json",
    )

    capability = Capability(
        id=capability_id,
        name=name,
        description=description,
        approval=Approval(status="draft"),
        contract=curated_contract,
        target=target_app,
        screens=screens,
        steps=curated_steps,
        outcomes=curated_outcomes,
        success=success,
        policy=cap_policy,
        provenance=provenance,
    ).with_content_hash()

    problems = capability.validate_semantics()
    if problems:
        raise RecorderError(
            "recorded capability failed validate_semantics(): " + "; ".join(problems)
        )

    raw_capability = Capability(
        id=capability_id,
        name=name,
        description=(
            description
            + " (raw/unpruned: includes failed and policy-blocked steps, and every "
            "checkpoint/outcome before linting)"
        ),
        approval=Approval(status="draft"),
        contract=raw_contract,
        target=target_app,
        screens=screens,
        steps=raw_steps,
        outcomes=raw_outcomes,
        success=success,
        policy=cap_policy,
        provenance=provenance.model_copy(update={"raw_artifact_ref": None}),
    ).with_content_hash()

    return RecordResult(
        capability=capability,
        raw_capability=raw_capability,
        pruning_log=pruning_log,
        warnings=warnings,
    )
