"""Tests for hands.agent.recorder.

The fixture below hand-builds a `DiscoveryTrace` (no browser, no network) that
mirrors a realistic LegacyCore-style frameset run: sign on (type username,
type password-as-secret, click Sign On) -> member lookup (type member
number, a FAILED search click, type a hard-coded "Last Name" note, a
REDUNDANT retype of that same note, then a successful search retry) ->
member detail (a note_condition, a POLICY-BLOCKED click on "Open
Sub-Account", then read the savings balance off a multi-row accounts
table) -> done.

Observations are built incrementally with `add_step`, which always appends
a TraceStep with `observation_index == len(observations) - 1` and then
appends the next observation (mirroring exactly what
`hands.agent.loop.DiscoveryLoop` does per turn: re-observe unless the turn
terminated the run) -- this keeps the before/after observation pairing
mechanically correct instead of hand-numbered.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import pytest

from hands.agent.loop import DiscoveryTrace, TraceStep
from hands.agent.recorder import RecorderError, record
from hands.safety.policy import Policy
from hands.safety.redact import Redactor
from hands.schema.capability import AppInfo, Capability, InputSpec, OutputSpec

# --------------------------------------------------------------------------
# Constants mirroring a LegacyCore-style frameset
# --------------------------------------------------------------------------

BASE = "http://127.0.0.1:5050"
FRAMESET_URL = BASE + "/"
MENU_URL = BASE + "/nav"
LOGIN_URL = BASE + "/main"
LOOKUP_URL = BASE + "/lookup"
MEMBER_URL = BASE + "/member?mno=12345"

TOP_INDEX = 0
MENU_INDEX = 1
WORK_INDEX = 2

TOP_TITLE = "LegacyCore Teller Workstation"
MENU_TITLE = "Menu"

ACCOUNTS_HEADERS = ["Acct No", "Type", "Curr Bal", "Avail Bal", "Status"]
ACCOUNTS = [
    ["0012345-S1", "SAV", "2,340.55", "2,300.00", "Open"],
    ["0012345-C1", "CHK", "812.10", "762.10", "Open"],
    ["0012345-L1", "LN", "-4,500.00", "0.00", "Open"],
]
TABLE_PATH = "member>table:accounts"

PASSWORD_LITERAL = "Passw0rd!"


# --------------------------------------------------------------------------
# Observation-building helpers
# --------------------------------------------------------------------------


def _frame(index: int, name: str, url: str, parent: Optional[int]) -> dict[str, Any]:
    return {"index": index, "name": name, "url": url, "parent_index": parent, "offset_x": 0.0, "offset_y": 0.0}


def _menu_elements() -> list[dict[str, Any]]:
    specs = [
        ("cell", "LegacyCore", "td"),
        ("link", "Home", "a"),
        ("link", "Member Inquiry", "a"),
        ("link", "Sign Off", "a"),
        ("cell", "Quick Search", "td"),
        ("textbox", "mno", "input"),
        ("button", "Search", "input"),
    ]
    out = []
    for i, (role, nm, tag) in enumerate(specs):
        out.append(
            {
                "idx": 0,
                "role": role,
                "name": nm,
                "tag": tag,
                "frame_index": MENU_INDEX,
                "frame_url": MENU_URL,
                "bbox": {"x": 0, "y": i * 14, "w": 60, "h": 12},
                "page_bbox": {"x": 0, "y": i * 14, "w": 60, "h": 12},
                "structural": f"menu>el:nth-of-type({i + 1})",
                "row_ctx": None,
                "attrs": {},
            }
        )
    return out


def _el(
    role: str,
    name: str,
    tag: str,
    structural: str,
    work_url: str,
    attrs: Optional[dict[str, Any]] = None,
    row_ctx: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    bbox = {"x": 10, "y": 10, "w": 80, "h": 14}
    return {
        "idx": 0,
        "role": role,
        "name": name,
        "tag": tag,
        "frame_index": WORK_INDEX,
        "frame_url": work_url,
        "bbox": bbox,
        "page_bbox": bbox,
        "structural": structural,
        "row_ctx": row_ctx,
        "attrs": attrs or {},
    }


def _login_elements(work_url: str, uid_val: str, pwd_val: str) -> list[dict[str, Any]]:
    return [
        _el("cell", "Teller Sign On", "td", "login>heading", work_url),
        _el("cell", "LegacyCore CBS 7.2", "td", "login>banner", work_url),
        _el("textbox", "Username", "input", "login>uid", work_url, attrs={"type": "text", "name": "uid", "value": uid_val}),
        _el(
            "textbox", "Password", "input", "login>pwd", work_url,
            attrs={"type": "password", "name": "pwd", "value": ("[masked]" if pwd_val else "")},
        ),
        _el("button", "Sign On", "input", "login>submit", work_url, attrs={"type": "submit", "value": "Sign On"}),
    ]


def _lookup_elements(work_url: str, mno_val: str, lnm_val: str) -> list[dict[str, Any]]:
    return [
        _el("cell", "Member Inquiry", "td", "lookup>heading", work_url),
        _el("cell", "LegacyCore CBS 7.2", "td", "lookup>banner", work_url),
        _el("cell", "Enter a member number to display the member record.", "td", "lookup>hint", work_url),
        _el("textbox", "Member No.", "input", "lookup>mno", work_url, attrs={"type": "text", "name": "mno", "value": mno_val}),
        _el("textbox", "Last Name", "input", "lookup>lnm", work_url, attrs={"type": "text", "name": "lnm", "value": lnm_val}),
        _el("button", "Search", "input", "lookup>submit", work_url, attrs={"type": "submit", "value": "Search"}),
        _el("button", "Clear", "input", "lookup>reset", work_url, attrs={"type": "reset", "value": "Clear"}),
    ]


def _account_row_elements(work_url: str, row_index: int, values: list[str]) -> list[dict[str, Any]]:
    return [
        _el(
            "cell", values[c], "td", f"member>table:accounts>row{row_index}>col{c}", work_url,
            row_ctx={"row_index": row_index, "headers": list(ACCOUNTS_HEADERS), "cells": list(values), "table_path": TABLE_PATH},
        )
        for c in range(len(values))
    ]


def _member_elements(work_url: str) -> list[dict[str, Any]]:
    els = [
        _el("cell", "Member Detail", "td", "member>heading", work_url),
        _el("cell", "LegacyCore CBS 7.2", "td", "member>banner", work_url),
        _el("cell", "Member No.", "td", "member>lbl1", work_url),
        _el("cell", "12345", "td", "member>val1", work_url),
        _el("cell", "Status", "td", "member>lbl2", work_url),
        _el("cell", "Active", "td", "member>val2", work_url),
        _el("cell", "Name", "td", "member>lbl3", work_url),
        _el("cell", "Jane Q. Member", "td", "member>val3", work_url),
        _el("cell", "Member Since", "td", "member>lbl4", work_url),
        _el("cell", "03/14/2011", "td", "member>val4", work_url),
        _el("cell", "Accounts", "td", "member>accountsheading", work_url),
    ]
    for i, row in enumerate(ACCOUNTS, start=1):
        els.extend(_account_row_elements(work_url, i, row))
    els.extend(
        [
            _el("link", "Open Sub-Account", "a", "member>opensub", work_url, attrs={"href": "/subaccount/new?mno=12345"}),
            _el("link", "New Inquiry", "a", "member>newinquiry", work_url, attrs={"href": "/lookup"}),
            _el("link", "Print", "a", "member>print", work_url, attrs={"href": "javascript:window.print()"}),
        ]
    )
    return els


def _obs(work_url: str, work_title: str, work_elements: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [
        _frame(TOP_INDEX, "", FRAMESET_URL, None),
        _frame(MENU_INDEX, "menu", MENU_URL, TOP_INDEX),
        _frame(WORK_INDEX, "work", work_url, TOP_INDEX),
    ]
    all_elements = _menu_elements() + list(work_elements)
    idx = 1
    for e in all_elements:
        e["idx"] = idx
        idx += 1

    titles = {TOP_INDEX: TOP_TITLE, MENU_INDEX: MENU_TITLE, WORK_INDEX: work_title}
    frame_texts: dict[str, str] = {}
    frame_titles: dict[str, str] = {}
    for f in frames:
        fi = f["index"]
        names = [e["name"] for e in all_elements if e["frame_index"] == fi and e.get("name")]
        frame_texts[str(fi)] = " ".join(names)
        frame_titles[str(fi)] = titles.get(fi, "")

    return {
        "url": FRAMESET_URL,
        "title": TOP_TITLE,
        "frames": frames,
        "elements": all_elements,
        "frame_texts": frame_texts,
        "frame_titles": frame_titles,
        "dialog": None,
        "screenshot_path": None,
        "ts": "2026-09-15T00:00:00.000Z",
    }


def find_el(obs: dict[str, Any], name: str, frame_index: int = WORK_INDEX) -> dict[str, Any]:
    for e in obs["elements"]:
        if e["frame_index"] == frame_index and e["name"] == name:
            return e
    raise KeyError(name)


# --------------------------------------------------------------------------
# Trace + Policy + Redactor fixtures
# --------------------------------------------------------------------------


def _policy() -> Policy:
    return Policy.from_dict(
        {
            "allowed_domains": ["127.0.0.1", "localhost"],
            "allowed_url_patterns": [r"^http://(127\.0\.0\.1|localhost):5050(/.*)?$"],
            "allowed_actions": ["navigate", "click", "type", "select", "press", "extract", "wait"],
            "risky_patterns": [r"(?i)\b(open|create|submit|confirm|transfer|close|delete|approve|post)\b"],
            "risky_mode": "block",
            "allow_coordinate_fallback": True,
            "capture_human_input_values": False,
            "human_wait_timeout_s": 600,
            "max_steps": 40,
            "step_timeout_ms": 10000,
            "run_timeout_s": 300,
            "recovery_budget": 5,
        }
    )


class _TraceBuilder:
    """Mechanically keeps `TraceStep.observation_index` correct: every
    non-terminal step consumes the CURRENT last observation as "before" and
    then appends the next observation, exactly like
    `hands.agent.loop.DiscoveryLoop` does per model turn (one tool call per
    turn, in this fixture).
    """

    def __init__(self, first_obs: dict[str, Any]) -> None:
        self.observations: list[dict[str, Any]] = [first_obs]
        self.steps: list[TraceStep] = []

    def add(
        self,
        tool: str,
        tool_input: dict[str, Any],
        reason: str,
        element: Optional[dict[str, Any]],
        *,
        ok: bool = True,
        error: Optional[str] = None,
        policy_decision: Optional[dict[str, Any]] = None,
        terminal: bool = False,
        next_obs: Optional[dict[str, Any]] = None,
    ) -> TraceStep:
        obs_index = len(self.observations) - 1
        before = self.observations[obs_index]
        action_result = {
            "ok": ok,
            "error": error,
            "duration_ms": 5,
            "url_after": before["frames"][WORK_INDEX]["url"],
            "dialogs": [],
        }
        ts = TraceStep(
            index=len(self.steps),
            tool=tool,
            tool_input=tool_input,
            reason=reason,
            element=element,
            action_result=action_result,
            url_before=before["url"],
            url_after=action_result["url_after"],
            screenshot=None,
            observation_index=obs_index,
            policy_decision=policy_decision,
            error=error,
        )
        self.steps.append(ts)
        if not terminal:
            assert next_obs is not None, "a non-terminal step must supply next_obs"
            self.observations.append(next_obs)
        return ts


@pytest.fixture
def built() -> dict[str, Any]:
    obs0 = _obs(LOGIN_URL, "Sign On", _login_elements(LOGIN_URL, "", ""))
    b = _TraceBuilder(obs0)

    # s: type username
    obs1 = _obs(LOGIN_URL, "Sign On", _login_elements(LOGIN_URL, "teller1", ""))
    b.add("type_text", {"element": 1, "text": "teller1", "reason": "enter the teller username"},
          "enter the teller username", find_el(obs0, "Username"), next_obs=obs1)

    # s: type password (already redacted, mirroring DiscoveryLoop's own Redactor)
    obs2 = _obs(LOGIN_URL, "Sign On", _login_elements(LOGIN_URL, "teller1", "x"))
    b.add("type_text", {"element": 1, "text": "[redacted:password]", "reason": "enter the teller password"},
          "enter the teller password", find_el(obs1, "Password"), next_obs=obs2)

    # s: click Sign On -> lookup screen
    obs3 = _obs(LOOKUP_URL, "Member Inquiry", _lookup_elements(LOOKUP_URL, "", ""))
    b.add("click", {"element": 1, "reason": "submit the login form"},
          "submit the login form", find_el(obs2, "Sign On"), next_obs=obs3)

    # s: type member number
    obs4 = _obs(LOOKUP_URL, "Member Inquiry", _lookup_elements(LOOKUP_URL, "12345", ""))
    b.add("type_text", {"element": 1, "text": "12345", "reason": "enter the member number to search for"},
          "enter the member number to search for", find_el(obs3, "Member No."), next_obs=obs4)

    # s: click Search -- FAILS (nothing changes)
    obs5 = _obs(LOOKUP_URL, "Member Inquiry", _lookup_elements(LOOKUP_URL, "12345", ""))
    b.add("click", {"element": 1, "reason": "run the member search"}, "run the member search",
          find_el(obs4, "Search"), ok=False, error="timeout: button not clickable", next_obs=obs5)

    # s: type a hard-coded note into Last Name (matches no input -- exercises the "keep literal, warn" branch)
    obs6 = _obs(LOOKUP_URL, "Member Inquiry", _lookup_elements(LOOKUP_URL, "12345", "Member"))
    b.add("type_text", {"element": 1, "text": "Member", "reason": "note the expected surname"},
          "note the expected surname", find_el(obs5, "Last Name"), next_obs=obs6)

    # s: REDUNDANT retype of the same note -- nothing changes -> checkpoint-lint-drop case
    obs7 = _obs(LOOKUP_URL, "Member Inquiry", _lookup_elements(LOOKUP_URL, "12345", "Member"))
    b.add("type_text", {"element": 1, "text": "Member", "reason": "re-confirm the surname note"},
          "re-confirm the surname note", find_el(obs6, "Last Name"), next_obs=obs7)

    # s: click Search -- succeeds, lands on member detail
    obs8 = _obs(MEMBER_URL, "Member Detail", _member_elements(MEMBER_URL))
    b.add("click", {"element": 1, "reason": "retry the member search"}, "retry the member search",
          find_el(obs7, "Search"), next_obs=obs8)

    # s: note_condition -- discovered outcome, never fires on this run's own observations
    obs9 = _obs(MEMBER_URL, "Member Detail", _member_elements(MEMBER_URL))
    b.add(
        "note_condition",
        {"code": "session_risk", "kind": "transient", "text_pattern": "Your session has expired",
         "reason": "flagging a possible session-risk banner just in case"},
        "flagging a possible session-risk banner just in case", None, next_obs=obs9,
    )

    # s: click "Open Sub-Account" -- POLICY BLOCKED (risky pattern "open")
    policy = _policy()
    opensub = find_el(obs9, "Open Sub-Account")
    decision = policy.check_action("click", MEMBER_URL, element_name="Open Sub-Account", element_attrs=opensub["attrs"])
    assert not decision.allowed
    obs10 = _obs(MEMBER_URL, "Member Detail", _member_elements(MEMBER_URL))
    b.add(
        "click", {"element": 1, "reason": "open a sub-account for this member"},
        "open a sub-account for this member", opensub, ok=False, error=decision.detail,
        policy_decision={"allowed": decision.allowed, "rule": decision.rule, "detail": decision.detail,
                          "requires_confirmation": decision.requires_confirmation},
        next_obs=obs10,
    )

    # s: read_value savings balance
    obs11 = _obs(MEMBER_URL, "Member Detail", _member_elements(MEMBER_URL))
    sav_cell = find_el(obs10, "2,340.55")
    b.add("read_value", {"element": 1, "output_name": "savings_balance", "reason": "capture the savings balance"},
          "capture the savings balance", sav_cell, next_obs=obs11)

    # s: done -- terminal, no further observation
    b.add(
        "done",
        {"outputs": [{"name": "savings_balance", "value": "2,340.55"}],
         "summary": "Captured the savings balance for member 12345.", "reason": "goal accomplished"},
        "goal accomplished", None, terminal=True,
    )

    trace = DiscoveryTrace(
        goal="Look up member 12345's savings balance",
        status="success",
        steps=b.steps,
        outputs={"savings_balance": "2,340.55"},
        summary="Captured the savings balance for member 12345.",
        stop_reason="done",
        observations=b.observations,
        intervention=None,
        model="claude-test-model",
        prompt_version="v1",
        usage_total={},
        started_at="2026-09-15T00:00:00.000Z",
        finished_at="2026-09-15T00:01:00.000Z",
    )
    return {"trace": trace, "policy": policy}


def _record_result(built: dict[str, Any]):
    trace = built["trace"]
    policy = built["policy"]
    redactor = Redactor()
    redactor.add("password", PASSWORD_LITERAL)

    inputs_spec = {
        "username": InputSpec(type="string", description="Teller login username", example="teller1"),
        "password": InputSpec(type="string", description="Teller login password", sensitive=True, source="vault"),
        "member_id": InputSpec(type="string", description="Member id to search for", pattern=r"^\d{5}$", example="12345"),
    }
    outputs_spec = {
        "savings_balance": OutputSpec(type="money", description="The member's current savings balance"),
    }
    app = AppInfo(vendor="LegacyCore", product="Teller Console", version="3.2")
    outcome_library = [
        {
            "code": "MEMBER_NOT_FOUND", "kind": "business", "origin": "library", "scope": "global",
            "detect": {"type": "text_present", "pattern": "No member found"},
            "message": "No member exists with that id.",
            "handling": {"action": "stop"},
        },
        {
            # deliberately loose: also matches "Member Detail"/"Member No." on the success page
            "code": "SYSTEM_BUSY", "kind": "transient", "origin": "library", "scope": "global",
            "detect": {"type": "text_present", "pattern": r"Member\b"},
            "message": "The system appears busy; retry the lookup.",
            "handling": {"action": "reload"},
        },
    ]

    return record(
        trace,
        capability_id="legacycore.lookup_savings_balance",
        name="Look up savings balance",
        description="Signs on, looks up a member by id, and reads their savings balance.",
        inputs_spec=inputs_spec,
        outputs_spec=outputs_spec,
        policy=policy,
        redactor=redactor,
        outcome_library=outcome_library,
        entry_url=FRAMESET_URL,
        app=app,
        discovery_run_id="disc-0001",
        inputs={"username": "teller1", "member_id": "12345"},
    )


# --------------------------------------------------------------------------
# 1. validates + validate_semantics() == []
# --------------------------------------------------------------------------


def test_capability_validates(built: dict[str, Any]) -> None:
    result = _record_result(built)
    assert result.capability.validate_semantics() == []


# --------------------------------------------------------------------------
# 2. screens keyed on the work frame, not the top document
# --------------------------------------------------------------------------


def test_screen_frame_targets_work_not_top(built: dict[str, Any]) -> None:
    result = _record_result(built)
    screen = result.capability.screens["lookup"]
    pattern = screen.frame.url_pattern
    assert pattern is not None
    assert re.search(pattern, LOOKUP_URL) is not None
    assert re.search(pattern, FRAMESET_URL) is None


# --------------------------------------------------------------------------
# 3. member number parameterised
# --------------------------------------------------------------------------


def test_member_id_is_parameterised(built: dict[str, Any]) -> None:
    result = _record_result(built)
    values = [s.value for s in result.capability.steps]
    assert "{{member_id}}" in values
    assert "12345" not in values


# --------------------------------------------------------------------------
# 4. password never serialised
# --------------------------------------------------------------------------


def test_password_never_serialised(built: dict[str, Any]) -> None:
    result = _record_result(built)
    dumped = result.capability.model_dump_json()
    assert PASSWORD_LITERAL not in dumped
    assert "[redacted:password]" not in dumped
    assert "{{password}}" in dumped
    password_steps = [s for s in result.capability.steps if s.value == "{{password}}"]
    assert len(password_steps) == 1
    assert password_steps[0].checkpoint is None  # sensitive type steps get no checkpoint


# --------------------------------------------------------------------------
# 5. savings balance extraction is a table_cell keyed on header + SAV row
# --------------------------------------------------------------------------


def test_savings_extraction_is_table_cell(built: dict[str, Any]) -> None:
    result = _record_result(built)
    extract_steps = [s for s in result.capability.steps if s.kind == "extract"]
    assert len(extract_steps) == 1
    ext = extract_steps[0].extract
    assert ext is not None
    assert ext.from_ == "table_cell"
    assert ext.table is not None
    assert re.search(ext.table.header_match, "Curr Bal")
    assert not re.search(ext.table.header_match, "Avail Bal")
    assert ext.table.row_where.header == "Type"
    assert re.search(ext.table.row_where.matches, "SAV")
    assert not re.search(ext.table.row_where.matches, "CHK")
    assert ext.parse is not None and ext.parse.type == "money"


# --------------------------------------------------------------------------
# 6. a checkpoint already-true-before is dropped
# --------------------------------------------------------------------------


def test_already_true_checkpoint_is_dropped(built: dict[str, Any]) -> None:
    result = _record_result(built)
    member_field_steps = [s for s in result.capability.steps if s.value == "Member"]
    assert len(member_field_steps) == 2
    assert member_field_steps[0].checkpoint is not None
    assert member_field_steps[1].checkpoint is None
    assert member_field_steps[1].wait.until == "load"

    checkpoint_prunes = [p for p in result.pruning_log if p["kind"] == "checkpoint"]
    assert len(checkpoint_prunes) == 1


# --------------------------------------------------------------------------
# 7. an outcome that also matches the success page is dropped
# --------------------------------------------------------------------------


def test_loose_outcome_is_dropped(built: dict[str, Any]) -> None:
    result = _record_result(built)
    codes = [o.code for o in result.capability.outcomes]
    assert "SYSTEM_BUSY" not in codes
    assert "MEMBER_NOT_FOUND" in codes
    assert "session_risk" in codes  # discovered outcome survives (never fires on this run)

    outcome_prunes = [p for p in result.pruning_log if p["kind"] == "outcome"]
    assert any(p["code"] == "SYSTEM_BUSY" for p in outcome_prunes)
    assert result.capability.contract.outcome_codes == codes


# --------------------------------------------------------------------------
# 8. failed / policy-blocked steps: absent from curated, present in raw
# --------------------------------------------------------------------------


def _target_names(steps) -> set[str]:
    names: set[str] = set()
    for s in steps:
        if not s.target:
            continue
        for strat in s.target.strategies:
            if strat.kind == "role":
                names.add(strat.params.get("name"))
    return names


def test_failed_and_blocked_steps_pruned_but_kept_in_raw(built: dict[str, Any]) -> None:
    result = _record_result(built)

    assert len(result.raw_capability.steps) == 10
    assert len(result.capability.steps) == 8

    step_prunes = [p for p in result.pruning_log if p["kind"] == "step"]
    assert len(step_prunes) == 2
    reasons = " ".join(p["reason"] for p in step_prunes)
    assert "blocked by policy" in reasons
    assert "action failed" in reasons

    assert "Open Sub-Account" in _target_names(result.raw_capability.steps)
    assert "Open Sub-Account" not in _target_names(result.capability.steps)


# --------------------------------------------------------------------------
# 9. targets: multiple, semantic-first, real rationales
# --------------------------------------------------------------------------


def test_targets_are_semantic_first_with_real_rationales(built: dict[str, Any]) -> None:
    result = _record_result(built)
    targets = [s.target for s in result.capability.steps if s.target is not None]
    assert targets

    all_rationales = []
    for t in targets:
        assert len(t.strategies) > 1
        kinds = [st.kind for st in t.strategies]
        if "role" in kinds and "structural" in kinds:
            assert kinds.index("role") < kinds.index("structural")
        for st in t.strategies:
            assert st.rationale.strip()
            all_rationales.append(st.rationale)

    assert len(set(all_rationales)) > 1


# --------------------------------------------------------------------------
# 10. content_hash set and stable across re-serialisation
# --------------------------------------------------------------------------


def test_content_hash_stable(built: dict[str, Any]) -> None:
    result = _record_result(built)
    cap = result.capability
    assert cap.content_hash is not None
    reloaded = Capability.model_validate_json(cap.model_dump_json())
    assert reloaded.content_hash == cap.content_hash
    assert reloaded.compute_content_hash() == cap.content_hash


# --------------------------------------------------------------------------
# extra: record() refuses a non-successful trace outright
# --------------------------------------------------------------------------


def test_record_refuses_non_success_trace(built: dict[str, Any]) -> None:
    trace = built["trace"]
    failed_trace = DiscoveryTrace(
        goal=trace.goal, status="stopped", steps=trace.steps, outputs=trace.outputs,
        summary=trace.summary, stop_reason="max_steps", observations=trace.observations,
        intervention=None, model=trace.model, prompt_version=trace.prompt_version,
        usage_total={}, started_at=trace.started_at, finished_at=trace.finished_at,
    )
    with pytest.raises(RecorderError):
        record(
            failed_trace,
            capability_id="x", name="x", description="x",
            inputs_spec={}, outputs_spec={}, policy=built["policy"], redactor=Redactor(),
            entry_url=FRAMESET_URL, app=AppInfo(vendor="x", product="x", version="1"),
            discovery_run_id="x",
        )
