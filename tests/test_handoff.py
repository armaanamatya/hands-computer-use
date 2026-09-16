r"""Real, end-to-end handoff tests: a live Flask mock app + headless
Playwright Chromium, driven through `hands.handoff`'s `SessionController`,
`OperatorServer`, and `SimulatedOperator` -- no LLM anywhere in this file.

Fixture pattern (mock server, browser, port-substituted policy) mirrors
`tests/test_replay.py`: every regex this project's `policy.yaml` carries is
path-only, so the only port-dependent piece is `allowed_url_patterns`,
patched here the same way test_replay.py patches it.

Test 4 (end-to-end handoff) is provided as BOTH variants the brief allows:
  - `test_end_to_end_handoff_via_replay_engine_and_simulated_operator` wires
    `make_escalation_handler()` into a REAL `ReplayEngine` run against the
    real mock app (the `lookup_savings_balance` capability, with an
    appended risky "Open Sub-Account" step) -- this is what actually
    exercises the engine-integration contract end to end.
  - `test_direct_mechanics_supervisor_override_on_live_page` (used by tests
    5/6 below via a shared helper) drives `SessionController` +
    `SimulatedOperator` directly against the mock app's member-31337
    supervisor-override screen -- the concrete "automation is blocked,
    a human enters a code it doesn't have, and the SAME live session
    continues" scenario the brief describes -- without authoring a second,
    larger capability JSON for the sub-account flow.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from hands.evidence.run_log import RunLog
from hands.handoff import OperatorServer, SessionController, SimulatedOperator, make_escalation_handler
from hands.replay.engine import ReplayEngine
from hands.safety.policy import Policy
from hands.schema.capability import Capability
from hands.schema.control import ControlState, InterventionRequest, NotInControl
from hands.schema.result import RunStatus
from hands.surface.base import Action
from hands.surface.web import WebSurface
from mockbank.app import app as flask_app

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "lookup_savings_balance.json"
POLICY_YAML_PATH = REPO_ROOT / "policy.yaml"

VALID_INPUTS = {"username": "teller1", "password": "Passw0rd!", "member_id": "12345"}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_server():
    srv = make_server("127.0.0.1", 0, flask_app, threaded=True)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        srv.shutdown()
        t.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    # module-, not session-, scoped: tests/test_replay.py (owned by another
    # agent, not modified here) declares its own session-scoped `browser`
    # fixture using the same `sync_playwright()` context manager.
    # Playwright's sync API only tolerates one such context being open at a
    # time per process -- a SECOND `with sync_playwright()` while a first
    # one is still open raises "already inside the asyncio loop". Scoping
    # THIS fixture to the module means it is fully torn down as soon as
    # this file's tests finish, before test_replay.py's own (session-
    # scoped, so alive until the whole run ends) fixture ever starts.
    #
    # This makes `uv run pytest -q` and `uv run pytest tests/test_handoff.py
    # -q` (this file's two Done-criteria invocations) reliable: pytest's
    # default collection sorts tests/ alphabetically, so test_handoff.py
    # always runs (and this fixture always tears down) before test_replay.py
    # starts -- verified by running the full suite twice in a row. It is
    # NOT bulletproof against every possible invocation: `pytest
    # tests/test_replay.py tests/test_handoff.py` (test_replay.py's path
    # given first, forcing it to collect and open its session-scoped
    # fixture before this one) reproduces the same "already inside the
    # asyncio loop" error, in test_replay.py's fixture this time -- and
    # there is no fix available from this side of that boundary short of a
    # shared tests/conftest.py, which is outside the set of files this
    # module is scoped to create or modify.
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def policy(mock_server) -> Policy:
    raw = yaml.safe_load(POLICY_YAML_PATH.read_text(encoding="utf-8"))
    raw["allowed_url_patterns"] = [rf"^http://(127\.0\.0\.1|localhost):{mock_server}(/.*)?$"]
    # Keep the handoff tests fast: policy.yaml's own 600s default would
    # otherwise be the fallback timeout make_escalation_handler uses
    # whenever a test doesn't pass its own timeout_s explicitly.
    raw["human_wait_timeout_s"] = 20
    return Policy.from_dict(raw)


@pytest.fixture
def policy_capture(policy: Policy) -> Policy:
    # Policy is a frozen dataclass -- dataclasses.replace, not mutation.
    return replace(policy, capture_human_input_values=True)


def _dummy_policy() -> Policy:
    """A minimal, self-contained Policy for the pure state-machine tests
    (1 and 3) that need no live browser and no mock server at all."""
    return Policy.from_dict({
        "allowed_domains": ["127.0.0.1"],
        "allowed_url_patterns": [r"^http://127\.0\.0\.1(/.*)?$"],
        "allowed_actions": ["navigate", "click", "type", "select", "press", "extract", "wait"],
        "risky_patterns": [r"(?i)\b(open|create|submit|confirm|transfer|close|delete|approve|post)\b"],
        "risky_mode": "gate",
        "allow_coordinate_fallback": True,
        "capture_human_input_values": False,
        "human_wait_timeout_s": 5,
        "max_steps": 40,
        "step_timeout_ms": 10000,
        "run_timeout_s": 300,
        "recovery_budget": 5,
    })


def _new_run_log(tmp_path: Path, run_id: str) -> RunLog:
    return RunLog(tmp_path, run_id)


def _new_session(browser: Any, run_id: str, run_log: RunLog, policy: Policy):
    """A fresh Playwright context/page wired to a fresh SessionController +
    WebSurface, in the real construction order: the controller must exist
    before the surface (the surface's `guard` callable IS
    `controller.guard`), so the surface is attached back onto the
    controller once built. A fresh context per test -> fresh cookies ->
    fresh Flask session per test, same reasoning as test_replay.py's
    `surface` fixture (not reused here directly because several of these
    tests need MULTIPLE independent sessions per test, e.g. one per policy
    variant)."""
    context = browser.new_context()
    page = context.new_page()
    controller = SessionController(run_id, run_log, policy, page=page)
    surface = WebSurface(page, guard=controller.guard)
    controller.attach(surface=surface)
    return context, page, controller, surface


def _login_and_open_member(surface: WebSurface, token, port: int, member_id: str) -> None:
    """Automation-side setup, performed with the current automation token:
    sign on and look up `member_id`, landing on the member-detail screen."""
    surface.act(Action(kind="navigate", url=f"http://127.0.0.1:{port}/"), token)
    page = surface.page
    work = next(f for f in page.frames if f.name == "work")
    work.fill("input[name=uid]", "teller1")
    work.fill("input[name=pwd]", "Passw0rd!")
    work.click("input[type=submit]")
    page.wait_for_load_state("load")
    work = next(f for f in page.frames if f.name == "work")
    work.fill("input[name=mno]", member_id)
    work.click("input[type=submit]")
    page.wait_for_load_state("load")


def _open_subaccount_new(surface: WebSurface) -> None:
    page = surface.page
    work = next(f for f in page.frames if f.name == "work")
    work.click("text=Open Sub-Account")
    page.wait_for_load_state("load")


def _run_supervisor_override_handoff(surface: WebSurface, controller: SessionController, port: int, nick_value: str):
    """Drive the mock app's actual human-in-the-loop trigger: member 31337
    needs a supervisor override code (`data.SUPERVISOR_OVERRIDE_CODE`) that
    is deliberately never given to automation (see mockbank/data.py). This
    performs automation's setup up to the point automation would be stuck,
    raises a real handoff, has a SimulatedOperator enter the code on the
    SAME live page, releases, and resumes -- returning the resulting
    HumanActionRecord for the caller to inspect.
    """
    token = controller.token()
    _login_and_open_member(surface, token, port, "31337")
    _open_subaccount_new(surface)

    req = InterventionRequest(
        id=f"{controller.run_id}-iv1", run_id=controller.run_id, mode="replay",
        reason_code="REQUIRES_HUMAN_STEP",
        summary="Supervisor override code required to open this sub-account",
    )
    handoff_id = controller.request_handoff(req)
    assert controller.state == ControlState.HANDOFF_PENDING

    operator = SimulatedOperator(controller, actions=[
        {"fill": {"frame_name": "work", "selector": "input[name=nick]", "value": nick_value}},
        {"click": {"frame_name": "work", "selector": "input[type=submit]"}},   # -> sub_review (needs_override)
        {"fill": {"frame_name": "work", "selector": "input[name=sup]", "value": "4242"}},
        {"click": {"frame_name": "work", "selector": "input[type=submit]"}},   # -> sub_done
    ])
    record = operator.run(handoff_id, resume_mode="auto", notes="entered supervisor override code")
    assert controller.state == ControlState.RESUME_PENDING

    fresh = controller.resume()
    assert controller.state == ControlState.AUTOMATION_ACTIVE

    # The run actually completed on the SAME session: the confirmation page
    # is now showing, reached entirely through the human's in-session work.
    result = surface.act(Action(kind="wait", ms=10), fresh)
    assert result.ok
    page = surface.page
    work = next(f for f in page.frames if f.name == "work")
    assert "sub-account has been opened" in work.inner_text("body").lower()
    return record


def _load_capability_with_risky_subaccount_step(port: int) -> Capability:
    data = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    entry_url = f"http://127.0.0.1:{port}/"
    data["target"]["entry_url"] = entry_url
    for step in data["steps"]:
        if step["kind"] == "navigate":
            step["value"] = entry_url
    # Same risky step test_replay.py's own escalation tests append: "Open
    # Sub-Account" matches policy.yaml's risky_patterns, and the capability
    # is left unapproved, so this always escalates RISKY_CONFIRMATION.
    data["steps"].append({
        "id": "s8", "kind": "click", "screen": "member_detail",
        "intent": "Open a sub-account for this member", "origin": "agent",
        "target": {
            "description": "Open Sub-Account link on the member detail page",
            "frame": {"url_pattern": None, "name": "work", "index": None},
            "strategies": [{
                "kind": "role",
                "params": {"role": "link", "name": "Open Sub-Account", "exact": True},
                "rationale": "the only link with this exact accessible name on the member detail page",
            }],
            "snapshot": {},
        },
        "value": None, "expect_dialog": None,
        "wait": {"until": "load", "timeout_ms": 10000},
        "checkpoint": None, "extract": None,
        "risk": {
            "level": "risky", "reason": "opens a new sub-account record for the member",
            "requires_confirmation": True, "idempotency_check": None,
        },
        "human_task": None,
    })
    cap = Capability.model_validate(data)
    assert cap.validate_semantics() == []
    return cap


# --------------------------------------------------------------------------
# 1. State machine: legal path succeeds, an illegal jump raises
# --------------------------------------------------------------------------


def test_legal_transition_path_then_illegal_jump_raises(tmp_path):
    run_log = _new_run_log(tmp_path, "test1-legal-path")
    controller = SessionController("test1-legal-path", run_log, _dummy_policy())
    assert controller.state == ControlState.AUTOMATION_ACTIVE
    assert controller.epoch == 0

    controller.transition(ControlState.HANDOFF_PENDING, reason="intervention raised")
    assert controller.state == ControlState.HANDOFF_PENDING
    assert controller.epoch == 0  # holder unchanged (still automation, unclaimed)

    controller.transition(ControlState.HUMAN_ACTIVE, reason="claimed")
    assert controller.state == ControlState.HUMAN_ACTIVE
    assert controller.epoch == 1  # holder changed: automation -> human

    controller.transition(ControlState.RESUME_PENDING, reason="released")
    assert controller.state == ControlState.RESUME_PENDING
    assert controller.epoch == 1  # holder unchanged (still human, until resume())

    controller.transition(ControlState.AUTOMATION_ACTIVE, reason="resumed")
    assert controller.state == ControlState.AUTOMATION_ACTIVE
    assert controller.epoch == 2  # holder changed: human -> automation

    # An illegal jump: AUTOMATION_ACTIVE -> HUMAN_ACTIVE skips HANDOFF_PENDING.
    with pytest.raises(ValueError):
        controller.transition(ControlState.HUMAN_ACTIVE, reason="illegal")
    # The illegal attempt must not have mutated anything.
    assert controller.state == ControlState.AUTOMATION_ACTIVE
    assert controller.epoch == 2


# --------------------------------------------------------------------------
# 2. Token/epoch discipline: a pre-handoff token is dead once a human claims
# --------------------------------------------------------------------------


def test_stale_automation_token_rejected_after_human_claims(browser, policy, tmp_path):
    run_log = _new_run_log(tmp_path, "test2-stale-token")
    context, page, controller, surface = _new_session(browser, "test2-stale-token", run_log, policy)
    try:
        stale_token = controller.token()
        # Sanity: the token is perfectly valid right now, before any handoff.
        assert surface.act(Action(kind="wait", ms=10), stale_token).ok

        req = InterventionRequest(
            id="test2-iv1", run_id="test2-stale-token", mode="replay",
            reason_code="STUCK_NO_PROGRESS", summary="automation is stuck",
        )
        handoff_id = controller.request_handoff(req)
        controller.claim(handoff_id, "operator-1")
        assert controller.epoch == 1

        # THE invariant this whole design exists to enforce: the token
        # captured before the handoff must be rejected now, both by the
        # controller directly and by the surface it guards.
        with pytest.raises(NotInControl):
            controller.guard(stale_token)
        with pytest.raises(NotInControl):
            surface.act(Action(kind="wait", ms=10), stale_token)

        # A freshly-minted token for the CURRENT epoch works again.
        controller.release(handoff_id, resume_mode="auto")
        fresh_token = controller.resume()
        assert surface.act(Action(kind="wait", ms=10), fresh_token).ok
    finally:
        context.close()


# --------------------------------------------------------------------------
# 3. control_state.json exists and reflects the current holder on disk
# --------------------------------------------------------------------------


def test_control_state_json_reflects_current_holder_after_each_transition(tmp_path):
    run_log = _new_run_log(tmp_path, "test3-disk-state")
    controller = SessionController("test3-disk-state", run_log, _dummy_policy())
    state_path = run_log.dir / "control_state.json"

    def _on_disk() -> dict:
        # Read from disk every time -- this test exists specifically to
        # NOT trust in-memory state.
        return json.loads(state_path.read_text(encoding="utf-8"))

    controller.transition(ControlState.HANDOFF_PENDING, reason="req")
    rec = _on_disk()
    assert rec["state"] == "HANDOFF_PENDING" and rec["holder"] == "automation" and rec["epoch"] == 0

    controller.transition(ControlState.HUMAN_ACTIVE, reason="claim")
    rec = _on_disk()
    assert rec["state"] == "HUMAN_ACTIVE" and rec["holder"] == "human" and rec["epoch"] == 1

    controller.transition(ControlState.RESUME_PENDING, reason="release")
    rec = _on_disk()
    assert rec["state"] == "RESUME_PENDING" and rec["holder"] == "human" and rec["epoch"] == 1

    controller.transition(ControlState.AUTOMATION_ACTIVE, reason="resume")
    rec = _on_disk()
    assert rec["state"] == "AUTOMATION_ACTIVE" and rec["holder"] == "automation" and rec["epoch"] == 2


# --------------------------------------------------------------------------
# 4a. End-to-end handoff through the real ReplayEngine (engine-integration)
# --------------------------------------------------------------------------


def test_end_to_end_handoff_via_replay_engine_and_simulated_operator(browser, policy, tmp_path, mock_server):
    """`make_escalation_handler()` wired straight into a real
    `ReplayEngine.run()` against the real mock app. Clicking "Open
    Sub-Account" is risky and the capability is unapproved, so it escalates
    RISKY_CONFIRMATION; a `SimulatedOperator` claims the SAME live session,
    looks at the target page (proving control genuinely transferred) and
    returns without submitting it, then releases with resume_mode="auto" --
    the control-layer's "the human handled it, don't redo it" signal, which
    `make_escalation_handler` maps to `HumanResolution(action="skip_step")`
    (see the mapping table's comment in hands/handoff/__init__.py). The run
    then completes normally.
    """
    run_log = _new_run_log(tmp_path, "test4a-engine")
    context, page, controller, surface = _new_session(browser, "test4a-engine", run_log, policy)
    try:
        cap = _load_capability_with_risky_subaccount_step(mock_server)
        operator = SimulatedOperator(controller, actions=[
            {"click": {"frame_name": "work", "selector": "text=Open Sub-Account"}},
            {"click": {"frame_name": "work", "selector": "text=Cancel"}},
        ])
        handler = make_escalation_handler(controller, operator=operator, timeout_s=15)
        engine = ReplayEngine(surface, policy, run_log, escalate=handler)

        result = engine.run(cap, VALID_INPUTS)

        assert result.status == RunStatus.SUCCESS, result.failure
        s8 = next(s for s in result.steps if s.step_id == "s8")
        assert s8.status == "human_performed"
        assert len(result.human_interventions) == 1

        # Exactly two holder changes happened: automation -> human (claim),
        # human -> automation (resume).
        assert controller.epoch == 2
        assert controller.state == ControlState.AUTOMATION_ACTIVE

        assert (run_log.dir / "human-actions.jsonl").stat().st_size > 0
        intervention_files = list(run_log.dir.glob("intervention-*.json"))
        assert len(intervention_files) == 1
        payload = json.loads(intervention_files[0].read_text(encoding="utf-8"))
        assert payload["reason_code"] == "RISKY_CONFIRMATION"
    finally:
        context.close()


# --------------------------------------------------------------------------
# 4b / 5. Direct-mechanics handoff on the live 31337 override screen, and
# human actions recorded with a target descriptor + capture-policy respected
# --------------------------------------------------------------------------


def test_human_actions_recorded_with_target_descriptor_and_capture_policy(browser, policy, tmp_path, mock_server):
    run_log = _new_run_log(tmp_path, "test5-recorded")
    context, page, controller, surface = _new_session(browser, "test5-recorded", run_log, policy)
    try:
        record = _run_supervisor_override_handoff(surface, controller, mock_server, nick_value="CaptureOff")

        assert record.actions, "expected the operator's actions to be recorded"

        click_actions = [a for a in record.actions if a.type == "click"]
        assert click_actions, "expected at least one recorded click with a target descriptor"
        assert click_actions[0].target is not None
        assert {"role", "name", "structural"} <= click_actions[0].target.keys()

        # policy.yaml's capture_human_input_values is False: the value the
        # operator typed into the (non-sensitive) nickname field must be
        # absent, but the fact that a value was entered must still show up.
        nick_actions = [a for a in record.actions if a.target and a.target.get("attrs", {}).get("name") == "nick"]
        assert nick_actions, "expected a recorded action targeting the nickname field"
        for a in nick_actions:
            assert a.value is None
            assert a.target.get("has_value") is True

        # persisted to disk, not just held in memory
        jsonl_path = run_log.dir / "human-actions.jsonl"
        assert jsonl_path.exists() and jsonl_path.stat().st_size > 0
        lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        disk_nick_actions = [
            a for rec in lines for a in rec["actions"]
            if (a.get("target") or {}).get("attrs", {}).get("name") == "nick"
        ]
        assert disk_nick_actions
        assert all(a["value"] is None and a["target"]["has_value"] is True for a in disk_nick_actions)
    finally:
        context.close()


# --------------------------------------------------------------------------
# 6. A password typed by the human is never recorded, even with capture on
# --------------------------------------------------------------------------


def test_password_never_recorded_even_with_capture_enabled(browser, policy_capture, tmp_path, mock_server):
    run_log = _new_run_log(tmp_path, "test6-password")
    context, page, controller, surface = _new_session(browser, "test6-password", run_log, policy_capture)
    try:
        assert policy_capture.capture_human_input_values is True
        record = _run_supervisor_override_handoff(surface, controller, mock_server, nick_value="CaptureOn")

        # With capture enabled, the non-sensitive field's value IS recorded
        # -- proving the earlier test's None wasn't just "always None".
        nick_actions = [a for a in record.actions if a.target and a.target.get("attrs", {}).get("name") == "nick"]
        assert any(a.value == "CaptureOn" for a in nick_actions)

        # The password field ("sup", type=password on sub_review.html) must
        # NEVER report a value, regardless of policy.
        sup_actions = [a for a in record.actions if a.target and a.target.get("attrs", {}).get("name") == "sup"]
        assert sup_actions, "expected a recorded action targeting the supervisor-override password field"
        assert all(a.value is None for a in sup_actions)
    finally:
        context.close()


# --------------------------------------------------------------------------
# 7. Timeout: no operator moves the run to ABANDONED
# --------------------------------------------------------------------------


def test_wait_for_human_timeout_abandons_run(browser, policy, tmp_path):
    run_log = _new_run_log(tmp_path, "test7-timeout")
    context, page, controller, surface = _new_session(browser, "test7-timeout", run_log, policy)
    try:
        req = InterventionRequest(
            id="test7-iv1", run_id="test7-timeout", mode="replay",
            reason_code="STUCK_NO_PROGRESS", summary="stuck, and nobody ever claims it",
        )
        handoff_id = controller.request_handoff(req)

        final_state = controller.wait_for_human(handoff_id, timeout_s=1.0)

        assert final_state == ControlState.ABANDONED
        assert controller.state == ControlState.ABANDONED
        on_disk = json.loads((run_log.dir / "control_state.json").read_text(encoding="utf-8"))
        assert on_disk["state"] == "ABANDONED"
        assert on_disk["reason"] == "HUMAN_TIMEOUT"
    finally:
        context.close()


# --------------------------------------------------------------------------
# 8. Operator server: the HTTP thread only queues; the main thread applies
# --------------------------------------------------------------------------


def test_operator_server_queues_on_http_thread_and_applies_on_drain(browser, policy, tmp_path):
    run_log = _new_run_log(tmp_path, "test8-operator-server")
    context, page, controller, surface = _new_session(browser, "test8-operator-server", run_log, policy)
    server = OperatorServer(controller, port=0)
    try:
        req = InterventionRequest(
            id="test8-iv1", run_id="test8-operator-server", mode="replay",
            reason_code="STUCK_NO_PROGRESS", summary="stuck",
        )
        handoff_id = controller.request_handoff(req)

        bound_port = server.start()
        base = f"http://127.0.0.1:{bound_port}"

        with urllib.request.urlopen(f"{base}/control") as resp:
            control = json.loads(resp.read())
        assert control["state"] == "HANDOFF_PENDING"
        assert control["handoff_id"] == handoff_id

        with urllib.request.urlopen(f"{base}/interventions") as resp:
            interventions = json.loads(resp.read())
        assert len(interventions) == 1
        assert interventions[0]["id"] == handoff_id

        body = json.dumps({"operator_id": "http-operator"}).encode("utf-8")
        post_req = urllib.request.Request(
            f"{base}/interventions/{handoff_id}/claim", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(post_req) as resp:
            assert resp.status == 202
            assert json.loads(resp.read())["queued"] is True

        # Proof the HTTP thread never touched Playwright/the controller
        # directly: right after the POST returns, nothing has been applied.
        assert controller.state == ControlState.HANDOFF_PENDING
        assert controller.epoch == 0

        applied = server.drain(controller)
        assert applied == 1
        assert controller.state == ControlState.HUMAN_ACTIVE
        assert controller.epoch == 1

        with urllib.request.urlopen(f"{base}/control") as resp:
            control_after = json.loads(resp.read())
        assert control_after["state"] == "HUMAN_ACTIVE"
        assert control_after["epoch"] == 1

        with urllib.request.urlopen(f"{base}/interventions") as resp:
            interventions_after = json.loads(resp.read())
        assert interventions_after == []  # claimed -> no longer open
    finally:
        server.stop()
        context.close()
