r"""Real, end-to-end replay tests: a live Flask mock app + a headless
Playwright browser, driven entirely through `hands.replay.engine.ReplayEngine`
-- no LLM anywhere in this file.

Port handling: every regex this test's capability/policy carries is
PATH-ONLY (e.g. `r"/lookup(\?|$)"`, never a host:port), so the only two
things that are actually port-dependent are `Capability.target.entry_url`
and `Policy.allowed_url_patterns` (policy.yaml pins port 5050, and the
mock server here binds a free port). Both are patched in `_load_capability`
/ `_load_policy` below by loading the real JSON/YAML and substituting the
fixture's real port before validating -- the capability and policy files
on disk are otherwise used completely unmodified.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from hands.replay.engine import HumanResolution, ReplayEngine
from hands.safety.policy import Policy
from hands.schema.capability import Capability
from hands.schema.result import RunStatus
from hands.evidence.run_log import RunLog
from hands.surface.web import WebSurface
from mockbank.app import app as flask_app

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "lookup_savings_balance.json"
OUTCOMES_LIBRARY_PATH = REPO_ROOT / "capabilities" / "library" / "legacycore.outcomes.json"
POLICY_YAML_PATH = REPO_ROOT / "policy.yaml"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_server():
    """Run the real LegacyCore mock app on a free localhost port, in a
    background thread, for the whole test session. `threaded=True` matters:
    the frameset loads `/nav` and `/main` as two concurrent requests, and a
    single-threaded dev server can stall on that.
    """
    srv = make_server("127.0.0.1", 0, flask_app, threaded=True)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        srv.shutdown()
        t.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def surface(browser):
    # A fresh browser CONTEXT per test -> fresh cookies -> a fresh Flask
    # session per test. Without this, server-side session state (the
    # busy-once marker, sa_counter, the logged-in uid) would leak between
    # tests that otherwise look independent.
    context = browser.new_context()
    page = context.new_page()
    s = WebSurface(page)
    yield s
    s.close()


@pytest.fixture
def policy(mock_server) -> Policy:
    raw = yaml.safe_load(POLICY_YAML_PATH.read_text(encoding="utf-8"))
    raw["allowed_url_patterns"] = [rf"^http://(127\.0\.0\.1|localhost):{mock_server}(/.*)?$"]
    return Policy.from_dict(raw)


@pytest.fixture
def run_log(tmp_path) -> RunLog:
    return RunLog(tmp_path, "test-run")


def _load_capability(port: int, mutate: Any = None) -> Capability:
    """Load the real capability JSON, point it at the fixture's port (the
    only port-dependent field -- see module docstring), optionally apply
    `mutate` (a callable over the raw dict, for the handful of tests that
    need a deliberately-broken variant), and validate.
    """
    data = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    entry_url = f"http://127.0.0.1:{port}/"
    data["target"]["entry_url"] = entry_url
    # step s1's own `value` is the literal navigate URL, baked in
    # separately from target.entry_url at authoring time (mirroring what a
    # real discovery transcript would have recorded) -- also port-dependent.
    for step in data["steps"]:
        if step["kind"] == "navigate":
            step["value"] = entry_url
    if mutate is not None:
        mutate(data)
    cap = Capability.model_validate(data)
    assert cap.validate_semantics() == []
    return cap


VALID_INPUTS = {"username": "teller1", "password": "Passw0rd!", "member_id": "12345"}


def _inputs(member_id: str) -> dict[str, str]:
    return {**VALID_INPUTS, "member_id": member_id}


# --------------------------------------------------------------------------
# 0. The artifacts themselves are schema-valid (sanity for deliverables 5/6)
# --------------------------------------------------------------------------


def test_outcomes_library_is_schema_valid() -> None:
    from hands.schema.capability import Outcome

    entries = json.loads(OUTCOMES_LIBRARY_PATH.read_text(encoding="utf-8"))
    assert len(entries) >= 6
    for entry in entries:
        outcome = Outcome.model_validate(entry)
        assert outcome.origin == "library"


def test_capability_file_is_schema_valid_and_semantically_clean(mock_server) -> None:
    cap = _load_capability(mock_server)
    assert cap.validate_semantics() == []
    assert cap.contract.outputs["savings_balance"].type == "money"


# --------------------------------------------------------------------------
# 1. Happy path: correct row picked out of a multi-row table
# --------------------------------------------------------------------------


def test_lookup_success_picks_the_savings_row_not_checking_or_loan(mock_server, surface, policy, run_log) -> None:
    cap = _load_capability(mock_server)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("12345"))

    assert result.status == RunStatus.SUCCESS
    assert result.exit_code() == 0
    # Money is parsed to a canonical decimal STRING (commas/currency
    # symbols stripped) -- see hands/surface/extract.py's _parse_money
    # docstring for why a string, not a float.
    assert result.outputs["savings_balance"] == "2340.55"
    assert result.outputs["savings_balance"] not in ("812.10", "-4500.00")
    assert result.outputs_complete is True


# --------------------------------------------------------------------------
# 2. Business outcome: member not found
# --------------------------------------------------------------------------


def test_member_not_found_is_a_business_outcome(mock_server, surface, policy, run_log) -> None:
    cap = _load_capability(mock_server)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("99999"))

    assert result.status == RunStatus.BUSINESS_OUTCOME
    assert result.outcome is not None
    assert result.outcome.code == "MEMBER_NOT_FOUND"
    assert result.exit_code() == 0


# --------------------------------------------------------------------------
# 3. Business outcome: restricted account
# --------------------------------------------------------------------------


def test_restricted_account_is_a_business_outcome(mock_server, surface, policy, run_log) -> None:
    cap = _load_capability(mock_server)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("77777"))

    assert result.status == RunStatus.BUSINESS_OUTCOME
    assert result.outcome is not None
    assert result.outcome.code == "ACCOUNT_RESTRICTED"
    assert result.exit_code() == 0
    # bonus: the outcome's result_outputs extraction ran too
    assert "restricted" in result.outputs.get("restriction_reason", "").lower()


# --------------------------------------------------------------------------
# 4. Interstitial recovery: system busy, clears on retry
# --------------------------------------------------------------------------


def test_system_busy_recovers_and_succeeds(mock_server, surface, policy, run_log) -> None:
    cap = _load_capability(mock_server)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("40404"))

    assert result.status == RunStatus.SUCCESS
    recovered_steps = [s for s in result.steps if s.status == "recovered"]
    assert len(recovered_steps) == 1
    assert "SYSTEM_BUSY" in recovered_steps[0].outcomes_matched

    assert len(engine.recovery_history) == 1
    record = engine.recovery_history[0]
    assert record.outcome_code == "SYSTEM_BUSY"
    assert record.action == "click"
    assert record.attempt == 1


# --------------------------------------------------------------------------
# 5. Hard failure: app error, with evidence on disk
# --------------------------------------------------------------------------


def test_app_error_fails_with_evidence_on_disk(mock_server, surface, policy, run_log) -> None:
    cap = _load_capability(mock_server)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("50000"))

    assert result.status == RunStatus.FAILED
    assert result.failure is not None
    assert result.failure.category.value == "APP_ERROR"
    assert result.failure.observed is not None
    assert "screenshot" in result.failure.evidence
    assert "dom" in result.failure.evidence
    screenshot_path = run_log.dir / result.failure.evidence["screenshot"]
    dom_path = run_log.dir / result.failure.evidence["dom"]
    assert screenshot_path.is_file() and screenshot_path.stat().st_size > 0
    assert dom_path.is_file() and dom_path.stat().st_size > 0


# --------------------------------------------------------------------------
# 6. Wrong-screen detection actually fires (frame-scoped, not decorative)
# --------------------------------------------------------------------------


def test_wrong_screen_declaration_is_detected(mock_server, surface, policy, run_log) -> None:
    def _mutate(data: dict[str, Any]) -> None:
        # s2 (kind="type", screen="login") is mutated to declare screen
        # "lookup" instead. Reality after s1's navigate is the login page
        # (/main), not /lookup -- so s2's OWN pre-action screen gate must
        # catch this before it ever tries to act. (s1 itself is a
        # `navigate` step and is deliberately NOT used for this: the gate
        # skips navigate steps, since they establish a screen rather than
        # presuppose one -- this test needs a non-navigate step to prove
        # the gate fires at all.)
        steps = data["steps"]
        s2 = next(s for s in steps if s["id"] == "s2")
        assert s2["kind"] == "type"
        s2["screen"] = "lookup"

    cap = _load_capability(mock_server, mutate=_mutate)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("12345"))

    assert result.status == RunStatus.FAILED
    assert result.failure is not None
    assert result.failure.category.value == "UNEXPECTED_SCREEN"
    assert result.failure.step is not None
    assert result.failure.step["id"] == "s2"
    # reality (the login screen) is correctly identified as the actual screen
    assert result.failure.observed is not None
    assert result.failure.observed.screen_guess == "login"


# --------------------------------------------------------------------------
# 7. Risky step on a non-approved capability escalates for confirmation
# --------------------------------------------------------------------------


def test_risky_unapproved_step_escalates_to_needs_human(mock_server, surface, policy, run_log) -> None:
    def _mutate(data: dict[str, Any]) -> None:
        # Append a risky step (opens a sub-account) after the extract step.
        # "Open Sub-Account" is a real link on member.html whose accessible
        # name contains the whole word "open", which policy.yaml's
        # risky_patterns matches -> risky_mode "gate" -> requires
        # confirmation. The capability's approval is left at its default
        # "draft" status and no approved_by is passed to run(), so this
        # must escalate rather than proceed.
        data["steps"].append(
            {
                "id": "s8",
                "kind": "click",
                "screen": "member_detail",
                "intent": "Open a sub-account for this member",
                "origin": "agent",
                "target": {
                    "description": "Open Sub-Account link on the member detail page",
                    "frame": {"url_pattern": None, "name": "work", "index": None},
                    "strategies": [
                        {
                            "kind": "role",
                            "params": {"role": "link", "name": "Open Sub-Account", "exact": True},
                            "rationale": "the only link with this exact accessible name on the member detail page",
                        }
                    ],
                    "snapshot": {},
                },
                "value": None,
                "expect_dialog": None,
                "wait": {"until": "load", "timeout_ms": 10000},
                "checkpoint": None,
                "extract": None,
                "risk": {
                    "level": "risky",
                    "reason": "opens a new sub-account record for the member",
                    "requires_confirmation": True,
                    "idempotency_check": None,
                },
                "human_task": None,
            }
        )

    cap = _load_capability(mock_server, mutate=_mutate)
    engine = ReplayEngine(surface, policy, run_log)  # no escalate handler configured
    result = engine.run(cap, _inputs("12345"), approved_by=None)

    assert result.status == RunStatus.NEEDS_HUMAN
    assert len(result.human_interventions) >= 1

    intervention_files = list(run_log.dir.glob("intervention-*.json"))
    assert len(intervention_files) >= 1
    payload = json.loads(intervention_files[0].read_text(encoding="utf-8"))
    assert payload["reason_code"] == "RISKY_CONFIRMATION"


# --------------------------------------------------------------------------
# 8. The sensitive password never leaks into evidence
# --------------------------------------------------------------------------


def test_password_never_appears_in_events_or_result(mock_server, surface, policy, run_log) -> None:
    cap = _load_capability(mock_server)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("12345"))
    assert result.status == RunStatus.SUCCESS

    events_text = (run_log.dir / "events.jsonl").read_text(encoding="utf-8")
    result_text = (run_log.dir / "result.json").read_text(encoding="utf-8")
    assert "Passw0rd!" not in events_text
    assert "Passw0rd!" not in result_text


# --------------------------------------------------------------------------
# 9. Locator fallthrough: a bad first strategy still resolves via a later one
# --------------------------------------------------------------------------


def test_locator_fallthrough_records_rank_and_drift(mock_server, surface, policy, run_log) -> None:
    def _mutate(data: dict[str, Any]) -> None:
        # s2's target (username field) gets a bogus-but-VALID-ARIA-role
        # strategy prepended, guaranteed to match zero elements (there is
        # no checkbox named "Search" anywhere in the work frame) -- a
        # deterministic rank-0 failure, rather than depending on how
        # Playwright happens to behave for a truly invalid role string.
        steps = data["steps"]
        s2 = next(s for s in steps if s["id"] == "s2")
        s2["target"]["strategies"].insert(
            0,
            {
                "kind": "role",
                "params": {"role": "checkbox", "name": "Search"},
                "rationale": "deliberately wrong, for test_locator_fallthrough_records_rank_and_drift",
            },
        )

    cap = _load_capability(mock_server, mutate=_mutate)
    engine = ReplayEngine(surface, policy, run_log)
    result = engine.run(cap, _inputs("12345"))

    assert result.status == RunStatus.SUCCESS
    s2_result = next(s for s in result.steps if s.step_id == "s2")
    assert s2_result.strategy_rank is not None and s2_result.strategy_rank > 0
    assert s2_result.attempts >= 2
    assert any("s2" in d and "strategy rank" in d for d in result.drift_signals)


# --------------------------------------------------------------------------
# extra: the escalate seam actually plugs in (skip_step path)
# --------------------------------------------------------------------------


def test_escalate_handler_skip_step_is_honoured(mock_server, surface, policy, run_log) -> None:
    def _mutate(data: dict[str, Any]) -> None:
        data["steps"].append(
            {
                "id": "s8",
                "kind": "click",
                "screen": "member_detail",
                "intent": "Open a sub-account for this member",
                "origin": "agent",
                "target": {
                    "description": "Open Sub-Account link on the member detail page",
                    "frame": {"url_pattern": None, "name": "work", "index": None},
                    "strategies": [
                        {
                            "kind": "role",
                            "params": {"role": "link", "name": "Open Sub-Account", "exact": True},
                            "rationale": "the only link with this exact accessible name",
                        }
                    ],
                    "snapshot": {},
                },
                "value": None, "expect_dialog": None,
                "wait": {"until": "load", "timeout_ms": 10000},
                "checkpoint": None, "extract": None,
                "risk": {"level": "risky", "reason": "opens a sub-account", "requires_confirmation": True, "idempotency_check": None},
                "human_task": None,
            }
        )

    cap = _load_capability(mock_server, mutate=_mutate)

    def handler(request):
        assert request.reason_code == "RISKY_CONFIRMATION"
        return HumanResolution(action="skip_step", note="operator declined to open a sub-account")

    engine = ReplayEngine(surface, policy, run_log, escalate=handler)
    result = engine.run(cap, _inputs("12345"), approved_by=None)

    human_steps = [s for s in result.steps if s.status == "human_performed"]
    assert len(human_steps) == 1
    assert human_steps[0].step_id == "s8"
