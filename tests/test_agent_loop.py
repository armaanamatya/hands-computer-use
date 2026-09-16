"""Tests for the discovery loop (hands.agent.loop), driven by FakeLLM against
the real WebSurface and the real mock app (LegacyCore). No network calls and
no ANTHROPIC_API_KEY are needed - FakeLLM stands in for the model.

Element indices used in the scripts below were confirmed against the running
mock app (see the frame/element dump each stage produces); they are stable
because the app is server-rendered with no client-side JS altering the DOM.

All five tests transitively depend on the `browser` fixture (shared,
session-scoped, defined in tests/conftest.py), which calls `pytest.skip(...)`
if Chromium cannot be launched - that is what lets this suite be skipped in
an environment with no browser.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hands.agent.llm import FakeLLM
from hands.agent.loop import DiscoveryLoop, DiscoveryTrace
from hands.evidence.run_log import RunLog, new_run_id
from hands.safety.redact import Redactor
from hands.schema.control import ControlHolder, ControlToken
from hands.surface.web import WebSurface

# Fake, seeded credential from mockbank/data.py (USERS = {"teller1": "Passw0rd!"}).
# Hardcoding it here (rather than reading MOCKBANK_PASSWORD from .env) keeps
# this suite self-contained; it is documented as fake, non-secret seed data.
MOCKBANK_PASSWORD = "Passw0rd!"


# --------------------------------------------------------------------- fixtures
#
# `mock_server` (yields the mock app's PORT) and `browser` are shared,
# session-scoped fixtures defined in tests/conftest.py.


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


@pytest.fixture()
def page(browser):
    p = browser.new_page()
    yield p
    p.close()


def _make_loop(
    tmp_path: Path,
    page: Any,
    entry_url: str,
    goal: str,
    script: list[dict[str, Any]],
    inputs: dict[str, str] | None = None,
    sensitive_input_names: set[str] | None = None,
    max_steps: int = 20,
) -> tuple[DiscoveryTrace, RunLog]:
    redactor = Redactor()
    run_id = new_run_id("test")
    run_log = RunLog(tmp_path, run_id, redactor=redactor)
    surface = WebSurface(page)
    provider = FakeLLM(script)
    control_token = ControlToken(run_id=run_id, holder=ControlHolder.AUTOMATION, epoch=0)
    loop = DiscoveryLoop(
        surface=surface,
        run_log=run_log,
        redactor=redactor,
        provider=provider,
        goal=goal,
        inputs=inputs or {},
        sensitive_input_names=sensitive_input_names or set(),
        policy=None,
        control_token=control_token,
        max_steps=max_steps,
        run_timeout_s=120,
        entry_url=entry_url,
    )
    trace = loop.run()
    return trace, run_log


# ----------------------------------------------------------------- test 1 & 2 & 5
# (share one recorded happy-path run: sign on, look up member 12345, read the
# savings balance, done. Reusing the run keeps the suite from driving the
# same browser flow three times.)

_HAPPY_PATH_SCRIPT: list[dict[str, Any]] = [
    {"tool": "type_text", "input": {"element": 11, "text": "{{username}}", "reason": "enter the teller user id to sign on"}},
    {"tool": "type_text", "input": {"element": 13, "text": "{{password}}", "reason": "enter the teller password to sign on"}},
    {"tool": "click", "input": {"element": 14, "reason": "submit the sign-on form"}},
    {"tool": "type_text", "input": {"element": 11, "text": "{{member_id}}", "reason": "enter the member number to look up"}},
    {"tool": "click", "input": {"element": 14, "reason": "search for the member using the member number"}},
    {"tool": "read_value", "input": {"element": 26, "output_name": "savings_balance", "reason": "capture the savings account's current balance"}},
    {
        "tool": "done",
        "input": {
            "outputs": [{"name": "savings_balance", "value": "2,340.55"}],
            "summary": "Signed on, looked up member 12345, and read the savings account's current balance.",
            "reason": "the goal is satisfied: the balance has been captured",
        },
    },
]


@pytest.fixture(scope="module")
def happy_path_result(tmp_path_factory: pytest.TempPathFactory, mock_server: int, browser: Any) -> tuple[DiscoveryTrace, RunLog]:
    tmp_path = tmp_path_factory.mktemp("happy")
    pg = browser.new_page()
    try:
        trace, run_log = _make_loop(
            tmp_path,
            pg,
            _base_url(mock_server) + "/",
            goal="Sign on, look up member 12345, and report the savings account's current balance.",
            script=_HAPPY_PATH_SCRIPT,
            inputs={"username": "teller1", "password": MOCKBANK_PASSWORD, "member_id": "12345"},
            sensitive_input_names={"password"},
        )
    finally:
        pg.close()
    return trace, run_log


def test_happy_path_reaches_success(happy_path_result: tuple[DiscoveryTrace, RunLog]) -> None:
    trace, _ = happy_path_result
    assert trace.status == "success"
    assert trace.stop_reason == "done"
    assert trace.outputs.get("savings_balance") == "2,340.55"
    # One TraceStep per tool call in the script.
    assert len(trace.steps) == len(_HAPPY_PATH_SCRIPT)
    tool_sequence = [s.tool for s in trace.steps]
    assert tool_sequence == ["type_text", "type_text", "click", "type_text", "click", "read_value", "done"]


def test_password_never_leaks(happy_path_result: tuple[DiscoveryTrace, RunLog]) -> None:
    _, run_log = happy_path_result
    events_text = (run_log.dir / "events.jsonl").read_text(encoding="utf-8")
    transcript_text = (run_log.dir / "transcript.jsonl").read_text(encoding="utf-8")

    assert MOCKBANK_PASSWORD not in events_text
    assert MOCKBANK_PASSWORD not in transcript_text
    # The redaction marker must actually appear - proof the redactor fired
    # on the resolved password value, not just that it was never logged.
    assert "[redacted:password]" in events_text


def test_every_trace_step_has_a_reason(happy_path_result: tuple[DiscoveryTrace, RunLog]) -> None:
    trace, _ = happy_path_result
    assert len(trace.steps) > 0
    for step in trace.steps:
        assert step.reason is not None
        assert step.reason.strip() != ""


# --------------------------------------------------------------------- test 3

def test_dead_end_detection_stops_run(tmp_path: Path, mock_server: int, page: Any) -> None:
    # The menu frame's "Home" link (idx 2) is stable across observations and,
    # unauthenticated, clicking it just reloads the same sign-on screen each
    # time - a genuine no-op repeated four times.
    script = [{"tool": "click", "input": {"element": 2, "reason": "go back to the home screen"}} for _ in range(4)]
    trace, _ = _make_loop(
        tmp_path,
        page,
        _base_url(mock_server) + "/",
        goal="Keep returning to the home screen.",
        script=script,
    )
    assert trace.status == "stopped"
    assert trace.stop_reason == "no_progress"
    # The dead-end must be caught before the script is exhausted (3rd repeat).
    assert len(trace.steps) <= 3


# --------------------------------------------------------------------- test 4

def test_request_human_produces_intervention(tmp_path: Path, mock_server: int, page: Any) -> None:
    script = [
        {
            "tool": "request_human",
            "input": {
                "reason_code": "UNEXPECTED_SCREEN",
                "reason": "the screen shown does not match anything the goal describes",
            },
        }
    ]
    trace, run_log = _make_loop(
        tmp_path,
        page,
        _base_url(mock_server) + "/",
        goal="Do something this application cannot be made to do.",
        script=script,
    )
    assert trace.status == "needs_human"
    assert trace.stop_reason == "request_human"
    assert trace.intervention is not None
    assert trace.intervention.reason_code == "UNEXPECTED_SCREEN"
    assert trace.intervention.run_id == run_log.run_id
    assert trace.intervention.mode == "discovery"
