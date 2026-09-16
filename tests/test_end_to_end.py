"""The join the whole system rests on: discover -> record -> replay.

Every other suite tests one half. `tests/test_agent_loop.py` proves the model
can drive the surface; `tests/test_replay.py` proves a *hand-authored*
capability replays. Neither proves that an artifact the RECORDER produced is
one the REPLAY ENGINE can actually execute, which is the entire premise
("the model discovers, the artifact becomes a reusable capability").

So this suite records an artifact from a discovery run and then replays it,
with no model in the loop the second time, against the same live app. The
discovery half is driven by FakeLLM so it needs no API key; what is under
test here is the recorder/replay contract, not the model.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from hands.agent.llm import FakeLLM
from hands.agent.loop import DiscoveryLoop
from hands.agent.recorder import record
from hands.evidence.run_log import RunLog, new_run_id
from hands.replay.engine import ReplayEngine
from hands.safety.policy import Policy
from hands.safety.redact import Redactor
from hands.schema.capability import AppInfo, Capability, InputSpec, OutputSpec
from hands.schema.control import ControlHolder, ControlToken
from hands.schema.result import RunStatus
from hands.surface.web import WebSurface

# Fake seeded credential from mockbank/data.py. Documented as non-secret.
MOCKBANK_PASSWORD = "Passw0rd!"

# Element indices are stable because the app is server-rendered with no
# client-side scripting that alters the DOM. Mirrors tests/test_agent_loop.py.
DISCOVERY_SCRIPT: list[dict[str, Any]] = [
    {"tool": "type_text", "input": {"element": 11, "text": "{{username}}", "reason": "enter the teller user id to sign on"}},
    {"tool": "type_text", "input": {"element": 13, "text": "{{password}}", "reason": "enter the teller password to sign on"}},
    {"tool": "click", "input": {"element": 14, "reason": "submit the sign-on form"}},
    {"tool": "type_text", "input": {"element": 11, "text": "{{member_id}}", "reason": "enter the member number to look up"}},
    {"tool": "click", "input": {"element": 14, "reason": "search for the member using the member number"}},
    {"tool": "read_value", "input": {"element": 26, "output_name": "savings_balance", "reason": "capture the savings account current balance"}},
    {
        "tool": "done",
        "input": {
            "outputs": [{"name": "savings_balance", "value": "2,340.55"}],
            "summary": "Signed on, looked up the member, and read the savings balance.",
            "reason": "the goal is satisfied: the balance has been captured",
        },
    },
]


@pytest.fixture(scope="module")
def mock_server() -> Iterator[str]:
    from werkzeug.serving import make_server

    from mockbank.app import app

    srv = make_server("127.0.0.1", 0, app, threaded=True)
    port = srv.socket.getsockname()[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        t.join(timeout=5)


def _policy_for(base_url: str) -> Policy:
    """policy.yaml pins the mock app to port 5050, but these tests bind an
    ephemeral port. Substitute it rather than loosening the shipped policy:
    a URL outside the allowlist is *supposed* to be blocked, and the first
    draft of this suite proved it by getting POLICY_BLOCKED.
    """
    import yaml

    raw = yaml.safe_load(Path("policy.yaml").read_text(encoding="utf-8"))
    port = base_url.rsplit(":", 1)[-1].rstrip("/")
    raw["allowed_url_patterns"] = [rf"^http://(127\.0\.0\.1|localhost):{port}(/.*)?$"]
    return Policy.from_dict(raw)


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover
        pytest.skip("playwright is not installed")
        return
    pw = sync_playwright().start()
    try:
        b = pw.chromium.launch(headless=True)
    except Exception as e:  # noqa: BLE001
        pw.stop()
        pytest.skip(f"Chromium is not available: {e}")
        return
    yield b
    b.close()
    pw.stop()


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory, mock_server: str, browser: Any):
    """Run discovery once, then record the trace into a Capability."""
    tmp_path = tmp_path_factory.mktemp("e2e_discover")
    inputs = {"username": "teller1", "password": MOCKBANK_PASSWORD, "member_id": "12345"}
    redactor = Redactor({"password": MOCKBANK_PASSWORD})
    run_id = new_run_id("e2e-discover")
    run_log = RunLog(tmp_path, run_id, redactor=redactor)
    page = browser.new_page(viewport={"width": 1100, "height": 750})
    try:
        loop = DiscoveryLoop(
            surface=WebSurface(page),
            run_log=run_log,
            redactor=redactor,
            provider=FakeLLM(DISCOVERY_SCRIPT),
            goal="Look up the member and read their current savings balance.",
            inputs=inputs,
            sensitive_input_names={"password"},
            policy=None,
            control_token=ControlToken(run_id=run_id, holder=ControlHolder.AUTOMATION, epoch=0),
            max_steps=20,
            run_timeout_s=120,
            entry_url=mock_server + "/",
        )
        trace = loop.run()
    finally:
        page.close()

    assert trace.status == "success", f"discovery did not succeed: {trace.stop_reason}"

    result = record(
        trace,
        capability_id="e2e_lookup_savings_balance",
        name="Look up member savings balance",
        description="Sign on, look up a member by number, and read the savings current balance.",
        inputs_spec={
            "username": InputSpec(type="string", description="Teller user id."),
            "password": InputSpec(type="string", description="Teller password.", sensitive=True),
            "member_id": InputSpec(type="string", description="Member number to look up."),
        },
        outputs_spec={"savings_balance": OutputSpec(type="money", description="Savings current balance.")},
        policy=_policy_for(mock_server),
        redactor=redactor,
        entry_url=mock_server + "/",
        app=AppInfo(vendor="LegacyCore", product="CBS", version="7.2"),
        discovery_run_id=run_id,
        inputs=inputs,
    )
    inputs["__base_url"] = mock_server + "/"
    return result, inputs


def test_recorded_artifact_is_coherent(recorded) -> None:
    result, _ = recorded
    cap = result.capability
    assert cap.validate_semantics() == [], "recorder emitted an artifact the schema rejects"
    assert cap.content_hash, "content hash must be set so the artifact is reviewable"
    assert cap.steps, "recorder produced no steps"


def test_recorded_artifact_does_not_contain_the_password(recorded) -> None:
    result, _ = recorded
    blob = result.capability.model_dump_json()
    assert MOCKBANK_PASSWORD not in blob, "a secret reached the artifact"
    assert "{{password}}" in blob, "the password step must be parameterised, not dropped"


def test_recorded_artifact_replays_without_a_model(recorded, tmp_path: Path, browser: Any) -> None:
    """The load-bearing test: replay the RECORDED artifact, no LLM involved.

    If the recorder and the replay engine disagree about anything - frame
    matchers, screen identification, locator strategies, checkpoints - this
    is where it surfaces.
    """
    result, inputs = recorded
    cap = Capability.model_validate_json(result.capability.model_dump_json())

    redactor = Redactor({"password": inputs["password"]})
    run_id = new_run_id("e2e-replay")
    run_log = RunLog(tmp_path, run_id, redactor=redactor)
    page = browser.new_page(viewport={"width": 1100, "height": 750})
    try:
        engine = ReplayEngine(
            surface=WebSurface(page, dialog_mode="hold"),
            policy=_policy_for(inputs["__base_url"]),
            run_log=run_log,
        )
        replay = engine.run(cap, {k: v for k, v in inputs.items() if not k.startswith("__")})
    finally:
        page.close()

    assert replay.status == RunStatus.SUCCESS, (
        f"recorded artifact failed to replay: {replay.status} "
        f"{replay.failure.category if replay.failure else ''} "
        f"{replay.failure.expected if replay.failure else ''}"
    )
    assert replay.outputs.get("savings_balance"), "replay captured no savings balance"
    assert "2340.55" in str(replay.outputs["savings_balance"]).replace(",", "")


def test_recorded_artifact_is_parameterised_over_the_member(recorded, tmp_path: Path, browser: Any) -> None:
    """Replay the recorded artifact with a DIFFERENT member than the one used
    during discovery. This is what separates a recording from a capability:
    if the member number had been baked in, this returns 12345's balance (or
    fails) instead of 23456's."""
    result, inputs = recorded
    cap = Capability.model_validate_json(result.capability.model_dump_json())

    redactor = Redactor({"password": inputs["password"]})
    run_id = new_run_id("e2e-replay-other")
    run_log = RunLog(tmp_path, run_id, redactor=redactor)
    page = browser.new_page(viewport={"width": 1100, "height": 750})
    try:
        engine = ReplayEngine(
            surface=WebSurface(page, dialog_mode="hold"),
            policy=_policy_for(inputs["__base_url"]),
            run_log=run_log,
        )
        replay = engine.run(cap, {**{k: v for k, v in inputs.items() if not k.startswith("__")}, "member_id": "23456"})
    finally:
        page.close()

    assert replay.status == RunStatus.SUCCESS, (
        f"parameterised replay failed: {replay.status} "
        f"{replay.failure.category if replay.failure else ''}"
    )
    balance = str(replay.outputs.get("savings_balance", "")).replace(",", "")
    assert "15020.31" in balance, f"expected member 23456's savings balance, got {balance!r}"


def test_recorded_artifact_reports_a_business_outcome_for_a_missing_member(
    recorded, tmp_path: Path, browser: Any
) -> None:
    """A recorded artifact only saw the happy path, so this also checks that
    the curated outcome library merged in at record time does its job."""
    result, inputs = recorded
    cap = Capability.model_validate_json(result.capability.model_dump_json())

    redactor = Redactor({"password": inputs["password"]})
    run_id = new_run_id("e2e-replay-missing")
    run_log = RunLog(tmp_path, run_id, redactor=redactor)
    page = browser.new_page(viewport={"width": 1100, "height": 750})
    try:
        engine = ReplayEngine(
            surface=WebSurface(page, dialog_mode="hold"),
            policy=_policy_for(inputs["__base_url"]),
            run_log=run_log,
        )
        replay = engine.run(cap, {**{k: v for k, v in inputs.items() if not k.startswith("__")}, "member_id": "99999"})
    finally:
        page.close()

    # Without a merged outcome library the recorder cannot know about
    # "no member found", so the honest expectation is: this must NOT be
    # reported as a success, and must not crash. Either a declared business
    # outcome or a clean, classified failure is acceptable.
    assert replay.status != RunStatus.SUCCESS
    if replay.status == RunStatus.BUSINESS_OUTCOME:
        assert replay.outcome and replay.outcome.code
        assert replay.exit_code() == 0
    else:
        assert replay.failure is not None and replay.failure.category
