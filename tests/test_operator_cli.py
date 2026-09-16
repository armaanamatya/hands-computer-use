r"""CLI tests for `hands operator` (hands/cli.py's operator subcommand
group): `status`/`list` read `control_state.json` / `intervention-*.json`
off disk, `claim`/`resume` POST to a running `OperatorServer` over HTTP.
Driven entirely through `main([...])`, mirroring tests/test_cli.py's own
convention of preferring that over a subprocess.

On-disk fixtures are built from the REAL Pydantic models in
hands/schema/control.py (`ControlStateRecord`, `InterventionRequest`), not
hand-rolled dicts, so their shape cannot silently drift from what
`SessionController` actually writes.

No Playwright anywhere in this file. `status`/`list` never touch a browser
(disk only). `claim` is proven against a page-less `SessionController`:
`OperatorServer`'s POST handlers only ever enqueue a `_Command` onto a
`queue.Queue` and return 202 -- they never call a controller method that
would need a live Playwright page (see operator_server.py's module
docstring: the HTTP thread must never touch Playwright, full stop). This
file therefore never requests the shared `browser` fixture from
tests/conftest.py, and imposes no ordering constraint relative to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hands import config
from hands.cli import main
from hands.evidence.run_log import RunLog
from hands.handoff.controller import SessionController
from hands.handoff.operator_server import OperatorServer
from hands.safety.policy import Policy
from hands.schema.control import ControlHolder, ControlState, ControlStateRecord, InterventionRequest


def _dummy_policy() -> Policy:
    """A minimal, self-contained Policy. None of these tests touch a live
    page, so almost every field here is inert -- mirrors
    tests/test_handoff.py's own `_dummy_policy()`."""
    return Policy.from_dict(
        {
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
        }
    )


# --------------------------------------------------------------------------
# 1. status prints the holder and state, read from control_state.json
# --------------------------------------------------------------------------


def test_status_prints_holder_and_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run-status"
    run_dir.mkdir()
    record = ControlStateRecord(
        run_id="status-run",
        state=ControlState.HUMAN_ACTIVE,
        holder=ControlHolder.HUMAN,
        epoch=1,
        handoff_id="status-run-iv1",
        reason="claimed_by:operator-1",
    )
    (run_dir / "control_state.json").write_text(json.dumps(record.model_dump(mode="json")), encoding="utf-8")

    rc = main(["operator", "status", "--run-dir", str(run_dir)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "HUMAN_ACTIVE" in out
    assert "human" in out
    assert "status-run-iv1" in out


# --------------------------------------------------------------------------
# 2. list prints the intervention's id and reason code
# --------------------------------------------------------------------------


def test_list_prints_intervention_id_and_reason_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run-list"
    run_dir.mkdir()
    req = InterventionRequest(
        id="list-run-iv1",
        run_id="list-run",
        mode="replay",
        reason_code="RISKY_CONFIRMATION",
        summary="opens a new sub-account record for the member",
    )
    (run_dir / f"intervention-{req.id}.json").write_text(json.dumps(req.model_dump(mode="json")), encoding="utf-8")

    rc = main(["operator", "list", "--run-dir", str(run_dir)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "list-run-iv1" in out
    assert "RISKY_CONFIRMATION" in out


# --------------------------------------------------------------------------
# 3. status with no --run-dir and no runs present: a clean exit, not a
#    traceback
# --------------------------------------------------------------------------


def test_status_with_no_run_dir_and_no_runs_exits_nonzero_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An EMPTY directory, not just a nonexistent one: this actual project's
    # evidence/runs/ already has real runs left over from earlier CLI/test
    # invocations, so config.EVIDENCE_RUNS must be pointed somewhere
    # genuinely empty for this test to be deterministic.
    empty_runs = tmp_path / "evidence_runs_empty"
    monkeypatch.setattr(config, "EVIDENCE_RUNS", empty_runs)

    rc = main(["operator", "status"])
    result = capsys.readouterr()
    combined = result.out + result.err

    assert rc != 0
    assert "Traceback" not in combined
    assert "hands:" in combined
    assert str(empty_runs) in combined


# --------------------------------------------------------------------------
# 4. claim against a running OperatorServer is accepted (202, queued)
# --------------------------------------------------------------------------


def test_claim_against_running_operator_server_is_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_log = RunLog(tmp_path, "claim-cli-run")
    controller = SessionController("claim-cli-run", run_log, _dummy_policy())  # page-less: no live browser needed
    req = InterventionRequest(
        id="claim-cli-run-iv1",
        run_id="claim-cli-run",
        mode="replay",
        reason_code="STUCK_NO_PROGRESS",
        summary="automation is stuck",
    )
    handoff_id = controller.request_handoff(req)
    assert controller.state == ControlState.HANDOFF_PENDING

    server = OperatorServer(controller, port=0)
    try:
        bound_port = server.start()

        rc = main(["operator", "claim", handoff_id, "--port", str(bound_port), "--operator-id", "cli-operator"])
        out = capsys.readouterr().out

        assert rc == 0
        # OperatorServer's own contract (operator_server.py's do_POST): every
        # POST returns 202 the instant the command is enqueued, with body
        # {"queued": True, ...} -- it does not wait for drain() to apply it.
        assert "queued" in out
        assert "True" in out

        # The HTTP thread must never touch the controller directly (see the
        # module docstring): right after the CLI call returns, nothing has
        # been applied yet.
        assert controller.state == ControlState.HANDOFF_PENDING
    finally:
        server.stop()
