r"""End-to-end tests for the `hands` CLI (hands/cli.py), driven entirely
through `main([...])` per the task's own instruction to prefer that over a
subprocess where practical.

Port handling for the replay tests mirrors tests/test_replay.py's own
module docstring exactly: the mock app is started on a free localhost port,
and `--base-url` is what rewrites the capability's (and, internally, the
operator policy's `allowed_url_patterns`) port to match -- see
`hands.cli._rewrite_capability_base_url` / `hands.cli._load_policy`.

The discover test needs the SAME free-port problem solved a different way,
because `discover` has no `--base-url` flag: it loads whatever policy.yaml
`--policy` points to as-is. So `_policy_for_port` below reproduces the same
port-only substitution `hands.cli._load_policy` does internally for
replay, but as a real policy.yaml file on disk, passed via `--policy`.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Iterator

import pytest
from werkzeug.serving import make_server

from hands.cli import main
from hands.schema.capability import Capability
from mockbank.app import app as flask_app

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "lookup_savings_balance.json"
POLICY_YAML_PATH = REPO_ROOT / "policy.yaml"

MOCKBANK_PASSWORD = "Passw0rd!"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_server() -> Iterator[int]:
    """The real LegacyCore mock app, on a free localhost port, for the
    whole module -- same pattern as tests/test_replay.py and
    tests/test_agent_loop.py.
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


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def _policy_for_port(tmp_path: Path, port: int) -> Path:
    """A real policy.yaml on disk, identical to the project's own except
    with the port in `allowed_url_patterns` substituted for `port` -- the
    same port-only substitution `hands.cli._load_policy` applies internally
    for `replay --base-url`, needed here because `discover` has no
    --base-url flag and loads --policy as a literal file.
    """
    text = POLICY_YAML_PATH.read_text(encoding="utf-8").replace(":5050", f":{port}")
    out = tmp_path / "policy.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def _replay_args(capability_path: Path, base_url: str, member_id: str, extra: list[str] | None = None) -> list[str]:
    return [
        "replay",
        str(capability_path),
        "--base-url",
        base_url,
        "--input",
        "username=teller1",
        "--input",
        f"password={MOCKBANK_PASSWORD}",
        "--input",
        f"member_id={member_id}",
        *(extra or []),
    ]


# The exact happy-path script from tests/test_agent_loop.py's
# _HAPPY_PATH_SCRIPT: sign on, look up member 12345, read the savings
# balance, done. Element indices are stable (the mock app is server-
# rendered with no client-side JS altering the DOM).
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


def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("HANDS_PROVIDER", raising=False)


def _forbid_playwright(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything tries to launch Playwright -- used to prove
    a code path returns before ever touching a browser.
    """
    import playwright.sync_api as pw_mod

    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("hands must not launch Playwright on this path")

    monkeypatch.setattr(pw_mod, "sync_playwright", _boom)


# --------------------------------------------------------------------------
# 1. schema export
# --------------------------------------------------------------------------


def test_schema_export_writes_schema_and_tool_definitions(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["schema", "export"])
    assert rc == 0

    schema_path = REPO_ROOT / "capabilities" / "schema" / "capability.schema.json"
    tool_path = REPO_ROOT / "capabilities" / "schema" / "legacycore.lookup_savings_balance.tool.json"
    assert schema_path.is_file()
    assert tool_path.is_file()

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema.get("title") == "Capability" or "properties" in schema

    tool_def = json.loads(tool_path.read_text(encoding="utf-8"))
    assert "input_schema" in tool_def
    props = tool_def["input_schema"]["properties"]
    assert set(props) == {"member_id", "username", "password"}
    assert tool_def["input_schema"]["required"]

    out = capsys.readouterr().out
    assert "wrote" in out
    assert str(schema_path) in out
    assert str(tool_path) in out


# --------------------------------------------------------------------------
# 2. inspect
# --------------------------------------------------------------------------


def test_inspect_prints_id_every_step_id_and_risk_levels(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["inspect", str(CAPABILITY_PATH)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "legacycore.lookup_savings_balance" in out

    cap = Capability.model_validate(json.loads(CAPABILITY_PATH.read_text(encoding="utf-8")))
    for step in cap.steps:
        assert re.search(rf"^\s*{re.escape(step.id)}\s", out, flags=re.MULTILINE), f"missing step id {step.id!r}"

    # s4 (Sign On) is recorded risky with an explicit reason; every other
    # step is safe -- both levels must actually appear in the output.
    assert "risk: risky -- submits credentials and establishes an authenticated teller session" in out
    assert "risk: safe" in out


# --------------------------------------------------------------------------
# 3-5. replay against the real mock app
# --------------------------------------------------------------------------


def test_replay_member_12345_succeeds_and_prints_the_savings_balance(
    mock_server: int, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(_replay_args(CAPABILITY_PATH, _base_url(mock_server), "12345"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "status: SUCCESS" in out
    assert "2340.55" in out


def test_replay_member_99999_is_a_business_outcome_not_a_failure(
    mock_server: int, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(_replay_args(CAPABILITY_PATH, _base_url(mock_server), "99999"))
    out = capsys.readouterr().out

    # A legitimate business answer ("no such member") must exit 0, never 1.
    assert rc == 0
    assert "BUSINESS_OUTCOME" in out
    assert "MEMBER_NOT_FOUND" in out


def test_replay_member_50000_fails_with_app_error_and_evidence_path(
    mock_server: int, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(_replay_args(CAPABILITY_PATH, _base_url(mock_server), "50000"))
    out = capsys.readouterr().out

    assert rc == 1
    assert "APP_ERROR" in out
    assert "evidence" in out.lower()
    assert "screenshot" in out


# --------------------------------------------------------------------------
# 6. a capability that fails validate_semantics() -- never touches a browser
# --------------------------------------------------------------------------


def test_replay_semantically_broken_capability_exits_1_without_a_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    # A schema-VALID but semantically broken capability: s2 declares a
    # screen id that is not in `screens` at all.
    steps = data["steps"]
    s2 = next(s for s in steps if s["id"] == "s2")
    s2["screen"] = "totally_unknown_screen"
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(data), encoding="utf-8")

    _forbid_playwright(monkeypatch)

    rc = main(
        [
            "replay",
            str(broken_path),
            "--input",
            "username=teller1",
            "--input",
            f"password={MOCKBANK_PASSWORD}",
            "--input",
            "member_id=12345",
        ]
    )
    result = capsys.readouterr()
    combined = result.out + result.err

    assert rc == 1
    assert "totally_unknown_screen" in combined
    assert "validate_semantics" in combined


# --------------------------------------------------------------------------
# 7. discover --fake-llm, no API key, end to end
# --------------------------------------------------------------------------


def test_discover_fake_llm_runs_with_no_api_key_and_writes_a_valid_capability(
    mock_server: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_api_keys(monkeypatch)

    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(_HAPPY_PATH_SCRIPT), encoding="utf-8")
    policy_path = _policy_for_port(tmp_path, mock_server)
    out_path = tmp_path / "discovered.json"

    rc = main(
        [
            "discover",
            "--goal",
            "Sign on, look up member 12345, and report the savings account's current balance.",
            "--url",
            _base_url(mock_server),
            "--input",
            "username=teller1",
            "--input",
            f"password={MOCKBANK_PASSWORD}",
            "--input",
            "member_id=12345",
            "--fake-llm",
            str(script_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--capability-id",
            "test.discover_happy_path",
            "--max-steps",
            "20",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert out_path.is_file()

    cap = Capability.model_validate(json.loads(out_path.read_text(encoding="utf-8")))
    assert cap.validate_semantics() == []
    assert cap.id == "test.discover_happy_path"

    assert "status: success" in out
    assert str(out_path) in out


# --------------------------------------------------------------------------
# 8. discover with no API key and no --fake-llm: clean failure, no browser
# --------------------------------------------------------------------------


def test_discover_without_api_key_or_fake_llm_fails_cleanly_without_a_browser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_api_keys(monkeypatch)
    _forbid_playwright(monkeypatch)

    rc = main(
        [
            "discover",
            "--goal",
            "Sign on and look up a member.",
            "--input",
            "username=teller1",
            "--input",
            f"password={MOCKBANK_PASSWORD}",
            "--input",
            "member_id=12345",
        ]
    )
    result = capsys.readouterr()
    combined = result.out + result.err

    assert rc != 0
    assert "ANTHROPIC_API_KEY" in combined


# --------------------------------------------------------------------------
# 9. a sensitive input value never appears in discover's stdout
# --------------------------------------------------------------------------


def test_discover_never_prints_a_sensitive_input_value(
    mock_server: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_api_keys(monkeypatch)

    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(_HAPPY_PATH_SCRIPT), encoding="utf-8")
    policy_path = _policy_for_port(tmp_path, mock_server)
    out_path = tmp_path / "discovered.json"

    rc = main(
        [
            "discover",
            "--goal",
            "Sign on, look up member 12345, and report the savings account's current balance.",
            "--url",
            _base_url(mock_server),
            "--input",
            "username=teller1",
            "--input",
            f"password={MOCKBANK_PASSWORD}",
            "--input",
            "member_id=12345",
            "--fake-llm",
            str(script_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--capability-id",
            "test.discover_no_secret_leak",
            "--max-steps",
            "20",
        ]
    )
    result = capsys.readouterr()
    combined = result.out + result.err

    assert rc == 0
    assert MOCKBANK_PASSWORD not in combined
    # The written artifact must not contain it either -- belt and suspenders
    # for the "never serialized" half of the same requirement.
    assert MOCKBANK_PASSWORD not in out_path.read_text(encoding="utf-8")
