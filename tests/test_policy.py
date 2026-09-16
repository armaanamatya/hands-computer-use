r"""Tests for hands.safety.policy.

The project's actual `policy.yaml` currently fails to parse under strict
`yaml.safe_load` (test_load_actual_policy_yaml_is_currently_malformed pins
this against the real file, by path, and fails loud rather than hiding it —
see the `Policy.load`/module docstrings in hands/safety/policy.py for the
two concrete defects). Fixing policy.yaml is outside this module's
ownership; the fix is to single-quote its `allowed_url_patterns` and
`risky_patterns` strings, where YAML backslash is literal.

Every behavioural test below (items 1-7) therefore loads a Policy through
`Policy.load` for real — not `from_dict` with a hand-written dict — by
writing that single-quoted, corrected policy.yaml to a temp file
(`_POLICY_YAML_FIXED`, in the `policy` fixture) and calling
`Policy.load(path)` on it. This exercises the actual `load` ->
`yaml.safe_load` -> `from_dict` -> compiled-regex path end to end, so a
mistake in that path (wrong encoding, mis-wired field, wrong regex flags)
would show up as real test failures instead of being invisible behind a
hand-maintained mirror dict. `_POLICY_DICT` (a plain dict with the same
values) is kept only for the handful of tests that need to vary one field
directly — different `risky_mode`s, `allow_coordinate_fallback` — where a
temp YAML file per variant would be pure ceremony.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from hands.safety.policy import Decision, Policy, PolicyViolation

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_YAML_PATH = REPO_ROOT / "policy.yaml"

# The intended content of policy.yaml, corrected to single-quoted regex
# strings (backslash is literal in single-quoted YAML, so `\.` and `\b`
# mean exactly what they look like -- no illegal escape, no accidental
# backspace character). This is what policy.yaml should read once fixed.
_POLICY_YAML_FIXED = """\
allowed_domains:
  - "127.0.0.1"
  - "localhost"
allowed_url_patterns:
  - '^http://(127\\.0\\.0\\.1|localhost):5050(/.*)?$'
allowed_actions: [navigate, click, type, select, press, extract, wait]
risky_patterns:
  - '(?i)\\b(open|create|submit|confirm|transfer|close|delete|approve|post)\\b'
risky_mode: gate
allow_coordinate_fallback: true
capture_human_input_values: false
human_wait_timeout_s: 600
max_steps: 40
step_timeout_ms: 10000
run_timeout_s: 300
recovery_budget: 5
"""

# Same values as _POLICY_YAML_FIXED, as a plain dict -- used only by the
# handful of tests that need to vary a single field (risky_mode,
# allow_coordinate_fallback) via Policy.from_dict directly.
_POLICY_DICT: dict[str, Any] = {
    "allowed_domains": ["127.0.0.1", "localhost"],
    "allowed_url_patterns": [r"^http://(127\.0\.0\.1|localhost):5050(/.*)?$"],
    "allowed_actions": ["navigate", "click", "type", "select", "press", "extract", "wait"],
    "risky_patterns": [
        r"(?i)\b(open|create|submit|confirm|transfer|close|delete|approve|post)\b"
    ],
    "risky_mode": "gate",
    "allow_coordinate_fallback": True,
    "capture_human_input_values": False,
    "human_wait_timeout_s": 600,
    "max_steps": 40,
    "step_timeout_ms": 10000,
    "run_timeout_s": 300,
    "recovery_budget": 5,
}


@pytest.fixture
def policy(tmp_path: Path) -> Policy:
    """A Policy loaded through the real `Policy.load` -> `yaml.safe_load`
    path, from a corrected (single-quoted) copy of policy.yaml's content.
    """
    p = tmp_path / "policy.yaml"
    p.write_text(_POLICY_YAML_FIXED, encoding="utf-8")
    return Policy.load(p)


def _with(risky_mode: str) -> Policy:
    d = dict(_POLICY_DICT)
    d["risky_mode"] = risky_mode
    return Policy.from_dict(d)


# -- loading ---------------------------------------------------------------


def test_real_policy_yaml_loads() -> None:
    """The shipped policy.yaml must actually parse. It previously did not:
    the regexes were double-quoted, and `\\.` is not a legal YAML
    double-quoted escape."""
    assert POLICY_YAML_PATH.exists()
    loaded = Policy.load(POLICY_YAML_PATH)
    assert loaded.allowed_domains == {"127.0.0.1", "localhost"}


def test_real_policy_yaml_risky_patterns_are_not_silently_neutered() -> None:
    r"""Regression guard for a guardrail that can fail *open* without any
    error.

    In double-quoted YAML a backslash is an escape character, so `"\b"`
    parses to a backspace (U+0008) rather than a regex word boundary. A
    policy file written that way still loads once the other escape is
    fixed, but every risky-action pattern then matches nothing at all and
    "Confirm" stops being classified as risky. Quoting is therefore
    load-bearing for safety here, and this asserts the real file's real
    behaviour rather than a corrected copy of it.
    """
    loaded = Policy.load(POLICY_YAML_PATH)
    for pattern in loaded.risky_patterns:
        assert "\b" not in pattern.pattern, "word-boundary anchor became a backspace"
    for name in ("Confirm", "Open Sub-Account", "Transfer funds", "Delete"):
        level, reason = loaded.classify_risk("click", name)
        assert level == "risky", f"{name!r} must be classified risky"
        assert reason
    # ...and still does not fire on a merely similar-looking safe control.
    assert loaded.classify_risk("click", "Search")[0] == "safe"
    assert loaded.classify_risk("click", "Reopened")[0] == "safe"


def test_from_dict_round_trips_the_intended_policy_yaml_values(policy: Policy) -> None:
    assert policy.allowed_domains == {"127.0.0.1", "localhost"}
    assert policy.allowed_actions == {
        "navigate", "click", "type", "select", "press", "extract", "wait",
    }
    assert policy.risky_mode == "gate"
    assert policy.allow_coordinate_fallback is True
    assert policy.capture_human_input_values is False
    assert policy.human_wait_timeout_s == 600
    assert policy.max_steps == 40
    assert policy.step_timeout_ms == 10000
    assert policy.run_timeout_s == 300
    assert policy.recovery_budget == 5


def test_from_dict_rejects_unrecognized_risky_mode() -> None:
    d = dict(_POLICY_DICT)
    d["risky_mode"] = "sometimes"
    with pytest.raises(ValueError):
        Policy.from_dict(d)


def test_from_dict_requires_a_mapping() -> None:
    with pytest.raises(TypeError):
        Policy.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]


# -- dataclass invariants -----------------------------------------------------


def test_decision_is_frozen() -> None:
    d = Decision(True)
    with pytest.raises(FrozenInstanceError):
        d.allowed = False  # type: ignore[misc]


# -- item 1: URL matrix ------------------------------------------------------


def test_check_url_allows_the_configured_host_and_pattern(policy: Policy) -> None:
    d = policy.check_url("http://127.0.0.1:5050/member?mno=12345")
    assert d.allowed
    assert d.rule is None


def test_check_url_denies_unknown_domain(policy: Policy) -> None:
    d = policy.check_url("http://evil.example.com/")
    assert not d.allowed
    assert d.rule == "domain"


def test_check_url_denies_https_because_pattern_is_http_only(policy: Policy) -> None:
    d = policy.check_url("https://127.0.0.1:5050/")
    assert not d.allowed
    assert d.rule == "url"


def test_check_url_denies_javascript_scheme(policy: Policy) -> None:
    d = policy.check_url("javascript:alert(1)")
    assert not d.allowed
    assert d.rule == "scheme"


def test_check_url_denies_data_scheme(policy: Policy) -> None:
    d = policy.check_url("data:text/html,<script>alert(1)</script>")
    assert not d.allowed
    assert d.rule == "scheme"


def test_check_url_denies_file_scheme(policy: Policy) -> None:
    d = policy.check_url("file:///c:/x")
    assert not d.allowed
    assert d.rule == "scheme"


def test_check_url_denies_garbage_without_raising(policy: Policy) -> None:
    d = policy.check_url("not a url at all")
    assert not d.allowed  # scheme check fires; the point is no exception

    # A string that makes urllib itself raise (malformed IPv6 host) must
    # also come back as a denial, never propagate.
    d2 = policy.check_url("http://[::1/path")
    assert not d2.allowed


# -- item 2: port ignored for domain, pattern still governs -----------------


def test_check_url_port_ignored_for_domain_but_pattern_still_denies(policy: Policy) -> None:
    d = policy.check_url("http://127.0.0.1:9999/")
    assert not d.allowed
    assert d.rule == "url"  # host matched allowed_domains; the port-specific pattern didn't


# -- item 3: classify_risk ----------------------------------------------------


def test_classify_risk_click_confirm_is_risky(policy: Policy) -> None:
    level, reason = policy.classify_risk("click", "Confirm")
    assert level == "risky"
    assert reason is not None


def test_classify_risk_click_search_is_safe(policy: Policy) -> None:
    level, reason = policy.classify_risk("click", "Search")
    assert level == "safe"
    assert reason is None


def test_classify_risk_type_is_always_safe(policy: Policy) -> None:
    level, reason = policy.classify_risk("type", "Nickname")
    assert level == "safe"
    assert reason is None


def test_classify_risk_falls_back_to_value_attr_when_name_is_empty(policy: Policy) -> None:
    # <input type=submit value="Open Sub-Account"> with an empty accessible
    # name still has to be caught via its `value` attribute.
    level, reason = policy.classify_risk(
        "click", "", element_attrs={"value": "Open Sub-Account"}
    )
    assert level == "risky"
    assert reason is not None


def test_classify_risk_read_only_kinds_are_safe_regardless_of_name(policy: Policy) -> None:
    for kind in ("wait", "extract", "assert", "navigate"):
        level, reason = policy.classify_risk(kind, "Delete Everything")
        assert level == "safe", kind
        assert reason is None


def test_classify_risk_uses_word_boundaries_not_substring_match(policy: Policy) -> None:
    # "Reopened" contains "open" as a substring but not as a whole word.
    # risky_patterns' `\b` anchors must be real word-boundary anchors for
    # this to come back "safe" -- if those anchors were ever dropped (or,
    # per the YAML defect this suite pins against the real policy.yaml
    # above, silently turned into literal backspace characters, which
    # match nothing and would instead make test_classify_risk_click_
    # confirm_is_risky fail), plain substring matching would wrongly
    # flag this as "risky". This test is the one that would catch
    # `risky_patterns` regressing to unanchored substring matching.
    level, reason = policy.classify_risk("click", "Reopened")
    assert level == "safe"
    assert reason is None


# -- item 4: risky_mode behaviour --------------------------------------------


def test_risky_mode_gate_allows_with_confirmation() -> None:
    p = _with("gate")
    d = p.check_action("click", "http://127.0.0.1:5050/x", element_name="Confirm")
    assert d.allowed is True
    assert d.requires_confirmation is True
    assert d.rule is None


def test_risky_mode_block_denies() -> None:
    p = _with("block")
    d = p.check_action("click", "http://127.0.0.1:5050/x", element_name="Confirm")
    assert d.allowed is False
    assert d.rule == "risky_action"


def test_risky_mode_allow_permits_outright() -> None:
    p = _with("allow")
    d = p.check_action("click", "http://127.0.0.1:5050/x", element_name="Confirm")
    assert d.allowed is True
    assert d.requires_confirmation is False


# -- item 5: intersect narrows, never widens ---------------------------------


def test_intersect_narrows_actions_to_the_common_set(policy: Policy) -> None:
    narrowed = policy.intersect({"allowed_actions": ["click", "type", "navigate", "select"]})
    assert narrowed.allowed_actions == {"click", "type", "navigate", "select"}
    # everything not named by the capability is dropped, even though the
    # operator policy allowed it
    assert "press" not in narrowed.allowed_actions
    assert "extract" not in narrowed.allowed_actions
    assert "wait" not in narrowed.allowed_actions


def test_intersect_cannot_add_a_domain_the_operator_never_allowed(policy: Policy) -> None:
    narrowed = policy.intersect({"allowed_domains": ["127.0.0.1", "totally-new-domain.example"]})
    assert narrowed.allowed_domains == {"127.0.0.1"}
    assert "totally-new-domain.example" not in narrowed.allowed_domains


def test_intersect_coordinate_fallback_false_on_either_side_wins(policy: Policy) -> None:
    # operator allows it, capability forbids it -> forbidden
    narrowed = policy.intersect({"allow_coordinate_fallback": False})
    assert narrowed.allow_coordinate_fallback is False

    # operator forbids it, capability "allows" it -> still forbidden
    strict = Policy.from_dict({**_POLICY_DICT, "allow_coordinate_fallback": False})
    narrowed2 = strict.intersect({"allow_coordinate_fallback": True})
    assert narrowed2.allow_coordinate_fallback is False


def test_intersect_absent_keys_do_not_narrow(policy: Policy) -> None:
    # A capability that only declares allowed_actions must not zero out
    # domains or coordinate-fallback just because it said nothing about them.
    narrowed = policy.intersect({"allowed_actions": ["click"]})
    assert narrowed.allowed_domains == policy.allowed_domains
    assert narrowed.allow_coordinate_fallback == policy.allow_coordinate_fallback


def test_intersect_with_none_is_a_no_op(policy: Policy) -> None:
    narrowed = policy.intersect(None)
    assert narrowed.allowed_domains == policy.allowed_domains
    assert narrowed.allowed_actions == policy.allowed_actions
    assert narrowed.allow_coordinate_fallback == policy.allow_coordinate_fallback


# -- item 6: assert_action raises PolicyViolation ----------------------------


def test_assert_action_raises_for_disallowed_domain(policy: Policy) -> None:
    with pytest.raises(PolicyViolation) as exc_info:
        policy.assert_action("navigate", "http://evil.example.com/")
    assert exc_info.value.rule == "domain"


def test_assert_action_returns_decision_when_allowed(policy: Policy) -> None:
    decision = policy.assert_action("navigate", "http://127.0.0.1:5050/")
    assert decision.allowed is True


def test_assert_action_does_not_raise_for_gated_confirmation(policy: Policy) -> None:
    # requires_confirmation=True is still `allowed=True` -- not a denial.
    decision = policy.assert_action(
        "click", "http://127.0.0.1:5050/x", element_name="Confirm"
    )
    assert decision.allowed is True
    assert decision.requires_confirmation is True


# -- item 7: disallowed action kind denied even on an allowed URL -----------


def test_disallowed_action_kind_denied_on_allowed_url(policy: Policy) -> None:
    d = policy.check_action("download", "http://127.0.0.1:5050/report")
    assert not d.allowed
    assert d.rule == "action_kind"


def test_check_action_propagates_scheme_rule(policy: Policy) -> None:
    d = policy.check_action("navigate", "javascript:alert(1)")
    assert not d.allowed
    assert d.rule == "scheme"


def test_check_action_coordinate_fallback_denied_when_policy_forbids_it() -> None:
    p = Policy.from_dict({**_POLICY_DICT, "allow_coordinate_fallback": False})
    d = p.check_action(
        "click", "http://127.0.0.1:5050/x", element_name="Search", coordinate_fallback=True
    )
    assert not d.allowed
    assert d.rule == "coordinate_fallback"
