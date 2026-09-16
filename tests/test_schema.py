"""Tests for the Pydantic v2 schema layer (hands.schema.capability/overlay/result/events).

A fixture, `legacycore_capability`, builds a small but complete Capability for
the LegacyCore mock app: login -> member lookup -> member detail, extracting
a savings balance. Individual tests mutate a deep copy of that fixture's
dumped dict rather than the live object, since `Capability.model_copy()` is a
shallow copy and would make hash/approval mutation tests pass vacuously.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

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
    ElementAbsent,
    Extraction,
    FrameMatcher,
    Handling,
    InputSpec,
    Not,
    Outcome,
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
    UrlMatches,
    WaitSpec,
)
from hands.schema.events import Event
from hands.schema.overlay import CapabilityOverlay, OverlayViolation, PatchOp, apply_overlay
from hands.schema.result import (
    CapabilityRef,
    ReplayResult,
    RunStatus,
)


# --------------------------------------------------------------------------
# Fixture: a small but complete LegacyCore capability
# --------------------------------------------------------------------------


def _target(description: str, kind: str, params: dict[str, Any], rationale: str) -> TargetSpec:
    return TargetSpec(
        description=description,
        frame=FrameMatcher(name="work"),
        strategies=[Strategy(kind=kind, params=params, rationale=rationale)],
    )


def _work(pattern: str) -> TextPresent:
    return TextPresent(pattern=pattern, frame=FrameMatcher(name="work"))


def make_legacycore_capability() -> Capability:
    """Login -> member lookup -> member detail, extracting a savings balance."""

    target_username = _target(
        "Username field on the login screen", "role", {"role": "textbox", "name": "Username"},
        "labelled by adjacent bare-<td> text 'Username', stable across theme changes",
    )
    target_password = _target(
        "Password field on the login screen", "role", {"role": "textbox", "name": "Password"},
        "labelled by adjacent bare-<td> text 'Password'",
    )
    target_login_button = _target(
        "Login submit button", "text", {"text": "Go"},
        "the only submit-typed input on the login form",
    )
    target_member_id_field = _target(
        "Member id search field", "role", {"role": "textbox", "name": "Member ID"},
        "only textbox in the lookup form's row",
    )
    target_search_button = _target(
        "Search submit button on the lookup screen", "structural", {"selector": "form#lookup input[type=submit]"},
        "disambiguates from the nav frame's own 'Search' button by frame scope",
    )
    target_balance_cell = _target(
        "Savings row current-balance cell", "relative",
        {"anchor": "Acct Typ", "same_row": "Curr Bal"},
        "balance column is anchored to the row whose Acct Typ cell reads SAV",
    )

    steps = [
        Step(
            id="s1", kind="navigate", screen="login", intent="Open the LegacyCore login page",
            value="http://127.0.0.1:5050/", wait=WaitSpec(), risk=Risk(level="safe"),
        ),
        Step(
            id="s2", kind="type", screen="login", intent="Enter the teller username",
            target=target_username, value="{{username}}", wait=WaitSpec(), risk=Risk(level="safe"),
        ),
        Step(
            id="s3", kind="type", screen="login", intent="Enter the teller password",
            target=target_password, value="{{password}}", wait=WaitSpec(), risk=Risk(level="safe"),
        ),
        Step(
            id="s4", kind="click", screen="login", intent="Submit the login form",
            target=target_login_button,
            wait=WaitSpec(until=_work("Member Lookup"), timeout_ms=10000),
            risk=Risk(
                level="risky",
                reason="submits credentials and establishes an authenticated session",
            ),
        ),
        Step(
            id="s5", kind="type", screen="lookup", intent="Enter the member id to search for",
            target=target_member_id_field, value="{{member_id}}", wait=WaitSpec(), risk=Risk(level="safe"),
        ),
        Step(
            id="s6", kind="click", screen="lookup", intent="Run the member search",
            target=target_search_button,
            wait=WaitSpec(until=_work("Savings"), timeout_ms=10000),
            risk=Risk(level="safe"),
        ),
        Step(
            id="s7", kind="extract", screen="member_detail", intent="Read the member's savings balance",
            wait=WaitSpec(),
            risk=Risk(level="safe"),
            extract=Extraction(
                target=target_balance_cell,
                from_="table_cell",
                table=TableSelector(header_match="Curr Bal", row_where=RowWhere(header="Acct Typ", matches="SAV")),
                parse=ParseSpec(type="money", regex=r"[\d,.]+", group=0),
                output="savings_balance",
            ),
        ),
    ]

    screens = {
        "login": Screen(
            id="login", description="Teller login form", frame=FrameMatcher(name="work"),
            url_pattern="**/login.html", landmarks=[_work("LegacyCore Teller Login")],
        ),
        "lookup": Screen(
            id="lookup", description="Member lookup form", frame=FrameMatcher(name="work"),
            url_pattern="**/lookup.html", landmarks=[_work("Member Lookup")],
        ),
        "member_detail": Screen(
            id="member_detail", description="Member detail with account balances",
            frame=FrameMatcher(name="work"), url_pattern="**/member.html", landmarks=[_work("Savings")],
        ),
    }

    outcomes = [
        Outcome(
            code="MEMBER_NOT_FOUND", kind="business", origin="library", scope="global",
            detect=_work("No member found"),
            message="No member exists with that id.",
            handling=Handling(action="stop"),
        ),
    ]

    contract = Contract(
        inputs={
            "username": InputSpec(type="string", description="Teller login username", example="teller1"),
            "password": InputSpec(
                type="string", description="Teller login password", sensitive=True, source="vault",
            ),
            "member_id": InputSpec(
                type="string", description="Member id to look up", pattern=r"^\d{5}$", example="12345",
            ),
        },
        outputs={
            "savings_balance": OutputSpec(
                type="money", description="The member's current savings balance", source_step="s7",
            ),
        },
        outcome_codes=["MEMBER_NOT_FOUND"],
        side_effects="reads",
        idempotent=True,
    )

    target = TargetApp(
        app=AppInfo(vendor="LegacyCore", product="Teller Console", version="3.2"),
        entry_url="http://127.0.0.1:5050/",
        fingerprint=[_work("LegacyCore")],
    )

    return Capability(
        id="legacycore.lookup_savings_balance",
        name="Look up savings balance",
        description="Logs into LegacyCore, looks up a member by id, and extracts their savings balance.",
        approval=Approval(),
        contract=contract,
        target=target,
        screens=screens,
        steps=steps,
        outcomes=outcomes,
        success=Checkpoint(condition=_work("Savings"), description="Member detail loaded with balance visible"),
        policy=CapabilityPolicy(
            allowed_domains=["127.0.0.1:5050"],
            allowed_actions=["navigate", "click", "type", "extract"],
        ),
        provenance=Provenance(
            source_goal="Look up a member's savings balance by member id",
            discovery_run_id="disc-0001",
            model="claude-opus-5",
            prompt_version="v1",
            recorded_against={"viewport": {"width": 1280, "height": 800}, "browser": "chromium"},
        ),
    )


@pytest.fixture
def cap() -> Capability:
    return make_legacycore_capability()


def _dump(capability: Capability) -> dict[str, Any]:
    return capability.model_dump(mode="json", by_alias=True)


# --------------------------------------------------------------------------
# 1. Round trip
# --------------------------------------------------------------------------


def test_round_trip_json(cap: Capability) -> None:
    restored = Capability.model_validate_json(cap.model_dump_json())
    assert restored == cap


def test_round_trip_by_alias(cap: Capability) -> None:
    # compute_content_hash() and apply_overlay() both dump with by_alias=True
    # (so `not_`/`from_` come out as `not`/`from`); round-trip that mode too.
    restored = Capability.model_validate(cap.model_dump(mode="json", by_alias=True))
    assert restored == cap


# --------------------------------------------------------------------------
# 2. compute_content_hash
# --------------------------------------------------------------------------


def test_hash_stable_across_reserialisation(cap: Capability) -> None:
    h1 = cap.compute_content_hash()
    reloaded = Capability.model_validate_json(cap.model_dump_json())
    assert reloaded.compute_content_hash() == h1


def test_hash_changes_when_step_target_changes(cap: Capability) -> None:
    h1 = cap.compute_content_hash()
    dump = _dump(cap)
    step = next(s for s in dump["steps"] if s["id"] == "s2")
    step["target"]["strategies"][0]["params"] = {"role": "textbox", "name": "Login ID"}
    mutated = Capability.model_validate(dump)
    assert mutated.compute_content_hash() != h1


def test_hash_unaffected_by_approval_or_provenance(cap: Capability) -> None:
    h1 = cap.compute_content_hash()
    dump = _dump(cap)
    dump["approval"] = {
        "status": "approved", "approved_by": "alice",
        "approved_at": "2026-01-01T00:00:00.000+00:00", "approved_hash": "deadbeef",
    }
    dump["provenance"]["discovery_run_id"] = "a-totally-different-run-id"
    dump["provenance"]["recorded_at"] = "2020-01-01T00:00:00.000+00:00"
    mutated = Capability.model_validate(dump)
    assert mutated.compute_content_hash() == h1


# --------------------------------------------------------------------------
# 3. is_effectively_approved
# --------------------------------------------------------------------------


def test_is_effectively_approved_false_after_mutating_approved_capability(cap: Capability) -> None:
    approved_dump = _dump(cap)
    approved_dump["approval"] = {
        "status": "approved", "approved_by": "alice",
        "approved_at": "2026-01-01T00:00:00.000+00:00",
        "approved_hash": cap.compute_content_hash(),
    }
    approved = Capability.model_validate(approved_dump)
    assert approved.is_effectively_approved() is True

    mutated_dump = copy.deepcopy(approved_dump)
    mutated_dump["steps"][1]["target"]["strategies"][0]["params"]["name"] = "changed"
    mutated = Capability.model_validate(mutated_dump)
    assert mutated.is_effectively_approved() is False


# --------------------------------------------------------------------------
# 4. validate_semantics
# --------------------------------------------------------------------------


def test_validate_semantics_clean_capability_has_no_problems(cap: Capability) -> None:
    assert cap.validate_semantics() == []


def test_validate_semantics_catches_unknown_step_screen(cap: Capability) -> None:
    dump = _dump(cap)
    dump["steps"][0]["screen"] = "nonexistent_screen"
    broken = Capability.model_validate(dump)
    problems = broken.validate_semantics()
    assert any("s1" in p and "nonexistent_screen" in p for p in problems)


def test_validate_semantics_catches_undeclared_extract_output(cap: Capability) -> None:
    dump = _dump(cap)
    extract_step = next(s for s in dump["steps"] if s["id"] == "s7")
    extract_step["extract"]["output"] = "totally_undeclared_output"
    broken = Capability.model_validate(dump)
    problems = broken.validate_semantics()
    assert any("totally_undeclared_output" in p for p in problems)


def test_validate_semantics_catches_undeclared_placeholder(cap: Capability) -> None:
    dump = _dump(cap)
    type_step = next(s for s in dump["steps"] if s["id"] == "s2")
    type_step["value"] = "{{typo_username}}"
    broken = Capability.model_validate(dump)
    problems = broken.validate_semantics()
    assert any("typo_username" in p for p in problems)


def test_validate_semantics_catches_outcome_codes_mismatch(cap: Capability) -> None:
    dump = _dump(cap)
    dump["contract"]["outcome_codes"] = ["MEMBER_NOT_FOUND", "NOT_A_REAL_OUTCOME"]
    broken = Capability.model_validate(dump)
    problems = broken.validate_semantics()
    assert any("NOT_A_REAL_OUTCOME" in p for p in problems)


# --------------------------------------------------------------------------
# 5. to_tool_definition
# --------------------------------------------------------------------------


def test_step_kind_validators_reject_missing_required_fields() -> None:
    common = dict(id="x", screen="login", intent="do a thing", wait=WaitSpec(), risk=Risk(level="safe"))
    with pytest.raises(ValidationError):
        Step(kind="click", **common)  # click requires target
    with pytest.raises(ValidationError):
        Step(kind="type", **common)  # type requires target
    with pytest.raises(ValidationError):
        Step(kind="select", **common)  # select requires target
    with pytest.raises(ValidationError):
        Step(kind="press", **common)  # press requires target
    with pytest.raises(ValidationError):
        Step(kind="extract", **common)  # extract requires extract
    with pytest.raises(ValidationError):
        Step(kind="navigate", **common)  # navigate requires value
    with pytest.raises(ValidationError):
        Step(kind="human_task", **common)  # human_task requires human_task


def test_handling_click_requires_target() -> None:
    with pytest.raises(ValidationError):
        Handling(action="click")


def test_strategy_rejects_blank_rationale() -> None:
    with pytest.raises(ValidationError):
        Strategy(kind="role", params={}, rationale="   ")


def test_to_tool_definition(cap: Capability) -> None:
    tool = cap.to_tool_definition()
    input_schema = tool["input_schema"]

    assert set(input_schema["properties"]) == {"username", "password", "member_id"}
    assert set(input_schema["required"]) == {"username", "password", "member_id"}
    assert input_schema["additionalProperties"] is False
    assert input_schema["properties"]["member_id"]["pattern"] == r"^\d{5}$"
    assert "vault" in input_schema["properties"]["password"]["description"]

    assert tool["outputs"]["savings_balance"]["type"] == "string"
    assert tool["outcome_codes"] == ["MEMBER_NOT_FOUND"]
    assert tool["side_effects"] == "reads"
    assert tool["idempotent"] is True


# --------------------------------------------------------------------------
# 6. Condition discriminated union round trip
# --------------------------------------------------------------------------


def test_condition_union_round_trips_nested_all_of_not() -> None:
    adapter = TypeAdapter(Condition)
    cond = AllOf(
        conditions=[
            UrlMatches(pattern=r".*/lookup\.html"),
            Not(not_=TextPresent(pattern="Error")),
        ],
    )
    dumped = adapter.dump_json(cond, by_alias=True)
    restored = adapter.validate_json(dumped)

    assert isinstance(restored, AllOf)
    assert isinstance(restored.conditions[0], UrlMatches)
    assert isinstance(restored.conditions[1], Not)
    assert isinstance(restored.conditions[1].not_, TextPresent)
    assert restored == cond


# --------------------------------------------------------------------------
# 7. Overlay
# --------------------------------------------------------------------------


def _overlay(cap: Capability, description: str, patches: list[PatchOp]) -> CapabilityOverlay:
    return CapabilityOverlay(
        base_id=cap.id, base_version=cap.version, tenant_id="acme",
        description=description, patches=patches,
    )


def test_overlay_legal_target_override_applies(cap: Capability) -> None:
    overlay = _overlay(
        cap, "tenant renamed the username field's aria label",
        [
            PatchOp(
                path="steps[s2].target.strategies", op="replace",
                value=[{"kind": "text", "params": {"text": "Login ID"}, "rationale": "tenant renamed the label"}],
            ),
        ],
    )
    patched = apply_overlay(cap, overlay)
    step = next(s for s in patched.steps if s.id == "s2")
    assert step.target is not None
    assert step.target.strategies[0].params == {"text": "Login ID"}
    # nothing else about the capability changed
    assert patched.id == cap.id
    assert len(patched.steps) == len(cap.steps)


def test_overlay_cannot_lower_risk(cap: Capability) -> None:
    overlay = _overlay(
        cap, "try to relax the login-submit risk level",
        [
            PatchOp(
                path="steps[s4].risk", op="replace",
                value={"level": "safe", "reason": None, "requires_confirmation": False, "idempotency_check": None},
            ),
        ],
    )
    with pytest.raises(OverlayViolation):
        apply_overlay(cap, overlay)


def test_overlay_cannot_widen_allowed_domains(cap: Capability) -> None:
    overlay = _overlay(
        cap, "try to widen the domain allowlist",
        [PatchOp(path="policy.allowed_domains", op="prepend", value="evil.example.com")],
    )
    with pytest.raises(OverlayViolation):
        apply_overlay(cap, overlay)


def test_overlay_cannot_change_step_kind(cap: Capability) -> None:
    overlay = _overlay(
        cap, "try to change what s2 does",
        [PatchOp(path="steps[s2].kind", op="replace", value="click")],
    )
    with pytest.raises(OverlayViolation):
        apply_overlay(cap, overlay)


def test_overlay_prepend_adds_new_landmark_first(cap: Capability) -> None:
    # exercises the spec's own example path and the "prepend" op's success path
    # (the widen-domains test only exercises prepend on a path that then raises)
    new_landmark = {"type": "title_matches", "pattern": "Member Lookup - v2", "frame": None}
    overlay = _overlay(
        cap, "tenant's lookup page also carries a versioned title",
        [PatchOp(path="screens.lookup.landmarks", op="prepend", value=new_landmark)],
    )
    patched = apply_overlay(cap, overlay)
    landmarks = patched.screens["lookup"].landmarks
    assert len(landmarks) == 2
    assert landmarks[0].type == "title_matches"
    assert landmarks[0].pattern == "Member Lookup - v2"
    assert landmarks[1] == cap.screens["lookup"].landmarks[0]


def test_overlay_semantics_delta_catches_risk_raised_without_reason(cap: Capability) -> None:
    # Raising risk (safe -> risky) is allowed by the risk-monotonicity rule in
    # isolation, but it is disallowed here because it leaves the artifact
    # failing "no step with risk.level != safe lacking a reason" -- a problem
    # the base capability did not have. This is what demonstrates the
    # re-validate-with-validate_semantics() delta check actually has teeth,
    # not just the explicit path allowlist / risk-direction check.
    overlay = _overlay(
        cap, "tenant marks the search step riskier but forgets to say why",
        [
            PatchOp(
                path="steps[s6].risk", op="replace",
                value={"level": "risky", "reason": None, "requires_confirmation": False, "idempotency_check": None},
            ),
        ],
    )
    with pytest.raises(OverlayViolation):
        apply_overlay(cap, overlay)


# --------------------------------------------------------------------------
# 8. ReplayResult.exit_code
# --------------------------------------------------------------------------


def test_exit_code_business_outcome_and_failed() -> None:
    ref = CapabilityRef(id="legacycore.lookup_savings_balance", version="1.0.0")
    business = ReplayResult(run_id="run-1", capability_ref=ref, status=RunStatus.BUSINESS_OUTCOME)
    failed = ReplayResult(run_id="run-2", capability_ref=ref, status=RunStatus.FAILED)
    assert business.exit_code() == 0
    assert failed.exit_code() == 1


# --------------------------------------------------------------------------
# events.py sanity (thin envelope, still worth one real assertion)
# --------------------------------------------------------------------------


def test_event_requires_known_kind_and_defaults_data() -> None:
    ev = Event(seq=1, kind="run_started", run_id="run-1")
    assert ev.data == {}
    assert ev.step_id is None
    with pytest.raises(ValidationError):
        Event(seq=2, kind="not_a_real_kind", run_id="run-1")
