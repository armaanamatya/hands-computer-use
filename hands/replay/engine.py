"""The deterministic replay engine: the production execution path.

`ReplayEngine.run(capability, inputs)` re-runs a recorded `Capability`
against a live `Surface` with NO LLM in the loop and returns a structured
`ReplayResult`. Every per-step decision below follows one fixed procedure,
in one fixed order, documented at each phase -- that procedure, not any
particular outcome, is the thing this module exists to get exactly right.

## Per-step decision procedure

For each step, in this exact order:

1. **Pre-action gate** -- may the step act at all, right now?
   (a) dialog check -- an unexpected open dialog gets one chance to be
       explained by a global outcome before failing;
   (b) screen check -- verify the step's declared screen against the
       frame its own matcher selects (skipped for `navigate`: a navigate
       step ESTABLISHES a screen, it does not presuppose one -- see
       `_gate_screen`'s docstring);
   (c) target resolution -- `hands.surface.locator.resolve()`, recording
       every `StrategyAttempt`;
   (d) policy check -- against the resolved element's live name; denied
       outright blocks the run, "risky + needs confirmation + not
       approved" escalates.
2. **Execute** -- bind `{{placeholders}}`, perform the action.
3. **Settle and evaluate** -- poll every 250ms until `step.wait.timeout_ms`,
   each tick checking, in order: (a) step-scoped outcomes, (b) the step's
   checkpoint, (c) global outcomes, (d) otherwise keep polling. See
   `_settle_and_evaluate`'s docstring for exactly why that order is the
   one that keeps a business outcome from ever being masked by a loosely-
   satisfied checkpoint.
4. **Timeout classification**, only reached if nothing above matched by
   the deadline: no declared screen's URL matches -> UNEXPECTED_SCREEN;
   page text matches a generic error/5xx heuristic -> APP_ERROR; a dialog
   is open -> UNEXPECTED_DIALOG; otherwise CHECKPOINT_FAILED.

`step.checkpoint` and `step.wait.until` are two names for the same idea at
different granularity: when a step declares its own `Checkpoint`, that
condition IS what phase 3(b) checks. When it does not, `step.wait.until`
(when it is a real `Condition`, not the literal `"load"`) stands in for it.
This mirrors how `hands/schema/capability.py`'s own worked examples use
`wait.until` for ordinary steps and reserve a separate `Checkpoint` for
the ones worth naming explicitly, and it means `WaitSpec.until` is never
inert: either it or `Step.checkpoint` is always the thing phase 3(b)
checks, never both silently, never neither.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from hands.evidence.run_log import RunLog
from hands.safety.policy import Policy, PolicyViolation
from hands.schema.capability import Capability, Outcome, Screen, Step, UrlMatches
from hands.schema.control import (
    ControlHolder,
    ControlToken,
    InterventionRequest,
    ReasonCode,
    utcnow_iso,
)
from hands.schema.result import (
    CapabilityRef,
    Failure,
    FailureCategory,
    Observed,
    OutcomeRef,
    RecoveryRecord,
    ReplayResult,
    RunStatus,
    StepResult,
)
from hands.surface import conditions
from hands.surface import extract as extract_mod
from hands.surface import locator as locator_mod
from hands.surface.base import Action, ActionResult, Observation

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

# Timeout-classification heuristic (phase 4, 2nd check): a generic
# server-error signature this mock app's error500.html actually renders
# ("Internal Server Error" / "CBS-0500"), plus a couple of common variants
# a real legacy app's error pages tend to share. Deliberately a heuristic,
# not an exhaustive taxonomy -- the brief calls this out as secondary to
# the outcome library, which is the primary, per-app-curated way APP_ERROR
# gets detected (see capabilities/library/legacycore.outcomes.json).
_APP_ERROR_HEURISTIC = re.compile(
    r"(?i)internal server error|unexpected error|http\s*5\d{2}\b|error\s*5\d{2}\b|CBS-0\d{3}"
)


# --------------------------------------------------------------------------
# The escalation seam
# --------------------------------------------------------------------------


@dataclass
class HumanResolution:
    """What an `escalate` handler hands back after a human has looked at an
    `InterventionRequest`.

    This is the whole surface a handoff controller needs to implement to
    plug into replay: `retry_step` ("the human resolved it live in the
    session -- re-check/continue now"), `skip_step` ("treat this step as
    done; a human performed the equivalent action out of band" -- recorded
    as `StepResult.status == "human_performed"`), or `abort` ("stop the
    run" -- `RunStatus.ABANDONED`). Kept intentionally small: this engine
    does not itself manage `ControlState`/`ControlToken` handoff across a
    real live session (that is the handoff controller's job, described in
    `hands/schema/control.py`); it only needs a synchronous answer to
    "what do we do now" to keep the per-step procedure well-defined.
    """

    action: Literal["retry_step", "skip_step", "abort"]
    note: Optional[str] = None


class _Terminate(Exception):
    """Internal control-flow signal: stop the run right now with this
    already-fully-built `ReplayResult`. Raised from deep inside the
    per-step procedure and caught once, at the top of `run()` -- the
    standard, simplest way to unwind an arbitrarily nested decision
    procedure in Python without threading a status code through every
    return statement in between.
    """

    def __init__(self, result: ReplayResult) -> None:
        self.result = result


class _StepDoneByHuman(Exception):
    """Internal control-flow signal: the CURRENT step is done (a human
    resolved it, or confirmed/performed it out of band) -- record it as
    `status == "human_performed"` and move on to the next step. Caught by
    `_run_step`/`_run_human_task_step`, never escapes this module.
    """

    def __init__(self, note: str) -> None:
        self.note = note


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class ReplayEngine:
    """Deterministic replay of one `Capability` against one live `Surface`.

    Construct once per run (or reuse across runs -- `run()` resets all
    per-run state first). `surface` is an already-constructed `Surface`
    (e.g. a `WebSurface` wrapping a live Playwright `Page`); this engine
    does not open browsers or navigate anywhere until `run()`'s Startup
    phase.
    """

    def __init__(
        self,
        surface: Any,
        policy: Policy,
        run_log: RunLog,
        escalate: Optional[Callable[[InterventionRequest], Optional[HumanResolution]]] = None,
    ) -> None:
        self.surface = surface
        self.base_policy = policy
        self.run_log = run_log
        self.escalate_fn = escalate
        self.token = ControlToken(run_id=run_log.run_id, holder=ControlHolder.AUTOMATION, epoch=0)

    # ---------------------------------------------------------------- run()

    def run(
        self,
        capability: Capability,
        inputs: dict[str, Any],
        approved_by: Optional[str] = None,
    ) -> ReplayResult:
        self._reset(capability, inputs, approved_by)
        try:
            self._startup(capability, inputs)
            for index, step in enumerate(capability.steps):
                self._last_screen = capability.screens.get(step.screen, self._last_screen)
                if step.kind == "human_task":
                    self._run_human_task_step(capability, step, index)
                else:
                    self._run_step(capability, step, index)
            result = self._finish(capability)
        except _Terminate as terminated:
            result = terminated.result
        self.run_log.write_json("result.json", result.model_dump(mode="json"))
        if self.recovery_history:
            self.run_log.write_json(
                "recovery_history.json", [r.model_dump(mode="json") for r in self.recovery_history]
            )
        return result

    def _reset(self, capability: Capability, inputs: dict[str, Any], approved_by: Optional[str]) -> None:
        self.capability = capability
        self.inputs = inputs
        self.approved_by = approved_by
        self.policy: Policy = self.base_policy
        self.outputs: dict[str, Any] = {}
        self.drift_signals: list[str] = []
        self.human_interventions: list[str] = []
        self.recovery_history: list[RecoveryRecord] = []
        self._recovery_attempts: dict[str, int] = {}
        self._recovery_total = 0
        self.side_effects_uncertain = False
        self.step_results: list[StepResult] = []
        self._intervention_seq = 0
        self._last_screen: Optional[Screen] = None
        self._pending_step_note: Optional[str] = None

    # --------------------------------------------------------------- startup

    def _startup(self, capability: Capability, inputs: dict[str, Any]) -> None:
        # 1. Register sensitive input values with the Redactor BEFORE
        # anything is logged -- including the input-validation failure
        # message below, which could otherwise echo a bad password value
        # straight into events.jsonl.
        for name, spec in capability.contract.inputs.items():
            value = inputs.get(name)
            if spec.sensitive and isinstance(value, str):
                self.run_log.redactor.add(name, value)

        self.run_log.event(
            "run_started", capability_id=capability.id, capability_version=capability.version
        )

        # 2. Validate inputs against the contract. A missing/malformed
        # argument is a caller mistake (INPUT_INVALID), not a safety
        # refusal (POLICY_BLOCKED) and not a runtime condition (FAILED
        # with some other category) -- it never touches the surface at all.
        problems = self._validate_inputs(capability, inputs)
        if problems:
            raise _Terminate(
                self._result(
                    RunStatus.FAILED,
                    failure=Failure(
                        category=FailureCategory.INPUT_INVALID,
                        expected="inputs satisfying contract.inputs",
                        suggested_action="; ".join(problems),
                    ),
                    outputs_complete=False,
                )
            )

        # 3. Effective policy = operator policy.yaml INTERSECT this
        # capability's own (narrower-only) policy block.
        self.policy = self.base_policy.intersect(capability.policy.model_dump())

        # 4. Dialog mode -> "hold". Replay must never let a native dialog
        # auto-resolve: every dialog is either explicitly expected
        # (step.expect_dialog) or handled by outcome recovery.
        self.surface.set_dialog_mode("hold")

        # 5. Navigate to entry_url, THEN check the fingerprint. Navigating
        # first means a fingerprint mismatch reflects a genuine "wrong
        # app/version" signal, not an artifact of the browser's blank
        # initial page. Capabilities conventionally still declare their
        # own explicit first `navigate` step (mirroring what the discovery
        # transcript actually did); that step is then a harmless, idempotent
        # re-navigation to the same URL.
        try:
            self.policy.assert_action("navigate", capability.target.entry_url)
        except PolicyViolation as e:
            raise _Terminate(
                self._result(
                    RunStatus.POLICY_BLOCKED,
                    failure=Failure(category=FailureCategory.POLICY_VIOLATION, suggested_action=e.detail),
                )
            )
        self.surface.act(Action(kind="navigate", url=capability.target.entry_url), self.token)

        if capability.target.fingerprint:
            obs = self._observe(label="startup-fingerprint")
            ctx = conditions.ConditionContext(
                self.surface, obs, self.outputs, default_frame=None, policy=self.policy
            )
            ok, explanations = conditions.evaluate_all(capability.target.fingerprint, ctx)
            self.run_log.event("policy_check", note="fingerprint check", ok=ok, detail="; ".join(explanations))
            if not ok:
                # Does NOT abort -- just a drift signal for the result.
                self.drift_signals.append(
                    "target fingerprint mismatch at startup: " + "; ".join(explanations)
                )

    def _validate_inputs(self, capability: Capability, inputs: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        for name, spec in capability.contract.inputs.items():
            if name not in inputs or inputs[name] is None:
                if spec.required:
                    problems.append(f"missing required input {name!r}")
                continue
            value = inputs[name]
            if spec.type == "string":
                if not isinstance(value, str):
                    problems.append(f"input {name!r} must be a string")
                    continue
            elif spec.type == "integer":
                if isinstance(value, bool) or not isinstance(value, (int, str)):
                    problems.append(f"input {name!r} must be an integer")
                    continue
                if isinstance(value, str) and not re.fullmatch(r"-?\d+", value):
                    problems.append(f"input {name!r} must be an integer")
                    continue
            elif spec.type == "enum":
                if spec.enum is not None and str(value) not in spec.enum:
                    problems.append(f"input {name!r}={value!r} is not one of {spec.enum!r}")
                    continue
            if spec.pattern and isinstance(value, str) and not re.fullmatch(spec.pattern, value):
                problems.append(f"input {name!r}={value!r} does not match pattern {spec.pattern!r}")
        return problems

    def _is_confirmed(self, capability: Capability) -> bool:
        return capability.is_effectively_approved() or self.approved_by is not None

    # ----------------------------------------------------------------- steps

    def _run_step(self, capability: Capability, step: Step, index: int) -> None:
        screen = capability.screens[step.screen]
        resolve_result: Optional[locator_mod.ResolveResult] = None
        try:
            # ---- Phase 1: pre-action gate ----
            self._gate_dialog(capability, step, screen, index)
            if step.kind != "navigate":
                # A navigate step ESTABLISHES a screen; it has no screen
                # precondition to verify. Every other kind's `screen` field
                # is the screen it expects to already be standing on.
                self._gate_screen(capability, step, screen, index)
            if step.target is not None:
                resolve_result = self._gate_resolve(step, index)
            self._gate_policy(capability, step, index, resolve_result)

            # ---- Phase 2: execute ----
            self._execute_step(step, resolve_result)

            # ---- Phase 3: settle & evaluate ----
            status, note, outcomes_matched = self._settle_and_evaluate(
                capability, step, screen, index, resolve_result
            )
        except _StepDoneByHuman as done:
            self.step_results.append(
                StepResult(step_id=step.id, index=index, intent=step.intent, status="human_performed", note=done.note)
            )
            return

        self.step_results.append(
            StepResult(
                step_id=step.id,
                index=index,
                intent=step.intent,
                status=status,
                strategy_used=(resolve_result.strategy_used if resolve_result else None),
                strategy_rank=(resolve_result.strategy_rank if resolve_result else None),
                attempts=(len(resolve_result.attempts) if resolve_result else 1),
                outcomes_matched=outcomes_matched,
                note=note,
            )
        )

    def _run_human_task_step(self, capability: Capability, step: Step, index: int) -> None:
        # human_task steps are NEVER performed by automation: they always
        # escalate, and completion_check is what verifies a human actually
        # did the thing.
        assert step.human_task is not None
        obs = self._observe(label=f"{step.id}-human-task")
        resolution = self._escalate(
            "REQUIRES_HUMAN_STEP",
            step,
            index,
            "human_task",
            obs,
            expected=step.human_task.instructions,
            suggested_actions=[step.human_task.instructions],
        )
        screen = capability.screens[step.screen]
        if resolution.action == "retry_step":
            # "Retry" has no automation-side action to repeat for a
            # human_task; interpreted as "the human says it's done now,
            # please re-check" -- verify via completion_check before
            # trusting it.
            obs2 = self._observe(label=f"{step.id}-human-task-check")
            ctx2 = conditions.ConditionContext(
                self.surface, obs2, self.outputs, default_frame=screen.frame, policy=self.policy, step_id=step.id
            )
            ok, why = conditions.evaluate(step.human_task.completion_check, ctx2)
            if not ok:
                self._fail(
                    FailureCategory.CHECKPOINT_FAILED, step, index, obs2,
                    expected=step.human_task.instructions,
                    suggested_action=f"human_task.completion_check not satisfied: {why}",
                    action_performed=False,
                )
            self.step_results.append(
                StepResult(step_id=step.id, index=index, intent=step.intent, status="human_performed",
                           note="verified via completion_check")
            )
            return
        # skip_step: trust the human outright, no automated verification.
        self.step_results.append(
            StepResult(step_id=step.id, index=index, intent=step.intent, status="human_performed",
                       note=resolution.note or "performed by human")
        )

    # ------------------------------------------------------- phase 1: gates

    def _gate_dialog(self, capability: Capability, step: Step, screen: Screen, index: int) -> None:
        """1(a). If a dialog is open and this step does not `expect_dialog`,
        give the outcome library one chance to explain it (a known
        interstitial, e.g. UNEXPECTED_CONFIRM) before failing outright.
        Recovering loops back to re-check -- bounded by the same recovery
        budget `_perform_recovery_action` enforces on every call.
        """
        while True:
            dlg = self.surface.dialog_state()
            if dlg is None or step.expect_dialog is not None:
                return
            obs = self._observe(label=f"{step.id}-dialog")
            ctx = conditions.ConditionContext(
                self.surface, obs, self.outputs, default_frame=screen.frame, policy=self.policy, step_id=step.id
            )
            global_outcomes = [o for o in capability.outcomes if o.scope == "global"]
            outcome = self._match_outcome(global_outcomes, ctx)
            if outcome is None:
                self._fail(
                    FailureCategory.UNEXPECTED_DIALOG, step, index, obs,
                    expected="no unexpected dialog before this step's action",
                    suggested_action=f"open dialog {dlg.type}: {dlg.message!r}; no outcome explains it",
                    action_performed=False,
                )
            self._handle_matched_outcome(step, screen, index, outcome, ctx, phase="pre_action")
            # Loop back and re-check for a dialog.

    def _gate_screen(self, capability: Capability, step: Step, screen: Screen, index: int) -> None:
        """1(b). Verify THIS step's declared screen against the frame its
        own `FrameMatcher` selects. On mismatch, look for whichever OTHER
        declared screen DOES match (screen_guess) before failing.

        This is a precondition re-check, not the thing that catches a
        business/interstitial outcome -- that already happened (or should
        have) inside the PREVIOUS step's own settle-and-evaluate phase. The
        two together form the interlock: a step's own phase 3 verifies
        where ITS action landed; the NEXT step's phase 1(b) independently
        re-verifies it before acting again. That second check is what
        catches a checkpoint that merely matched leftover page text --
        exactly the case the wrong-screen test exists to prove is real.
        """
        obs = self._observe(label=f"{step.id}-pre")
        ok, explanations = self._screen_matches(screen, obs)
        if ok:
            return
        guess = self._guess_screen(capability, obs, exclude_id=step.screen)
        self._fail(
            FailureCategory.UNEXPECTED_SCREEN, step, index, obs,
            expected=f"screen {step.screen!r}: {screen.description}",
            screen_guess=guess,
            suggested_action="; ".join(explanations),
            action_performed=False,
        )

    def _gate_resolve(self, step: Step, index: int) -> locator_mod.ResolveResult:
        """1(c). Resolve `step.target`, recording every `StrategyAttempt`.
        Zero matches on every strategy -> TARGET_NOT_FOUND; every strategy
        matching more than one element (never zero) -> TARGET_AMBIGUOUS.
        """
        assert step.target is not None
        try:
            result = locator_mod.resolve(
                self.surface, step.target, self.policy, locator_mod.ResolveContext(step_id=step.id)
            )
        except locator_mod.TargetResolutionError as e:
            obs = self._observe(label=f"{step.id}-resolve-failed")
            category = (
                FailureCategory.TARGET_AMBIGUOUS if e.category == "ambiguous"
                else FailureCategory.TARGET_NOT_FOUND
            )
            detail = "; ".join(
                f"{a.strategy}: {a.error}" if a.error else f"{a.strategy}: {a.match_count} matches"
                for a in e.attempts
            )
            self._fail(
                category, step, index, obs, expected=step.target.description,
                suggested_action=detail, target_resolution=e.attempts, action_performed=False,
            )
            raise AssertionError("unreachable")  # _fail always raises

        self.run_log.event(
            "policy_check", step_id=step.id, note="target resolved",
            strategy=result.strategy_used, strategy_rank=result.strategy_rank,
            coordinate_fallback=result.coordinate_fallback,
        )
        if result.strategy_rank > 0:
            self.drift_signals.append(
                f"step {step.id}: resolved via strategy rank {result.strategy_rank} "
                f"({result.strategy_used}) rather than the top-ranked strategy"
            )
        if result.coordinate_fallback:
            self.drift_signals.append(f"step {step.id}: resolved via coordinate (bbox) fallback")
        return result

    def _gate_policy(
        self, capability: Capability, step: Step, index: int, resolve_result: Optional[locator_mod.ResolveResult]
    ) -> None:
        """1(d). Policy check against the resolved element's live name.
        Denied outright -> POLICY_BLOCKED. Risky + requires_confirmation
        (per the LIVE policy's own classification of the actual resolved
        element -- not the recorded `step.risk`, which is a separate,
        advisory, discovery-time annotation used for idempotency/semantics
        elsewhere) + not effectively approved + no `approved_by` -> escalate.
        """
        if step.kind == "navigate":
            url = _bind_placeholders(step.value or "", self.inputs)
            action_kind, element_name, coordinate_fallback = "navigate", None, False
        else:
            url = self.surface.url()
            action_kind = step.kind
            element_name = resolve_result.element_name if resolve_result else None
            coordinate_fallback = bool(resolve_result and resolve_result.coordinate_fallback)

        decision = self.policy.check_action(
            action_kind, url, element_name=element_name, coordinate_fallback=coordinate_fallback
        )
        self.run_log.event(
            "policy_check", step_id=step.id, rule=decision.rule, allowed=decision.allowed,
            requires_confirmation=decision.requires_confirmation, detail=decision.detail,
        )

        if not decision.allowed:
            obs = self._observe(label=f"{step.id}-policy-blocked")
            raise _Terminate(
                self._result(
                    RunStatus.POLICY_BLOCKED,
                    failure=Failure(
                        category=FailureCategory.POLICY_VIOLATION,
                        step={"id": step.id, "index": index, "intent": step.intent, "kind": step.kind},
                        expected="an action permitted by the effective policy",
                        observed=self._observed_from(obs),
                        evidence=self._capture_evidence(f"{step.id}-policy-blocked", obs),
                        suggested_action=decision.detail,
                    ),
                )
            )

        if decision.requires_confirmation and not self._is_confirmed(capability):
            obs = self._observe(label=f"{step.id}-risky-confirm")
            resolution = self._escalate(
                "RISKY_CONFIRMATION", step, index, "pre_action", obs,
                expected="human confirmation of a risky action before it is performed",
                suggested_actions=[f"confirm or deny: {decision.detail}"],
            )
            if resolution.action == "skip_step":
                raise _StepDoneByHuman(f"risky action skipped by human confirmation: {decision.detail}")
            # retry_step: the human confirmed it; fall through to execute.

    # ---------------------------------------------------------- phase 2/3

    def _execute_step(self, step: Step, resolve_result: Optional[locator_mod.ResolveResult]) -> None:
        value = _bind_placeholders(step.value, self.inputs) if step.value is not None else None

        if step.kind == "navigate":
            result = self.surface.act(Action(kind="navigate", url=value), self.token)
        elif resolve_result is not None and resolve_result.coordinate_fallback:
            # bbox fallback: there is no Locator, only a page-coordinate
            # point. Per the Surface protocol, `click_at()` takes no
            # ControlToken -- policy gating for this path already happened
            # explicitly in phase 1(d) via check_action(coordinate_
            # fallback=True); control-token gating simply is not available
            # for this primitive given the Surface protocol as specified.
            assert resolve_result.bbox_point is not None
            self.surface.click_at(*resolve_result.bbox_point)
            result = ActionResult(ok=True, url_after=self.surface.url())
        elif step.kind in ("click", "type", "select", "press"):
            loc = resolve_result.locator if resolve_result else None
            action = Action(
                kind=step.kind, value=value, key=(value if step.kind == "press" else None), locator=loc
            )
            result = self.surface.act(action, self.token)
        else:
            # extract / assert / wait: nothing to click/type -- these kinds
            # exist purely to be checked in phase 3, not acted on in phase 2.
            self.run_log.event("action", step_id=step.id, action_kind=step.kind, ok=True, note="no-op action phase")
            return

        self.run_log.event(
            "action", step_id=step.id, action_kind=step.kind, ok=result.ok, error=result.error,
            duration_ms=result.duration_ms,
        )

        if step.expect_dialog is not None:
            dlg = self.surface.dialog_state()
            if dlg is not None:
                self.surface.answer_dialog(accept=step.expect_dialog.accept)
                self.run_log.event(
                    "action", step_id=step.id, note="answered expected dialog", accepted=step.expect_dialog.accept
                )

    def _settle_and_evaluate(
        self, capability: Capability, step: Step, screen: Screen, index: int,
        resolve_result: Optional[locator_mod.ResolveResult],
    ) -> tuple[str, Optional[str], list[str]]:
        """Poll every 250ms until `step.wait.timeout_ms`, checking, EACH
        POLL, in this exact order:
          (a) step-scoped outcomes (outcomes whose `scope` names this step)
              -- most specific wins;
          (b) the step's checkpoint (or `wait.until` when no Checkpoint is
              declared) -- if it passes, run any extraction and move on;
          (c) global outcomes;
          (d) otherwise keep polling.

        WHY this order: a step-scoped business outcome must beat a loosely-
        satisfied checkpoint, so "account restricted" is never masked by a
        checkpoint that merely matched leftover page text (a stale landmark
        heading, a generic label present on both the success page and an
        error page). Checking outcomes strictly BEFORE the checkpoint on
        every single tick -- not just once, not just on timeout -- is what
        makes that guarantee hold at every point in the poll, not only at
        the end of it.
        """
        step_scoped = [o for o in capability.outcomes if isinstance(o.scope, list) and step.id in o.scope]
        global_outcomes = [o for o in capability.outcomes if o.scope == "global"]
        checkpoint_condition = (
            step.checkpoint.condition if step.checkpoint
            else (step.wait.until if step.wait.until != "load" else None)
        )
        deadline = time.monotonic() + step.wait.timeout_ms / 1000.0
        had_recovery = False
        outcomes_matched: list[str] = []

        while True:
            obs = self._observe(label=f"{step.id}-poll")
            ctx = conditions.ConditionContext(
                self.surface, obs, self.outputs, default_frame=screen.frame, policy=self.policy, step_id=step.id
            )

            outcome = self._match_outcome(step_scoped, ctx)  # (a)
            if outcome is not None:
                outcomes_matched.append(outcome.code)
                signal = self._handle_matched_outcome(step, screen, index, outcome, ctx, phase="settle")
                had_recovery = True
                if signal == "reexec":
                    self._execute_step(step, resolve_result)
                continue

            if checkpoint_condition is None:  # (b), vacuous: nothing more specific to verify
                break
            ok, _why = conditions.evaluate(checkpoint_condition, ctx)
            if ok:
                break

            outcome = self._match_outcome(global_outcomes, ctx)  # (c)
            if outcome is not None:
                outcomes_matched.append(outcome.code)
                signal = self._handle_matched_outcome(step, screen, index, outcome, ctx, phase="settle")
                had_recovery = True
                if signal == "reexec":
                    self._execute_step(step, resolve_result)
                continue

            if time.monotonic() >= deadline:  # (d) -> timeout classification
                self._timeout_failure(capability, step, index, obs)
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

        if step.extract is not None:
            obs_final = self._observe(label=f"{step.id}-extract")
            ctx_final = conditions.ConditionContext(
                self.surface, obs_final, self.outputs, default_frame=screen.frame, policy=self.policy, step_id=step.id
            )
            value, err = extract_mod.run_extraction(self.surface, step.extract, ctx_final)
            if err is not None:
                self._fail(
                    FailureCategory.EXTRACTION_FAILED, step, index, obs_final,
                    expected=f"extraction for output {step.extract.output!r}", suggested_action=err,
                )
            self.outputs[step.extract.output] = value
            self.run_log.event("checkpoint", step_id=step.id, note="extraction ok", output=step.extract.output)

        note = self._pending_step_note
        self._pending_step_note = None
        return ("recovered" if had_recovery else "ok"), note, outcomes_matched

    # -------------------------------------------------------------- outcomes

    def _match_outcome(self, outcomes: list[Outcome], ctx: conditions.ConditionContext) -> Optional[Outcome]:
        for outcome in outcomes:
            ok, _why = conditions.evaluate(outcome.detect, ctx)
            if ok:
                return outcome
        return None

    def _handle_matched_outcome(
        self, step: Step, screen: Screen, index: int, outcome: Outcome,
        ctx: conditions.ConditionContext, phase: Literal["pre_action", "settle"],
    ) -> str:
        """Branch on `outcome.kind`, then on `handling.action`. Returns a
        signal used only by the phase-3 poll loop ("reexec" | "continue")
        -- terminal cases (business/app_error/requires_human/stop/escalate-
        no-resolution/abort) raise `_Terminate` (or `_StepDoneByHuman`)
        directly and never return at all.
        """
        self.run_log.event(
            "outcome_detected", step_id=step.id, code=outcome.code, outcome_kind=outcome.kind, scope=outcome.scope
        )

        if outcome.kind == "business":
            # A normal return, not a failure -- run result_outputs so the
            # caller still gets data (e.g. the restriction reason text).
            self._apply_result_outputs(outcome, ctx)
            raise _Terminate(
                self._result(
                    RunStatus.BUSINESS_OUTCOME,
                    outcome=OutcomeRef(code=outcome.code, kind=outcome.kind, detected_at_step=step.id, message=outcome.message),
                    # A business outcome terminates before every declared
                    # contract.output is necessarily reached; the Finish
                    # phase's completeness check only runs on the success
                    # path (see `_finish`), so this is descriptive, not a
                    # failure signal in its own right.
                    outputs_complete=False,
                )
            )

        if outcome.kind == "app_error":
            self._fail(
                FailureCategory.APP_ERROR, step, index, ctx.observation,
                expected="no application error", suggested_action=outcome.message,
            )

        if outcome.kind == "requires_human" or outcome.handling.action == "escalate":
            resolution = self._escalate(
                "STUCK_NO_PROGRESS", step, index, phase, ctx.observation,
                expected=f"resolution of outcome {outcome.code}", suggested_actions=[outcome.message],
            )
            if resolution.action == "skip_step":
                raise _StepDoneByHuman(f"outcome {outcome.code} resolved by human: {resolution.note or outcome.message}")
            return "continue"  # retry_step: human resolved it live; re-check now.

        if outcome.handling.action == "stop":
            self._fail(
                FailureCategory.CHECKPOINT_FAILED, step, index, ctx.observation,
                expected="a recoverable condition",
                suggested_action=f"outcome {outcome.code} handling is 'stop': {outcome.message}",
            )

        # transient / interstitial / auth, handling.action in
        # (dismiss_dialog, reload, click): perform the recovery primitive.
        self._perform_recovery_action(step, index, outcome, ctx)

        if phase == "pre_action":
            # Nothing has executed yet at this point (we're still in phase
            # 1(a)); there is no "step action" to decide whether to redo.
            # The caller (`_gate_dialog`) just loops back and re-checks.
            return "continue"

        return self._decide_reexec(step, screen, outcome)

    def _perform_recovery_action(self, step: Step, index: int, outcome: Outcome, ctx: conditions.ConditionContext) -> None:
        handling = outcome.handling
        attempt_n = self._recovery_attempts.get(outcome.code, 0) + 1
        if self._recovery_total >= self.policy.recovery_budget or attempt_n > handling.max_attempts:
            self._fail(
                FailureCategory.RECOVERY_EXHAUSTED, step, index, ctx.observation,
                expected="a successful recovery within budget",
                suggested_action=(
                    f"outcome {outcome.code}: exhausted recovery budget "
                    f"(attempt {attempt_n}, run total {self._recovery_total})"
                ),
            )

        self._recovery_attempts[outcome.code] = attempt_n
        self._recovery_total += 1

        try:
            if handling.action == "dismiss_dialog":
                self.surface.answer_dialog(accept=False)
                result_desc = "dismissed the unexpected dialog"
            elif handling.action == "reload":
                self.surface.act(Action(kind="navigate", url=self.surface.url()), self.token)
                result_desc = "reloaded the current page"
            elif handling.action == "click":
                assert handling.target is not None
                rr = locator_mod.resolve(
                    self.surface, handling.target, self.policy, locator_mod.ResolveContext(step_id=step.id)
                )
                if rr.coordinate_fallback:
                    assert rr.bbox_point is not None
                    self.surface.click_at(*rr.bbox_point)
                else:
                    self.surface.act(Action(kind="click", locator=rr.locator), self.token)
                result_desc = f"clicked recovery target (strategy {rr.strategy_used}, rank {rr.strategy_rank})"
            else:
                result_desc = f"no recovery primitive for handling.action={handling.action!r}"
        except Exception as e:  # noqa: BLE001 - a failed recovery attempt still counts against budget
            result_desc = f"recovery action failed: {e.__class__.__name__}: {e}"

        record = RecoveryRecord(outcome_code=outcome.code, attempt=attempt_n, action=handling.action, result=result_desc)
        self.recovery_history.append(record)
        self.run_log.event(
            "recovery", step_id=step.id, outcome_code=outcome.code, attempt=attempt_n,
            action=handling.action, result=result_desc,
        )

    def _decide_reexec(self, step: Step, screen: Screen, outcome: Outcome) -> str:
        """After a recovery action has run, decide whether the STEP's own
        original action needs to be redone.

        `click` (and any future recovery primitive that targets its OWN
        designated control, e.g. a "Retry" button) IS the retry: it already
        ran through `Surface.act()`'s own settle, and re-performing the
        step's original action on top of that would act on whatever the
        step originally targeted -- which may no longer even be on the page
        (e.g. a lookup form's Search button, after the recovery already
        navigated on to the result page). So for `click`, simply resume
        polling; the checkpoint gets re-checked on the next tick.

        `reload`/`dismiss_dialog` clear an obstacle WITHOUT resubmitting the
        step's own action, so the step's effect (if any) may or may not
        have happened yet. Before blindly repeating a possibly-risky
        action, check `step.risk.idempotency_check` (falling back to the
        step's own checkpoint): if it is already satisfied, the effect was
        already applied -- do not re-execute, just accept this poll pass.
        If it is not satisfied and the step is risky with no idempotency
        signal at all, do not guess -- escalate instead of silently
        double-submitting a risky action.
        """
        handling = outcome.handling
        if handling.action not in ("reload", "dismiss_dialog"):
            return "continue"

        idem_condition = step.risk.idempotency_check or (step.checkpoint.condition if step.checkpoint else None)
        if idem_condition is not None:
            obs2 = self._observe(label=f"{step.id}-idempotency")
            ctx2 = conditions.ConditionContext(
                self.surface, obs2, self.outputs, default_frame=screen.frame, policy=self.policy, step_id=step.id
            )
            ok, why = conditions.evaluate(idem_condition, ctx2)
            self.run_log.event("recovery", step_id=step.id, note="idempotency_check", result=why, satisfied=ok)
            if ok:
                self._pending_step_note = (
                    f"outcome {outcome.code} recovery cleared the obstacle; idempotency_check "
                    "confirmed the step's effect was already applied, so it was not re-executed"
                )
                return "continue"  # already applied; do not re-execute
            if step.risk.level != "safe":
                self._escalate_risky_retry(step, ctx2.observation, outcome)
                return "continue"
            return "reexec"

        if step.risk.level != "safe":
            self._escalate_risky_retry(step, None, outcome)
            return "continue"
        return "reexec"

    def _escalate_risky_retry(self, step: Step, obs: Optional[Observation], outcome: Outcome) -> None:
        if obs is None:
            obs = self._observe(label=f"{step.id}-recovery-confirm")
        resolution = self._escalate(
            "RISKY_CONFIRMATION", step, None, "recovery", obs,
            expected="a human to confirm it is safe to retry this risky step after recovering",
            suggested_actions=[
                f"step {step.id} is risky and recovered from outcome {outcome.code} with no "
                "idempotency signal confirming its effect; confirm before it is retried"
            ],
        )
        if resolution.action == "skip_step":
            raise _StepDoneByHuman("risky step's post-recovery state confirmed by human; not re-executed")
        # retry_step: human authorized it; caller resumes polling either way
        # (a human confirming here means "go ahead and let it resolve",
        # not "perform an extra automated action").

    def _apply_result_outputs(self, outcome: Outcome, ctx: conditions.ConditionContext) -> None:
        for name, extraction in outcome.result_outputs.items():
            value, err = extract_mod.run_extraction(self.surface, extraction, ctx)
            if err is None:
                self.outputs[name] = value
            else:
                self.run_log.event("checkpoint", note=f"result_outputs extraction {name!r} failed: {err}")

    # --------------------------------------------------------------- timeout

    def _timeout_failure(self, capability: Capability, step: Step, index: int, obs: Observation) -> None:
        category = self._classify_timeout(capability, obs)
        guess = self._guess_screen(capability, obs, exclude_id=step.screen) if category == FailureCategory.UNEXPECTED_SCREEN else None
        self._fail(
            category, step, index, obs,
            expected=(step.checkpoint.description if step.checkpoint else step.intent),
            screen_guess=guess, suggested_action="checkpoint/outcome polling timed out",
        )

    def _classify_timeout(self, capability: Capability, obs: Observation) -> FailureCategory:
        # Order matches the brief precisely. This order is safe even with a
        # dialog open: `_observe()` returns EMPTY frames/frame_texts in
        # that case (see its docstring), so the URL/text checks below find
        # nothing and fall through harmlessly to the dialog check.
        any_screen_matches = any(self._screen_matches(screen, obs)[0] for screen in capability.screens.values())
        if not any_screen_matches:
            return FailureCategory.UNEXPECTED_SCREEN
        if any(_APP_ERROR_HEURISTIC.search(t or "") for t in obs.frame_texts.values()):
            return FailureCategory.APP_ERROR
        if obs.dialog is not None:
            return FailureCategory.UNEXPECTED_DIALOG
        return FailureCategory.CHECKPOINT_FAILED

    # ----------------------------------------------------------------- screens

    def _screen_matches(self, screen: Screen, obs: Observation) -> tuple[bool, list[str]]:
        ctx = conditions.ConditionContext(
            self.surface, obs, self.outputs, default_frame=screen.frame, policy=self.policy
        )
        explanations: list[str] = []
        ok = True
        if screen.url_pattern:
            # frame=None: ctx.default_frame is screen.frame, and a condition
            # with no frame of its own falls back to ctx.default_frame (see
            # conditions._resolve_frame_info's precedence).
            u_ok, why = conditions.evaluate(UrlMatches(pattern=screen.url_pattern, frame=None), ctx)
            explanations.append(why)
            ok = ok and u_ok
        if screen.landmarks:
            l_ok, land_why = conditions.evaluate_all(screen.landmarks, ctx)
            explanations.extend(land_why)
            ok = ok and l_ok
        return ok, explanations

    def _guess_screen(self, capability: Capability, obs: Observation, exclude_id: Optional[str]) -> Optional[str]:
        for sid, screen in capability.screens.items():
            if sid == exclude_id:
                continue
            ok, _explanations = self._screen_matches(screen, obs)
            if ok:
                return sid
        return None

    # ------------------------------------------------------------- escalation

    def _escalate(
        self, reason_code: ReasonCode, step: Optional[Step], index: Optional[int], phase: str,
        obs: Observation, expected: Optional[str], suggested_actions: list[str],
    ) -> HumanResolution:
        self._intervention_seq += 1
        req_id = f"{self.run_log.run_id}-iv{self._intervention_seq}"
        evidence = self._capture_evidence(f"intervention-{req_id}", obs)
        observed = {
            "url": obs.url,
            "title": obs.title,
            "text_excerpt": self._text_excerpt(obs),
            "open_dialog": (f"{obs.dialog.type}: {obs.dialog.message}" if obs.dialog else None),
        }
        request = InterventionRequest(
            id=req_id,
            run_id=self.run_log.run_id,
            mode="replay",
            capability_ref={
                "id": self.capability.id, "version": self.capability.version,
                "content_hash": self.capability.compute_content_hash(),
            },
            reason_code=reason_code,
            summary=expected or f"automation cannot safely proceed at step {step.id if step else '?'}",
            step=({"id": step.id, "index": index, "intent": step.intent, "phase": phase} if step else None),
            expected=expected,
            observed=observed,
            evidence=evidence,
            suggested_actions=suggested_actions,
        )
        self.human_interventions.append(request.id)
        self.run_log.event(
            "intervention_raised", step_id=(step.id if step else None), reason_code=reason_code, intervention_id=request.id
        )

        if self.escalate_fn is None:
            self.run_log.write_json(f"intervention-{request.id}.json", request.model_dump(mode="json"))
            raise _Terminate(self._result(RunStatus.NEEDS_HUMAN, outputs_complete=False))

        resolution = self.escalate_fn(request)
        if resolution is None:
            self.run_log.write_json(f"intervention-{request.id}.json", request.model_dump(mode="json"))
            raise _Terminate(self._result(RunStatus.NEEDS_HUMAN, outputs_complete=False))

        self.run_log.event(
            "control_transfer", step_id=(step.id if step else None), intervention_id=request.id,
            resolution=resolution.action, note=resolution.note,
        )
        if resolution.action == "abort":
            raise _Terminate(self._result(RunStatus.ABANDONED, outputs_complete=False))
        return resolution

    # ------------------------------------------------------------ evidence

    def _observe(self, label: Optional[str] = None) -> Observation:
        # A held dialog blocks the page's JS thread; do not call
        # surface.observe() in that state (it would either come back with
        # everything empty anyway, per hands/surface/web.py's own dialog
        # handling, or risk a hang on a call that needs the page's JS
        # context). dialog_state()+url() are both implemented without
        # evaluate(), so they're the only two calls made here.
        dlg = self.surface.dialog_state()
        if dlg is not None:
            return Observation(
                url=self.surface.url(), title="", frames=[], elements=[], frame_texts={},
                frame_titles={}, dialog=dlg, screenshot_path=None, ts=utcnow_iso(),
            )
        shot = self.run_log.next_screenshot(label) if label else None
        return self.surface.observe(screenshot_path=shot)

    def _text_excerpt(self, obs: Observation) -> str:
        return " | ".join(v for v in obs.frame_texts.values() if v)[:500]

    def _observed_from(self, obs: Observation, screen_guess: Optional[str] = None) -> Observed:
        return Observed(
            url=obs.url, title=obs.title, screen_guess=screen_guess, matched_conditions=[],
            text_excerpt=self._text_excerpt(obs),
            open_dialog=(f"{obs.dialog.type}: {obs.dialog.message}" if obs.dialog else None),
        )

    def _capture_evidence(self, label: str, obs: Optional[Observation]) -> dict[str, str]:
        evidence: dict[str, str] = {}
        shot_path = self.run_log.next_screenshot(label)
        if self.surface.screenshot(shot_path):
            evidence["screenshot"] = self.run_log.rel(shot_path) or shot_path
        if obs is not None and obs.dialog is None:
            html_path, elements_path = self.run_log.dom_paths(label)
            if self.surface.snapshot_dom(html_path):
                evidence["dom"] = self.run_log.rel(html_path) or html_path
            try:
                data = [e.to_dict() for e in obs.elements]
                Path(elements_path).write_text(
                    json.dumps(self.run_log.redactor.obj(data), indent=2, ensure_ascii=False), encoding="utf-8"
                )
                evidence["elements"] = self.run_log.rel(elements_path) or elements_path
            except Exception:  # noqa: BLE001 - evidence capture must never itself crash the run
                pass
        return evidence

    # --------------------------------------------------------------- failure

    def _fail(
        self, category: FailureCategory, step: Optional[Step], index: Optional[int], obs: Optional[Observation],
        expected: Optional[str] = None, screen_guess: Optional[str] = None, suggested_action: Optional[str] = None,
        evidence: Optional[dict[str, str]] = None, target_resolution: Optional[list[Any]] = None,
        action_performed: bool = True, outputs_complete: bool = False,
    ) -> None:
        # Any RISKY step that fails AFTER its action was performed leaves
        # the underlying system in a state we can no longer vouch for: the
        # click/submit happened, but its effect is unconfirmed. Flagging
        # this is what an operations team reconciles against by hand,
        # rather than silently trusting a "clean" failure that might not
        # be clean at all. Pre-action gate failures (screen mismatch,
        # target not found, policy denial) never set this -- nothing was
        # done yet.
        if action_performed and step is not None and step.risk.level != "safe":
            self.side_effects_uncertain = True

        if evidence is None:
            evidence = self._capture_evidence((step.id if step else "run") + "-failed", obs)
        observed = self._observed_from(obs, screen_guess=screen_guess) if obs is not None else None
        failure = Failure(
            category=category,
            step=({"id": step.id, "index": index, "intent": step.intent, "kind": step.kind} if step else None),
            expected=expected,
            observed=observed,
            target_resolution=list(target_resolution or []),
            recovery_history=list(self.recovery_history),
            evidence=evidence,
            suggested_action=suggested_action,
        )
        raise _Terminate(self._result(RunStatus.FAILED, failure=failure, outputs_complete=outputs_complete))

    # ---------------------------------------------------------------- finish

    def _finish(self, capability: Capability) -> ReplayResult:
        """Evaluate `capability.success`; check every declared
        `contract.output` is present. Only reached when every step
        completed without an early termination."""
        screen = self._last_screen
        obs = self._observe(label="success-checkpoint")
        ctx = conditions.ConditionContext(
            self.surface, obs, self.outputs, default_frame=(screen.frame if screen else None), policy=self.policy
        )
        ok, why = conditions.evaluate(capability.success.condition, ctx)
        if not ok:
            self._fail(
                FailureCategory.CHECKPOINT_FAILED, None, None, obs,
                expected=capability.success.description, suggested_action=why, action_performed=False,
            )

        missing = [name for name in capability.contract.outputs if name not in self.outputs]
        if missing:
            self._fail(
                FailureCategory.EXTRACTION_FAILED, None, None, obs,
                expected="all contract.outputs present", suggested_action=f"missing outputs: {missing!r}",
                action_performed=False,
            )

        return self._result(RunStatus.SUCCESS, outputs_complete=True)

    def _result(
        self, status: RunStatus, outcome: Optional[OutcomeRef] = None, failure: Optional[Failure] = None,
        outputs_complete: bool = True,
    ) -> ReplayResult:
        return ReplayResult(
            run_id=self.run_log.run_id,
            capability_ref=CapabilityRef(
                id=self.capability.id, version=self.capability.version,
                content_hash=self.capability.compute_content_hash(),
            ),
            status=status,
            outcome=outcome,
            outputs=dict(self.outputs),
            outputs_complete=outputs_complete,
            steps=list(self.step_results),
            drift_signals=list(self.drift_signals),
            human_interventions=list(self.human_interventions),
            approved_by=self.approved_by,
            side_effects_uncertain=self.side_effects_uncertain,
            failure=failure,
            finished_at=utcnow_iso(),
            evidence_dir=str(self.run_log.dir),
        )


def _bind_placeholders(value: str, inputs: dict[str, Any]) -> str:
    """Bind `{{name}}` placeholders in a step's `value` from `inputs`.
    `Capability.validate_semantics()` already guarantees every placeholder
    names a declared input, so a missing key here would mean the artifact
    was never validated -- `.get(..., "")` is defence in depth, not the
    primary safeguard.
    """
    return _PLACEHOLDER_RE.sub(lambda m: str(inputs.get(m.group(1), "")), value)
