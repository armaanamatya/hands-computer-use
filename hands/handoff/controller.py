r"""The control-transfer state machine: who may drive the live session, right now.

`SessionController` is the single authority for `hands/schema/control.py`'s
`ControlState`/`ControlToken` contract. It owns exactly one thing well: when
automation cannot safely proceed, a human takes over the SAME live Playwright
page -- not a copy, not a replay of evidence -- and hands it back so the run
resumes from wherever the human left it. Everything here exists to make that
handover provably safe and provably auditable from disk.

Two invariants make that true:

1. **Epoch-gated tokens.** `ControlToken.epoch` only ever advances when the
   HOLDER actually changes (automation <-> human). `guard()` rejects any
   token whose epoch is not the CURRENT epoch, even if its holder still says
   "automation". This is what makes the handoff a hard boundary rather than
   an advisory one: a step issued (and possibly already in flight) before a
   handoff can never land after it, because the token it was issued with is
   provably stale the instant control changes hands. See `guard()`'s
   docstring for the concrete failure this prevents.

2. **Disk is the source of truth.** Every legal `transition()` -- not just
   the terminal ones -- overwrites `control_state.json` in the run
   directory and appends a `control_transfer` event. If this process dies
   mid-handoff, "who is holding the session right now" is answerable by
   reading a file, not by trusting in-memory state that no longer exists.

## Why RESUME_PENDING is HUMAN-held, not AUTOMATION-held

`_STATE_HOLDER` maps `RESUME_PENDING` to `ControlHolder.HUMAN`, not
`AUTOMATION`, even though the docstring for that state says "automation
verifying where it is". This is not cosmetic: the spec is explicit that
`claim()` bumps the epoch and `resume()` bumps the epoch again -- two
distinct bumps. An epoch bump only happens here when the computed holder
for the new state differs from the current holder (see `transition()`). If
`RESUME_PENDING` were mapped to AUTOMATION, the holder change would already
have happened at `release()` (HUMAN_ACTIVE -> RESUME_PENDING) and `resume()`
(RESUME_PENDING -> AUTOMATION_ACTIVE) would bump nothing at all --
contradicting "resume() ... increments the epoch" directly. Keeping
RESUME_PENDING HUMAN-held also means no automation token can pass `guard()`
during that window even if one leaked out somehow: automation is not
trusted to act again until `resume()` has explicitly re-established it.

## Dialog handling during HUMAN_ACTIVE

`claim()` forces `surface.set_dialog_mode("hold")`. In headed mode the
native browser dialog is visible to the person sitting at the keyboard --
they answer it themselves by clicking the real button. If the surface were
left in (or switched to) "auto_accept", it would race the human's own
click and resolve the dialog out from under them, leaving what they saw on
screen inconsistent with what the app actually did. "hold" is also what
replay already runs in for the whole run (see `ReplayEngine._startup`), so
this is not even a mode switch in the normal case -- just re-asserted,
idempotently, at the moment a human takes over.

## The Playwright-thread pump constraint (see `wait_for_human`)

Playwright's sync API is not a passive object: `expose_binding` callbacks
(here, the human-action recorder) are only delivered to Python while the
main thread is *inside* a Playwright call -- it is Playwright's own sync
wrapper that pumps its driver connection for incoming messages, and it only
does that pumping while one of its own blocking calls is on the stack. A
bare `time.sleep()` does not touch Playwright at all, so for the case this
matters most -- a REAL human, clicking in a headed browser while this
process is otherwise idle -- it would receive zero recorder events for its
whole duration (this is a documented, real constraint of Playwright's sync
API, not a hypothetical; it is also not something this project's own test
suite can exercise directly, since simulating it safely would mean driving
the same page from a second thread, which Playwright's sync API forbids
outright). `wait_for_human()` therefore polls with `page.wait_for_timeout`,
never `time.sleep`, whenever a page is attached -- see `SimulatedOperator`,
which sidesteps the whole question by never needing `wait_for_human` at
all: it drives the page directly and synchronously, on this same thread.

## The recorder's two injection paths, and a real Playwright gotcha

`hands/handoff/human_recorder.js` is a bare arrow-function expression:
`() => { ...; return "installed"; }`. That string is used two different
ways, and they are NOT interchangeable:

- `frame.evaluate(RECORDER_JS)` -- Playwright's `evaluate()` auto-invokes a
  string that parses as a function expression, so this both installs the
  listeners AND returns "installed"/"already-installed"/"no-binding".
- `context.add_init_script(...)` -- this is backed by CDP's
  `Page.addScriptToEvaluateOnNewDocument`, which just runs the given string
  as an ordinary top-level script. A bare `() => {...}` is a *syntactically
  legal* expression statement (function object constructed, then
  discarded) -- there is no error, the body simply never executes. This is
  an easy, silent failure: the symptom is "the recorder works on the page
  that was open at claim() time, but every subsequently-navigated document
  has none of its listeners installed, with no exception anywhere." The fix
  is to wrap it as an IIFE before handing it to `add_init_script`: see
  `_INIT_SCRIPT` below.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

from hands.evidence.run_log import RunLog
from hands.safety.policy import Policy
from hands.schema.control import (
    TRANSITIONS,
    ControlHolder,
    ControlState,
    ControlStateRecord,
    ControlToken,
    HumanAction,
    HumanActionRecord,
    InterventionRequest,
    NotInControl,
    utcnow_iso,
)

RECORDER_JS = (Path(__file__).parent / "human_recorder.js").read_text(encoding="utf-8")

# See the module docstring's "recorder gotcha" section: add_init_script runs
# its argument as a plain script, not an auto-invoked expression, so the
# arrow function must be wrapped in an IIFE here. frame.evaluate() does NOT
# need this wrapper (and must not get it double-wrapped), so this constant
# is used only for add_init_script.
_INIT_SCRIPT = f"({RECORDER_JS})();"

# Which ControlHolder each ControlState implies. A state absent from this
# map (ABANDONED, FINISHED -- both terminal) does not change the holder:
# whoever held the session when the run ended is who the record shows it
# ended with. See the module docstring for why RESUME_PENDING maps to HUMAN.
_STATE_HOLDER: dict[ControlState, ControlHolder] = {
    ControlState.AUTOMATION_ACTIVE: ControlHolder.AUTOMATION,
    ControlState.HANDOFF_PENDING: ControlHolder.AUTOMATION,
    ControlState.HUMAN_ACTIVE: ControlHolder.HUMAN,
    ControlState.RESUME_PENDING: ControlHolder.HUMAN,
}


class SessionController:
    """Owns the control state machine, the current `ControlToken`, and the
    human-action recorder for one run.

    `page`/`context` (or a pre-built `surface`, i.e. a `WebSurface`) are
    optional at construction because a controller may legitimately be built
    before a browser page exists yet; attach them with `attach()` once they
    do. They ARE required by the time `claim()` first runs (the recorder
    needs a live `context`/`page` to install into), and `surface` is
    required for dialog handling (`claim()`'s `set_dialog_mode("hold")`)
    and for `SimulatedOperator`'s `answer_dialog` action.
    """

    def __init__(
        self,
        run_id: str,
        run_log: RunLog,
        policy: Policy,
        page: Any = None,
        context: Any = None,
        surface: Any = None,
    ) -> None:
        self.run_id = run_id
        self.run_log = run_log
        self.policy = policy
        self.surface = surface
        self.page = page if page is not None else getattr(surface, "page", None)
        self.context = context if context is not None else (self.page.context if self.page is not None else None)

        self._state = ControlState.AUTOMATION_ACTIVE
        self._holder = ControlHolder.AUTOMATION
        self._epoch = 0
        self._handoff_id: Optional[str] = None
        self._open_request: Optional[InterventionRequest] = None
        self._active_record: Optional[HumanActionRecord] = None
        self.last_record: Optional[HumanActionRecord] = None
        self._recorder_installed_once = False

    # ------------------------------------------------------------- attach

    def attach(self, page: Any = None, context: Any = None, surface: Any = None) -> None:
        """Late-bind the live Playwright handles / surface. Needed when a
        controller is constructed before its `WebSurface` (the surface
        needs `controller.guard` as its own `guard` callable, so the
        controller must exist first) -- the usual order is:
        `controller = SessionController(...)`,
        `surface = WebSurface(page, guard=controller.guard)`,
        `controller.attach(page=page, surface=surface)`.
        """
        if surface is not None:
            self.surface = surface
        if page is not None:
            self.page = page
        elif self.page is None and surface is not None:
            self.page = getattr(surface, "page", None)
        if context is not None:
            self.context = context
        elif self.context is None and self.page is not None:
            self.context = self.page.context

    # ---------------------------------------------------------- properties

    @property
    def state(self) -> ControlState:
        return self._state

    @property
    def holder(self) -> ControlHolder:
        return self._holder

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def handoff_id(self) -> Optional[str]:
        return self._handoff_id

    def token(self) -> ControlToken:
        return ControlToken(run_id=self.run_id, holder=self._holder, epoch=self._epoch, handoff_id=self._handoff_id)

    # -------------------------------------------------------- transitions

    def transition(self, new_state: ControlState, reason: Optional[str] = None) -> ControlStateRecord:
        """Validate `self.state -> new_state` against `TRANSITIONS` (raise
        `ValueError` if illegal), apply it, and persist it.

        Persisting on every transition -- not only the ones a caller thinks
        of as "important" -- is what makes "who is in control?" answerable
        from disk even after a crash mid-handoff: there is no window where
        the only record of the current holder/epoch is this process's
        memory. `control_state.json` is a snapshot (overwritten each time),
        the `control_transfer` event is the append-only history of how it
        got there.
        """
        legal = TRANSITIONS.get(self._state, set())
        if new_state not in legal:
            raise ValueError(
                f"illegal control transition {self._state.value} -> {new_state.value} "
                f"(legal from {self._state.value}: {sorted(s.value for s in legal) or 'none (terminal)'})"
            )

        new_holder = _STATE_HOLDER.get(new_state, self._holder)
        if new_holder != self._holder:
            # The epoch only moves when the HOLDER actually changes. This is
            # deliberate and is the entire mechanism guard() relies on: a
            # token minted under the old holder must never be honoured
            # under the new one, even a moment later. See guard()'s and the
            # module's docstrings.
            self._epoch += 1
        self._state = new_state
        self._holder = new_holder

        record = ControlStateRecord(
            run_id=self.run_id, state=self._state, holder=self._holder,
            epoch=self._epoch, handoff_id=self._handoff_id, reason=reason,
        )
        self.run_log.write_json("control_state.json", record.model_dump(mode="json"))
        self.run_log.event(
            "control_transfer", state=self._state.value, holder=self._holder.value,
            epoch=self._epoch, handoff_id=self._handoff_id, reason=reason,
        )
        return record

    def guard(self, token: ControlToken) -> None:
        """Raise `NotInControl` unless `token` is an AUTOMATION token at the
        CURRENT epoch, AND the controller's own live state currently agrees
        that automation holds the session. Wired as
        `WebSurface(page, guard=controller.guard)`.

        Why the epoch, specifically, and not just the holder: `holder` alone
        would let a token minted at epoch 3 (before some handoff) still pass
        at epoch 5 (after two more), because its `holder` field still reads
        "automation" -- tokens are immutable snapshots, nothing rewrites
        them in place. The epoch is the part that is guaranteed to have
        moved by the time control comes back, so comparing it is what turns
        "a step issued before a handoff" into something that is
        *structurally* incapable of landing after it, rather than merely
        unlikely to.

        Why ALSO check live state, on top of holder+epoch: "who is in
        control" should be answerable from exactly one authority -- this
        controller's own `_state`/`_holder` -- not reconstructed piecemeal
        from a token's fields plus a separate epoch comparison. Two sources
        that can disagree is the bug this design exists to prevent, and the
        extra check costs nothing. Concretely, `HANDOFF_PENDING` maps to
        `ControlHolder.AUTOMATION` (see `_STATE_HOLDER`) with NO epoch bump
        -- a request_handoff() alone does not change who last held a valid
        token. Without this check, a token minted just before
        `request_handoff()` (holder=automation, epoch still current) would
        pass `guard()` for the whole HANDOFF_PENDING window, even though a
        handoff is already in flight and automation should not be acting.
        Requiring `self._state == ControlState.AUTOMATION_ACTIVE` (not just
        `self._holder == ControlHolder.AUTOMATION`) closes that window: a
        token is only honoured while the controller is unambiguously,
        currently, automation's to act with.
        """
        live_ok = self._state == ControlState.AUTOMATION_ACTIVE and self._holder == ControlHolder.AUTOMATION
        if token.holder != ControlHolder.AUTOMATION or token.epoch != self._epoch or not live_ok:
            raise NotInControl(
                f"token (holder={token.holder.value}, epoch={token.epoch}) is not in control; "
                f"current live state={self._state.value} holder={self._holder.value} epoch={self._epoch}. "
                f"A token issued before a handoff must not be able to act after it, and automation may only "
                f"act while the controller's own live state agrees it holds the session "
                f"(state=AUTOMATION_ACTIVE, holder=automation) -- this is that guarantee firing."
            )

    # ------------------------------------------------------------ handoff

    def request_handoff(self, req: InterventionRequest) -> str:
        """AUTOMATION_ACTIVE -> HANDOFF_PENDING. Persists the intervention
        request to `intervention-<id>.json` and returns its id (the
        "handoff id" used by every subsequent call in this handoff)."""
        self._handoff_id = req.id
        self.transition(ControlState.HANDOFF_PENDING, reason=req.reason_code)
        self._open_request = req
        self.run_log.write_json(f"intervention-{req.id}.json", req.model_dump(mode="json"))
        return req.id

    def list_open_interventions(self) -> list[InterventionRequest]:
        """Intervention requests raised but not yet claimed. At most one:
        this controller models a single run's control state, and the state
        machine only ever has one handoff in flight at a time."""
        if self._state == ControlState.HANDOFF_PENDING and self._open_request is not None:
            return [self._open_request]
        return []

    def claim(self, handoff_id: str, operator_id: str) -> ControlToken:
        """HANDOFF_PENDING -> HUMAN_ACTIVE. Bumps the epoch (killing
        automation's pre-handoff token, see `guard()`), opens a
        `HumanActionRecord`, forces the surface to hold dialogs, and
        installs the recorder into the live page."""
        if handoff_id != self._handoff_id:
            raise ValueError(
                f"claim(): handoff_id {handoff_id!r} does not match the active handoff {self._handoff_id!r}"
            )
        self.transition(ControlState.HUMAN_ACTIVE, reason=f"claimed_by:{operator_id}")
        self._open_request = None
        self._active_record = HumanActionRecord(
            handoff_id=handoff_id, run_id=self.run_id, operator_id=operator_id, claimed_at=utcnow_iso()
        )
        if self.surface is not None:
            self.surface.set_dialog_mode("hold")
        self.install_recorder()
        return self.token()

    def release(
        self, handoff_id: str, resume_mode: str, notes: Optional[str] = None, record_as: str = "omit"
    ) -> HumanActionRecord:
        """HUMAN_ACTIVE -> RESUME_PENDING. Finalises the `HumanActionRecord`
        (does NOT bump the epoch -- that happens at `resume()`, see the
        module docstring for why) and appends it to `human-actions.jsonl`.
        Leaves the page exactly as the human left it: nothing is
        uninstalled or reset here."""
        if handoff_id != self._handoff_id:
            raise ValueError(
                f"release(): handoff_id {handoff_id!r} does not match the active handoff {self._handoff_id!r}"
            )
        if self._active_record is None:
            raise RuntimeError("release() called with no active HumanActionRecord -- was claim() ever called?")
        self.transition(ControlState.RESUME_PENDING, reason=f"released:{resume_mode}")
        record = self._active_record
        record.released_at = utcnow_iso()
        record.resume_mode = resume_mode  # type: ignore[assignment]
        record.notes = notes
        record.record_as = record_as  # type: ignore[assignment]
        self.run_log.append_jsonl("human-actions.jsonl", record.model_dump(mode="json"))
        self._active_record = None
        self.last_record = record
        return record

    def resume(self) -> ControlToken:
        """RESUME_PENDING -> AUTOMATION_ACTIVE. Bumps the epoch and returns
        a fresh automation token -- the only token from this point on that
        `guard()` will accept."""
        self._handoff_id = None
        self.transition(ControlState.AUTOMATION_ACTIVE, reason="resumed")
        return self.token()

    def abandon(self, reason: str) -> ControlStateRecord:
        """HANDOFF_PENDING or HUMAN_ACTIVE -> ABANDONED (no human ever
        came, or the human/operator explicitly gave up). Terminal:
        `TRANSITIONS[ABANDONED]` is empty."""
        record = self.transition(ControlState.ABANDONED, reason=reason)
        self._open_request = None
        self._active_record = None
        return record

    def wait_for_human(
        self, handoff_id: str, timeout_s: float, drain: Optional[Callable[[], None]] = None
    ) -> ControlState:
        """Block the calling (main) thread until this handoff leaves
        HANDOFF_PENDING/HUMAN_ACTIVE, or `timeout_s` elapses (in which case
        it calls `abandon("HUMAN_TIMEOUT")`).

        `drain`, when given, is called once per poll iteration -- this is
        how a real `OperatorServer`'s queued HTTP commands actually reach
        the controller: the HTTP thread only ever enqueues them (see
        `operator_server.py`'s module docstring), and THIS loop, running on
        the same thread that owns Playwright, is what applies them.

        See the module docstring's "Playwright-thread pump" section for why
        this polls with `page.wait_for_timeout(200)` rather than
        `time.sleep`: only a live Playwright call pumps the sync API's
        connection for incoming messages, which is what actually delivers
        `expose_binding` callbacks (the recorder) to Python. A bare sleep
        here would risk silently starving the human-action log for a real
        headed operator while still otherwise appearing to work (the wait
        still completes once `drain`/an operator changes the state) --
        this loop would still correctly resolve the handoff, it just might
        miss recording what the human actually did along the way.
        """
        deadline = time.monotonic() + timeout_s
        while self._state in (ControlState.HANDOFF_PENDING, ControlState.HUMAN_ACTIVE) and self._handoff_id == handoff_id:
            if drain is not None:
                drain()
                if self._state not in (ControlState.HANDOFF_PENDING, ControlState.HUMAN_ACTIVE) or self._handoff_id != handoff_id:
                    break
            if time.monotonic() >= deadline:
                self.abandon("HUMAN_TIMEOUT")
                break
            if self.page is not None:
                self.page.wait_for_timeout(200)
            else:
                time.sleep(0.2)
        return self._state

    # ------------------------------------------------------------ recorder

    def install_recorder(self, context: Any = None, page: Any = None) -> None:
        """Register the `hands_record` binding once, then inject the
        recorder both into every currently-loaded frame (`frame.evaluate`)
        and into every future document (`context.add_init_script`). Safe to
        call more than once (e.g. on a second handoff in the same run): the
        binding/init-script are only ever registered once per controller,
        and the JS itself is idempotent per document
        (`window.__handsRecorderInstalled`).
        """
        context = context or self.context
        page = page or self.page
        if context is None or page is None:
            raise RuntimeError(
                "SessionController has no live Playwright context/page attached; "
                "call attach(page=..., context=...) before claim()"
            )
        if not self._recorder_installed_once:
            context.expose_binding("hands_record", self._on_record)
            # IIFE-wrap for add_init_script -- see the module docstring's
            # "recorder gotcha" section for exactly why this wrapper is
            # required here but must NOT be used for frame.evaluate below.
            context.add_init_script(_INIT_SCRIPT)
            self._recorder_installed_once = True
        for frame in page.frames:
            try:
                frame.evaluate(RECORDER_JS)
            except Exception:  # noqa: BLE001 - detached/cross-origin frame; future docs still get it via add_init_script
                pass

    def record_human_action(self, action: HumanAction) -> None:
        """Append a `HumanAction` that did not come through the DOM
        recorder -- e.g. a dialog answer, which fires no DOM event at all.
        Used by `SimulatedOperator`'s `answer_dialog` step and available to
        a real operator console for the same purpose."""
        if self._active_record is None:
            raise RuntimeError("record_human_action() called with no active handoff (call claim() first)")
        self._active_record.actions.append(action)

    def _on_record(self, source: Any, payload: dict[str, Any]) -> None:
        """`expose_binding` callback for `window.hands_record`. Builds a
        `HumanAction` from the recorder's payload, respecting
        `policy.capture_human_input_values` and, independent of that
        policy, never recording a password field's value under any
        circumstance (belt-and-suspenders: the JS side already withholds it
        for password fields; this checks again from the `sensitive` flag
        AND from the target's own `attrs.type`, in case either signal is
        ever missing)."""
        if self._active_record is None or not isinstance(payload, dict):
            return  # a stray event after release(); nothing to attach it to
        kind = str(payload.get("type", "unknown"))
        frame_url = payload.get("frame_url")
        target = payload.get("target")
        target_desc: Optional[dict[str, Any]] = dict(target) if isinstance(target, dict) else None
        sensitive = bool(payload.get("sensitive"))
        attrs = target_desc.get("attrs") if target_desc else None
        is_password = sensitive or (isinstance(attrs, dict) and str(attrs.get("type") or "").lower() == "password")

        value: Optional[str] = None
        if not is_password and self.policy.capture_human_input_values and "value" in payload:
            value = payload.get("value")

        if target_desc is not None:
            if "has_value" in payload:
                target_desc["has_value"] = bool(payload.get("has_value"))
            if "checked" in payload:
                target_desc["checked"] = payload.get("checked")

        action = HumanAction(ts=utcnow_iso(), type=kind, target=target_desc, value=value, frame_url=frame_url)
        self._active_record.actions.append(action)
