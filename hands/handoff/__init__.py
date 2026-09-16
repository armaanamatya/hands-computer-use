r"""Human-in-the-loop escalation and control transfer.

When `hands.replay.engine.ReplayEngine` cannot safely proceed, it calls its
`escalate` callable with an `InterventionRequest` and expects back an
`Optional[HumanResolution]` (see `hands/replay/engine.py`'s "escalation
seam" section). This package is what plugs a REAL handoff -- a human taking
over the very same live Playwright session, doing the manual step, and
handing it back -- into that seam, via `make_escalation_handler()` below.

    controller = SessionController(run_id, run_log, policy, page=page)
    surface = WebSurface(page, guard=controller.guard)
    controller.attach(surface=surface)
    engine = ReplayEngine(surface, policy, run_log,
                           escalate=make_escalation_handler(controller, operator=simulated_or_None))

See `controller.py` for the state machine and token discipline,
`operator_server.py` for the (optional) real-operator HTTP console, and
`simulated_operator.py` for the scripted stand-in used in tests/CI.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from hands.handoff.controller import RECORDER_JS, SessionController
from hands.handoff.operator_server import OperatorServer
from hands.handoff.simulated_operator import SimulatedOperator
from hands.replay.engine import HumanResolution
from hands.schema.control import ControlState, HumanActionRecord, InterventionRequest

__all__ = [
    "RECORDER_JS",
    "SessionController",
    "OperatorServer",
    "SimulatedOperator",
    "make_escalation_handler",
]

# The control layer's `HumanActionRecord.resume_mode` ("auto" | "retry_step"
# | "abort") and the engine's `HumanResolution.action` ("retry_step" |
# "skip_step" | "abort") are two views of the same three-way decision, named
# from two different vantage points -- and two of the three names already
# line up. The one that doesn't ("auto" <-> "skip_step") is not a free
# naming choice: at `ReplayEngine._gate_policy`'s RISKY_CONFIRMATION site,
# `HumanResolution(action="retry_step")` makes the engine EXECUTE the step's
# own action; `"skip_step"` makes it raise `_StepDoneByHuman` and move on
# without executing anything. If a human operator already performed the
# equivalent action live in the session (the normal case: they resolve
# whatever automation was stuck on and hand back "go ahead, I'm done" --
# `resume_mode="auto"`), mapping that to `"retry_step"` would make the
# engine perform the SAME action a second time -- e.g. double-submitting a
# form the human already submitted. `"auto"` therefore maps to
# `"skip_step"`: nothing is left for automation to redo. `"retry_step"` at
# the control layer means the human only cleared an OBSTACLE (dismissed
# something, corrected a value) without performing the step's own action
# themselves, so automation's own retry is still the right next move, and
# maps straight across. `"abort"` needs no translation either way.
_RESUME_MODE_TO_ACTION: dict[str, str] = {
    "auto": "skip_step",
    "retry_step": "retry_step",
    "abort": "abort",
}


def make_escalation_handler(
    controller: SessionController,
    operator: Optional[SimulatedOperator] = None,
    timeout_s: Optional[float] = None,
) -> Callable[[InterventionRequest], Optional[HumanResolution]]:
    """Return a callable matching `ReplayEngine`'s `escalate` contract
    exactly: `Callable[[InterventionRequest], Optional[HumanResolution]]`.

    On every call it:
      1. `controller.request_handoff(request)` -- AUTOMATION_ACTIVE ->
         HANDOFF_PENDING, persists `intervention-<id>.json`.
      2. Gets a human into the seat: if `operator` (a `SimulatedOperator`)
         was given, drives it synchronously -- claim, act, release, all on
         this same (main/Playwright-owning) thread, no waiting required.
         Otherwise waits for a REAL operator via
         `controller.wait_for_human(handoff_id, timeout_s or
         policy.human_wait_timeout_s)`, which pumps Playwright so any
         attached `OperatorServer`'s queued commands (see
         `operator_server.py`) and the live recorder's `expose_binding`
         callbacks are actually delivered while it waits.
      3. If the handoff ended in RESUME_PENDING (a human released it),
         calls `controller.resume()` -- AUTOMATION_ACTIVE again, fresh
         epoch -- and converts `controller.last_record.resume_mode` into a
         `HumanResolution` via `_RESUME_MODE_TO_ACTION` (see that table's
         comment for exactly why `"auto"` becomes `"skip_step"`).
      4. If it ended in ABANDONED (timeout, or an explicit abort), returns
         `HumanResolution(action="abort")` -- the engine finishes the run
         `RunStatus.ABANDONED`.

    Never returns `None`: `None` in the engine's contract means "no
    escalation handler configured at all" (`ReplayEngine.escalate_fn is
    None`), which is a different, degenerate case this factory does not
    produce -- this handler always resolves one way or another.
    """

    def handle(request: InterventionRequest) -> Optional[HumanResolution]:
        handoff_id = controller.request_handoff(request)

        if operator is not None:
            operator.run(handoff_id)
        else:
            effective_timeout = controller.policy.human_wait_timeout_s if timeout_s is None else timeout_s
            controller.wait_for_human(handoff_id, effective_timeout)

        if controller.state == ControlState.RESUME_PENDING:
            controller.resume()
            record: Optional[HumanActionRecord] = controller.last_record
            resume_mode = record.resume_mode if record is not None else "abort"
            action = _RESUME_MODE_TO_ACTION.get(resume_mode, "abort")
            note = record.notes if record is not None else None
            return HumanResolution(action=action, note=note)  # type: ignore[arg-type]

        if controller.state == ControlState.ABANDONED:
            return HumanResolution(
                action="abort",
                note=f"handoff {handoff_id} ended without a human resuming it (state=ABANDONED)",
            )

        # Any other resting state here would mean the controller was driven
        # outside this handler (e.g. something called resume() early) --
        # fail loudly rather than silently guess what the engine should do.
        raise RuntimeError(
            f"make_escalation_handler: unexpected control state {controller.state.value!r} "
            f"after handoff {handoff_id!r}"
        )

    return handle
