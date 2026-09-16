r"""A scripted stand-in for a human operator.

This exists so the handoff MECHANICS -- claim, act on the live page,
release, resume; the epoch/token discipline that goes with them; the
human-action log they produce -- are exercised deterministically, in CI and
in recorded evidence, without a person actually sitting at the keyboard.
`SimulatedOperator` goes through `SessionController` exactly the way a real
operator (or a real console driving the same controller) would: it does not
bypass `claim()`/`release()`, it does not reach into `ControlToken`
internals, and its DOM interactions run through the SAME live Playwright
page the automation was just using -- not a copy, not a re-run.

## This is a stand-in, not a console -- what a real one would add

- **A UI.** A real operator console shows the person the actual
  `InterventionRequest` -- its `reason_code`, `summary`, `evidence`
  (screenshot/DOM), `suggested_actions` -- so a HUMAN can decide what to do.
  `SimulatedOperator` never looks at the request at all; it blindly runs
  whatever `actions` it was constructed with, decided in advance by
  whoever wrote the test/evidence run.
- **Judgement.** A real operator can recognise "this isn't what I
  expected, I should abort" or improvise a fix the script never
  anticipated. This class has none of that: if `actions` doesn't match
  what's actually on the page, it fails loudly (a Playwright timeout/
  `LookupError`), it does not adapt.
- **Native interaction.** A real human, in headed mode, clicks and types
  with the real mouse/keyboard and answers native dialogs themselves --
  which is exactly why `SessionController.claim()` puts the surface into
  dialog-`"hold"` mode instead of auto-answering. `SimulatedOperator`'s
  `answer_dialog` step exists only because nothing here is actually
  headed; a script cannot click a real native dialog button.

Each action is a single-key dict: `{"fill": {...}}`, `{"click": {...}}`,
`{"answer_dialog": True | False}`. `fill`/`click` take `selector` plus
either `frame_name` or `frame_url_pattern` to pick which frame (a frameset
target, like this project's mock app, can have several frames with
independent documents/URLs); omitting both targets the page's main frame.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from hands.schema.control import HumanAction, HumanActionRecord, utcnow_iso


class SimulatedOperator:
    def __init__(self, controller: Any, actions: list[dict[str, Any]], operator_id: str = "sim-operator") -> None:
        self.controller = controller
        self.actions = list(actions)
        self.operator_id = operator_id

    def run(
        self,
        handoff_id: str,
        resume_mode: str = "auto",
        notes: Optional[str] = None,
        record_as: str = "omit",
    ) -> HumanActionRecord:
        """Claim the given handoff, perform every scripted action against
        the controller's live page, then release. Deliberately does NOT
        call `controller.resume()` -- that belongs to whoever is driving
        the escalation (see `make_escalation_handler`), exactly the same as
        it would for a real operator: a person hands control back with
        "resume"; they do not themselves flip automation back on.
        """
        self.controller.claim(handoff_id, self.operator_id)
        page = self.controller.page
        if page is None:
            raise RuntimeError("SimulatedOperator needs a live Playwright page (controller.page is None)")
        for step in self.actions:
            self._perform(page, step)
        return self.controller.release(handoff_id, resume_mode=resume_mode, notes=notes, record_as=record_as)

    # ------------------------------------------------------------------

    def _frame(self, page: Any, frame_url_pattern: Optional[str], frame_name: Optional[str]) -> Any:
        if frame_url_pattern is None and frame_name is None:
            return page.main_frame
        for f in page.frames:
            if frame_name is not None and f.name == frame_name:
                return f
            if frame_url_pattern is not None and re.search(frame_url_pattern, f.url):
                return f
        raise LookupError(
            f"SimulatedOperator: no live frame matched frame_name={frame_name!r} "
            f"frame_url_pattern={frame_url_pattern!r}"
        )

    def _perform(self, page: Any, step: dict[str, Any]) -> None:
        if "fill" in step:
            spec = step["fill"]
            frame = self._frame(page, spec.get("frame_url_pattern"), spec.get("frame_name"))
            loc = frame.locator(spec["selector"])
            loc.fill(spec["value"])
            # The recorder (human_recorder.js) listens for "change", not
            # "input". Playwright's fill() was observed (against this
            # project's own mock app) to already fire a native "change" on
            # the filled element, so this explicit dispatch is usually a
            # harmless duplicate -- but that behaviour is an implementation
            # detail of fill(), not a documented guarantee, and in
            # particular nothing guarantees a "change" at all for a fill()
            # that is the LAST scripted action (no subsequent focus shift
            # to trigger one). Dispatching it explicitly here removes that
            # dependency on step ordering rather than relying on it.
            loc.dispatch_event("change", {"bubbles": True, "cancelable": True})
        elif "click" in step:
            spec = step["click"]
            frame = self._frame(page, spec.get("frame_url_pattern"), spec.get("frame_name"))
            frame.locator(spec["selector"]).click()
        elif "answer_dialog" in step:
            accept = bool(step["answer_dialog"])
            surface = self.controller.surface
            if surface is None:
                raise RuntimeError("SimulatedOperator: 'answer_dialog' needs controller.surface (a WebSurface)")
            surface.answer_dialog(accept=accept)
            # Native dialogs fire no DOM event the JS recorder could see;
            # log the decision explicitly so it still lands in the human
            # action record.
            self.controller.record_human_action(
                HumanAction(ts=utcnow_iso(), type="dialog", target=None, value=str(accept), frame_url=page.url)
            )
        else:
            raise ValueError(f"SimulatedOperator: unrecognized action {step!r}")
