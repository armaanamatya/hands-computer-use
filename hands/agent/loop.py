"""The discovery loop: the model observes a live browser UI, decides, and
acts until a natural-language goal is met (or a clear stop condition fires).

This module owns the agentic loop only. It never constructs a Policy (that
type belongs to `hands.safety.policy`, written by another agent and possibly
not landed yet) - it accepts an already-constructed policy-like object via
the constructor and calls `.check_action(...)` on it duck-typed, defaulting
to None which means "no gate, everything allowed".
"""
from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hands.agent.llm import LLMProvider
from hands.agent.prompts import PROMPT_VERSION, build_system_prompt, render_observation
from hands.agent.tools import TOOL_DEFS, TOOL_NAMES, TOOL_TO_ACTION_KIND, build_action
from hands.evidence.run_log import RunLog
from hands.safety.redact import Redactor
from hands.schema.control import ControlHolder, ControlToken, InterventionRequest
from hands.surface.base import Action, Element, Observation, Surface

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


@dataclass
class TraceStep:
    index: int
    tool: str
    tool_input: dict[str, Any]
    reason: str
    element: Optional[dict[str, Any]]
    action_result: dict[str, Any]
    url_before: Optional[str]
    url_after: Optional[str]
    screenshot: Optional[str]
    observation_index: int
    policy_decision: Optional[dict[str, Any]]
    error: Optional[str]


@dataclass
class DiscoveryTrace:
    goal: str
    status: str  # "success" | "needs_human" | "stopped" | "failed"
    steps: list[TraceStep]
    outputs: dict[str, Any]
    summary: str
    stop_reason: str
    observations: list[dict[str, Any]]
    intervention: Optional[InterventionRequest]
    model: Optional[str]
    prompt_version: str
    usage_total: dict[str, int]
    started_at: str
    finished_at: str


class DiscoveryLoop:
    # "N consecutive turns produce no observable change" - N is a design
    # choice the spec leaves open; 3 gives the model room to try a couple of
    # distinct approaches before we call it a dead end.
    NO_CHANGE_STREAK_LIMIT = 3
    # "the same (tool, element identity, value) triple repeats 3 times" - given verbatim.
    TRIPLE_REPEAT_LIMIT = 3

    def __init__(
        self,
        surface: Surface,
        run_log: RunLog,
        redactor: Redactor,
        provider: LLMProvider,
        goal: str,
        inputs: dict[str, str],
        sensitive_input_names: Optional[set[str]] = None,
        policy: Any = None,
        control_token: Optional[ControlToken] = None,
        max_steps: int = 40,
        run_timeout_s: int = 300,
        entry_url: str = "",
    ) -> None:
        self.surface = surface
        self.run_log = run_log
        self.redactor = redactor
        self.provider = provider
        self.goal = goal
        self.inputs = dict(inputs or {})
        self.sensitive_input_names = set(sensitive_input_names or set())
        self.policy = policy
        self.control_token = control_token or ControlToken(
            run_id=run_log.run_id, holder=ControlHolder.AUTOMATION, epoch=0
        )
        self.max_steps = max_steps
        self.run_timeout_s = run_timeout_s
        self.entry_url = entry_url

        # Register every sensitive value with the Redactor at construction so
        # it can never leak into events.jsonl / transcript.jsonl, no matter
        # where it later shows up as a plain string.
        for name in self.sensitive_input_names:
            value = self.inputs.get(name)
            if value:
                self.redactor.add(name, value)

        self.captured_outputs: dict[str, str] = {}
        self._action_counts: dict[tuple[Any, ...], int] = {}
        self._obs_index = 0

    # ------------------------------------------------------------------ run

    def run(self) -> DiscoveryTrace:
        started = datetime.now(timezone.utc)
        t0 = time.monotonic()

        steps: list[TraceStep] = []
        observations: list[dict[str, Any]] = []
        usage_total: dict[str, int] = {}
        status = "stopped"
        stop_reason = "unknown"
        summary = ""
        outputs: dict[str, Any] = {}
        intervention: Optional[InterventionRequest] = None
        model_name: Optional[str] = None

        inputs_desc = self._build_inputs_desc()
        system_prompt = build_system_prompt(inputs_desc)
        messages: list[dict[str, Any]] = []

        nav_reason = "open the target application"
        nav_result = self.surface.act(
            Action(kind="navigate", url=self.entry_url, reason=nav_reason), self.control_token
        )
        shot = self.run_log.next_screenshot("entry")
        obs = self.surface.observe(screenshot_path=shot)
        observations.append(obs.to_dict())
        self._obs_index = 0
        self.run_log.event(
            "action",
            tool="navigate",
            reason=nav_reason,
            tool_input={"url": self.entry_url},
            policy_decision=None,
            element=None,
            action_result={"ok": nav_result.ok, "error": nav_result.error, "url_after": nav_result.url_after},
        )

        prev_signature = self._signature(obs)
        no_change_streak = 0
        step_index = 0

        def elapsed_s() -> float:
            return time.monotonic() - t0

        def terminate(new_status: str, new_reason: str, note: str = "") -> None:
            nonlocal status, stop_reason, summary
            status, stop_reason = new_status, new_reason
            if note:
                summary = note
            self.run_log.event("run_terminated", status=status, stop_reason=stop_reason, note=note)

        while True:
            if step_index >= self.max_steps:
                terminate("stopped", "max_steps")
                break
            if elapsed_s() >= self.run_timeout_s:
                terminate("stopped", "run_timeout")
                break

            messages.append({"role": "user", "content": self._user_turn_content(obs, inputs_desc)})
            rendered_for_log = render_observation(obs, inputs_desc)

            resp = self.provider.complete(system=system_prompt, messages=messages, tools=TOOL_DEFS)
            model_name = resp.model
            for k, v in (resp.usage or {}).items():
                usage_total[k] = usage_total.get(k, 0) + v

            messages.append({"role": "assistant", "content": resp.content_blocks})

            tool_call_log = [
                {
                    "name": getattr(b, "name", None),
                    "input": getattr(b, "input", None),
                    "reason": (getattr(b, "input", None) or {}).get("reason"),
                }
                for b in resp.content_blocks
                if getattr(b, "type", None) == "tool_use"
            ]
            self.run_log.transcript(
                {
                    "message_id": resp.message_id,
                    "model": resp.model,
                    "usage": resp.usage,
                    "latency_ms": resp.latency_ms,
                    "stop_reason": resp.stop_reason,
                    "observation": rendered_for_log,
                    "screenshot": self.run_log.rel(obs.screenshot_path),
                    "tool_calls": tool_call_log,
                }
            )

            if resp.stop_reason != "tool_use":
                # Every non-tool_use stop_reason is a clean, logged
                # termination - never an exception. `tool_use` is the only
                # value that continues the loop.
                mapped = {
                    "end_turn": "end_turn",
                    "max_tokens": "max_tokens",
                    "refusal": "refusal",
                    "pause_turn": "pause_turn",
                    "model_context_window_exceeded": "context_window_exceeded",
                }.get(resp.stop_reason, f"unhandled_stop_reason_{resp.stop_reason}")
                note = f"model stopped with stop_reason={resp.stop_reason!r}"
                if resp.stop_details:
                    note += f"; stop_details={resp.stop_details!r}"
                terminate("failed", mapped, note)
                break

            tool_use_blocks = [b for b in resp.content_blocks if getattr(b, "type", None) == "tool_use"]
            if not tool_use_blocks:
                terminate("failed", "tool_use_no_blocks", "stop_reason was tool_use but no tool_use block was present")
                break

            tool_results: list[dict[str, Any]] = []
            turn_terminate: Optional[dict[str, Any]] = None
            for i, block in enumerate(tool_use_blocks):
                if turn_terminate is not None:
                    # A prior tool call in this same turn ended the run.
                    # Still answer every tool_use block so `messages` stays
                    # API-valid for anyone resuming this trace.
                    tool_results.append(self._tool_result(getattr(block, "id", ""), "not executed: run ended", is_error=True))
                    continue
                if step_index >= self.max_steps:
                    tool_results.append(self._tool_result(getattr(block, "id", ""), "max_steps reached; not executed", is_error=True))
                    continue
                step, tool_result, step_terminate = self._execute_tool(block, obs, step_index)
                steps.append(step)
                tool_results.append(tool_result)
                step_index += 1
                if step_terminate is not None:
                    turn_terminate = step_terminate
            messages.append({"role": "user", "content": tool_results})

            if turn_terminate is not None:
                status = turn_terminate["status"]
                stop_reason = turn_terminate["stop_reason"]
                summary = turn_terminate.get("summary", summary)
                outputs = turn_terminate.get("outputs", outputs)
                intervention = turn_terminate.get("intervention", intervention)
                self.run_log.event("run_terminated", status=status, stop_reason=stop_reason, note=summary)
                break

            shot = self.run_log.next_screenshot(f"obs{len(observations)}")
            obs = self.surface.observe(screenshot_path=shot)
            observations.append(obs.to_dict())
            self._obs_index = len(observations) - 1

            sig = self._signature(obs)
            no_change_streak = no_change_streak + 1 if sig == prev_signature else 0
            prev_signature = sig
            if no_change_streak >= self.NO_CHANGE_STREAK_LIMIT:
                terminate("stopped", "no_progress", f"no observable change across {no_change_streak} consecutive turns")
                break

        finished = datetime.now(timezone.utc)
        return DiscoveryTrace(
            goal=self.goal,
            status=status,
            steps=steps,
            outputs=outputs,
            summary=summary,
            stop_reason=stop_reason,
            observations=observations,
            intervention=intervention,
            model=model_name,
            prompt_version=PROMPT_VERSION,
            usage_total=usage_total,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
        )

    # -------------------------------------------------------------- helpers

    def _build_inputs_desc(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in self.inputs:
            if name in self.sensitive_input_names:
                out[name] = "sensitive credential - value withheld; reference only via the placeholder"
            else:
                out[name] = "run input value - reference via the placeholder, or type a literal value if the goal already gives it to you"
        return out

    def _user_turn_content(self, obs: Observation, inputs_desc: dict[str, str]) -> list[dict[str, Any]]:
        rendered = render_observation(obs, inputs_desc)
        content: list[dict[str, Any]] = [{"type": "text", "text": f"GOAL: {self.goal}\n\n{rendered}"}]
        b64 = self._b64_image(obs.screenshot_path)
        if b64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
        return content

    @staticmethod
    def _b64_image(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            data = Path(path).read_bytes()
        except OSError:
            return None
        if not data:
            return None
        return base64.standard_b64encode(data).decode("utf-8")

    @staticmethod
    def _signature(obs: Observation) -> tuple[Any, ...]:
        frame_urls = tuple(sorted(f.url for f in obs.frames))
        names = tuple(sorted(f"{e.frame_index}:{e.role}:{e.name}" for e in obs.elements))
        return (frame_urls, names)

    @staticmethod
    def _element_identity(el: Optional[Element]) -> str:
        if el is None:
            return "-"
        return f"{el.frame_index}:{el.role}:{el.name}"

    @staticmethod
    def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}

    def _substitute(self, text: str) -> tuple[str, list[str]]:
        missing: list[str] = []

        def repl(m: "re.Match[str]") -> str:
            name = m.group(1)
            if name not in self.inputs:
                missing.append(name)
                return m.group(0)
            return self.inputs[name]

        resolved = _PLACEHOLDER_RE.sub(repl, text or "")
        return resolved, missing

    def _check_policy(self, action_kind: str, url: str, element: Optional[Element]) -> Any:
        if self.policy is None:
            return None
        return self.policy.check_action(
            action_kind,
            url,
            element_name=(element.name if element else None),
            element_attrs=(element.attrs if element else None),
            coordinate_fallback=False,
        )

    @staticmethod
    def _decision_to_dict(decision: Any) -> Optional[dict[str, Any]]:
        if decision is None:
            return None
        return {
            "allowed": getattr(decision, "allowed", True),
            "rule": getattr(decision, "rule", None),
            "detail": getattr(decision, "detail", None),
            "requires_confirmation": getattr(decision, "requires_confirmation", False),
        }

    def _handle_read_value(self, tool_input: dict[str, Any], obs: Observation) -> tuple[Optional[str], Optional[str]]:
        idx = tool_input.get("element")
        output_name = tool_input.get("output_name")
        if not output_name:
            return None, "read_value requires output_name"
        try:
            el = obs.element(idx)
        except (KeyError, TypeError):
            return None, f"element {idx} not found in the last observation"
        value = el.attrs.get("value") or el.name
        self.captured_outputs[output_name] = value
        return value, None

    @staticmethod
    def _normalize_claimed_outputs(raw: Any) -> dict[str, str]:
        """`done` declares outputs as a list of {name, value} pairs.

        The list form is used because an open-ended object cannot satisfy
        strict function calling (see tools.py). A mapping is still accepted
        so that a transcript recorded before that change, or a hand-written
        replay script, keeps working.
        """
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items()}
        claimed: dict[str, str] = {}
        for entry in raw or []:
            if isinstance(entry, dict) and "name" in entry:
                claimed[str(entry["name"])] = entry.get("value")
        return claimed

    def _handle_done(self, tool_use_id: str, tool_input: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        outputs_claimed = self._normalize_claimed_outputs(tool_input.get("outputs"))
        summary = tool_input.get("summary") or ""
        missing = [k for k in outputs_claimed if k not in self.captured_outputs]
        final_outputs = {k: self.captured_outputs.get(k, v) for k, v in outputs_claimed.items()}
        self.run_log.event(
            "done",
            reason=tool_input.get("reason", ""),
            outputs=final_outputs,
            summary=summary,
            missing_reads=missing,
        )
        if missing:
            summary = (
                summary
                + f"\n[DISCREPANCY: output(s) {missing} were declared in done() but never captured with read_value]"
            ).strip()
            control = {
                "status": "failed",
                "stop_reason": "done_unvalidated_outputs",
                "summary": summary,
                "outputs": final_outputs,
            }
        else:
            control = {"status": "success", "stop_reason": "done", "summary": summary, "outputs": final_outputs}
        return control, self._tool_result(tool_use_id, "run finished")

    def _handle_request_human(
        self, tool_use_id: str, tool_input: dict[str, Any], obs: Observation, shot_path: str, step_index: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reason_code = tool_input.get("reason_code", "STUCK_NO_PROGRESS")
        reason = tool_input.get("reason", "")
        text_excerpt = next((t for t in obs.frame_texts.values() if t), "")[:300]
        intervention = InterventionRequest(
            id=f"{self.run_log.run_id}-ih{step_index + 1}",
            run_id=self.run_log.run_id,
            mode="discovery",
            reason_code=reason_code,
            summary=reason or "the agent requested a human",
            step={"index": step_index, "intent": reason, "phase": "discovery"},
            observed={
                "url": obs.url,
                "title": obs.title,
                "text_excerpt": text_excerpt,
                "open_dialog": obs.dialog is not None,
            },
            evidence={"screenshot": self.run_log.rel(shot_path) or ""},
        )
        self.run_log.event("request_human", reason=reason, reason_code=reason_code, intervention_id=intervention.id)
        control = {
            "status": "needs_human",
            "stop_reason": "request_human",
            "summary": reason,
            "intervention": intervention,
        }
        return control, self._tool_result(tool_use_id, "human requested; run pausing")

    def _execute_tool(
        self, block: Any, obs_before: Observation, step_index: int
    ) -> tuple[TraceStep, dict[str, Any], Optional[dict[str, Any]]]:
        name = getattr(block, "name", "") or ""
        tool_input = dict(getattr(block, "input", None) or {})
        tool_use_id = getattr(block, "id", "")
        reason = tool_input.get("reason") or f"(no reason provided for {name})"

        element_idx = tool_input.get("element")
        element_obj: Optional[Element] = None
        if isinstance(element_idx, int):
            try:
                element_obj = obs_before.element(element_idx)
            except KeyError:
                element_obj = None

        url_before = self.surface.url()
        shot_path = self.run_log.next_screenshot(f"step{step_index + 1}-{name}")
        self.surface.screenshot(shot_path)

        terminate: Optional[dict[str, Any]] = None
        error: Optional[str] = None
        policy_decision_dict: Optional[dict[str, Any]] = None
        action_result_dict: dict[str, Any] = {
            "ok": True,
            "error": None,
            "duration_ms": 0,
            "url_after": url_before,
            "dialogs": [],
        }
        tool_result: dict[str, Any]

        if name not in TOOL_NAMES:
            error = f"unknown tool: {name}"
            action_result_dict = {"ok": False, "error": error, "duration_ms": 0, "url_after": url_before, "dialogs": []}
            tool_result = self._tool_result(tool_use_id, error, is_error=True)
            self.run_log.event("tool_error", tool=name, reason=reason, error=error)

        elif name == "done":
            terminate, tool_result = self._handle_done(tool_use_id, tool_input)

        elif name == "request_human":
            terminate, tool_result = self._handle_request_human(tool_use_id, tool_input, obs_before, shot_path, step_index)

        elif name == "note_condition":
            self.run_log.event(
                "condition_noted",
                reason=reason,
                condition_code=tool_input.get("code"),
                condition_kind=tool_input.get("kind"),
                text_pattern=tool_input.get("text_pattern"),
            )
            tool_result = self._tool_result(tool_use_id, "condition noted")

        elif name == "read_value":
            value, err = self._handle_read_value(tool_input, obs_before)
            error = err
            tool_result = (
                self._tool_result(tool_use_id, err, is_error=True)
                if err
                else self._tool_result(tool_use_id, f"captured {tool_input.get('output_name')!r}")
            )
            self.run_log.event(
                "read_value",
                reason=reason,
                output_name=tool_input.get("output_name"),
                element=element_obj.to_dict() if element_obj else None,
                ok=err is None,
                error=err,
            )

        else:
            # Surface-mutating tool: click / type_text / select_option / press_key / navigate / wait.
            action_kind = TOOL_TO_ACTION_KIND[name]
            resolved_value: Optional[str] = None
            missing: list[str] = []
            if name == "type_text":
                resolved_value, missing = self._substitute(tool_input.get("text", ""))

            if missing:
                error = f"unknown input placeholder(s): {missing}"
                action_result_dict = {"ok": False, "error": error, "duration_ms": 0, "url_after": url_before, "dialogs": []}
                tool_result = self._tool_result(tool_use_id, error, is_error=True)
                self.run_log.event(
                    "action_error",
                    tool=name,
                    reason=reason,
                    tool_input=self.redactor.obj(tool_input),
                    error=error,
                )
            else:
                decision = self._check_policy(action_kind, obs_before.url, element_obj)
                policy_decision_dict = self._decision_to_dict(decision)
                if decision is not None and not getattr(decision, "allowed", True):
                    detail = getattr(decision, "detail", None) or getattr(decision, "rule", None) or "denied by policy"
                    error = detail
                    action_result_dict = {"ok": False, "error": detail, "duration_ms": 0, "url_after": url_before, "dialogs": []}
                    tool_result = self._tool_result(tool_use_id, f"blocked by policy: {detail}", is_error=True)
                    self.run_log.event(
                        "policy_blocked",
                        tool=name,
                        reason=reason,
                        tool_input=self.redactor.obj(tool_input),
                        policy_decision=policy_decision_dict,
                        element=element_obj.to_dict() if element_obj else None,
                    )
                else:
                    try:
                        action = build_action(name, tool_input, resolved_value=resolved_value)
                        assert action is not None
                        result = self.surface.act(action, self.control_token)
                        action_result_dict = {
                            "ok": result.ok,
                            "error": result.error,
                            "duration_ms": result.duration_ms,
                            "url_after": result.url_after,
                            "dialogs": [d.__dict__ for d in result.dialogs],
                        }
                        error = result.error
                        tool_result = self._tool_result(
                            tool_use_id,
                            f"ok, url now {result.url_after}" if result.ok else f"error: {result.error}",
                            is_error=not result.ok,
                        )
                    except (KeyError, LookupError) as e:
                        error = f"{e.__class__.__name__}: {e}"
                        action_result_dict = {"ok": False, "error": error, "duration_ms": 0, "url_after": url_before, "dialogs": []}
                        tool_result = self._tool_result(tool_use_id, error, is_error=True)

                    self.run_log.event(
                        "action",
                        tool=name,
                        reason=reason,
                        tool_input=self.redactor.obj(tool_input),
                        # The resolved (placeholder-substituted) value is logged
                        # here deliberately, through the Redactor, so the log
                        # proves the redaction layer actually fired on any
                        # sensitive value that passed through - rather than
                        # being vacuously clean because the value was never
                        # logged anywhere at all.
                        resolved_value=self.redactor.text(resolved_value) if resolved_value is not None else None,
                        policy_decision=policy_decision_dict,
                        element=element_obj.to_dict() if element_obj else None,
                        action_result=action_result_dict,
                    )

                    value_key = (
                        tool_input.get("text")
                        if name == "type_text"
                        else tool_input.get("option")
                        if name == "select_option"
                        else tool_input.get("key")
                        if name == "press_key"
                        else tool_input.get("url")
                        if name == "navigate"
                        else tool_input.get("ms")
                    )
                    key = (name, self._element_identity(element_obj), value_key)
                    self._action_counts[key] = self._action_counts.get(key, 0) + 1
                    if self._action_counts[key] >= self.TRIPLE_REPEAT_LIMIT:
                        terminate = {
                            "status": "stopped",
                            "stop_reason": "no_progress",
                            "summary": (
                                f"repeated {name} on the same element with the same value "
                                f"{self._action_counts[key]} times with no progress."
                            ),
                        }

        step = TraceStep(
            index=step_index,
            tool=name,
            tool_input=self.redactor.obj(tool_input),
            reason=reason,
            element=element_obj.to_dict() if element_obj else None,
            action_result=action_result_dict,
            url_before=url_before,
            url_after=action_result_dict.get("url_after"),
            screenshot=self.run_log.rel(shot_path),
            observation_index=self._obs_index,
            policy_decision=policy_decision_dict,
            error=error,
        )
        return step, tool_result, terminate
