"""Typed envelope for the evidence log.

`RunLog` (`hands/evidence/run_log.py`) writes plain dicts to
`events.jsonl` -- one JSON object per line, so it's append-only and
streamable. This module exists purely to document and validate that shape
in one place, so every writer agrees on what a line looks like, instead of
each call site inventing its own field set. It is deliberately thin: an
`Event` carries a free-form `data` bag for whatever the specific `kind`
needs, rather than a distinct pydantic model per kind -- the log is meant to
be cheap to write from many call sites (perception, the LLM loop, the
replay engine, the handoff controller), not a second schema to keep in sync
with all of them.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from hands.schema.control import utcnow_iso

EventKind = Literal[
    "run_started",
    "observation",
    "action",
    "action_result",
    "checkpoint",
    "outcome_detected",
    "recovery",
    "policy_check",
    "policy_blocked",
    "intervention_raised",
    "control_transfer",
    "human_action",
    "step_result",
    "run_finished",
    "model_turn",
]


class Event(BaseModel):
    """One line of `events.jsonl`.

    `seq` is a per-run monotonic counter (not a timestamp) so event order is
    unambiguous even when two events share a millisecond. `step_id` and
    `reason` are optional because run-level events (`run_started`,
    `run_finished`, `control_transfer`) aren't about any one step.
    """

    model_config = ConfigDict(extra="forbid")

    seq: int
    ts: str = Field(default_factory=utcnow_iso)
    kind: EventKind
    run_id: str
    step_id: Optional[str] = None
    reason: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)
