"""Control-transfer contract: who may drive the live session right now.

The Playwright process owns the browser for the whole run. What changes during
a handoff is *who may command it*. That is enforced with a ControlToken: every
mutating call on the surface must present a token whose holder is `automation`
and whose epoch is current. The epoch increments on every transfer, so an
in-flight step that was issued before a handoff can never act after it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ControlHolder(str, Enum):
    AUTOMATION = "automation"
    HUMAN = "human"


class ControlState(str, Enum):
    AUTOMATION_ACTIVE = "AUTOMATION_ACTIVE"
    HANDOFF_PENDING = "HANDOFF_PENDING"      # intervention raised, waiting for a human to claim
    HUMAN_ACTIVE = "HUMAN_ACTIVE"            # human holds the session
    RESUME_PENDING = "RESUME_PENDING"        # human released; automation verifying where it is
    ABANDONED = "ABANDONED"                  # no human came, or the human aborted
    FINISHED = "FINISHED"


# Legal transitions. Anything else raises.
TRANSITIONS: dict[ControlState, set[ControlState]] = {
    ControlState.AUTOMATION_ACTIVE: {ControlState.HANDOFF_PENDING, ControlState.FINISHED},
    ControlState.HANDOFF_PENDING: {ControlState.HUMAN_ACTIVE, ControlState.ABANDONED},
    ControlState.HUMAN_ACTIVE: {ControlState.RESUME_PENDING, ControlState.ABANDONED},
    ControlState.RESUME_PENDING: {ControlState.AUTOMATION_ACTIVE, ControlState.HANDOFF_PENDING, ControlState.FINISHED},
    ControlState.ABANDONED: set(),
    ControlState.FINISHED: set(),
}


class ControlToken(BaseModel):
    run_id: str
    holder: ControlHolder
    epoch: int
    handoff_id: Optional[str] = None


class NotInControl(RuntimeError):
    """Raised when automation tries to act without holding the current token."""


class ControlStateRecord(BaseModel):
    """Persisted to control_state.json on every transition, so 'who is in
    control' is answerable from disk even after a crash."""

    run_id: str
    state: ControlState
    holder: ControlHolder
    epoch: int
    handoff_id: Optional[str] = None
    since: str = Field(default_factory=utcnow_iso)
    reason: Optional[str] = None


ReasonCode = Literal[
    "STUCK_NO_PROGRESS",
    "TARGET_NOT_FOUND",
    "UNEXPECTED_SCREEN",
    "RISKY_CONFIRMATION",
    "REQUIRES_HUMAN_STEP",
    "RECOVERY_EXHAUSTED",
    "MODEL_REQUESTED",
]


class InterventionRequest(BaseModel):
    id: str
    run_id: str
    mode: Literal["discovery", "replay"]
    capability_ref: Optional[dict[str, Any]] = None
    reason_code: ReasonCode
    summary: str
    step: Optional[dict[str, Any]] = None          # {id, index, intent, phase}
    expected: Optional[str] = None
    observed: Optional[dict[str, Any]] = None      # {url, title, text_excerpt, open_dialog}
    evidence: dict[str, str] = Field(default_factory=dict)  # {screenshot, dom, elements}
    suggested_actions: list[str] = Field(default_factory=list)
    required_role: str = "operator"
    allowed_resume_modes: list[str] = Field(default_factory=lambda: ["auto", "retry_step", "abort"])
    requested_at: str = Field(default_factory=utcnow_iso)
    expires_at: Optional[str] = None


class HumanAction(BaseModel):
    ts: str
    type: str                                # click | input | keypress | navigate | dialog
    target: Optional[dict[str, Any]] = None  # same descriptor shape as perceived elements
    value: Optional[str] = None              # null unless policy allows capture
    frame_url: Optional[str] = None
    url_before: Optional[str] = None
    url_after: Optional[str] = None


class HumanActionRecord(BaseModel):
    handoff_id: str
    run_id: str
    operator_id: str
    claimed_at: str
    released_at: Optional[str] = None
    actions: list[HumanAction] = Field(default_factory=list)
    notes: Optional[str] = None
    resume_mode: Literal["auto", "retry_step", "abort"] = "auto"
    record_as: Literal["human_task", "omit"] = "omit"
