"""The result contract: what a replay produced, and how to interpret it.

A replay does not just succeed or fail. It can reach a legitimate business
answer ("no such member"), stop and hand off to a human, get refused by
policy, or genuinely fail. `RunStatus` keeps these apart at the type level
-- as distinct enum members, not a boolean plus a free-text string -- so a
caller can `match` on it instead of parsing a message. Conflating "no such
member" with a crash is exactly the mistake the brief calls out.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from hands.schema.control import utcnow_iso


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    BUSINESS_OUTCOME = "BUSINESS_OUTCOME"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


class FailureCategory(str, Enum):
    UNEXPECTED_SCREEN = "UNEXPECTED_SCREEN"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    TARGET_AMBIGUOUS = "TARGET_AMBIGUOUS"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    TIMEOUT = "TIMEOUT"
    APP_ERROR = "APP_ERROR"
    UNEXPECTED_DIALOG = "UNEXPECTED_DIALOG"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    INPUT_INVALID = "INPUT_INVALID"
    RECOVERY_EXHAUSTED = "RECOVERY_EXHAUSTED"
    DRIFT_SUSPECTED = "DRIFT_SUSPECTED"
    POLICY_VIOLATION = "POLICY_VIOLATION"


class _Model(BaseModel):
    """Shared strictness: an unexpected field in a result means the writer
    and this schema have drifted, which should fail loudly, not silently
    drop data evidence later depends on."""

    model_config = ConfigDict(extra="forbid")


class StrategyAttempt(_Model):
    """One resolution attempt for a target, recorded whether it succeeded or
    not -- this is what makes "the recorder used role, replay fell back to
    text" a fact in evidence instead of an inference."""

    strategy: str
    params: dict[str, Any]
    match_count: int
    elapsed_ms: int
    error: Optional[str] = None


class StepResult(_Model):
    step_id: str
    index: int
    intent: str
    status: Literal["ok", "skipped", "recovered", "human_performed", "failed", "blocked"]
    strategy_used: Optional[str] = None
    strategy_rank: Optional[int] = None
    attempts: int = 1
    resolve_ms: Optional[int] = None
    action_ms: Optional[int] = None
    outcomes_matched: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class Observed(_Model):
    """What replay actually saw at the moment a decision was made -- the
    evidence side of a `Failure`, so "why did it think this" is answerable
    without re-running the browser."""

    url: str
    title: str
    screen_guess: Optional[str] = None
    matched_conditions: list[str]
    text_excerpt: str
    open_dialog: Optional[str] = None


class RecoveryRecord(_Model):
    outcome_code: str
    attempt: int
    action: str
    result: str


class Failure(_Model):
    category: FailureCategory
    step: Optional[dict[str, Any]] = None
    expected: Optional[str] = None
    observed: Optional[Observed] = None
    target_resolution: list[StrategyAttempt] = Field(default_factory=list)
    recovery_history: list[RecoveryRecord] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)
    suggested_action: Optional[str] = None


class CapabilityRef(_Model):
    """Which capability, at which content, produced this result -- pinned by
    hash so a result can be traced back to the exact behaviour that ran even
    if the artifact file has since been edited."""

    id: str
    version: str
    content_hash: Optional[str] = None


class OutcomeRef(_Model):
    code: str
    kind: str
    detected_at_step: Optional[str] = None
    message: str


class ReplayResult(_Model):
    """The full record of one replay run: status, outcome, extracted
    outputs, per-step detail, and (on failure) the evidence needed to debug
    it without re-running the browser."""

    run_id: str
    capability_ref: CapabilityRef
    status: RunStatus
    outcome: Optional[OutcomeRef] = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    outputs_complete: bool = True
    steps: list[StepResult] = Field(default_factory=list)
    drift_signals: list[str] = Field(default_factory=list)
    human_interventions: list[str] = Field(default_factory=list)
    approved_by: Optional[str] = None
    side_effects_uncertain: bool = False
    failure: Optional[Failure] = None
    started_at: str = Field(default_factory=utcnow_iso)
    finished_at: Optional[str] = None
    evidence_dir: Optional[str] = None

    def exit_code(self) -> int:
        """Map this result to a process exit code for the CLI.

        0 covers both SUCCESS and BUSINESS_OUTCOME: a legitimate business
        answer such as "no such member" is a result the caller must handle,
        not a crash -- conflating the two is the mistake the brief calls
        out, and a non-zero exit here would train callers (and CI) to treat
        a correct "not found" answer as a broken run. NEEDS_HUMAN (2) and
        POLICY_BLOCKED (3) get their own codes so a script can tell "come
        back after a human resolves this" and "this was refused by policy"
        apart from an unclassified failure (1).
        """
        if self.status in (RunStatus.SUCCESS, RunStatus.BUSINESS_OUTCOME):
            return 0
        if self.status == RunStatus.NEEDS_HUMAN:
            return 2
        if self.status == RunStatus.POLICY_BLOCKED:
            return 3
        return 1
