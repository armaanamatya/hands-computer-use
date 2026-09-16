"""The capability artifact: a discovered UI flow, typed and replayable.

A `Capability` is the thing a discovery run produces and a replay run
consumes with no model in the loop. Every field on it exists to answer a
question the replay engine (or a human reviewing the artifact) will actually
ask at run time: which screen am I on, how do I find this element again, is
this step safe to retry, what does "done" look like. Nothing here is
decorative -- if a field couldn't change a replay decision it isn't in this
file.

Design rule used throughout this module: a field defaults to a value only
when (a) the artifact spec explicitly shows `= value` for it, (b) it is
typed `X | None`, in which case it defaults to `None` -- following
`hands.schema.control`'s own convention of defaulting every `Optional`
field, or (c) it is a "when was this recorded" timestamp, which defaults via
`utcnow_iso()` for the same reason `control.py` does. Everything else is
required. For a safety-relevant, machine-produced artifact, forcing explicit
construction (e.g. `Step.risk`, `Step.wait`, `Screen.landmarks`,
`TargetSpec.frame`) is worth the extra verbosity: it stops a field from
being silently unset by omission.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hands.schema.control import utcnow_iso


class _Model(BaseModel):
    """Shared strictness for this module's models.

    `extra="forbid"` so a typo in a hand-authored capability, overlay patch,
    or outcomes-library entry fails validation loudly instead of silently
    dropping a field the recorder or replay engine expected to see.
    """

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Frame scoping
# --------------------------------------------------------------------------


class FrameMatcher(_Model):
    """Selects which frame a condition, screen, or target is evaluated against.

    Matched in this order of preference: `url_pattern`, then `name`, then
    `index`. An unnamed `<frame>` element reports `name == ""` in the DOM
    (verified against the mock app's frameset), so `name` alone can't
    disambiguate the common case; `url_pattern` is therefore primary and
    `index` -- positional, and brittle across reloads or frame reordering --
    is the last resort.
    """

    url_pattern: str | None = None
    name: str | None = None
    index: int | None = None


# --------------------------------------------------------------------------
# Targeting
# --------------------------------------------------------------------------


class Strategy(_Model):
    """One way to (re)locate a target element, tried in list order at replay.

    `rationale` is required and non-empty: it is where the recorder (or a
    reviewer) records *why* this strategy should keep resolving to the right
    element when the DOM shifts slightly. That reasoning-about-robustness is
    what the brief asks for -- a strategy with no rationale is just a guess
    that happened to work once.
    """

    kind: Literal["role", "text", "relative", "structural", "bbox"]
    params: dict[str, Any]
    rationale: str = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def _rationale_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale must not be blank")
        return v


class TargetSpec(_Model):
    """Everything needed to find one element again on replay, without a model.

    `strategies` is ordered: replay tries them in sequence and records every
    attempt (see `hands.schema.result.StrategyAttempt`), so a later,
    coarser strategy (e.g. `bbox`) is a documented fallback, not a hidden
    one. `snapshot` carries a discovery-time-only descriptor (role, name,
    bbox, viewport) used purely as forensic context on failure; replay never
    resolves against it.
    """

    description: str
    frame: FrameMatcher
    strategies: list[Strategy] = Field(min_length=1)
    snapshot: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Condition algebra (closed set, discriminated union on `type`)
# --------------------------------------------------------------------------


class UrlMatches(_Model):
    type: Literal["url_matches"] = "url_matches"
    pattern: str
    frame: FrameMatcher | None = None


class TitleMatches(_Model):
    type: Literal["title_matches"] = "title_matches"
    pattern: str
    frame: FrameMatcher | None = None


class TextPresent(_Model):
    type: Literal["text_present"] = "text_present"
    pattern: str
    negate: bool = False
    frame: FrameMatcher | None = None


class ElementVisible(_Model):
    type: Literal["element_visible"] = "element_visible"
    target: TargetSpec
    frame: FrameMatcher | None = None


class ElementAbsent(_Model):
    type: Literal["element_absent"] = "element_absent"
    target: TargetSpec
    frame: FrameMatcher | None = None


class ValueEquals(_Model):
    type: Literal["value_equals"] = "value_equals"
    target: TargetSpec
    value: str
    frame: FrameMatcher | None = None


class OutputMatches(_Model):
    type: Literal["output_matches"] = "output_matches"
    output: str
    pattern: str
    frame: FrameMatcher | None = None


class DialogOpen(_Model):
    type: Literal["dialog_open"] = "dialog_open"
    pattern: str | None = None
    frame: FrameMatcher | None = None


class AllOf(_Model):
    type: Literal["all_of"] = "all_of"
    conditions: list[Condition]
    frame: FrameMatcher | None = None


class AnyOf(_Model):
    type: Literal["any_of"] = "any_of"
    conditions: list[Condition]
    frame: FrameMatcher | None = None


class Not(_Model):
    """Negation. `not` is a Python keyword, so the payload field is `not_`
    in code and aliased to `not` on the wire; `populate_by_name` lets callers
    construct it either way."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["not_"] = "not_"
    not_: Condition = Field(alias="not")
    frame: FrameMatcher | None = None


Condition = Annotated[
    Union[
        UrlMatches,
        TitleMatches,
        TextPresent,
        ElementVisible,
        ElementAbsent,
        ValueEquals,
        OutputMatches,
        DialogOpen,
        AllOf,
        AnyOf,
        Not,
    ],
    Field(discriminator="type"),
]
"""The closed set of things a replay can check against the live page.

Every variant may carry an optional `frame` scope: conditions are evaluated
against the frame it selects, not necessarily the top document (see
`Screen`'s docstring for why that distinction matters). `AllOf`/`AnyOf`/`Not`
are self-referential, so their forward references are resolved below with
`model_rebuild()`.
"""

for _cls in (AllOf, AnyOf, Not):
    _cls.model_rebuild()


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


class RowWhere(_Model):
    """Selects a row in an extracted table by matching one of its cells."""

    header: str
    matches: str


class TableSelector(_Model):
    header_match: str
    row_where: RowWhere


class ParseSpec(_Model):
    """Post-processing applied to a raw extracted string before it becomes an output."""

    type: Literal["string", "int", "money", "date"]
    regex: str | None = None
    group: int = 0


class Extraction(_Model):
    """Reads one value out of the page and binds it to a named contract output.

    `from_` is aliased to `from` on the wire for the same reason `Not.not_`
    is: `from` is a Python keyword.
    """

    model_config = ConfigDict(populate_by_name=True)

    target: TargetSpec | None = None
    from_: Literal["text", "value", "attribute", "table_cell"] = Field(alias="from")
    attribute: str | None = None
    table: TableSelector | None = None
    parse: ParseSpec | None = None
    output: str


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


class Risk(_Model):
    """A step's blast radius, and how to find out if it already happened.

    `idempotency_check` is what lets a recovery ask "did this already go
    through?" before retrying a step that failed after it may have already
    taken effect. A risky (or irreversible) step with no idempotency_check
    is never retried by the replay engine -- there is no safe way to tell
    whether a retry would double the effect, so the only sound move is to
    stop and escalate.
    """

    level: Literal["safe", "risky", "irreversible"]
    reason: str | None = None
    requires_confirmation: bool = False
    idempotency_check: Condition | None = None


# --------------------------------------------------------------------------
# Waiting, checkpoints, human tasks
# --------------------------------------------------------------------------


class WaitSpec(_Model):
    until: Condition | Literal["load"] = "load"
    timeout_ms: int = 10000


class Checkpoint(_Model):
    """A condition that must hold for a step (or the whole capability) to
    count as having actually succeeded, independent of the action itself
    reporting no error."""

    condition: Condition
    description: str


class HumanTask(_Model):
    """A step the recorder captured as performed by a human, not automation
    (e.g. a step requiring a decision replay can't make on its own)."""

    instructions: str
    required_role: str = "supervisor"
    completion_check: Condition


class ExpectDialog(_Model):
    type: Literal["confirm", "alert", "prompt"]
    pattern: str | None = None
    accept: bool = True


# --------------------------------------------------------------------------
# Step
# --------------------------------------------------------------------------


class Step(_Model):
    """One recorded action (or human task) in a capability's fixed sequence."""

    id: str
    kind: Literal[
        "navigate", "click", "type", "select", "press", "extract", "assert", "wait", "human_task"
    ]
    screen: str
    intent: str
    origin: Literal["agent", "human"] = "agent"
    target: TargetSpec | None = None
    value: str | None = None
    expect_dialog: ExpectDialog | None = None
    wait: WaitSpec
    checkpoint: Checkpoint | None = None
    extract: Extraction | None = None
    risk: Risk
    human_task: HumanTask | None = None

    @model_validator(mode="after")
    def _kind_requirements(self) -> Step:
        if self.kind == "human_task" and self.human_task is None:
            raise ValueError("kind == 'human_task' requires human_task")
        if self.kind == "extract" and self.extract is None:
            raise ValueError("kind == 'extract' requires extract")
        if self.kind in ("click", "type", "select", "press") and self.target is None:
            raise ValueError(f"kind == {self.kind!r} requires target")
        if self.kind == "navigate" and self.value is None:
            raise ValueError("kind == 'navigate' requires value")
        return self


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------


class Handling(_Model):
    action: Literal["stop", "dismiss_dialog", "reload", "click", "escalate"]
    target: TargetSpec | None = None
    max_attempts: int = 2

    @model_validator(mode="after")
    def _click_requires_target(self) -> Handling:
        if self.action == "click" and self.target is None:
            raise ValueError("handling.action == 'click' requires target")
        return self


class Outcome(_Model):
    """A runtime condition replay must be able to recognise and react to.

    `origin` records where this outcome came from: `discovered` (the model
    hit it during the recording run itself), `library` (merged in from a
    curated per-app outcomes library), or `reviewer` (hand-authored). A
    single happy-path discovery run cannot observe things like "record not
    found" or "system busy" -- they simply don't happen on the path that was
    recorded -- so most of an artifact's outcomes are necessarily merged in
    from elsewhere. Recording provenance per outcome keeps that honest
    instead of implying every outcome was empirically observed.
    """

    code: str
    kind: Literal["business", "transient", "interstitial", "auth", "requires_human", "app_error"]
    origin: Literal["discovered", "library", "reviewer"]
    scope: Literal["global"] | list[str] = "global"
    detect: Condition
    message: str
    handling: Handling
    result_outputs: dict[str, Extraction] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------


class Screen(_Model):
    """A named, recognisable state of the app, used to detect drift.

    `url_pattern` and `landmarks` are evaluated against the frame that
    `frame` selects, never the top document. In a frameset (nav + main, as
    in the mock app), the top-level URL never changes across login, lookup,
    and member detail -- only the inner frame's URL does. Matching the top
    document would make `UNEXPECTED_SCREEN` structurally unfireable, since
    every screen would look identical from that vantage point.
    """

    id: str
    description: str
    frame: FrameMatcher
    url_pattern: str | None = None
    landmarks: list[Condition]


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


class InputSpec(_Model):
    type: Literal["string", "integer", "enum"]
    required: bool = True
    description: str
    sensitive: bool = False
    source: Literal["caller", "vault"] = "caller"
    pattern: str | None = None
    enum: list[str] | None = None
    example: str | None = None


class OutputSpec(_Model):
    type: Literal["string", "integer", "money", "date"]
    description: str
    source_step: str | None = None


class Contract(_Model):
    """The agent-callable surface of a capability: what it needs, what it
    returns, what it can end in, and what kind of effect it has on the app."""

    inputs: dict[str, InputSpec]
    outputs: dict[str, OutputSpec]
    outcome_codes: list[str]
    side_effects: Literal["none", "reads", "creates_record", "modifies"]
    idempotent: bool
    preconditions: list[Condition] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Target application
# --------------------------------------------------------------------------


class AppInfo(_Model):
    vendor: str
    product: str
    version: str


class TargetApp(_Model):
    surface: Literal["web"] = "web"
    app: AppInfo
    entry_url: str
    fingerprint: list[Condition] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Approval and provenance
# --------------------------------------------------------------------------


class Approval(_Model):
    status: Literal["draft", "approved", "deprecated"] = "draft"
    approved_by: str | None = None
    approved_at: str | None = None
    approved_hash: str | None = None


class Provenance(_Model):
    """How this artifact came to exist, for audit and for reproducing a discovery run."""

    source_goal: str
    discovery_run_id: str
    model: str
    prompt_version: str
    recorded_at: str = Field(default_factory=utcnow_iso)
    recorded_against: dict[str, Any]
    evidence_ref: str | None = None
    raw_artifact_ref: str | None = None


class CapabilityPolicy(_Model):
    """The safety envelope a capability is willing to operate inside; the
    effective policy at replay time is this narrowed against the deployment
    policy (narrowest wins), enforced outside this module."""

    allowed_domains: list[str]
    allowed_actions: list[str]
    allow_coordinate_fallback: bool = False


# --------------------------------------------------------------------------
# Capability
# --------------------------------------------------------------------------

_BEHAVIOURAL_FIELDS = ("steps", "outcomes", "screens", "success", "contract", "target")

_INPUT_JSON_TYPE: dict[str, str] = {"string": "string", "integer": "integer", "enum": "string"}
_OUTPUT_JSON_TYPE: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "money": "string",
    "date": "string",
}


class Capability(_Model):
    """A recorded, typed, replayable UI flow.

    This is the artifact a discovery run produces and a replay run consumes
    with no model in the loop. `compute_content_hash` / `with_content_hash`
    / `is_effectively_approved` exist to make "was this exact behaviour
    reviewed" a checkable fact rather than a matter of trusting the
    `approval.status` field in isolation.
    """

    schema_version: Literal["1.0"] = "1.0"
    id: str
    name: str
    version: str = "1.0.0"
    description: str
    content_hash: str | None = None
    approval: Approval
    contract: Contract
    target: TargetApp
    screens: dict[str, Screen]
    steps: list[Step]
    outcomes: list[Outcome]
    success: Checkpoint
    policy: CapabilityPolicy
    provenance: Provenance

    def compute_content_hash(self) -> str:
        """SHA-256 over the capability's *behavioural* content only.

        Excludes `content_hash` itself, `approval`, and `provenance` so that
        re-approving a capability, or re-recording where/when it was
        discovered, never changes the hash an approval is pinned to. Only a
        change to steps, outcomes, screens, success, contract, or target --
        the things that actually change what replay does -- should.
        """
        full = self.model_dump(mode="json", by_alias=True)
        payload = {key: full[key] for key in _BEHAVIOURAL_FIELDS}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def with_content_hash(self) -> Capability:
        """Return a copy with `content_hash` set to `compute_content_hash()`."""
        return self.model_copy(update={"content_hash": self.compute_content_hash()})

    def is_effectively_approved(self) -> bool:
        """True only if `approval.status == "approved"` AND the approval is
        pinned to the *current* content hash.

        Editing an approved artifact (changing a step's target, adding an
        outcome, ...) doesn't touch `approval.status` -- nothing forces that
        to happen. Comparing the pinned hash against what the content
        actually hashes to now is what silently demotes an edited-but-still-
        marked-approved artifact back to effectively-draft, instead of
        letting a stale approval keep authorizing risky replays.
        """
        return (
            self.approval.status == "approved"
            and self.approval.approved_hash is not None
            and self.approval.approved_hash == self.compute_content_hash()
        )

    def to_tool_definition(self) -> dict[str, Any]:
        """The agent-callable contract for this capability: name, description,
        a strict JSON-Schema input shape, and the outputs/outcomes/effects an
        LLM (or an orchestrator) needs to decide whether and how to call it.

        Sensitive inputs are still listed -- an LLM planning a call needs to
        know the input exists -- but their description says the value comes
        from a vault and is never echoed, so nothing here invites the model
        to ask the user to type a secret into the conversation.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, spec in self.contract.inputs.items():
            prop: dict[str, Any] = {"type": _INPUT_JSON_TYPE[spec.type]}
            prop["description"] = (
                f"{spec.description} (supplied from a vault; never echoed)"
                if spec.sensitive
                else spec.description
            )
            if spec.pattern is not None:
                prop["pattern"] = spec.pattern
            if spec.enum is not None:
                prop["enum"] = spec.enum
            if spec.example is not None:
                prop["examples"] = [spec.example]
            properties[name] = prop
            if spec.required:
                required.append(name)

        outputs = {
            name: {"type": _OUTPUT_JSON_TYPE[spec.type], "description": spec.description}
            for name, spec in self.contract.outputs.items()
        }

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "outputs": outputs,
            "outcome_codes": list(self.contract.outcome_codes),
            "side_effects": self.contract.side_effects,
            "idempotent": self.contract.idempotent,
        }

    def validate_semantics(self) -> list[str]:
        """Cross-reference checks that field-level typing can't express.

        Returns a list of human-readable problems; an empty list means
        clean. This is a plain method, not a pydantic validator, because
        these checks are about consistency *across* the artifact (does this
        step's screen exist, does this placeholder name a declared input),
        which only makes sense once the whole object graph exists.
        """
        problems: list[str] = []
        step_ids = {s.id for s in self.steps}
        screen_ids = set(self.screens.keys())
        declared_inputs = set(self.contract.inputs.keys())
        declared_outputs = set(self.contract.outputs.keys())

        for step in self.steps:
            if step.screen not in screen_ids:
                problems.append(f"step {step.id!r} references unknown screen {step.screen!r}")
            if step.value:
                for placeholder in re.findall(r"\{\{(\w+)\}\}", step.value):
                    if placeholder not in declared_inputs:
                        problems.append(
                            f"step {step.id!r} value references undeclared input "
                            f"{{{{{placeholder}}}}}"
                        )
            if step.extract is not None and step.extract.output not in declared_outputs:
                problems.append(
                    f"step {step.id!r} extract.output {step.extract.output!r} "
                    "not declared in contract.outputs"
                )
            if step.risk.level != "safe" and not step.risk.reason:
                problems.append(
                    f"step {step.id!r} has risk.level={step.risk.level!r} but no risk.reason"
                )

        for outcome in self.outcomes:
            if isinstance(outcome.scope, list):
                for step_id in outcome.scope:
                    if step_id not in step_ids:
                        problems.append(
                            f"outcome {outcome.code!r} scope references unknown step {step_id!r}"
                        )

        declared_codes = set(self.contract.outcome_codes)
        real_codes = {o.code for o in self.outcomes}
        for code in sorted(declared_codes - real_codes):
            problems.append(f"contract.outcome_codes lists {code!r} but no outcome has that code")
        for code in sorted(real_codes - declared_codes):
            problems.append(f"outcome {code!r} is not listed in contract.outcome_codes")

        return problems
