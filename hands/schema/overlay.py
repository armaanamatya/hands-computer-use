"""Per-tenant capability overlays.

Hundreds of tenants can run the same vendor product (same login screen,
same member-lookup flow) with small per-deployment differences: a renamed
button, a narrower confirm-dialog wording, a smaller set of allowed
domains. An overlay lets one recorded, reviewed `Capability` be specialised
per tenant instead of re-discovering and re-approving the whole flow from
scratch for each one. The merge rules enforced by `apply_overlay` are what
keep that specialisation from becoming a backdoor: a tenant overlay can
retarget *how* a step finds its element or *which* screen it's looking at,
but it cannot change *what* the capability does (step kinds/ids, its
input/output contract, its outcome codes), and it cannot make the
capability less safe (lower a risk level, widen what it's allowed to touch).
"""
from __future__ import annotations

import copy
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from hands.schema.capability import Capability

_RISK_RANK: dict[str, int] = {"safe": 0, "risky": 1, "irreversible": 2}

# Path prefixes an overlay patch is allowed to touch. Anything else is a
# structural or contractual change and must go through re-recording /
# re-approval, not a lightweight per-tenant overlay.
#   - screens.*            any part of any screen (id, description, frame,
#                           url_pattern, landmarks) -- screens are exactly
#                           the per-deployment detection heuristics tenants
#                           legitimately differ on.
#   - steps[id].target.*   how a step relocates its element.
#   - steps[id].risk.*     a tenant's own risk assessment may only ever
#                           raise a step's risk (enforced in
#                           _check_invariants below), never lower it.
#   - target.entry_url     the tenant's own base URL for this app.
#   - policy.*             narrowed in _check_invariants below.


class OverlayViolation(Exception):
    """Raised when a patch would touch something an overlay may not change,
    would weaken safety (lower a risk level, widen a policy allowlist), or
    would introduce a new semantic problem (e.g. orphan a step's screen
    reference) that `Capability.validate_semantics()` did not already have."""


class PatchOp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    op: Literal["replace", "prepend"]
    value: Any


class CapabilityOverlay(BaseModel):
    """A tenant-scoped set of patches against one base capability, identified
    by id and version so an overlay can't silently apply to the wrong
    artifact after the base is re-recorded."""

    model_config = ConfigDict(extra="forbid")

    base_id: str
    base_version: str
    tenant_id: str
    description: str
    patches: list[PatchOp]


def _split_path(path: str) -> list[str]:
    """Turn `"steps[s3].target.strategies"` into `["steps", "s3", "target",
    "strategies"]`. A bracketed segment (`[s3]`) is a lookup by the `id`
    field of an element in a list, not a positional index -- step order is
    behavioural and an overlay must never depend on it."""
    tokens: list[str] = []
    for part in path.split("."):
        m = re.match(r"^(\w+)((?:\[[^\]]+\])*)$", part)
        if not m:
            raise OverlayViolation(f"malformed patch path segment {part!r} in {path!r}")
        name, brackets = m.groups()
        tokens.append(name)
        tokens.extend(re.findall(r"\[([^\]]+)\]", brackets))
    if not tokens:
        raise OverlayViolation(f"empty patch path {path!r}")
    return tokens


def _check_path_allowed(tokens: list[str], path: str) -> None:
    head = tokens[0]
    if head == "screens":
        return
    if head == "policy":
        return
    if head == "target" and len(tokens) >= 2 and tokens[1] == "entry_url":
        return
    if head == "steps" and len(tokens) >= 3 and tokens[2] in ("target", "risk"):
        return
    raise OverlayViolation(f"overlay may not modify path {path!r}")


def _navigate(dump: dict[str, Any], tokens: list[str], path: str) -> tuple[dict[str, Any], str]:
    """Walk `tokens` into `dump`, returning `(container, key)` for the final
    segment so the caller can set `container[key]`."""
    current: Any = dump
    for token in tokens[:-1]:
        if isinstance(current, list):
            match = next((item for item in current if isinstance(item, dict) and item.get("id") == token), None)
            if match is None:
                raise OverlayViolation(f"no element with id {token!r} while applying path {path!r}")
            current = match
        elif isinstance(current, dict):
            if token not in current:
                raise OverlayViolation(f"unknown path segment {token!r} in {path!r}")
            current = current[token]
        else:
            raise OverlayViolation(
                f"cannot navigate into {type(current).__name__} at {token!r} in {path!r}"
            )
    last = tokens[-1]
    if not isinstance(current, dict):
        raise OverlayViolation(f"path {path!r} must address an object field")
    return current, last


def _apply_op(container: dict[str, Any], key: str, patch: PatchOp) -> None:
    if patch.op == "replace":
        container[key] = patch.value
        return
    if patch.op == "prepend":
        existing = container.get(key)
        if not isinstance(existing, list):
            raise OverlayViolation(
                f"prepend requires a list at {patch.path!r}, found {type(existing).__name__}"
            )
        container[key] = [patch.value] + existing
        return
    raise OverlayViolation(f"unknown patch op {patch.op!r}")  # pragma: no cover - Literal-enforced


def _check_invariants(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Post-hoc structural diff, independent of which paths were touched --
    defense in depth on top of `_check_path_allowed`."""
    before_steps = {s["id"]: s for s in before["steps"]}
    after_steps = {s["id"]: s for s in after["steps"]}
    if set(before_steps) != set(after_steps):
        raise OverlayViolation("overlay may not add or remove steps")
    if [s["id"] for s in before["steps"]] != [s["id"] for s in after["steps"]]:
        raise OverlayViolation("overlay may not reorder steps")

    for step_id, before_step in before_steps.items():
        after_step = after_steps[step_id]
        if before_step["kind"] != after_step["kind"]:
            raise OverlayViolation(f"overlay may not change step {step_id!r} kind")
        before_rank = _RISK_RANK[before_step["risk"]["level"]]
        after_rank = _RISK_RANK[after_step["risk"]["level"]]
        if after_rank < before_rank:
            raise OverlayViolation(f"overlay may not lower risk on step {step_id!r}")

    if before["contract"]["inputs"] != after["contract"]["inputs"]:
        raise OverlayViolation("overlay may not change contract inputs")
    if before["contract"]["outputs"] != after["contract"]["outputs"]:
        raise OverlayViolation("overlay may not change contract outputs")
    if set(before["contract"]["outcome_codes"]) != set(after["contract"]["outcome_codes"]):
        raise OverlayViolation("overlay may not change outcome codes")

    before_domains = set(before["policy"]["allowed_domains"])
    after_domains = set(after["policy"]["allowed_domains"])
    if not after_domains <= before_domains:
        raise OverlayViolation("overlay may not widen policy.allowed_domains")

    before_actions = set(before["policy"]["allowed_actions"])
    after_actions = set(after["policy"]["allowed_actions"])
    if not after_actions <= before_actions:
        raise OverlayViolation("overlay may not widen policy.allowed_actions")


def apply_overlay(cap: Capability, overlay: CapabilityOverlay) -> Capability:
    """Apply a tenant overlay to a base capability and return a new,
    re-validated `Capability`.

    Each patch's `path` is dotted/bracket navigation over the base
    capability dumped to a dict (e.g. `screens.lookup.landmarks`, or
    `steps[s3].target.strategies` where `s3` is a step **id**, not an
    index). After all patches are applied the result is re-validated as a
    `Capability` and checked against the merge rules documented on this
    module; any breach raises `OverlayViolation` and the base capability is
    left untouched.
    """
    if overlay.base_id != cap.id or overlay.base_version != cap.version:
        raise OverlayViolation(
            f"overlay targets {overlay.base_id!r}@{overlay.base_version!r}, "
            f"not {cap.id!r}@{cap.version!r}"
        )

    before = cap.model_dump(mode="json", by_alias=True)
    after = copy.deepcopy(before)

    for patch in overlay.patches:
        tokens = _split_path(patch.path)
        _check_path_allowed(tokens, patch.path)
        container, key = _navigate(after, tokens, patch.path)
        _apply_op(container, key, patch)

    new_cap = Capability.model_validate(after)

    _check_invariants(before, after)

    # The overlay may not introduce semantic problems that weren't already
    # present in the base artifact (e.g. renaming a screen id out from
    # under a step that still points at the old name).
    before_problems = set(cap.validate_semantics())
    after_problems = set(new_cap.validate_semantics())
    new_problems = after_problems - before_problems
    if new_problems:
        raise OverlayViolation(
            "overlay introduces semantic problems: " + "; ".join(sorted(new_problems))
        )

    return new_cap
