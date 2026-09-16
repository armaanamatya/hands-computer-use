r"""Safety/guardrail gate for computer-use actions.

Both the agent loop and the replay engine call `Policy.check_action` (or its
raising twin `assert_action`) before every action reaches a `Surface`. This
module has three jobs, in order:

1. Load and honour every key in `policy.yaml` (the operator's ceiling on
   what the automation may ever do).
2. Compute the intersection of that ceiling with a capability's own,
   narrower `policy` block (see `Policy.intersect`) — a capability can only
   ever *restrict* what the operator permits, never grant something the
   operator did not already allow.
3. Gate a single action: is the action kind allowed, is the target URL
   allowed, is a coordinate-fallback locator allowed, and is the action
   "risky" (an action with an irreversible side effect on the institution's
   records) under the configured `risky_mode`.

Design notes a reviewer will ask about:

* `Policy.load`/`Policy.from_dict` call `yaml.safe_load` and validate; they
  do not repair or reinterpret a malformed config. A loader that "fixes"
  its own policy file cannot be trusted to notice when a fix silently
  changes behaviour. Case in point: this project's actual `policy.yaml`
  currently has two YAML defects in its double-quoted regex strings, and
  they are not equally dangerous. `"^http://(127\.0\.0\.1|...)"` on the
  `allowed_url_patterns` line uses `\.`, which is not a legal YAML
  double-quoted escape, so `yaml.safe_load` raises immediately — this one
  is the *benign* defect precisely because it is loud: nobody ships it.
  The `risky_patterns` line uses `\b`, which *is* a legal YAML
  double-quoted escape (backspace, U+0008): it parses without error into
  a regex containing literal backspace characters where the author meant
  word-boundary anchors, and would silently classify "Confirm"/
  "Transfer"/"Delete" clicks as safe forever. THIS is the one that
  matters: fixing only line 9 (to stop the crash) without also fixing the
  `\b` line ships the dangerous defect. Both need single-quoting (where
  backslash is literal in YAML) — that fix belongs in `policy.yaml`, not
  in this loader. A gate that fails loud on a bad config is safe; a gate
  that repairs its own config, or one that merely doesn't crash, is not a
  gate.
* Every check has a single, named `rule` string so a caller (and the audit
  log) can tell exactly which line of defence fired, and `Decision.detail`
  is always human-readable.
* `check_url` never raises on a malformed URL; a URL the parser chokes on
  is treated as untrusted input and denied, not an exception the caller
  must remember to catch.
* An unrecognized `risky_mode` fails closed twice over: `from_dict` raises
  at load time (same "don't guess" stance as the YAML handling above), and
  `check_action` also refuses risky actions at the point of use as
  defence-in-depth, in case a `Policy` is ever hand-constructed outside
  `load`/`from_dict` with a bad value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

# Action kinds classify_risk treats as having no irreversible side effect by
# virtue of their kind alone. `wait`/`extract`/`assert` read or pause, never
# mutate. `navigate` is included here too, but only for classify_risk's
# purposes: whether a navigation is *safe* depends on *where* it goes, and
# that is check_url's job (called separately by check_action), not a
# pattern match against an element name.
_SAFE_BY_KIND = frozenset({"wait", "extract", "assert", "navigate"})

# Action kinds for which classify_risk actually inspects the target: only
# these can trigger an irreversible, hard-to-undo effect on the underlying
# application's records (submitting a form, pressing Enter on a submit
# button, clicking "Delete"). Typing text is reversible — it can be
# overwritten or the field cleared before anything is submitted — so `type`
# is never risk-checked against risky_patterns.
_RISK_CHECKED_KINDS = frozenset({"click", "press"})

_VALID_RISKY_MODES = ("gate", "block", "allow")


class PolicyViolation(Exception):
    """Raised by `Policy.assert_action` when an action is not allowed.

    `rule` is the short, machine-readable name of the check that failed
    (e.g. "domain", "action_kind", "risky_action"); `detail` is the
    human-readable explanation. Mirrors `Decision.rule`/`Decision.detail`
    for the denial that caused the raise.
    """

    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"{rule}: {detail}")


@dataclass(frozen=True)
class Decision:
    """The outcome of a single policy check.

    `allowed=False` always carries a non-None `rule` and `detail` naming
    exactly why. `allowed=True` may still set `requires_confirmation=True`
    (risky_mode "gate"): the action is permitted to proceed only after a
    human confirms it — it is not yet cleared to execute unattended.
    """

    allowed: bool
    rule: str | None = None
    detail: str | None = None
    requires_confirmation: bool = False


@dataclass(frozen=True)
class Policy:
    """An operator's safety policy, loaded from `policy.yaml` (or narrowed
    from one via `intersect`). See the module docstring for the loading
    and risk-classification rules this class enforces.
    """

    allowed_domains: frozenset[str]
    allowed_url_patterns: tuple[re.Pattern[str], ...]
    allowed_actions: frozenset[str]
    risky_patterns: tuple[re.Pattern[str], ...]
    risky_mode: str
    allow_coordinate_fallback: bool
    capture_human_input_values: bool
    human_wait_timeout_s: int
    max_steps: int
    step_timeout_ms: int
    run_timeout_s: int
    recovery_budget: int

    # -- construction ---------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        """Load a Policy from a policy.yaml file at `path`.

        Uses `yaml.safe_load` and nothing else: a malformed file raises
        `yaml.YAMLError` straight out of the parser, and a well-formed but
        incomplete/mistyped mapping raises out of `from_dict` (KeyError for
        a missing key, ValueError for a bad `risky_mode`). See the module
        docstring for why this loader does not try to be lenient.
        """
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Policy:
        """Build a Policy from a plain mapping shaped like policy.yaml.

        Every key is required and taken literally — no defaults are
        guessed for a missing key, and no unrecognized `risky_mode` is
        accepted. `Policy.load` and `Policy.from_dict` are the only two
        ways to produce a real, usable Policy.
        """
        if not isinstance(d, dict):
            raise TypeError(f"policy config must be a mapping, got {type(d).__name__}")

        risky_mode = d["risky_mode"]
        if risky_mode not in _VALID_RISKY_MODES:
            raise ValueError(
                f"risky_mode must be one of {_VALID_RISKY_MODES}, got {risky_mode!r}"
            )

        return cls(
            allowed_domains=frozenset(str(x).lower() for x in d["allowed_domains"]),
            allowed_url_patterns=tuple(re.compile(str(p)) for p in d["allowed_url_patterns"]),
            allowed_actions=frozenset(str(x) for x in d["allowed_actions"]),
            # Always case-insensitive regardless of whether the source
            # pattern also carries an inline (?i) — belt and suspenders,
            # and the spec for this module calls for it explicitly.
            risky_patterns=tuple(
                re.compile(str(p), re.IGNORECASE) for p in d["risky_patterns"]
            ),
            risky_mode=risky_mode,
            allow_coordinate_fallback=bool(d["allow_coordinate_fallback"]),
            capture_human_input_values=bool(d["capture_human_input_values"]),
            human_wait_timeout_s=int(d["human_wait_timeout_s"]),
            max_steps=int(d["max_steps"]),
            step_timeout_ms=int(d["step_timeout_ms"]),
            run_timeout_s=int(d["run_timeout_s"]),
            recovery_budget=int(d["recovery_budget"]),
        )

    # -- narrowing --------------------------------------------------------

    def intersect(self, capability_policy: dict[str, Any] | None) -> Policy:
        """Return a NEW Policy narrowed by a capability's own policy block.

        Precedence rule: the narrower set always wins. A capability may
        declare its own `allowed_domains`, `allowed_actions`, and
        `allow_coordinate_fallback`; the result takes the SET
        INTERSECTION of `allowed_domains` and of `allowed_actions` with
        this Policy's own values, and the logical AND of
        `allow_coordinate_fallback`. Concretely: a capability can only
        ever *narrow* what this operator policy.yaml permits — it can
        never widen it. A domain or action absent from this Policy can
        never appear in the result no matter what the capability
        declares, and if either side disables coordinate fallback the
        result disables it too.

        A key that is ABSENT from `capability_policy` (or a `None`
        `capability_policy` entirely) means that dimension is not
        narrowed at all: this Policy's value passes through unchanged for
        that key. This matters — intersecting against an *empty* set
        would zero out, say, every allowed domain just because a
        capability only bothered to declare `allowed_actions`, which
        would be a much stronger (and wrong) restriction than "this
        capability didn't say anything about domains".

        All other fields (`allowed_url_patterns`, `risky_patterns`,
        `risky_mode`, and the budget/timeout fields) are operator-only:
        they are never affected by a capability's policy block.
        """
        cap = capability_policy or {}

        if "allowed_domains" in cap:
            cap_domains = {str(x).lower() for x in cap["allowed_domains"]}
            new_domains = self.allowed_domains & cap_domains
        else:
            new_domains = self.allowed_domains

        if "allowed_actions" in cap:
            cap_actions = {str(x) for x in cap["allowed_actions"]}
            new_actions = self.allowed_actions & cap_actions
        else:
            new_actions = self.allowed_actions

        if "allow_coordinate_fallback" in cap:
            new_fallback = self.allow_coordinate_fallback and bool(
                cap["allow_coordinate_fallback"]
            )
        else:
            new_fallback = self.allow_coordinate_fallback

        return replace(
            self,
            allowed_domains=frozenset(new_domains),
            allowed_actions=frozenset(new_actions),
            allow_coordinate_fallback=new_fallback,
        )

    # -- checks -----------------------------------------------------------

    def check_url(self, url: str) -> Decision:
        """Check a URL against scheme, domain, and URL-pattern rules.

        ALL of the following must hold, in this order:
          1. scheme is http or https (rejects `javascript:`, `data:`,
             `file:`, and anything else outright — rule "scheme");
          2. the hostname (port stripped, lowercased) is in
             `allowed_domains` (rule "domain");
          3. the full URL matches at least one of `allowed_url_patterns`
             (rule "url").

        Domain and pattern are deliberately two separate checks, not
        "pattern implies domain": `allowed_url_patterns` pins down
        scheme+host+port+path together (e.g. http-only, one specific
        port), while `allowed_domains` is the coarser host allowlist.
        Requiring both closes the gap where a pattern happens to be loose
        on host, or a domain entry is loose on scheme/port.

        Never raises: a URL the parser cannot make sense of (or whose
        host the parser rejects, e.g. a malformed IPv6 literal) is denied
        like any other untrusted input, not an exception the caller must
        remember to catch.
        """
        try:
            parts = urlsplit(url)
            hostname = parts.hostname  # note: never touch `.port` here — it
            # raises ValueError on a malformed port, and we don't need it:
            # port-matching is `allowed_url_patterns`'s job, not ours.
        except ValueError:
            return Decision(False, "url", f"could not parse URL: {url!r}")

        scheme = (parts.scheme or "").lower()
        if scheme not in ("http", "https"):
            return Decision(
                False, "scheme", f"scheme {scheme!r} is not http/https: {url!r}"
            )

        if not hostname:
            return Decision(False, "domain", f"URL has no hostname: {url!r}")

        hostname = hostname.lower()
        if hostname not in self.allowed_domains:
            return Decision(
                False, "domain", f"host {hostname!r} is not in allowed_domains"
            )

        if not any(p.match(url) for p in self.allowed_url_patterns):
            return Decision(
                False, "url", f"URL does not match any allowed_url_patterns: {url!r}"
            )

        return Decision(True, None, "url allowed")

    def classify_risk(
        self,
        action_kind: str,
        element_name: str | None,
        element_attrs: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """Classify an action as "safe" or "risky".

        The risky class is about *irreversible side effects on the
        institution's records* — a click that submits, confirms,
        transfers, deletes, or otherwise commits something — not about
        whether the automation might click the wrong thing or extract the
        wrong text. Reading, waiting, and typing can all be gotten wrong
        too, but they can be inspected and corrected before anything is
        committed; that is why they are "safe" here even though they are
        not risk-free in the colloquial sense.

        Rules, in order:
          1. `wait`, `extract`, `assert`, `navigate` are safe by kind
             (navigate's actual destination is checked separately by
             `check_url`, not here);
          2. `click`/`press` are risky if the element's name — or,
             failing that (name is falsy), the element's `value`
             attribute — matches any `risky_patterns` regex; the reason
             names the matched pattern;
          3. `type` is safe: typing is reversible;
          4. everything else is safe.

        Returns `(level, reason)`; `reason` is None for "safe".
        """
        if action_kind in _SAFE_BY_KIND:
            return ("safe", None)

        if action_kind in _RISK_CHECKED_KINDS:
            target = element_name or (element_attrs or {}).get("value")
            if target:
                target = str(target)
                for pattern in self.risky_patterns:
                    if pattern.search(target):
                        return (
                            "risky",
                            f"{action_kind} target {target!r} matched risky pattern "
                            f"{pattern.pattern!r}",
                        )
            return ("safe", None)

        # `type`, `select`, and any other kind not covered above: no
        # irreversible side effect, so safe by default.
        return ("safe", None)

    def check_action(
        self,
        action_kind: str,
        url: str,
        element_name: str | None = None,
        element_attrs: dict[str, Any] | None = None,
        coordinate_fallback: bool = False,
    ) -> Decision:
        """The single gate the agent loop and the replay engine call
        before every action reaches a `Surface`.

        Order of checks (first failure wins):
          1. `action_kind` is in `allowed_actions` (rule "action_kind");
          2. `check_url(url)` passes (rule "scheme"/"domain"/"url",
             propagated verbatim from `check_url`'s own Decision);
          3. if `coordinate_fallback` is True, `allow_coordinate_fallback`
             must also be True (rule "coordinate_fallback");
          4. `classify_risk(...)`: under risky_mode "block" a risky
             action is denied (rule "risky_action"); under "gate" it is
             allowed with `requires_confirmation=True`; under "allow" it
             is allowed outright.

        Always returns a Decision with a human-readable `detail`, even
        when allowed.
        """
        if action_kind not in self.allowed_actions:
            return Decision(
                False,
                "action_kind",
                f"action kind {action_kind!r} is not in allowed_actions",
            )

        url_decision = self.check_url(url)
        if not url_decision.allowed:
            return url_decision

        if coordinate_fallback and not self.allow_coordinate_fallback:
            return Decision(
                False,
                "coordinate_fallback",
                "coordinate-fallback locators are not permitted by policy",
            )

        level, reason = self.classify_risk(action_kind, element_name, element_attrs)
        if level == "risky":
            if self.risky_mode == "block":
                return Decision(False, "risky_action", reason or "action classified as risky")
            if self.risky_mode == "gate":
                return Decision(
                    True,
                    None,
                    reason or "action classified as risky; human confirmation required",
                    requires_confirmation=True,
                )
            if self.risky_mode == "allow":
                return Decision(
                    True,
                    None,
                    reason or "action classified as risky; allowed by risky_mode=allow",
                    requires_confirmation=False,
                )
            # Defence in depth: `from_dict` already rejects an unrecognized
            # risky_mode at load time, but if a Policy was ever
            # hand-constructed (bypassing load/from_dict) with a bad
            # value, fail closed rather than silently letting a risky
            # action through.
            return Decision(
                False,
                "risky_action",
                f"unrecognized risky_mode {self.risky_mode!r}; failing closed",
            )

        return Decision(True, None, "action allowed")

    def assert_action(
        self,
        action_kind: str,
        url: str,
        element_name: str | None = None,
        element_attrs: dict[str, Any] | None = None,
        coordinate_fallback: bool = False,
    ) -> Decision:
        """Same signature and checks as `check_action`, but raises
        `PolicyViolation` instead of returning a denial. A `requires_
        confirmation=True` gate result is NOT a denial (it's `allowed=
        True`) so this does not raise for it — it is returned normally,
        same as any other allowed Decision.
        """
        decision = self.check_action(
            action_kind,
            url,
            element_name=element_name,
            element_attrs=element_attrs,
            coordinate_fallback=coordinate_fallback,
        )
        if not decision.allowed:
            raise PolicyViolation(decision.rule or "policy", decision.detail or "action not allowed")
        return decision
