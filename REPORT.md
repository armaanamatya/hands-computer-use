# Design write-up

A small system that gives an AI agent hands inside a legacy application with no API: a model works out the flow once, the run is recorded as a typed capability, and that capability replays deterministically with no model in the loop. Python, Playwright, Pydantic, Anthropic or OpenAI.

## Architecture

Five parts, one process, files on disk. There are no queues, services, or databases, because none of them would make any decision in this system better; the brief explicitly says designing abstractions that *could* scale is valuable and prematurely building the scaling is not.

**Surface** is the perception and action layer. A JS walker runs in every frame and returns a flat list of visible elements with a computed role, an accessible name, a bounding box, a structural path, and table row context. It is a `Protocol`, and the element list is its own shape rather than Playwright's, because that is the seam a desktop backend implements from UI Automation or an accessibility API.

**Discovery** is an LLM tool-use loop over that surface. I wrote the loop by hand rather than using an SDK helper because a policy gate has to run before every action, every turn has to be recorded as evidence, and the loop has to be able to stop mid-flight and hand the browser to a human. None of those fit inside a helper that owns the loop.

**Recorder** turns a successful trace into a `Capability`. **Replay** executes a `Capability` with no model. **Handoff** owns the control-transfer state machine. Safety and evidence are cross-cutting and are called by everything.

The seam that matters is between *how we perceive and act on a surface* and *what the recorded flow says*. A capability never refers to a Playwright locator. It refers to a frame matcher, an ordered list of targeting strategies, and conditions over an observation. Anything that can produce that observation shape can execute one.

## Artifact schema

A capability is a contract, not a step list. The parts I would defend hardest:

**`contract`** is what a calling agent needs: typed inputs (each marked `sensitive` and `caller` or `vault`), typed outputs, `side_effects`, `idempotent`, and the `outcome_codes` the caller must handle. `to_tool_definition()` emits it as a JSON-Schema tool definition, so a capability is directly callable. Publishing the outcome codes is deliberate: `MEMBER_NOT_FOUND` is part of the interface, not an exception.

**`screens`** is a first-class map and every step names one. A screen's URL pattern and landmarks are evaluated against the frame its matcher selects, never the top document. In a frameset the top URL never changes, so a screen keyed on it can never fail to match and "wrong screen" becomes undetectable. Making screens explicit also turns a wrong screen into its own debuggable category instead of the misleading "target not found".

**Targets** carry ordered strategies, each with its own `rationale`: role and accessible name first, then text, then a row-relative strategy for legacy tables ("the control in the row whose type cell says SAV"), then a structural path, then a coordinate box gated by policy. Each must match exactly one element or the next is tried, and the rank that succeeded is reported as a drift signal.

**`risk`** is an object, not a flag. `idempotency_check` is load-bearing: it is how a recovery asks "did this already go through?" before retrying, and a risky step without one is never retried.

**`Outcome.origin`** records whether a condition was discovered by the model, merged from a curated per-app library, or added by a reviewer. This matters more than it looks. A single happy-path discovery run cannot observe "record not found" or "system busy", so claiming the model discovered them would be false. Outcomes are a per-app library the recorder merges and lints, and the artifact says so.

**`content_hash`** covers behaviour only and excludes approval and provenance, so re-approving does not change it and editing an approved artifact silently demotes it to draft.

## Determinism & error handling

Replay verifies rather than assumes. Per step: check the screen against its own frame, resolve the target recording every strategy attempt, run the policy gate against the **live** element name so a control since relabelled "Delete" is caught, execute, then poll.

The poll order is the load-bearing decision: step-scoped outcomes, then the checkpoint, then global outcomes. A step-scoped business outcome must beat a loosely-satisfied checkpoint, otherwise "account restricted" is masked by a checkpoint that merely matched leftover page text. When nothing matches before the timeout, the failure is *classified* rather than reported as a timeout: unknown screen, application error, unexpected dialog, or failed checkpoint.

The result contract separates three things the brief warns about conflating:

| Member | Status | Detail |
|---|---|---|
| 12345 | `SUCCESS` | `savings_balance` 2340.55, exit 0 |
| 99999 | `BUSINESS_OUTCOME` | `MEMBER_NOT_FOUND`, exit 0 |
| 40404 | `SUCCESS` | one step `recovered` after a "system busy" interstitial |
| 50000 | `FAILED` | `APP_ERROR`, exit 1, with screenshot and DOM snapshot |

"No such member" exits 0 at the shell. It is an answer the caller asked for.

Recovery never blindly re-runs a step: it evaluates the idempotency check first, and a risky step with no idempotency signal escalates instead of retrying. A risky step that fails *after* its action fired sets `side_effects_uncertain`, which is the flag an operations team reconciles against.

Two linting passes keep recorded artifacts honest, both pure functions over recorded observations. A checkpoint must be false before its action and true after; one that was already true proves nothing and would let replay "verify" a click that never happened. An outcome detector must not match the success page, or a loose "not found" pattern silently shadows a successful run.

The most useful thing I built was the test that replays an artifact the **recorder itself produced**. Until then, discovery and replay were each tested against the other's stand-in. That join found four real defects, and one is worth stating plainly: the recorder chose screen landmarks by finding text unique to each screen in the discovery run, and on the member detail screen the most distinctive text was the member's own data. It landmarked that screen on `LN`, the loan row of the one member used during discovery, so the artifact replayed perfectly for member 12345 and rejected the member screen for every member without a loan account. That is exactly the difference between a recording and a capability. Screens are now identified by the work frame's document title, which the application chose to name the screen and which does not vary by record.

## Heterogeneity & multi-tenant

**Other surfaces.** The element list, not the DOM, is the contract. `role`, `text`, and `relative` survive on a desktop surface because UI Automation and the macOS accessibility API expose the same role/name/bounds triple; `bbox` survives anywhere a screenshot does, which is the true no-DOM fallback. `structural` and the frame matchers are web-only, and I would expect a desktop backend to supply a window/control-tree path in their place. I am scoping that claim deliberately rather than asserting the whole design is surface-agnostic.

**Many tenants, one vendor product.** Re-recording per tenant does not scale to hundreds of institutions, so a capability is a base and a `CapabilityOverlay` specialises it per tenant. The merge rules are the interesting part: an overlay may retarget, re-fingerprint, and change URL patterns, but may only *raise* risk and only *narrow* the allowlist, and may never change step kinds or the contract. A tenant can adapt a capability to its branding; it cannot weaken its safety envelope or silently change what the capability returns to a calling agent.

**Drift** is observable rather than assumed. Every run reports which strategy rank succeeded, whether the coordinate fallback was used, and whether the app fingerprint matched. A tenant whose replays quietly start succeeding at rank 3 is drifting and can be re-recorded before it breaks. Overlays are implemented and tested as a type with enforced merge rules; I did not build a tenant registry, and a second tenant variant of the mock app is not shipped.

One known limit: an overlay can patch a `Screen.id` field without being rejected. It is not exploitable, because everything resolves screens by the map key rather than that field, but it is unchecked.

## Escalation & handoff

The session is never recreated. One process owns the browser for the whole run; what a handoff changes is *who may command it*. That is enforced with a token carrying a holder and an epoch, and the epoch increments on every transfer, so a step issued before a handoff cannot land after it. Its token is stale and the surface refuses it. A test asserts exactly that, because it is the invariant the design exists to protect. `guard()` also checks the controller's live state, so "who is in control" has one authority rather than two that can disagree.

State is written to `control_state.json` on every transition, so that question is answerable from disk after a crash, not only from memory.

Escalation is triggered by a model asking for help, a replay hitting a `requires_human` outcome, an unresolvable step, or a risky step on an unapproved capability. It raises an `InterventionRequest` carrying the reason code, the step and its phase (before or after the action fired, since the human's job differs if Submit was already clicked), what was expected, what was observed, and the evidence paths.

Resuming re-verifies rather than trusting: if the capability's success condition now holds the run finishes, if the step's checkpoint holds the step is marked `human_performed`, otherwise the step is retried or control goes back to a human. One mapping is load-bearing: when a human has just performed a risky step by hand, resume must mark it performed rather than retry it, or the same irreversible action is submitted twice.

Two constraints shaped the implementation and are verified, not assumed. The operator HTTP server runs on its own thread and never touches Playwright, whose sync objects are thread-bound; it enqueues commands the main loop drains. The wait loop pumps Playwright rather than sleeping, because the injected recorder delivers events only while the main thread is inside a Playwright call.

What the human did is recorded with the same target descriptor shape perception produces, so a human-performed step is describable like an agent-performed one. Values are withheld by default and a password field reports none regardless. The operator console is a minimal HTTP surface plus a scripted stand-in operator; a real one would add live co-browsing and a queue. The mechanism, not the console, is what I built.

## Safety

The allowlist is data, not code, because it is the kind of thing an institution's security team must review and change without touching Python. Effective policy is `policy.yaml` intersected with the capability's own block, so a capability can only ever narrow what the operator permits. Non-HTTP schemes are rejected outright, which matters here because the menu frame contains `javascript:` links.

Risk is about irreversible effects on the institution's records, not about whether the automation might be wrong. Typing is reversible and stays safe; clicking a control named "Confirm" or "Open Sub-Account" is risky. Under the default `gate` mode a risky step on an unapproved capability routes to a human rather than proceeding.

Secrets are handled in two layers. Sensitive values are never shown to the model, which sees only a placeholder name and a description; and the recorder refuses to serialize a literal that matches a secret, raising rather than writing it. Evidence is redacted on write rather than trusting callers to remember.

The best thing in this section is a bug I wrote and then found. `policy.yaml` quoted its regexes with double quotes. That made the file fail to parse, and worse, once the obvious break was fixed the `\b` word-boundary anchors in the risky-action patterns would have parsed as backspace characters. The file would load without error and nothing would ever be classified risky again: a guardrail failing open, silently. It is fixed, and there is a test that loads the real file and asserts "Confirm" still classifies as risky. Quoting is load-bearing for safety here, which is not obvious until it bites.

A second bug came from adding a second model provider. The `done` tool declared its outputs as an open-ended map, which can never satisfy strict function calling, so a live OpenAI run would have been rejected at the final step. Supporting two providers is what surfaced it; there is now an audit asserting every tool schema stays strict-clean.

## Cuts

Everything here is a deliberate cut with a documented reason, not an oversight.

- **Session-expiry re-login.** Detected as an `auth` outcome and escalated. Recovering it means running a sub-flow inside an outcome handler with its own restart and idempotency semantics, which is a second interpreter. *Next:* outcome handling that can invoke another capability by reference.
- **Operator preemption, resume-from-arbitrary-step, promoting recorded human actions into replayable steps, claim-expiry timers.** Each is a real feature; together they are most of an operator product. The state machine has the seams. *Next:* preemption first, since it is a single check at a step boundary.
- **A real operator console.** A minimal HTTP surface and a scripted operator stand in. *Next:* live screen streaming and a work queue.
- **Desktop surface.** Designed for, not built. *Next:* a UI Automation backend producing the same element list.
- **Multi-tenant registry and a second tenant variant.** Overlays are implemented and tested; the infrastructure around them is exactly the premature scaling the brief warns against. *Next:* a tenant variant of the mock app to prove the overlay against real divergence.
- **Approval workflow.** `approval` is a field with real semantics (editing an approved artifact demotes it to draft) but there is no review UI. *Next:* an `approve` command gated on a passing replay.
- **Capability health statistics** (strategy hit rates over time). Drift signals are emitted per run but never aggregated. *Next:* persist them and alert on rank drift.
- **A dry-run boundary** so a replay can stop before the first irreversible step. Cheap to add; I ran out of time box before need.

The largest outstanding item is not a cut: the brief requires at least one genuinely model-driven discovery run with evidence, and that needs an API key this machine does not have. Everything else runs without one, including a scripted-provider discovery run that exercises the identical code path. See `evidence/discovery/PENDING.md` for the exact command.
