# hands

Gives an AI agent hands inside a legacy application that has no API.

A model works out how to do a task once, by driving a real UI the way a person would. The successful run is recorded as a typed, versioned **capability**. From then on that capability replays deterministically with no model in the loop, verifies every step, returns typed outputs, and hands the live session to a human when it cannot safely continue.

```
  discover               record                replay                  result
  a model drives   --->  a capability    --->  no model; every   --->  typed outputs, or a
  the real UI once       JSON on disk          step is verified        declared outcome, or
                                                     |                 a classified failure
                                                     v
                                               handoff: a human takes over the same
                                               live browser session, then hands it back
```

Read next:

- **[REPORT.md](REPORT.md)**: design decisions, trade-offs, and the bugs the design caught.
- **[evidence/README.md](evidence/README.md)**: real runs, one directory per outcome class, exactly as the system wrote them.
- **[capabilities/lookup_savings_balance.json](capabilities/lookup_savings_balance.json)**: what a capability looks like.

## Contents

- [Quick start](#quick-start)
- [What a run returns](#what-a-run-returns)
- [The target application](#the-target-application)
- [Walkthrough](#walkthrough)
- [Example run](#example-run)
- [Inside a capability](#inside-a-capability)
- [Safety](#safety)
- [Tests](#tests)
- [Layout](#layout)
- [Status and known gaps](#status-and-known-gaps)
- [Troubleshooting](#troubleshooting)

## Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). No API key is needed for anything below.

```bash
uv sync
uv run playwright install chromium
cp .env.example .env               # fake credentials for the mock app; no real secrets
```

Start the target application in one terminal and leave it running:

```bash
uv run hands serve-mock            # LegacyCore on http://127.0.0.1:5050
```

In a second terminal, replay the saved capability in a real browser session:

```bash
uv run hands replay capabilities/lookup_savings_balance.json \
  --input username=teller1 \
  --input password=@env:MOCKBANK_PASSWORD \
  --input member_id=12345
```

```
run_id: replay-20261022T005647Z-cb180e
capability: legacycore.lookup_savings_balance v1.0.0
status: SUCCESS
outputs:
  savings_balance: 2340.55
outputs_complete: True
steps:
  s1     ok
  s2     ok               strategy=relative rank=0
  s3     ok               strategy=relative rank=0
  s4     ok               strategy=role rank=0
  s5     ok               strategy=relative rank=0
  s6     ok               strategy=role rank=0
  s7     ok
evidence dir: .../evidence/runs/replay-20261022T005647Z-cb180e
```

That signed on, navigated a frameset, picked the savings row out of a multi-row accounts table, and extracted a typed value, with no model involved. Every step reports which targeting strategy resolved it and at what rank, which is the drift signal. The evidence directory holds a screenshot before every action, a structured event log, and the result as JSON.

`@env:` reads a value from the environment (including `.env`, which every command loads) so a secret never appears on the command line or in shell history. An input named `password` is always treated as sensitive. On Windows, set `PYTHONUTF8=1` if your console is not already UTF-8.

## What a run returns

Replay separates three things that are easy to conflate: the task succeeded, the application gave a legitimate answer that is not the happy path, and something actually broke. The shell exit code follows the same split.

| Status | Meaning | Exit |
|---|---|---|
| `SUCCESS` | Outputs captured and the success condition verified | 0 |
| `BUSINESS_OUTCOME` | The application answered with a declared outcome such as `MEMBER_NOT_FOUND` or `ACCOUNT_RESTRICTED` | 0 |
| `NEEDS_HUMAN` | The run escalated to a human and was not resumed | 2 |
| `POLICY_BLOCKED` | A step was refused by the allowlist or the risk policy | 3 |
| `FAILED` | A real failure, classified as one of `UNEXPECTED_SCREEN`, `TARGET_NOT_FOUND`, `TARGET_AMBIGUOUS`, `CHECKPOINT_FAILED`, `APP_ERROR`, `UNEXPECTED_DIALOG`, `EXTRACTION_FAILED`, `TIMEOUT`, `RECOVERY_EXHAUSTED`, `DRIFT_SUSPECTED`, `POLICY_VIOLATION`, `INPUT_INVALID` | 1 |
| `ABANDONED` | An operator aborted the run, or the wait for a human timed out | 1 |

"No such member" exits 0. It is an answer the caller asked for, not a crash. The outcome codes a capability can return are published in its contract, so a calling agent has to handle them rather than discover them in production.

A failure never comes back as a bare timeout. It carries the step, what was expected, what was observed (URL, title, visible text), a suggested action, and paths to the screenshot, the full DOM across frames, and the perceived element list captured at the moment it failed.

## The target application

`mockbank` is **LegacyCore**, a deliberately hostile stand-in for a bank back-office system. The brief forbids automating a real one, and a clean demo site would flatter the automation, so this app has the properties that make the problem hard:

- a `<frameset>` (menu frame + work frame), so anything perceiving only the top document sees nothing, and the top URL never changes between screens
- table layout, `<font>` tags, no `id`s, no `data-*` attributes, no ARIA
- cryptic control names (`mno`, `acct_typ`), labels that exist only as the text of the cell to the left of a field
- submit buttons as `<input type=submit value="...">`, and `javascript:` links
- the word "Search" as a button in **both** frames, so an unscoped target is genuinely ambiguous
- an accounts table with several rows under a `Curr Bal` header, so reading the savings balance means picking the right row

Sign-on is `teller1` / `Passw0rd!`. These are fake seeded values, documented here and kept out of artifacts and logs by treating the password as a sensitive input.

Runtime conditions are injected by member number, which is how each outcome class is reproduced on demand:

| Member | Behaviour | Class |
|---|---|---|
| 12345, 23456 | happy path | success |
| 99999 | "No member found" | business outcome |
| 77777 | record restricted | business outcome (permission denial) |
| 40404 | "System busy" once, then succeeds on retry | recoverable interstitial |
| 30303 | 4-second page load | recoverable (wait) |
| 20202 | `confirm()` dialog on sub-account submit | recoverable dialog |
| 50500 | session expires | auth outcome, escalates |
| 31337 | sub-account needs a supervisor override code | requires a human |
| 50000 | HTTP 500 | hard failure |

## Walkthrough

The mock app must be running (`uv run hands serve-mock`). Every command below takes the same three `--input` flags as the quick start; after the first use they are abbreviated as `...`.

### 1. Discovery: a model works out the flow

```bash
uv run hands discover \
  --goal "Sign on, look up member 12345, and read their savings account current balance." \
  --url http://127.0.0.1:5050/ \
  --input username=teller1 \
  --input password=@env:MOCKBANK_PASSWORD \
  --input member_id=12345 \
  --capability-id legacycore.lookup_savings_balance \
  --out capabilities/lookup_savings_balance.recorded.json
```

The model sees a flat list of visible elements across every frame, each with a role, accessible name, bounding box, structural path, and table row context. It acts through a small tool set: `click`, `type_text`, `select_option`, `press_key`, `navigate`, `read_value`, `wait`, `note_condition`, `request_human`, `done`. A policy gate runs before every action and every turn is written to the evidence log. The password is never shown to the model; it sees a placeholder and a description.

When the run succeeds, the recorder turns the trace into a capability. It prunes dead ends and logs what it pruned, lints checkpoints and outcome detectors against the recorded observations, merges the curated per-app outcome library from `capabilities/library/`, and refuses to serialize any literal that matches a sensitive value.

**A live run needs one API key** in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...     # default provider, model claude-opus-5
OPENAI_API_KEY=sk-...            # alternative, model gpt-5.6
```

Select with `--provider` or `HANDS_PROVIDER`; `--model` or `HANDS_MODEL` overrides the model.

**With no key**, add `--fake-llm evidence/discovery/script.json`. A scripted provider stands in for the model and the identical loop, tools, policy gate, and recorder run. It is not a real discovery run and is never presented as one; see `evidence/discovery/PENDING.md` and [Status and known gaps](#status-and-known-gaps).

Two capabilities are in the repo. `lookup_savings_balance.json` is hand-authored, and `lookup_savings_balance.recorded.json` is what the recorder produced from the command above. Both replay with the same results across the whole member table, including members the recording never saw.

### 2. Replay: no model in the loop

Replay verifies rather than assumes. For each step it checks the expected screen in its own frame, resolves the target by trying each strategy in order, runs the policy gate against the live element name, acts, then polls for the step's own outcomes, its checkpoint, and the global outcomes, in that order.

A member who does not exist:

```bash
uv run hands replay capabilities/lookup_savings_balance.json ... --input member_id=99999
```

```
status: BUSINESS_OUTCOME
outcome: MEMBER_NOT_FOUND (business) -- No member exists with that member number.
outputs_complete: False
steps:
  s1     ok
  s2     ok               strategy=relative rank=0
  ...
  s5     ok               strategy=relative rank=0
```

Exit code 0. The run stopped at the step where the outcome appeared and named it.

A restricted record. The outcome carries its own typed output, so the caller gets the reason without parsing a screenshot:

```bash
uv run hands replay capabilities/lookup_savings_balance.json ... --input member_id=77777
```

```
status: BUSINESS_OUTCOME
outcome: ACCOUNT_RESTRICTED (business) -- This member's record is restricted; contact a supervisor.
outputs:
  restriction_reason: Member 77777 record is restricted. Contact a supervisor. (SEC-0403)
```

A "system busy" interstitial. The outcome's handler clicks Retry, re-verifies, and the step reports `recovered` instead of being silently absorbed:

```bash
uv run hands replay capabilities/lookup_savings_balance.json ... --input member_id=40404
```

```
status: SUCCESS
outputs:
  savings_balance: 980.00
steps:
  ...
  s6     recovered        strategy=role rank=0
  s7     ok
```

A real failure:

```bash
uv run hands replay capabilities/lookup_savings_balance.json ... --input member_id=50000
```

```
status: FAILED
failure category: APP_ERROR
  step: {'id': 's6', 'index': 5, 'intent': 'Run the member search', 'kind': 'click'}
  expected: no application error
  observed: url='http://127.0.0.1:5050/' title='LegacyCore Teller Workstation' text='... Internal Server Error ... Reference: CBS-0500 ...'
  suggested_action: The host returned an internal server error (CBS-0500).
  evidence: {'screenshot': 'screenshots/step-13-s6-failed.png', 'dom': 'dom/s6-failed.html', 'elements': 'dom/s6-failed-elements.json'}
```

Exit code 1.

Flags worth knowing:

| Flag | What it does |
|---|---|
| `--repeat N` | Run N times and print a stability summary: status counts, and whether any step ever needed a non-zero strategy rank |
| `--json` | Print the raw result object instead of the summary |
| `--approved-by NAME` | Confirm risky steps on an unapproved capability; the name is recorded in the result |
| `--headed` | Watch the browser |
| `--base-url URL` | Run the same artifact against another host or port |

### 3. Review a capability

```bash
uv run hands inspect capabilities/lookup_savings_balance.json
```

Prints the contract a calling agent sees (typed inputs, typed outputs, the outcome codes it must handle, side effects, idempotency), each screen with its frame and URL pattern, each step with its intent, target description, ordered strategies, and risk level, and every outcome with its origin.

```bash
uv run hands schema export
```

Writes the capability JSON Schema and a callable tool definition per capability into `capabilities/schema/`. The tool definition is what an orchestrating agent would load to call this capability like any other tool.

### 4. Human handoff

When replay cannot safely continue, it raises an intervention request: a reason code, the step and whether its action had already fired, what was expected, what was observed, and the evidence paths. A human claims the **same live browser session**, does what is needed, and hands it back. The run then re-verifies rather than trusting: if the success condition now holds it finishes, if the step's checkpoint holds the step is marked `human_performed`, otherwise it retries or escalates again.

Control is a token with a holder and an epoch, and the epoch increments on every transfer. A command issued before a handoff cannot land after it; its token is stale and the surface refuses it. State is written to `control_state.json` on every transition, so "who holds the session" is answerable from disk after a crash.

The operator console is a minimal HTTP surface driven from the terminal:

```bash
uv run hands operator serve                     # start a console; prints its run directory
uv run hands operator status --run-dir <dir>    # who holds the session, read from disk
uv run hands operator list   --run-dir <dir>    # open intervention requests
uv run hands operator claim  <id>               # take control
uv run hands operator resume <id> --mode auto   # hand it back: auto | retry_step | abort
```

The end-to-end path with a live browser is `tests/test_handoff.py`. It wires the escalation handler into a real replay, hits a risky "Open Sub-Account" click on an unapproved capability, has a scripted operator claim the session and act in it, and asserts the run finishes with that step marked `human_performed` and the epoch advanced exactly twice. A second test drives the supervisor-override screen for member `31337` directly: the automation is blocked by a code it does not have, a human enters it, and the same session continues. `operator serve` on its own starts a console with no browser attached; the state machine, persistence, claim and resume are real, the page is not.

## Example run

One pass through the whole system, exactly as it ran: discover and record, then replay the recording against a member it never saw. This run used the scripted provider, so the model column says `fake-llm` and the token counts are zero. A live model run writes the same files.

### Discover and record

```bash
uv run hands discover \
  --goal "Sign on, look up member 12345, and read their savings account current balance." \
  --url http://127.0.0.1:5050/ \
  --input username=teller1 --input password=@env:MOCKBANK_PASSWORD --input member_id=12345 \
  --capability-id legacycore.lookup_savings_balance --name "Look up savings balance" \
  --fake-llm evidence/discovery/script.json \
  --out lookup.recorded.json
```

```
status: success   stop_reason: done   steps: 7
summary: Signed on as the teller, looked up the member by number, and read the current balance of the savings account.
outputs:
  savings_balance: 2,340.55
token usage: {'input_tokens': 0, 'output_tokens': 0}
evidence dir: .../evidence/runs/disc-20261022T024818Z-0259f6
capability written: lookup.recorded.json
```

The run directory:

```
evidence/runs/disc-20261022T024818Z-0259f6/
  transcript.jsonl      one line per model turn: the exact observation shown, each tool call with its stated reason, usage, latency
  events.jsonl          every action with the policy decision taken before it, the resolved value, and the result
  pruning_log.json      what the recorder dropped from the trace, and why
  capability.raw.json   the unpruned artifact, including any failed or blocked steps
  screenshots/          step-01-entry.png, step-02-step1-type_text.png, step-03-obs1.png ... step-14-step7-done.png
```

What the model saw on its first turn. Perception is one flat list across every frame, with table row context on each element. The password placeholder is offered; the value is not:

```
URL: http://127.0.0.1:5050/
Title: LegacyCore Teller Workstation

Frames:
  [frame 0] (unnamed) http://127.0.0.1:5050/
  [frame 1] "menu" http://127.0.0.1:5050/nav
  [frame 2] "work" http://127.0.0.1:5050/main

Elements in frame 1:
  [2] link "Home" (href="/main"; row 1 of table with headers ['LegacyCore'])
  [3] link "Member Inquiry" (href="/lookup"; row 2 of table with headers ['LegacyCore'])
  [4] link "Sign Off" (href="javascript:parent.work.location='/signoff'"; row 3 of table with headers ['LegacyCore'])
  [6] textbox "mno" (row 4 of table with headers ['LegacyCore'])
  [7] button "Search" (type=submit; value="Search"; row 4 of table with headers ['LegacyCore'])

Elements in frame 2:
  [11] textbox "User ID" (row 0 of table with headers ['User ID', ''])
  [13] textbox "Password" (type=password; row 1 of table with headers ['User ID', ''])
  [14] button "Sign On" (type=submit; value="Sign On"; row 2 of table with headers ['User ID', ''])

Available input placeholders: {{member_id}}, {{password}}, {{username}}
```

Three lines from the event log, trimmed. Every action carries the model's own reason, the policy decision made before it ran, and the value that was typed, redacted on write:

```
{"seq": 2, "kind": "action", "tool": "type_text", "reason": "enter the teller user id so the session can be signed on",
 "tool_input": {"element": 11, "text": "{{username}}"}, "resolved_value": "teller1",
 "policy_decision": {"allowed": true, "requires_confirmation": false}}
{"seq": 3, "kind": "action", "tool": "type_text", "reason": "enter the teller password to complete the sign-on credentials",
 "tool_input": {"element": 13, "text": "{{password}}"}, "resolved_value": "[redacted:password]",
 "policy_decision": {"allowed": true, "requires_confirmation": false}}
{"seq": 7, "kind": "read_value", "reason": "capture the current balance from the savings row, which is the value the goal asks for",
 "output_name": "savings_balance", "element": {"idx": 26, "role": "cell", "name": "2,340.55", "frame_index": 2}}
```

The pruning log shows the recorder's checkpoint lint at work. Two checkpoints were already true before their action ran, so they were dropped rather than kept as false proof:

```json
[
  {"kind": "checkpoint", "trace_step_index": 0, "reason": "checkpoint condition was already true before the action executed; proves nothing"},
  {"kind": "checkpoint", "trace_step_index": 3, "reason": "checkpoint condition was already true before the action executed; proves nothing"}
]
```

### The recorded capability

Six steps over three screens. The `done` call is the stop signal, not a step, so seven tool calls became six steps. Each target carries strategies ordered from most to least semantic, and each strategy carries its own rationale:

```
s1  type     main    role, structural, bbox             enter the teller user id so the session can be signed on
s2  type     main    role, structural, bbox             enter the teller password to complete the sign-on credentials
s3  click    main    role, text, structural, bbox       submit the sign-on form to reach the member inquiry screen
s4  type     lookup  role, structural, bbox             enter the member number so the record can be retrieved
s5  click    lookup  role, text, structural, bbox       run the member search to open the member detail screen
s6  extract  member  role, relative, structural, bbox   capture the current balance from the savings row
```

The checkpoint on the search click asserts the destination's title, not only its path, because `/member` also serves the not-found, busy and error pages:

```json
"checkpoint": {
  "condition": {"type": "all_of", "conditions": [
    {"type": "url_matches",   "pattern": "^https?://[^/]+/member(?:$|\\?)"},
    {"type": "title_matches", "pattern": "Member\\ Detail"}
  ]},
  "description": "work frame navigated to path '/member' showing 'Member Detail'"
}
```

The extraction step does not trust a cell position. It locates the accounts table, then reads the cell by header and row match, so the value is parameterised over the member even though the recording only ever saw one:

```json
"extract": {
  "from": "table_cell",
  "table": {"header_match": "Curr\\ Bal", "row_where": {"header": "Type", "matches": "SAV"}},
  "parse": {"type": "money", "regex": "-?[\\d,]+\\.\\d+"},
  "output": "savings_balance"
}
```

The eight outcomes (`MEMBER_NOT_FOUND`, `ACCOUNT_RESTRICTED`, `SYSTEM_BUSY`, `SESSION_EXPIRED`, `APP_ERROR`, `VALIDATION_ERROR`, `SUPERVISOR_OVERRIDE_REQUIRED`, `UNEXPECTED_CONFIRM`) all carry `"origin": "library"`. A happy-path run cannot observe them, and the artifact says so.

### Replay the recording on a member it never saw

```bash
uv run hands replay lookup.recorded.json \
  --input username=teller1 --input password=@env:MOCKBANK_PASSWORD --input member_id=23456
```

```
run_id: replay-20261022T024821Z-7c6977
capability: legacycore.lookup_savings_balance v1.0.0
status: SUCCESS
outputs:
  savings_balance: 15020.31
outputs_complete: True
steps:
  s1     ok               strategy=structural rank=1
  s2     ok               strategy=structural rank=1
  s3     ok               strategy=role rank=0
  s4     ok               strategy=structural rank=1
  s5     ok               strategy=role rank=0
  s6     ok               strategy=structural rank=2
drift signals:
  - step s1: resolved via strategy rank 1 (structural) rather than the top-ranked strategy
  - step s2: resolved via strategy rank 1 (structural) rather than the top-ranked strategy
  - step s4: resolved via strategy rank 1 (structural) rather than the top-ranked strategy
  - step s6: resolved via strategy rank 2 (structural) rather than the top-ranked strategy
evidence dir: .../evidence/runs/replay-20261022T024821Z-7c6977
```

Member 23456 lists checking before savings, the opposite order from 12345, and the right balance came back because extraction matches the `SAV` row rather than a position.

The drift signals are worth reading rather than skipping. Steps s1, s2 and s4 resolve at rank 1 on this artifact for every member, including the one it was recorded on (compare `evidence/replay-success/`), so that is a property of the recording, not drift between members. Step s6 is more telling: the recorder wrote the discovered balance `2,340.55` into the `name` of the role and relative strategies, so neither can match any other member and the structural path carried it. The run still succeeds because extraction reads by header, but the signal is doing its job: it is pointing at a target that should be re-recorded or edited before the structural path breaks too.

The result object, trimmed:

```json
{
  "run_id": "replay-20261022T024821Z-7c6977",
  "capability_ref": {"id": "legacycore.lookup_savings_balance", "version": "1.0.0", "content_hash": "55c7dbab..."},
  "status": "SUCCESS",
  "outcome": null,
  "outputs": {"savings_balance": "15020.31"},
  "outputs_complete": true,
  "steps": [
    {"step_id": "s6", "intent": "capture the current balance from the savings row, ...",
     "status": "ok", "strategy_used": "structural", "strategy_rank": 2, "attempts": 3, "outcomes_matched": []}
  ],
  "drift_signals": ["step s6: resolved via strategy rank 2 (structural) rather than the top-ranked strategy", "..."],
  "human_interventions": [],
  "side_effects_uncertain": false,
  "failure": null
}
```

The replay run directory:

```
evidence/runs/replay-20261022T024821Z-7c6977/
  result.json           the object above, in full
  events.jsonl          run_started, the app fingerprint check, then per step: target resolved (strategy, rank), policy check, action
  screenshots/          step-01-startup-fingerprint.png, step-02-s1-pre.png, step-03-s1-poll.png ... step-15-success-checkpoint.png
```

The password appears in none of it:

```bash
grep -rc "Passw0rd" lookup.recorded.json evidence/runs/disc-20261022T024818Z-0259f6/   # 0 in every file
grep -o "\[redacted:[a-z_]*\]" evidence/runs/disc-20261022T024818Z-0259f6/events.jsonl  # [redacted:password]
```

## Inside a capability

A capability is a contract, not a step list. The fields worth knowing before opening one:

- **`contract`**: typed inputs (each marked `sensitive`, and sourced from the `caller` or a `vault`), typed outputs, `side_effects`, `idempotent`, and the `outcome_codes` the caller must handle. `to_tool_definition()` emits it as a JSON-Schema tool.
- **`screens`**: a first-class map. Every step names the screen it expects, and a screen is matched in its own frame by URL pattern and landmarks, never by the top document. The recorder identifies screens by the work frame's document title, which does not vary by record.
- **`steps`**: each has an intent, a screen, a target with **ordered strategies** (`role`, `text`, `relative`, `structural`, `bbox`) each carrying its own rationale, a checkpoint, and a `risk` object. A strategy must match exactly one element or the next is tried, and the rank that succeeded is reported as a drift signal.
- **`outcomes`**: conditions over an observation with a kind (`business`, `interstitial`, `auth`, `app_error`, `requires_human`), a handler (`stop`, `click`, `dismiss_dialog`, `escalate`), optional result outputs, and an `origin` recording whether the model discovered it, the per-app library supplied it, or a reviewer added it. A single happy-path run cannot observe "not found", so the artifact says where each detector came from.
- **`risk`** and **`idempotency_check`**: a risky step is never retried without an idempotency signal. A risky step that fails after its action fired sets `side_effects_uncertain`.
- **`content_hash`** and **`approval`**: the hash covers behaviour only. Editing an approved artifact silently demotes it to draft.
- **`provenance`**: the discovery run, model, prompt version, and viewport it was recorded against, plus a pointer to the raw unpruned artifact.

A capability never refers to a browser locator. It refers to a frame matcher, strategies, and conditions over an element list, which is the seam a desktop backend would implement. `CapabilityOverlay` specialises a base capability per tenant: an overlay may retarget and re-fingerprint, but may only raise risk and only narrow the allowlist, and may never change step kinds or the contract.

## Safety

- **The allowlist is data.** `policy.yaml` holds allowed domains, URL patterns, and actions. Effective policy is that file intersected with the capability's own block, so a capability can only narrow what the operator permits. Non-HTTP schemes are rejected outright; the menu frame is full of `javascript:` links.
- **Risk is gated, not guessed.** A step whose live target name matches a risky pattern ("Confirm", "Submit", "Transfer", ...) is routed to a human unless the capability is approved or the caller passes `--approved-by`. The gate runs against the live element name at replay, so a control since relabelled "Delete" is caught.
- **Secrets in two layers.** Sensitive values are never shown to the model, and the recorder raises rather than writing a literal that matches one. Evidence is redacted on write. `grep -r "Passw0rd!" evidence/` finds nothing, and `grep -rho "\[redacted:[a-z_]*\]" evidence/discovery-offline` finds the marker proving the redactor fired on a real value.
- **Guardrails that fail open are tested.** A YAML quoting mistake once turned every `\b` in the risky patterns into a backspace, which would have classified nothing as risky with no error. `tests/test_policy.py` loads the real file and asserts "Confirm" is still risky.

## Tests

```bash
uv run pytest
```

Browser tests run against the real mock app in headless Chromium, started on a free port by the test session. Nothing requires an API key or network access.

| File | What it proves |
|---|---|
| `test_end_to_end.py` | Records an artifact from a discovery run, then replays it with no model, including against a **different** member than it was recorded on |
| `test_replay.py` | Picks the savings row over checking and loan, every outcome class on the member table, wrong-screen detection, locator fallthrough with rank and drift, risky unapproved steps escalate, the password never reaches events or results |
| `test_recorder.py` | Screens target the work frame not the top, the member id is parameterised, already-true checkpoints and loose outcome detectors are dropped, pruned steps survive in the raw artifact, the password is never serialised |
| `test_handoff.py` | Stale tokens rejected after a claim, control state on disk after every transition, the full escalate / claim / resume path, human actions recorded without values |
| `test_schema.py` | Round trips, hash stable across approval and provenance, approved artifacts demote on edit, semantic validation, overlay rules, exit codes |
| `test_policy.py` | URL allowlist, scheme rejection, risk classification, and the real `policy.yaml` |
| `test_agent_loop.py` | The discovery loop with a scripted provider: dead-end detection, `request_human`, every step has a reason |
| `test_llm_providers.py` | Anthropic and OpenAI translation, provider resolution, reasoning-span ordering |
| `test_cli.py`, `test_operator_cli.py` | The commands above, including exit codes and that `discover` never prints a sensitive value |

Read `test_end_to_end.py` first. That join is the system's central claim, and it found four real bugs the two halves could not find separately (REPORT.md, Determinism).

## Layout

```
hands/
  surface/      perception (a JS walker over every frame) and acting; locators, conditions, extraction
  agent/        the LLM loop, tool definitions, providers, and the recorder
  replay/       the deterministic engine
  handoff/      control-transfer state machine, operator server, human action recorder
  safety/       allowlist and risk policy, redaction
  schema/       the capability contract, results, control tokens, overlays
  evidence/     per-run structured logs
  cli.py        serve-mock, discover, replay, inspect, schema, operator
mockbank/       LegacyCore, the target application
capabilities/   saved artifacts, the per-app outcome library, exported schema
evidence/       curated runs (raw output under evidence/runs/ is ignored)
tests/
policy.yaml     operator-owned allowlist and risk policy
REPORT.md       design write-up
```

## Status and known gaps

- **The live model-driven discovery run has not been done.** The machine this was built on has no API key. The scripted run in `evidence/discovery-offline/` exercises the identical code path and is labelled as such; `evidence/discovery/PENDING.md` has the exact command to produce the real one.
- Session-expiry re-login is detected and escalated, not recovered.
- Overlays are implemented and tested as a type; there is no tenant registry or second tenant variant of the mock app.
- The operator console is a terminal surface with a scripted stand-in operator, not a product.
- `approval` has real semantics but there is no review command yet.

Each cut and its reason is in REPORT.md under Cuts.

## Troubleshooting

- **`Executable doesn't exist at ...chromium_headless_shell...`**: run `uv run playwright install chromium`. The pinned Playwright version wants its own Chromium build, and a different one already on the machine will not do.
- **Garbled characters on Windows**: set `PYTHONUTF8=1`.
- **Port 5050 is taken**: start the mock with `--port`, then pass `--base-url http://127.0.0.1:<port>` to `replay` or `--url` to `discover`.
- **`discover` fails with no key and no `--fake-llm`**: that is intended. It names the missing variable and never opens a browser.

## Repo conventions

Secrets live in `.env` and never in the repo. Raw run output under `evidence/runs/` is ignored; curated runs are committed. The assignment brief itself is not committed.
