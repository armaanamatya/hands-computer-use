# hands

Gives an AI agent hands inside a legacy application that has no API.

A model works out how to do a task once, by driving a real UI the way a person would. The successful run is recorded as a typed, versioned **capability**. From then on that capability replays deterministically with no model in the loop, verifies what it did, returns typed outputs, and asks a human to take over the live session when it cannot safely continue.

Design decisions and trade-offs are in **[REPORT.md](REPORT.md)**.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run playwright install chromium
cp .env.example .env
```

**No API key is needed to run anything in this repo except a live discovery run.** Replay never calls a model, and discovery has a scripted-provider mode that exercises the identical code path offline. To do a live discovery run, put one key in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...     # default provider, model claude-opus-5
OPENAI_API_KEY=sk-...            # alternative, model gpt-5.6
```

Either provider works; select with `--provider` or `HANDS_PROVIDER`. `HANDS_MODEL` overrides the model.

On Windows, set `PYTHONUTF8=1` if your console is not already UTF-8.

## The target application

`mockbank` is **LegacyCore**, a deliberately hostile stand-in for a bank back-office system. The brief forbids automating a real one, and a clean demo site would flatter the automation, so this app has the properties that make the problem hard:

- a `<frameset>` (menu frame + work frame), so anything perceiving only the top document sees nothing, and the top URL never changes between screens
- table layout, `<font>` tags, no `id`s, no `data-*` attributes, no ARIA
- cryptic control names (`mno`, `acct_typ`), labels that exist only as the text of the cell to the left of a field
- submit buttons as `<input type=submit value="...">`, and `javascript:` links
- the word "Search" as a button in **both** frames, so an unscoped target is genuinely ambiguous
- an accounts table with several rows under a `Curr Bal` header, so reading the savings balance means picking the right row

Sign-on is `teller1` / `Passw0rd!`. These are fake seeded values, documented here and kept out of artifacts and logs by treating the password as a sensitive input.

Runtime conditions are injected by member number, which is how the evidence runs reproduce each class of outcome on demand:

| Member | Behaviour | Class |
|---|---|---|
| 12345, 23456 | happy path | success |
| 99999 | "No member found" | business outcome |
| 77777 | record restricted | business outcome (permission denial) |
| 40404 | "System busy" once, then succeeds on retry | recoverable interstitial |
| 30303 | 4-second page load | recoverable (wait) |
| 20202 | `confirm()` dialog on submit | recoverable dialog |
| 50500 | session expires | auth outcome, escalates |
| 31337 | needs a supervisor override code | requires a human |
| 50000 | HTTP 500 | hard failure |

## Demo

Start the target app in one terminal and leave it running:

```bash
uv run hands serve-mock            # http://127.0.0.1:5050
```

### 1. Discovery: a model works out the flow

```bash
uv run hands discover \
  --goal "Sign on, look up member 12345, and read their savings account current balance." \
  --url http://127.0.0.1:5050/ \
  --input username=teller1 \
  --input password=@env:MOCKBANK_PASSWORD \
  --input member_id=12345 \
  --out capabilities/lookup_savings_balance.recorded.json
```

`@env:` reads a value from the environment so a secret never appears on the command line or in shell history. The password is treated as sensitive automatically: the model is shown a placeholder and a description, never the value.

To run the same path with **no API key**, add `--fake-llm evidence/discovery/script.json` to substitute a scripted provider. That exercises the identical loop, tools, policy gate and recorder; it is not a real discovery run, and `evidence/discovery/PENDING.md` says so plainly.

Two capabilities are in the repo: `lookup_savings_balance.json` is hand-authored, and `lookup_savings_balance.recorded.json` is what the recorder produced from the run above. They replay identically. Either path works in the commands below.

### 2. Replay: no model in the loop

```bash
uv run hands replay capabilities/lookup_savings_balance.json \
  --input username=teller1 --input password=@env:MOCKBANK_PASSWORD \
  --input member_id=12345
```

```
status: SUCCESS
outputs:
  savings_balance: 2340.55
steps:
  s1     ok
  s2     ok               strategy=relative rank=0
  s3     ok               strategy=relative rank=0
  s4     ok               strategy=role rank=0
  s5     ok               strategy=relative rank=0
  s6     ok               strategy=role rank=0
  s7     ok
EXIT CODE: 0
```

Now the same capability against a member who does not exist:

```bash
uv run hands replay capabilities/lookup_savings_balance.json \
  --input username=teller1 --input password=@env:MOCKBANK_PASSWORD \
  --input member_id=99999
```

```
status: BUSINESS_OUTCOME
outcome: MEMBER_NOT_FOUND (business) -- No member exists with that member number.
EXIT CODE: 0
```

**Exit code 0.** "No such member" is an answer the caller asked for, not a crash. Conflating the two is the mistake the brief calls out, so it is visible at the shell rather than buried in a JSON field.

Two more worth running:

```bash
# recovers from a "system busy" interstitial: one step reports `recovered`
uv run hands replay capabilities/lookup_savings_balance.json ... --input member_id=40404

# a real failure: APP_ERROR, exit 1, with a screenshot and DOM snapshot
uv run hands replay capabilities/lookup_savings_balance.json ... --input member_id=50000
```

### 3. Review a capability

```bash
uv run hands inspect capabilities/lookup_savings_balance.json
```

Prints the contract a calling agent sees (typed inputs, typed outputs, the outcome codes it must handle, side effects), each screen, and each step with its intent, target, ordered targeting strategies and risk level.

```bash
uv run hands schema export
```

Writes the capability JSON Schema and a callable tool definition per capability into `capabilities/schema/`.

### 4. Human handoff

```bash
uv run hands operator serve        # local operator surface
uv run hands operator status       # who holds the session, read from disk
uv run hands operator list         # open intervention requests
uv run hands operator claim <id>   # take control of the live session
uv run hands operator resume <id> --mode auto
```

Member `31337` needs a supervisor override code the automation does not have. Replaying that flow escalates, a human completes the step in the same live browser window, and the run resumes. `tests/test_handoff.py` drives this end to end with a scripted stand-in operator.

## Tests

```bash
uv run pytest
```

Browser tests run against the real mock app in headless Chromium. Nothing requires an API key or network access.

The one to read first is `tests/test_end_to_end.py`: it records an artifact from a discovery run and then replays that artifact with no model involved, including against a **different** member than the one it was recorded on. That join is the system's central claim, and it found four real bugs the two halves could not find separately (see REPORT.md § Determinism).

## Layout

```
hands/
  surface/    perception (a JS walker over every frame) and acting; locators, conditions, extraction
  agent/      the LLM loop, tool definitions, providers, and the recorder
  replay/     the deterministic engine
  handoff/    control-transfer state machine, operator surface, action recorder
  safety/     allowlist and risk policy, redaction
  schema/     the capability contract, results, control, overlays
  evidence/   per-run structured logs
mockbank/     LegacyCore, the target application
capabilities/ saved artifacts, the per-app outcome library, exported schema
evidence/     curated runs
```

## Repo conventions

Secrets live in `.env` and never in the repo. Raw run output under `evidence/runs/` is ignored; curated runs are committed. The assignment brief itself is not committed.
