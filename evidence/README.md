# Evidence

Real runs against the LegacyCore mock app, produced by the commands in the
root README. Each directory is one run exactly as the system wrote it.

| Directory | What it shows | Result |
|---|---|---|
| `discovery-offline/` | A discovery run recorded into a capability | `success`, artifact written |
| `replay-success/` | Replaying that artifact for member 12345 | `SUCCESS`, `savings_balance` 2340.55, exit 0 |
| `replay-not-found/` | Member 99999 | `BUSINESS_OUTCOME` `MEMBER_NOT_FOUND`, **exit 0** |
| `replay-restricted/` | Member 77777, a permission denial | `BUSINESS_OUTCOME` `ACCOUNT_RESTRICTED`, exit 0 |
| `replay-busy-retry/` | Member 40404, a "system busy" interstitial | `SUCCESS` after a `recovered` step |
| `replay-app-error/` | Member 50000, an HTTP 500 | `FAILED` `APP_ERROR`, exit 1, with screenshot and DOM |
| `discovery/` | How to produce the live model-driven run | see `PENDING.md` |

The two replays worth opening are `replay-not-found/` and `replay-app-error/`.
They are the distinction the brief calls out: "no such member" is an answer
the caller asked for and exits 0, while a real failure exits non-zero and
carries the step, what was expected, what was observed, and the evidence
needed to debug it.

## What is in a run directory

- `result.json` — the terminal result contract: status, outcome, outputs,
  per-step status with the targeting strategy that succeeded and its rank,
  drift signals, and on failure a classified failure object.
- `events.jsonl` — every action with the reason it was taken, every policy
  check, every outcome detected. Redacted on write.
- `transcript.jsonl` — discovery only: one line per model turn.
- `screenshots/` — one before each action, plus one at the point of failure.
- `dom/` — on failure, the full DOM across frames and the perceived element
  list, so a failure can be diagnosed without reproducing it.
- `capability.json` — discovery only: the artifact that run produced.
- `terminal-output.txt` — what the command printed.

## A note on the discovery run

`discovery-offline/` used a scripted provider rather than a live model. It
exercises the identical code path (same loop, same tools, same policy gate,
same recorder), which is what makes the system reviewable without an API key
— but it is **not** the genuine model-driven run the brief requires, and it
is labelled that way rather than presented as one. `discovery/PENDING.md` has
the exact command to produce the real one.

## Verifying the secret handling

The mock app's seeded password is passed to every run as a sensitive input.
It appears in none of these files:

```bash
grep -r "Passw0rd!" evidence/replay-* evidence/discovery-offline   # no matches
grep -rho "\[redacted:[a-z_]*\]" evidence/discovery-offline        # [redacted:password]
```

The marker proves the redactor actually fired on the resolved value, rather
than the value simply never having been written.
