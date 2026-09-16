# The live discovery run is outstanding

The brief requires at least one genuinely model-driven discovery run against a
live surface, with the evidence here to show it happened. **That run has not
been done**, because the machine this was built on has no API key configured
for either provider. Stating that plainly is better than shipping a scripted
run and letting it be mistaken for a real one.

Everything the run needs is built and exercised. The discovery loop, the
recorder, replay, safety, evidence and handoff are all tested end to end using
a scripted provider that drives the identical code path, including a test that
records an artifact and replays it against a different record than the one it
was recorded on (`tests/test_end_to_end.py`).

## To produce it

Put a key in `.env` at the repo root (either provider works):

```
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
```

Start the target app:

```bash
uv run hands serve-mock
```

Then run discovery, from the repo root:

```bash
uv run hands discover \
  --goal "Sign on, look up member 12345, and read their savings account current balance." \
  --url http://127.0.0.1:5050/ \
  --input username=teller1 \
  --input password=@env:MOCKBANK_PASSWORD \
  --input member_id=12345 \
  --out capabilities/lookup_savings_balance.json
```

Add `--provider openai` to use OpenAI instead of the default.

The run writes a directory under `evidence/runs/<run_id>/` containing
`transcript.jsonl` (one line per model turn: message id, model, token usage,
latency, the exact observation shown, and every tool call with the model's
own stated reason), `events.jsonl`, per-step screenshots, `capability.raw.json`
alongside the curated artifact, and the recorder's pruning log. Copy that
directory here.

Then confirm the artifact replays with no model involved, by unsetting the key
first so it cannot silently fall back to one:

```bash
uv run hands replay capabilities/lookup_savings_balance.json \
  --input username=teller1 --input password=@env:MOCKBANK_PASSWORD \
  --input member_id=12345
```

## What the evidence should keep

Keep the wasted actions and any wrong turns. A transcript with no dead ends
looks scripted, and the recorder's pruning log exists precisely so a reviewer
can see what the model did that did not make it into the artifact.
