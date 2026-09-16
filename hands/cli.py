"""Command-line entry point: the glue between everything else in this repo
and a reviewer typing commands at a terminal.

Every subcommand follows the same shape: call `config.load_env()` first,
parse/validate CLI-caused input as early as possible (before touching a
browser), then do the work, wrapping any Playwright session in try/finally
so the browser always closes. `main()` never lets a user-causable error
escape as a raw traceback: helpers in this module and in `hands.config`
signal a clean failure by raising `SystemExit(<one-line message>)`, and
`main()` is the single place that catches it, prints it, and turns it into
a plain non-zero return value -- so `main([...])` is easy to drive from
tests (always returns an int, never raises for an ordinary user mistake).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from pydantic import ValidationError

from hands import config
from hands.agent.loop import DiscoveryLoop
from hands.agent.recorder import RecorderError, record
from hands.evidence.run_log import RunLog, new_run_id
from hands.safety.policy import Policy
from hands.safety.redact import Redactor
from hands.schema.capability import AppInfo, Capability, InputSpec, OutputSpec
from hands.schema.control import ControlStateRecord, InterventionRequest
from hands.surface.web import WebSurface

# An input literally named "password" is always treated as sensitive, even
# if the caller never passes --sensitive password (see `discover`'s help).
_ALWAYS_SENSITIVE = {"password"}

_MONEY_RE = re.compile(r"-?\$?[\d,]+\.\d{2}")
_INT_RE = re.compile(r"-?\d+")

# `hands operator serve`'s default HTTP port. `claim`/`resume` default to the
# SAME port, so the common case -- one terminal running `hands operator
# serve`, another running `hands operator claim <id>` -- works with no
# --port flag on either side.
_DEFAULT_OPERATOR_PORT = 8765


# ==========================================================================
# small shared helpers
# ==========================================================================


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_").lower()
    return slug[:60]


def _read_json_file(path: str) -> dict[str, Any]:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"hands: could not read {path!r}: {e.strerror or e}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"hands: {path!r} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"hands: {path!r} does not contain a JSON object")
    return data


def _split_host_port(netloc: str) -> tuple[str, Optional[str]]:
    if ":" in netloc:
        host, _, port = netloc.rpartition(":")
        return host, port
    return netloc, None


def _load_policy(path: Any, rewrite_netloc: Optional[tuple[str, str]] = None) -> Policy:
    """Load a Policy from `path`. When `rewrite_netloc` is given (a
    (old_netloc, new_netloc) pair, as returned by
    `_rewrite_capability_base_url`), the same --base-url change is also
    applied to `allowed_url_patterns` -- otherwise a capability pointed at a
    different host:port would be denied by the operator's own policy before
    it ever got a chance to run.

    Only the PORT is substituted (":5050" -> ":<newport>"), not the whole
    netloc: `policy.yaml`'s patterns are regexes with backslash-escaped dots
    (e.g. `127\\.0\\.0\\.1:5050`), so a plain netloc ("127.0.0.1:5050")
    never occurs there as a literal substring -- the port suffix is the one
    part that is never escaped and safely substitutable either way. This
    mirrors exactly what tests/test_replay.py's own `policy` fixture does
    (it substitutes only `mock_server`'s port into a freshly built pattern).
    """
    import yaml

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"hands: could not read policy file {path!r}: {e.strerror or e}")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SystemExit(f"hands: policy file {path!r} is not valid YAML: {e}")
    if not isinstance(raw, dict):
        raise SystemExit(f"hands: policy file {path!r} must contain a mapping")

    if rewrite_netloc is not None:
        old_netloc, new_netloc = rewrite_netloc
        _, old_port = _split_host_port(old_netloc)
        _, new_port = _split_host_port(new_netloc)
        patterns = raw.get("allowed_url_patterns") or []
        if old_port and new_port:
            raw["allowed_url_patterns"] = [
                (p.replace(f":{old_port}", f":{new_port}") if isinstance(p, str) else p) for p in patterns
            ]

    try:
        return Policy.from_dict(raw)
    except (KeyError, ValueError, TypeError) as e:
        raise SystemExit(f"hands: policy file {path!r} is invalid: {e.__class__.__name__}: {e}")


def _load_fake_script(path: str) -> list[dict[str, Any]]:
    """Load a --fake-llm script: either a plain JSON list of turns
    (`[{"tool": ..., "input": ...}, ...]`), or a JSONL transcript in the
    shape `RunLog.transcript` writes (one JSON object per line, each with a
    `tool_calls` list) -- the same shape `FakeLLM.from_jsonl` already knows
    how to read.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"hands: could not read --fake-llm script {path!r}: {e.strerror or e}")
    stripped = text.strip()
    if not stripped:
        raise SystemExit(f"hands: --fake-llm script {path!r} is empty")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return parsed

    from hands.agent.llm import FakeLLM

    try:
        fake = FakeLLM.from_jsonl(path)
    except OSError as e:
        raise SystemExit(f"hands: could not read --fake-llm script {path!r}: {e}")
    if not fake.script:
        raise SystemExit(
            f"hands: --fake-llm script {path!r} is neither a JSON list of turns "
            "nor a JSONL transcript with any tool calls"
        )
    return fake.script


def _load_outcome_library(path: Optional[str]) -> Optional[list[dict[str, Any]]]:
    """Load the curated per-app outcome library that gets merged into a
    recorded capability.

    This matters more than it looks. A discovery run only ever walks the
    happy path, so it cannot observe "record not found", "system busy", or a
    500 page -- those conditions simply never appear. Without a library
    merged in, the recorded artifact has no detectors for them, and replay
    then reports a legitimate business answer as UNEXPECTED_SCREEN: it can
    tell it is not where it expected to be, but not why, and a caller cannot
    distinguish "no such member" from a broken automation.

    So the library is the per-app knowledge a human curates once and every
    capability recorded against that app inherits. Each entry is marked
    `origin: "library"` in the artifact, and the recorder lints every
    detector against the recorded success path so a loose pattern cannot
    shadow a successful run.
    """
    if path is not None and path.strip().lower() == "none":
        return None
    if path is None:
        default = config.CAPABILITIES_DIR / "library" / "legacycore.outcomes.json"
        if not default.exists():
            return None
        path = str(default)
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"hands: could not read outcome library {str(p)!r}: {e.strerror or e}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"hands: outcome library {str(p)!r} is not valid JSON: {e}")
    if not isinstance(data, list):
        raise SystemExit(f"hands: outcome library {str(p)!r} must contain a JSON list of outcomes")
    return data


def _sniff_output_type(value: str) -> str:
    v = (value or "").strip()
    if _MONEY_RE.fullmatch(v):
        return "money"
    if _INT_RE.fullmatch(v):
        return "integer"
    return "string"


def _rewrite_capability_base_url(data: dict[str, Any], base_url: str) -> tuple[str, str]:
    """Rewrite every occurrence of the capability's current host:port with
    `base_url`'s host:port, in place, across the whole capability dict --
    this covers `target.entry_url`, any `navigate` step's `value`, and any
    `FrameMatcher.url_pattern`/`Screen.url_pattern` that happens to embed a
    host:port (this project's own capabilities keep those path-only, per
    tests/test_replay.py's module docstring, but a hand-authored one is not
    guaranteed to). Returns (old_netloc, new_netloc) so the caller can apply
    the identical substitution to the operator policy's own
    allowed_url_patterns (see `_load_policy`).
    """
    old_entry = (data.get("target") or {}).get("entry_url", "")
    old_netloc = urlsplit(old_entry).netloc
    new_netloc = urlsplit(base_url).netloc
    if not old_netloc or not new_netloc:
        raise SystemExit(
            f"hands: --base-url {base_url!r} must be a full URL with a host (and usually a port), "
            f"and the capability's own target.entry_url {old_entry!r} must be too"
        )

    def walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.replace(old_netloc, new_netloc) if old_netloc in obj else obj
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    rewritten = walk(data)
    data.clear()
    data.update(rewritten)
    return old_netloc, new_netloc


# ==========================================================================
# serve-mock (pre-existing)
# ==========================================================================


def _cmd_serve_mock(args: argparse.Namespace) -> int:
    config.load_env()
    from mockbank.app import run

    print(f"LegacyCore mock listening on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    run(host=args.host, port=args.port)
    return 0


# ==========================================================================
# discover
# ==========================================================================


def _cmd_discover(args: argparse.Namespace) -> int:
    config.load_env()

    inputs = config.parse_inputs(args.input)
    sensitive_names = set(_ALWAYS_SENSITIVE) | set(args.sensitive or [])

    fake_script = _load_fake_script(args.fake_llm) if args.fake_llm else None
    # Raises a clear SystemExit naming the missing env var when no key is
    # configured and no --fake-llm script was given -- BEFORE anything
    # Playwright-related is imported or launched.
    provider = config.build_provider(provider=args.provider, model=args.model, fake_script=fake_script)

    capability_id = args.capability_id or _slugify(args.goal) or "capability"
    name = args.name or (args.goal or "").strip() or capability_id
    out_path = Path(args.out) if args.out else config.CAPABILITIES_DIR / f"{capability_id}.json"

    policy = _load_policy(args.policy or config.DEFAULT_POLICY)
    entry_url = args.url or config.DEFAULT_MOCK_URL

    redactor = Redactor({n: inputs[n] for n in sensitive_names if n in inputs})
    run_id = new_run_id("disc")
    run_log = RunLog(config.EVIDENCE_RUNS, run_id, redactor=redactor)

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            try:
                context = browser.new_context()
                page = context.new_page()
                surface = WebSurface(page)
                try:
                    loop = DiscoveryLoop(
                        surface=surface,
                        run_log=run_log,
                        redactor=redactor,
                        provider=provider,
                        goal=args.goal,
                        inputs=inputs,
                        sensitive_input_names=sensitive_names,
                        policy=policy,
                        max_steps=args.max_steps,
                        entry_url=entry_url,
                    )
                    trace = loop.run()
                finally:
                    surface.close()
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001 - never let a browser/setup crash traceback out
        print(f"hands: discovery run failed to start or crashed: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    artifact_path: Optional[Path] = None
    record_error: Optional[str] = None

    if trace.status == "success" and not args.no_record:
        inputs_spec = {
            iname: InputSpec(
                type="string",
                required=True,
                description=(
                    "Sensitive credential, supplied from a vault at run time"
                    if iname in sensitive_names
                    else "Run input"
                ),
                sensitive=(iname in sensitive_names),
                source=("vault" if iname in sensitive_names else "caller"),
                example=(None if iname in sensitive_names else inputs.get(iname)),
            )
            for iname in inputs
        }
        outputs_spec = {
            oname: OutputSpec(type=_sniff_output_type(str(oval)), description=f"Captured output {oname!r}")
            for oname, oval in (trace.outputs or {}).items()
        }
        app = AppInfo(vendor="discovered", product=name, version="0.0.1")
        outcome_library = _load_outcome_library(getattr(args, "outcome_library", None))
        try:
            result = record(
                trace,
                capability_id=capability_id,
                name=name,
                description=args.goal,
                inputs_spec=inputs_spec,
                outputs_spec=outputs_spec,
                policy=policy,
                redactor=redactor,
                outcome_library=outcome_library,
                entry_url=entry_url,
                app=app,
                discovery_run_id=run_id,
                evidence_ref=str(run_log.dir),
                inputs=inputs,
            )
        except RecorderError as e:
            record_error = str(e)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result.capability.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            run_log.write_json("capability.raw.json", result.raw_capability.model_dump(mode="json", by_alias=True))
            run_log.write_json("pruning_log.json", result.pruning_log)
            artifact_path = out_path

    lines = [
        f"status: {trace.status}   stop_reason: {trace.stop_reason}   steps: {len(trace.steps)}",
    ]
    if trace.summary:
        lines.append(f"summary: {trace.summary}")
    if trace.outputs:
        lines.append("outputs:")
        for k, v in trace.outputs.items():
            shown = "[redacted]" if k in sensitive_names else v
            lines.append(f"  {k}: {shown}")
    if trace.usage_total:
        lines.append(f"token usage: {trace.usage_total}")
    lines.append(f"evidence dir: {run_log.dir}")
    if artifact_path is not None:
        lines.append(f"capability written: {artifact_path}")
    elif args.no_record and trace.status == "success":
        lines.append("capability not recorded (--no-record)")
    if record_error is not None:
        lines.append(f"capability was NOT recorded: {record_error}")
    print(redactor.text("\n".join(lines)))

    if trace.status == "needs_human":
        return 2
    if trace.status == "success" and artifact_path is not None:
        return 0
    return 1


# ==========================================================================
# replay
# ==========================================================================


def _print_replay_summary(result: Any, redactor: Redactor) -> None:
    lines: list[str] = [
        f"run_id: {result.run_id}",
        f"capability: {result.capability_ref.id} v{result.capability_ref.version}",
        f"status: {result.status.value}",
    ]
    if result.outcome is not None:
        lines.append(f"outcome: {result.outcome.code} ({result.outcome.kind}) -- {result.outcome.message}")
    if result.outputs:
        lines.append("outputs:")
        for k, v in result.outputs.items():
            lines.append(f"  {k}: {v}")
    lines.append(f"outputs_complete: {result.outputs_complete}")
    if result.steps:
        lines.append("steps:")
        for s in result.steps:
            strat = f"  strategy={s.strategy_used} rank={s.strategy_rank}" if s.strategy_used is not None else ""
            note = f"  note={s.note!r}" if s.note else ""
            lines.append(f"  {s.step_id:<6} {s.status:<15}{strat}{note}")
    if result.drift_signals:
        lines.append("drift signals:")
        for d in result.drift_signals:
            lines.append(f"  - {d}")
    if result.human_interventions:
        lines.append(f"human interventions: {', '.join(result.human_interventions)}")
    if result.side_effects_uncertain:
        lines.append("WARNING: side_effects_uncertain -- a risky step failed after its action was performed")
    if result.failure is not None:
        f = result.failure
        lines.append(f"failure category: {f.category.value}")
        if f.step:
            lines.append(f"  step: {f.step}")
        if f.expected:
            lines.append(f"  expected: {f.expected}")
        if f.observed is not None:
            lines.append(
                f"  observed: url={f.observed.url!r} title={f.observed.title!r} "
                f"text={f.observed.text_excerpt[:200]!r}"
            )
        if f.suggested_action:
            lines.append(f"  suggested_action: {f.suggested_action}")
        if f.evidence:
            lines.append(f"  evidence: {f.evidence}")
    lines.append(f"evidence dir: {result.evidence_dir}")
    print(redactor.text("\n".join(lines)))


def _print_repeat_summary(results: list[Any], redactor: Redactor) -> None:
    lines: list[str] = [f"repeated {len(results)} time(s)"]
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    lines.append("status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    flaky_steps: dict[str, int] = {}
    for r in results:
        for s in r.steps:
            if s.strategy_rank:
                flaky_steps[s.step_id] = max(flaky_steps.get(s.step_id, 0), s.strategy_rank)
    if flaky_steps:
        lines.append(
            "steps that ever needed a non-zero strategy rank: "
            + ", ".join(f"{k} (max rank {v})" for k, v in sorted(flaky_steps.items()))
        )
    else:
        lines.append("no step ever needed a non-zero strategy rank across all runs")
    print(redactor.text("\n".join(lines)))


def _cmd_replay(args: argparse.Namespace) -> int:
    config.load_env()

    inputs = config.parse_inputs(args.input)
    data = _read_json_file(args.capability_path)

    rewrite_netloc: Optional[tuple[str, str]] = None
    if args.base_url:
        rewrite_netloc = _rewrite_capability_base_url(data, args.base_url)

    try:
        capability = Capability.model_validate(data)
    except ValidationError as e:
        print(f"hands: {args.capability_path!r} does not match the Capability schema: {e}", file=sys.stderr)
        return 1

    problems = capability.validate_semantics()
    if problems:
        print(f"hands: capability {capability.id!r} failed validate_semantics():", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    policy = _load_policy(args.policy or config.DEFAULT_POLICY, rewrite_netloc=rewrite_netloc)

    redactor = Redactor(
        {
            name: inputs[name]
            for name, spec in capability.contract.inputs.items()
            if spec.sensitive and isinstance(inputs.get(name), str)
        }
    )

    repeat = max(1, args.repeat)
    results: list[Any] = []
    try:
        from playwright.sync_api import sync_playwright
        from hands.replay.engine import ReplayEngine

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            try:
                for _ in range(repeat):
                    context = browser.new_context()
                    page = context.new_page()
                    surface = WebSurface(page)
                    try:
                        run_id = new_run_id("replay")
                        run_log = RunLog(config.EVIDENCE_RUNS, run_id)
                        engine = ReplayEngine(surface, policy, run_log)
                        result = engine.run(capability, inputs, approved_by=args.approved_by)
                        results.append(result)
                    finally:
                        surface.close()
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001 - never let a browser/setup crash traceback out
        print(f"hands: replay failed to start or crashed: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    if repeat > 1:
        _print_repeat_summary(results, redactor)
        codes = [r.exit_code() for r in results]
        return 0 if all(c == 0 for c in codes) else next(c for c in codes if c != 0)

    result = results[0]
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _print_replay_summary(result, redactor)
    return result.exit_code()


# ==========================================================================
# schema export
# ==========================================================================


def _cmd_schema_export(args: argparse.Namespace) -> int:
    config.load_env()

    schema_dir = config.CAPABILITIES_DIR / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    schema_path = schema_dir / "capability.schema.json"
    schema_path.write_text(
        json.dumps(Capability.model_json_schema(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {schema_path}")

    written = 0
    for cap_path in sorted(config.CAPABILITIES_DIR.glob("*.json")):
        try:
            cap_data = json.loads(cap_path.read_text(encoding="utf-8"))
            cap = Capability.model_validate(cap_data)
        except Exception as e:  # noqa: BLE001 - one bad file must not abort the whole export
            print(f"hands: skipping {cap_path} (not a valid capability): {e.__class__.__name__}: {e}", file=sys.stderr)
            continue
        tool_def = cap.to_tool_definition()
        tool_path = schema_dir / f"{cap.id}.tool.json"
        tool_path.write_text(json.dumps(tool_def, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {tool_path}")
        written += 1

    if written == 0:
        print(f"hands: no valid capabilities found under {config.CAPABILITIES_DIR}", file=sys.stderr)
    return 0


# ==========================================================================
# inspect
# ==========================================================================


def _cmd_inspect(args: argparse.Namespace) -> int:
    config.load_env()

    data = _read_json_file(args.capability_path)
    try:
        cap = Capability.model_validate(data)
    except ValidationError as e:
        print(f"hands: {args.capability_path!r} does not match the Capability schema: {e}", file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append(f"Capability: {cap.id}  (name: {cap.name!r}, version: {cap.version})")
    approved_by = f"  approved_by: {cap.approval.approved_by}" if cap.approval.approved_by else ""
    lines.append(f"  approval status: {cap.approval.status}{approved_by}")
    computed_hash = cap.compute_content_hash()
    hash_note = (
        "matches recorded content_hash"
        if cap.content_hash == computed_hash
        else "DOES NOT MATCH recorded content_hash -- behavioural content changed since it was hashed"
    )
    lines.append(f"  content_hash: {cap.content_hash}  ({hash_note})")
    lines.append(f"  description: {cap.description}")
    lines.append("")

    lines.append("Contract:")
    lines.append("  inputs:")
    for iname, spec in cap.contract.inputs.items():
        sens = "  [SENSITIVE: never shown to a model, never serialized]" if spec.sensitive else ""
        req = "required" if spec.required else "optional"
        lines.append(f"    - {iname}: {spec.type} ({req}, source={spec.source}){sens} -- {spec.description}")
    lines.append("  outputs:")
    for oname, ospec in cap.contract.outputs.items():
        lines.append(f"    - {oname}: {ospec.type} (from step {ospec.source_step}) -- {ospec.description}")
    lines.append(f"  outcome_codes: {', '.join(cap.contract.outcome_codes) or '(none)'}")
    lines.append(f"  side_effects: {cap.contract.side_effects}   idempotent: {cap.contract.idempotent}")
    lines.append("")

    lines.append(f"Target: {cap.target.app.vendor} {cap.target.app.product} {cap.target.app.version}")
    lines.append(f"  entry_url: {cap.target.entry_url}")
    lines.append("")

    lines.append("Screens:")
    for sid, screen in cap.screens.items():
        lines.append(f"  - {sid}: {screen.description}  (url_pattern={screen.url_pattern!r})")
    lines.append("")

    lines.append("Steps:")
    for step in cap.steps:
        target_desc = step.target.description if step.target else "(no target)"
        strat_kinds = ", ".join(s.kind for s in step.target.strategies) if step.target else "(none)"
        risk_note = f" -- {step.risk.reason}" if step.risk.reason else ""
        lines.append(f"  {step.id}  {step.kind}  {step.screen}  {step.intent}")
        lines.append(f"      target: {target_desc}")
        lines.append(f"      strategies (in order): {strat_kinds}")
        lines.append(f"      risk: {step.risk.level}{risk_note}")
    lines.append("")

    lines.append("Outcomes:")
    for o in cap.outcomes:
        lines.append(f"  - {o.code} ({o.kind}, origin={o.origin}): {o.message}")
    lines.append("")

    lines.append(f"Success: {cap.success.description}")

    print("\n".join(lines))
    return 0


# ==========================================================================
# operator
# ==========================================================================
#
# The handoff layer (hands/handoff/) ships a state-machine controller and an
# HTTP console for it, but had no CLI path onto either -- this subcommand
# group is that path. `status`/`list` read evidence off disk (never from a
# live process): "who is in control?" must be answerable from a crashed run
# just as well as a live one, which is the whole point of
# `SessionController` persisting `control_state.json` on every transition
# (see hands/handoff/controller.py's module docstring). `claim`/`resume`
# instead talk to an ALREADY-RUNNING `OperatorServer` over HTTP, since that
# server fronts a live, in-process `SessionController` that a separate CLI
# invocation cannot reach any other way (see hands/handoff/operator_server.py's
# module docstring for why it is HTTP-enqueue-only in the first place).


def _most_recent_run_dir(base: Path) -> Optional[Path]:
    """The most recently modified subdirectory of `base`, or None if `base`
    doesn't exist or has none. mtime, not name, so this is correct even for
    run ids that don't happen to be lexicographically sortable."""
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_run_dir(run_dir_arg: Optional[str]) -> Path:
    """`--run-dir` if given (must exist), otherwise the most recently
    modified run under `config.EVIDENCE_RUNS` -- a clear SystemExit, never a
    traceback, if neither resolves to anything."""
    if run_dir_arg:
        p = Path(run_dir_arg)
        if not p.is_dir():
            raise SystemExit(f"hands: --run-dir {run_dir_arg!r} is not a directory")
        return p
    found = _most_recent_run_dir(config.EVIDENCE_RUNS)
    if found is None:
        raise SystemExit(
            f"hands: no --run-dir given and no runs found under {config.EVIDENCE_RUNS} "
            "-- pass --run-dir explicitly"
        )
    return found


def _cmd_operator_status(args: argparse.Namespace) -> int:
    config.load_env()
    run_dir = _resolve_run_dir(args.run_dir)
    state_path = run_dir / "control_state.json"
    if not state_path.is_file():
        raise SystemExit(
            f"hands: no control_state.json in {run_dir} "
            "(not a handoff-capable run, or no control transition has happened yet)"
        )
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"hands: could not read {state_path}: {e}")
    # Validate through the real model, not just dict.get(): a field renamed
    # in hands/schema/control.py should surface here as a clear error, not
    # silently print `None` for whatever key moved.
    try:
        record = ControlStateRecord.model_validate(raw)
    except ValidationError as e:
        raise SystemExit(f"hands: {state_path} does not match the ControlStateRecord schema: {e}")

    lines = [
        f"run_dir: {run_dir}",
        f"run_id: {record.run_id}",
        f"state: {record.state.value}",
        f"holder: {record.holder.value}",
        f"epoch: {record.epoch}",
    ]
    if record.handoff_id:
        lines.append(f"handoff_id: {record.handoff_id}")
    lines.append(f"since: {record.since}")
    if record.reason:
        lines.append(f"reason: {record.reason}")
    print("\n".join(lines))
    return 0


def _cmd_operator_list(args: argparse.Namespace) -> int:
    config.load_env()
    run_dir = _resolve_run_dir(args.run_dir)
    files = sorted(run_dir.glob("intervention-*.json"))
    if not files:
        print(f"no interventions recorded in {run_dir}")
        return 0
    for f in files:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"hands: skipping {f} (could not read): {e}", file=sys.stderr)
            continue
        try:
            data = InterventionRequest.model_validate(raw)
        except ValidationError as e:
            print(f"hands: skipping {f} (does not match the InterventionRequest schema): {e}", file=sys.stderr)
            continue
        print(f"id: {data.id}  reason_code: {data.reason_code}  mode: {data.mode}")
        if data.summary:
            print(f"    summary: {data.summary}")
    return 0


def _operator_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST `payload` as JSON to a running `OperatorServer` and return its
    (also-JSON) body. Every server endpoint in `operator_server.py`'s HTTP
    contract answers 202 the instant a command is enqueued (never waiting
    for it to be applied -- see that module's docstring); anything else, or
    a connection failure, becomes a one-line `SystemExit`, never a traceback.
    """
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.URLError as e:
        raise SystemExit(f"hands: could not reach operator server at {url}: {e.reason if hasattr(e, 'reason') else e}")
    if status != 202:
        raise SystemExit(f"hands: operator server at {url} returned unexpected status {status}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _cmd_operator_claim(args: argparse.Namespace) -> int:
    config.load_env()
    url = f"http://{args.host}:{args.port}/interventions/{args.intervention_id}/claim"
    result = _operator_post(url, {"operator_id": args.operator_id})
    print(f"claim queued for intervention {args.intervention_id!r} (operator_id={args.operator_id!r}): {result}")
    return 0


def _cmd_operator_resume(args: argparse.Namespace) -> int:
    config.load_env()
    payload: dict[str, Any] = {"resume_mode": args.mode}
    if args.note:
        payload["notes"] = args.note
    url = f"http://{args.host}:{args.port}/interventions/{args.intervention_id}/resume"
    result = _operator_post(url, payload)
    print(f"resume (mode={args.mode!r}) queued for intervention {args.intervention_id!r}: {result}")
    return 0


def _cmd_operator_serve(args: argparse.Namespace) -> int:
    """Start a bare operator HTTP console -- a fresh, page-less
    `SessionController` (no live browser attached) wrapped in an
    `OperatorServer` -- and drain queued commands against it until
    interrupted. `claim`/`release`/`abort` all still go through the real
    `SessionController` state machine and are persisted to
    `control_state.json` in the printed run directory exactly as a live
    discover/replay run's handoff would be; the only thing missing is an
    actual Playwright page for `claim()` to install its human-action
    recorder into (a command that needs one is caught and logged, not
    crashed -- see `OperatorServer.drain`'s docstring).
    """
    config.load_env()
    import time

    from hands.handoff.controller import SessionController
    from hands.handoff.operator_server import OperatorServer
    from hands.schema.control import ControlState

    policy = _load_policy(config.DEFAULT_POLICY)
    root = Path(args.run_dir) if args.run_dir else config.EVIDENCE_RUNS
    run_id = new_run_id("operator")
    run_log = RunLog(root, run_id)
    controller = SessionController(run_id, run_log, policy)

    server = OperatorServer(controller, host=args.host, port=args.port)
    bound_port = server.start()
    print(f"hands operator server listening on http://{args.host}:{bound_port}/  (Ctrl+C to stop)")
    print(f"run dir: {run_log.dir}")
    # NOTE: --run-dir means something different for `status`/`list` -- it
    # names the exact run directory to read, not a root a new one gets
    # created under. Copy the FULL path printed above, not whatever
    # --run-dir was passed to THIS command.
    print(f"from another terminal: hands operator status --run-dir {run_log.dir}")
    try:
        while True:
            server.drain(controller)
            if controller.state in (ControlState.ABANDONED, ControlState.FINISHED):
                print(f"run ended: {controller.state.value}")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("hands: operator server stopped (Ctrl+C)")
    finally:
        server.stop()
    return 0


# ==========================================================================
# argument parsing
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hands",
        description="Computer-use automation: discover, record, replay, hand off.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sm = sub.add_parser("serve-mock", help="Run the LegacyCore mock target app.")
    sm.add_argument("--host", default="127.0.0.1")
    sm.add_argument("--port", type=int, default=5050)
    sm.set_defaults(func=_cmd_serve_mock)

    d = sub.add_parser(
        "discover",
        help="Run the discovery loop against a live app and (by default) record a capability.",
        description=(
            "Drive an LLM (or a --fake-llm script) through a live browser to accomplish --goal, "
            "then curate the resulting trace into a replayable capability artifact."
        ),
    )
    d.add_argument("--goal", required=True, help="Natural-language goal for the discovery run.")
    d.add_argument("--url", default=None, help=f"Entry URL to open (default: {config.DEFAULT_MOCK_URL}).")
    d.add_argument(
        "--input", action="append", default=[], metavar="name=value",
        help="A run input as name=value. A value of the form @env:VAR is read from the environment "
        "instead of appearing on the command line. May be repeated.",
    )
    d.add_argument(
        "--sensitive", action="append", default=[], metavar="NAME",
        help="Mark an input sensitive: its value is never shown to the model and never serialized "
        "into the recorded capability (a {{placeholder}} is used instead). May be repeated. "
        "An input literally named 'password' is always treated as sensitive by default, "
        "whether or not it is listed here.",
    )
    d.add_argument("--out", default=None, help="Where to write the curated capability JSON (default: capabilities/<capability-id>.json).")
    d.add_argument("--provider", choices=("anthropic", "openai"), default=None, help="LLM provider (default: auto-detected from the environment).")
    d.add_argument("--model", default=None, help="Model name override.")
    d.add_argument(
        "--fake-llm", default=None, metavar="PATH",
        help="Run with a scripted FakeLLM instead of a real provider -- no API key required. "
        "PATH is either a JSON list of {\"tool\":...,\"input\":...} turns, or a JSONL transcript "
        "(the format RunLog.transcript writes).",
    )
    d.add_argument("--headed", action="store_true", help="Run the browser headed instead of headless.")
    d.add_argument("--max-steps", type=int, default=40, help="Maximum discovery steps before stopping (default: 40).")
    d.add_argument("--capability-id", default=None, help="Capability id to record (default: derived from --goal).")
    d.add_argument("--name", default=None, help="Human-readable capability name (default: derived from --goal).")
    d.add_argument("--policy", default=None, help=f"Path to policy.yaml (default: {config.DEFAULT_POLICY}).")
    d.add_argument("--no-record", action="store_true", help="Run discovery without recording a capability artifact.")
    d.add_argument(
        "--outcome-library",
        default=None,
        metavar="PATH",
        help=(
            "Curated per-app outcome definitions to merge into the recorded capability "
            "(default: capabilities/library/legacycore.outcomes.json if present). A discovery "
            "run only walks the happy path, so it cannot observe conditions like 'record not "
            "found' or 'system busy'; without these, replay can only report that it is on an "
            "unexpected screen, not which business outcome occurred. Pass 'none' to skip."
        ),
    )
    d.set_defaults(func=_cmd_discover)

    r = sub.add_parser(
        "replay",
        help="Deterministically replay a recorded capability against a live app.",
        description=(
            "Replay a saved capability with no model in the loop. Exit code follows "
            "ReplayResult.exit_code(): 0 for SUCCESS or a legitimate BUSINESS_OUTCOME (e.g. "
            "'no such member' is not a failure), 2 for NEEDS_HUMAN, 3 for POLICY_BLOCKED, "
            "1 for everything else (a real failure)."
        ),
    )
    r.add_argument("capability_path", metavar="CAPABILITY_PATH")
    r.add_argument("--input", action="append", default=[], metavar="name=value", help="A run input as name=value (see 'discover --input' for @env:VAR). May be repeated.")
    r.add_argument("--approved-by", default=None, metavar="NAME", help="Confirm this replay for any risky step that needs human confirmation.")
    r.add_argument("--headed", action="store_true", help="Run the browser headed instead of headless.")
    r.add_argument("--policy", default=None, help=f"Path to policy.yaml (default: {config.DEFAULT_POLICY}).")
    r.add_argument("--json", action="store_true", help="Print the raw ReplayResult JSON instead of a human-readable summary.")
    r.add_argument("--base-url", default=None, metavar="URL", help="Rewrite the capability's entry URL and any embedded host:port to run against a different host (e.g. a locally started mock app on a free port).")
    r.add_argument("--repeat", type=int, default=1, metavar="N", help="Run the replay N times and report a stability summary instead of one result.")
    r.set_defaults(func=_cmd_replay)

    s = sub.add_parser("schema", help="Capability JSON Schema and per-capability tool-definition export.")
    s_sub = s.add_subparsers(dest="schema_command", required=True)
    se = s_sub.add_parser(
        "export",
        help="Write capabilities/schema/capability.schema.json, plus one *.tool.json per capabilities/*.json.",
    )
    se.set_defaults(func=_cmd_schema_export)

    i = sub.add_parser("inspect", help="Print a human-readable review of a saved capability artifact.")
    i.add_argument("capability_path", metavar="CAPABILITY_PATH")
    i.set_defaults(func=_cmd_inspect)

    o = sub.add_parser(
        "operator",
        help="Operator console: inspect and drive a handoff from the terminal.",
        description=(
            "Read a run's control state / open interventions from disk (status, list), or drive an "
            "already-running operator HTTP server (serve, claim, resume)."
        ),
    )
    o_sub = o.add_subparsers(dest="operator_command", required=True)

    o_serve = o_sub.add_parser(
        "serve", help="Start a bare operator HTTP console (no live browser) for a new run."
    )
    o_serve.add_argument("--host", default="127.0.0.1")
    o_serve.add_argument("--port", type=int, default=_DEFAULT_OPERATOR_PORT)
    o_serve.add_argument(
        "--run-dir", default=None,
        help=(
            f"Root directory under which to create the new run's evidence directory "
            f"(default: {config.EVIDENCE_RUNS}). NOTE: this is a ROOT, not the run directory itself -- "
            "pass the full path this command prints at startup to `status`/`list --run-dir`, not this value."
        ),
    )
    o_serve.set_defaults(func=_cmd_operator_serve)

    o_status = o_sub.add_parser(
        "status", help="Print the current control state for a run, read from control_state.json on disk."
    )
    o_status.add_argument(
        "--run-dir", default=None,
        help=(
            "The run's OWN evidence directory, e.g. evidence/runs/<run_id>/ (default: the most recently "
            "modified run under evidence/runs/). Not the --run-dir root passed to `operator serve`."
        ),
    )
    o_status.set_defaults(func=_cmd_operator_status)

    o_list = o_sub.add_parser(
        "list", help="List intervention requests recorded for a run, read from intervention-*.json on disk."
    )
    o_list.add_argument(
        "--run-dir", default=None,
        help=(
            "The run's OWN evidence directory, e.g. evidence/runs/<run_id>/ (default: the most recently "
            "modified run under evidence/runs/). Not the --run-dir root passed to `operator serve`."
        ),
    )
    o_list.set_defaults(func=_cmd_operator_list)

    o_claim = o_sub.add_parser("claim", help="Claim an open intervention on a running operator server.")
    o_claim.add_argument("intervention_id", metavar="INTERVENTION_ID")
    o_claim.add_argument("--operator-id", default="cli-operator")
    o_claim.add_argument("--host", default="127.0.0.1")
    o_claim.add_argument("--port", type=int, default=_DEFAULT_OPERATOR_PORT)
    o_claim.set_defaults(func=_cmd_operator_claim)

    o_resume = o_sub.add_parser(
        "resume", help="Resume (or abort) a claimed intervention on a running operator server."
    )
    o_resume.add_argument("intervention_id", metavar="INTERVENTION_ID")
    o_resume.add_argument("--mode", choices=("auto", "retry_step", "abort"), default="auto")
    o_resume.add_argument("--note", default=None, metavar="TEXT")
    o_resume.add_argument("--host", default="127.0.0.1")
    o_resume.add_argument("--port", type=int, default=_DEFAULT_OPERATOR_PORT)
    o_resume.set_defaults(func=_cmd_operator_resume)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return args.func(args)
    except SystemExit as e:
        code = e.code
        if isinstance(code, str):
            print(code if code.startswith("hands:") else f"hands: {code}", file=sys.stderr)
            return 1
        if code is None:
            return 0
        return int(code) if isinstance(code, int) else 1
