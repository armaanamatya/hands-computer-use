"""Per-run evidence directory and structured, redacted event log.

Layout of evidence/runs/<run_id>/:
  events.jsonl        one line per event: what happened and why (redacted)
  transcript.jsonl    discovery only: one line per model turn
  screenshots/        step-NN-<label>.png
  dom/                step-NN.html + elements-NN.json (on failure / handoff)
  result.json         the run's terminal result contract
  ...                 intervention-*.json, human-actions.jsonl, control_state.json
All paths recorded inside JSON are POSIX-style and relative to the run dir.
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hands.safety.redact import Redactor


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunLog:
    def __init__(self, root: Path | str, run_id: str, redactor: Optional[Redactor] = None) -> None:
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "screenshots").mkdir(exist_ok=True)
        (self.dir / "dom").mkdir(exist_ok=True)
        self.redactor = redactor or Redactor()
        self._seq = 0
        self._shot = 0

    # ---------------------------------------------------------------- paths
    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def rel(self, p: Path | str | None) -> Optional[str]:
        if p is None:
            return None
        try:
            return Path(p).resolve().relative_to(self.dir.resolve()).as_posix()
        except ValueError:
            return Path(p).as_posix()

    def next_screenshot(self, label: str) -> str:
        self._shot += 1
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        return str(self.dir / "screenshots" / f"step-{self._shot:02d}-{safe}.png")

    def dom_paths(self, label: str) -> tuple[str, str]:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        return str(self.dir / "dom" / f"{safe}.html"), str(self.dir / "dom" / f"{safe}-elements.json")

    # --------------------------------------------------------------- writing
    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        self._seq += 1
        rec = {"seq": self._seq, "ts": _now(), "kind": kind, **fields}
        rec = self.redactor.obj(rec)
        with open(self.dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def transcript(self, entry: dict[str, Any]) -> None:
        entry = {"ts": _now(), **entry}
        entry = self.redactor.obj(entry)
        with open(self.dir / "transcript.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def append_jsonl(self, name: str, entry: dict[str, Any]) -> None:
        entry = self.redactor.obj({"ts": _now(), **entry})
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.redactor.obj(obj), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.redactor.text(text), encoding="utf-8")
        return p
