"""Project paths, environment loading, and input parsing helpers."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_RUNS = PROJECT_ROOT / "evidence" / "runs"
CAPABILITIES_DIR = PROJECT_ROOT / "capabilities"
DEFAULT_POLICY = PROJECT_ROOT / "policy.yaml"
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MOCK_URL = "http://127.0.0.1:5050/"


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def model_name(explicit: str | None = None) -> str:
    return explicit or os.environ.get("HANDS_MODEL") or DEFAULT_MODEL


def parse_inputs(pairs: list[str]) -> dict[str, str]:
    """Parse --input name=value. A value of the form @env:VAR is read from the
    environment so secrets never appear on the command line or in shell history."""
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--input expects name=value, got {p!r}")
        k, v = p.split("=", 1)
        if v.startswith("@env:"):
            var = v[5:]
            if var not in os.environ:
                raise SystemExit(f"--input {k}: environment variable {var} is not set")
            v = os.environ[var]
        out[k.strip()] = v
    return out
