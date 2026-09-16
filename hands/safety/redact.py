"""Redaction. Applied to everything that is persisted: events, transcripts,
artifacts, human-action logs, intervention requests.

Two layers:
1. Known sensitive values (inputs declared `sensitive`, e.g. a password) are
   replaced by an ASCII marker `[redacted:<name>]` wherever they occur.
2. Pattern scrubbing for regulated identifiers that could appear in page text
   even when no input was declared sensitive (SSN, card numbers).

The marker is ASCII on purpose: Windows consoles under cp1252 mangle
non-ASCII markers, and evidence must be greppable everywhere.
"""
from __future__ import annotations

import re
from typing import Any

MARKER = "[redacted:{name}]"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]


class Redactor:
    def __init__(self, sensitive: dict[str, str] | None = None) -> None:
        # name -> value ; only values with length >= 3 are substituted, to
        # avoid shredding unrelated text on trivially short values.
        self._values: list[tuple[str, str]] = sorted(
            [(n, v) for n, v in (sensitive or {}).items() if v and len(v) >= 3],
            key=lambda nv: -len(nv[1]),
        )

    def add(self, name: str, value: str) -> None:
        if value and len(value) >= 3:
            self._values.append((name, value))
            self._values.sort(key=lambda nv: -len(nv[1]))

    def has_value(self, value: str) -> str | None:
        """Return the sensitive name if `value` equals a known sensitive value."""
        for n, v in self._values:
            if value == v:
                return n
        return None

    def text(self, s: str) -> str:
        if not s:
            return s
        for name, value in self._values:
            if value in s:
                s = s.replace(value, MARKER.format(name=name))
        for name, pat in _PATTERNS:
            s = pat.sub(MARKER.format(name=name), s)
        return s

    def obj(self, o: Any) -> Any:
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, dict):
            return {k: self.obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.obj(v) for v in o]
        return o
