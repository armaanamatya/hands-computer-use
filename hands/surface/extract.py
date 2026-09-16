"""Read one typed output value out of the live page.

`run_extraction()` implements `Extraction.from_`'s four cases. The
important one is `table_cell`: it locates the TABLE (via `target`), then
reads the header row and matches columns/rows by REGEX rather than by
position, so "the savings balance" can be read off a multi-row accounts
table without the replay depending on which row order the app happens to
render this time. `target` for a `table_cell` extraction therefore points
at the table itself, not at a specific cell -- resolving to a specific
cell would silently reintroduce the row-order dependency this extraction
type exists to avoid.

`parse` (money/int/date/string, with an optional regex+group applied
first) is shared post-processing applied uniformly regardless of which
`from_` produced the raw string.
"""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from hands.schema.capability import Extraction, ParseSpec
from hands.surface import locator as locator_mod
from hands.surface.conditions import ConditionContext


class _NoCoordinateFallback:
    """See `hands.surface.conditions._NoCoordinateFallback` -- same
    purpose, duplicated locally rather than imported so this module does
    not reach into another module's underscore-prefixed internals."""

    allow_coordinate_fallback = False


def run_extraction(surface: Any, extraction: Extraction, ctx: ConditionContext) -> tuple[Any, Optional[str]]:
    """Read and parse one extraction. Returns `(value, error)`: exactly one
    of the two is non-None. A missing/unmatched extraction is never an
    exception -- it's a clear error string, which the replay engine turns
    into `EXTRACTION_FAILED`.
    """
    raw, err = _read_raw(surface, extraction, ctx)
    if err is not None:
        return None, err
    return _parse(raw, extraction.parse)


def _read_raw(surface: Any, extraction: Extraction, ctx: ConditionContext) -> tuple[Optional[str], Optional[str]]:
    if extraction.from_ == "table_cell":
        return _read_table_cell(surface, extraction, ctx)

    if extraction.target is None:
        return None, f"extraction from {extraction.from_!r} requires a target"

    policy = ctx.policy or _NoCoordinateFallback()
    try:
        result = locator_mod.resolve(
            surface, extraction.target, policy, locator_mod.ResolveContext(step_id=ctx.step_id)
        )
    except locator_mod.TargetResolutionError as e:
        return None, f"extraction target {extraction.target.description!r} did not resolve: {e.message}"

    if result.locator is None:
        return None, "extraction target resolved to a coordinate-fallback location; no readable value"

    try:
        if extraction.from_ == "text":
            return result.locator.inner_text(), None
        if extraction.from_ == "value":
            return result.locator.input_value(), None
        if extraction.from_ == "attribute":
            if not extraction.attribute:
                return None, "extraction from 'attribute' requires extraction.attribute"
            value = result.locator.get_attribute(extraction.attribute)
            if value is None:
                return None, f"attribute {extraction.attribute!r} is not present on the resolved element"
            return value, None
    except Exception as e:  # noqa: BLE001
        return None, f"could not read {extraction.from_!r} from the resolved element: {e}"

    return None, f"unsupported extraction.from {extraction.from_!r}"


def _read_table_cell(surface: Any, extraction: Extraction, ctx: ConditionContext) -> tuple[Optional[str], Optional[str]]:
    if extraction.target is None or extraction.table is None:
        return None, "table_cell extraction requires both target and table"

    policy = ctx.policy or _NoCoordinateFallback()
    try:
        result = locator_mod.resolve(
            surface, extraction.target, policy, locator_mod.ResolveContext(step_id=ctx.step_id)
        )
    except locator_mod.TargetResolutionError as e:
        return None, f"table target {extraction.target.description!r} did not resolve: {e.message}"

    table = result.locator
    if table is None:
        return None, "table_cell extraction target resolved to a coordinate-fallback location"

    try:
        rows = table.locator("tr")
        n_rows = rows.count()
        if n_rows == 0:
            return None, "resolved table has no rows"
        headers = [h.strip() for h in rows.nth(0).locator("th,td").all_inner_texts()]
    except Exception as e:  # noqa: BLE001
        return None, f"could not read the table's header row: {e}"

    header_match = extraction.table.header_match
    value_col = next((i for i, h in enumerate(headers) if re.search(header_match, h)), None)
    if value_col is None:
        return None, f"no header matches {header_match!r} (headers seen: {headers!r})"

    row_where = extraction.table.row_where
    # `row_where.header` names a column by its exact header text (it finds
    # WHICH column selects the row, not a pattern within that column --
    # `row_where.matches`, applied per-row below, is the regex).
    key_col = next((i for i, h in enumerate(headers) if h == row_where.header), None)
    if key_col is None:
        return None, f"no header named {row_where.header!r} to select a row by (headers seen: {headers!r})"

    for r in range(1, n_rows):
        try:
            cells = rows.nth(r).locator("td,th").all_inner_texts()
        except Exception:  # noqa: BLE001 - a stale/detached row; skip it
            continue
        if key_col >= len(cells):
            continue
        if not re.search(row_where.matches, cells[key_col].strip()):
            continue
        if value_col >= len(cells):
            return None, f"row matched {row_where.matches!r} but has no cell at column {value_col}"
        return cells[value_col], None

    return None, f"no row where {row_where.header!r} matches {row_where.matches!r}"


def _parse(raw: Optional[str], spec: Optional[ParseSpec]) -> tuple[Any, Optional[str]]:
    if raw is None:
        return None, "no value to parse"
    text: str = raw
    if spec is None:
        return text.strip(), None

    if spec.regex:
        m = re.search(spec.regex, text)
        if not m:
            return None, f"parse.regex {spec.regex!r} did not match extracted text {raw!r}"
        try:
            text = m.group(spec.group)
        except IndexError as e:
            return None, f"parse.regex group {spec.group} is not present in the match: {e}"
    text = text.strip()

    if spec.type == "string":
        return text, None

    if spec.type == "int":
        cleaned = text.replace(",", "").strip()
        try:
            return int(cleaned), None
        except ValueError:
            return None, f"could not parse {text!r} as int"

    if spec.type == "money":
        try:
            return _parse_money(text), None
        except (InvalidOperation, ValueError) as e:
            return None, f"could not parse {text!r} as money: {e}"

    if spec.type == "date":
        parsed = _parse_date(text)
        if parsed is None:
            return None, f"could not parse {text!r} as a date"
        return parsed, None

    return None, f"unknown parse type {spec.type!r}"


def _parse_money(raw: str) -> str:
    """Strip currency symbols/commas, treat a parenthesized or leading-minus
    value as negative, and return a canonical decimal STRING (e.g.
    "2340.55", "-4500.00").

    A string, not a float: `OutputSpec.type == "money"` maps to JSON-Schema
    type "string" in `Capability.to_tool_definition` (see
    `hands/schema/capability.py`'s `_OUTPUT_JSON_TYPE`), precisely because
    currency is not safely representable as IEEE-754 float (rounding error
    on a money amount is a real-money bug, not a cosmetic one). `Decimal`
    is used here only to parse and normalize the sign unambiguously across
    both negative conventions this mock app's legacy ledger uses
    ("-4,500.00" and, hypothetically, "(4,500.00)"); the `Decimal` object
    itself is discarded and only its canonical string form is kept, so the
    output stays a plain JSON string end to end.
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty money value")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s.startswith("-"):
        negative = True
        s = s[1:]
    if s in ("", "."):
        raise ValueError(f"no digits in money value {raw!r}")
    dec = Decimal(s)
    if negative:
        dec = -dec
    return str(dec)


def _parse_date(raw: str) -> Optional[str]:
    """Normalize a handful of common legacy-ledger date formats to ISO
    (YYYY-MM-DD). Not used by `lookup_savings_balance`'s own extraction,
    but implemented for completeness of the closed `ParseSpec.type` set."""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None
