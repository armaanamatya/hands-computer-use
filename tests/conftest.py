r"""Shared fixtures for tests/*.py: ONE Playwright driver, ONE Chromium
`browser`, and ONE mock LegacyCore `mock_server`, for the whole test
session.

## The bug this file exists to fix

Playwright's sync API tolerates only ONE open `sync_playwright()` context
per process at a time: opening a second one while a first is still alive
raises "It looks like you are using Playwright Sync API inside the asyncio
loop." (verified directly: opening a second `sync_playwright()` while a
first one's `browser` is still around reproduces this exact error, even
though the first context is sitting idle at that moment -- Playwright's
sync wrapper marks a thread-local "current running loop" for as long as its
own context is open, not just while a call is in flight).

Before this file existed, tests/test_replay.py, tests/test_handoff.py,
tests/test_agent_loop.py and tests/test_end_to_end.py each opened their own
`sync_playwright()` context, at various scopes. That only worked by an
accident of file-collection order (module-scoped ones tore down before the
next file's opened) -- `pytest tests/test_replay.py tests/test_handoff.py`
(replay's session-scoped context forced open first) broke it. Centralizing
to one instance, shared via this conftest, removes that fragility.

## The one remaining wrinkle: tests/test_cli.py

tests/test_cli.py drives `hands.cli.main()`, which -- for `replay` and
`discover` -- opens and fully closes its OWN `sync_playwright()` context
per call (see hands/cli.py's `_cmd_replay` / `_cmd_discover`; that module is
out of scope to change here). If the `browser` fixture below is already
alive when one of those calls runs, it collides with it exactly the same
way. Default (alphabetical) collection order puts tests/test_agent_loop.py
-- which DOES request `browser` -- before tests/test_cli.py, so a plain
`uv run pytest -q` would instantiate the shared `browser` first and then
break test_cli.py's own Playwright usage. `pytest_collection_modifyitems`
below reorders tests/test_cli.py's items to run before anything that might
request `browser`, regardless of invocation order, so its self-contained
open/close cycles never overlap with the shared one. (Grep confirms
tests/test_cli.py is the only file with this shape: every other file either
consumes the shared `browser` fixture below, or never touches Playwright at
all.)
"""
from __future__ import annotations

import threading
from typing import Any, Iterator

import pytest


@pytest.fixture(scope="session")
def mock_server() -> Iterator[int]:
    """The real LegacyCore mock app (mockbank.app), on a free localhost
    port, in a background thread, for the whole test session. `threaded=True`
    matters: the frameset loads `/nav` and `/main` as two concurrent
    requests, and a single-threaded dev server can stall on that.

    Yields the PORT (int). A couple of files (tests/test_agent_loop.py,
    tests/test_end_to_end.py) historically yielded a full base URL instead;
    they now build that URL at their own call sites
    (`f"http://127.0.0.1:{mock_server}/"`) rather than this fixture
    guessing which shape a given call site wants.
    """
    from werkzeug.serving import make_server

    from mockbank.app import app as flask_app

    srv = make_server("127.0.0.1", 0, flask_app, threaded=True)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        srv.shutdown()
        t.join(timeout=5)


@pytest.fixture(scope="session")
def browser() -> Iterator[Any]:
    """ONE Chromium instance, launched from ONE `sync_playwright()` context,
    for the whole test session. `pytest.skip`s (rather than fails) if
    Playwright is not installed or Chromium cannot launch -- the same
    skip behaviour the original per-module fixtures had.

    Tests needing isolated state build their own fresh `browser.new_context()`
    / `browser.new_page()` per test (see e.g. tests/test_replay.py's
    `surface` fixture, tests/test_handoff.py's `_new_session` helper): a
    shared `browser` process is fine to share, a shared context/page is not
    -- that is what would leak Flask session state between tests.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not installed")
        return

    pw = sync_playwright().start()
    try:
        b = pw.chromium.launch(headless=True)
    except Exception as e:  # noqa: BLE001 - any launch failure means "skip", not "fail"
        pw.stop()
        pytest.skip(f"Chromium is not available: {e}")
        return
    yield b
    b.close()
    pw.stop()


def pytest_collection_modifyitems(config: Any, items: list) -> None:
    """Run tests/test_cli.py's items before anything that might request the
    shared `browser` fixture above -- see this module's docstring for why.
    A no-op whenever tests/test_cli.py isn't part of the current run.
    """
    cli_items = [it for it in items if it.path.name == "test_cli.py"]
    if not cli_items:
        return
    other_items = [it for it in items if it.path.name != "test_cli.py"]
    items[:] = cli_items + other_items
