r"""A minimal local HTTP operator console, stdlib only (`http.server.
ThreadingHTTPServer`) -- no new dependency for what is deliberately a thin
surface: the real design work in this feature is `SessionController`'s
state machine, not the console.

## The constraint this whole module is built around

`ThreadingHTTPServer` serves every request on its own worker thread. The
`SessionController` it fronts owns Playwright objects (`page`, `context`),
and Playwright's SYNC API is thread-bound: calling a sync `Page`/`Frame`/
`Locator` method from a thread other than the one that created it raises.
So request handlers in this module must NEVER call a controller method that
touches Playwright -- that rules out `claim()` (installs the recorder),
`release()` (writes evidence but is still controller-owned state a second
thread must not mutate concurrently with the main thread), `resume()`, and
`abandon()`.

The fix is indirection, not a lock: handler methods only ever (a) read
plain Python attributes off the controller for the two `GET` endpoints --
`state`/`holder`/`epoch`/`handoff_id` are enums/ints/strings, cheap and
safe to read from another thread under the GIL -- or (b) push a `_Command`
onto a `queue.Queue` for the four `POST` endpoints and return immediately
(HTTP 202: queued, not yet applied). The main thread -- the one that owns
Playwright, already looping inside `SessionController.wait_for_human` -- is
the only thread that ever calls `drain(controller)`, which pops every
queued command and applies it via the controller's real methods. This is
also why `wait_for_human` accepts a `drain` callable: it is the one place
in this codebase guaranteed to be on the main thread AND already polling
in a loop, so it is the natural place to also drain the operator queue.

## HTTP contract

    GET  /control                                -> current control snapshot
    GET  /interventions                           -> open (unclaimed) InterventionRequests
    POST /interventions/<id>/claim   {operator_id} -> queue a claim
    POST /interventions/<id>/resume  {resume_mode, notes, record_as} -> queue a release
    POST /interventions/<id>/abort   {reason}      -> queue an abandon

Every POST returns 202 with `{"queued": true, ...}` the instant the command
is enqueued -- it does NOT wait for `drain()` to apply it, because doing so
would mean blocking the HTTP thread on the main thread's schedule, which
reintroduces exactly the cross-thread coupling this design avoids. An
operator (or a real console) confirms the effect by polling `GET /control`
afterwards.
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlsplit


@dataclass
class _Command:
    op: str  # "claim" | "release" | "abort"
    handoff_id: str
    payload: dict[str, Any] = field(default_factory=dict)


def _make_handler_class(controller: Any, cmd_queue: "queue.Queue[_Command]") -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "HandsOperator/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - stdlib override signature
            pass  # keep test/CI output quiet; nothing here is a debugging aid

        # -- helpers ----------------------------------------------------

        def _send_json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}
            return data if isinstance(data, dict) else {}

        # -- GET: read-only attribute snapshots, never a Playwright call --

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = urlsplit(self.path).path
            if path == "/control":
                self._send_json(200, {
                    "state": controller.state.value,
                    "holder": controller.holder.value,
                    "epoch": controller.epoch,
                    "handoff_id": controller.handoff_id,
                })
                return
            if path == "/interventions":
                self._send_json(200, [r.model_dump(mode="json") for r in controller.list_open_interventions()])
                return
            self._send_json(404, {"error": f"no such resource: {path}"})

        # -- POST: enqueue only ------------------------------------------

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            parts = [p for p in urlsplit(self.path).path.split("/") if p]
            if len(parts) != 3 or parts[0] != "interventions" or parts[2] not in ("claim", "resume", "abort"):
                self._send_json(404, {"error": f"no such resource: {self.path}"})
                return
            handoff_id, verb = parts[1], parts[2]
            op = {"claim": "claim", "resume": "release", "abort": "abort"}[verb]
            body = self._read_json_body()
            cmd_queue.put(_Command(op=op, handoff_id=handoff_id, payload=body))
            self._send_json(202, {"queued": True, "op": op, "handoff_id": handoff_id})

    return Handler


class OperatorServer:
    """`OperatorServer(controller, host, port).start()` binds and serves in
    a background daemon thread; `stop()` shuts it down. `drain(controller)`
    must be called from the thread that owns `controller`'s Playwright
    objects (see the module docstring) -- typically from inside
    `SessionController.wait_for_human`'s `drain` callback.
    """

    def __init__(self, controller: Any, host: str = "127.0.0.1", port: int = 0) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        self._queue: "queue.Queue[_Command]" = queue.Queue()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> int:
        handler_cls = _make_handler_class(self.controller, self._queue)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="hands-operator-http", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None

    def drain(self, controller: Any) -> int:
        """Apply every queued operator command to `controller`, on
        whatever thread calls this. MUST be the thread that owns
        `controller`'s Playwright objects -- see the module docstring.
        Returns the number of commands DEQUEUED (not the number that
        necessarily succeeded -- a command still counts toward the return
        value even if its controller call raised). One bad command (a
        stale/mismatched handoff id, a Playwright error inside `claim()`,
        etc.) must not wedge the drain loop and stop every command behind
        it from being applied, so a failure here is caught -- but it is
        always explicitly logged via `controller.run_log`, so a real bug
        (e.g. a cross-thread Playwright violation) surfaces in
        `events.jsonl` instead of vanishing silently.
        """
        applied = 0
        while True:
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                break
            applied += 1
            try:
                if cmd.op == "claim":
                    controller.claim(cmd.handoff_id, cmd.payload.get("operator_id", "operator"))
                elif cmd.op == "release":
                    controller.release(
                        cmd.handoff_id,
                        cmd.payload.get("resume_mode", "auto"),
                        notes=cmd.payload.get("notes"),
                        record_as=cmd.payload.get("record_as", "omit"),
                    )
                elif cmd.op == "abort":
                    controller.abandon(cmd.payload.get("reason", "operator_abort"))
                else:
                    raise ValueError(f"unknown operator command op {cmd.op!r}")
            except Exception as e:  # noqa: BLE001 - one bad command must not wedge the drain loop
                controller.run_log.event(
                    "control_transfer", note="operator command failed to apply", op=cmd.op,
                    handoff_id=cmd.handoff_id, error=f"{e.__class__.__name__}: {e}",
                )
        return applied
