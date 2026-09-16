"""Command-line entry point. Subcommands are filled in as the build progresses."""
from __future__ import annotations

import argparse
import sys


def _cmd_serve_mock(args: argparse.Namespace) -> int:
    from mockbank.app import run

    print(f"LegacyCore mock listening on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    run(host=args.host, port=args.port)
    return 0


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
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)
