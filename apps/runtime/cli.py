"""CLI entrypoint for the unified runtime scaffold."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from libs.config.loader import load_strategy_registry
from libs.config.registry import write_registry_artifact

from .runtime import RuntimeShell


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate runtime config and registry")
    preflight.add_argument("--config-dir", default="config")
    preflight.add_argument("--json", action="store_true", dest="as_json")
    preflight.add_argument(
        "--write-registry-artifact",
        nargs="?",
        const="data/strategy-registry.json",
        default=None,
    )

    run = subparsers.add_parser("run", help="Start the unified runtime")
    run.add_argument("--config-dir", default="config")
    run.add_argument("--shadow", action="store_true")
    run.add_argument("--connect-ib", action="store_true")
    run.add_argument("--once", action="store_true")
    run.add_argument(
        "--family",
        default=None,
        choices=["swing", "momentum", "stock"],
        help="Run only strategies for the specified family",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    log = logging.getLogger(__name__)

    if args.command == "preflight":
        try:
            shell = RuntimeShell(args.config_dir)
            checks = shell.run_preflight()
        except Exception as exc:
            log.error("Preflight failed: %s", exc, exc_info=True)
            return 1
        if args.write_registry_artifact:
            registry = load_strategy_registry(args.config_dir)
            path = write_registry_artifact(registry, Path(args.write_registry_artifact))
            log.info("Wrote registry artifact to %s", path)

        payload = [
            {"name": check.name, "ok": check.ok, "detail": check.detail}
            for check in checks
        ]
        if args.as_json:
            print(json.dumps(payload, indent=2))
        else:
            for check in checks:
                status = "OK" if check.ok else "FAIL"
                print(f"[{status}] {check.name}: {check.detail}")
        return 0 if all(check.ok for check in checks) else 1

    if args.command == "run":
        try:
            shell = RuntimeShell(args.config_dir)
            asyncio.run(shell.run(
                shadow=args.shadow,
                connect_ib=args.connect_ib,
                once=args.once,
                family_filter=args.family,
            ))
            return 0
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            log.error("Runtime failed: %s", exc, exc_info=True)
            return 1

    parser.error(f"Unknown command {args.command!r}")
    return 2

