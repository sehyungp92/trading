"""Retired compatibility entrypoint for the former TPC context recovery.

TPC is now an explicitly ETF-only strategy. Historical promotion artifacts
remain in ``backtests/output/swing/tpc/round_8`` for auditability, but this
command must not mutate the current round or restore an external data lane.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Retired: TPC is ETF-only. Use the current Round-8 optimized config "
        "and ordinary TPC replay/optimization commands."
    )


if __name__ == "__main__":
    main()
