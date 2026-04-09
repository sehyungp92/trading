"""Standalone CLI for ALCB P13 phased auto-optimization.

Usage::

    python -m backtests.stock.auto.alcb_p13_phase --max-workers 2 -v
    python -m backtests.stock.auto.alcb_p13_phase --phase 1
    python -m backtests.stock.auto.alcb_p13_phase --experiments p13_rvol_soft_200 p13_no_combined
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_runner(args: argparse.Namespace):
    from backtests.shared.auto.phase_runner import PhaseRunner

    from .plugin import ALCBP13Plugin

    experiment_names = set(args.experiments) if getattr(args, "experiments", None) else None
    plugin = ALCBP13Plugin(
        data_dir=Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        experiment_names=experiment_names,
    )
    return PhaseRunner(
        plugin=plugin,
        output_dir=Path(args.output_dir),
        round_name="alcb_p13",
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alcb-p13-phase",
        description="ALCB P13 phased auto-optimization (5 phases, 62 candidates)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--data-dir", default="backtests/stock/data/raw")
    parser.add_argument("--output-dir", default="backtests/stock/auto/alcb_p13_phase/output")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help="Run a single phase instead of all phases")
    parser.add_argument("--experiments", nargs="*", help="Filter to specific experiment names")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    runner = _build_runner(args)

    if args.phase is not None:
        state = runner.load_state()
        missing = [p for p in range(1, args.phase) if p not in state.completed_phases]
        if missing:
            print(
                f"Cannot run phase {args.phase} yet. Missing earlier phases: {missing}. "
                "Run without --phase or complete prior phases first.",
                file=sys.stderr,
            )
            sys.exit(1)
        state = runner.run_phase(args.phase, state)
        result = state.phase_results.get(args.phase, {})
        print(f"ALCB P13 phase {args.phase} complete.")
        print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
        print(f"Accepted: {len(result.get('kept_features', []))}")
    else:
        state = runner.run_all_phases()
        print("ALCB P13 phased auto-optimization complete.")
        print(f"Completed phases: {state.completed_phases}")
        print(f"Final mutations: {len(state.cumulative_mutations)}")


if __name__ == "__main__":
    main()
