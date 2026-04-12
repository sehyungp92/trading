"""ATRSS phased auto-optimization CLI."""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.auto.phase_gates import evaluate_gate
from backtests.shared.auto.phase_state import save_phase_state
from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
from backtests.swing.auto.atrss.plugin import ATRSSPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = _root / "backtests/swing/auto/atrss/output"


def _build_runner(args: argparse.Namespace) -> PhaseRunner:
    plugin = ATRSSPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
    )
    return PhaseRunner(
        plugin=plugin,
        output_dir=OUTPUT_DIR,
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.005),
        max_retries=getattr(args, "max_retries", 2),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_phase_run(args: argparse.Namespace) -> None:
    """Run a single phase."""
    runner = _build_runner(args)
    state = runner.run_phase(args.phase)
    result = state.phase_results.get(args.phase, {})
    print(f"Phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    """Check if a phase passes its gate."""
    runner = _build_runner(args)
    state = runner.load_state()
    if args.phase not in state.phase_results:
        print(f"Phase {args.phase} has not been completed yet.")
        return

    plugin = runner.plugin
    phase_mutations = _mutations_through_phase(state, args.phase)
    metrics = plugin.compute_final_metrics(phase_mutations)
    spec = plugin.get_phase_spec(args.phase, state)
    gate = evaluate_gate(spec.gate_criteria_fn(metrics))
    state.record_gate(args.phase, {
        "passed": gate.passed,
        "criteria": [c.__dict__ for c in gate.criteria],
        "failure_category": gate.failure_category,
    })
    save_phase_state(state, OUTPUT_DIR)
    print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for c in gate.criteria:
        status = "OK" if c.passed else "FAIL"
        print(f"  [{status}] {c.name}: {c.actual:.4f} (target: {c.target:.4f})")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    """Run all phases automatically."""
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print("ATRSS auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    final_muts = state.cumulative_mutations
    print(f"Cumulative mutations: {len(final_muts)}")
    for k, v in sorted(final_muts.items()):
        print(f"  {k}: {v}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current optimization status."""
    runner = _build_runner(args)
    state = runner.load_state()
    print(f"Completed phases: {state.completed_phases}")
    print(f"Cumulative mutations ({len(state.cumulative_mutations)}):")
    for k, v in sorted(state.cumulative_mutations.items()):
        print(f"  {k}: {v}")
    for phase in sorted(state.phase_results.keys()):
        res = state.phase_results[phase]
        n_kept = len(res.get("kept_features", []))
        score = res.get("final_score", 0.0)
        print(f"  Phase {phase}: {n_kept} accepted, score={score:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ATRSS R1 phased auto-optimization")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw",
                        help="Path to OHLCV data directory")
    parser.add_argument("--equity", type=float, default=10_000.0,
                        help="Initial equity (default: $10,000)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: CPU count - 1)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Max greedy rounds per phase")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="Min score improvement to accept (default: 0.005)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries per phase on failure")

    sub = parser.add_subparsers(dest="cmd")

    # phase run
    p_run = sub.add_parser("run", help="Run a single phase")
    p_run.add_argument("phase", type=int, choices=[1, 2, 3, 4])
    p_run.set_defaults(func=cmd_phase_run)

    # phase gate
    p_gate = sub.add_parser("gate", help="Check phase gate")
    p_gate.add_argument("phase", type=int, choices=[1, 2, 3, 4])
    p_gate.set_defaults(func=cmd_phase_gate)

    # auto (all phases)
    p_auto = sub.add_parser("auto", help="Run all phases automatically")
    p_auto.set_defaults(func=cmd_phase_auto)

    # status
    p_status = sub.add_parser("status", help="Show optimization status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
