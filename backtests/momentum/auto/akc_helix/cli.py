"""AKC Helix phased auto-optimization CLI."""
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
from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
from backtests.shared.auto.phase_state import save_phase_state
from backtests.shared.auto.round_manager import RoundManager

from .phase_candidates import PHASE_FOCUS
from .plugin import AKCHelixPlugin, AKCHelixRebasePlugin
from .rebase_candidates import REBASE_PHASE_FOCUS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUND_MANAGER = RoundManager("momentum", "helix")
REBASE_ROUND_MANAGER = RoundManager("momentum", "helix")


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    plugin = AKCHelixPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
    )
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if round_num > 1:
        plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(round_num)
    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 2),
        round_manager=ROUND_MANAGER,
        round_num=round_num,
    )


def cmd_phase_run(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_phase(args.phase)
    result = state.phase_results.get(args.phase, {})
    print(f"Phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print("AKC Helix auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False)
    state = runner.load_state()
    if args.phase not in state.phase_results:
        print(f"Phase {args.phase} has not been completed yet.")
        return

    phase_mutations = _mutations_through_phase(state, args.phase)
    metrics = runner.plugin.compute_final_metrics(phase_mutations)
    spec = runner.plugin.get_phase_spec(args.phase, state)
    gate = evaluate_gate(spec.gate_criteria_fn(metrics))
    state.record_gate(args.phase, {
        "passed": gate.passed,
        "criteria": [criterion.__dict__ for criterion in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": list(gate.recommendations),
    })
    save_phase_state(state, runner.state_path)

    print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for criterion in gate.criteria:
        marker = "[PASS]" if criterion.passed else "[FAIL]"
        print(f"  {marker} {criterion.name}: {criterion.actual:.4f} (target {criterion.target:.4f})")


def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    _, round_dir = ROUND_MANAGER.resolve_round(getattr(args, "round", None), for_write=False)
    diag_path = round_dir / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.")
        return
    print(diag_path.read_text(encoding="utf-8"))


def _build_rebase_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    plugin = AKCHelixRebasePlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
    )
    round_num, round_dir = REBASE_ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if round_num > 1:
        plugin.initial_mutations = REBASE_ROUND_MANAGER.get_previous_mutations(round_num)
    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 2),
        round_manager=REBASE_ROUND_MANAGER,
        round_num=round_num,
    )


def cmd_rebase_baseline(args: argparse.Namespace) -> None:
    plugin = AKCHelixRebasePlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
    )
    metrics = plugin.compute_final_metrics({})
    print("=" * 60)
    print("  AKC HELIX REBASE -- BASELINE (no mutations)")
    print("=" * 60)
    print(f"  Trades:      {int(metrics.get('total_trades', 0))}")
    print(f"  Win rate:    {metrics.get('win_rate', 0.0):.1%}")
    print(f"  PF:          {metrics.get('profit_factor', 0.0):.2f}")
    print(f"  Net profit:  ${metrics.get('net_profit', 0.0):,.0f}")
    print(f"  Max DD:      {metrics.get('max_drawdown_pct', 0.0):.1%}")
    print(f"  Sharpe:      {metrics.get('sharpe', 0.0):.2f}")
    print(f"  Sortino:     {metrics.get('sortino', 0.0):.2f}")
    print(f"  Calmar:      {metrics.get('calmar', 0.0):.2f}")
    print(f"  Expectancy:  ${metrics.get('expectancy_dollar', 0.0):,.0f}")
    print(f"  Avg hold:    {metrics.get('avg_hold_hours', 0.0):.1f}h")


def cmd_rebase_phase_run(args: argparse.Namespace) -> None:
    runner = _build_rebase_runner(args)
    state = runner.run_phase(args.phase)
    result = state.phase_results.get(args.phase, {})
    print(f"Rebase phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_rebase_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_rebase_runner(args)
    state = runner.run_all_phases()
    print("AKC Helix rebase auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_rebase_phase_gate(args: argparse.Namespace) -> None:
    runner = _build_rebase_runner(args, for_write=False)
    state = runner.load_state()
    if args.phase not in state.phase_results:
        print(f"Rebase phase {args.phase} has not been completed yet.")
        return

    phase_mutations = _mutations_through_phase(state, args.phase)
    metrics = runner.plugin.compute_final_metrics(phase_mutations)
    spec = runner.plugin.get_phase_spec(args.phase, state)
    gate = evaluate_gate(spec.gate_criteria_fn(metrics))
    state.record_gate(args.phase, {
        "passed": gate.passed,
        "criteria": [criterion.__dict__ for criterion in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": list(gate.recommendations),
    })
    save_phase_state(state, runner.state_path)

    print(f"Rebase phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for criterion in gate.criteria:
        marker = "[PASS]" if criterion.passed else "[FAIL]"
        print(f"  {marker} {criterion.name}: {criterion.actual:.4f} (target {criterion.target:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="akc-helix-auto", description="AKC Helix phased auto-optimization")
    sub = parser.add_subparsers(dest="command")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", default="backtests/momentum/data/raw")
        command.add_argument("--equity", type=float, default=10_000.0)
        command.add_argument("--round", type=int, default=None)

    # --- Standard pipeline commands ---
    phase_run = sub.add_parser("phase-run", help="Run a single AKC Helix phase")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_FOCUS))
    phase_run.add_argument("--max-rounds", type=int, default=20)
    phase_run.add_argument("--max-workers", type=int, default=None)
    phase_run.add_argument("--min-delta", type=float, default=0.001)

    phase_auto = sub.add_parser("phase-auto", help="Run all AKC Helix phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=20)
    phase_auto.add_argument("--max-workers", type=int, default=None)
    phase_auto.add_argument("--min-delta", type=float, default=0.001)
    phase_auto.add_argument("--max-retries", type=int, default=2)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed AKC Helix phase gate")
    add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_FOCUS))

    phase_diag = sub.add_parser("phase-diagnostics", help="Print AKC Helix phase diagnostics")
    phase_diag.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_FOCUS))
    phase_diag.add_argument("--round", type=int, default=None)

    # --- Rebase pipeline commands ---
    rebase_baseline = sub.add_parser("rebase-baseline", help="Evaluate vanilla config on corrected data")
    rebase_baseline.add_argument("--data-dir", default="backtests/momentum/data/raw")
    rebase_baseline.add_argument("--equity", type=float, default=10_000.0)

    rebase_run = sub.add_parser("rebase-phase-run", help="Run a single rebase phase")
    add_common(rebase_run)
    rebase_run.add_argument("--phase", type=int, required=True, choices=sorted(REBASE_PHASE_FOCUS))
    rebase_run.add_argument("--max-rounds", type=int, default=20)
    rebase_run.add_argument("--max-workers", type=int, default=None)
    rebase_run.add_argument("--min-delta", type=float, default=0.001)

    rebase_auto = sub.add_parser("rebase-phase-auto", help="Run all rebase phases")
    add_common(rebase_auto)
    rebase_auto.add_argument("--max-rounds", type=int, default=20)
    rebase_auto.add_argument("--max-workers", type=int, default=None)
    rebase_auto.add_argument("--min-delta", type=float, default=0.001)
    rebase_auto.add_argument("--max-retries", type=int, default=2)

    rebase_gate = sub.add_parser("rebase-phase-gate", help="Check a completed rebase phase gate")
    add_common(rebase_gate)
    rebase_gate.add_argument("--phase", type=int, required=True, choices=sorted(REBASE_PHASE_FOCUS))

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    elif args.command == "rebase-baseline":
        cmd_rebase_baseline(args)
    elif args.command == "rebase-phase-run":
        cmd_rebase_phase_run(args)
    elif args.command == "rebase-phase-auto":
        cmd_rebase_phase_auto(args)
    elif args.command == "rebase-phase-gate":
        cmd_rebase_phase_gate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
