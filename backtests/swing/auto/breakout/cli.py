"""Breakout phased auto-optimization CLI."""
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

from .plugin import BreakoutPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUND_MANAGER = RoundManager("swing", "breakout")


def _parse_symbols(value: str) -> list[str]:
    return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]


def _sanitize_breakout_baseline_mutations(mutations: dict[str, object]) -> dict[str, object]:
    cleaned = dict(mutations)
    try:
        from strategies.swing.breakout.config import REGIME_CHOP_BLOCK
    except Exception:
        REGIME_CHOP_BLOCK = None
    if cleaned.get("flags.disable_regime_chop_block") and REGIME_CHOP_BLOCK is False:
        cleaned.pop("flags.disable_regime_chop_block", None)
    legacy_fresh_stop = bool(cleaned.pop("param_overrides.ENTRY_C_FRESH_USE_STOP_LIMIT", False))
    if legacy_fresh_stop and "param_overrides.ENTRY_C_FRESH_STOP_ENABLE" not in cleaned:
        cleaned["param_overrides.ENTRY_C_FRESH_STOP_ENABLE"] = True
    return cleaned


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    plugin = BreakoutPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        symbols=_parse_symbols(args.symbols),
    )
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if round_num > 1:
        plugin.initial_mutations = _sanitize_breakout_baseline_mutations(
            ROUND_MANAGER.get_previous_mutations(round_num)
        )
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
    state = runner.load_state()
    preflight = getattr(runner.plugin, "run_preflight_ablations", None)
    if callable(preflight) and not state.completed_phases:
        preflight(runner.output_dir, dict(state.cumulative_mutations))
    state = runner.run_all_phases()
    print("Breakout auto-optimization complete.")
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="breakout-auto", description="Breakout phased auto-optimization")
    sub = parser.add_subparsers(dest="command")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", default="backtests/swing/data/raw")
        command.add_argument("--equity", type=float, default=10_000.0)
        command.add_argument("--symbols", default="QQQ,GLD")
        command.add_argument("--round", type=int, default=None)

    phase_run = sub.add_parser("phase-run", help="Run a single Breakout phase")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])
    phase_run.add_argument("--max-rounds", type=int, default=20)
    phase_run.add_argument("--max-workers", type=int, default=None)
    phase_run.add_argument("--min-delta", type=float, default=0.001)

    phase_auto = sub.add_parser("phase-auto", help="Run all Breakout phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=20)
    phase_auto.add_argument("--max-workers", type=int, default=None)
    phase_auto.add_argument("--min-delta", type=float, default=0.001)
    phase_auto.add_argument("--max-retries", type=int, default=2)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed Breakout phase gate")
    add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])

    phase_diag = sub.add_parser("phase-diagnostics", help="Print Breakout phase diagnostics")
    phase_diag.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])
    phase_diag.add_argument("--round", type=int, default=None)

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
