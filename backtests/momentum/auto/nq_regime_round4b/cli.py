"""NQ_REGIME round_4b phased auto-optimization CLI."""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.momentum.auto.nq_regime_round4b.plugin import NqRegimeRound4bPlugin
from backtests.shared.auto.phase_gates import evaluate_gate
from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
from backtests.shared.auto.phase_state import _utc_now_iso, save_phase_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PHASE_CHOICES = list(range(1, NqRegimeRound4bPlugin.num_phases + 1))


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    del for_write
    plugin = NqRegimeRound4bPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        analysis_symbol=args.analysis_symbol,
        trade_symbol=args.trade_symbol,
        num_phases=NqRegimeRound4bPlugin.num_phases,
    )
    baseline_config = getattr(args, "baseline_config", None)
    if baseline_config:
        plugin.initial_mutations = _load_mutations(Path(baseline_config))
    return PhaseRunner(
        plugin=plugin,
        output_dir=Path(args.output_dir),
        round_name=getattr(args, "round_name", "") or Path(args.output_dir).name,
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.003),
        max_retries=getattr(args, "max_retries", 0),
    )


def cmd_phase_run(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_phase(args.phase)
    _write_round_artifacts(args, runner, state)
    result = state.phase_results.get(args.phase, {})
    print(f"Phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    _write_round_artifacts(args, runner, state)
    print("NQ_REGIME round_4b auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False)
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
        "criteria": [criterion.__dict__ for criterion in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": list(gate.recommendations),
    })
    save_phase_state(state, runner.state_path)
    print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for criterion in gate.criteria:
        marker = "[PASS]" if criterion.passed else "[FAIL]"
        print(f"  {marker} {criterion.name}: {criterion.actual:.4f} (target {criterion.target:.4f})")


def _load_mutations(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("mutations"), dict):
            return dict(data["mutations"])
        if isinstance(data.get("cumulative_mutations"), dict):
            return dict(data["cumulative_mutations"])
        return dict(data)
    raise TypeError(f"Unexpected optimized config payload in {path}")


def _write_round_artifacts(args: argparse.Namespace, runner: PhaseRunner, state) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_mutations(Path(args.baseline_config)) if getattr(args, "baseline_config", None) else {}
    final_metrics = dict(runner.plugin.compute_final_metrics(state.cumulative_mutations))
    round_name = getattr(args, "round_name", "") or output_dir.name
    spec = {
        "family": "momentum",
        "strategy": "nq_regime",
        "strategy_name": "nq_regime",
        "round": round_name,
        "description": "round_4b isolated nq_2 structural_expansion reproducible-alpha optimization",
        "generated_at_utc": _utc_now_iso(),
        "baseline_source": str(Path(args.baseline_config).resolve()) if getattr(args, "baseline_config", None) else None,
        "baseline_mutation_count": len(baseline),
        "baseline_mutations": baseline,
        "scoring_weights": dict(runner.plugin.get_phase_spec(1, state).scoring_weights or {}),
    }
    summary = {
        "family": "momentum",
        "strategy": "nq_regime",
        "round": round_name,
        "generated_at_utc": _utc_now_iso(),
        "completed_phases": list(state.completed_phases),
        "mutation_count": len(state.cumulative_mutations),
        "cumulative_mutations": dict(state.cumulative_mutations),
        "final_metrics": final_metrics,
    }
    (output_dir / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=False), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8")
    (output_dir / "optimized_config.json").write_text(json.dumps(dict(state.cumulative_mutations), indent=2, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nq-regime-round4b-auto", description="NQ_REGIME round_4b phased auto-optimization")
    sub = parser.add_subparsers(dest="command")

    def add_common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--data-dir", default="backtests/momentum/data/raw")
        cmd.add_argument("--equity", type=float, default=10_000.0)
        cmd.add_argument("--analysis-symbol", default="NQ")
        cmd.add_argument("--trade-symbol", default="MNQ")
        cmd.add_argument("--output-dir", required=True)
        cmd.add_argument("--baseline-config", required=True)
        cmd.add_argument("--round-name", default="round_4b")

    phase_run = sub.add_parser("phase-run", help="Run one NQ_REGIME round_4b phase")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=PHASE_CHOICES)
    phase_run.add_argument("--max-rounds", type=int, default=8)
    phase_run.add_argument("--max-workers", type=int, default=None)
    phase_run.add_argument("--min-delta", type=float, default=0.003)
    phase_run.add_argument("--max-retries", type=int, default=0)

    phase_auto = sub.add_parser("phase-auto", help="Run all NQ_REGIME round_4b phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=8)
    phase_auto.add_argument("--max-workers", type=int, default=None)
    phase_auto.add_argument("--min-delta", type=float, default=0.003)
    phase_auto.add_argument("--max-retries", type=int, default=0)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed phase gate")
    add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=PHASE_CHOICES)

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
