"""Run full diagnostics on the R4 hybrid optimized config from phase_state.json.

Usage::

    python -m backtests.stock.auto.iaric_pullback.run_optimized_diagnostics
    python -m backtests.stock.auto.iaric_pullback.run_optimized_diagnostics --phase-state path/to/phase_state.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtests.diagnostic_snapshot import build_group_snapshot


DEFAULT_PHASE_STATE = Path("backtests/stock/auto/iaric_pullback/output_multiphase/phase_state.json")
DEFAULT_OUTPUT = Path("backtests/stock/auto/iaric_pullback/output_multiphase/r4_hybrid_full_diagnostics.txt")
DATA_DIR = Path("backtests/stock/data/raw")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0


def _reference_metrics(
    phase_state: dict,
    phase_result: str | None,
    mutation_kind: str,
) -> dict | None:
    if phase_result:
        phase_entry = phase_state.get("phase_results", {}).get(str(phase_result))
    else:
        current_phase = phase_state.get("current_phase")
        phase_entry = phase_state.get("phase_results", {}).get(str(current_phase))
    if not isinstance(phase_entry, dict):
        return None

    if mutation_kind == "base":
        return {
            "score": phase_entry.get("base_score"),
            "metrics": phase_entry.get("base_metrics", {}),
        }
    return {
        "score": phase_entry.get("final_score"),
        "metrics": phase_entry.get("final_metrics", {}),
    }


def _format_reference_lines(reference: dict | None) -> list[str]:
    if not reference:
        return []
    metrics = reference.get("metrics") or {}
    score = reference.get("score")
    parts: list[str] = []
    if score is not None:
        parts.append(f"score={float(score):.4f}")
    if metrics.get("total_trades") is not None:
        parts.append(f"trades={int(metrics['total_trades'])}")
    if metrics.get("profit_factor") is not None:
        parts.append(f"PF={float(metrics['profit_factor']):.2f}")
    if metrics.get("net_profit") is not None:
        parts.append(f"net=${float(metrics['net_profit']):,.2f}")
    return [
        "Phase-state reference (pre-fix optimization record):",
        ("  " + "  ".join(parts)) if parts else "  unavailable",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-state", default=str(DEFAULT_PHASE_STATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--phase-result", help="Optional phase id to load from phase_results instead of cumulative mutations.")
    parser.add_argument(
        "--mutation-kind",
        choices=["base", "final", "current"],
        default="current",
        help="When --phase-result is supplied, choose base_mutations, final_mutations, or cumulative_mutations.",
    )
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--equity", type=float, default=INITIAL_EQUITY)
    parser.add_argument("--title", default="IARIC PULLBACK OPTIMIZED DIAGNOSTICS")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output path.")
    args = parser.parse_args()

    from backtests.stock.analysis.iaric_pullback_diagnostics import pullback_full_diagnostic
    from backtests.stock.analysis.iaric_pullback_diagnostics import _trade_stats as _iaric_trade_stats
    from backtests.stock.auto.config_mutator import mutate_iaric_config
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    # Load optimized mutations
    phase_state = json.loads(Path(args.phase_state).read_text(encoding="utf-8"))
    if args.phase_result:
        if args.mutation_kind == "current":
            mutations = phase_state["cumulative_mutations"]
        else:
            phase_result = phase_state.get("phase_results", {}).get(str(args.phase_result))
            if phase_result is None:
                raise SystemExit(f"Phase result {args.phase_result} not found in {args.phase_state}")
            key = "base_mutations" if args.mutation_kind == "base" else "final_mutations"
            mutations = phase_result.get(key, {})
    else:
        mutations = phase_state["cumulative_mutations"]
    print(f"Loaded {len(mutations)} mutations from {args.phase_state}")
    print(f"Date range: {args.start} -- {args.end}")
    print(f"Initial equity: ${args.equity:,.0f}")
    print()

    # Build config with mutations applied
    data_dir = Path(DATA_DIR)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    base_config = IARICBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=3,
        data_dir=data_dir,
    )
    config = mutate_iaric_config(base_config, mutations)

    print("Running IARIC T3 pullback backtest (optimized R4 hybrid)...")
    engine = IARICPullbackEngine(config, replay, collect_diagnostics=True)
    result = engine.run()
    print(f"Completed: {len(result.trades)} trades\n")

    diag = pullback_full_diagnostic(
        result.trades,
        replay=replay,
        daily_selections=result.daily_selections,
        candidate_ledger=result.candidate_ledger,
        funnel_counters=result.funnel_counters,
        rejection_log=result.rejection_log,
        shadow_outcomes=result.shadow_outcomes,
        selection_attribution=result.selection_attribution,
        fsm_log=result.fsm_log,
    )
    snapshot = build_group_snapshot(
        "IARIC Strength / Weakness Snapshot",
        result.trades,
        [
            ("symbol", lambda trade: getattr(trade, "symbol", None)),
            ("entry variant", lambda trade: getattr(trade, "metadata", {}).get("entry_variant")),
            ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
        ],
        min_count=5,
    )
    reference = _reference_metrics(phase_state, args.phase_result, args.mutation_kind)
    header_lines = [
        "=" * 72,
        f"  {args.title}",
        "=" * 72,
        f"  Mutation source: {Path(args.phase_state)}",
        f"  Phase result:    {args.phase_result or 'current'}",
        f"  Mutation kind:   {args.mutation_kind}",
        f"  Mutation count:  {len(mutations)}",
        f"  Date range:      {args.start} -- {args.end}",
        f"  Initial equity:  ${args.equity:,.0f}",
    ]
    if mutations:
        header_lines.append("  Mutations:")
        for key, value in sorted(mutations.items()):
            header_lines.append(f"    {key}: {value}")
    reference_lines = _format_reference_lines(reference)
    if reference_lines:
        header_lines.extend([""] + reference_lines)

    full_report = "\n".join(header_lines) + "\n\n" + snapshot + "\n\n" + diag
    print(full_report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_report, encoding="utf-8")
    print(f"\nFull diagnostics saved to {out_path}")

    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_name(f"{out_path.stem}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trade_stats = _iaric_trade_stats(result.trades)
    summary_payload = {
        "strategy": "iaric_pullback",
        "title": args.title,
        "mutation_source": str(Path(args.phase_state)),
        "phase_result": args.phase_result or "current",
        "mutation_kind": args.mutation_kind,
        "date_range": {"start": args.start, "end": args.end},
        "initial_equity": args.equity,
        "mutation_count": len(mutations),
        "mutations": mutations,
        "live_metrics": trade_stats,
        "phase_state_reference": reference,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
