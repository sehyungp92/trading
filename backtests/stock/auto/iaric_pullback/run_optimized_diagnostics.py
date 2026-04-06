"""Run full diagnostics on the R4 hybrid optimized config from phase_state.json.

Usage::

    python -m research.backtests.stock.auto.iaric_pullback.run_optimized_diagnostics
    python -m research.backtests.stock.auto.iaric_pullback.run_optimized_diagnostics --phase-state path/to/phase_state.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


DEFAULT_PHASE_STATE = Path("research/backtests/stock/auto/iaric_pullback/output_multiphase/phase_state.json")
DEFAULT_OUTPUT = Path("research/backtests/stock/auto/iaric_pullback/output_multiphase/r4_hybrid_full_diagnostics.txt")
DATA_DIR = Path("research/backtests/stock/data/raw")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-state", default=str(DEFAULT_PHASE_STATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--equity", type=float, default=INITIAL_EQUITY)
    args = parser.parse_args()

    from research.backtests.stock._aliases import install
    install()

    from research.backtests.stock.analysis.iaric_pullback_diagnostics import pullback_full_diagnostic
    from research.backtests.stock.auto.config_mutator import mutate_iaric_config
    from research.backtests.stock.config_iaric import IARICBacktestConfig
    from research.backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine

    # Load optimized mutations
    phase_state = json.loads(Path(args.phase_state).read_text(encoding="utf-8"))
    mutations = phase_state["cumulative_mutations"]
    print(f"Loaded {len(mutations)} cumulative mutations from {args.phase_state}")
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
    print(diag)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(diag, encoding="utf-8")
    print(f"\nFull diagnostics saved to {out_path}")


if __name__ == "__main__":
    main()
