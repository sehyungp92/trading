"""Run full diagnostics on conservative P12 config (overfit mutations removed).

Conservative P12 = P11 base + use_combined_breakout=false + sector_mult_financials=0.5
Removed: healthcare=0.3, communication=0.5, industrials->0.3, tuesday=0.5, entry_window_end=11:30

Usage::

    python -m research.backtests.stock.auto.alcb_p12_phase.run_conservative_diagnostics
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_OUTPUT = Path("research/backtests/stock/auto/alcb_p12_phase/output_multiphase/p12_conservative_full_diagnostics.txt")
DATA_DIR = Path("research/backtests/stock/data/raw")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0

CONSERVATIVE_MUTATIONS = {
    # P11 base (7 mutations)
    "ablation.use_or_width_min": True,
    "ablation.use_partial_takes": False,
    "param_overrides.carry_min_cpr": 0.6,
    "param_overrides.carry_min_r": 0.5,
    "param_overrides.flow_reversal_min_hold_bars": 12,
    "param_overrides.or_width_min_pct": 0.0015,
    "param_overrides.sector_mult_industrials": 0.5,
    # P12 high-confidence additions
    "ablation.use_combined_breakout": False,
    "param_overrides.sector_mult_financials": 0.5,
}


def main() -> None:
    from research.backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
    from research.backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
    from research.backtests.stock.analysis.reports import full_report
    from research.backtests.stock.auto.config_mutator import mutate_alcb_config
    from research.backtests.stock.config_alcb import ALCBBacktestConfig
    from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine

    print(f"Conservative P12: {len(CONSERVATIVE_MUTATIONS)} mutations")
    for k, v in CONSERVATIVE_MUTATIONS.items():
        print(f"  {k} = {v}")
    print(f"\nDate range: {START_DATE} -- {END_DATE}")
    print(f"Initial equity: ${INITIAL_EQUITY:,.0f}\n")

    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    print("Loading bar data...")
    replay.load_all_data()

    base_config = ALCBBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_equity=INITIAL_EQUITY,
        tier=2,
        data_dir=DATA_DIR,
    )
    config = mutate_alcb_config(base_config, CONSERVATIVE_MUTATIONS)

    shadow_tracker = ALCBShadowTracker()
    engine = ALCBIntradayEngine(config, replay)
    engine.shadow_tracker = shadow_tracker

    print("Running ALCB T2 backtest (conservative P12)...")
    result = engine.run()
    print(f"Completed: {len(result.trades)} trades\n")

    report = full_report(
        result.trades, result.equity_curve, result.timestamps,
        config.initial_equity, strategy="ALCB Conservative P12",
        daily_selections=result.daily_selections,
    )
    print(report)

    diag = alcb_full_diagnostic(
        result.trades,
        shadow_tracker=shadow_tracker,
        daily_selections=result.daily_selections,
    )
    print(diag)

    output_path = DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{report}\n\n{diag}", encoding="utf-8")
    print(f"\nDiagnostics saved to {output_path}")


if __name__ == "__main__":
    main()
