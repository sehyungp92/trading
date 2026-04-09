"""Run full diagnostics on P14 final config + fr_mfe_grace_r=0.15.

Usage::
    python -m backtests.stock.auto.alcb_p14_phase.run_p14_final_diagnostics
"""
from __future__ import annotations

import io
import sys
import time as _time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from backtests.stock._aliases import install

install()

from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
from backtests.stock.auto.config_mutator import mutate_alcb_config
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.research_replay import ResearchReplayEngine

from .time_utils import hydrate_time_mutations

# P14 final + fr_mfe_grace_r tuning (21 cumulative mutations)
P14_FINAL_PLUS: dict = {
    # P13 base (14 mutations)
    "ablation.use_or_width_min": True,
    "ablation.use_partial_takes": False,
    "param_overrides.carry_min_cpr": 0.6,
    "param_overrides.carry_min_r": 0.5,
    "param_overrides.flow_reversal_min_hold_bars": 12,
    "param_overrides.or_width_min_pct": 0.0015,
    "param_overrides.sector_mult_industrials": 0.5,
    "param_overrides.rvol_threshold": 2.0,
    "ablation.use_combined_quality_gate": True,
    "param_overrides.combined_breakout_score_min": 5,
    "param_overrides.combined_breakout_min_rvol": 2.5,
    "param_overrides.fr_cpr_threshold": 0.3,
    "param_overrides.sector_mult_consumer_disc": 1.2,
    "param_overrides.sector_mult_financials": 0.5,
    # P14 accepted (6 mutations)
    "ablation.use_adaptive_trail": True,
    "param_overrides.adaptive_trail_start_bars": 25,
    "param_overrides.adaptive_trail_tighten_bars": 25,
    "param_overrides.adaptive_trail_late_activate_r": 0.25,
    "param_overrides.adaptive_trail_late_distance_r": 0.20,
    "param_overrides.combined_avwap_cap_pct": 0.003,
    # Alpha recovery accepted (1 mutation)
    "param_overrides.fr_mfe_grace_r": 0.15,
}


def main() -> None:
    data_dir = Path("backtests/stock/data/raw")
    output_dir = Path("backtests/stock/auto/alcb_p14_phase/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = ResearchReplayEngine(data_dir=data_dir)
    replay.load_all_data()

    base_cfg = ALCBBacktestConfig(
        start_date="2024-01-01",
        end_date="2026-03-01",
        initial_equity=10_000.0,
        tier=2,
        data_dir=data_dir,
    )
    muts = hydrate_time_mutations(dict(P14_FINAL_PLUS))
    cfg = mutate_alcb_config(base_cfg, muts)

    print("Running P14 final + fr_grace_015 diagnostics...")
    t0 = _time.time()

    engine = ALCBIntradayEngine(cfg, replay)
    shadow = ALCBShadowTracker()
    engine.shadow_tracker = shadow
    result = engine.run()

    elapsed = _time.time() - t0
    print(f"Engine run complete in {elapsed:.0f}s -- {len(result.trades)} trades")

    report = alcb_full_diagnostic(
        result.trades,
        shadow_tracker=shadow,
        daily_selections=result.daily_selections,
    )

    out_path = output_dir / "p14_final_fr015_diagnostics.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"Diagnostics saved to {out_path}")
    print()
    print(report[:3000])


if __name__ == "__main__":
    main()
