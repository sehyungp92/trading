"""Quick A/B test of high-value alpha recovery approaches on P14 final config.

Tests ~12 experiments targeting:
1. Compound MFE conviction (MFE + current R floor)
2. FR tuning (grace R, CPR threshold, min hold bars)
3. Adaptive trail variants (mid+late vs late-only)
4. Best combinations

Usage::
    python -m backtests.stock.auto.alcb_p14_phase.test_alpha_recovery
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

import numpy as np

from backtests.stock.auto.config_mutator import mutate_alcb_config
from backtests.stock.auto.scoring import extract_metrics
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.research_replay import ResearchReplayEngine

from .time_utils import hydrate_time_mutations

# P14 final cumulative mutations (from round_evaluation.txt)
P14_FINAL: dict = {
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
    # P14 accepted mutations
    "ablation.use_adaptive_trail": True,
    "param_overrides.adaptive_trail_start_bars": 25,
    "param_overrides.adaptive_trail_tighten_bars": 25,
    "param_overrides.adaptive_trail_late_activate_r": 0.25,
    "param_overrides.adaptive_trail_late_distance_r": 0.20,
    "param_overrides.combined_avwap_cap_pct": 0.003,
}

# Experiments: (name, extra_mutations)
EXPERIMENTS: list[tuple[str, dict]] = [
    # --- FR Tuning ---
    ("fr_grace_020", {"param_overrides.fr_mfe_grace_r": 0.20}),
    ("fr_grace_015", {"param_overrides.fr_mfe_grace_r": 0.15}),
    ("fr_grace_010", {"param_overrides.fr_mfe_grace_r": 0.10}),
    ("fr_cpr_040", {"param_overrides.fr_cpr_threshold": 0.4}),
    ("fr_cpr_050", {"param_overrides.fr_cpr_threshold": 0.5}),
    ("fr_hold_16", {"param_overrides.flow_reversal_min_hold_bars": 16}),
    ("fr_hold_20", {"param_overrides.flow_reversal_min_hold_bars": 20}),

    # --- Compound MFE Conviction ---
    ("cmpd_14bar_015mfe_n015r", {
        "ablation.use_mfe_conviction_exit": True,
        "param_overrides.mfe_conviction_check_bars": 14,
        "param_overrides.mfe_conviction_min_r": 0.15,
        "param_overrides.mfe_conviction_floor_r": -0.15,
    }),
    ("cmpd_14bar_020mfe_n020r", {
        "ablation.use_mfe_conviction_exit": True,
        "param_overrides.mfe_conviction_check_bars": 14,
        "param_overrides.mfe_conviction_min_r": 0.20,
        "param_overrides.mfe_conviction_floor_r": -0.20,
    }),
    ("cmpd_12bar_015mfe_n020r", {
        "ablation.use_mfe_conviction_exit": True,
        "param_overrides.mfe_conviction_check_bars": 12,
        "param_overrides.mfe_conviction_min_r": 0.15,
        "param_overrides.mfe_conviction_floor_r": -0.20,
    }),

    # --- Adaptive Trail: Mid+Late vs Late-only ---
    ("trail_mid_late_13_25", {
        "param_overrides.adaptive_trail_start_bars": 13,
        "param_overrides.adaptive_trail_mid_activate_r": 0.20,
        "param_overrides.adaptive_trail_mid_distance_r": 0.40,
    }),
    ("trail_mid_late_15_25", {
        "param_overrides.adaptive_trail_start_bars": 15,
        "param_overrides.adaptive_trail_mid_activate_r": 0.25,
        "param_overrides.adaptive_trail_mid_distance_r": 0.35,
    }),

    # --- Best FR + Compound combo ---
    ("combo_fr015_cmpd14", {
        "param_overrides.fr_mfe_grace_r": 0.15,
        "ablation.use_mfe_conviction_exit": True,
        "param_overrides.mfe_conviction_check_bars": 14,
        "param_overrides.mfe_conviction_min_r": 0.15,
        "param_overrides.mfe_conviction_floor_r": -0.15,
    }),
    ("combo_fr020_hold16", {
        "param_overrides.fr_mfe_grace_r": 0.20,
        "param_overrides.flow_reversal_min_hold_bars": 16,
    }),
]


def run_experiment(
    replay: ResearchReplayEngine,
    base_cfg: ALCBBacktestConfig,
    base_muts: dict,
    extra_muts: dict,
) -> dict:
    muts = dict(base_muts)
    muts.update(extra_muts)
    cfg = mutate_alcb_config(base_cfg, muts)
    result = ALCBIntradayEngine(cfg, replay).run()
    perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, 10_000.0)

    # Exit breakdown
    exits: dict[str, list[float]] = {}
    for t in result.trades:
        er = getattr(t, "exit_reason", "?")
        exits.setdefault(er, []).append(float(t.r_multiple))

    return {
        "trades": int(perf.total_trades),
        "wr": float(perf.win_rate),
        "total_r": float(perf.expectancy * perf.total_trades),
        "pf": float(perf.profit_factor),
        "dd": float(perf.max_drawdown_pct),
        "exp_dollar": float(perf.expectancy_dollar),
        "tpm": float(perf.trades_per_month),
        "exits": {k: (len(v), sum(v)) for k, v in exits.items()},
    }


def main() -> None:
    data_dir = Path("backtests/stock/data/raw")
    replay = ResearchReplayEngine(data_dir=data_dir)
    replay.load_all_data()

    base_cfg = ALCBBacktestConfig(
        start_date="2024-01-01",
        end_date="2026-03-01",
        initial_equity=10_000.0,
        tier=2,
        data_dir=data_dir,
    )
    base_muts = hydrate_time_mutations(dict(P14_FINAL))

    # Run baseline
    print("Running baseline...")
    t0 = _time.time()
    baseline = run_experiment(replay, base_cfg, base_muts, {})
    print(f"Baseline ({_time.time() - t0:.0f}s): {baseline['trades']} trades, "
          f"R={baseline['total_r']:+.1f}, PF={baseline['pf']:.2f}, "
          f"DD={baseline['dd']:.2%}, $/t={baseline['exp_dollar']:.2f}")
    for er, (n, r) in sorted(baseline["exits"].items(), key=lambda x: -x[1][0]):
        print(f"  {er}: n={n}, R={r:+.1f}")
    print()

    # Run experiments
    results: list[tuple[str, dict]] = []
    for name, extra in EXPERIMENTS:
        t0 = _time.time()
        res = run_experiment(replay, base_cfg, base_muts, extra)
        elapsed = _time.time() - t0
        delta_r = res["total_r"] - baseline["total_r"]
        delta_pf = res["pf"] - baseline["pf"]
        results.append((name, res))
        flag = ">>>" if delta_r > 1.0 else "   "
        print(f"{flag} {name} ({elapsed:.0f}s): {res['trades']}t, "
              f"R={res['total_r']:+.1f} ({delta_r:+.1f}), "
              f"PF={res['pf']:.2f} ({delta_pf:+.2f}), "
              f"DD={res['dd']:.2%}, $/t={res['exp_dollar']:.2f}, "
              f"tpm={res['tpm']:.1f}")

    # Summary sorted by total R
    print("\n" + "=" * 80)
    print("RANKED BY TOTAL R (vs baseline)")
    print("=" * 80)
    print(f"{'Name':<35} {'Trades':>6} {'TotalR':>8} {'DeltaR':>8} {'PF':>6} {'DD':>7} {'$/t':>7} {'TPM':>6}")
    print("-" * 80)
    print(f"{'BASELINE':<35} {baseline['trades']:>6} {baseline['total_r']:>+8.1f} {'--':>8} {baseline['pf']:>6.2f} {baseline['dd']:>6.2%} {baseline['exp_dollar']:>7.2f} {baseline['tpm']:>6.1f}")
    for name, res in sorted(results, key=lambda x: -x[1]["total_r"]):
        delta = res["total_r"] - baseline["total_r"]
        print(f"{name:<35} {res['trades']:>6} {res['total_r']:>+8.1f} {delta:>+8.1f} {res['pf']:>6.2f} {res['dd']:>6.2%} {res['exp_dollar']:>7.2f} {res['tpm']:>6.1f}")


if __name__ == "__main__":
    main()
