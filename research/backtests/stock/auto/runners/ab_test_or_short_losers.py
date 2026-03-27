"""A/B tests targeting OR_BREAKOUT short losers (0-2 bars, -42.6R drag).

Tests against P9 optimal baseline (score 0.9011, 1159 trades).
Each experiment modifies one aspect to reduce short-hold OR losses.

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    PYTHONUNBUFFERED=1 python -u -m research.backtests.stock.auto.output.ab_test_or_short_losers
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from research.backtests.stock.auto.greedy_optimize import (
    ALCB_T2_P9_BASE_MUTATIONS,
    _score_config,
    _run_engine,
    _make_config,
)
from research.backtests.stock.engine.research_replay import ResearchReplayEngine

DATA_DIR = Path("research/backtests/stock/data/raw")

# P9 optimal = P9 base + 3 greedy-selected features
P9_OPTIMAL: dict = {
    **ALCB_T2_P9_BASE_MUTATIONS,
    # Greedy round 1: MFE trailing fix
    "param_overrides.fr_trailing_distance_r": 0.4,
    # Greedy round 2: disable quick exit
    "ablation.use_time_based_quick_exit": False,
    # Greedy round 3: OR quality gate
    "ablation.use_or_quality_gate": True,
    "param_overrides.or_breakout_score_min": 5,
}

# A/B experiments: each is (name, description, mutations_to_add_or_override)
EXPERIMENTS: list[tuple[str, str, dict]] = [
    # --- OR RVOL filters ---
    ("or_rvol_min_1.5", "OR requires RVOL >= 1.5", {
        "param_overrides.or_breakout_min_rvol": 1.5,
    }),
    ("or_rvol_min_2.0", "OR requires RVOL >= 2.0", {
        "param_overrides.or_breakout_min_rvol": 2.0,
    }),
    ("or_rvol_min_2.5", "OR requires RVOL >= 2.5", {
        "param_overrides.or_breakout_min_rvol": 2.5,
    }),

    # --- OR score tightening ---
    ("or_score_min_6", "OR requires momentum score >= 6", {
        "param_overrides.or_breakout_score_min": 6,
    }),

    # --- Ultra-early stage 1 QE (no stage 2, targets 0-2 bar reversals) ---
    ("early_qe_2bar_n05", "Stage1 QE: bar 2, R < -0.5 (worst reversals only)", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 2,
        "param_overrides.qe_stage1_min_r": -0.5,
    }),
    ("early_qe_2bar_n03", "Stage1 QE: bar 2, R < -0.3", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 2,
        "param_overrides.qe_stage1_min_r": -0.3,
    }),
    ("early_qe_3bar_n03", "Stage1 QE: bar 3, R < -0.3", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 3,
        "param_overrides.qe_stage1_min_r": -0.3,
    }),
    ("early_qe_3bar_n05", "Stage1 QE: bar 3, R < -0.5", {
        "ablation.use_quick_exit_stage1": True,
        "param_overrides.qe_stage1_bars": 3,
        "param_overrides.qe_stage1_min_r": -0.5,
    }),
]


def _short_hold_stats(trades, max_bars: int = 6) -> dict:
    """Compute stats for short-hold trades (<=max_bars)."""
    short = [t for t in trades if t.hold_bars <= max_bars]
    or_short = [t for t in short if t.entry_type == "OR_BREAKOUT"]
    if not short:
        return {"n": 0, "or_n": 0}
    short_r = [t.r_multiple for t in short]
    or_r = [t.r_multiple for t in or_short]
    return {
        "n": len(short),
        "wr": sum(1 for r in short_r if r > 0) / len(short),
        "total_r": sum(short_r),
        "mean_r": sum(short_r) / len(short),
        "or_n": len(or_short),
        "or_total_r": sum(or_r) if or_r else 0,
        "or_mean_r": sum(or_r) / len(or_r) if or_r else 0,
    }


def main():
    print("=" * 70)
    print("A/B TEST: OR_BREAKOUT Short Loser Optimization")
    print("=" * 70, flush=True)

    print(f"\nExperiments: {len(EXPERIMENTS)}")
    print("Baseline: P9 optimal (score 0.9011)\n")

    print("Loading data...", flush=True)
    t0 = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t0:.1f}s\n", flush=True)

    cfg_kwargs = dict(
        strategy="alcb", tier=2, initial_equity=10_000.0,
        start_date="2024-01-01", end_date="2026-03-01",
        data_dir=DATA_DIR,
    )

    # --- Baseline ---
    print("Running baseline (P9 optimal)...", flush=True)
    t1 = time.time()
    base_score, base_metrics = _score_config(replay, P9_OPTIMAL, 10_000.0, cfg_kwargs)

    config_b = _make_config(P9_OPTIMAL, cfg_kwargs)
    trades_b, eq_b, ts_b = _run_engine(replay, "alcb", 2, config_b)
    base_sh = _short_hold_stats(trades_b)

    print(f"  Baseline: score={base_score.total:.4f}  trades={base_metrics.total_trades}"
          f"  PF={base_metrics.profit_factor:.2f}  DD={base_metrics.max_drawdown_pct:.1%}"
          f"  return={base_metrics.net_profit / 10_000:.1%}")
    print(f"  Short hold: n={base_sh['n']}  WR={base_sh.get('wr',0):.1%}"
          f"  total_R={base_sh['total_r']:+.1f}"
          f"  OR: n={base_sh['or_n']}  total_R={base_sh['or_total_r']:+.1f}")
    print(f"  ({time.time() - t1:.1f}s)\n", flush=True)

    # --- Experiments ---
    results = []
    for name, desc, mutations in EXPERIMENTS:
        print(f"Testing {name}: {desc}...", flush=True)
        t1 = time.time()

        test_mutations = {**P9_OPTIMAL, **mutations}
        score, metrics = _score_config(replay, test_mutations, 10_000.0, cfg_kwargs)

        config_t = _make_config(test_mutations, cfg_kwargs)
        trades_t, eq_t, ts_t = _run_engine(replay, "alcb", 2, config_t)
        sh = _short_hold_stats(trades_t)

        delta = (score.total - base_score.total) / base_score.total * 100
        elapsed = time.time() - t1

        results.append((name, desc, score.total, delta, metrics, sh, elapsed))
        print(f"  score={score.total:.4f}  delta={delta:+.2f}%  trades={metrics.total_trades}"
              f"  PF={metrics.profit_factor:.2f}  DD={metrics.max_drawdown_pct:.1%}"
              f"  short_R={sh['total_r']:+.1f}  OR_short_R={sh['or_total_r']:+.1f}"
              f"  ({elapsed:.1f}s)", flush=True)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"{'Experiment':<25} {'Score':>8} {'Delta':>8} {'Trades':>7} {'PF':>6}"
          f" {'DD':>6} {'ShortR':>8} {'OR_ShR':>8}")
    print("-" * 90)
    print(f"{'P9 BASELINE':<25} {base_score.total:>8.4f} {'—':>8} {base_metrics.total_trades:>7}"
          f" {base_metrics.profit_factor:>6.2f} {base_metrics.max_drawdown_pct:>6.1%}"
          f" {base_sh['total_r']:>+8.1f} {base_sh['or_total_r']:>+8.1f}")
    for name, desc, score, delta, metrics, sh, _ in sorted(results, key=lambda x: -x[2]):
        marker = " ***" if delta > 0 else ""
        print(f"{name:<25} {score:>8.4f} {delta:>+7.2f}% {metrics.total_trades:>7}"
              f" {metrics.profit_factor:>6.2f} {metrics.max_drawdown_pct:>6.1%}"
              f" {sh['total_r']:>+8.1f} {sh['or_total_r']:>+8.1f}{marker}")

    print(f"\nTotal time: {time.time() - t0:.0f}s ({(time.time() - t0) / 60:.1f}min)")


if __name__ == "__main__":
    main()
