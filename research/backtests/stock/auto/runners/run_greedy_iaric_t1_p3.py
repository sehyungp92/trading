"""Run IARIC T1 Phase 3 greedy optimization.

Phase 3: Full re-optimization from pre-P1 baseline with ALL 36 candidates
from P1 + P2 pooled together. Tests whether the greedy algorithm finds a
different/better path when all candidates are available simultaneously
instead of being split across two phases.

Base = pre-P1 baseline (6 hardcoded params, no greedy selections).
Candidates = 18 from P1 (param upgrades + new features)
           + 18 from P2 (structural improvements + speculative)
           = 36 total.

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    PYTHONUNBUFFERED=1 python -u -m research.backtests.stock.auto.output.run_greedy_iaric_t1_p3
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from research.backtests.stock.auto.greedy_optimize import run_greedy, save_result
from research.backtests.stock.engine.research_replay import ResearchReplayEngine

DATA_DIR = Path("research/backtests/stock/data/raw")
OUTPUT_DIR = Path("research/backtests/stock/auto/output")

# Base: true pre-P1 baseline (reset P2-deployed defaults to original values)
BASE_MUTATIONS: dict = {
    "param_overrides.carry_only_entry": True,
    "param_overrides.use_close_stop": True,
    "param_overrides.base_risk_fraction": 0.005,
    "param_overrides.max_carry_days": 5,
    "param_overrides.min_conviction_multiplier": 0.25,
    "max_per_sector": 2,
    # Reset P2-deployed config defaults to original pre-P1 values
    "param_overrides.t1_entry_flow_gate": False,
    "param_overrides.dow_friday_mult": 1.0,
}

# ALL candidates from P1 + P2, pooled together (36 total)
CANDIDATES: list[tuple[str, dict]] = [
    # ══════════════════════════════════════════════════════════════════
    # Phase 1 candidates (param upgrades + new features)
    # ══════════════════════════════════════════════════════════════════

    # Risk fraction alternatives
    ("risk_0075", {"param_overrides.base_risk_fraction": 0.0075}),
    ("risk_010", {"param_overrides.base_risk_fraction": 0.01}),
    ("risk_015", {"param_overrides.base_risk_fraction": 0.015}),

    # Carry duration
    ("carry_days_7", {"param_overrides.max_carry_days": 7}),
    ("carry_days_10", {"param_overrides.max_carry_days": 10}),

    # Carry R minimum
    ("carry_r_050", {"param_overrides.min_carry_r": 0.50}),
    ("carry_r_075", {"param_overrides.min_carry_r": 0.75}),
    ("carry_r_100", {"param_overrides.min_carry_r": 1.00}),

    # Flow reversal lookback
    ("flow_lb_1", {"param_overrides.flow_reversal_lookback": 1}),
    ("flow_lb_3", {"param_overrides.flow_reversal_lookback": 3}),

    # Stop risk cap
    ("stop_cap_010", {"param_overrides.stop_risk_cap_pct": 0.01}),
    ("stop_cap_015", {"param_overrides.stop_risk_cap_pct": 0.015}),

    # Entry ordering / filtering
    ("conviction_order", {"param_overrides.entry_order_by_conviction": True}),
    ("intraday_flow", {"param_overrides.intraday_flow_check": True}),
    ("top_quartile", {"param_overrides.carry_top_quartile": True}),
    ("strong_only", {"param_overrides.strong_only_entry": True}),

    # Regime B carry
    ("regime_b_carry_06", {"param_overrides.regime_b_carry_mult": 0.6}),

    # Conviction gate removal
    ("no_conviction_gate", {"param_overrides.min_conviction_multiplier": 0.0}),

    # ══════════════════════════════════════════════════════════════════
    # Phase 2 candidates (structural improvements)
    # ══════════════════════════════════════════════════════════════════

    # ── Tier 1: Structurally defensible ──

    # Entry flow gate: require positive flow proxy at entry
    ("entry_flow_pos_1", {"param_overrides.t1_entry_flow_gate": True, "param_overrides.t1_entry_flow_lookback": 1}),
    ("entry_flow_pos_2", {"param_overrides.t1_entry_flow_gate": True, "param_overrides.t1_entry_flow_lookback": 2}),

    # Intraday partial takes: lock in profit at R-trigger
    ("partial_05r_025f", {"param_overrides.t1_partial_takes": True, "param_overrides.t1_partial_r_trigger": 0.5, "param_overrides.t1_partial_fraction": 0.25}),
    ("partial_075r_033f", {"param_overrides.t1_partial_takes": True, "param_overrides.t1_partial_r_trigger": 0.75, "param_overrides.t1_partial_fraction": 0.33}),
    ("partial_10r_050f", {"param_overrides.t1_partial_takes": True, "param_overrides.t1_partial_r_trigger": 1.0, "param_overrides.t1_partial_fraction": 0.50}),

    # Gap-down entry filter: skip entries where open gaps below prev close
    ("gap_skip_005", {"param_overrides.t1_gap_down_skip_pct": 0.005}),
    ("gap_skip_010", {"param_overrides.t1_gap_down_skip_pct": 0.01}),
    ("gap_skip_015", {"param_overrides.t1_gap_down_skip_pct": 0.015}),

    # Carry close-pct: parameterize hardcoded 0.75 threshold
    ("carry_close_050", {"param_overrides.carry_top_quartile": True, "param_overrides.carry_close_pct_min": 0.50}),
    ("carry_close_060", {"param_overrides.carry_top_quartile": True, "param_overrides.carry_close_pct_min": 0.60}),
    ("carry_close_065", {"param_overrides.carry_top_quartile": True, "param_overrides.carry_close_pct_min": 0.65}),

    # ── Tier 2: Defensible with caveats ──

    # Carry trailing stop: protect accumulated carry profit
    ("carry_trail_d1_15atr", {"param_overrides.carry_trail_activate_days": 1, "param_overrides.carry_trail_atr_mult": 1.5}),
    ("carry_trail_d2_10atr", {"param_overrides.carry_trail_activate_days": 2, "param_overrides.carry_trail_atr_mult": 1.0}),
    ("carry_breakeven_d1", {"param_overrides.carry_trail_activate_days": 1, "param_overrides.carry_trail_atr_mult": 0.0}),

    # Conviction minimum tightening
    ("conviction_035", {"param_overrides.min_conviction_multiplier": 0.35}),
    ("conviction_050", {"param_overrides.min_conviction_multiplier": 0.50}),

    # ── Tier 3: Speculative ──

    # Day-of-week sizing
    ("tue_half", {"param_overrides.dow_tuesday_mult": 0.5}),
    ("fri_125", {"param_overrides.dow_friday_mult": 1.25}),
]


def main():
    print("=" * 60)
    print("IARIC T1 Phase 3 — Full Pool Re-Optimization")
    print("=" * 60)
    print(f"Base: pre-P1 baseline ({len(BASE_MUTATIONS)} params)")
    print(f"Candidates: {len(CANDIDATES)} (P1 + P2 pooled)")
    print("=" * 60, flush=True)

    # Load data
    print("\n[1/2] Loading data...", flush=True)
    t0 = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Run greedy
    print("\n[2/2] Running greedy optimization...", flush=True)
    checkpoint = OUTPUT_DIR / "greedy_checkpoint_iaric_t1_p3.json"
    result = run_greedy(
        replay,
        strategy="iaric",
        tier=1,
        base_mutations=BASE_MUTATIONS,
        candidates=CANDIDATES,
        max_workers=3,
        checkpoint_path=checkpoint,
    )

    # Save
    out_path = OUTPUT_DIR / "greedy_optimal_iaric_t1_p3.json"
    save_result(result, out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Final score: {result.final_score:.6f}")
    print(f"  Kept features: {result.kept_features}")
    total = time.time() - t0
    print(f"  Total time: {total:.0f}s ({total / 60:.1f}min)")


if __name__ == "__main__":
    main()
