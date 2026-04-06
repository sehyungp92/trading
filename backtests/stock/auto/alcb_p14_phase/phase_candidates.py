"""ALCB P14 phase candidates -- 28 experiments across 3 phases.

P14 builds on P13 final config (14 cumulative mutations). Introduces 3 engine
changes to address structural alpha leaks:
  1. MFE Conviction Exit -- kill purgatory zone trades with no conviction
  2. Adaptive Trailing Stop -- time-phased trail to improve MFE capture
  3. COMBINED-specific entry filters -- AVWAP/breakout caps for COMBINED entries

P13 Final: 609 trades, PF 1.55, +52.79R, DD 3.8%, $6.08/trade, 23.5 tpm.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# P13 cumulative mutations -- P14 baseline (14 mutations)
# ---------------------------------------------------------------------------
BASE_MUTATIONS: dict[str, Any] = {
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
}


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Purgatory zone fix",
        [
            "expectancy_dollar",
            "expected_total_r",
            "profit_factor",
            "trades_per_month",
            "entry_quality",
            "inv_dd",
        ],
    ),
    2: (
        "Winner capture",
        [
            "mfe_capture_efficiency",
            "long_hold_capture",
            "expectancy_dollar",
            "expected_total_r",
            "profit_factor",
            "inv_dd",
        ],
    ),
    3: (
        "Entry refinement",
        [
            "entry_quality",
            "expectancy_dollar",
            "expected_total_r",
            "profit_factor",
            "trades_per_month",
            "inv_dd",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Phase candidates (28 total)
# ---------------------------------------------------------------------------
PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Purgatory Zone Fix (10 experiments) -------------------------
    # Target: Kill bars 7-24 trades that never showed conviction (-6.12R).
    1: [
        ("p14_mfe_12bar_015r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 12,
            "param_overrides.mfe_conviction_min_r": 0.15,
        }),
        ("p14_mfe_12bar_020r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 12,
            "param_overrides.mfe_conviction_min_r": 0.20,
        }),
        ("p14_mfe_12bar_010r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 12,
            "param_overrides.mfe_conviction_min_r": 0.10,
        }),
        ("p14_mfe_10bar_015r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 10,
            "param_overrides.mfe_conviction_min_r": 0.15,
        }),
        ("p14_mfe_8bar_015r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 8,
            "param_overrides.mfe_conviction_min_r": 0.15,
        }),
        ("p14_mfe_14bar_020r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 14,
            "param_overrides.mfe_conviction_min_r": 0.20,
        }),
        ("p14_mfe_10bar_010r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 10,
            "param_overrides.mfe_conviction_min_r": 0.10,
        }),
        ("p14_mfe_12bar_025r", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 12,
            "param_overrides.mfe_conviction_min_r": 0.25,
        }),
        # Compound: MFE conviction + stage1 early cut
        ("p14_mfe_12bar_015r_s1", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 12,
            "param_overrides.mfe_conviction_min_r": 0.15,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.30,
        }),
        # Compound: MFE conviction + breakeven stop
        ("p14_mfe_12bar_015r_be020", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 12,
            "param_overrides.mfe_conviction_min_r": 0.15,
            "param_overrides.close_stop_be_after_r": 0.20,
        }),
    ],

    # -- Phase 2: Winner Capture (10 experiments) -----------------------------
    # Target: Improve MFE capture from 42.7% toward 50%+.
    # Protect 25-48 bar alpha zone (+33.78R).
    2: [
        ("p14_adapt_13_25_020_040_030_025", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 13,
            "param_overrides.adaptive_trail_tighten_bars": 25,
            "param_overrides.adaptive_trail_mid_activate_r": 0.20,
            "param_overrides.adaptive_trail_mid_distance_r": 0.40,
            "param_overrides.adaptive_trail_late_activate_r": 0.30,
            "param_overrides.adaptive_trail_late_distance_r": 0.25,
        }),
        ("p14_adapt_13_25_025_035_035_020", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 13,
            "param_overrides.adaptive_trail_tighten_bars": 25,
            "param_overrides.adaptive_trail_mid_activate_r": 0.25,
            "param_overrides.adaptive_trail_mid_distance_r": 0.35,
            "param_overrides.adaptive_trail_late_activate_r": 0.35,
            "param_overrides.adaptive_trail_late_distance_r": 0.20,
        }),
        ("p14_adapt_10_20_015_045_025_025", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 10,
            "param_overrides.adaptive_trail_tighten_bars": 20,
            "param_overrides.adaptive_trail_mid_activate_r": 0.15,
            "param_overrides.adaptive_trail_mid_distance_r": 0.45,
            "param_overrides.adaptive_trail_late_activate_r": 0.25,
            "param_overrides.adaptive_trail_late_distance_r": 0.25,
        }),
        ("p14_adapt_13_30_020_040_040_020", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 13,
            "param_overrides.adaptive_trail_tighten_bars": 30,
            "param_overrides.adaptive_trail_mid_activate_r": 0.20,
            "param_overrides.adaptive_trail_mid_distance_r": 0.40,
            "param_overrides.adaptive_trail_late_activate_r": 0.40,
            "param_overrides.adaptive_trail_late_distance_r": 0.20,
        }),
        ("p14_adapt_15_30_025_035_030_020", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 15,
            "param_overrides.adaptive_trail_tighten_bars": 30,
            "param_overrides.adaptive_trail_mid_activate_r": 0.25,
            "param_overrides.adaptive_trail_mid_distance_r": 0.35,
            "param_overrides.adaptive_trail_late_activate_r": 0.30,
            "param_overrides.adaptive_trail_late_distance_r": 0.20,
        }),
        # Late-only: no mid trail, direct to tight
        ("p14_adapt_late_25_030_025", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 25,
            "param_overrides.adaptive_trail_tighten_bars": 25,
            "param_overrides.adaptive_trail_late_activate_r": 0.30,
            "param_overrides.adaptive_trail_late_distance_r": 0.25,
        }),
        ("p14_adapt_late_25_025_020", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 25,
            "param_overrides.adaptive_trail_tighten_bars": 25,
            "param_overrides.adaptive_trail_late_activate_r": 0.25,
            "param_overrides.adaptive_trail_late_distance_r": 0.20,
        }),
        # Mid-only: never tighten
        ("p14_adapt_mid_13_020_040", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 13,
            "param_overrides.adaptive_trail_tighten_bars": 999,
            "param_overrides.adaptive_trail_mid_activate_r": 0.20,
            "param_overrides.adaptive_trail_mid_distance_r": 0.40,
        }),
        # Compound: adaptive trail + MFE conviction
        ("p14_adapt_compound_mfe", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 13,
            "param_overrides.adaptive_trail_tighten_bars": 25,
            "param_overrides.adaptive_trail_mid_activate_r": 0.20,
            "param_overrides.adaptive_trail_mid_distance_r": 0.40,
            "param_overrides.adaptive_trail_late_activate_r": 0.30,
            "param_overrides.adaptive_trail_late_distance_r": 0.25,
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 12,
            "param_overrides.mfe_conviction_min_r": 0.15,
        }),
        # Compound: adaptive trail + breakeven stop
        ("p14_adapt_compound_be", {
            "ablation.use_adaptive_trail": True,
            "param_overrides.adaptive_trail_start_bars": 13,
            "param_overrides.adaptive_trail_tighten_bars": 25,
            "param_overrides.adaptive_trail_mid_activate_r": 0.20,
            "param_overrides.adaptive_trail_mid_distance_r": 0.40,
            "param_overrides.adaptive_trail_late_activate_r": 0.30,
            "param_overrides.adaptive_trail_late_distance_r": 0.25,
            "param_overrides.close_stop_be_after_r": 0.20,
        }),
    ],

    # -- Phase 3: Entry Refinement (8 experiments) ----------------------------
    # Target: COMBINED drag (-1.53R) and extended entry filtering.
    3: [
        ("p14_comb_avwap_005", {
            "param_overrides.combined_avwap_cap_pct": 0.005,
        }),
        ("p14_comb_avwap_003", {
            "param_overrides.combined_avwap_cap_pct": 0.003,
        }),
        ("p14_comb_bkout_04r", {
            "param_overrides.combined_breakout_cap_r": 0.4,
        }),
        ("p14_comb_bkout_06r", {
            "param_overrides.combined_breakout_cap_r": 0.6,
        }),
        ("p14_comb_avwap005_bkout04r", {
            "param_overrides.combined_avwap_cap_pct": 0.005,
            "param_overrides.combined_breakout_cap_r": 0.4,
        }),
        # Global AVWAP cap re-test with P14 engine changes
        ("p14_avwap_global_005", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.005,
        }),
        # Full COMBINED disable re-test with P14 engine changes
        ("p14_no_combined", {
            "ablation.use_combined_breakout": False,
        }),
        # Tighter COMBINED quality gate
        ("p14_comb_gate_s6_r30", {
            "param_overrides.combined_breakout_score_min": 6,
            "param_overrides.combined_breakout_min_rvol": 3.0,
        }),
    ],
}


def get_phase_candidates(
    phase: int,
    *,
    experiment_filter: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    experiments = list(PHASE_CANDIDATES.get(phase, []))
    if experiment_filter:
        experiments = [
            (name, mutations)
            for name, mutations in experiments
            if name in experiment_filter
        ]
    return experiments
