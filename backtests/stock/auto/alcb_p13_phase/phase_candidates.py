"""ALCB P13 phase candidates -- 48 experiments across 4 phases.

P13 starts fresh from P11 base (7 mutations). Key fixes from P12 failure:
raised normalization ceilings, absolute floor hard rejects, fewer focused
experiments (48 vs 80), minimum 50-trade sample backing for defensive sizing
changes, Consumer Disc size-UP to amplify strongest alpha source.

P11 Baseline: 807 trades, PF 1.35, +54.47R, DD 4.7%, $3,147 net, 57.5% WR.
"""
from __future__ import annotations

from datetime import time
from typing import Any


def _time(hour: int, minute: int) -> time:
    return time(hour, minute)


# ---------------------------------------------------------------------------
# P11 cumulative mutations -- P13 baseline (identical to P12 baseline)
# ---------------------------------------------------------------------------
BASE_MUTATIONS: dict[str, Any] = {
    "ablation.use_or_width_min": True,
    "ablation.use_partial_takes": False,
    "param_overrides.carry_min_cpr": 0.6,
    "param_overrides.carry_min_r": 0.5,
    "param_overrides.flow_reversal_min_hold_bars": 12,
    "param_overrides.or_width_min_pct": 0.0015,
    "param_overrides.sector_mult_industrials": 0.5,
}


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Entry quality filtering",
        [
            "expectancy_dollar",
            "entry_quality",
            "expected_total_r",
            "high_rvol_edge",
            "profit_factor",
            "trades_per_month",
        ],
    ),
    2: (
        "Loss management",
        [
            "profit_protection",
            "short_hold_drag_inverse",
            "expected_total_r",
            "expectancy_dollar",
            "trades_per_month",
            "inv_dd",
        ],
    ),
    3: (
        "Winner capture",
        [
            "long_hold_capture",
            "expectancy_dollar",
            "profit_protection",
            "expected_total_r",
            "profit_factor",
            "inv_dd",
        ],
    ),
    4: (
        "Sizing & polish",
        [
            "expectancy_dollar",
            "expected_total_r",
            "inv_dd",
            "profit_factor",
            "trades_per_month",
            "entry_quality",
        ],
    ),
    5: (
        "Capture & recovery",
        [
            "long_hold_capture",
            "expectancy_dollar",
            "expected_total_r",
            "inv_dd",
            "profit_factor",
            "trades_per_month",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Phase candidates (48 total)
# ---------------------------------------------------------------------------
PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Entry Quality Filtering (14 experiments) -------------------
    # Target: RVOL 1.5-2.0 leak (-6.55R), COMBINED dead weight (PF 1.03),
    # far/extended entries.
    1: [
        # Soft RVOL: changes when bar_vol_surge score factor fires
        ("p13_rvol_soft_200", {"param_overrides.rvol_threshold": 2.0}),
        ("p13_rvol_soft_225", {"param_overrides.rvol_threshold": 2.25}),
        # Hard RVOL: eliminates ALL entries with bar RVOL below threshold
        ("p13_rvol_hard_200", {"param_overrides.breakout_min_rvol_d": 2.0}),
        # COMBINED management
        ("p13_no_combined", {"ablation.use_combined_breakout": False}),
        ("p13_comb_gate_s5_r25", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 5,
            "param_overrides.combined_breakout_min_rvol": 2.5,
        }),
        ("p13_comb_gate_s6_r30", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 6,
            "param_overrides.combined_breakout_min_rvol": 3.0,
        }),
        # Breakout distance caps
        ("p13_bkout_cap_06r", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.6,
        }),
        ("p13_bkout_cap_08r", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.8,
        }),
        # AVWAP distance caps
        ("p13_avwap_cap_005", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.005,
        }),
        ("p13_avwap_cap_008", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.008,
        }),
        # Compound: hard RVOL + distance cap
        ("p13_rvol_hard200_bkout06r", {
            "param_overrides.breakout_min_rvol_d": 2.0,
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.6,
        }),
        # Compound: soft RVOL + AVWAP cap
        ("p13_rvol_soft200_avwap005", {
            "param_overrides.rvol_threshold": 2.0,
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.005,
        }),
        # Compound: soft RVOL + COMBINED gate
        ("p13_rvol_soft200_comb_s5r25", {
            "param_overrides.rvol_threshold": 2.0,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 5,
            "param_overrides.combined_breakout_min_rvol": 2.5,
        }),
        # Compound: P12's top pick re-evaluated with fixed scorer
        ("p13_rvol_hard200_no_combined", {
            "param_overrides.breakout_min_rvol_d": 2.0,
            "ablation.use_combined_breakout": False,
        }),
    ],

    # -- Phase 2: Loss Management (12 experiments) --------------------------
    # Target: Short-hold drag (-24.51R from 632 trades <=24 bars).
    # Purgatory zone bars 7-24: -24.44R.
    2: [
        # Stage 1 quick exit (ultra-early kill)
        ("p13_qe_s1_1bar_n010", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.10,
        }),
        ("p13_qe_s1_2bar_n015", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.15,
        }),
        ("p13_qe_s1_2bar_n020", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.20,
        }),
        # Time-based quick exit (purgatory kill)
        ("p13_qe_6bar_p0", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p13_qe_8bar_p005", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 8,
            "param_overrides.quick_exit_min_r": 0.05,
        }),
        ("p13_qe_12bar_p010", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 12,
            "param_overrides.quick_exit_min_r": 0.10,
        }),
        # Breakeven stop
        ("p13_be_015r", {"param_overrides.close_stop_be_after_r": 0.15}),
        ("p13_be_020r", {"param_overrides.close_stop_be_after_r": 0.20}),
        ("p13_be_025r", {"param_overrides.close_stop_be_after_r": 0.25}),
        # Tighter initial stop
        ("p13_stop_atr_075", {"param_overrides.stop_atr_multiple": 0.75}),
        # Compound: layered exit (stage1 + timed)
        ("p13_qe_full_6bar", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.10,
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        # Compound: BE + stage1
        ("p13_be020_s1_1bar", {
            "param_overrides.close_stop_be_after_r": 0.20,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.10,
        }),
    ],

    # -- Phase 3: Winner Capture (10 experiments) ---------------------------
    # Target: Improve 42.5% MFE capture. Winners give back 0.377R avg.
    3: [
        # Trailing stop variants
        ("p13_trail_025_015", {
            "param_overrides.fr_trailing_activate_r": 0.25,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p13_trail_030_020", {
            "param_overrides.fr_trailing_activate_r": 0.30,
            "param_overrides.fr_trailing_distance_r": 0.20,
        }),
        ("p13_trail_040_020", {
            "param_overrides.fr_trailing_activate_r": 0.40,
            "param_overrides.fr_trailing_distance_r": 0.20,
        }),
        ("p13_trail_050_025", {
            "param_overrides.fr_trailing_activate_r": 0.50,
            "param_overrides.fr_trailing_distance_r": 0.25,
        }),
        # FR MFE grace -- broader protection from premature FR exits
        ("p13_fr_mfe_020", {"param_overrides.fr_mfe_grace_r": 0.20}),
        ("p13_fr_mfe_015", {"param_overrides.fr_mfe_grace_r": 0.15}),
        # FR max hold -- disable FR after N bars to protect alpha zone
        ("p13_fr_maxhold_24", {"param_overrides.fr_max_hold_bars": 24}),
        ("p13_fr_maxhold_36", {"param_overrides.fr_max_hold_bars": 36}),
        # Compound: trail + broader FR protection
        ("p13_trail030_mfe020", {
            "param_overrides.fr_trailing_activate_r": 0.30,
            "param_overrides.fr_trailing_distance_r": 0.20,
            "param_overrides.fr_mfe_grace_r": 0.20,
        }),
        # FR selectivity: only fire when CPR < threshold
        ("p13_fr_cpr_030", {"param_overrides.fr_cpr_threshold": 0.3}),
    ],

    # -- Phase 4: Sizing & Polish (12 experiments) --------------------------
    # Target: Reduce sector drag, amplify Consumer Disc alpha, timing polish.
    # 50-trade minimum for defensive mults (HC/Ind/Staples excluded).
    4: [
        # Communication: 188 trades, PF 0.92 -- adequate sample
        ("p13_comm_070", {"param_overrides.sector_mult_communication": 0.7}),
        ("p13_comm_050", {"param_overrides.sector_mult_communication": 0.5}),
        # Financials: 64 trades, PF 0.87 -- borderline sample
        ("p13_fin_070", {"param_overrides.sector_mult_financials": 0.7}),
        ("p13_fin_050", {"param_overrides.sector_mult_financials": 0.5}),
        # Tuesday: 156 trades, PF 0.96 -- only negative DoW
        ("p13_tue_080", {"param_overrides.tuesday_sizing_mult": 0.8}),
        # Consumer Disc SIZE UP: 126 trades, PF 2.91, +40.25R
        ("p13_cd_120", {"param_overrides.sector_mult_consumer_disc": 1.2}),
        ("p13_cd_130", {"param_overrides.sector_mult_consumer_disc": 1.3}),
        # Compound: reduce drag + amplify alpha
        ("p13_sector_compound", {
            "param_overrides.sector_mult_communication": 0.7,
            "param_overrides.sector_mult_financials": 0.7,
            "param_overrides.sector_mult_consumer_disc": 1.2,
        }),
        # Regime: enable B-regime at half size
        ("p13_regime_b_050", {"param_overrides.regime_mult_b": 0.50}),
        # Late entry gates
        ("p13_late5_1030", {
            "param_overrides.late_entry_cutoff": _time(10, 30),
            "param_overrides.late_entry_score_min": 5,
        }),
        ("p13_late5_1100", {
            "param_overrides.late_entry_cutoff": _time(11, 0),
            "param_overrides.late_entry_score_min": 5,
        }),
        # Frequency boost
        ("p13_max_pos_8", {"param_overrides.max_positions": 8}),
    ],

    # -- Phase 5: Capture & Recovery (14 experiments) ----------------------
    # Target: long_hold_capture at 0.484 (winners give back MFE).
    # Phase 3 trails (0.25-0.50R activation) were too tight -> hard rejected.
    # Phase 2 loss mgmt was redundant (Phase 1 solved short-hold drag).
    # Approach: wider trail activations, FR hold protection, gentle compounds.
    5: [
        # Wide trailing stops -- only lock profits on big winners
        ("p13_trail_060_030", {
            "param_overrides.fr_trailing_activate_r": 0.60,
            "param_overrides.fr_trailing_distance_r": 0.30,
        }),
        ("p13_trail_080_040", {
            "param_overrides.fr_trailing_activate_r": 0.80,
            "param_overrides.fr_trailing_distance_r": 0.40,
        }),
        ("p13_trail_100_050", {
            "param_overrides.fr_trailing_activate_r": 1.00,
            "param_overrides.fr_trailing_distance_r": 0.50,
        }),
        # FR hold protection -- disable FR on mature positions
        ("p13_fr_maxhold_48", {"param_overrides.fr_max_hold_bars": 48}),
        ("p13_fr_maxhold_60", {"param_overrides.fr_max_hold_bars": 60}),
        # FR MFE grace -- broader protection from premature FR exits
        ("p13_fr_mfe_025", {"param_overrides.fr_mfe_grace_r": 0.25}),
        ("p13_fr_mfe_030", {"param_overrides.fr_mfe_grace_r": 0.30}),
        # Compound: wide trail + FR hold protection
        ("p13_trail060_maxhold48", {
            "param_overrides.fr_trailing_activate_r": 0.60,
            "param_overrides.fr_trailing_distance_r": 0.30,
            "param_overrides.fr_max_hold_bars": 48,
        }),
        ("p13_trail080_mfe025", {
            "param_overrides.fr_trailing_activate_r": 0.80,
            "param_overrides.fr_trailing_distance_r": 0.40,
            "param_overrides.fr_mfe_grace_r": 0.25,
        }),
        # Even more selective FR (lower CPR from 0.3 to 0.25)
        ("p13_fr_cpr_025", {"param_overrides.fr_cpr_threshold": 0.25}),
        # Gentle purgatory exit (very long bar threshold)
        ("p13_qe_20bar_p010", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 20,
            "param_overrides.quick_exit_min_r": 0.10,
        }),
        # Conservative breakeven stop
        ("p13_be_040r", {"param_overrides.close_stop_be_after_r": 0.40}),
        # Carry recovery -- lower carry threshold to allow more overnight holds
        ("p13_carry_r030", {"param_overrides.carry_min_r": 0.3}),
        # Relax combined gate to recover some filtered trades
        ("p13_comb_gate_s4_r20", {
            "param_overrides.combined_breakout_score_min": 4,
            "param_overrides.combined_breakout_min_rvol": 2.0,
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
