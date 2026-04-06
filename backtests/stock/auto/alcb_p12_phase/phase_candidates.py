"""ALCB P12 phase candidates -- 80 experiments across 5 phases.

P12 starts from P11 final cumulative mutations and targets ~40-60R of
recoverable alpha across RVOL floor, COMBINED dead weight, short-hold drag,
weak MFE capture, and sector drag.
"""
from __future__ import annotations

from datetime import time
from typing import Any


def _time(hour: int, minute: int) -> time:
    return time(hour, minute)


# ---------------------------------------------------------------------------
# P11 cumulative mutations -- P12 baseline
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
        "RVOL + COMBINED filtering",
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
        "Short-hold drag kill",
        [
            "profit_protection",
            "short_hold_drag_inverse",
            "early_1000_drag_inverse",
            "expectancy_dollar",
            "expected_total_r",
            "trades_per_month",
        ],
    ),
    3: (
        "Winner capture and trail optimization",
        [
            "long_hold_capture",
            "profit_protection",
            "expectancy_dollar",
            "expected_total_r",
            "profit_factor",
            "inv_dd",
        ],
    ),
    4: (
        "Sector and DoW sizing",
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
        "Timing and score polish",
        [
            "expectancy_dollar",
            "expected_total_r",
            "profit_factor",
            "trades_per_month",
            "inv_dd",
            "entry_quality",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Phase candidates (80 total)
# ---------------------------------------------------------------------------
PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # ── Phase 1: RVOL + COMBINED Filtering (18 experiments) ──────────────
    1: [
        ("p12_rvol_175", {"param_overrides.rvol_threshold": 1.75}),
        ("p12_rvol_200", {"param_overrides.rvol_threshold": 2.0}),
        ("p12_rvol_225", {"param_overrides.rvol_threshold": 2.25}),
        ("p12_rvol_250", {"param_overrides.rvol_threshold": 2.5}),
        ("p12_no_combined", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
        }),
        ("p12_no_combined_direct", {
            "ablation.use_combined_breakout": False,
        }),
        ("p12_combined_s7_r25", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 7,
            "param_overrides.combined_breakout_min_rvol": 2.5,
        }),
        ("p12_combined_s7_r30", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 7,
            "param_overrides.combined_breakout_min_rvol": 3.0,
        }),
        ("p12_rvol200_no_combined", {
            "param_overrides.rvol_threshold": 2.0,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
        }),
        ("p12_rvol200_no_comb_direct", {
            "param_overrides.rvol_threshold": 2.0,
            "ablation.use_combined_breakout": False,
        }),
        ("p12_rvol175_no_combined", {
            "param_overrides.rvol_threshold": 1.75,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
        }),
        ("p12_rvol225_no_combined", {
            "param_overrides.rvol_threshold": 2.25,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
        }),
        ("p12_rvol200_comb_s7r30", {
            "param_overrides.rvol_threshold": 2.0,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 7,
            "param_overrides.combined_breakout_min_rvol": 3.0,
        }),
        ("p12_or_score5_rvol200", {
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_score_min": 5,
            "param_overrides.rvol_threshold": 2.0,
        }),
        ("p12_avwap_cap_005", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.005,
        }),
        ("p12_avwap_cap_008", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.008,
        }),
        ("p12_avwap_cap_005_rvol200", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.005,
            "param_overrides.rvol_threshold": 2.0,
        }),
        ("p12_bkout_cap_06r_rvol200", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.6,
            "param_overrides.rvol_threshold": 2.0,
        }),
    ],

    # ── Phase 2: Short-Hold Drag Kill (16 experiments) ───────────────────
    2: [
        ("p12_qe_s1_1bar_n010", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.10,
        }),
        ("p12_qe_s1_1bar_n015", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.15,
        }),
        ("p12_qe_s1_2bar_n010", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.10,
        }),
        ("p12_qe_s1_2bar_n020", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.20,
        }),
        ("p12_qe_full_4bar", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.10,
            "param_overrides.quick_exit_max_bars": 4,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p12_qe_full_6bar", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.15,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p12_qe_full_8bar_p005", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.15,
            "param_overrides.quick_exit_max_bars": 8,
            "param_overrides.quick_exit_min_r": 0.05,
        }),
        ("p12_qe_10bar_p005", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 10,
            "param_overrides.quick_exit_min_r": 0.05,
        }),
        ("p12_qe_12bar_p010", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 12,
            "param_overrides.quick_exit_min_r": 0.10,
        }),
        ("p12_be_015r", {"param_overrides.close_stop_be_after_r": 0.15}),
        ("p12_be_020r", {"param_overrides.close_stop_be_after_r": 0.20}),
        ("p12_be_025r", {"param_overrides.close_stop_be_after_r": 0.25}),
        ("p12_be_020r_qe_s1_1bar", {
            "param_overrides.close_stop_be_after_r": 0.20,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.10,
        }),
        ("p12_be_015r_qe_full_6bar", {
            "param_overrides.close_stop_be_after_r": 0.15,
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.15,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p12_stop_atr_075", {"param_overrides.stop_atr_multiple": 0.75}),
        ("p12_stop_atr_125", {"param_overrides.stop_atr_multiple": 1.25}),
    ],

    # ── Phase 3: Winner Capture & Trail Optimization (15 experiments) ────
    3: [
        ("p12_trail_020_010", {
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.10,
        }),
        ("p12_trail_020_015", {
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p12_trail_025_015", {
            "param_overrides.fr_trailing_activate_r": 0.25,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p12_trail_030_020", {
            "param_overrides.fr_trailing_activate_r": 0.30,
            "param_overrides.fr_trailing_distance_r": 0.20,
        }),
        ("p12_trail_040_020", {
            "param_overrides.fr_trailing_activate_r": 0.40,
            "param_overrides.fr_trailing_distance_r": 0.20,
        }),
        ("p12_trail_050_025", {
            "param_overrides.fr_trailing_activate_r": 0.50,
            "param_overrides.fr_trailing_distance_r": 0.25,
        }),
        ("p12_be_015_trail_020_010", {
            "param_overrides.close_stop_be_after_r": 0.15,
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.10,
        }),
        ("p12_be_020_trail_025_015", {
            "param_overrides.close_stop_be_after_r": 0.20,
            "param_overrides.fr_trailing_activate_r": 0.25,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p12_be_025_trail_030_020", {
            "param_overrides.close_stop_be_after_r": 0.25,
            "param_overrides.fr_trailing_activate_r": 0.30,
            "param_overrides.fr_trailing_distance_r": 0.20,
        }),
        ("p12_be_020_trail_050_025", {
            "param_overrides.close_stop_be_after_r": 0.20,
            "param_overrides.fr_trailing_activate_r": 0.50,
            "param_overrides.fr_trailing_distance_r": 0.25,
        }),
        ("p12_fr_g18_below_trail", {
            "param_overrides.flow_reversal_min_hold_bars": 18,
            "param_overrides.flow_reversal_require_below_entry": True,
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p12_fr_mfe_025", {"param_overrides.fr_mfe_grace_r": 0.25}),
        ("p12_fr_mfe_020", {"param_overrides.fr_mfe_grace_r": 0.20}),
        ("p12_fr_mfe_020_trail_025", {
            "param_overrides.fr_mfe_grace_r": 0.20,
            "param_overrides.fr_trailing_activate_r": 0.25,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p12_fr_max_hold_36", {"param_overrides.fr_max_hold_bars": 36}),
    ],

    # ── Phase 4: Sector & DoW Sizing (17 experiments) ────────────────────
    4: [
        ("p12_hc_050", {"param_overrides.sector_mult_healthcare": 0.5}),
        ("p12_hc_030", {"param_overrides.sector_mult_healthcare": 0.3}),
        ("p12_hc_000", {"param_overrides.sector_mult_healthcare": 0.0}),
        ("p12_fin_050", {"param_overrides.sector_mult_financials": 0.5}),
        ("p12_fin_030", {"param_overrides.sector_mult_financials": 0.3}),
        ("p12_fin_000", {"param_overrides.sector_mult_financials": 0.0}),
        ("p12_comm_050", {"param_overrides.sector_mult_communication": 0.5}),
        ("p12_comm_040", {"param_overrides.sector_mult_communication": 0.4}),
        ("p12_tue_080", {"param_overrides.tuesday_sizing_mult": 0.8}),
        ("p12_tue_050", {"param_overrides.tuesday_sizing_mult": 0.5}),
        ("p12_tue_000", {"param_overrides.tuesday_sizing_mult": 0.0}),
        ("p12_sector_v2", {
            "param_overrides.sector_mult_healthcare": 0.3,
            "param_overrides.sector_mult_financials": 0.5,
            "param_overrides.sector_mult_communication": 0.5,
        }),
        ("p12_sector_v3_tue", {
            "param_overrides.sector_mult_healthcare": 0.3,
            "param_overrides.sector_mult_financials": 0.5,
            "param_overrides.sector_mult_communication": 0.5,
            "param_overrides.tuesday_sizing_mult": 0.5,
        }),
        ("p12_ind_030", {"param_overrides.sector_mult_industrials": 0.3}),
        ("p12_risk_0075", {"param_overrides.base_risk_fraction": 0.0075}),
        ("p12_max8", {"param_overrides.max_positions": 8}),
        ("p12_regime_b_050", {"param_overrides.regime_mult_b": 0.50}),
    ],

    # ── Phase 5: Timing & Score Polish (14 experiments) ──────────────────
    5: [
        ("p12_end_1030", {"param_overrides.entry_window_end": _time(10, 30)}),
        ("p12_end_1100", {"param_overrides.entry_window_end": _time(11, 0)}),
        ("p12_end_1130", {"param_overrides.entry_window_end": _time(11, 30)}),
        ("p12_late5_1030", {
            "param_overrides.late_entry_cutoff": _time(10, 30),
            "param_overrides.late_entry_score_min": 5,
        }),
        ("p12_late6_1030", {
            "param_overrides.late_entry_cutoff": _time(10, 30),
            "param_overrides.late_entry_score_min": 6,
        }),
        ("p12_late5_1100", {
            "param_overrides.late_entry_cutoff": _time(11, 0),
            "param_overrides.late_entry_score_min": 5,
        }),
        ("p12_score_min3", {"param_overrides.momentum_score_min": 3}),
        ("p12_score_min4", {"param_overrides.momentum_score_min": 4}),
        ("p12_score_min5", {"param_overrides.momentum_score_min": 5}),
        ("p12_adx_15", {"param_overrides.adx_threshold": 15.0}),
        ("p12_adx_25", {"param_overrides.adx_threshold": 25.0}),
        ("p12_rs_050", {"param_overrides.min_rs_percentile": 0.50}),
        ("p12_rs_060", {"param_overrides.min_rs_percentile": 0.60}),
        ("p12_or_6bars", {"param_overrides.opening_range_bars": 6}),
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
