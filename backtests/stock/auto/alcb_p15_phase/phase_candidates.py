"""ALCB P15 phased candidates.

P15 starts from the latest optimized P14 stack plus the post-round alpha
recovery and regime-B COMBINED cleanup:

* P14 accepted adaptive late trail and COMBINED AVWAP cap.
* FR MFE grace at 0.15R.
* COMBINED_BREAKOUT blocked in Tier B.

The round tests broad, structural alpha hypotheses before incremental sizing:
anti-chase discrimination, pullback/reclaim entries, exit/sizing repair, then
US_ORB-inspired quality, acceptance, and management extensions.
"""
from __future__ import annotations

from typing import Any


BASE_MUTATIONS: dict[str, Any] = {
    # P13/P14 optimized base.
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
    "ablation.use_adaptive_trail": True,
    "param_overrides.adaptive_trail_start_bars": 25,
    "param_overrides.adaptive_trail_tighten_bars": 25,
    "param_overrides.adaptive_trail_late_activate_r": 0.25,
    "param_overrides.adaptive_trail_late_distance_r": 0.20,
    "param_overrides.combined_avwap_cap_pct": 0.003,
    # Latest optimized cleanup.
    "param_overrides.opening_range_bars": 6,
    "param_overrides.entry_window_end": "12:30:00",
    "param_overrides.fr_trailing_activate_r": 0.0,
    "param_overrides.fr_mfe_grace_r": 0.15,
    "param_overrides.block_combined_regime_b": True,
    "param_overrides.regime_mult_b": 0.5,
}


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Entry discrimination / anti-chase",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "trades_per_month",
            "entry_quality",
            "mfe_capture_efficiency",
            "inv_dd",
        ],
    ),
    2: (
        "Structural reclaim entries",
        [
            "expected_total_r",
            "net_profit",
            "trades_per_month",
            "profit_factor",
            "entry_quality",
            "long_hold_capture",
            "inv_dd",
        ],
    ),
    3: (
        "Trade management and sizing",
        [
            "expected_total_r",
            "net_profit",
            "expectancy_dollar",
            "trades_per_month",
            "profit_factor",
            "mfe_capture_efficiency",
            "long_hold_capture",
            "inv_dd",
        ],
    ),
    4: (
        "US_ORB quality / danger filters",
        [
            "expected_total_r",
            "net_profit",
            "trades_per_month",
            "profit_factor",
            "expectancy_dollar",
            "profit_protection",
            "short_hold_drag_inverse",
            "inv_dd",
        ],
    ),
    5: (
        "US_ORB acceptance / retest entries",
        [
            "expected_total_r",
            "net_profit",
            "trades_per_month",
            "profit_factor",
            "expectancy_dollar",
            "entry_quality",
            "mfe_capture_efficiency",
            "inv_dd",
        ],
    ),
    6: (
        "US_ORB scratch / retracement exits",
        [
            "expected_total_r",
            "net_profit",
            "trades_per_month",
            "profit_factor",
            "expectancy_dollar",
            "mfe_capture_efficiency",
            "profit_protection",
            "inv_dd",
        ],
    ),
}


PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        ("p15_avwap_cap_0075", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.0075,
        }),
        ("p15_avwap_cap_0050", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.005,
        }),
        ("p15_breakout_cap_075r", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.75,
        }),
        ("p15_breakout_cap_060r", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.60,
        }),
        ("p15_avwap005_breakout075", {
            "ablation.use_avwap_distance_cap": True,
            "param_overrides.avwap_distance_cap_pct": 0.005,
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.75,
        }),
        ("p15_or_quality_rvol25", {
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_min_rvol": 2.5,
        }),
        ("p15_or_quality_score5_rvol25", {
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_score_min": 5,
            "param_overrides.or_breakout_min_rvol": 2.5,
        }),
        ("p15_late_score5_1030", {
            "param_overrides.late_entry_cutoff": "10:30:00",
            "param_overrides.late_entry_score_min": 5,
        }),
        ("p15_min_selection_6", {
            "param_overrides.min_selection_score": 6,
        }),
    ],
    2: [
        ("p15_reclaim_or", {
            "param_overrides.reclaim_entry_mode": "or",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
        }),
        ("p15_reclaim_or_rvol25", {
            "param_overrides.reclaim_entry_mode": "or",
            "param_overrides.reclaim_min_rvol": 2.5,
            "param_overrides.reclaim_cpr_threshold": 0.55,
        }),
        ("p15_reclaim_or_avwap", {
            "param_overrides.reclaim_entry_mode": "or_avwap",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
        }),
        ("p15_reclaim_or_avwap_tight", {
            "param_overrides.reclaim_entry_mode": "or_avwap",
            "param_overrides.reclaim_min_rvol": 2.5,
            "param_overrides.reclaim_cpr_threshold": 0.60,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.005,
        }),
        ("p15_reclaim_or_pdh", {
            "param_overrides.reclaim_entry_mode": "or_pdh",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
        }),
        ("p15_reclaim_or_pdh_avwap", {
            "param_overrides.reclaim_entry_mode": "or_pdh_avwap",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
        }),
        ("p15_reclaim_or_avwap_late1230", {
            "param_overrides.reclaim_entry_mode": "or_avwap",
            "param_overrides.reclaim_min_rvol": 2.25,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.entry_window_end": "12:30:00",
        }),
        ("p15_reclaim_or_avwap_wide_touch", {
            "param_overrides.reclaim_entry_mode": "or_avwap",
            "param_overrides.reclaim_min_rvol": 2.5,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_touch_tolerance_pct": 0.002,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
        }),
    ],
    3: [
        ("p15_be_035", {
            "param_overrides.close_stop_be_after_r": 0.35,
        }),
        ("p15_be_045", {
            "param_overrides.close_stop_be_after_r": 0.45,
        }),
        ("p15_fr_grace_020", {
            "param_overrides.fr_mfe_grace_r": 0.20,
        }),
        ("p15_fr_grace_025", {
            "param_overrides.fr_mfe_grace_r": 0.25,
        }),
        ("p15_fr_hold_16", {
            "param_overrides.flow_reversal_min_hold_bars": 16,
        }),
        ("p15_fr_hold_20", {
            "param_overrides.flow_reversal_min_hold_bars": 20,
        }),
        ("p15_fr_require_below_entry", {
            "param_overrides.flow_reversal_require_below_entry": True,
        }),
        ("p15_trail_mid13_wide", {
            "param_overrides.adaptive_trail_start_bars": 13,
            "param_overrides.adaptive_trail_tighten_bars": 25,
            "param_overrides.adaptive_trail_mid_activate_r": 0.35,
            "param_overrides.adaptive_trail_mid_distance_r": 0.45,
        }),
        ("p15_trail_mid16_wide", {
            "param_overrides.adaptive_trail_start_bars": 16,
            "param_overrides.adaptive_trail_tighten_bars": 28,
            "param_overrides.adaptive_trail_mid_activate_r": 0.35,
            "param_overrides.adaptive_trail_mid_distance_r": 0.40,
            "param_overrides.adaptive_trail_late_activate_r": 0.30,
            "param_overrides.adaptive_trail_late_distance_r": 0.20,
        }),
        ("p15_mfe_conviction_floor", {
            "ablation.use_mfe_conviction_exit": True,
            "param_overrides.mfe_conviction_check_bars": 16,
            "param_overrides.mfe_conviction_min_r": 0.20,
            "param_overrides.mfe_conviction_floor_r": -0.15,
        }),
        ("p15_regime_b_06", {
            "param_overrides.regime_mult_b": 0.6,
        }),
        ("p15_regime_b_07", {
            "param_overrides.regime_mult_b": 0.7,
        }),
    ],
    4: [
        ("p15_orb_quality_55", {
            "ablation.use_orb_quality_gate": True,
            "param_overrides.orb_quality_score_min": 55.0,
        }),
        ("p15_orb_quality_60", {
            "ablation.use_orb_quality_gate": True,
            "param_overrides.orb_quality_score_min": 60.0,
        }),
        ("p15_orb_quality_size_55", {
            "ablation.use_orb_quality_gate": True,
            "param_overrides.orb_quality_score_min": 55.0,
            "param_overrides.orb_quality_size_floor": 0.75,
            "param_overrides.orb_quality_top_score": 85.0,
            "param_overrides.orb_quality_top_mult": 1.15,
        }),
        ("p15_orb_gap_size", {
            "ablation.use_orb_gap_policy": True,
            "param_overrides.orb_gap_policy_mode": "size",
        }),
        ("p15_orb_gap_filter", {
            "ablation.use_orb_gap_policy": True,
            "param_overrides.orb_gap_policy_mode": "filter",
            "param_overrides.orb_quality_score_min": 55.0,
        }),
        ("p15_orb_range_cap_175r", {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 1.75,
        }),
        ("p15_orb_time_decay_mild", {
            "param_overrides.orb_late_rvol_add_per_30m": 0.15,
            "param_overrides.orb_late_size_decay_per_30m": 0.08,
            "param_overrides.orb_late_size_floor": 0.65,
        }),
        ("p15_orb_quality_gap_size", {
            "ablation.use_orb_quality_gate": True,
            "param_overrides.orb_quality_score_min": 55.0,
            "ablation.use_orb_gap_policy": True,
            "param_overrides.orb_gap_policy_mode": "size",
        }),
    ],
    5: [
        ("p15_orb_avwap_reclaim", {
            "param_overrides.reclaim_entry_mode": "or_avwap_entry",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
        }),
        ("p15_orb_avwap_reclaim_rvol225", {
            "param_overrides.reclaim_entry_mode": "or_avwap_entry",
            "param_overrides.reclaim_min_rvol": 2.25,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
        }),
        ("p15_orb_avwap_reclaim_quality55", {
            "param_overrides.reclaim_entry_mode": "or_avwap_entry",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
            "ablation.use_orb_quality_gate": True,
            "param_overrides.orb_quality_score_min": 55.0,
        }),
        ("p15_orb_avwap_reclaim_gap_size", {
            "param_overrides.reclaim_entry_mode": "or_avwap_entry",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
            "ablation.use_orb_gap_policy": True,
            "param_overrides.orb_gap_policy_mode": "size",
        }),
        ("p15_orb_reclaim_structure_stop", {
            "param_overrides.orb_structure_stop_mode": "reclaim",
            "param_overrides.orb_structure_stop_buffer_pct": 0.0015,
            "param_overrides.orb_structure_min_risk_pct": 0.60,
        }),
        ("p15_orb_avwap_reclaim_structure_stop", {
            "param_overrides.reclaim_entry_mode": "or_avwap_entry",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
            "param_overrides.orb_structure_stop_mode": "reclaim",
            "param_overrides.orb_structure_stop_buffer_pct": 0.0015,
            "param_overrides.orb_structure_min_risk_pct": 0.60,
        }),
        ("p15_orb_all_structure_stop", {
            "param_overrides.orb_structure_stop_mode": "all",
            "param_overrides.orb_structure_stop_buffer_pct": 0.0015,
            "param_overrides.orb_structure_min_risk_pct": 0.70,
        }),
        ("p15_orb_avwap_reclaim_time_decay", {
            "param_overrides.reclaim_entry_mode": "or_avwap_entry",
            "param_overrides.reclaim_min_rvol": 2.0,
            "param_overrides.reclaim_cpr_threshold": 0.55,
            "param_overrides.reclaim_max_avwap_premium_pct": 0.0075,
            "param_overrides.orb_late_rvol_add_per_30m": 0.10,
            "param_overrides.orb_late_size_decay_per_30m": 0.06,
            "param_overrides.orb_late_size_floor": 0.70,
        }),
    ],
    6: [
        ("p15_orb_scratch_2bar_020", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_qe_no_recycle": True,
            "param_overrides.quick_exit_max_bars": 2,
            "param_overrides.quick_exit_min_r": 0.20,
        }),
        ("p15_orb_scratch_2bar_040", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_qe_no_recycle": True,
            "param_overrides.quick_exit_max_bars": 2,
            "param_overrides.quick_exit_min_r": 0.40,
        }),
        ("p15_orb_scratch_3bar_020", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_qe_no_recycle": True,
            "param_overrides.quick_exit_max_bars": 3,
            "param_overrides.quick_exit_min_r": 0.20,
        }),
        ("p15_orb_partial_1r_33", {
            "ablation.use_partial_takes": True,
            "param_overrides.partial_r_trigger": 1.0,
            "param_overrides.partial_fraction": 0.33,
            "param_overrides.move_stop_to_be": True,
        }),
        ("p15_orb_partial_08r_25", {
            "ablation.use_partial_takes": True,
            "param_overrides.partial_r_trigger": 0.80,
            "param_overrides.partial_fraction": 0.25,
            "param_overrides.move_stop_to_be": True,
        }),
        ("p15_orb_retrace_trail_45_72", {
            "ablation.use_orb_retracement_trail": True,
            "param_overrides.orb_retracement_trail_start_bars": 3,
            "param_overrides.orb_retracement_trail_tighten_bars": 12,
            "param_overrides.orb_retracement_trail_min_mfe_r": 0.40,
            "param_overrides.orb_retracement_trail_early": 0.45,
            "param_overrides.orb_retracement_trail_late": 0.72,
        }),
        ("p15_orb_retrace_replace_45_72", {
            "ablation.use_adaptive_trail": False,
            "ablation.use_orb_retracement_trail": True,
            "param_overrides.orb_retracement_trail_start_bars": 3,
            "param_overrides.orb_retracement_trail_tighten_bars": 12,
            "param_overrides.orb_retracement_trail_min_mfe_r": 0.40,
            "param_overrides.orb_retracement_trail_early": 0.45,
            "param_overrides.orb_retracement_trail_late": 0.72,
        }),
        ("p15_orb_scratch_retrace", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_qe_no_recycle": True,
            "param_overrides.quick_exit_max_bars": 2,
            "param_overrides.quick_exit_min_r": 0.20,
            "ablation.use_orb_retracement_trail": True,
            "param_overrides.orb_retracement_trail_start_bars": 3,
            "param_overrides.orb_retracement_trail_tighten_bars": 12,
            "param_overrides.orb_retracement_trail_min_mfe_r": 0.40,
            "param_overrides.orb_retracement_trail_early": 0.45,
            "param_overrides.orb_retracement_trail_late": 0.72,
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
