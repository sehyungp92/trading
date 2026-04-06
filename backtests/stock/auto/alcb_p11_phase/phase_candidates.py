from __future__ import annotations

import json
from datetime import time
from typing import Any

# P10 constants were removed during legacy cleanup; P11 is superseded by P14.
# Empty list preserves structure if this module is ever imported.
ALCB_T2_P10_CANDIDATES: list[tuple[str, dict]] = []


def _time(hour: int, minute: int) -> time:
    return time(hour, minute)


# ---------------------------------------------------------------------------
# Clean baseline -- start from default engine config, not bug-era P10b.
# ---------------------------------------------------------------------------
BASE_MUTATIONS: dict[str, Any] = {}


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Entry quality gating",
        [
            "expectancy_dollar",
            "entry_quality",
            "expected_total_r",
            "high_rvol_edge",
            "trades_per_month",
            "profit_factor",
            "inv_dd",
        ],
    ),
    2: (
        "Loss control -- FR reform, quick exit, stop width",
        [
            "profit_protection",
            "short_hold_drag_inverse",
            "flow_reversal_short_inverse",
            "early_1000_drag_inverse",
            "expectancy_dollar",
            "trades_per_month",
        ],
    ),
    3: (
        "Winner capture",
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
        "Carry and overnight alpha",
        [
            "carry_capture",
            "long_hold_capture",
            "expectancy_dollar",
            "expected_total_r",
            "inv_dd",
            "profit_factor",
        ],
    ),
    5: (
        "Sizing, sectors, and risk",
        [
            "expectancy_dollar",
            "expected_total_r",
            "inv_dd",
            "profit_factor",
            "trades_per_month",
            "long_hold_capture",
            "carry_capture",
        ],
    ),
    6: (
        "Session timing polish",
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
# PHASE_EXTRA_CANDIDATES: P10-plus unique extras + P11 candidates (ablation-
# fixed).  Duplicates between P10-plus and P11 resolved in favor of P11 name.
# ---------------------------------------------------------------------------
PHASE_EXTRA_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # ── Phase 1: Entry Quality & Signal Gating ────────────────────────────
    1: [
        # P10-plus extras (14 unique)
        ("p10x_or_5bars", {"param_overrides.opening_range_bars": 5}),
        ("p10x_breakout_cap_12r", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 1.2,
        }),
        ("p10x_breakout_cap_12r_rvol225", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 1.2,
            "param_overrides.rvol_threshold": 2.25,
        }),
        ("p10x_rvol_min_250", {"param_overrides.rvol_threshold": 2.5}),
        ("p10x_rvol_min_275", {"param_overrides.rvol_threshold": 2.75}),
        ("p10x_or_score4_rvol225", {
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_score_min": 4,
            "param_overrides.or_breakout_min_rvol": 2.25,
        }),
        ("p10x_or_score5_rvol200", {
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_score_min": 5,
            "param_overrides.or_breakout_min_rvol": 2.0,
        }),
        ("p10x_or_score5_rvol225", {
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_score_min": 5,
            "param_overrides.or_breakout_min_rvol": 2.25,
        }),
        ("p10x_combined_score7_rvol20", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 7,
            "param_overrides.combined_breakout_min_rvol": 2.0,
        }),
        ("p10x_combined_score7_rvol25", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 7,
            "param_overrides.combined_breakout_min_rvol": 2.5,
        }),
        # OR-only (ablation fix: enable combined gate + score 99 to suppress)
        ("p10x_or_only_rvol225", {
            "ablation.use_prior_day_high_breakout": False,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
            "param_overrides.rvol_threshold": 2.25,
        }),
        ("p10x_or_only_orwidth020", {
            "ablation.use_prior_day_high_breakout": False,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.002,
        }),
        ("p10x_or_only_score5_rvol225", {
            "ablation.use_prior_day_high_breakout": False,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_score_min": 5,
            "param_overrides.or_breakout_min_rvol": 2.25,
        }),
        ("p10x_rvol250_orwidth020", {
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.002,
            "param_overrides.rvol_threshold": 2.5,
        }),
        # P11 candidates (19)
        ("p11_rvol_175", {"param_overrides.rvol_threshold": 1.75}),
        ("p11_rvol_200", {"param_overrides.rvol_threshold": 2.0}),
        ("p11_rvol_225", {"param_overrides.rvol_threshold": 2.25}),
        ("p11_orwidth_015", {
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.0015,
        }),
        ("p11_orwidth_020", {
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.002,
        }),
        ("p11_orwidth_025", {
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.0025,
        }),
        # COMBINED suppression (ablation fix: enable combined gate)
        ("p11_no_combined", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
        }),
        ("p11_combined_s7r30", {
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 7,
            "param_overrides.combined_breakout_min_rvol": 3.0,
        }),
        # PDH suppression
        ("p11_no_pdh", {"ablation.use_prior_day_high_breakout": False}),
        # OR score floor (ablation fix: enable OR quality gate)
        ("p11_or_score5", {
            "ablation.use_or_quality_gate": True,
            "param_overrides.or_breakout_score_min": 5,
        }),
        # Breakout distance (ablation fix: enable breakout distance cap)
        ("p11_bkout_cap_10r", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 1.0,
        }),
        ("p11_bkout_cap_08r", {
            "ablation.use_breakout_distance_cap": True,
            "param_overrides.breakout_distance_cap_r": 0.8,
        }),
        # Compound: RVOL + OR width
        ("p11_rvol200_orw015", {
            "param_overrides.rvol_threshold": 2.0,
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.0015,
        }),
        ("p11_rvol225_orw020", {
            "param_overrides.rvol_threshold": 2.25,
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.002,
        }),
        # OR-only mode (ablation fix: enable combined gate to suppress)
        ("p11_or_only", {
            "ablation.use_prior_day_high_breakout": False,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
        }),
        ("p11_or_only_rvol200", {
            "ablation.use_prior_day_high_breakout": False,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
            "param_overrides.rvol_threshold": 2.0,
        }),
        ("p11_or_only_rvol225_orw020", {
            "ablation.use_prior_day_high_breakout": False,
            "ablation.use_combined_quality_gate": True,
            "param_overrides.combined_breakout_score_min": 99,
            "param_overrides.rvol_threshold": 2.25,
            "ablation.use_or_width_min": True,
            "param_overrides.or_width_min_pct": 0.002,
        }),
        # OR bar count
        ("p11_or_6bars", {"param_overrides.opening_range_bars": 6}),
        ("p11_or_9bars", {"param_overrides.opening_range_bars": 9}),
    ],

    # ── Phase 2: Loss Control ─────────────────────────────────────────────
    2: [
        # P10-plus extras (9)
        ("p10x_qe_1n01_4p00", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.1,
            "param_overrides.quick_exit_max_bars": 4,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p10x_qe_2n03_8p00", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.3,
            "param_overrides.quick_exit_max_bars": 8,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p10x_qe_2n02_6p00", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.2,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p10x_qe_2n01_6p00", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.1,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p10x_qe_3n02_8p00", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 3,
            "param_overrides.qe_stage1_min_r": -0.2,
            "param_overrides.quick_exit_max_bars": 8,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p10x_fr_grace18_below", {
            "param_overrides.flow_reversal_min_hold_bars": 18,
            "param_overrides.flow_reversal_require_below_entry": True,
        }),
        ("p10x_fr_grace18_below_mfe02", {
            "param_overrides.flow_reversal_min_hold_bars": 18,
            "param_overrides.flow_reversal_require_below_entry": True,
            "param_overrides.fr_mfe_grace_r": 0.2,
        }),
        ("p10x_fr_grace24_below_mfe03", {
            "param_overrides.flow_reversal_min_hold_bars": 24,
            "param_overrides.flow_reversal_require_below_entry": True,
            "param_overrides.fr_mfe_grace_r": 0.3,
        }),
        ("p10x_fr_grace24_below_mfe02_qe2", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.flow_reversal_min_hold_bars": 24,
            "param_overrides.flow_reversal_require_below_entry": True,
            "param_overrides.fr_mfe_grace_r": 0.2,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.2,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        # P11 candidates (21)
        ("p11_no_fr", {"ablation.use_flow_reversal_exit": False}),
        ("p11_fr_grace6", {"param_overrides.flow_reversal_min_hold_bars": 6}),
        ("p11_fr_grace12", {"param_overrides.flow_reversal_min_hold_bars": 12}),
        ("p11_fr_g12_below", {
            "param_overrides.flow_reversal_min_hold_bars": 12,
            "param_overrides.flow_reversal_require_below_entry": True,
        }),
        ("p11_fr_g6_below", {
            "param_overrides.flow_reversal_min_hold_bars": 6,
            "param_overrides.flow_reversal_require_below_entry": True,
        }),
        ("p11_fr_mfe_015", {"param_overrides.fr_mfe_grace_r": 0.15}),
        ("p11_fr_mfe_020", {"param_overrides.fr_mfe_grace_r": 0.20}),
        ("p11_fr_cmpd_g12_below_mfe015", {
            "param_overrides.flow_reversal_min_hold_bars": 12,
            "param_overrides.flow_reversal_require_below_entry": True,
            "param_overrides.fr_mfe_grace_r": 0.15,
        }),
        ("p11_fr_cmpd_g6_below_mfe020", {
            "param_overrides.flow_reversal_min_hold_bars": 6,
            "param_overrides.flow_reversal_require_below_entry": True,
            "param_overrides.fr_mfe_grace_r": 0.20,
        }),
        # QE stage1 (ablation fix: enable use_quick_exit_stage1)
        ("p11_qe_1bar_n015", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.15,
        }),
        ("p11_qe_2bar_n020", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.20,
        }),
        ("p11_qe_2bar_n010", {
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.10,
        }),
        # QE full (already has both ablation flags)
        ("p11_qe_full_6bar", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 2,
            "param_overrides.qe_stage1_min_r": -0.15,
            "param_overrides.quick_exit_max_bars": 6,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        ("p11_qe_full_4bar", {
            "ablation.use_time_based_quick_exit": True,
            "ablation.use_quick_exit_stage1": True,
            "param_overrides.qe_stage1_bars": 1,
            "param_overrides.qe_stage1_min_r": -0.10,
            "param_overrides.quick_exit_max_bars": 4,
            "param_overrides.quick_exit_min_r": 0.0,
        }),
        # Purgatory zone (ablation fix: enable use_time_based_quick_exit)
        ("p11_qe_10bar_p005", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 10,
            "param_overrides.quick_exit_min_r": 0.05,
        }),
        ("p11_qe_8bar_p010", {
            "ablation.use_time_based_quick_exit": True,
            "param_overrides.quick_exit_max_bars": 8,
            "param_overrides.quick_exit_min_r": 0.10,
        }),
        # Stop width
        ("p11_stop_atr_150", {"param_overrides.stop_atr_multiple": 1.5}),
        ("p11_stop_atr_200", {"param_overrides.stop_atr_multiple": 2.0}),
        # Breakeven stop
        ("p11_be_010r", {"param_overrides.close_stop_be_after_r": 0.10}),
        ("p11_be_015r", {"param_overrides.close_stop_be_after_r": 0.15}),
        ("p11_be_020r", {"param_overrides.close_stop_be_after_r": 0.20}),
    ],

    # ── Phase 3: Winner Capture ───────────────────────────────────────────
    3: [
        # P10-plus extras (6)
        ("p10x_mfe_trail_02_02_be_02", {
            "param_overrides.fr_trailing_activate_r": 0.2,
            "param_overrides.fr_trailing_distance_r": 0.2,
            "param_overrides.close_stop_be_after_r": 0.2,
        }),
        ("p10x_mfe_trail_025_02_be_02", {
            "param_overrides.fr_trailing_activate_r": 0.25,
            "param_overrides.fr_trailing_distance_r": 0.2,
            "param_overrides.close_stop_be_after_r": 0.2,
        }),
        ("p10x_mfe_trail_02_03_be_02", {
            "param_overrides.fr_trailing_activate_r": 0.2,
            "param_overrides.fr_trailing_distance_r": 0.3,
            "param_overrides.close_stop_be_after_r": 0.2,
        }),
        ("p10x_mfe_trail_03_02_be_02", {
            "param_overrides.fr_trailing_activate_r": 0.3,
            "param_overrides.fr_trailing_distance_r": 0.2,
            "param_overrides.close_stop_be_after_r": 0.2,
        }),
        ("p10x_partial_075r_020f_be_02", {
            "ablation.use_partial_takes": True,
            "param_overrides.partial_r_trigger": 0.75,
            "param_overrides.partial_fraction": 0.20,
            "param_overrides.close_stop_be_after_r": 0.2,
        }),
        ("p10x_partial_10r_025f_be_03", {
            "ablation.use_partial_takes": True,
            "param_overrides.partial_r_trigger": 1.0,
            "param_overrides.partial_fraction": 0.25,
            "param_overrides.close_stop_be_after_r": 0.3,
        }),
        # P11 candidates (14)
        ("p11_part_050_015f", {
            "param_overrides.partial_r_trigger": 0.50,
            "param_overrides.partial_fraction": 0.15,
        }),
        ("p11_part_050_020f", {
            "param_overrides.partial_r_trigger": 0.50,
            "param_overrides.partial_fraction": 0.20,
        }),
        ("p11_part_075_015f", {
            "param_overrides.partial_r_trigger": 0.75,
            "param_overrides.partial_fraction": 0.15,
        }),
        ("p11_part_075_020f", {
            "param_overrides.partial_r_trigger": 0.75,
            "param_overrides.partial_fraction": 0.20,
        }),
        ("p11_part_100_020f", {
            "param_overrides.partial_r_trigger": 1.00,
            "param_overrides.partial_fraction": 0.20,
        }),
        ("p11_trail_020_015", {
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p11_trail_025_015", {
            "param_overrides.fr_trailing_activate_r": 0.25,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
        ("p11_trail_050_025", {
            "param_overrides.fr_trailing_activate_r": 0.50,
            "param_overrides.fr_trailing_distance_r": 0.25,
        }),
        ("p11_part050_trail020_be010", {
            "param_overrides.partial_r_trigger": 0.50,
            "param_overrides.partial_fraction": 0.15,
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.15,
            "param_overrides.close_stop_be_after_r": 0.10,
        }),
        ("p11_part075_trail025_be015", {
            "param_overrides.partial_r_trigger": 0.75,
            "param_overrides.partial_fraction": 0.20,
            "param_overrides.fr_trailing_activate_r": 0.25,
            "param_overrides.fr_trailing_distance_r": 0.15,
            "param_overrides.close_stop_be_after_r": 0.15,
        }),
        ("p11_part050_033f_trail050", {
            "param_overrides.partial_r_trigger": 0.50,
            "param_overrides.partial_fraction": 0.33,
            "param_overrides.fr_trailing_activate_r": 0.50,
            "param_overrides.fr_trailing_distance_r": 0.30,
        }),
        ("p11_nopart_trail020_be010", {
            "ablation.use_partial_takes": False,
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.15,
            "param_overrides.close_stop_be_after_r": 0.10,
        }),
        ("p11_be010_trail015", {
            "param_overrides.close_stop_be_after_r": 0.10,
            "param_overrides.fr_trailing_activate_r": 0.15,
            "param_overrides.fr_trailing_distance_r": 0.10,
        }),
        ("p11_be015_trail020", {
            "param_overrides.close_stop_be_after_r": 0.15,
            "param_overrides.fr_trailing_activate_r": 0.20,
            "param_overrides.fr_trailing_distance_r": 0.15,
        }),
    ],

    # ── Phase 4: Carry and Overnight Alpha ────────────────────────────────
    4: [
        # P10-plus extras (1)
        ("p10x_no_carry_risk015", {
            "ablation.use_carry_logic": False,
            "param_overrides.base_risk_fraction": 0.015,
        }),
        # P11 candidates (8)
        ("p11_carry_r025", {"param_overrides.carry_min_r": 0.25}),
        ("p11_carry_r015", {"param_overrides.carry_min_r": 0.15}),
        ("p11_carry_r000", {"param_overrides.carry_min_r": 0.0}),
        ("p11_carry_cpr030", {"param_overrides.carry_min_cpr": 0.30}),
        ("p11_carry_days3", {"param_overrides.max_carry_days": 3}),
        ("p11_carry_days5", {"param_overrides.max_carry_days": 5}),
        ("p11_carry_r015_d5", {
            "param_overrides.carry_min_r": 0.15,
            "param_overrides.max_carry_days": 5,
        }),
        ("p11_carry_r000_d3", {
            "param_overrides.carry_min_r": 0.0,
            "param_overrides.max_carry_days": 3,
        }),
    ],

    # ── Phase 5: Sizing, Sectors, and Risk ────────────────────────────────
    5: [
        # P10-plus extras (5)
        ("p10x_risk_0125", {"param_overrides.base_risk_fraction": 0.0125}),
        ("p10x_risk_015_stop_0125", {
            "param_overrides.base_risk_fraction": 0.015,
            "param_overrides.stop_risk_cap_pct": 0.0125,
        }),
        ("p10x_max10_risk015", {
            "param_overrides.max_positions": 10,
            "param_overrides.base_risk_fraction": 0.015,
        }),
        ("p10x_max12_risk0125", {
            "param_overrides.max_positions": 12,
            "param_overrides.base_risk_fraction": 0.0125,
        }),
        ("p10x_sector_balanced", {
            "param_overrides.sector_mult_financials": 0.9,
            "param_overrides.sector_mult_communication": 0.9,
            "param_overrides.sector_mult_consumer_disc": 1.05,
        }),
        # P11 candidates (14)
        ("p11_fin_060", {"param_overrides.sector_mult_financials": 0.6}),
        ("p11_fin_040", {"param_overrides.sector_mult_financials": 0.4}),
        ("p11_cd_120", {"param_overrides.sector_mult_consumer_disc": 1.2}),
        ("p11_comm_060", {"param_overrides.sector_mult_communication": 0.6}),
        ("p11_sector_v1", {
            "param_overrides.sector_mult_financials": 0.5,
            "param_overrides.sector_mult_communication": 0.6,
            "param_overrides.sector_mult_consumer_disc": 1.2,
        }),
        ("p11_thu_115", {"param_overrides.thursday_sizing_mult": 1.15}),
        ("p11_thu_125", {"param_overrides.thursday_sizing_mult": 1.25}),
        ("p11_risk_0075", {"param_overrides.base_risk_fraction": 0.0075}),
        ("p11_risk_010", {"param_overrides.base_risk_fraction": 0.010}),
        ("p11_regime_b_025", {"param_overrides.regime_mult_b": 0.25}),
        ("p11_regime_b_050", {"param_overrides.regime_mult_b": 0.50}),
        ("p11_max8", {"param_overrides.max_positions": 8}),
        ("p11_max8_risk0075", {
            "param_overrides.max_positions": 8,
            "param_overrides.base_risk_fraction": 0.0075,
        }),
        ("p11_max12_risk0075", {
            "param_overrides.max_positions": 12,
            "param_overrides.base_risk_fraction": 0.0075,
        }),
    ],

    # ── Phase 6: Session Timing ───────────────────────────────────────────
    6: [
        # P10-plus extras (4)
        ("p10x_entry_end_1015", {"param_overrides.entry_window_end": _time(10, 15)}),
        ("p10x_entry_end_1045", {"param_overrides.entry_window_end": _time(10, 45)}),
        ("p10x_score5_rs60", {
            "param_overrides.momentum_score_min": 5,
            "param_overrides.min_rs_percentile": 0.60,
        }),
        ("p10x_score5_rs70", {
            "param_overrides.momentum_score_min": 5,
            "param_overrides.min_rs_percentile": 0.70,
        }),
        # P11 candidates (10)
        ("p11_end_1030", {"param_overrides.entry_window_end": _time(10, 30)}),
        ("p11_end_1100", {"param_overrides.entry_window_end": _time(11, 0)}),
        ("p11_end_1130", {"param_overrides.entry_window_end": _time(11, 30)}),
        ("p11_late5_1030", {
            "param_overrides.late_entry_cutoff": _time(10, 30),
            "param_overrides.late_entry_score_min": 5,
        }),
        ("p11_late6_1030", {
            "param_overrides.late_entry_cutoff": _time(10, 30),
            "param_overrides.late_entry_score_min": 6,
        }),
        ("p11_late5_1100", {
            "param_overrides.late_entry_cutoff": _time(11, 0),
            "param_overrides.late_entry_score_min": 5,
        }),
        ("p11_rs_050", {"param_overrides.min_rs_percentile": 0.50}),
        ("p11_rs_060", {"param_overrides.min_rs_percentile": 0.60}),
        ("p11_heat_8r", {"param_overrides.heat_cap_r": 8.0}),
        ("p11_heat_12r", {"param_overrides.heat_cap_r": 12.0}),
    ],
}


# ---------------------------------------------------------------------------
# P10 union reclassification into 6 phases
# ---------------------------------------------------------------------------
def _phase_for_p10_candidate(name: str, mutations: dict) -> int:
    """Classify a P10 union candidate into one of 6 phases.

    Returns -1 to skip IARIC-only candidates that use parameters not wired
    in the ALCB engine.
    """
    # Filter IARIC-only candidates
    mutations_str = str(mutations)
    if "carry_top_quartile" in mutations_str or "carry_close_pct_min" in mutations_str:
        return -1

    # Phase 6: Session timing (checked FIRST -- carved from old Phase 1)
    if name.startswith("entry_end_") or name.startswith("late_") or name.startswith("rs_pct_"):
        return 6

    # Phase 1: Entry quality + signal gating
    if (
        (name.startswith("or_") and not name.startswith("or_width_") and not name.startswith("or_score_"))
        or name.startswith("score_min_")
        or name.startswith("sel_score_")
        or name.startswith("atr_usd_")
        or name in {"no_avwap_filter", "no_pdh_breakout"}
        or name.startswith("rvol_min_")
        or name.startswith("rvol_max_")
        or name.startswith("combined_")
        or name.startswith("avwap_cap_")
        or name.startswith("or_width_min_")
        or name.startswith("breakout_dist_cap_")
        or name.startswith("or_score_min_")
    ):
        return 1

    # Phase 2: Loss control
    if (
        name == "no_flow_reversal" or name == "fr_below_entry"
        or name == "fr_grace12_below" or name == "fr_grace36_below"
        or name.startswith("fr_grace_") or name.startswith("fr_mfe_")
        or name.startswith("fr_cpr_") or name.startswith("fr_max_hold_")
        or name.startswith("fr_compound")
        or name.startswith("quick_exit_") or name.startswith("qe_")
        or name.startswith("early_qe_")
    ):
        return 2

    # Phase 3: Winner capture
    if (
        name == "no_partials"
        or name.startswith("partial_") or name.startswith("fr_trail_")
        or name.startswith("be_after_") or name.startswith("be_stop_")
        or name.startswith("mfe_trail_") or name.startswith("runner_")
    ):
        return 3

    # Phase 4: Carry & overnight
    if name in {"no_carry", "no_carry_tweaks", "cpr_0.5"} or name.startswith("carry_"):
        return 4

    # Phase 5: Sizing & risk
    if (
        name.startswith("max_pos_") or name == "bigger_portfolio"
        or name.startswith("risk_") or name.startswith("stop_")
        or name.startswith("regime_b_") or name.startswith("thursday_")
        or name.startswith("sector_sz_")
    ):
        return 5

    raise ValueError(f"Unclassified P10 candidate: {name}")


def _mutation_sig(mutations: dict) -> str:
    """Deterministic signature for dedup: json of sorted items."""
    return json.dumps(sorted(mutations.items()), default=str)


def _build_phase_candidates() -> dict[int, list[tuple[str, dict[str, Any]]]]:
    """Merge P10 union + PHASE_EXTRA_CANDIDATES with mutation-dict dedup.

    PHASE_EXTRA entries take priority -- if a P10 union candidate has the
    same mutation dict as an extra in the same phase, only the extra is kept.
    """
    phase_map: dict[int, list[tuple[str, dict[str, Any]]]] = {
        phase: list(PHASE_EXTRA_CANDIDATES.get(phase, []))
        for phase in range(1, 7)
    }

    # Build per-phase mutation signature sets from extras for dedup
    extra_sigs: dict[int, set[str]] = {}
    for phase in phase_map:
        sigs = set()
        for _, mutations in phase_map[phase]:
            sigs.add(_mutation_sig(mutations))
        extra_sigs[phase] = sigs

    # Add P10 union candidates, skipping IARIC-only and mutation duplicates
    for name, mutations in ALCB_T2_P10_CANDIDATES:
        phase = _phase_for_p10_candidate(name, dict(mutations))
        if phase == -1:
            continue
        mut_dict = dict(mutations)
        sig = _mutation_sig(mut_dict)
        if sig in extra_sigs.get(phase, set()):
            continue
        extra_sigs.setdefault(phase, set()).add(sig)
        phase_map[phase].append((name, mut_dict))

    return phase_map


PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = _build_phase_candidates()

# Validate no duplicate candidate names across phases
_phase_names = [
    name
    for phase in sorted(PHASE_CANDIDATES)
    for name, _ in PHASE_CANDIDATES[phase]
]
if len(_phase_names) != len(set(_phase_names)):
    _seen: set[str] = set()
    for _n in _phase_names:
        if _n in _seen:
            raise RuntimeError(f"Duplicate candidate name: {_n}")
        _seen.add(_n)


def get_phase_candidates(
    phase: int,
    *,
    experiment_filter: set[str] | None = None,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    experiments = list(PHASE_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    if experiment_filter:
        experiments = [
            (name, mutations)
            for name, mutations in experiments
            if name in experiment_filter
        ]
    return experiments
