"""ALCB P16 phased candidates.

P16 starts from the latest accepted P15 winner bundle and directly targets the
remaining entry leaks visible in the final diagnostics:

* weak 10:10 / bar-9 behavior
* weaker late entries
* clear AVWAP-premium degradation above +0.5%
* PDH entries lagging OR entries
* imperfect momentum-score monotonicity
"""
from __future__ import annotations

from typing import Any


BASE_MUTATIONS: dict[str, Any] = {
    # Optimized P15 winner bundle.
    "ablation.use_adaptive_trail": True,
    "ablation.use_combined_quality_gate": True,
    "ablation.use_mfe_conviction_exit": True,
    "ablation.use_or_width_min": True,
    "ablation.use_partial_takes": False,
    "param_overrides.adaptive_trail_late_activate_r": 0.25,
    "param_overrides.adaptive_trail_late_distance_r": 0.20,
    "param_overrides.adaptive_trail_start_bars": 25,
    "param_overrides.adaptive_trail_tighten_bars": 25,
    "param_overrides.block_combined_regime_b": True,
    "param_overrides.carry_min_cpr": 0.6,
    "param_overrides.carry_min_r": 0.5,
    "param_overrides.combined_avwap_cap_pct": 0.003,
    "param_overrides.combined_breakout_min_rvol": 2.5,
    "param_overrides.combined_breakout_score_min": 5,
    "param_overrides.entry_window_end": "12:30:00",
    "param_overrides.flow_reversal_min_hold_bars": 12,
    "param_overrides.fr_cpr_threshold": 0.3,
    "param_overrides.fr_mfe_grace_r": 0.20,
    "param_overrides.fr_trailing_activate_r": 0.0,
    "param_overrides.mfe_conviction_check_bars": 16,
    "param_overrides.mfe_conviction_floor_r": -0.15,
    "param_overrides.mfe_conviction_min_r": 0.20,
    "param_overrides.opening_range_bars": 6,
    "param_overrides.or_width_min_pct": 0.0015,
    "param_overrides.regime_mult_b": 0.7,
    "param_overrides.rvol_threshold": 2.0,
    "param_overrides.sector_mult_consumer_disc": 1.2,
    "param_overrides.sector_mult_financials": 0.5,
    "param_overrides.sector_mult_industrials": 0.5,
}


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "AVWAP premium discipline",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "expectancy_dollar",
            "trades_per_month",
            "extended_avwap_inverse",
        ],
    ),
    2: (
        "Bar-9 and late-entry repair",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "trades_per_month",
            "bar9_inverse",
            "late_entry_quality",
        ],
    ),
    3: (
        "PDH discrimination",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "expectancy_dollar",
            "pdh_avg_r",
            "trades_per_month",
        ],
    ),
    4: (
        "Momentum score-shape normalization",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "expectancy_dollar",
            "score_monotonicity",
            "inv_dd",
        ],
    ),
    5: (
        "Targeted synthesis",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "expectancy_dollar",
            "trades_per_month",
            "inv_dd",
        ],
    ),
}


PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        (
            "p16_avwap_cap_0050",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0050,
            },
        ),
        (
            "p16_avwap_cap_0045",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0045,
            },
        ),
        (
            "p16_avwap_cap_0060",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0060,
            },
        ),
        (
            "p16_pdh_avwap_cap_0050",
            {
                "param_overrides.pdh_avwap_cap_pct": 0.0050,
            },
        ),
        (
            "p16_late_avwap_cap_0050",
            {
                "param_overrides.late_avwap_cap_pct": 0.0050,
            },
        ),
        (
            "p16_bar9_avwap_cap_0040",
            {
                "param_overrides.bar9_avwap_cap_pct": 0.0040,
            },
        ),
        (
            "p16_pdh_avwap_cap_0045_late_0050",
            {
                "param_overrides.pdh_avwap_cap_pct": 0.0045,
                "param_overrides.late_avwap_cap_pct": 0.0050,
            },
        ),
        (
            "p16_avwap_cap_0050_plus_late_score6",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0050,
                "param_overrides.late_entry_cutoff": "11:00:00",
                "param_overrides.late_entry_score_min": 6,
            },
        ),
    ],
    2: [
        (
            "p16_bar9_score6",
            {
                "param_overrides.bar9_score_min": 6,
            },
        ),
        (
            "p16_bar9_rvol30",
            {
                "param_overrides.bar9_rvol_min": 3.0,
            },
        ),
        (
            "p16_bar9_score6_cap004",
            {
                "param_overrides.bar9_score_min": 6,
                "param_overrides.bar9_avwap_cap_pct": 0.0040,
            },
        ),
        (
            "p16_late_score6_1100",
            {
                "param_overrides.late_entry_cutoff": "11:00:00",
                "param_overrides.late_entry_score_min": 6,
            },
        ),
        (
            "p16_late_score6_1030",
            {
                "param_overrides.late_entry_cutoff": "10:30:00",
                "param_overrides.late_entry_score_min": 6,
            },
        ),
        (
            "p16_end_1200",
            {
                "param_overrides.entry_window_end": "12:00:00",
            },
        ),
        (
            "p16_end_1130",
            {
                "param_overrides.entry_window_end": "11:30:00",
            },
        ),
        (
            "p16_late_size_070_1130",
            {
                "param_overrides.late_entry_cutoff": "11:30:00",
                "param_overrides.late_entry_size_mult": 0.70,
            },
        ),
    ],
    3: [
        (
            "p16_pdh_score6",
            {
                "param_overrides.pdh_breakout_score_min": 6,
            },
        ),
        (
            "p16_pdh_rvol25",
            {
                "param_overrides.pdh_breakout_min_rvol": 2.5,
            },
        ),
        (
            "p16_pdh_score6_rvol25",
            {
                "param_overrides.pdh_breakout_score_min": 6,
                "param_overrides.pdh_breakout_min_rvol": 2.5,
            },
        ),
        (
            "p16_pdh_score6_rvol30",
            {
                "param_overrides.pdh_breakout_score_min": 6,
                "param_overrides.pdh_breakout_min_rvol": 3.0,
            },
        ),
        (
            "p16_pdh_cap005",
            {
                "param_overrides.pdh_avwap_cap_pct": 0.0050,
            },
        ),
        (
            "p16_pdh_end1100",
            {
                "param_overrides.pdh_entry_window_end": "11:00:00",
            },
        ),
        (
            "p16_pdh_size075",
            {
                "param_overrides.pdh_size_mult": 0.75,
            },
        ),
        (
            "p16_pdh_score6_cap005_end1100",
            {
                "param_overrides.pdh_breakout_score_min": 6,
                "param_overrides.pdh_avwap_cap_pct": 0.0050,
                "param_overrides.pdh_entry_window_end": "11:00:00",
            },
        ),
    ],
    4: [
        (
            "p16_score_shape_flat5",
            {
                "param_overrides.momentum_size_mult_score_3": 1.00,
                "param_overrides.momentum_size_mult_score_4": 1.10,
                "param_overrides.momentum_size_mult_score_5": 1.05,
                "param_overrides.momentum_size_mult_score_6": 1.20,
                "param_overrides.momentum_size_mult_score_7_plus": 1.22,
            },
        ),
        (
            "p16_score_shape_reward4",
            {
                "param_overrides.momentum_size_mult_score_3": 1.00,
                "param_overrides.momentum_size_mult_score_4": 1.15,
                "param_overrides.momentum_size_mult_score_5": 1.10,
                "param_overrides.momentum_size_mult_score_6": 1.20,
                "param_overrides.momentum_size_mult_score_7_plus": 1.22,
            },
        ),
        (
            "p16_score_shape_topheavy125",
            {
                "param_overrides.momentum_size_mult_score_3": 1.00,
                "param_overrides.momentum_size_mult_score_4": 1.05,
                "param_overrides.momentum_size_mult_score_5": 1.10,
                "param_overrides.momentum_size_mult_score_6": 1.22,
                "param_overrides.momentum_size_mult_score_7_plus": 1.25,
            },
        ),
        (
            "p16_score_shape_reward4_topheavy",
            {
                "param_overrides.momentum_size_mult_score_3": 1.00,
                "param_overrides.momentum_size_mult_score_4": 1.15,
                "param_overrides.momentum_size_mult_score_5": 1.10,
                "param_overrides.momentum_size_mult_score_6": 1.22,
                "param_overrides.momentum_size_mult_score_7_plus": 1.25,
            },
        ),
        (
            "p16_score_shape_flat5_topheavy",
            {
                "param_overrides.momentum_size_mult_score_3": 1.00,
                "param_overrides.momentum_size_mult_score_4": 1.10,
                "param_overrides.momentum_size_mult_score_5": 1.05,
                "param_overrides.momentum_size_mult_score_6": 1.22,
                "param_overrides.momentum_size_mult_score_7_plus": 1.25,
            },
        ),
        (
            "p16_score_shape_reward4_flat5",
            {
                "param_overrides.momentum_size_mult_score_3": 1.00,
                "param_overrides.momentum_size_mult_score_4": 1.15,
                "param_overrides.momentum_size_mult_score_5": 1.05,
                "param_overrides.momentum_size_mult_score_6": 1.20,
                "param_overrides.momentum_size_mult_score_7_plus": 1.25,
            },
        ),
    ],
    5: [
        (
            "p16_synth_avwap005_late1100score6",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0050,
                "param_overrides.late_entry_cutoff": "11:00:00",
                "param_overrides.late_entry_score_min": 6,
            },
        ),
        (
            "p16_synth_avwap005_pdhscore6_rvol25",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0050,
                "param_overrides.pdh_breakout_score_min": 6,
                "param_overrides.pdh_breakout_min_rvol": 2.5,
            },
        ),
        (
            "p16_synth_bar9score6_late1100score6",
            {
                "param_overrides.bar9_score_min": 6,
                "param_overrides.late_entry_cutoff": "11:00:00",
                "param_overrides.late_entry_score_min": 6,
            },
        ),
        (
            "p16_synth_pdhscore6_cap005_end1100",
            {
                "param_overrides.pdh_breakout_score_min": 6,
                "param_overrides.pdh_avwap_cap_pct": 0.0050,
                "param_overrides.pdh_entry_window_end": "11:00:00",
            },
        ),
        (
            "p16_synth_avwap005_late1100score6_pdhscore6",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0050,
                "param_overrides.late_entry_cutoff": "11:00:00",
                "param_overrides.late_entry_score_min": 6,
                "param_overrides.pdh_breakout_score_min": 6,
            },
        ),
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
