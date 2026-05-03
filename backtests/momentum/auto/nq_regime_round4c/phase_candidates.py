from __future__ import annotations

from typing import Any

BASE_MUTATIONS: dict[str, Any] = {
    "flags.enable_structural_expansion": True,
    "flags.enable_liquidity_reversion": True,
    "flags.enable_second_wind": True,
    "param_overrides.REGIME_MIN_CONFIDENCE": 0.65,
    "param_overrides.REGIME_MIN_MARGIN": 0.15,
    "param_overrides.STRUCTURAL_MIN_SCORE": 9,
    "param_overrides.STRUCTURAL_A_PLUS_SCORE": 10,
    "param_overrides.REVERSION_MIN_SCORE": 7,
    "param_overrides.REVERSION_A_SCORE": 8,
    "param_overrides.REVERSION_A_PLUS_SCORE": 11,
    "param_overrides.SECOND_WIND_MIN_SCORE": 7,
    "param_overrides.SECOND_WIND_A_SCORE": 8,
    "param_overrides.SECOND_WIND_A_PLUS_SCORE": 10,
    "param_overrides.TARGET_ROOM_MIN_R": 0.75,
    "param_overrides.REVERSION_STANDARD_STOP_CAP": 10.0,
    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 12.0,
    "param_overrides.MAX_TRADES_PER_DAY": 2,
    "param_overrides.MAX_FULL_RISK_TRADES": 2,
    "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -2.0,
    "param_overrides.PROFIT_FLOOR_ENABLED": True,
    "param_overrides.PROFIT_FLOOR_TRIGGER_R": 0.75,
    "param_overrides.PROFIT_FLOOR_LOCK_R": 0.25,
    "param_overrides.MFE_RATCHET_ENABLED": True,
    "param_overrides.MFE_RATCHET_TRIGGER_R": 1.0,
    "param_overrides.MFE_RATCHET_FLOOR_PCT": 0.65,
    "param_overrides.TIME_STOP_ENABLED": False,
    "param_overrides.REVERSION_ENTRY_MODEL": "swept_level_retest",
    "param_overrides.REVERSION_RETEST_OFFSET_TICKS": 1,
    "param_overrides.ENTRY_TTL_RETEST_MINUTES": 60,
    "param_overrides.ENTRY_TTL_MOMENTUM_MINUTES": 15,
    "param_overrides.ALLOW_LATE_PM_REVERSION": True,
    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "nq_3 Capacity And Routing Priority",
        [
            "module_liquidity_reversion_trades",
            "module_liquidity_reversion_total_r_per_month",
            "routing_liquidity_reversion_selected_to_fill_rate",
        ],
    ),
    2: (
        "nq_3 Entry Conversion",
        [
            "routing_liquidity_reversion_request_to_fill_rate",
            "module_liquidity_reversion_trades",
            "module_liquidity_reversion_avg_r",
        ],
    ),
    3: (
        "nq_3 Level And Session Expansion",
        [
            "module_liquidity_reversion_trades",
            "module_liquidity_reversion_total_r_per_month",
            "module_liquidity_reversion_profit_factor",
        ],
    ),
    4: (
        "nq_3 Discrimination And Stop Quality",
        [
            "module_liquidity_reversion_avg_r",
            "module_liquidity_reversion_profit_factor",
            "module_liquidity_reversion_positive_mfe_loser_rate",
        ],
    ),
    5: (
        "nq_3 Reversion Trade Management",
        [
            "module_liquidity_reversion_mfe_capture",
            "module_liquidity_reversion_avg_r",
            "module_liquidity_reversion_total_r_per_month",
        ],
    ),
    6: (
        "nq_3 Final Alpha/Frequency Stack",
        [
            "module_liquidity_reversion_total_r_per_month",
            "module_liquidity_reversion_trades",
            "max_drawdown_pct",
        ],
    ),
    7: (
        "nq_3 Exit Capture And Trade Management",
        [
            "module_liquidity_reversion_mfe_capture",
            "module_liquidity_reversion_positive_mfe_loser_rate",
            "module_liquidity_reversion_total_r_per_month",
        ],
    ),
}


def get_phase_candidates(phase: int, current_mutations: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    del current_mutations
    if phase == 1:
        return [
            (
                "nq3_only_capacity3",
                {
                    "flags.enable_structural_expansion": False,
                    "flags.enable_second_wind": False,
                    "param_overrides.MAX_TRADES_PER_DAY": 3,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                },
            ),
            (
                "nq3_priority_capacity3",
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 3,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
                },
            ),
            (
                "nq3_priority_disable_second_wind",
                {
                    "flags.enable_second_wind": False,
                    "param_overrides.MAX_TRADES_PER_DAY": 3,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
                },
            ),
            (
                "nq3_capacity4_guarded",
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 4,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -2.5,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
                },
            ),
            (
                "nq3_candidate_led_score7_room0p75",
                {
                    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 7,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
                },
            ),
            (
                "nq3_candidate_led_score8_room0p25",
                {
                    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.25,
                },
            ),
            (
                "nq3_candidate_led_score9_room0",
                {
                    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 9,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.0,
                },
            ),
        ]
    if phase == 2:
        return [
            (
                "nq3_retest_offset0_ttl90",
                {
                    "param_overrides.REVERSION_RETEST_OFFSET_TICKS": 0,
                    "param_overrides.ENTRY_TTL_RETEST_MINUTES": 90,
                },
            ),
            (
                "nq3_retest_offset2_ttl90",
                {
                    "param_overrides.REVERSION_RETEST_OFFSET_TICKS": 2,
                    "param_overrides.ENTRY_TTL_RETEST_MINUTES": 90,
                },
            ),
            (
                "nq3_retest_offset3_ttl120",
                {
                    "param_overrides.REVERSION_RETEST_OFFSET_TICKS": 3,
                    "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
                },
            ),
            (
                "nq3_reclaim_close_score8",
                {
                    "param_overrides.REVERSION_ENTRY_MODEL": "reclaim_close",
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                    "param_overrides.REVERSION_A_SCORE": 8,
                    "param_overrides.TARGET_ROOM_MIN_R": 1.0,
                },
            ),
            (
                "nq3_structure_shift_ttl20",
                {
                    "param_overrides.REVERSION_ENTRY_MODEL": "structure_shift",
                    "param_overrides.ENTRY_TTL_MOMENTUM_MINUTES": 20,
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                },
            ),
            (
                "nq3_adaptive_market_high_room",
                {
                    "param_overrides.REVERSION_ENTRY_MODEL": "adaptive_reclaim_retest",
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_SCORE": 11,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MAX_PENETRATION_PTS": 8.0,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R": 2.0,
                },
            ),
            (
                "nq3_adaptive_market_medium_room",
                {
                    "param_overrides.REVERSION_ENTRY_MODEL": "adaptive_reclaim_retest",
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_SCORE": 10,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MAX_PENETRATION_PTS": 6.0,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R": 1.5,
                },
            ),
        ]
    if phase == 3:
        return [
            (
                "nq3_swing_levels_conservative",
                {
                    "param_overrides.REVERSION_ENABLE_SWING_LEVELS": True,
                    "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 36,
                    "param_overrides.REVERSION_SWING_RADIUS": 2,
                    "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 4,
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                },
            ),
            (
                "nq3_swing_levels_broad",
                {
                    "param_overrides.REVERSION_ENABLE_SWING_LEVELS": True,
                    "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 48,
                    "param_overrides.REVERSION_SWING_RADIUS": 2,
                    "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 6,
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                },
            ),
            (
                "nq3_early_sweep_score8",
                {
                    "param_overrides.REVERSION_ALLOW_EARLY_SWEEP": True,
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                },
            ),
            (
                "nq3_shallow_sweep_with_value",
                {
                    "param_overrides.REVERSION_MIN_PENETRATION_PTS": 1.0,
                    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 10.0,
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                    "param_overrides.TARGET_ROOM_MIN_R": 1.0,
                },
            ),
            (
                "nq3_deeper_sweep_room0p75",
                {
                    "param_overrides.REVERSION_MIN_PENETRATION_PTS": 3.0,
                    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 14.0,
                    "param_overrides.TARGET_ROOM_MIN_R": 0.75,
                },
            ),
            (
                "nq3_room0p5_score8",
                {
                    "param_overrides.TARGET_ROOM_MIN_R": 0.50,
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                    "param_overrides.REVERSION_A_SCORE": 8,
                },
            ),
            (
                "nq3_value_factor_score7",
                {
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 2,
                    "param_overrides.REVERSION_MIN_SCORE": 7,
                    "param_overrides.TARGET_ROOM_MIN_R": 0.50,
                },
            ),
        ]
    if phase == 4:
        return [
            (
                "nq3_score8_room0p75",
                {
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                    "param_overrides.REVERSION_A_SCORE": 8,
                    "param_overrides.TARGET_ROOM_MIN_R": 0.75,
                },
            ),
            (
                "nq3_score9_room0p5",
                {
                    "param_overrides.REVERSION_MIN_SCORE": 9,
                    "param_overrides.REVERSION_A_SCORE": 9,
                    "param_overrides.TARGET_ROOM_MIN_R": 0.50,
                },
            ),
            (
                "nq3_a_plus10_stop10",
                {
                    "param_overrides.REVERSION_A_PLUS_SCORE": 10,
                    "param_overrides.REVERSION_STANDARD_STOP_CAP": 10.0,
                    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 12.0,
                },
            ),
            (
                "nq3_tight_stop8_10",
                {
                    "param_overrides.REVERSION_STANDARD_STOP_CAP": 8.0,
                    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 10.0,
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                },
            ),
            (
                "nq3_wide_stop12_score9",
                {
                    "param_overrides.REVERSION_STANDARD_STOP_CAP": 12.0,
                    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 14.0,
                    "param_overrides.REVERSION_MIN_SCORE": 9,
                    "param_overrides.REVERSION_A_SCORE": 9,
                },
            ),
            (
                "nq3_value_factor1_penetration_cap10",
                {
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 10.0,
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                },
            ),
            (
                "nq3_block_overextension",
                {
                    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 8.0,
                    "param_overrides.REVERSION_STANDARD_STOP_CAP": 9.0,
                    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 11.0,
                },
            ),
        ]
    if phase == 5:
        return [
            (
                "nq3_floor_0p75_lock0p25",
                {
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
                },
            ),
            (
                "nq3_floor_1p0_lock0p5",
                {
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 1.0,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.5,
                },
            ),
            (
                "nq3_ratchet_1p0_50pct",
                {
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.0,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.50,
                },
            ),
            (
                "nq3_ratchet_1p5_55pct",
                {
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.5,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.55,
                },
            ),
            (
                "nq3_vwap_touch_exit",
                {
                    "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": True,
                    "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": False,
                },
            ),
            (
                "nq3_vwap_reaction_floor",
                {
                    "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
                },
            ),
            (
                "nq3_runner_heavier_targets",
                {
                    "param_overrides.REVERSION_TARGET1_QTY_FRACTION": 0.30,
                    "param_overrides.REVERSION_TARGET2_QTY_FRACTION": 0.25,
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.5,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.50,
                },
            ),
        ]
    if phase == 6:
        return [
            (
                "nq3_quality_frequency_stack",
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 3,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                },
            ),
            (
                "nq3_adaptive_capacity_stack",
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 4,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.REVERSION_ENTRY_MODEL": "adaptive_reclaim_retest",
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_SCORE": 10,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MAX_PENETRATION_PTS": 6.0,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R": 1.5,
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                },
            ),
            (
                "nq3_swing_adaptive_stack",
                {
                    "param_overrides.REVERSION_ENABLE_SWING_LEVELS": True,
                    "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 48,
                    "param_overrides.REVERSION_SWING_RADIUS": 2,
                    "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 6,
                    "param_overrides.REVERSION_ENTRY_MODEL": "adaptive_reclaim_retest",
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_SCORE": 11,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R": 2.0,
                },
            ),
            (
                "nq3_discriminatory_runner_stack",
                {
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                    "param_overrides.REVERSION_A_SCORE": 8,
                    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 10.0,
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.5,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.55,
                    "param_overrides.REVERSION_TARGET1_QTY_FRACTION": 0.30,
                    "param_overrides.REVERSION_TARGET2_QTY_FRACTION": 0.25,
                },
            ),
            (
                "nq3_reclaim_close_capacity3",
                {
                    "param_overrides.REVERSION_ENTRY_MODEL": "reclaim_close",
                    "param_overrides.MAX_TRADES_PER_DAY": 3,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.REVERSION_MIN_SCORE": 8,
                    "param_overrides.TARGET_ROOM_MIN_R": 1.0,
                },
            ),
            (
                "nq3_only_final_stack",
                {
                    "flags.enable_structural_expansion": False,
                    "flags.enable_second_wind": False,
                    "param_overrides.MAX_TRADES_PER_DAY": 4,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
                    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
                },
            ),
            (
                "nq3_conservative_final_stack",
                {
                    "param_overrides.REVERSION_MIN_SCORE": 9,
                    "param_overrides.REVERSION_A_SCORE": 9,
                    "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                    "param_overrides.REVERSION_STANDARD_STOP_CAP": 10.0,
                    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 12.0,
                    "param_overrides.MAX_TRADES_PER_DAY": 3,
                    "param_overrides.MAX_FULL_RISK_TRADES": 2,
                },
            ),
        ]
    if phase == 7:
        return [
            (
                "nq3_capture_floor_0p5_lock0p25",
                {
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.50,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
                },
            ),
            (
                "nq3_capture_floor_0p75_lock0p5",
                {
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.50,
                },
            ),
            (
                "nq3_capture_ratchet_0p75_50pct",
                {
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 0.75,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.50,
                },
            ),
            (
                "nq3_capture_ratchet_1p0_65pct",
                {
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.00,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.65,
                },
            ),
            (
                "nq3_capture_ratchet_1p5_70pct",
                {
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.50,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.70,
                },
            ),
            (
                "nq3_capture_delayed_vwap_reaction",
                {
                    "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": False,
                    "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.25,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.55,
                },
            ),
            (
                "nq3_capture_no_vwap_flatten_ratchet",
                {
                    "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": False,
                    "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": False,
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 1.00,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.50,
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.25,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.65,
                },
            ),
            (
                "nq3_capture_touch_plus_runner_ratchet",
                {
                    "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": True,
                    "param_overrides.REVERSION_TARGET1_QTY_FRACTION": 0.30,
                    "param_overrides.REVERSION_TARGET2_QTY_FRACTION": 0.20,
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.25,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.55,
                },
            ),
            (
                "nq3_capture_de_risked_t1_floor",
                {
                    "param_overrides.REVERSION_TARGET1_QTY_FRACTION": 0.50,
                    "param_overrides.REVERSION_TARGET2_QTY_FRACTION": 0.25,
                    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
                    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
                    "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                    "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.00,
                    "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.50,
                },
            ),
            (
                "nq3_capture_stall_exit_6bars_0p5",
                {
                    "param_overrides.REVERSION_TIME_STOP_ENABLED": True,
                    "param_overrides.REVERSION_TIME_STOP_BARS": 6,
                    "param_overrides.REVERSION_TIME_STOP_MIN_MFE_R": 0.50,
                },
            ),
            (
                "nq3_capture_stall_exit_8bars_0p75",
                {
                    "param_overrides.REVERSION_TIME_STOP_ENABLED": True,
                    "param_overrides.REVERSION_TIME_STOP_BARS": 8,
                    "param_overrides.REVERSION_TIME_STOP_MIN_MFE_R": 0.75,
                },
            ),
            (
                "nq3_capture_stall_exit_with_reaction",
                {
                    "param_overrides.REVERSION_TIME_STOP_ENABLED": True,
                    "param_overrides.REVERSION_TIME_STOP_BARS": 8,
                    "param_overrides.REVERSION_TIME_STOP_MIN_MFE_R": 0.50,
                    "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": False,
                    "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": True,
                },
            ),
        ]
    return []
