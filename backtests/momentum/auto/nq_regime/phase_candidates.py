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
    "param_overrides.SECOND_WIND_STOP_CAP": 55.0,
    "param_overrides.SECOND_WIND_ENTRY_STOP_INVALIDATION_ENABLED": True,
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
    "param_overrides.STRUCTURAL_ENTRY_MODE": "retest",
    "param_overrides.REVERSION_ENTRY_MODEL": "swept_level_retest",
    "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_midpoint",
    "param_overrides.ENTRY_TTL_RETEST_MINUTES": 60,
    "param_overrides.ENTRY_TTL_MOMENTUM_MINUTES": 15,
    "param_overrides.ALLOW_LATE_PM_REVERSION": True,
    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "nq_1 Integrity And Discrimination",
        [
            "module_second_wind_avg_r",
            "module_second_wind_profit_factor",
            "module_second_wind_top_trade_share",
        ],
    ),
    2: (
        "nq_1 Continuation Signal Expansion",
        [
            "module_second_wind_trades",
            "module_second_wind_total_r_per_month",
            "module_second_wind_avg_r",
        ],
    ),
    3: (
        "nq_1 Entry And Fill Conversion",
        [
            "routing_second_wind_request_to_fill_rate",
            "module_second_wind_trades",
            "execution_conversion",
        ],
    ),
    4: (
        "nq_1 Routing And Session Activation",
        [
            "routing_second_wind_select_rate_when_valid",
            "routing_second_wind_selected_to_fill_rate",
            "module_second_wind_trades",
        ],
    ),
    5: (
        "nq_1 Trade Management",
        [
            "module_second_wind_mfe_capture",
            "module_second_wind_avg_r",
            "module_second_wind_positive_mfe_loser_rate",
        ],
    ),
    6: (
        "nq_1 Alpha/Frequency Stack",
        [
            "module_second_wind_total_r_per_month",
            "module_second_wind_trades",
            "max_drawdown_pct",
        ],
    ),
    7: (
        "nq_1 Weak Subfamily Selectivity",
        [
            "module_second_wind_pm_vwap_reclaim_avg_r",
            "module_second_wind_pm_vwap_reclaim_profit_factor",
            "module_second_wind_pm_second_leg_avg_r",
            "module_second_wind_pm_second_leg_profit_factor",
            "module_second_wind_trades",
            "module_second_wind_total_r_per_month",
        ],
    ),
}


def get_phase_candidates(phase: int, current_mutations: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    del current_mutations
    if phase == 1:
        return [
            ("nq1_disable_global_candidate_led", {
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
            }),
            ("nq1_module_led_replace_global", {
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_score8_only", {
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
            }),
            ("nq1_score8_volume1p2", {
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 1.2,
            }),
            ("nq1_stop_cap30_score8", {
                "param_overrides.SECOND_WIND_MAX_STOP_PTS": 30.0,
                "param_overrides.SECOND_WIND_STOP_CAP": 30.0,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
            }),
            ("nq1_stop_cap25_room3", {
                "param_overrides.SECOND_WIND_MAX_STOP_PTS": 25.0,
                "param_overrides.SECOND_WIND_STOP_CAP": 25.0,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.TARGET_ROOM_MIN_R": 3.0,
            }),
            ("nq1_pm_score055", {
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_cutoff_1500", {
                "param_overrides.SECOND_WIND_MAX_ENTRY_MINUTE_ET": 15 * 60,
            }),
            ("nq1_score9_strict", {
                "param_overrides.SECOND_WIND_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_A_SCORE": 9,
            }),
        ]
    if phase == 2:
        return [
            ("nq1_vwap_reclaim_score8", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.50,
            }),
            ("nq1_micro_compression_score8", {
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.50,
            }),
            ("nq1_range_acceptance_score8", {
                "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.50,
            }),
            ("nq1_second_leg_score8", {
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.50,
            }),
            ("nq1_vwap_micro_combo", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_all_family_score8_stop30", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MAX_STOP_PTS": 30.0,
                "param_overrides.SECOND_WIND_STOP_CAP": 30.0,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_all_family_score9", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_A_SCORE": 9,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.55,
            }),
        ]
    if phase == 3:
        return [
            ("nq1_trigger_close", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_close",
            }),
            ("nq1_breakout_stop", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "breakout_stop",
                "param_overrides.ENTRY_TTL_MOMENTUM_MINUTES": 20,
            }),
            ("nq1_ema_pullback_ttl90", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "ema_pullback",
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 90,
            }),
            ("nq1_midpoint_ttl120", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_midpoint",
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
            }),
            ("nq1_range_retest_ttl90", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "range_retest",
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 90,
            }),
            ("nq1_market_family_tight_stop", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_close",
                "param_overrides.SECOND_WIND_MAX_STOP_PTS": 30.0,
                "param_overrides.SECOND_WIND_STOP_CAP": 30.0,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
            }),
            ("nq1_pullback_family_ttl120", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "ema_pullback",
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
            }),
        ]
    if phase == 4:
        return [
            ("nq1_candidate_led_score8_room3_pm055", {
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_clean_candidate_led_score8_room3", {
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_candidate_led_score8_room1p5_pm055", {
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_candidate_led_score9_room1_pm05", {
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.50,
            }),
            ("nq1_candidate_led_score7_room5_pm06", {
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 7,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 5.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.60,
            }),
            ("nq1_capacity3_candidate_led", {
                "param_overrides.MAX_TRADES_PER_DAY": 3,
                "param_overrides.MAX_FULL_RISK_TRADES": 3,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
            }),
            ("nq1_pm_window_1315", {
                "param_overrides.SECOND_WIND_MIN_ENTRY_MINUTE_ET": 13 * 60 + 15,
            }),
            ("nq1_no_late_restricted", {
                "param_overrides.SECOND_WIND_MAX_ENTRY_MINUTE_ET": 14 * 60 + 45,
            }),
        ]
    if phase == 5:
        return [
            ("nq1_floor_1p0_lock0p5", {
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R": 1.00,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_LOCK_R": 0.50,
            }),
            ("nq1_floor_0p75_lock0p25", {
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R": 0.75,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_LOCK_R": 0.25,
            }),
            ("nq1_ratchet_1p5_50pct", {
                "param_overrides.SECOND_WIND_MFE_RATCHET_ENABLED": True,
                "param_overrides.SECOND_WIND_MFE_RATCHET_TRIGGER_R": 1.50,
                "param_overrides.SECOND_WIND_MFE_RATCHET_FLOOR_PCT": 0.50,
            }),
            ("nq1_ratchet_2p0_55pct", {
                "param_overrides.SECOND_WIND_MFE_RATCHET_ENABLED": True,
                "param_overrides.SECOND_WIND_MFE_RATCHET_TRIGGER_R": 2.00,
                "param_overrides.SECOND_WIND_MFE_RATCHET_FLOOR_PCT": 0.55,
            }),
            ("nq1_ema_trail_after_partial", {
                "param_overrides.SECOND_WIND_EMA_TRAIL_EXIT_ENABLED": True,
                "param_overrides.SECOND_WIND_EMA_TRAIL_REQUIRES_PARTIAL": True,
            }),
            ("nq1_ema_trail_immediate", {
                "param_overrides.SECOND_WIND_EMA_TRAIL_EXIT_ENABLED": True,
                "param_overrides.SECOND_WIND_EMA_TRAIL_REQUIRES_PARTIAL": False,
            }),
            ("nq1_time_stop_4bars", {
                "param_overrides.SECOND_WIND_TIME_STOP_ENABLED": True,
                "param_overrides.SECOND_WIND_TIME_STOP_BARS": 4,
                "param_overrides.SECOND_WIND_TIME_STOP_MIN_MFE_R": 0.50,
            }),
        ]
    if phase == 6:
        return [
            ("nq1_vwap_reclaim_route_floor", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R": 1.00,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_LOCK_R": 0.50,
            }),
            ("nq1_micro_route_ratchet", {
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
                "param_overrides.SECOND_WIND_MFE_RATCHET_ENABLED": True,
                "param_overrides.SECOND_WIND_MFE_RATCHET_TRIGGER_R": 1.50,
                "param_overrides.SECOND_WIND_MFE_RATCHET_FLOOR_PCT": 0.50,
            }),
            ("nq1_all_family_strict_management", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_A_SCORE": 8,
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.55,
                "param_overrides.SECOND_WIND_MAX_STOP_PTS": 30.0,
                "param_overrides.SECOND_WIND_STOP_CAP": 30.0,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R": 1.00,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_LOCK_R": 0.50,
            }),
            ("nq1_frequency_stack_capacity4", {
                "param_overrides.MAX_TRADES_PER_DAY": 4,
                "param_overrides.MAX_FULL_RISK_TRADES": 4,
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
            }),
            ("nq1_high_quality_frequency", {
                "param_overrides.MAX_TRADES_PER_DAY": 3,
                "param_overrides.MAX_FULL_RISK_TRADES": 3,
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
                "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 1.1,
            }),
            ("nq1_close_entry_route_stack", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_close",
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
            }),
            ("nq1_conservative_stack", {
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_MAX_STOP_PTS": 25.0,
                "param_overrides.SECOND_WIND_STOP_CAP": 25.0,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R": 1.00,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_LOCK_R": 0.50,
            }),
        ]
    if phase == 7:
        return [
            ("nq1_vwap_reclaim_disable", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": False,
            }),
            ("nq1_second_leg_disable", {
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": False,
            }),
            ("nq1_keep_range_micro_prune_weak_families", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": False,
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": False,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
            }),
            ("nq1_vwap_reclaim_quality_gate", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE": 1.30,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR": 0.08,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT": True,
            }),
            ("nq1_vwap_reclaim_transition_only", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE": 0.65,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE": 1.40,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION": 0.75,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR": 0.12,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_REQUIRE_PM_TRANSITION": True,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT": True,
            }),
            ("nq1_vwap_reclaim_no_chase_quality", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE": 1.20,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION": 0.67,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR": 0.05,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MAX_RECLAIM_ATR": 0.65,
            }),
            ("nq1_second_leg_quality_gate", {
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE": 1.20,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_ATR": 0.05,
                "param_overrides.SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE": True,
            }),
            ("nq1_second_leg_decisive_break", {
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_SCORE": 10,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_PM_SCORE": 0.65,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE": 1.40,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION": 0.75,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_ATR": 0.10,
                "param_overrides.SECOND_WIND_SECOND_LEG_REQUIRE_PM_TRANSITION": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_PULLBACK_BUFFER_ATR_MULT": 0.25,
            }),
            ("nq1_selective_vwap_off_second_leg_strict", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": False,
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE": 1.20,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_ATR": 0.05,
                "param_overrides.SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE": True,
            }),
            ("nq1_selective_vwap_strict_second_leg_off", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": False,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE": 1.30,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR": 0.08,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT": True,
            }),
            ("nq1_weak_subfamilies_strict_combo", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE": 1.30,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR": 0.08,
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE": 1.20,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_ATR": 0.05,
                "param_overrides.SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE": True,
            }),
        ]
    return []
