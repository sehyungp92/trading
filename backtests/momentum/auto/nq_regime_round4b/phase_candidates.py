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
    "param_overrides.TARGET_ROOM_MIN_R": 0.75,
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
    "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 1,
    "param_overrides.STRUCTURAL_MIN_BODY_PCT": 0.55,
    "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.75,
    "param_overrides.STRUCTURAL_MAX_STOP_PTS": 999.0,
    "param_overrides.STRUCTURAL_BLOCK_NORMAL_IB": False,
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
        "nq_2 Structural Fill Causality And Conversion",
        [
            "module_structural_expansion_trades",
            "routing_structural_expansion_selected",
            "module_structural_expansion_avg_r",
        ],
    ),
    2: (
        "nq_2 Sample-Floor Standalone Alpha",
        [
            "module_structural_expansion_trades",
            "module_structural_expansion_total_r",
            "module_structural_expansion_avg_r",
        ],
    ),
    3: (
        "nq_2 Signal Discrimination",
        [
            "module_structural_expansion_avg_r",
            "module_structural_expansion_profit_factor",
            "module_structural_expansion_trades",
        ],
    ),
    4: (
        "nq_2 Reproducible Edge Validation",
        [
            "module_structural_expansion_trades",
            "module_structural_expansion_total_r",
            "module_structural_expansion_profit_factor",
        ],
    ),
    5: (
        "nq_2 Trade Management And Leakage",
        [
            "module_structural_expansion_mfe_capture",
            "module_structural_expansion_positive_mfe_loser_rate",
            "module_structural_expansion_avg_r",
        ],
    ),
    6: (
        "nq_2 Final Alpha Guardrails",
        [
            "module_structural_expansion_trades",
            "module_structural_expansion_total_r",
            "module_structural_expansion_avg_r",
        ],
    ),
}


def _merge(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


RETEST_TTL90 = {
    "param_overrides.ENTRY_TTL_RETEST_MINUTES": 90,
    "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 2,
}

RECENT_STOP_QUALITY = {
    "param_overrides.STRUCTURAL_STOP_MODEL": "recent_5m",
    "param_overrides.STRUCTURAL_MIN_STOP_PTS": 10.0,
}

PULLBACK_RECLAIM_SCORE8 = {
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENABLED": True,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENTRY_MODE": "close",
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MAX_AGE_MINUTES": 240,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R": 1.0,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_VOLUME_MULTIPLE": 0.8,
}

QUALITY_SIDE_FLOORS = {
    "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 10,
    "param_overrides.STRUCTURAL_LONG_MIN_SCORE": 8,
}

HYBRID_SCORE8_SAMPLE_EDGE = _merge(
    RETEST_TTL90,
    RECENT_STOP_QUALITY,
    PULLBACK_RECLAIM_SCORE8,
    QUALITY_SIDE_FLOORS,
    {
        "param_overrides.STRUCTURAL_ENTRY_MODE": "hybrid_close_adaptive",
        "param_overrides.STRUCTURAL_MIN_SCORE": 8,
        "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.65,
        "param_overrides.STRUCTURAL_HYBRID_CLOSE_MIN_SCORE": 8,
        "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 50.0,
        "param_overrides.TARGET_ROOM_MIN_R": 0.50,
        "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
        "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
        "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
        "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 1,
        "param_overrides.STRUCTURAL_MAX_STOP_PTS": 50.0,
        "param_overrides.STRUCTURAL_ALLOW_MIN_MICRO_SIZE": True,
        "param_overrides.STRUCTURAL_MIN_MICRO_MAX_RISK_PCT": 0.01,
        "param_overrides.MAX_TRADES_PER_DAY": 4,
        "param_overrides.MAX_FULL_RISK_TRADES": 3,
    },
)

ADAPTIVE_RETEST_SCORE8 = _merge(
    RETEST_TTL90,
    RECENT_STOP_QUALITY,
    PULLBACK_RECLAIM_SCORE8,
    QUALITY_SIDE_FLOORS,
    {
        "param_overrides.STRUCTURAL_ENTRY_MODE": "adaptive_retest",
        "param_overrides.STRUCTURAL_ADAPTIVE_RETEST_PREFERS_FVG": True,
        "param_overrides.STRUCTURAL_MIDPOINT_RETEST_ENABLED": True,
        "param_overrides.STRUCTURAL_MIN_SCORE": 8,
        "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
        "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 1,
        "param_overrides.STRUCTURAL_MAX_STOP_PTS": 35.0,
        "param_overrides.MAX_TRADES_PER_DAY": 4,
        "param_overrides.MAX_FULL_RISK_TRADES": 3,
    },
)


def get_phase_candidates(phase: int, current_mutations: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    del current_mutations
    if phase == 1:
        return [
            ("struct_retest_ttl45_offset0", {
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 45,
                "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 0,
            }),
            ("struct_retest_ttl90_offset2", RETEST_TTL90),
            ("struct_retest_ttl120_offset1", {
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
                "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 1,
            }),
            ("struct_recent_stop_ttl90_min10", _merge(RETEST_TTL90, RECENT_STOP_QUALITY)),
            ("struct_recent_stop_ttl120_min10", _merge(RECENT_STOP_QUALITY, {
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
                "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 1,
            })),
            ("struct_capacity_3_full2", {
                "param_overrides.MAX_TRADES_PER_DAY": 3,
                "param_overrides.MAX_FULL_RISK_TRADES": 2,
            }),
            ("struct_capacity_4_full3", {
                "param_overrides.MAX_TRADES_PER_DAY": 4,
                "param_overrides.MAX_FULL_RISK_TRADES": 3,
            }),
        ]
    if phase == 2:
        return [
            ("struct_hybrid_score8_sample_edge", HYBRID_SCORE8_SAMPLE_EDGE),
            ("struct_hybrid_score8_close70", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.70,
            })),
            ("struct_hybrid_score8_stop45", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 45.0,
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 45.0,
            })),
            ("struct_hybrid_score8_room075", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.TARGET_ROOM_MIN_R": 0.75,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
            })),
            ("struct_adaptive_retest_score8_sample", ADAPTIVE_RETEST_SCORE8),
            ("struct_fvg_retest_score8_quality", _merge(ADAPTIVE_RETEST_SCORE8, {
                "param_overrides.STRUCTURAL_ENTRY_MODE": "fvg_retest",
                "param_overrides.STRUCTURAL_FVG_RETEST_ENABLED": True,
                "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 0,
            })),
            ("struct_pullback_reclaim_score8_quality", _merge(RETEST_TTL90, RECENT_STOP_QUALITY, PULLBACK_RECLAIM_SCORE8, {
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 25.0,
                "param_overrides.MAX_TRADES_PER_DAY": 4,
                "param_overrides.MAX_FULL_RISK_TRADES": 3,
            })),
            ("struct_retest_score8_quality", _merge(RETEST_TTL90, RECENT_STOP_QUALITY, QUALITY_SIDE_FLOORS, {
                "param_overrides.STRUCTURAL_MIN_SCORE": 8,
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 35.0,
                "param_overrides.TARGET_ROOM_MIN_R": 0.50,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
                "param_overrides.MAX_TRADES_PER_DAY": 4,
                "param_overrides.MAX_FULL_RISK_TRADES": 3,
            })),
        ]
    if phase == 3:
        return [
            ("struct_score9_quality", {
                "param_overrides.STRUCTURAL_MIN_SCORE": 9,
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MIN_SCORE": 9,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
            }),
            ("struct_body60_close70", {
                "param_overrides.STRUCTURAL_MIN_BODY_PCT": 0.60,
                "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.70,
            }),
            ("struct_body65_close75", {
                "param_overrides.STRUCTURAL_MIN_BODY_PCT": 0.65,
                "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.75,
            }),
            ("struct_target_room075", {
                "param_overrides.TARGET_ROOM_MIN_R": 0.75,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
            }),
            ("struct_short_score11", {
                "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 11,
                "param_overrides.STRUCTURAL_LONG_MIN_SCORE": 8,
            }),
            ("struct_stop_cap35", {
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 35.0,
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 35.0,
            }),
            ("struct_pullback_require_trend", {
                "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_REQUIRE_TREND": True,
                "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R": 1.25,
            }),
            ("struct_after_1045", {"param_overrides.STRUCTURAL_MIN_ENTRY_MINUTE_ET": 10 * 60 + 45}),
        ]
    if phase == 4:
        return [
            ("struct_hybrid_score8_sample_edge_retest", HYBRID_SCORE8_SAMPLE_EDGE),
            ("struct_hybrid_score8_sample_edge_floor", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.20,
            })),
            ("struct_hybrid_score8_sample_edge_fast", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.STRUCTURAL_FAST_FAILURE_EXIT_ENABLED": True,
            })),
            ("struct_hybrid_score8_close70_floor", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.20,
            })),
            ("struct_adaptive_retest_score8_sample_retest", ADAPTIVE_RETEST_SCORE8),
            ("struct_hybrid_score8_stop45_floor", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 45.0,
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 45.0,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.20,
            })),
            ("struct_hybrid_score8_minmicro_halfpct", _merge(HYBRID_SCORE8_SAMPLE_EDGE, {
                "param_overrides.STRUCTURAL_MIN_MICRO_MAX_RISK_PCT": 0.005,
            })),
        ]
    if phase == 5:
        return [
            ("struct_fast_failure_exit", {"param_overrides.STRUCTURAL_FAST_FAILURE_EXIT_ENABLED": True}),
            ("struct_floor_0p50_lock0p10", {
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.10,
            }),
            ("struct_floor_0p50_lock0p20", {
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.20,
            }),
            ("struct_floor_0p75_lock0p25", {
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.75,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.25,
            }),
            ("struct_ratchet_1p0_0p70", {
                "param_overrides.STRUCTURAL_MFE_RATCHET_ENABLED": True,
                "param_overrides.STRUCTURAL_MFE_RATCHET_TRIGGER_R": 1.00,
                "param_overrides.STRUCTURAL_MFE_RATCHET_FLOOR_PCT": 0.70,
            }),
            ("struct_target_frac_60_20", {
                "param_overrides.STRUCTURAL_TARGET1_QTY_FRACTION": 0.60,
                "param_overrides.STRUCTURAL_TARGET2_QTY_FRACTION": 0.20,
            }),
            ("struct_wide_t1_0p35", {
                "param_overrides.STRUCTURAL_WIDE_TARGET1_R": 0.35,
                "param_overrides.STRUCTURAL_WIDE_TARGET2_R": 0.80,
                "param_overrides.STRUCTURAL_WIDE_TARGET3_R": 1.50,
            }),
        ]
    if phase == 6:
        return [
            ("struct_final_quality_score9", {
                "param_overrides.STRUCTURAL_MIN_SCORE": 9,
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MIN_SCORE": 9,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
            }),
            ("struct_final_quality_close70", {
                "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.70,
            }),
            ("struct_final_quality_stop45", {
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 45.0,
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 45.0,
            }),
            ("struct_final_quality_room075", {
                "param_overrides.TARGET_ROOM_MIN_R": 0.75,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
            }),
            ("struct_final_quality_daycap3", {
                "param_overrides.MAX_TRADES_PER_DAY": 3,
                "param_overrides.MAX_FULL_RISK_TRADES": 2,
            }),
            ("struct_final_quality_no_minmicro", {
                "param_overrides.STRUCTURAL_ALLOW_MIN_MICRO_SIZE": False,
            }),
            ("struct_final_quality_floor", {
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.20,
            }),
        ]
    return []
