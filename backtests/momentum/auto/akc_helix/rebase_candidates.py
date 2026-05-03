"""Candidate pools for Helix rebase frequency expansion.

Round 4 starts from the latest optimized rebase config and targets the main
remaining weakness: too few trades.  The structural candidates stay inside the
existing trend-continuation theme and use shared signal logic consumed by both
the live and backtest adapters.
"""
from __future__ import annotations

from backtests.shared.auto.types import Experiment

REBASE_BASE_MUTATIONS: dict[str, object] = {}

REBASE_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("DIRECTIONAL M5 CONTINUATION BRIDGE", ["total_trades", "net_profit", "profit_factor"]),
    2: ("COMPLETED M15 LONG CONTINUATION", ["total_trades", "net_profit", "expectancy_dollar"]),
    3: ("COMPLETED M15 SHORT AND TWO-SIDED CONTINUATION", ["total_trades", "net_profit", "profit_factor"]),
    4: ("STRUCTURAL STACKS WITHOUT BROAD GATE LOOSENING", ["total_trades", "net_profit", "profit_factor"]),
    5: ("COHORT-SAFE MANAGEMENT AND STOP CALIBRATION", ["net_profit", "calmar", "expectancy_dollar"]),
    6: ("RISK SIZING AFTER FREQUENCY IS PROVEN", ["net_profit", "calmar", "max_drawdown_pct"]),
    7: ("FINAL QUALITY-FREQUENCY INTEGRATION", ["net_profit", "total_trades", "profit_factor"]),
}


_FOLLOWTHROUGH = {
    "param_overrides.FOLLOWTHROUGH_CATCHUP_ENABLED": True,
    "param_overrides.FOLLOWTHROUGH_ALLOW_CLASS_M": True,
    "param_overrides.FOLLOWTHROUGH_ALLOW_CLASS_T": True,
    "param_overrides.FOLLOWTHROUGH_ALLOW_RTH_PRIME1": True,
    "param_overrides.FOLLOWTHROUGH_ALLOW_RTH_PRIME2": False,
    "param_overrides.FOLLOWTHROUGH_ALLOW_ETH_QUALITY_PM": True,
    "param_overrides.FOLLOWTHROUGH_MIN_ALIGNMENT_SCORE": 1,
    "param_overrides.FOLLOWTHROUGH_MAX_GAP_ATR": 0.25,
}

_FAILED_RECLAIM_SHORT = {
    "param_overrides.FAILED_RECLAIM_SHORT_ENABLED": True,
    "param_overrides.FAILED_RECLAIM_SHORT_ALLOW_RTH_DEAD": False,
    "param_overrides.FAILED_RECLAIM_SHORT_ALLOW_RTH_PRIME2": True,
    "param_overrides.FAILED_RECLAIM_SHORT_ALLOW_ETH_QUALITY_PM": True,
    "param_overrides.FAILED_RECLAIM_SHORT_MIN_VOL_PCT": 45,
    "param_overrides.FAILED_RECLAIM_SHORT_MAX_VOL_PCT": 92,
    "param_overrides.FAILED_RECLAIM_SHORT_MIN_ALIGNMENT_SCORE": 0,
    "param_overrides.FAILED_RECLAIM_SHORT_MIN_CONTEXT_SCORE": 2,
    "param_overrides.FAILED_RECLAIM_SHORT_REQUIRE_STRONG_TREND": False,
}

_H1_RETEST = {
    "param_overrides.TREND_RETEST_ENABLED": True,
    "param_overrides.TREND_RETEST_ALLOW_LONG": True,
    "param_overrides.TREND_RETEST_ALLOW_SHORT": False,
    "param_overrides.TREND_RETEST_ALLOW_RTH_PRIME1": True,
    "param_overrides.TREND_RETEST_ALLOW_RTH_PRIME2": False,
    "param_overrides.TREND_RETEST_ALLOW_RTH_DEAD": True,
    "param_overrides.TREND_RETEST_ALLOW_ETH_QUALITY_PM": False,
    "param_overrides.TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG": 1,
    "param_overrides.TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 2,
    "param_overrides.TREND_RETEST_REQUIRE_STRONG_TREND": False,
    "param_overrides.TREND_RETEST_MIN_TREND_STRENGTH": 0.30,
    "param_overrides.TREND_RETEST_LOOKBACK_BARS": 4,
    "param_overrides.TREND_RETEST_MAX_BREAKOUT_GAP_ATR": 0.45,
}

_MICRO_FAST = {
    "param_overrides.MICRO_TREND_RETEST_ENABLED": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_LONG": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_SHORT": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_RTH_PRIME1": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_RTH_PRIME2": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_RTH_DEAD": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_AM": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
    "param_overrides.MICRO_TREND_RETEST_MIN_VOL_PCT": 20,
    "param_overrides.MICRO_TREND_RETEST_MAX_VOL_PCT": 80,
    "param_overrides.MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG": 1,
    "param_overrides.MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT": 1,
    "param_overrides.MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 2,
    "param_overrides.MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT": 2,
    "param_overrides.MICRO_TREND_RETEST_REQUIRE_STRONG_TREND": False,
    "param_overrides.MICRO_TREND_RETEST_MIN_TREND_STRENGTH": 0.10,
    "param_overrides.MICRO_TREND_RETEST_LOOKBACK_BARS": 4,
    "param_overrides.MICRO_TREND_RETEST_MIN_BARS_BETWEEN": 3,
    "param_overrides.MICRO_TREND_RETEST_PULLBACK_EMA_TOL_ATR": 0.60,
    "param_overrides.MICRO_TREND_RETEST_MAX_PULLBACK_ATR": 3.00,
    "param_overrides.MICRO_TREND_RETEST_MAX_BREAKOUT_GAP_ATR": 1.10,
}

_MICRO_DIRECTIONAL_CLEAN = {
    **_MICRO_FAST,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME1": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME2": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_LONG_RTH_DEAD": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_AM": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_PM": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME1": True,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME2": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_SHORT_RTH_DEAD": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_AM": False,
    "param_overrides.MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_PM": True,
}

_M15_LONG = {
    "param_overrides.M15_TREND_RETEST_ENABLED": True,
    "param_overrides.M15_TREND_RETEST_ALLOW_LONG": True,
    "param_overrides.M15_TREND_RETEST_ALLOW_SHORT": False,
    "param_overrides.M15_TREND_RETEST_ALLOW_RTH_PRIME1": True,
    "param_overrides.M15_TREND_RETEST_ALLOW_RTH_PRIME2": False,
    "param_overrides.M15_TREND_RETEST_ALLOW_RTH_DEAD": True,
    "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_AM": False,
    "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": False,
    "param_overrides.M15_TREND_RETEST_MIN_VOL_PCT": 0,
    "param_overrides.M15_TREND_RETEST_MAX_VOL_PCT": 95,
    "param_overrides.M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG": 1,
    "param_overrides.M15_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 2,
    "param_overrides.M15_TREND_RETEST_REQUIRE_STRONG_TREND": False,
    "param_overrides.M15_TREND_RETEST_MIN_TREND_STRENGTH": 0.15,
    "param_overrides.M15_TREND_RETEST_LOOKBACK_BARS": 5,
    "param_overrides.M15_TREND_RETEST_MIN_BARS_BETWEEN": 4,
    "param_overrides.M15_TREND_RETEST_PULLBACK_EMA_TOL_ATR": 0.55,
    "param_overrides.M15_TREND_RETEST_MAX_PULLBACK_ATR": 2.60,
    "param_overrides.M15_TREND_RETEST_MAX_BREAKOUT_GAP_ATR": 0.90,
}

_M15_SHORT = {
    **_M15_LONG,
    "param_overrides.M15_TREND_RETEST_ALLOW_LONG": False,
    "param_overrides.M15_TREND_RETEST_ALLOW_SHORT": True,
    "param_overrides.M15_TREND_RETEST_ALLOW_RTH_PRIME1": True,
    "param_overrides.M15_TREND_RETEST_ALLOW_RTH_DEAD": False,
    "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
    "param_overrides.M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT": 1,
    "param_overrides.M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT": 2,
}


_PHASE_1_CANDIDATES: list[Experiment] = [
    Experiment("micro_fast_no_rth2_follow", {**_MICRO_FAST, **_FOLLOWTHROUGH}),
    Experiment("micro_directional_clean", _MICRO_DIRECTIONAL_CLEAN),
    Experiment(
        "micro_directional_clean_follow",
        {**_MICRO_DIRECTIONAL_CLEAN, **_FOLLOWTHROUGH},
    ),
    Experiment(
        "micro_directional_rtd_short_probe",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            "param_overrides.MICRO_TREND_RETEST_ALLOW_SHORT_RTH_DEAD": True,
        },
    ),
    Experiment(
        "micro_directional_pm_long_probe",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            "param_overrides.MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_PM": True,
        },
    ),
    Experiment(
        "micro_directional_vol20_95",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            "param_overrides.MICRO_TREND_RETEST_MIN_VOL_PCT": 20,
            "param_overrides.MICRO_TREND_RETEST_MAX_VOL_PCT": 95,
        },
    ),
    Experiment(
        "micro_directional_highvol_only",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            "param_overrides.MICRO_TREND_RETEST_MIN_VOL_PCT": 80,
            "param_overrides.MICRO_TREND_RETEST_MAX_VOL_PCT": 97,
            "param_overrides.MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 1,
            "param_overrides.MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT": 1,
        },
    ),
    Experiment(
        "micro_quality_slow_directional",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            "param_overrides.MICRO_TREND_RETEST_MIN_TREND_STRENGTH": 0.20,
            "param_overrides.MICRO_TREND_RETEST_LOOKBACK_BARS": 6,
            "param_overrides.MICRO_TREND_RETEST_MIN_BARS_BETWEEN": 6,
            "param_overrides.MICRO_TREND_RETEST_MAX_PULLBACK_ATR": 2.25,
            "param_overrides.MICRO_TREND_RETEST_MAX_BREAKOUT_GAP_ATR": 0.80,
        },
    ),
]


_PHASE_2_CANDIDATES: list[Experiment] = [
    Experiment("m15_long_prime1_rtd", _M15_LONG),
    Experiment(
        "m15_long_prime1_only",
        {**_M15_LONG, "param_overrides.M15_TREND_RETEST_ALLOW_RTH_DEAD": False},
    ),
    Experiment(
        "m15_long_prime1_pm",
        {**_M15_LONG, "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True},
    ),
    Experiment(
        "m15_long_highvol",
        {
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_MIN_VOL_PCT": 75,
            "param_overrides.M15_TREND_RETEST_MAX_VOL_PCT": 97,
            "param_overrides.M15_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 1,
        },
    ),
    Experiment(
        "m15_long_fast",
        {
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_LOOKBACK_BARS": 4,
            "param_overrides.M15_TREND_RETEST_MIN_BARS_BETWEEN": 3,
            "param_overrides.M15_TREND_RETEST_MAX_BREAKOUT_GAP_ATR": 1.10,
            "param_overrides.M15_TREND_RETEST_MAX_PULLBACK_ATR": 3.00,
        },
    ),
    Experiment(
        "m15_long_score2",
        {
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG": 2,
            "param_overrides.M15_TREND_RETEST_MIN_TREND_STRENGTH": 0.25,
        },
    ),
    Experiment(
        "m15_long_h1_retest_combo",
        {**_M15_LONG, **_H1_RETEST},
    ),
]


_PHASE_3_CANDIDATES: list[Experiment] = [
    Experiment("m15_short_prime1_pm", _M15_SHORT),
    Experiment(
        "m15_short_prime1_only",
        {**_M15_SHORT, "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": False},
    ),
    Experiment(
        "m15_short_pm_only",
        {
            **_M15_SHORT,
            "param_overrides.M15_TREND_RETEST_ALLOW_RTH_PRIME1": False,
            "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
        },
    ),
    Experiment(
        "m15_short_rtd_probe",
        {
            **_M15_SHORT,
            "param_overrides.M15_TREND_RETEST_ALLOW_RTH_DEAD": True,
            "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": False,
        },
    ),
    Experiment(
        "m15_both_clean",
        {
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_ALLOW_SHORT": True,
            "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
            "param_overrides.M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT": 1,
            "param_overrides.M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT": 2,
        },
    ),
    Experiment(
        "m15_both_highvol",
        {
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_ALLOW_SHORT": True,
            "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
            "param_overrides.M15_TREND_RETEST_MIN_VOL_PCT": 75,
            "param_overrides.M15_TREND_RETEST_MAX_VOL_PCT": 97,
            "param_overrides.M15_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 1,
            "param_overrides.M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT": 1,
        },
    ),
    Experiment(
        "m15_short_with_failed_reclaim",
        {**_M15_SHORT, **_FAILED_RECLAIM_SHORT},
    ),
]


_PHASE_4_CANDIDATES: list[Experiment] = [
    Experiment("stack_micro_clean_m15_long", {**_MICRO_DIRECTIONAL_CLEAN, **_M15_LONG}),
    Experiment("stack_micro_clean_m15_short", {**_MICRO_DIRECTIONAL_CLEAN, **_M15_SHORT}),
    Experiment(
        "stack_micro_clean_m15_both",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_ALLOW_SHORT": True,
            "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
        },
    ),
    Experiment(
        "stack_micro_highvol_m15_highvol",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            **_M15_LONG,
            "param_overrides.MICRO_TREND_RETEST_MIN_VOL_PCT": 75,
            "param_overrides.M15_TREND_RETEST_MIN_VOL_PCT": 75,
            "param_overrides.M15_TREND_RETEST_MAX_VOL_PCT": 97,
        },
    ),
    Experiment("stack_micro_clean_h1_retest", {**_MICRO_DIRECTIONAL_CLEAN, **_H1_RETEST}),
    Experiment("stack_micro_clean_failed_reclaim", {**_MICRO_DIRECTIONAL_CLEAN, **_FAILED_RECLAIM_SHORT}),
    Experiment(
        "stack_micro_m15_followthrough",
        {**_MICRO_DIRECTIONAL_CLEAN, **_M15_LONG, **_FOLLOWTHROUGH},
    ),
    Experiment(
        "stack_all_clean_structural",
        {**_MICRO_DIRECTIONAL_CLEAN, **_M15_LONG, **_H1_RETEST, **_FAILED_RECLAIM_SHORT},
    ),
]


_PHASE_5_CANDIDATES: list[Experiment] = [
    Experiment(
        "micro_stop_tighter_h1",
        {
            "param_overrides.MICRO_TREND_RETEST_STOP_H1_ATR_FRAC": 0.25,
            "param_overrides.MICRO_TREND_RETEST_MAX_STOP_DIST_ATR": 1.55,
        },
    ),
    Experiment(
        "micro_stop_roomier_ltf",
        {
            "param_overrides.MICRO_TREND_RETEST_STOP_M5_ATR_MULT": 1.65,
            "param_overrides.MICRO_TREND_RETEST_STOP_H1_ATR_FRAC": 0.30,
            "param_overrides.MICRO_TREND_RETEST_MAX_STOP_DIST_ATR": 2.05,
        },
    ),
    Experiment(
        "m15_stop_tighter_h1",
        {
            "param_overrides.M15_TREND_RETEST_STOP_H1_ATR_FRAC": 0.25,
            "param_overrides.M15_TREND_RETEST_MAX_STOP_DIST_ATR": 1.55,
        },
    ),
    Experiment(
        "m15_stop_roomier_ltf",
        {
            "param_overrides.M15_TREND_RETEST_STOP_M15_ATR_MULT": 1.35,
            "param_overrides.M15_TREND_RETEST_STOP_H1_ATR_FRAC": 0.30,
            "param_overrides.M15_TREND_RETEST_MAX_STOP_DIST_ATR": 2.00,
        },
    ),
    Experiment("early_adverse_3_-0.60", {"param_overrides.EARLY_ADVERSE_BARS": 3, "param_overrides.EARLY_ADVERSE_R": -0.60}),
    Experiment("early_adverse_5_-0.70", {"param_overrides.EARLY_ADVERSE_BARS": 5, "param_overrides.EARLY_ADVERSE_R": -0.70}),
    Experiment("time_decay_10_progress_0.40", {"param_overrides.TIME_DECAY_BARS": 10, "param_overrides.TIME_DECAY_PROGRESS_R": 0.40}),
    Experiment("partial1_0.85_0.35", {"param_overrides.PARTIAL1_R": 0.85, "param_overrides.PARTIAL1_FRAC": 0.35}),
    Experiment("partial2_1.25_0.30", {"param_overrides.PARTIAL2_R": 1.25, "param_overrides.PARTIAL2_FRAC": 0.30}),
    Experiment("mfe_ratchet_1.0_0.50", {"param_overrides.MFE_RATCHET_ACTIVATION_R": 1.00, "param_overrides.MFE_RATCHET_FLOOR_PCT": 0.50}),
]


_PHASE_6_CANDIDATES: list[Experiment] = [
    Experiment("fixed_qty_8", {"fixed_qty": 8}),
    Experiment("fixed_qty_12", {"fixed_qty": 12}),
    Experiment("fixed_qty_16", {"fixed_qty": 16}),
    Experiment("fixed_qty_20", {"fixed_qty": 20}),
    Experiment("risk_based_1.5pct", {"fixed_qty": None, "param_overrides.BASE_RISK_PCT": 0.015}),
    Experiment("heat_2.5_2.0", {"param_overrides.HEAT_CAP_R": 2.5, "param_overrides.HEAT_CAP_DIR_R": 2.0}),
    Experiment("heat_3.5_3.0", {"param_overrides.HEAT_CAP_R": 3.5, "param_overrides.HEAT_CAP_DIR_R": 3.0}),
    Experiment("vol5080_size_0.70", {"flags.vol_50_80_sizing_mult": 0.70}),
]


_PHASE_7_CANDIDATES: list[Experiment] = [
    Experiment(
        "final_clean_micro_m15_fixed12",
        {**_MICRO_DIRECTIONAL_CLEAN, **_M15_LONG, "fixed_qty": 12},
    ),
    Experiment(
        "final_clean_micro_m15_fixed16",
        {**_MICRO_DIRECTIONAL_CLEAN, **_M15_LONG, "fixed_qty": 16},
    ),
    Experiment(
        "final_clean_micro_m15_follow_fixed12",
        {**_MICRO_DIRECTIONAL_CLEAN, **_M15_LONG, **_FOLLOWTHROUGH, "fixed_qty": 12},
    ),
    Experiment(
        "final_clean_micro_m15_both_fixed12",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_ALLOW_SHORT": True,
            "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
            "fixed_qty": 12,
        },
    ),
    Experiment(
        "final_highvol_stack_fixed12",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            **_M15_LONG,
            "param_overrides.MICRO_TREND_RETEST_MIN_VOL_PCT": 75,
            "param_overrides.M15_TREND_RETEST_MIN_VOL_PCT": 75,
            "fixed_qty": 12,
        },
    ),
    Experiment(
        "final_all_clean_fixed12",
        {**_MICRO_DIRECTIONAL_CLEAN, **_M15_LONG, **_H1_RETEST, **_FAILED_RECLAIM_SHORT, "fixed_qty": 12},
    ),
    Experiment(
        "final_120_quality_guard",
        {
            **_MICRO_DIRECTIONAL_CLEAN,
            **_M15_LONG,
            "param_overrides.MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 3,
            "param_overrides.M15_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 3,
            "fixed_qty": 12,
        },
    ),
    Experiment(
        "final_180_frequency_guarded",
        {
            **_MICRO_FAST,
            **_M15_LONG,
            "param_overrides.M15_TREND_RETEST_ALLOW_SHORT": True,
            "param_overrides.M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM": True,
            "param_overrides.MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_LONG": 2,
            "param_overrides.MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT": 2,
            "fixed_qty": 8,
        },
    ),
]


_REBASE_PHASE_CANDIDATES: dict[int, list[Experiment]] = {
    1: _PHASE_1_CANDIDATES,
    2: _PHASE_2_CANDIDATES,
    3: _PHASE_3_CANDIDATES,
    4: _PHASE_4_CANDIDATES,
    5: _PHASE_5_CANDIDATES,
    6: _PHASE_6_CANDIDATES,
    7: _PHASE_7_CANDIDATES,
}


def get_rebase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    candidates = [
        (experiment.name, dict(experiment.mutations))
        for experiment in _REBASE_PHASE_CANDIDATES.get(phase, [])
    ]
    if suggested_experiments:
        existing = {name for name, _ in candidates}
        for name, mutations in suggested_experiments:
            if name not in existing:
                candidates.append((name, dict(mutations)))
                existing.add(name)
    return candidates
