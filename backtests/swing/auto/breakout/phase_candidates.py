from __future__ import annotations

from typing import Any

BASE_MUTATIONS = {
    "flags.disable_score_threshold": True,
    "flags.disable_reentry_gate": True,
    "flags.disable_stale_exit": True,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("ALPHA-ACCRETIVE ENTRY EXPANSION", ["edge_velocity", "net_profit", "trades_per_month"]),
    2: ("ENTRY QUALITY VALIDATION", ["edge_velocity", "expectancy_dollar", "trades_per_month"]),
    3: ("BRANCH-SPECIFIC EXIT MONETISATION", ["edge_velocity", "net_profit", "expectancy_dollar"]),
    4: ("CARRY + INTEGRATION", ["edge_velocity", "net_profit", "trades_per_month"]),
}

_DEFAULTS: dict[str, Any] = {
    "flags.disable_score_threshold": False,
    "flags.disable_reentry_gate": False,
    "flags.disable_stale_exit": False,
    "param_overrides.HARD_BLOCK_SLOPE_MULT": 0.12,
    "param_overrides.Q_DISP": 0.60,
    "param_overrides.PRE_RUNNER_LOCK_FRAC": 0.30,
    "param_overrides.TP1_R_ALIGNED": 0.20,
    "param_overrides.TRAIL_MULT_BASE_FACTOR": 5.0,
    "param_overrides.TRAIL_4H_ATR_MULT": 0.40,
    "param_overrides.ENTRY_C_EARLY_ENABLE": False,
    "param_overrides.ENTRY_C_EARLY_MIN_SCORE": 0,
    "param_overrides.ENTRY_C_EARLY_MIN_QUALITY": 0.45,
    "param_overrides.ENTRY_C_EARLY_REQUIRE_ALIGNED": False,
    "param_overrides.ENTRY_C_EARLY_ALLOW_NEUTRAL": True,
    "param_overrides.ENTRY_C_EARLY_CLV_Q": 0.25,
    "param_overrides.ENTRY_C_EARLY_MAX_DISP_H": 2.75,
    "param_overrides.ENTRY_C_EARLY_MAX_BREAKOUT_BARS": 6,
    "param_overrides.ENTRY_C_EARLY_MIN_RVOL_H": 0.85,
    "param_overrides.ENTRY_C_FRESH_ENABLE": False,
    "param_overrides.ENTRY_C_FRESH_MIN_SCORE": 0,
    "param_overrides.ENTRY_C_FRESH_MIN_QUALITY": 0.68,
    "param_overrides.ENTRY_C_FRESH_REQUIRE_ALIGNED": True,
    "param_overrides.ENTRY_C_FRESH_ALLOW_COUNTERTREND": True,
    "param_overrides.ENTRY_C_FRESH_CLV_Q": 0.25,
    "param_overrides.ENTRY_C_FRESH_MAX_DISP_H": 3.00,
    "param_overrides.ENTRY_C_FRESH_TOUCH_TOL_ATR_H": 0.50,
    "param_overrides.ENTRY_C_FRESH_MAX_BREAKOUT_BARS": 4,
    "param_overrides.ENTRY_C_FRESH_USE_STOP_LIMIT": False,
    "param_overrides.ENTRY_C_FRESH_STOP_ENABLE": False,
    "param_overrides.ENTRY_C_FRESH_STOP_MIN_SCORE": 1,
    "param_overrides.ENTRY_C_FRESH_STOP_MIN_QUALITY": 0.55,
    "param_overrides.ENTRY_C_FRESH_STOP_REQUIRE_ALIGNED": True,
    "param_overrides.ENTRY_C_FRESH_STOP_ALLOW_COUNTERTREND": False,
    "param_overrides.ENTRY_C_FRESH_STOP_CLV_Q": 0.30,
    "param_overrides.ENTRY_C_FRESH_STOP_MAX_DISP_H": 2.50,
    "param_overrides.ENTRY_C_FRESH_STOP_TOUCH_TOL_ATR_H": 0.40,
    "param_overrides.ENTRY_C_FRESH_STOP_MAX_BREAKOUT_BARS": 4,
    "param_overrides.ENTRY_C_FRESH_STOP_MIN_RVOL_H": 0.95,
    "param_overrides.ENTRY_C_FRESH_STOP_BUFFER_ATR_H": 0.05,
    "param_overrides.ENTRY_C_FRESH_LIMIT_OFFSET_ATR_H": 0.10,
    "param_overrides.ENTRY_C_MOMENTUM_ENABLE": False,
    "param_overrides.ENTRY_C_MOMENTUM_MIN_SCORE": 1,
    "param_overrides.ENTRY_C_MOMENTUM_MIN_QUALITY": 0.75,
    "param_overrides.ENTRY_C_MOMENTUM_REQUIRE_ALIGNED": True,
    "param_overrides.ENTRY_C_MOMENTUM_CLV_Q": 0.30,
    "param_overrides.ENTRY_C_MOMENTUM_MAX_DISP_H": 2.75,
    "param_overrides.ENTRY_C_MOMENTUM_USE_STOP_LIMIT": False,
    "param_overrides.ENTRY_C_CONTINUATION_ENABLE": False,
    "param_overrides.ENTRY_C_CONTINUATION_MIN_SCORE": 0,
    "param_overrides.ENTRY_C_CONTINUATION_MIN_QUALITY": 0.60,
    "param_overrides.ENTRY_C_CONTINUATION_REQUIRE_ALIGNED": False,
    "param_overrides.ENTRY_C_CONTINUATION_ALLOW_NEUTRAL": True,
    "param_overrides.ENTRY_C_CONTINUATION_CLV_Q": 0.20,
    "param_overrides.ENTRY_C_CONTINUATION_HOLD_BARS": 1,
    "param_overrides.ENTRY_C_CONTINUATION_PAUSE_ATR_H": 1.25,
    "param_overrides.ENTRY_C_CONTINUATION_MAX_DISP_H": 3.50,
    "param_overrides.ENTRY_C_CONTINUATION_MAX_BREAKOUT_BARS": 999,
    "param_overrides.ENTRY_C_CONTINUATION_MIN_RVOL_H": 0.70,
    "param_overrides.ENTRY_C_STANDARD_ALLOW_CONTINUATION": True,
    "param_overrides.ENTRY_A_STRONG_ENABLE": False,
    "param_overrides.ENTRY_A_STRONG_MIN_SCORE": 1,
    "param_overrides.ENTRY_A_STRONG_MIN_QUALITY": 0.75,
    "param_overrides.ENTRY_A_STRONG_REQUIRE_ALIGNED": True,
    "param_overrides.ENTRY_A_STRONG_CLV_Q": 0.30,
    "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_ENABLE": False,
    "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_MIN_SCORE": 1,
    "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_MIN_QUALITY": 0.75,
    "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_REQUIRE_ALIGNED": True,
    "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_TTL_HOURS": 24,
    "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_FRESH_ONLY": False,
    "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_A_OR_FRESH_ONLY": False,
    "param_overrides.FAST_BRANCH_MANAGEMENT_ENABLE": True,
    "param_overrides.FAST_BRANCH_TP1_R_ALIGNED": 0.22,
    "param_overrides.FAST_BRANCH_TP1_R_NEUTRAL": 0.18,
    "param_overrides.FAST_BRANCH_TP1_R_CAUTION": 0.12,
    "param_overrides.FAST_BRANCH_PRE_RUNNER_LOCK_FRAC": 0.35,
    "param_overrides.FAST_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R": 0.05,
    "param_overrides.FAST_BRANCH_TRAIL_MULT_BASE_FACTOR": 5.0,
    "param_overrides.FAST_BRANCH_TRAIL_4H_ATR_MULT": 0.35,
    "param_overrides.C_MOMENTUM_RISK_MULT": 1.00,
    "param_overrides.MOMENTUM_BRANCH_MANAGEMENT_ENABLE": False,
    "param_overrides.MOMENTUM_BRANCH_TP1_R_ALIGNED": 0.32,
    "param_overrides.MOMENTUM_BRANCH_TP1_R_NEUTRAL": 0.26,
    "param_overrides.MOMENTUM_BRANCH_TP1_R_CAUTION": 0.16,
    "param_overrides.MOMENTUM_BRANCH_TP1_PARTIAL_FRAC": 0.20,
    "param_overrides.MOMENTUM_BRANCH_PRE_RUNNER_LOCK_FRAC": 0.70,
    "param_overrides.MOMENTUM_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R": 0.18,
    "param_overrides.MOMENTUM_BRANCH_TRAIL_MULT_BASE_FACTOR": 5.5,
    "param_overrides.MOMENTUM_BRANCH_TRAIL_4H_ATR_MULT": 0.42,
    "param_overrides.CONTINUATION_BRANCH_MANAGEMENT_ENABLE": False,
    "param_overrides.CONTINUATION_BRANCH_TP1_R_ALIGNED": 0.18,
    "param_overrides.CONTINUATION_BRANCH_TP1_R_NEUTRAL": 0.14,
    "param_overrides.CONTINUATION_BRANCH_TP1_R_CAUTION": 0.10,
    "param_overrides.CONTINUATION_BRANCH_TP1_PARTIAL_FRAC": 0.45,
    "param_overrides.CONTINUATION_BRANCH_PRE_RUNNER_LOCK_FRAC": 0.60,
    "param_overrides.CONTINUATION_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R": 0.10,
    "param_overrides.CONTINUATION_BRANCH_TRAIL_MULT_BASE_FACTOR": 4.0,
    "param_overrides.CONTINUATION_BRANCH_TRAIL_4H_ATR_MULT": 0.30,
    "param_overrides.C_STANDARD_LATE_RISK_MULT": 0.85,
    "param_overrides.C_STANDARD_HIGH_DISP_RISK_MULT": 0.75,
    "param_overrides.C_CONTINUATION_RISK_MULT": 0.80,
    "param_overrides.C_STANDARD_LATE_BREAKOUT_BARS": 6,
    "param_overrides.C_STANDARD_HIGH_DISP_THRESHOLD": 2.75,
}


def get_phase_candidates(
    phase: int,
    current_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    current = dict(current_mutations or {})
    if phase == 1:
        candidates = _phase_1_candidates(current)
    elif phase == 2:
        candidates = _phase_2_candidates(current)
    elif phase == 3:
        candidates = _phase_3_candidates(current)
    elif phase == 4:
        candidates = _phase_4_candidates(current)
    else:
        candidates = []

    if suggested_experiments:
        existing = {name for name, _ in candidates}
        for name, mutations in suggested_experiments:
            if name not in existing:
                candidates.append((name, dict(mutations)))
                existing.add(name)
    return candidates


def _phase_1_candidates(current: dict[str, Any]) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    _append(
        candidates,
        current,
        "continuation_route_core",
        _merge(
            _continuation_route_mutation(
                min_score=0,
                min_quality=0.55,
                require_aligned=False,
                allow_neutral=True,
                clv_q=0.20,
                hold_bars=1,
                pause_atr_h=1.25,
                max_disp_h=3.50,
                max_breakout_bars=14,
                min_rvol_h=0.70,
                risk_mult=0.80,
                allow_standard=False,
            ),
            _continuation_management_mutation(
                tp1_aligned=0.18,
                tp1_neutral=0.14,
                tp1_caution=0.10,
                partial_frac=0.45,
                lock_frac=0.60,
                lock_threshold=0.10,
                trail_base=4.0,
                trail_atr=0.30,
            ),
        ),
    )
    _append(
        candidates,
        current,
        "continuation_route_quality",
        _merge(
            _continuation_route_mutation(
                min_score=0,
                min_quality=0.62,
                require_aligned=False,
                allow_neutral=True,
                clv_q=0.20,
                hold_bars=1,
                pause_atr_h=0.95,
                max_disp_h=3.00,
                max_breakout_bars=10,
                min_rvol_h=0.80,
                risk_mult=0.70,
                allow_standard=False,
            ),
            _continuation_management_mutation(
                tp1_aligned=0.16,
                tp1_neutral=0.12,
                tp1_caution=0.08,
                partial_frac=0.55,
                lock_frac=0.65,
                lock_threshold=0.08,
                trail_base=3.8,
                trail_atr=0.28,
            ),
        ),
    )
    _append(
        candidates,
        current,
        "momentum_exit_redesign",
        _merge(
            _momentum_branch_mutation(
                min_score=0,
                min_quality=0.55,
                require_aligned=False,
                clv_q=0.20,
                max_disp_h=3.00,
            ),
            _momentum_management_mutation(
                tp1_aligned=0.32,
                tp1_neutral=0.26,
                tp1_caution=0.16,
                partial_frac=0.20,
                lock_frac=0.70,
                lock_threshold=0.18,
                trail_base=5.5,
                trail_atr=0.42,
                risk_mult=1.05,
            ),
        ),
    )
    _append(
        candidates,
        current,
        "continuation_momentum_combo",
        _merge(
            _continuation_route_mutation(
                min_score=0,
                min_quality=0.55,
                require_aligned=False,
                allow_neutral=True,
                clv_q=0.20,
                hold_bars=1,
                pause_atr_h=1.25,
                max_disp_h=3.50,
                max_breakout_bars=14,
                min_rvol_h=0.70,
                risk_mult=0.80,
                allow_standard=False,
            ),
            _continuation_management_mutation(
                tp1_aligned=0.18,
                tp1_neutral=0.14,
                tp1_caution=0.10,
                partial_frac=0.45,
                lock_frac=0.60,
                lock_threshold=0.10,
                trail_base=4.0,
                trail_atr=0.30,
            ),
            _momentum_branch_mutation(
                min_score=0,
                min_quality=0.55,
                require_aligned=False,
                clv_q=0.20,
                max_disp_h=3.00,
            ),
            _momentum_management_mutation(
                tp1_aligned=0.32,
                tp1_neutral=0.26,
                tp1_caution=0.16,
                partial_frac=0.20,
                lock_frac=0.70,
                lock_threshold=0.18,
                trail_base=5.5,
                trail_atr=0.42,
                risk_mult=1.05,
            ),
        ),
    )
    _append(
        candidates,
        current,
        "broad_continuation_momentum_combo",
        _merge(
            _continuation_route_mutation(
                min_score=0,
                min_quality=0.25,
                require_aligned=False,
                allow_neutral=True,
                clv_q=0.00,
                hold_bars=1,
                pause_atr_h=999.0,
                max_disp_h=999.0,
                max_breakout_bars=999,
                min_rvol_h=0.00,
                risk_mult=0.80,
                allow_standard=False,
            ),
            _continuation_management_mutation(
                tp1_aligned=0.18,
                tp1_neutral=0.14,
                tp1_caution=0.10,
                partial_frac=0.45,
                lock_frac=0.60,
                lock_threshold=0.10,
                trail_base=4.0,
                trail_atr=0.30,
            ),
            _momentum_branch_mutation(
                min_score=0,
                min_quality=0.55,
                require_aligned=False,
                clv_q=0.20,
                max_disp_h=3.00,
            ),
            _momentum_management_mutation(
                tp1_aligned=0.32,
                tp1_neutral=0.26,
                tp1_caution=0.16,
                partial_frac=0.20,
                lock_frac=0.70,
                lock_threshold=0.18,
                trail_base=5.5,
                trail_atr=0.42,
                risk_mult=1.05,
            ),
        ),
    )
    _append(
        candidates,
        current,
        "momentum_stop_entry",
        _merge(
            _momentum_branch_mutation(
                min_score=0,
                min_quality=0.55,
                require_aligned=False,
                clv_q=0.20,
                max_disp_h=3.00,
                use_stop_limit=True,
            ),
            _momentum_management_mutation(
                tp1_aligned=0.30,
                tp1_neutral=0.24,
                tp1_caution=0.15,
                partial_frac=0.25,
                lock_frac=0.65,
                lock_threshold=0.15,
                trail_base=5.2,
                trail_atr=0.40,
                risk_mult=1.00,
            ),
        ),
    )
    return candidates


def _phase_2_candidates(current: dict[str, Any]) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    _append(
        candidates,
        current,
        "late_flow_risk_split",
        _late_flow_risk_mutation(late_mult=0.70, high_disp_mult=0.60, late_bars=5, high_disp_threshold=2.50),
    )
    _append(
        candidates,
        current,
        "late_flow_soft_split",
        _late_flow_risk_mutation(late_mult=0.80, high_disp_mult=0.70, late_bars=6, high_disp_threshold=2.75),
    )
    _append(
        candidates,
        current,
        "continuation_quality_raise",
        _continuation_route_mutation(
            min_score=0,
            min_quality=0.65,
            require_aligned=False,
            allow_neutral=True,
            clv_q=0.20,
            hold_bars=1,
            pause_atr_h=0.95,
            max_disp_h=3.00,
            max_breakout_bars=10,
            min_rvol_h=0.85,
            risk_mult=0.70,
            allow_standard=False,
        ),
    )
    _append(
        candidates,
        current,
        "continuation_risk_demote",
        {"param_overrides.C_CONTINUATION_RISK_MULT": 0.65},
    )
    _append(
        candidates,
        current,
        "momentum_risk_promote",
        {"param_overrides.C_MOMENTUM_RISK_MULT": 1.10},
    )
    return candidates


def _phase_3_candidates(current: dict[str, Any]) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    _append(candidates, current, "momentum_runner_capture", _momentum_management_mutation(
        tp1_aligned=0.36,
        tp1_neutral=0.28,
        tp1_caution=0.18,
        partial_frac=0.15,
        lock_frac=0.65,
        lock_threshold=0.22,
        trail_base=6.0,
        trail_atr=0.45,
        risk_mult=1.05,
    ))
    _append(candidates, current, "momentum_quick_capture", _momentum_management_mutation(
        tp1_aligned=0.22,
        tp1_neutral=0.18,
        tp1_caution=0.12,
        partial_frac=0.35,
        lock_frac=0.60,
        lock_threshold=0.08,
        trail_base=4.5,
        trail_atr=0.34,
        risk_mult=1.00,
    ))
    _append(candidates, current, "continuation_quick_bank", _continuation_management_mutation(
        tp1_aligned=0.16,
        tp1_neutral=0.12,
        tp1_caution=0.08,
        partial_frac=0.55,
        lock_frac=0.70,
        lock_threshold=0.08,
        trail_base=3.5,
        trail_atr=0.25,
    ))
    _append(candidates, current, "continuation_runner_capture", _continuation_management_mutation(
        tp1_aligned=0.22,
        tp1_neutral=0.18,
        tp1_caution=0.12,
        partial_frac=0.35,
        lock_frac=0.50,
        lock_threshold=0.14,
        trail_base=4.5,
        trail_atr=0.35,
    ))
    return candidates


def _phase_4_candidates(current: dict[str, Any]) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    _append(candidates, current, "strict_continuation_split", {
        "param_overrides.ENTRY_C_CONTINUATION_ENABLE": True,
        "param_overrides.ENTRY_C_STANDARD_ALLOW_CONTINUATION": False,
    })
    _append(candidates, current, "aligned_continuation_only", {
        "param_overrides.ENTRY_C_CONTINUATION_REQUIRE_ALIGNED": True,
        "param_overrides.ENTRY_C_CONTINUATION_ALLOW_NEUTRAL": False,
    })
    _append(candidates, current, "fresh_market_off_validation", {
        "param_overrides.ENTRY_C_FRESH_ENABLE": False,
    })
    _append(candidates, current, "fresh_stop_off_validation", {
        "param_overrides.ENTRY_C_FRESH_STOP_ENABLE": False,
    })
    _append(candidates, current, "carry_off_validation", {
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_ENABLE": False,
    })
    return candidates


def _append(
    candidates: list[tuple[str, dict]],
    current: dict[str, Any],
    name: str,
    mutations: dict[str, Any],
) -> None:
    if any(_current_value(current, key) != value for key, value in mutations.items()):
        candidates.append((name, mutations))


def _current_value(current: dict[str, Any], key: str) -> Any:
    if key in current:
        return current[key]
    return _DEFAULTS.get(key)


def _merge(*groups: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for group in groups:
        merged.update(group)
    return merged


def _early_standard_mutation(
    *,
    min_quality: float,
    require_aligned: bool,
    allow_neutral: bool,
    max_breakout_bars: int,
    max_disp_h: float,
    min_rvol_h: float,
) -> dict[str, Any]:
    return {
        "param_overrides.ENTRY_C_EARLY_ENABLE": True,
        "param_overrides.ENTRY_C_EARLY_MIN_SCORE": 0,
        "param_overrides.ENTRY_C_EARLY_MIN_QUALITY": min_quality,
        "param_overrides.ENTRY_C_EARLY_REQUIRE_ALIGNED": require_aligned,
        "param_overrides.ENTRY_C_EARLY_ALLOW_NEUTRAL": allow_neutral,
        "param_overrides.ENTRY_C_EARLY_CLV_Q": 0.25,
        "param_overrides.ENTRY_C_EARLY_MAX_DISP_H": max_disp_h,
        "param_overrides.ENTRY_C_EARLY_MAX_BREAKOUT_BARS": max_breakout_bars,
        "param_overrides.ENTRY_C_EARLY_MIN_RVOL_H": min_rvol_h,
    }


def _momentum_branch_mutation(
    *,
    min_score: int,
    min_quality: float,
    require_aligned: bool,
    clv_q: float,
    max_disp_h: float,
    use_stop_limit: bool = False,
) -> dict[str, Any]:
    return {
        "param_overrides.ENTRY_C_MOMENTUM_ENABLE": True,
        "param_overrides.ENTRY_C_MOMENTUM_MIN_SCORE": min_score,
        "param_overrides.ENTRY_C_MOMENTUM_MIN_QUALITY": min_quality,
        "param_overrides.ENTRY_C_MOMENTUM_REQUIRE_ALIGNED": require_aligned,
        "param_overrides.ENTRY_C_MOMENTUM_CLV_Q": clv_q,
        "param_overrides.ENTRY_C_MOMENTUM_MAX_DISP_H": max_disp_h,
        "param_overrides.ENTRY_C_MOMENTUM_USE_STOP_LIMIT": use_stop_limit,
    }


def _continuation_route_mutation(
    *,
    min_score: int,
    min_quality: float,
    require_aligned: bool,
    allow_neutral: bool,
    clv_q: float,
    hold_bars: int,
    pause_atr_h: float,
    max_disp_h: float,
    max_breakout_bars: int,
    min_rvol_h: float,
    risk_mult: float,
    allow_standard: bool,
) -> dict[str, Any]:
    return {
        "param_overrides.ENTRY_C_CONTINUATION_ENABLE": True,
        "param_overrides.ENTRY_C_CONTINUATION_MIN_SCORE": min_score,
        "param_overrides.ENTRY_C_CONTINUATION_MIN_QUALITY": min_quality,
        "param_overrides.ENTRY_C_CONTINUATION_REQUIRE_ALIGNED": require_aligned,
        "param_overrides.ENTRY_C_CONTINUATION_ALLOW_NEUTRAL": allow_neutral,
        "param_overrides.ENTRY_C_CONTINUATION_CLV_Q": clv_q,
        "param_overrides.ENTRY_C_CONTINUATION_HOLD_BARS": hold_bars,
        "param_overrides.ENTRY_C_CONTINUATION_PAUSE_ATR_H": pause_atr_h,
        "param_overrides.ENTRY_C_CONTINUATION_MAX_DISP_H": max_disp_h,
        "param_overrides.ENTRY_C_CONTINUATION_MAX_BREAKOUT_BARS": max_breakout_bars,
        "param_overrides.ENTRY_C_CONTINUATION_MIN_RVOL_H": min_rvol_h,
        "param_overrides.C_CONTINUATION_RISK_MULT": risk_mult,
        "param_overrides.ENTRY_C_STANDARD_ALLOW_CONTINUATION": allow_standard,
    }


def _momentum_management_mutation(
    *,
    tp1_aligned: float,
    tp1_neutral: float,
    tp1_caution: float,
    partial_frac: float,
    lock_frac: float,
    lock_threshold: float,
    trail_base: float,
    trail_atr: float,
    risk_mult: float,
) -> dict[str, Any]:
    return {
        "param_overrides.MOMENTUM_BRANCH_MANAGEMENT_ENABLE": True,
        "param_overrides.MOMENTUM_BRANCH_TP1_R_ALIGNED": tp1_aligned,
        "param_overrides.MOMENTUM_BRANCH_TP1_R_NEUTRAL": tp1_neutral,
        "param_overrides.MOMENTUM_BRANCH_TP1_R_CAUTION": tp1_caution,
        "param_overrides.MOMENTUM_BRANCH_TP1_PARTIAL_FRAC": partial_frac,
        "param_overrides.MOMENTUM_BRANCH_PRE_RUNNER_LOCK_FRAC": lock_frac,
        "param_overrides.MOMENTUM_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R": lock_threshold,
        "param_overrides.MOMENTUM_BRANCH_TRAIL_MULT_BASE_FACTOR": trail_base,
        "param_overrides.MOMENTUM_BRANCH_TRAIL_4H_ATR_MULT": trail_atr,
        "param_overrides.C_MOMENTUM_RISK_MULT": risk_mult,
    }


def _continuation_management_mutation(
    *,
    tp1_aligned: float,
    tp1_neutral: float,
    tp1_caution: float,
    partial_frac: float,
    lock_frac: float,
    lock_threshold: float,
    trail_base: float,
    trail_atr: float,
) -> dict[str, Any]:
    return {
        "param_overrides.CONTINUATION_BRANCH_MANAGEMENT_ENABLE": True,
        "param_overrides.CONTINUATION_BRANCH_TP1_R_ALIGNED": tp1_aligned,
        "param_overrides.CONTINUATION_BRANCH_TP1_R_NEUTRAL": tp1_neutral,
        "param_overrides.CONTINUATION_BRANCH_TP1_R_CAUTION": tp1_caution,
        "param_overrides.CONTINUATION_BRANCH_TP1_PARTIAL_FRAC": partial_frac,
        "param_overrides.CONTINUATION_BRANCH_PRE_RUNNER_LOCK_FRAC": lock_frac,
        "param_overrides.CONTINUATION_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R": lock_threshold,
        "param_overrides.CONTINUATION_BRANCH_TRAIL_MULT_BASE_FACTOR": trail_base,
        "param_overrides.CONTINUATION_BRANCH_TRAIL_4H_ATR_MULT": trail_atr,
    }


def _late_flow_risk_mutation(
    *,
    late_mult: float,
    high_disp_mult: float,
    late_bars: int,
    high_disp_threshold: float,
) -> dict[str, Any]:
    return {
        "param_overrides.C_STANDARD_LATE_RISK_MULT": late_mult,
        "param_overrides.C_STANDARD_HIGH_DISP_RISK_MULT": high_disp_mult,
        "param_overrides.C_STANDARD_LATE_BREAKOUT_BARS": late_bars,
        "param_overrides.C_STANDARD_HIGH_DISP_THRESHOLD": high_disp_threshold,
    }


def _a_strong_mutation(
    *,
    min_score: int,
    min_quality: float,
    require_aligned: bool,
    clv_q: float,
) -> dict[str, Any]:
    return {
        "param_overrides.ENTRY_A_STRONG_ENABLE": True,
        "param_overrides.ENTRY_A_STRONG_MIN_SCORE": min_score,
        "param_overrides.ENTRY_A_STRONG_MIN_QUALITY": min_quality,
        "param_overrides.ENTRY_A_STRONG_REQUIRE_ALIGNED": require_aligned,
        "param_overrides.ENTRY_A_STRONG_CLV_Q": clv_q,
    }


def _carry_mutation(
    *,
    min_score: int,
    min_quality: float,
    ttl_hours: int,
    require_aligned: bool,
    fresh_only: bool = False,
    a_or_fresh_only: bool = False,
) -> dict[str, Any]:
    return {
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_ENABLE": True,
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_MIN_SCORE": min_score,
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_MIN_QUALITY": min_quality,
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_REQUIRE_ALIGNED": require_aligned,
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_TTL_HOURS": ttl_hours,
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_FRESH_ONLY": fresh_only,
        "param_overrides.ENTRY_OUTSIDE_WINDOW_CARRY_A_OR_FRESH_ONLY": a_or_fresh_only,
    }


def _fast_branch_mutation(
    *,
    tp1_aligned: float,
    tp1_neutral: float,
    tp1_caution: float,
    lock_frac: float,
    lock_threshold: float,
    trail_base: float,
    trail_atr: float,
) -> dict[str, Any]:
    return {
        "param_overrides.FAST_BRANCH_MANAGEMENT_ENABLE": True,
        "param_overrides.FAST_BRANCH_TP1_R_ALIGNED": tp1_aligned,
        "param_overrides.FAST_BRANCH_TP1_R_NEUTRAL": tp1_neutral,
        "param_overrides.FAST_BRANCH_TP1_R_CAUTION": tp1_caution,
        "param_overrides.FAST_BRANCH_PRE_RUNNER_LOCK_FRAC": lock_frac,
        "param_overrides.FAST_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R": lock_threshold,
        "param_overrides.FAST_BRANCH_TRAIL_MULT_BASE_FACTOR": trail_base,
        "param_overrides.FAST_BRANCH_TRAIL_4H_ATR_MULT": trail_atr,
    }
