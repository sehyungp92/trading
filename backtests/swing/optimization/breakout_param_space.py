"""Breakout v3.3-ETF parameter space definition.

Reuses ParamRange, latin_hypercube_sample from the shared param_space module.
"""
from __future__ import annotations

from backtest.optimization.param_space import ParamRange


def _breakout_symbols() -> list[str]:
    try:
        from strategy_3.config import SYMBOLS
        return list(SYMBOLS)
    except ImportError:
        return ["QQQ", "GLD"]


def _build_per_symbol_atr_stop() -> list[ParamRange]:
    """Per-symbol ATR stop multiplier."""
    ranges: list[ParamRange] = []
    for sym in _breakout_symbols():
        ranges.append(ParamRange(
            f"atr_stop_mult_{sym}", 0.6, 1.5, 0.05, symbol=sym,
        ))
    return ranges


def _build_per_symbol_base_risk() -> list[ParamRange]:
    """Per-symbol base risk percentage."""
    ranges: list[ParamRange] = []
    for sym in _breakout_symbols():
        ranges.append(ParamRange(
            f"base_risk_pct_{sym}", 0.003, 0.010, 0.001, symbol=sym,
        ))
    return ranges


# ---------------------------------------------------------------------------
# Breakout parameter space (~40 params)
# ---------------------------------------------------------------------------

BREAKOUT_PARAM_SPACE: list[ParamRange] = [
    # --- Adaptive L (compression lookback) ---
    ParamRange("l_low", 6, 12, 1, True),
    ParamRange("l_mid", 10, 16, 1, True),
    ParamRange("l_high", 14, 22, 1, True),
    ParamRange("atr_ratio_low", 0.50, 0.90, 0.05),
    ParamRange("atr_ratio_high", 1.00, 1.50, 0.05),

    # --- Displacement ---
    ParamRange("q_disp", 0.55, 0.85, 0.05),
    ParamRange("q_disp_atr_expand_adj", 0.02, 0.10, 0.01),
    ParamRange("disp_strong_mult", 1.05, 1.30, 0.05),

    # --- Evidence score thresholds ---
    ParamRange("score_threshold_normal", 1, 4, 1, True),
    ParamRange("score_threshold_degraded", 2, 5, 1, True),
    ParamRange("score_threshold_range", 2, 5, 1, True),

    # --- Entry parameters ---
    ParamRange("entryb_rvolh_min", 0.5, 1.2, 0.1),
    ParamRange("reclaim_buffer_atr_h_mult", 0.06, 0.20, 0.02),
    ParamRange("sweep_depth_atr_d_mult", 0.15, 0.40, 0.05),
    ParamRange("entry_a_ttl_rth_hours", 4, 10, 1, True),
    ParamRange("entry_c_ttl_rth_hours", 2, 8, 1, True),

    # --- Add parameters ---
    ParamRange("max_adds_per_campaign", 1, 3, 1, True),
    ParamRange("campaign_risk_budget_mult", 1.2, 2.0, 0.1),
    ParamRange("add_risk_mult", 0.3, 0.7, 0.1),
    ParamRange("add_clv_strong_q", 0.20, 0.45, 0.05),

    # --- TP targets (per tier) ---
    ParamRange("tp1_r_aligned", 1.0, 2.0, 0.25),
    ParamRange("tp2_r_aligned", 2.0, 4.0, 0.25),
    ParamRange("tp1_r_neutral", 0.7, 1.5, 0.1),
    ParamRange("tp2_r_neutral", 1.5, 3.0, 0.25),
    ParamRange("tp1_r_caution", 0.5, 1.0, 0.1),
    ParamRange("tp2_r_caution", 1.0, 2.0, 0.25),

    # --- Trailing stop ---
    ParamRange("trail_4h_atr_mult", 0.3, 0.8, 0.05),
    ParamRange("be_buffer_atr_mult", 0.05, 0.20, 0.05),

    # --- Stale exit ---
    ParamRange("stale_exit_days_min", 7, 14, 1, True),
    ParamRange("stale_exit_days_max", 10, 18, 1, True),
    ParamRange("stale_r_thresh", 0.3, 0.8, 0.1),

    # --- Expiry decay ---
    ParamRange("expiry_bars_min", 2, 5, 1, True),
    ParamRange("expiry_bars_max", 7, 14, 1, True),
    ParamRange("expiry_decay_step", 0.06, 0.20, 0.02),

    # --- Chop mode ---
    ParamRange("chop_atr_pctl_high", 0.60, 0.90, 0.05),
    ParamRange("chop_cross_threshold", 3, 6, 1, True),
    ParamRange("chop_degraded_size_mult", 0.50, 1.00, 0.05),

    # --- Pending ---
    ParamRange("pending_max_rth_hours", 6, 20, 2, True),

    # --- Portfolio heat ---
    ParamRange("max_portfolio_heat", 0.015, 0.040, 0.005),

    # --- Per-symbol ATR stop and base risk ---
    *_build_per_symbol_atr_stop(),
    *_build_per_symbol_base_risk(),
]


def breakout_params_to_overrides(sample: dict[str, float]) -> dict[str, float]:
    """Convert a Breakout sample dict to config overrides.

    Breakout params map 1:1 to strategy_3.config module-level constants
    and SymbolConfig fields. Per-symbol params like atr_stop_mult_QQQ
    are passed through.
    """
    overrides = {}
    for key, value in sample.items():
        overrides[key] = value
    return overrides
