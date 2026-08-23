"""Pullback signal logic for IARIC V2 (hybrid intraday engine).

Replaces T1 FSM setup detection with 7-trigger daily scoring,
trend tier classification, and intraday entry score bundling.
"""
from __future__ import annotations

import numpy as np

from .config import StrategySettings
from .indicators import (
    atr,
    bollinger_pctb,
    consecutive_down_days,
    ema,
    pullback_depth,
    rate_of_change,
    relative_strength_ratio,
    rolling_sma,
    rsi,
    volume_climax_ratio,
)
from .models import Bar, ResearchSymbol
from .core.residual import align_values_by_date


# ---------------------------------------------------------------------------
# Indicator cache builder (called once per symbol per day during research)
# ---------------------------------------------------------------------------

def compute_indicator_cache(
    symbol: ResearchSymbol,
    benchmark_closes: np.ndarray | None = None,
    benchmark_dates: list[object] | None = None,
) -> dict[str, np.ndarray]:
    """Compute all pullback indicators from daily bars.

    Returns a dict of indicator name -> numpy array (same length as daily_bars).
    """
    bars = symbol.daily_bars
    if len(bars) < 20:
        return {}
    closes = np.array([b.close for b in bars], dtype=np.float64)
    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)
    volumes = np.array([b.volume for b in bars], dtype=np.float64)

    atr14 = atr(highs, lows, closes, 14)
    aligned_benchmark = benchmark_closes
    if benchmark_closes is not None and benchmark_dates is not None:
        aligned_benchmark = np.asarray(
            align_values_by_date(
                [bar.trade_date for bar in bars],
                benchmark_dates,
                benchmark_closes,
            ),
            dtype=np.float64,
        )
    return {
        "rsi2": rsi(closes, 2),
        "rsi5": rsi(closes, 5),
        "atr14": atr14,
        "cdd": consecutive_down_days(closes),
        "depth": pullback_depth(highs, closes, atr14, lookback=10),
        "bb_pctb": bollinger_pctb(closes, 20, 2.0),
        "vcr": volume_climax_ratio(volumes, 20),
        "rs_ratio": relative_strength_ratio(closes, aligned_benchmark, 20),
        "roc5": rate_of_change(closes, 5),
        "rsi14": rsi(closes, 14),
        "sma50": rolling_sma(closes, 50),
        "sma200": rolling_sma(closes, 200),
        "ema10": ema(closes, 10),
    }


# ---------------------------------------------------------------------------
# Daily candidate scoring (7 triggers)
# ---------------------------------------------------------------------------


def _peak01(value: float, *, target: float, width: float) -> float:
    if np.isnan(value):
        return 0.5
    return _clip01(1.0 - abs(float(value) - float(target)) / max(float(width), 1e-6))


def score_daily_pullback_context(
    *,
    trend_tier: str,
    rsi2: float,
    rsi5: float,
    cdd: int,
    depth_atr: float,
    bb_pctb: float,
    volume_climax: float,
    is_down_day: bool,
    rs_ratio: float,
    roc5: float,
    sma_slope_positive: bool = True,
    sma_dist_pct: float = 0.0,
    trigger_tier: str = "MEDIUM",
    n_triggers: int = 1,
    regime_tier: str = "B",
    persistence: float = 0.0,
) -> dict[str, float]:
    """Score the intended trend-pullback context with seven components.

    This is the shared/live form of the historically useful replay context
    model, grouped into seven independent economic ideas.  Pullback depth is
    peak-shaped; intraday reclaim quality remains the primary discriminator.
    """

    trend = 7.0 if sma_slope_positive else 0.0
    trend += 7.0 if trend_tier == "STRONG" else 4.0 if trend_tier == "SECULAR" else 0.0
    if 0.0 <= sma_dist_pct <= 12.0:
        trend += 8.0
    elif -5.0 <= sma_dist_pct <= 20.0:
        trend += 4.0
    elif trend_tier != "EXCLUDED":
        trend += 1.0

    if np.isnan(depth_atr):
        pullback = 0.0
    elif depth_atr > 3.5:
        pullback = 4.0
    elif depth_atr > 2.5:
        pullback = 10.0
    elif depth_atr > 1.5:
        pullback = 18.0
    elif depth_atr > 1.0:
        pullback = 12.0
    elif depth_atr > 0.5:
        pullback = 7.0
    else:
        pullback = 2.0

    trigger = 15.0 if trigger_tier == "HIGH" else 10.0
    if n_triggers >= 2:
        trigger += 3.0

    rsi2_score = 15.0 if rsi2 < 5 else 12.0 if rsi2 < 10 else 9.0 if rsi2 < 15 else 6.0 if rsi2 < 20 else 0.0
    rsi5_score = 12.0 if rsi5 < 20 else 9.0 if rsi5 < 25 else 6.0 if rsi5 < 30 else 0.0
    oversold = max(rsi2_score, rsi5_score)

    exhaustion = 3.0
    if is_down_day:
        exhaustion = 10.0 if volume_climax > 2.5 else 8.0 if volume_climax > 2.0 else 6.0 if volume_climax > 1.5 else 3.0
    elif volume_climax > 2.0:
        exhaustion = 4.0

    relative_strength = 1.0
    if not np.isnan(rs_ratio):
        relative_strength = 8.0 if rs_ratio > 1.10 else 7.0 if rs_ratio > 1.05 else 5.0 if rs_ratio > 1.00 else 3.0 if rs_ratio > 0.95 else 1.0
    regime = 6.0 if regime_tier == "A" else 4.0 if regime_tier == "B" else 1.0
    context = regime + (4.0 if persistence > 0.70 else 3.0 if persistence > 0.50 else 1.0 if persistence > 0.0 else 0.0)
    components = {
        "trend_quality": float(trend),
        "pullback_quality": float(pullback),
        "trigger_quality": float(trigger),
        "oversold_quality": float(oversold),
        "exhaustion_quality": float(exhaustion),
        "relative_strength": float(relative_strength),
        "market_persistence_context": float(context),
    }
    components["score"] = float(sum(components.values()))
    return components

def score_daily_candidate(
    symbol: ResearchSymbol,
    indicators: dict[str, np.ndarray],
    config: StrategySettings,
    *,
    regime_tier: str = "B",
) -> tuple[float, list[str]]:
    """Score a symbol for pullback candidacy using 7 triggers.

    Returns (composite_score, list_of_triggered_names).
    """
    if not indicators or len(symbol.daily_bars) < 20:
        return 0.0, []

    idx = -1  # latest bar
    def _val(key: str) -> float:
        arr = indicators.get(key)
        if arr is None or len(arr) == 0:
            return float("nan")
        v = arr[idx]
        return float(v) if not np.isnan(v) else float("nan")

    rsi2_val = _val("rsi2")
    rsi5_val = _val("rsi5")
    cdd_val = int(_val("cdd")) if not np.isnan(_val("cdd")) else 0
    depth_val = _val("depth")
    bb_val = _val("bb_pctb")
    vcr_val = _val("vcr")
    rs_val = _val("rs_ratio")
    roc5_val = _val("roc5")

    triggers: list[str] = []

    # Trigger A: RSI(2) < 15
    if not np.isnan(rsi2_val) and rsi2_val < config.pb_v2_rsi2_thresh:
        triggers.append("RSI2")

    # Trigger B: RSI(5) < 30 + CDD >= 2
    if not np.isnan(rsi5_val) and rsi5_val < config.pb_v2_rsi5_thresh and cdd_val >= config.pb_v2_cdd_min_for_rsi5:
        triggers.append("RSI5_CDD")

    # Trigger C: Pullback depth > 1.5 ATR
    if not np.isnan(depth_val) and depth_val > config.pb_v2_depth_thresh:
        triggers.append("DEPTH")

    # Trigger D: Bollinger %B < 0.05
    if not np.isnan(bb_val) and bb_val < config.pb_v2_bb_pctb_thresh:
        triggers.append("BB_PCTB")

    # Trigger E: Volume climax ratio > 2.0
    if not np.isnan(vcr_val) and vcr_val > config.pb_v2_vol_climax_thresh:
        triggers.append("VOL_CLIMAX")

    # Trigger F: Relative strength > 1.02 OR ROC(5) < -3%
    if not np.isnan(rs_val) and rs_val > config.pb_v2_rs_ratio_thresh:
        triggers.append("RS_STRONG")
    elif not np.isnan(roc5_val) and roc5_val < config.pb_v2_roc_thresh:
        triggers.append("ROC5_DROP")

    if not triggers:
        return 0.0, []

    trend_tier = compute_trend_tier(symbol, indicators, config)
    is_down_day = len(symbol.daily_bars) >= 2 and symbol.daily_bars[-1].close < symbol.daily_bars[-2].close
    sma50 = indicators.get("sma50")
    sma50_now = float(sma50[-1]) if sma50 is not None and len(sma50) and not np.isnan(sma50[-1]) else 0.0
    sma50_prior = float(sma50[-11]) if sma50 is not None and len(sma50) >= 11 and not np.isnan(sma50[-11]) else sma50_now
    close_now = float(symbol.daily_bars[-1].close)
    sma_dist_pct = (close_now - sma50_now) / sma50_now * 100.0 if sma50_now > 0 else 0.0
    high_quality_triggers = {"RSI2", "BB_PCTB", "VOL_CLIMAX"}
    trigger_tier = "HIGH" if any(name in high_quality_triggers for name in triggers) else "MEDIUM"
    flow_window = symbol.flow_proxy_history[-10:]
    persistence = sum(value > 0 for value in flow_window) / len(flow_window) if flow_window else 0.0
    bundle = score_daily_pullback_context(
        trend_tier=trend_tier,
        rsi2=rsi2_val,
        rsi5=rsi5_val,
        cdd=cdd_val,
        depth_atr=depth_val,
        bb_pctb=bb_val,
        volume_climax=vcr_val,
        is_down_day=is_down_day,
        rs_ratio=rs_val,
        roc5=roc5_val,
        sma_slope_positive=sma50_now > sma50_prior,
        sma_dist_pct=sma_dist_pct,
        trigger_tier=trigger_tier,
        n_triggers=len(triggers),
        regime_tier=regime_tier,
        persistence=persistence,
    )
    return min(100.0, float(bundle["score"])), triggers


def compute_trigger_tier(score: float, config: StrategySettings) -> tuple[str, float]:
    """Map daily signal score to tier and sizing multiplier.

    Returns (tier_name, sizing_mult).
    """
    if score >= config.pb_v2_signal_floor:
        return "PREMIUM", config.pb_v2_sizing_premium
    if score >= 60.0:
        return "STANDARD", config.pb_v2_sizing_standard
    if score >= 45.0:
        return "REDUCED", config.pb_v2_sizing_reduced
    if score >= 30.0:
        return "MINIMUM", config.pb_v2_sizing_minimum
    return "SKIP", 0.0


def compute_trend_tier(
    symbol: ResearchSymbol,
    indicators: dict[str, np.ndarray],
    config: StrategySettings,
) -> str:
    """Classify trend tier based on SMA50/SMA200 position.

    STRONG: price above SMA50
    SECULAR: price below SMA50 but above SMA200
    EXCLUDED: price below both
    """
    if not symbol.daily_bars:
        return "EXCLUDED"
    close = symbol.daily_bars[-1].close

    sma50_val = float("nan")
    sma200_val = float("nan")
    if "sma50" in indicators and len(indicators["sma50"]) > 0:
        sma50_val = float(indicators["sma50"][-1])
    if "sma200" in indicators and len(indicators["sma200"]) > 0:
        sma200_val = float(indicators["sma200"][-1])

    if np.isnan(sma50_val):
        return "EXCLUDED"

    if close > sma50_val:
        return "STRONG"
    if not np.isnan(sma200_val) and close > sma200_val and sma50_val > sma200_val and config.pb_v2_allow_secular:
        return "SECULAR"
    return "EXCLUDED"


# ---------------------------------------------------------------------------
# Daily watchlist builder
# ---------------------------------------------------------------------------

def build_daily_watchlist(
    universe: dict[str, ResearchSymbol],
    indicators_cache: dict[str, dict[str, np.ndarray]],
    config: StrategySettings,
    *,
    regime_tier: str = "B",
) -> list[tuple[str, float, list[str]]]:
    """Score all symbols, filter by minimum score, rank.

    Returns list of (symbol_name, score, trigger_list) sorted by score desc.
    """
    candidates: list[tuple[str, float, list[str]]] = []
    for sym_name, symbol in universe.items():
        ind = indicators_cache.get(sym_name, {})
        if not ind:
            continue
        score, triggers = score_daily_candidate(symbol, ind, config, regime_tier=regime_tier)
        minimum_score = config.pb_v2_signal_floor if config.pb_v2_enabled else config.pb_daily_signal_min_score
        if score > 0.0 and score >= minimum_score:
            candidates.append((sym_name, score, triggers))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Enforce minimum candidate floor
    if len(candidates) < config.pb_min_candidates_day and len(candidates) > 0:
        pass  # keep what we have, don't pad

    return candidates


# ---------------------------------------------------------------------------
# Intraday entry score bundle (port of research _entry_score_bundle)
# ---------------------------------------------------------------------------

def _clip01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _peak_score(value: float, *, target: float, width: float) -> float:
    """Bell-curve transform centered on target with given width."""
    width = max(float(width), 1e-6)
    return _clip01(1.0 - abs(float(value) - float(target)) / width)


def compute_entry_score_bundle(
    bar: Bar,
    daily_signal_score: float,
    session_vwap: float | None,
    reclaim_level: float,
    stop_level: float,
    daily_atr: float,
    volume_ratio: float,
    ready_min_volume_ratio: float,
    micropressure: str,
    rescue_candidate: bool,
    route_family: str,
    flush_bar_idx: int,
    bar_idx: int,
    config: StrategySettings,
) -> tuple[float, dict[str, float]]:
    """Compute intraday entry quality score (0-100).

    Faithful port of research engine _entry_score_bundle() with
    meanrev_sweetspot_v1 weights and peak_score bell-curve transforms.
    Returns (total_score, components_dict).
    """
    # --- Raw input scores (same as research engine) ---
    daily_signal = _clip01(daily_signal_score / 100.0)

    reclaim_score = 0.0
    if stop_level > 0 and bar.close > reclaim_level:
        reclaim_score = min(
            max((bar.close - reclaim_level) / max(bar.close - stop_level, 0.01), 0.0),
            1.5,
        ) / 1.5

    vol_norm = min(
        max(volume_ratio / max(ready_min_volume_ratio, 0.25), 0.0), 1.25,
    ) / 1.25

    vwap = session_vwap or bar.close
    vwap_score = 0.0
    if daily_atr > 0:
        vwap_score = _clip01((bar.close - vwap) / max(daily_atr * 0.75, 0.01))

    cpr_score = _clip01(bar.cpr)

    reclaim_bars = max(bar_idx - flush_bar_idx + 1, 1)
    speed_score = _clip01(1.0 - (reclaim_bars - 1) / 8.0)

    # --- Route-specific peak_score transforms ---
    is_opening = route_family == "OPENING_RECLAIM"
    reclaim_target = 0.55 if is_opening else 0.45
    vwap_target = 0.28 if is_opening else 0.20
    cpr_target = 0.68 if is_opening else 0.62

    reclaim_component = _peak_score(reclaim_score, target=reclaim_target, width=0.45)
    vwap_component = _peak_score(vwap_score, target=vwap_target, width=0.28)
    cpr_component = _peak_score(cpr_score, target=cpr_target, width=0.28)

    # Extension penalty (overextended entries)
    extension_penalty = 0.0
    if reclaim_score > 0.85:
        extension_penalty -= _clip01((reclaim_score - 0.85) / 0.15) * 4.0
    if vwap_score > 0.60:
        extension_penalty -= _clip01((vwap_score - 0.60) / 0.40) * 6.0
    if cpr_score > 0.85:
        extension_penalty -= _clip01((cpr_score - 0.85) / 0.15) * 6.0

    # --- meanrev_sweetspot_v1 weights ---
    daily_weight = 54.0
    reclaim_weight = 8.0
    volume_weight = 12.0
    vwap_weight = 5.0
    cpr_weight = 6.0
    speed_weight = 8.0
    context_low = -4.0
    context_high = 2.0
    distribute_penalty = -12.0
    neutral_penalty = -5.0
    weak_vwap_penalty_value = -10.0
    rescue_penalty_value = -8.0

    # Context (30m bonus not available in live -- defaults to 0)
    context_adjust = max(context_low, min(context_high, 0.0))
    micro_penalty = (
        distribute_penalty if micropressure == "DISTRIBUTE"
        else neutral_penalty if micropressure == "NEUTRAL"
        else 0.0
    )
    weak_vwap_pen = weak_vwap_penalty_value if bar.close < vwap else 0.0
    rescue_pen = rescue_penalty_value if rescue_candidate else 0.0

    route_flag = 0.0 if is_opening else 1.0
    total = (
        daily_signal * daily_weight
        + reclaim_component * reclaim_weight
        + vol_norm * volume_weight
        + vwap_component * vwap_weight
        + cpr_component * cpr_weight
        + speed_score * speed_weight
        + context_adjust
        + micro_penalty
        + weak_vwap_pen
        + rescue_pen
        + extension_penalty
    )

    components = {
        "route_family": route_flag,
        "daily_signal": float(daily_signal * daily_weight),
        "reclaim": float(reclaim_component * reclaim_weight),
        "volume": float(vol_norm * volume_weight),
        "vwap_hold": float(vwap_component * vwap_weight),
        "cpr": float(cpr_component * cpr_weight),
        "speed": float(speed_score * speed_weight),
        "context": float(context_adjust + micro_penalty + weak_vwap_pen + rescue_pen),
        "extension": float(extension_penalty),
        "score": float(max(0.0, min(100.0, total))),
    }
    return float(components["score"]), components


# ---------------------------------------------------------------------------
# Micropressure proxy (used by backtest engines when tick data unavailable)
# ---------------------------------------------------------------------------

def compute_micropressure_proxy(
    bar_5m: Bar,
    expected_volume: float,
    median20_volume: float,
    reclaim_level: float,
) -> str:
    """Proxy micropressure from 5m bar when tick data unavailable."""
    if expected_volume > 0:
        surge = bar_5m.volume / expected_volume
    else:
        surge = 0.0
    bar_range = bar_5m.high - bar_5m.low
    cpr = (bar_5m.close - bar_5m.low) / bar_range if bar_range > 0 else 0.0
    bullish_close = bar_5m.close > bar_5m.open

    # Path 1: volume surge + reclaim + strong close
    if surge >= 1.3 and bar_5m.close >= reclaim_level and cpr >= 0.60 and bullish_close:
        return "ACCUMULATE"

    # Path 2: strong CPR + bullish + median volume confirmation
    if cpr >= 0.75 and bullish_close and bar_5m.volume >= 1.3 * median20_volume:
        return "ACCUMULATE"

    return "NEUTRAL"
