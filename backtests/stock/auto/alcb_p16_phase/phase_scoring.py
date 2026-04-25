from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


_ET = ZoneInfo("America/New_York")
_BUCKET_1000_START = time(10, 0)
_BUCKET_1000_END = time(10, 30)
_LATE_BUCKET_START = time(11, 30)
_NEUTRAL_STRUCTURAL_SCORE = 0.50


PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "expected_total_r": 0.26,
        "net_profit": 0.18,
        "profit_factor": 0.18,
        "expectancy_dollar": 0.14,
        "trades_per_month": 0.12,
        "extended_avwap_inverse": 0.12,
    },
    2: {
        "expected_total_r": 0.24,
        "net_profit": 0.18,
        "profit_factor": 0.16,
        "trades_per_month": 0.16,
        "bar9_inverse": 0.16,
        "late_entry_quality": 0.10,
    },
    3: {
        "expected_total_r": 0.24,
        "net_profit": 0.18,
        "profit_factor": 0.18,
        "expectancy_dollar": 0.12,
        "pdh_avg_r": 0.18,
        "trades_per_month": 0.10,
    },
    4: {
        "expected_total_r": 0.22,
        "net_profit": 0.18,
        "profit_factor": 0.16,
        "expectancy_dollar": 0.14,
        "score_monotonicity": 0.20,
        "inv_dd": 0.10,
    },
    5: {
        "expected_total_r": 0.28,
        "net_profit": 0.20,
        "profit_factor": 0.16,
        "expectancy_dollar": 0.14,
        "trades_per_month": 0.12,
        "inv_dd": 0.10,
    },
}

NORMALIZATION_RANGES: dict[str, tuple[float, float]] = {
    "expected_total_r": (95.0, 115.0),
    "net_profit": (7800.0, 9600.0),
    "trades_per_month": (20.0, 25.5),
    "expectancy": (0.15, 0.21),
    "expectancy_dollar": (13.0, 16.5),
    "profit_factor": (1.60, 1.82),
    "mfe_capture_efficiency": (0.72, 0.82),
    "profit_protection": (0.70, 0.82),
    "inv_dd": (0.60, 0.78),
    "entry_quality": (0.70, 0.95),
    "extended_avwap_inverse": (0.45, 0.75),
    "bar9_inverse": (0.35, 0.80),
    "late_entry_quality": (0.45, 0.75),
    "pdh_avg_r": (0.08, 0.18),
    "score_monotonicity": (0.30, 0.80),
}


def merge_alcb_metrics(performance_metrics: Any, trades: list[Any]) -> dict[str, float]:
    metrics = asdict(performance_metrics) if is_dataclass(performance_metrics) else dict(performance_metrics)
    metrics.update(compute_alcb_phase_metrics(trades))
    return enrich_alcb_phase_metrics(metrics)


def score_alcb_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    weights = dict(PHASE_SCORING_WEIGHTS.get(phase, {}))
    if scoring_weights:
        weights.update(scoring_weights)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    enriched = enrich_alcb_phase_metrics(metrics)
    return sum(
        (weight / total_weight) * _normalize(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


def enrich_alcb_phase_metrics(metrics: dict[str, float]) -> dict[str, float]:
    enriched = dict(metrics)
    max_dd = float(metrics.get("max_drawdown_pct", 0.0))
    total_trades = float(metrics.get("total_trades", 0.0))
    expectancy_r = float(metrics.get("expectancy", 0.0))
    short_hold_total_r = float(metrics.get("short_hold_total_r", 0.0))
    flow_reversal_short_total_r = float(metrics.get("flow_reversal_short_total_r", 0.0))
    early_1000_short_total_r = float(metrics.get("early_1000_short_total_r", 0.0))
    long_hold_total_r = float(metrics.get("long_hold_total_r", 0.0))
    carry_total_r = float(metrics.get("carry_total_r", 0.0))
    positive_total_r = float(metrics.get("positive_total_r", 0.0))

    enriched["inv_dd"] = _clip01(1.0 - max_dd / 0.12)
    enriched["expected_total_r"] = expectancy_r * total_trades
    enriched["short_hold_drag_inverse"] = _clip01(1.0 - abs(min(short_hold_total_r, 0.0)) / 180.0)
    enriched["flow_reversal_short_inverse"] = _clip01(1.0 - abs(min(flow_reversal_short_total_r, 0.0)) / 140.0)
    enriched["early_1000_drag_inverse"] = _clip01(1.0 - abs(min(early_1000_short_total_r, 0.0)) / 150.0)
    enriched["long_hold_capture"] = float(np.mean([
        _clip01(max(long_hold_total_r, 0.0) / 80.0),
        _clip01(max(long_hold_total_r, 0.0) / max(positive_total_r, 1.0)),
    ]))
    enriched["carry_capture"] = float(np.mean([
        _clip01(max(carry_total_r, 0.0) / 12.0),
        _clip01(max(carry_total_r, 0.0) / max(positive_total_r, 1.0)),
    ]))
    enriched["profit_protection"] = float(np.mean([
        enriched["short_hold_drag_inverse"],
        enriched["flow_reversal_short_inverse"],
        enriched["early_1000_drag_inverse"],
    ]))
    return enriched


def compute_alcb_phase_metrics(trades: list[Any]) -> dict[str, float]:
    total = len(trades)
    global_avg_r = _avg_r(trades)

    short_hold = [trade for trade in trades if _hold_bars(trade) <= 6]
    short_flow = [
        trade for trade in short_hold
        if _normalize_exit(getattr(trade, "exit_reason", "")) == "FLOW_REVERSAL"
    ]
    early_1000_short = [trade for trade in short_hold if _in_1000_bucket(trade)]
    long_hold = [trade for trade in trades if _hold_bars(trade) > 24]
    carry_trades = [trade for trade in trades if _is_carry_trade(trade)]
    or_trades = [trade for trade in trades if _entry_type(trade) in {"OR_BREAKOUT", "OR_RECLAIM", "AVWAP_RECLAIM"}]
    combined_trades = [trade for trade in trades if _entry_type(trade).startswith("COMBINED")]
    pdh_trades = [trade for trade in trades if _entry_type(trade) in {"PDH_BREAKOUT", "PDH_RECLAIM"}]
    tight_or = [trade for trade in trades if _or_width_pct(trade) < 0.2]
    mid_rvol = [trade for trade in trades if 1.5 <= _rvol(trade) < 2.0]
    strong_rvol = [trade for trade in trades if _rvol(trade) >= 2.0]
    high_rvol = [trade for trade in trades if _rvol(trade) >= 3.0]

    slight_premium = [
        trade for trade in trades
        if 0.0 < _avwap_premium_pct(trade) <= 0.005
    ]
    extended_premium = [
        trade for trade in trades
        if _avwap_premium_pct(trade) > 0.005
    ]
    bar9_trades = [trade for trade in trades if _entry_bar_number(trade) == 9]
    late_trades = [trade for trade in trades if _is_late_trade(trade)]

    score_4 = [trade for trade in trades if _momentum_score(trade) == 4]
    score_5 = [trade for trade in trades if _momentum_score(trade) == 5]
    score_6 = [trade for trade in trades if _momentum_score(trade) == 6]
    score_7 = [trade for trade in trades if _momentum_score(trade) >= 7]

    or_avg_r = _avg_r(or_trades)
    combined_avg_r = _avg_r(combined_trades)
    pdh_avg_r = _avg_r(pdh_trades)
    strong_rvol_avg_r = _avg_r(strong_rvol)
    high_rvol_avg_r = _avg_r(high_rvol)

    tight_or_inverse = _clip01(1.0 - abs(min(_total_r(tight_or), 0.0)) / 45.0)
    mid_rvol_inverse = _clip01(1.0 - abs(min(_total_r(mid_rvol), 0.0)) / 40.0)
    combined_inverse = _clip01(1.0 - abs(min(combined_avg_r, 0.0)) / 0.12)
    pdh_inverse = _clip01(1.0 - abs(min(pdh_avg_r, 0.0)) / 0.12)
    or_edge = _clip01((or_avg_r + 0.02) / 0.10)
    strong_rvol_edge = _clip01((strong_rvol_avg_r + 0.02) / 0.10)
    high_rvol_edge = float(np.mean([
        strong_rvol_edge,
        _clip01((high_rvol_avg_r + 0.02) / 0.10),
        mid_rvol_inverse,
    ]))
    entry_quality = float(np.mean([
        or_edge,
        combined_inverse,
        pdh_inverse,
        tight_or_inverse,
        mid_rvol_inverse,
        high_rvol_edge,
    ]))

    positive_total_r = float(sum(max(float(trade.r_multiple), 0.0) for trade in trades))

    winners = [t for t in trades if float(t.r_multiple) > 0]
    if winners:
        total_winner_r = sum(float(t.r_multiple) for t in winners)
        total_winner_mfe = sum(_meta_float(t, "mfe_r", 0.0) for t in winners)
        mfe_capture_eff = total_winner_r / total_winner_mfe if total_winner_mfe > 0 else 0.0
    else:
        mfe_capture_eff = 0.0

    slight_premium_avg_r = _shrunk_avg_r(slight_premium, global_avg_r, prior_strength=12.0)
    extended_premium_avg_r = _shrunk_avg_r(extended_premium, global_avg_r, prior_strength=18.0)
    extended_raw = _clip01(1.0 - max(0.0, slight_premium_avg_r - extended_premium_avg_r) / 0.18)
    extended_avwap_inverse = _with_neutral_sample_floor(
        extended_raw,
        sample_size=len(extended_premium),
        full_weight_at=24,
    )

    bar9_avg_r = _shrunk_avg_r(bar9_trades, global_avg_r, prior_strength=12.0)
    bar9_raw = _clip01((bar9_avg_r + 0.08) / 0.18)
    bar9_inverse = _with_neutral_sample_floor(
        bar9_raw,
        sample_size=len(bar9_trades),
        full_weight_at=24,
    )

    late_avg_r = _shrunk_avg_r(late_trades, global_avg_r, prior_strength=14.0)
    late_raw = _clip01((late_avg_r + 0.02) / 0.20)
    late_entry_quality = _with_neutral_sample_floor(
        late_raw,
        sample_size=len(late_trades),
        full_weight_at=40,
    )

    score_bucket_avgs = {
        4: _shrunk_avg_r(score_4, global_avg_r, prior_strength=12.0),
        5: _shrunk_avg_r(score_5, global_avg_r, prior_strength=12.0),
        6: _shrunk_avg_r(score_6, global_avg_r, prior_strength=12.0),
        7: _shrunk_avg_r(score_7, global_avg_r, prior_strength=12.0),
    }
    score_bucket_counts = {
        4: len(score_4),
        5: len(score_5),
        6: len(score_6),
        7: len(score_7),
    }
    score_monotonicity_components = []
    for lower, higher in ((4, 5), (5, 6), (6, 7)):
        gap = score_bucket_avgs[higher] - score_bucket_avgs[lower]
        raw = _clip01((gap + 0.03) / 0.12)
        score_monotonicity_components.append(
            _with_neutral_sample_floor(
                raw,
                sample_size=min(score_bucket_counts[lower], score_bucket_counts[higher]),
                full_weight_at=35,
            )
        )
    score_monotonicity = float(np.mean(score_monotonicity_components)) if score_monotonicity_components else _NEUTRAL_STRUCTURAL_SCORE

    return {
        "short_hold_total_r": _total_r(short_hold),
        "flow_reversal_short_total_r": _total_r(short_flow),
        "early_1000_short_total_r": _total_r(early_1000_short),
        "long_hold_total_r": _total_r(long_hold),
        "carry_total_r": _total_r(carry_trades),
        "positive_total_r": positive_total_r,
        "or_avg_r": or_avg_r,
        "combined_avg_r": combined_avg_r,
        "pdh_avg_r": pdh_avg_r,
        "strong_rvol_avg_r": strong_rvol_avg_r,
        "high_rvol_avg_r": high_rvol_avg_r,
        "tight_or_inverse": tight_or_inverse,
        "mid_rvol_inverse": mid_rvol_inverse,
        "combined_inverse": combined_inverse,
        "pdh_inverse": pdh_inverse,
        "or_edge": or_edge,
        "high_rvol_edge": high_rvol_edge,
        "entry_quality": entry_quality,
        "short_hold_share": _share(len(short_hold), total),
        "long_hold_share": _share(len(long_hold), total),
        "carry_share": _share(len(carry_trades), total),
        "mfe_capture_efficiency": mfe_capture_eff,
        "slight_premium_avg_r": slight_premium_avg_r,
        "extended_premium_avg_r": extended_premium_avg_r,
        "extended_avwap_inverse": extended_avwap_inverse,
        "bar9_avg_r": bar9_avg_r,
        "bar9_inverse": bar9_inverse,
        "late_avg_r": late_avg_r,
        "late_entry_quality": late_entry_quality,
        "score_4_avg_r": score_bucket_avgs[4],
        "score_5_avg_r": score_bucket_avgs[5],
        "score_6_avg_r": score_bucket_avgs[6],
        "score_7_avg_r": score_bucket_avgs[7],
        "score_monotonicity": score_monotonicity,
    }


def _entry_type(trade: Any) -> str:
    if getattr(trade, "entry_type", None):
        return str(trade.entry_type)
    metadata = getattr(trade, "metadata", None) or {}
    return str(metadata.get("entry_type", "UNKNOWN"))


def _hold_bars(trade: Any) -> int:
    try:
        return int(getattr(trade, "hold_bars", 0))
    except (TypeError, ValueError):
        return 0


def _rvol(trade: Any) -> float:
    return _meta_float(trade, "rvol_at_entry", 0.0)


def _or_width_pct(trade: Any) -> float:
    or_high = _meta_float(trade, "or_high", 0.0)
    or_low = _meta_float(trade, "or_low", 0.0)
    if or_high <= 0:
        return 0.0
    return ((or_high - or_low) / or_high) * 100.0


def _meta_float(trade: Any, key: str, default: float = 0.0) -> float:
    metadata = getattr(trade, "metadata", None) or {}
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _signal_bar_index(trade: Any) -> int:
    metadata = getattr(trade, "metadata", None) or {}
    raw_value = metadata.get("signal_bar_index", getattr(trade, "signal_bar_index", -1))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def _momentum_score(trade: Any) -> int:
    metadata = getattr(trade, "metadata", None) or {}
    raw_value = metadata.get("momentum_score", getattr(trade, "momentum_score", 0))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0


def _avwap_premium_pct(trade: Any) -> float:
    avwap = _meta_float(trade, "avwap_at_entry", 0.0)
    entry_price = float(getattr(trade, "entry_price", 0.0))
    if avwap <= 0 or entry_price <= 0:
        return 0.0
    return (entry_price - avwap) / avwap


def _avg_r(trades: list[Any]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([float(trade.r_multiple) for trade in trades]))


def _shrunk_avg_r(trades: list[Any], prior_mean: float, prior_strength: float = 10.0) -> float:
    if not trades:
        return float(prior_mean)
    sample_total = sum(float(trade.r_multiple) for trade in trades)
    return float((sample_total + prior_mean * prior_strength) / (len(trades) + prior_strength))


def _total_r(trades: list[Any]) -> float:
    return float(sum(float(trade.r_multiple) for trade in trades))


def _share(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize(metric_name: str, metrics: dict[str, float]) -> float:
    value = float(metrics.get(metric_name, 0.0))
    lo, hi = NORMALIZATION_RANGES.get(metric_name, (0.0, 1.0))
    if hi <= lo:
        return 0.0
    return _clip01((value - lo) / (hi - lo))


def _normalize_exit(exit_reason: str | None) -> str:
    return str(exit_reason or "").strip().upper()


def _entry_time_et(trade: Any):
    fill_time = getattr(trade, "fill_time", None) or getattr(trade, "entry_time", None)
    if fill_time is None:
        metadata = getattr(trade, "metadata", None) or {}
        fill_time = metadata.get("fill_time")
    if fill_time is None:
        return None
    try:
        return fill_time.astimezone(_ET)
    except Exception:
        return fill_time


def _entry_bar_number(trade: Any) -> int:
    entry_dt = _entry_time_et(trade)
    if entry_dt is None:
        return 0
    minutes_from_open = (entry_dt.hour * 60 + entry_dt.minute) - 570
    if minutes_from_open < 0:
        return 0
    return minutes_from_open // 5 + 1


def _in_1000_bucket(trade: Any) -> bool:
    entry_dt = _entry_time_et(trade)
    if entry_dt is None:
        return False
    entry_t = entry_dt.timetz().replace(tzinfo=None)
    return _BUCKET_1000_START <= entry_t < _BUCKET_1000_END


def _is_late_trade(trade: Any) -> bool:
    entry_dt = _entry_time_et(trade)
    if entry_dt is None:
        return False
    entry_t = entry_dt.timetz().replace(tzinfo=None)
    return entry_t >= _LATE_BUCKET_START


def _is_carry_trade(trade: Any) -> bool:
    entry_time = getattr(trade, "entry_time", None)
    exit_time = getattr(trade, "exit_time", None)
    if entry_time is None or exit_time is None:
        return False
    return exit_time.date() > entry_time.date()


def _with_neutral_sample_floor(value: float, *, sample_size: int, full_weight_at: int) -> float:
    if full_weight_at <= 0:
        return _clip01(value)
    weight = _clip01(float(sample_size) / float(full_weight_at))
    return float((_NEUTRAL_STRUCTURAL_SCORE * (1.0 - weight)) + (_clip01(value) * weight))


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
