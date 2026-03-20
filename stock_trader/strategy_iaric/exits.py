"""Exit helpers for IARIC."""

from __future__ import annotations

from datetime import datetime, time

from .config import ET, StrategySettings
from .models import Bar, MarketSnapshot, PositionState, WatchlistItem


def classify_trade(market: MarketSnapshot, position: PositionState) -> str:
    if market.last_price is None or not market.bars_5m:
        return position.setup_tag
    last_bar = market.bars_5m[-1]
    prior_high = max((bar.high for bar in list(market.bars_5m)[-4:-1]), default=position.entry_price)
    if last_bar.close > prior_high and last_bar.cpr >= 0.6:
        return "MOMENTUM_CONTINUATION"
    if last_bar.close > position.entry_price and last_bar.high < prior_high:
        return "MEAN_REVERSION_BOUNCE"
    if market.avwap_live is not None and market.last_price >= market.avwap_live:
        return "FLOW_DRIVEN_GRIND"
    return "FAILED"


def regime_adjusted_partial(regime_multiplier: float, settings: StrategySettings) -> tuple[float, float]:
    t = min(1.0, max(0.0, (regime_multiplier - 0.35) / 0.65))
    r_trigger = settings.partial_r_min + t * (settings.partial_r_max - settings.partial_r_min)
    fraction = settings.partial_frac_max - t * (settings.partial_frac_max - settings.partial_frac_min)
    return round(r_trigger, 3), round(fraction, 3)


def should_take_partial(
    position: PositionState,
    market_price: float,
    settings: StrategySettings,
    regime_multiplier: float | None = None,
) -> tuple[bool, float]:
    if position.partial_taken:
        return False, settings.partial_exit_fraction
    if regime_multiplier is not None:
        r_trigger, fraction = regime_adjusted_partial(regime_multiplier, settings)
    else:
        r_trigger = settings.partial_r_multiple
        fraction = settings.partial_exit_fraction
    one_r = position.initial_risk_per_share
    triggered = market_price >= position.entry_price + (r_trigger * one_r)
    return triggered, fraction


def should_exit_for_time_stop(position: PositionState, now: datetime, market_price: float) -> bool:
    if position.time_stop_deadline is None:
        return False
    return now >= position.time_stop_deadline and market_price <= position.entry_price


def should_exit_for_avwap_breakdown(bar_30m: Bar, avwap_live: float, avg_30m_volume: float, settings: StrategySettings) -> bool:
    return (
        bar_30m.close < avwap_live * (1 - settings.avwap_breakdown_pct)
        and bar_30m.volume > settings.avwap_breakdown_volume_mult * max(avg_30m_volume, 1.0)
    )


def carry_eligible(
    item: WatchlistItem,
    market: MarketSnapshot,
    position: PositionState,
    flow_reversal_flag: bool = False,
) -> tuple[bool, str]:
    if item.regime_tier != "A":
        return False, "regime_not_a"
    if flow_reversal_flag:
        return False, "flow_reversal_flag"
    if position.setup_tag not in {"MOMENTUM_CONTINUATION", "FLOW_DRIVEN_GRIND"}:
        return False, "setup_not_carry"
    if market.last_30m_bar is None or market.avwap_live is None or market.last_price is None:
        return False, "missing_close_context"
    if market.last_price <= position.entry_price:
        return False, "not_in_profit"
    if market.last_30m_bar.close < market.avwap_live:
        return False, "close_below_avwap"
    session_high = market.session_high if market.session_high is not None else market.last_30m_bar.high
    session_low = market.session_low if market.session_low is not None else market.last_30m_bar.low
    daily_range = max(session_high - session_low, 1e-9)
    close_pct = (market.last_price - session_low) / daily_range
    if close_pct < 0.75:
        return False, "close_not_in_top_quartile"
    if item.sponsorship_state != "STRONG":
        return False, "sponsorship_not_strong"
    if item.earnings_risk_flag or item.blacklist_flag:
        return False, "event_risk"
    return True, "eligible"


def flow_reversal_exit_due(flow_reversal_flag: bool, now: datetime, opened_today: bool) -> bool:
    if not flow_reversal_flag or opened_today:
        return False
    return now.astimezone(ET).time() >= time(9, 31)
