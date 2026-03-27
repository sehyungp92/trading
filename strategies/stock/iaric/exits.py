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
    *,
    settings: StrategySettings | None = None,
) -> tuple[bool, str]:
    if item.regime_tier == "A":
        pass  # always allowed
    elif item.regime_tier == "B" and settings is not None and settings.regime_b_carry_mult > 0:
        pass  # Regime B carry enabled
    else:
        return False, "regime_not_eligible"
    if flow_reversal_flag:
        return False, "flow_reversal_flag"
    if settings is not None and settings.max_carry_days > 0:
        days_held = (datetime.now(ET).date() - position.entry_time.astimezone(ET).date()).days
        if days_held >= settings.max_carry_days:
            return False, "max_carry_days"
    if item.sponsorship_state != "STRONG":
        return False, "sponsorship_not_strong"
    if item.earnings_risk_flag or item.blacklist_flag:
        return False, "event_risk"
    if market.last_price is None:
        return False, "missing_market_data"
    if market.last_price <= position.entry_price:
        return False, "not_in_profit"
    if settings is not None and settings.min_carry_r > 0:
        one_r = position.initial_risk_per_share
        if one_r > 0:
            cur_r = (market.last_price - position.entry_price) / one_r
            if cur_r <= settings.min_carry_r:
                return False, "below_min_carry_r"
    if settings is not None and settings.carry_top_quartile:
        session_high = market.session_high if market.session_high is not None else market.last_price
        session_low = market.session_low if market.session_low is not None else market.last_price
        daily_range = max(session_high - session_low, 1e-9)
        close_pct = (market.last_price - session_low) / daily_range
        if close_pct < 0.75:
            return False, "close_not_in_top_quartile"
    return True, "eligible"


def flow_reversal_exit_due(flow_reversal_flag: bool, now: datetime, opened_today: bool) -> bool:
    if not flow_reversal_flag or opened_today:
        return False
    return now.astimezone(ET).time() >= time(9, 31)
