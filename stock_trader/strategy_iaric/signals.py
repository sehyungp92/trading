"""Intraday signal and FSM helpers for IARIC."""

from __future__ import annotations

from datetime import datetime

from .config import ET, StrategySettings
from .models import Bar, MarketSnapshot, SymbolIntradayState, WatchlistItem


def detect_setup(item: WatchlistItem, market: MarketSnapshot, last_1m_bar: Bar, settings: StrategySettings) -> str | None:
    in_avwap_band = (
        item.avwap_band_lower <= last_1m_bar.low <= item.avwap_band_upper
        or item.avwap_band_lower <= last_1m_bar.close <= item.avwap_band_upper
    )
    if not in_avwap_band:
        return None
    if market.drop_from_hod >= settings.panic_flush_drop_pct and market.minutes_since_hod <= settings.panic_flush_minutes:
        return "PANIC_FLUSH"
    if market.drop_from_hod >= settings.drift_exhaustion_drop_pct and market.minutes_since_hod >= settings.drift_exhaustion_minutes:
        return "DRIFT_EXHAUSTION"
    return None


def lock_setup(sym: SymbolIntradayState, bar_1m: Bar, atr_5m_pct: float, reason: str = "") -> None:
    offset = max(0.003, 0.25 * atr_5m_pct)
    sym.setup_low = bar_1m.low
    sym.reclaim_level = sym.setup_low * (1 + offset)
    sym.stop_level = sym.setup_low * (1 - offset)
    sym.setup_time = bar_1m.end_time
    sym.acceptance_count = 0
    sym.fsm_state = "SETUP_DETECTED"
    sym.last_transition_reason = reason or "setup_locked"


def compute_location_grade(item: WatchlistItem, market: MarketSnapshot) -> str:
    if market.session_vwap is None or market.avwap_live is None or market.last_price is None:
        return "B"
    near_avwap = (
        item.avwap_band_lower <= market.avwap_live <= item.avwap_band_upper
        or item.avwap_band_lower <= market.last_price <= item.avwap_band_upper
    )
    discount_pct = (market.session_vwap - market.last_price) / max(market.session_vwap, 1e-9)
    if near_avwap and 0.015 <= discount_pct <= 0.045:
        return "A"
    if near_avwap:
        return "B"
    if discount_pct >= 0.015:
        return "C"
    return "B"


def compute_required_acceptance(
    item: WatchlistItem,
    sym: SymbolIntradayState,
    now: datetime,
    settings: StrategySettings,
    market_wide_institutional_selling: bool = False,
) -> tuple[int, list[str]]:
    req = settings.acceptance_base_closes
    adders: list[str] = []
    if sym.micropressure_mode == "PROXY":
        req += 1
        adders.append("proxy_mode")
    if item.sponsorship_state == "STALE":
        req += 1
        adders.append("sponsorship_stale")
    if item.regime_tier == "B":
        req += 1
        adders.append("regime_b")
    if now.astimezone(ET).time().hour >= 14:
        req += 1
        adders.append("late_day")
    if sym.location_grade != "A":
        req += 1
        adders.append("location_non_a")
    if sym.flowproxy_signal == "UNAVAILABLE":
        req += 1
        adders.append("flow_unavailable")
    if market_wide_institutional_selling:
        req += 1
        adders.append("market_selling")
    return req, adders


def update_acceptance(sym: SymbolIntradayState, bar_5m: Bar) -> None:
    if sym.reclaim_level is None:
        return
    if bar_5m.close >= sym.reclaim_level:
        sym.acceptance_count += 1
        sym.last_5m_bar_time = bar_5m.end_time


def compute_micropressure_from_ticks(ticks_window) -> str:
    uptick_value = sum(value for _, value in ticks_window if value > 0)
    downtick_value = abs(sum(value for _, value in ticks_window if value < 0))
    if uptick_value > 1.5 * max(downtick_value, 1.0):
        return "ACCUMULATE"
    if downtick_value > 1.5 * max(uptick_value, 1.0):
        return "DISTRIBUTE"
    return "NEUTRAL"


def compute_micropressure_proxy(bar_5m: Bar, expected_volume: float, median20_volume: float, reclaim_level: float) -> str:
    surge = bar_5m.volume / max(expected_volume, 1e-9)
    cpr = bar_5m.cpr
    if surge >= 1.3 and bar_5m.close >= reclaim_level and cpr >= 0.60 and bar_5m.close > bar_5m.open:
        return "ACCUMULATE"
    if cpr >= 0.75 and bar_5m.close > bar_5m.open and bar_5m.volume >= 1.3 * max(median20_volume, 1e-9):
        return "ACCUMULATE"
    return "NEUTRAL"


def compute_flowproxy_signal(flow_value: str | None) -> str:
    if not flow_value:
        return "UNAVAILABLE"
    normalized = flow_value.upper()
    if normalized in {"ACCUMULATE", "DISTRIBUTE", "NEUTRAL", "STALE"}:
        return normalized
    return "UNAVAILABLE"


def resolve_confidence(sym: SymbolIntradayState) -> str:
    if sym.sponsorship_signal == "DISTRIBUTE":
        return "RED"
    if sym.micropressure_signal == "DISTRIBUTE":
        return "RED"
    if sym.flowproxy_signal == "DISTRIBUTE":
        return "RED"

    positives = 0
    if sym.sponsorship_signal == "STRONG":
        positives += 1
    if sym.micropressure_signal == "ACCUMULATE":
        positives += 1
    if sym.flowproxy_signal == "ACCUMULATE":
        positives += 1

    if sym.flowproxy_signal != "UNAVAILABLE":
        return "GREEN" if positives >= 2 else "YELLOW"
    return "GREEN" if sym.sponsorship_signal == "STRONG" and sym.micropressure_signal == "ACCUMULATE" else "YELLOW"


def cooldown_expired(sym: SymbolIntradayState, now: datetime, settings: StrategySettings) -> bool:
    if sym.invalidated_at is None:
        return True
    return (now - sym.invalidated_at).total_seconds() >= settings.invalidation_cooldown_minutes * 60


def reset_setup_state(sym: SymbolIntradayState) -> None:
    sym.fsm_state = "IDLE"
    sym.setup_type = None
    sym.setup_low = None
    sym.reclaim_level = None
    sym.stop_level = None
    sym.setup_time = None
    sym.acceptance_count = 0
    sym.required_acceptance_count = 0
    sym.location_grade = None
    sym.confidence = None
    sym.last_transition_reason = "reset"


def alpha_step(
    item: WatchlistItem,
    sym: SymbolIntradayState,
    market: MarketSnapshot,
    bar_1m: Bar | None,
    bar_5m: Bar | None,
    now: datetime,
    atr_5m_pct: float,
    settings: StrategySettings,
    market_wide_institutional_selling: bool = False,
) -> tuple[str, list[str]]:
    if sym.stop_level is not None and market.last_price is not None and market.last_price <= sym.stop_level:
        sym.fsm_state = "INVALIDATED"
        sym.invalidated_at = now
        sym.last_transition_reason = "stop_breach"
        return "INVALIDATE", []

    if sym.setup_time is not None and sym.fsm_state in {"SETUP_DETECTED", "ACCEPTING"}:
        if (now - sym.setup_time).total_seconds() >= settings.setup_stale_minutes * 60:
            sym.fsm_state = "INVALIDATED"
            sym.invalidated_at = now
            sym.last_transition_reason = "setup_stale"
            return "INVALIDATE", []

    if sym.fsm_state == "IDLE":
        setup_type = detect_setup(item, market, bar_1m, settings) if bar_1m else None
        if setup_type:
            sym.setup_type = setup_type
            lock_setup(sym, bar_1m, atr_5m_pct, reason=setup_type)
            sym.location_grade = compute_location_grade(item, market)
            return "SETUP_DETECTED", []

    elif sym.fsm_state == "SETUP_DETECTED":
        reclaim_touched = False
        if sym.reclaim_level is not None:
            if bar_1m is not None and bar_1m.high >= sym.reclaim_level:
                reclaim_touched = True
            elif bar_5m is not None and bar_5m.high >= sym.reclaim_level:
                reclaim_touched = True
            elif market.last_price is not None and market.last_price >= sym.reclaim_level:
                reclaim_touched = True
        if reclaim_touched:
            sym.fsm_state = "ACCEPTING"
            required, adders = compute_required_acceptance(
                item=item,
                sym=sym,
                now=now,
                settings=settings,
                market_wide_institutional_selling=market_wide_institutional_selling,
            )
            sym.required_acceptance_count = required
            sym.last_transition_reason = "reclaim_hit"
            return "MOVE_TO_ACCEPTING", adders

    elif sym.fsm_state == "ACCEPTING":
        if bar_5m:
            update_acceptance(sym, bar_5m)
            sym.confidence = resolve_confidence(sym)
            if sym.acceptance_count >= sym.required_acceptance_count and sym.confidence != "RED":
                return "READY_TO_ENTER", []

    elif sym.fsm_state == "INVALIDATED":
        if cooldown_expired(sym, now, settings):
            reset_setup_state(sym)
            return "RESET_TO_IDLE", []

    elif sym.fsm_state == "IN_POSITION":
        return "MANAGE_POSITION", []

    return "NO_ACTION", []


def update_symbol_tier(item: WatchlistItem, sym: SymbolIntradayState, market: MarketSnapshot, settings: StrategySettings) -> str:
    if sym.in_position or sym.fsm_state in {"SETUP_DETECTED", "ACCEPTING", "IN_POSITION"}:
        return "HOT"
    if market.last_price is None:
        return "COLD"
    in_band = item.avwap_band_lower <= market.last_price <= item.avwap_band_upper
    if in_band or market.drop_from_hod >= settings.hot_drop_pct:
        return "HOT"
    if market.drop_from_hod >= settings.warm_drop_pct:
        return "WARM"
    if market.avwap_live is not None and abs(market.last_price - market.avwap_live) / max(market.avwap_live, 1e-9) <= settings.avwap_band_pct * 2:
        return "WARM"
    return "COLD"
