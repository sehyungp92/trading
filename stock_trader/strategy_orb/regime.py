"""Broad-market regime helpers."""

from __future__ import annotations

from collections import deque

from .config import StrategySettings
from .models import QuoteSnapshot, RegimeSnapshot, SymbolContext


def ewma(previous: float, sample: float, alpha: float) -> float:
    return (alpha * sample) + ((1.0 - alpha) * previous)


def flow_regime(spy_ewma: float, qqq_ewma: float) -> str:
    if spy_ewma > 0 and qqq_ewma > 0:
        return "strong_inflow"
    if spy_ewma < 0 and qqq_ewma < 0:
        return "outflow"
    return "mixed"


def leader_count(symbols: dict[str, SymbolContext]) -> int:
    count = 0
    for ctx in symbols.values():
        if (
            ctx.surge >= 3.0
            and ctx.rvol_1m >= 1.8
            and ctx.vwap is not None
            and ctx.last_price is not None
            and ctx.last_price >= ctx.vwap
            and ctx.spread_pct <= 0.0035
        ):
            count += 1
    return count


def from_open_return(bars: deque, last_price: float | None) -> float:
    if not bars or last_price is None or bars[0].open <= 0:
        return 0.0
    return (last_price - bars[0].open) / bars[0].open


def risk_off(
    spy: QuoteSnapshot | None,
    qqq: QuoteSnapshot | None,
    spy_from_open: float,
    qqq_from_open: float,
    breadth_negative_persistent: bool,
) -> bool:
    if spy is None or qqq is None:
        return False
    spy_below_vwap = spy.vwap is not None and spy.last < spy.vwap
    qqq_below_vwap = qqq.vwap is not None and qqq.last < qqq.vwap
    return (
        spy_below_vwap
        and qqq_below_vwap
        and (spy_from_open <= -0.006 or qqq_from_open <= -0.008)
        and breadth_negative_persistent
    )


def chop(weak_directional_progress: bool, repeated_vwap_crossings: bool, poor_leader_follow_through: bool) -> bool:
    return weak_directional_progress and repeated_vwap_crossings and poor_leader_follow_through


def compute_regime(
    symbols: dict[str, SymbolContext],
    proxies: dict[str, QuoteSnapshot],
    proxy_returns: dict[str, float],
    flow_ewma: dict[str, float],
    settings: StrategySettings,
    breadth_negative_persistent: bool = False,
    weak_directional_progress: bool = False,
    repeated_vwap_crossings: bool = False,
    poor_leader_follow_through: bool = False,
) -> RegimeSnapshot:
    leaders = leader_count(symbols)
    snapshot = RegimeSnapshot(
        leader_count=leaders,
        leader_breadth_ok=leaders >= settings.leader_breadth_min,
        flow_regime=flow_regime(flow_ewma.get("SPY", 0.0), flow_ewma.get("QQQ", 0.0)),
        spy_from_open=proxy_returns.get("SPY", 0.0),
        qqq_from_open=proxy_returns.get("QQQ", 0.0),
        iwm_from_open=proxy_returns.get("IWM", 0.0),
        breadth_or_tick_negative_persistent=breadth_negative_persistent,
        weak_directional_progress=weak_directional_progress,
        repeated_vwap_crossings=repeated_vwap_crossings,
        poor_leader_follow_through=poor_leader_follow_through,
    )
    snapshot.risk_off = risk_off(
        proxies.get("SPY"),
        proxies.get("QQQ"),
        snapshot.spy_from_open,
        snapshot.qqq_from_open,
        breadth_negative_persistent,
    )
    snapshot.chop = chop(
        weak_directional_progress,
        repeated_vwap_crossings,
        poor_leader_follow_through,
    )
    snapshot.regime_ok = (not snapshot.risk_off) and (not snapshot.chop) and snapshot.leader_breadth_ok
    return snapshot
