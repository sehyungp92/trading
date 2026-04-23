from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.stock.us_orb.config import StrategySettings
from strategies.stock.us_orb.models import (
    AcceptanceState,
    CachedSymbol,
    DangerSnapshot,
    DangerState,
    MinuteBar,
    QuoteSnapshot,
    RegimeSnapshot,
    State,
    SymbolContext,
)
from strategies.stock.us_orb.signals import (
    acceptance_passed,
    candidate_pre_score,
    live_gate_pass,
    quality_score,
)


def _cached_symbol() -> CachedSymbol:
    return CachedSymbol(
        symbol="NFLX",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        adv20=30_000_000.0,
        prior_close=100.0,
        sma20=95.0,
        sma60=90.0,
        sma20_slope=1.0,
        atr1m14=1.0,
        opening_value15_baseline_0935_0950=1_000_000.0,
        minute_volume_baseline_0935_1115={"09:51": 100_000.0},
        sector="technology",
        catalyst_tag="earnings",
        tech_tag=True,
    )


def _context(*, tick_flow_available: bool, now: datetime) -> SymbolContext:
    ctx = SymbolContext(cached=_cached_symbol(), state=State.WAIT_ACCEPTANCE)
    ctx.tick_flow_available = tick_flow_available
    ctx.quote = QuoteSnapshot(
        ts=now,
        bid=101.0,
        ask=101.1,
        last=101.05,
        tick_flow_available=tick_flow_available,
        bid_size=1000.0,
        ask_size=900.0,
        vwap=100.5,
        spread_pct=0.001,
    )
    ctx.bars.append(
        MinuteBar(
            ts=now - timedelta(minutes=1),
            open=100.8,
            high=101.2,
            low=100.7,
            close=101.05,
            volume=10_000.0,
            dollar_value=1_010_500.0,
        )
    )
    ctx.vdm = DangerSnapshot(state=DangerState.SAFE)
    ctx.surge = 4.0
    ctx.rvol_1m = 3.0
    ctx.imbalance_90s = 0.35
    ctx.relative_strength_5m = 0.01
    ctx.spread_pct = 0.001
    ctx.vwap = 100.5
    ctx.or_high = 100.8
    ctx.or_low = 99.8
    ctx.or_mid = 100.3
    ctx.or_pct = 0.015
    ctx.acceptance = AcceptanceState(
        break_time=now - timedelta(minutes=2),
        deadline=now + timedelta(minutes=3),
        pulled_back=True,
        held_support=True,
        reclaimed=True,
        retest_low=100.7,
    )
    return ctx


def test_tick_flow_unavailable_removes_imbalance_bonus_and_blocks_entry_gates() -> None:
    market = RegimeSnapshot(regime_ok=True, flow_regime="strong_inflow")
    settings = StrategySettings()
    now = datetime(2026, 4, 22, 13, 55, tzinfo=timezone.utc)
    live_ctx = _context(tick_flow_available=True, now=now)
    degraded_ctx = _context(tick_flow_available=False, now=now)

    assert candidate_pre_score(live_ctx, market) > candidate_pre_score(degraded_ctx, market)
    assert quality_score(live_ctx, market) > quality_score(degraded_ctx, market)
    assert live_gate_pass(live_ctx, market, now, settings) is True
    assert live_gate_pass(degraded_ctx, market, now, settings) is False
    assert acceptance_passed(live_ctx, now) is True
    assert acceptance_passed(degraded_ctx, now) is False
