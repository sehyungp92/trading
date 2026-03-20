import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy_iaric.config import ET, StrategySettings
from strategy_iaric.data import CanonicalBarBuilder
from strategy_iaric.exits import flow_reversal_exit_due
from strategy_iaric.models import Bar, MarketSnapshot, PortfolioState, SymbolIntradayState, WatchlistItem
from strategy_iaric.risk import adjust_qty_for_portfolio_constraints
from strategy_iaric.signals import alpha_step, update_symbol_tier


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 12, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _item() -> WatchlistItem:
    return WatchlistItem(
        symbol="AAPL",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        regime_score=0.8,
        regime_tier="A",
        regime_risk_multiplier=1.0,
        sector_score=1.2,
        sector_rank_weight=1.0,
        sponsorship_score=1.4,
        sponsorship_state="STRONG",
        persistence=0.9,
        intensity_z=1.1,
        accel_z=0.9,
        rs_percentile=92.0,
        leader_pass=True,
        trend_pass=True,
        trend_strength=0.18,
        earnings_risk_flag=False,
        blacklist_flag=False,
        anchor_date=_ts(9, 30).date(),
        anchor_type="SPONSORSHIP_STREAK",
        acceptance_pass=True,
        avwap_ref=100.0,
        avwap_band_lower=99.5,
        avwap_band_upper=100.5,
        daily_atr_estimate=2.0,
        intraday_atr_seed=0.01,
        daily_rank=1.5,
        tradable_flag=True,
        conviction_bucket="TOP",
        conviction_multiplier=1.5,
        recommended_risk_r=1.5,
    )


def test_alpha_step_detects_setup_and_reaches_ready_to_enter():
    settings = StrategySettings()
    item = _item()
    sym = SymbolIntradayState(symbol="AAPL", tier="HOT")
    market = MarketSnapshot(symbol="AAPL", last_price=100.2, session_high=104.0, session_vwap=102.5, avwap_live=100.0)
    pullback_bar = Bar("AAPL", _ts(9, 39), _ts(9, 40), 101.1, 101.2, 99.9, 100.2, 100_000)

    action, _ = alpha_step(item, sym, market, pullback_bar, None, pullback_bar.end_time, 0.01, settings)
    assert action == "SETUP_DETECTED"
    assert sym.fsm_state == "SETUP_DETECTED"
    assert sym.reclaim_level is not None

    market.last_price = sym.reclaim_level + 0.05
    sym.micropressure_mode = "TICK"
    action, adders = alpha_step(item, sym, market, None, None, _ts(9, 41), 0.01, settings)
    assert action == "MOVE_TO_ACCEPTING"
    assert sym.required_acceptance_count == 3
    assert "flow_unavailable" in adders

    sym.sponsorship_signal = "STRONG"
    sym.micropressure_signal = "ACCUMULATE"
    for minute in (45, 50, 55):
        bar_5m = Bar("AAPL", _ts(9, minute - 5), _ts(9, minute), 100.4, 101.0, 100.2, sym.reclaim_level + 0.1, 250_000)
        action, _ = alpha_step(item, sym, market, None, bar_5m, _ts(9, minute), 0.01, settings)
    assert action == "READY_TO_ENTER"
    assert sym.acceptance_count == 3


def test_tier_promotion_and_sector_risk_adjustment():
    settings = replace(StrategySettings(), sector_risk_cap_pct=0.35)
    item = _item()
    sym = SymbolIntradayState(symbol="AAPL")
    market = MarketSnapshot(symbol="AAPL", last_price=100.0, session_high=102.0, avwap_live=100.1)
    assert update_symbol_tier(item, sym, market, settings) == "HOT"

    portfolio = PortfolioState(account_equity=100_000.0, base_risk_fraction=0.0025)
    portfolio.open_positions["MSFT"] = type("Pos", (), {"qty_open": 400, "entry_price": 105.0, "current_stop": 100.0})()
    qty, reason = adjust_qty_for_portfolio_constraints(
        portfolio=portfolio,
        item=item,
        intended_qty=500,
        entry_price=100.0,
        stop_level=98.0,
        symbol_to_sector={"MSFT": "Technology", "AAPL": "Technology"},
        settings=settings,
    )
    assert qty < 500
    assert reason in {"sector_risk_reduced", "sector_risk_cap", "risk_budget_reduced", "risk_budget_cap"}


def test_market_wide_selling_adds_acceptance_penalty():
    settings = StrategySettings()
    item = _item()
    sym = SymbolIntradayState(symbol="AAPL", tier="HOT")
    market = MarketSnapshot(symbol="AAPL", last_price=100.2, session_high=104.0, session_vwap=102.5, avwap_live=100.0)
    pullback_bar = Bar("AAPL", _ts(9, 39), _ts(9, 40), 101.1, 101.2, 99.9, 100.2, 100_000)

    action, _ = alpha_step(item, sym, market, pullback_bar, None, pullback_bar.end_time, 0.01, settings)
    assert action == "SETUP_DETECTED"

    sym.micropressure_mode = "TICK"
    reclaim_bar = Bar("AAPL", _ts(9, 40), _ts(9, 41), 100.2, sym.reclaim_level + 0.02, 100.1, sym.reclaim_level + 0.01, 110_000)
    action, adders = alpha_step(
        item,
        sym,
        market,
        reclaim_bar,
        None,
        _ts(9, 41),
        0.01,
        settings,
        market_wide_institutional_selling=True,
    )
    assert action == "MOVE_TO_ACCEPTING"
    assert sym.required_acceptance_count == 4
    assert "market_selling" in adders


def test_canonical_bar_builder_aligns_to_clock_boundaries():
    builder = CanonicalBarBuilder()
    for minute in range(37, 45):
        builder.ingest_bar(
            Bar(
                "AAPL",
                _ts(9, minute),
                _ts(9, minute + 1),
                100.0,
                100.2,
                99.8,
                100.1,
                10_000,
            )
        )

    bars_5m = builder.aggregate_new_bars("AAPL", 5)
    assert len(bars_5m) == 1
    assert bars_5m[0].start_time == _ts(9, 40)
    assert bars_5m[0].end_time == _ts(9, 45)


def test_flow_reversal_exit_window_uses_time_of_day():
    assert not flow_reversal_exit_due(True, _ts(9, 30), opened_today=False)
    assert flow_reversal_exit_due(True, _ts(10, 5), opened_today=False)
