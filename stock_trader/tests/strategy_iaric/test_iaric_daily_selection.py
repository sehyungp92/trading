import json
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy_iaric.artifact_store import load_watchlist_artifact
from strategy_iaric.config import ET, StrategySettings
from strategy_iaric.diagnostics import JsonlDiagnostics
from strategy_iaric.research import daily_selection_from_snapshot, run_daily_selection
from strategy_iaric.models import (
    HeldPositionResearch,
    MarketResearch,
    ResearchDailyBar,
    ResearchSnapshot,
    ResearchSymbol,
    SectorResearch,
)


def _bars(start_price: float, count: int, drift: float, impulse_index: int | None = None) -> list[ResearchDailyBar]:
    rows = []
    price = start_price
    start_day = date(2025, 12, 15)
    for index in range(count):
        open_price = price
        close_price = price + drift + (0.15 if index % 5 == 0 else 0.0)
        high = max(open_price, close_price) + 0.6
        low = min(open_price, close_price) - 0.6
        volume = 1_000_000 + (index * 12_500)
        tag = ""
        if impulse_index is not None and index == impulse_index:
            close_price += 2.5
            high = close_price + 0.8
            volume *= 4
            tag = "BREAKOUT"
        rows.append(
            ResearchDailyBar(
                trade_date=start_day + timedelta(days=index),
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
                event_tag=tag,
            )
        )
        price = close_price
    return rows


def _symbol(symbol: str, sector: str, start_price: float, flow: list[float], drift: float) -> ResearchSymbol:
    return ResearchSymbol(
        symbol=symbol,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector=sector,
        price=start_price + 30,
        adv20_usd=75_000_000.0,
        median_spread_pct=0.0015,
        earnings_within_sessions=10,
        blacklist_flag=False,
        halted_flag=False,
        severe_news_flag=False,
        flow_proxy_history=flow,
        daily_bars=_bars(start_price=start_price, count=70, drift=drift, impulse_index=45),
        sector_return_20d=0.06,
        sector_return_60d=0.18,
        intraday_atr_seed=0.012,
    )


def _settings(tmp_path: Path) -> StrategySettings:
    return replace(
        StrategySettings(),
        research_dir=tmp_path / "research",
        artifact_dir=tmp_path / "artifacts",
        diagnostics_dir=tmp_path / "diag",
        state_dir=tmp_path / "state",
    )


def test_run_daily_selection_persists_artifact_and_marks_flow_reversal(tmp_path):
    trade_date = date(2026, 3, 12)
    settings = _settings(tmp_path)

    snapshot = ResearchSnapshot(
        trade_date=trade_date,
        market=MarketResearch(
            price_ok=True,
            breadth_pct_above_20dma=72.0,
            vix_percentile_1y=40.0,
            hy_spread_5d_bps_change=3.0,
        ),
        sectors={
            "Technology": SectorResearch("Technology", flow_trend_20d=2.5, breadth_20d=0.78, participation=0.68),
            "Healthcare": SectorResearch("Healthcare", flow_trend_20d=0.8, breadth_20d=0.52, participation=0.44),
        },
        symbols={
            "AAPL": _symbol("AAPL", "Technology", 100.0, [5, 6, 8, 7, 9, 8, 9, 8, 7, 8, 6, 7, 8, 9, 7, 8, 9, 8, 7, 8], 0.55),
            "NVDA": _symbol("NVDA", "Technology", 90.0, [4, 4, 5, 4, 6, 5, 5, 6, 5, 4, 4, 5, 4, 6, 5, 6, 5, 4, 4, 5], 0.48),
            "PFE": _symbol("PFE", "Healthcare", 50.0, [2, 2, 1, 0, 1, 0, -1, -2, -3, -4, -1, -2, -1, -3, -2, -4, -2, -3, -5, -6], 0.12),
        },
        held_positions=[
            HeldPositionResearch(
                symbol="PFE",
                entry_time=datetime(2026, 3, 11, 15, 30, tzinfo=ET).astimezone(timezone.utc),
                entry_price=78.5,
                size=100,
                stop=76.0,
                initial_r=2.5,
                setup_tag="FLOW_DRIVEN_GRIND",
                carry_eligible_flag=True,
            )
        ],
    )

    research_path = settings.research_dir / f"{trade_date.isoformat()}.json"
    research_path.parent.mkdir(parents=True, exist_ok=True)
    with open(research_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "trade_date": trade_date.isoformat(),
                "market": {
                    "price_ok": snapshot.market.price_ok,
                    "breadth_pct_above_20dma": snapshot.market.breadth_pct_above_20dma,
                    "vix_percentile_1y": snapshot.market.vix_percentile_1y,
                    "hy_spread_5d_bps_change": snapshot.market.hy_spread_5d_bps_change,
                },
                "sectors": {
                    name: {
                        "flow_trend_20d": sector.flow_trend_20d,
                        "breadth_20d": sector.breadth_20d,
                        "participation": sector.participation,
                    }
                    for name, sector in snapshot.sectors.items()
                },
                "symbols": {
                    name: {
                        "exchange": symbol.exchange,
                        "primary_exchange": symbol.primary_exchange,
                        "currency": symbol.currency,
                        "tick_size": symbol.tick_size,
                        "point_value": symbol.point_value,
                        "sector": symbol.sector,
                        "price": symbol.price,
                        "adv20_usd": symbol.adv20_usd,
                        "median_spread_pct": symbol.median_spread_pct,
                        "earnings_within_sessions": symbol.earnings_within_sessions,
                        "blacklist_flag": symbol.blacklist_flag,
                        "halted_flag": symbol.halted_flag,
                        "severe_news_flag": symbol.severe_news_flag,
                        "flow_proxy_history": symbol.flow_proxy_history,
                        "sector_return_20d": symbol.sector_return_20d,
                        "sector_return_60d": symbol.sector_return_60d,
                        "intraday_atr_seed": symbol.intraday_atr_seed,
                        "daily_bars": [
                            {
                                "trade_date": bar.trade_date.isoformat(),
                                "open": bar.open,
                                "high": bar.high,
                                "low": bar.low,
                                "close": bar.close,
                                "volume": bar.volume,
                                "event_tag": bar.event_tag,
                            }
                            for bar in symbol.daily_bars
                        ],
                    }
                    for name, symbol in snapshot.symbols.items()
                },
                "held_positions": [
                    {
                        "symbol": held.symbol,
                        "entry_time": held.entry_time.isoformat(),
                        "entry_price": held.entry_price,
                        "size": held.size,
                        "stop": held.stop,
                        "initial_r": held.initial_r,
                        "setup_tag": held.setup_tag,
                        "carry_eligible_flag": held.carry_eligible_flag,
                    }
                    for held in snapshot.held_positions
                ],
            },
            handle,
            indent=2,
        )

    artifact = run_daily_selection(trade_date, settings=settings, diagnostics=JsonlDiagnostics(settings.diagnostics_dir, enabled=False))

    assert artifact.regime.tier == "A"
    assert artifact.tradable
    assert artifact.tradable[0].symbol == "AAPL"
    assert artifact.tradable[0].anchor_type in {"SPONSORSHIP_STREAK", "BREAKOUT", "IMPULSE_DAY"}
    assert artifact.held_positions[0].flow_reversal_flag is True
    assert artifact.items[0].symbol == "AAPL"

    persisted = load_watchlist_artifact(trade_date, settings=settings)
    assert persisted.tradable[0].symbol == artifact.tradable[0].symbol
    assert (settings.artifact_dir / f"{trade_date.isoformat()}.json").exists()


def test_daily_selection_tier_c_returns_no_new_trades(tmp_path):
    settings = _settings(tmp_path)
    snapshot = ResearchSnapshot(
        trade_date=date(2026, 3, 12),
        market=MarketResearch(
            price_ok=False,
            breadth_pct_above_20dma=20.0,
            vix_percentile_1y=95.0,
            hy_spread_5d_bps_change=40.0,
        ),
        sectors={"Technology": SectorResearch("Technology", flow_trend_20d=0.0, breadth_20d=0.4, participation=0.4)},
        symbols={"AAPL": _symbol("AAPL", "Technology", 100.0, [3] * 20, 0.45)},
    )

    artifact = daily_selection_from_snapshot(snapshot, settings=settings, diagnostics=JsonlDiagnostics(settings.diagnostics_dir, enabled=False))
    assert artifact.regime.tier == "C"
    assert artifact.items == []
    assert artifact.tradable == []
