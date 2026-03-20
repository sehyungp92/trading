import ast
import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy_alcb.config import ET, ScannerSettings, StrategySettings
from strategy_alcb.data import StrategyDataStore, _MinuteAccumulator, aggregate_bars
from strategy_alcb.execution import build_entry_order
from strategy_alcb.exits import ratchet_runner_stop, stale_exit_needed, update_dirty_state
from strategy_alcb.models import (
    Bar,
    Box,
    BreakoutQualification,
    Campaign,
    CampaignState,
    CandidateItem,
    CompressionTier,
    Direction,
    EntryType,
    PortfolioState,
    PositionState,
    ResearchDailyBar,
    Regime,
)
from strategy_alcb.risk import choose_stop, position_size, quality_mult, regime_mult
from strategy_alcb.research_generator import _fetch_scanner_symbols
from strategy_alcb.signals import entry_a_trigger, intraday_evidence_score, market_regime_from_proxies


def _ts(day: int, hour: int, minute: int) -> datetime:
    return datetime(2026, 3, day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _daily_bars() -> list[ResearchDailyBar]:
    bars: list[ResearchDailyBar] = []
    start = date(2026, 1, 5)
    for idx in range(25):
        close = 90.0 + (0.35 * idx)
        bars.append(
            ResearchDailyBar(
                trade_date=start + timedelta(days=idx),
                open=close - 0.3,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1_000_000 + (idx * 10_000),
            )
        )
    bars[-2] = ResearchDailyBar(
        trade_date=bars[-2].trade_date,
        open=99.6,
        high=100.8,
        low=99.2,
        close=100.4,
        volume=1_200_000,
    )
    bars[-1] = ResearchDailyBar(
        trade_date=bars[-1].trade_date,
        open=100.2,
        high=102.0,
        low=99.9,
        close=101.4,
        volume=1_800_000,
    )
    return bars


def _bars_30m() -> list[Bar]:
    return [
        Bar("AAPL", _ts(12, 9, 30), _ts(12, 10, 0), 100.2, 100.9, 100.0, 100.5, 100_000),
        Bar("AAPL", _ts(12, 10, 0), _ts(12, 10, 30), 100.6, 101.2, 99.9, 101.0, 180_000),
    ]


def _bars_4h() -> list[Bar]:
    bars: list[Bar] = []
    base = _ts(10, 9, 30)
    for idx in range(12):
        open_price = 90.0 + (idx * 0.8)
        bars.append(
            Bar(
                "AAPL",
                base + timedelta(hours=4 * idx),
                base + timedelta(hours=(4 * idx) + 4),
                open_price,
                open_price + 1.4,
                open_price - 0.6,
                open_price + 1.0,
                500_000,
            )
        )
    return bars


def _campaign() -> Campaign:
    return Campaign(
        symbol="AAPL",
        box=Box(
            start_date="2026-03-03",
            end_date="2026-03-11",
            L_used=8,
            high=100.0,
            low=97.0,
            mid=98.5,
            height=3.0,
            containment=0.9,
            squeeze_metric=0.7,
            tier=CompressionTier.GOOD,
        ),
        breakout=BreakoutQualification(
            direction=Direction.LONG,
            breakout_date="2026-03-11",
            structural_pass=True,
            displacement_pass=True,
            disp_value=1.1,
            disp_threshold=0.8,
            breakout_rejected=False,
            rvol_d=1.8,
        ),
        avwap_anchor_ts="2026-03-03T09:30:00-05:00",
    )


def _item() -> CandidateItem:
    return CandidateItem(
        symbol="AAPL",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        adv20_usd=125_000_000.0,
        median_spread_pct=0.001,
        selection_score=9,
        selection_detail={"rs": 3, "trend": 2, "compression": 2, "accumulation": 2},
        stock_regime=Regime.BULL.value,
        market_regime=Regime.BULL.value,
        sector_regime=Regime.BULL.value,
        daily_trend_sign=1,
        relative_strength_percentile=0.95,
        accumulation_score=0.8,
        ttm_squeeze_bonus=0,
        average_30m_volume=150_000.0,
        median_30m_volume=140_000.0,
        tradable_flag=True,
        direction_bias="LONG",
        price=101.0,
        earnings_risk_flag=False,
        campaign=_campaign(),
        daily_bars=_daily_bars(),
        bars_30m=_bars_30m(),
    )


def test_intraday_evidence_score_supports_entry_a_long():
    item = _item()
    market = type("Market", (), {"symbol": "AAPL", "bars_4h": _bars_4h(), "bars_30m": [], "daily_bars": []})()
    store = StrategyDataStore({"AAPL": item}, {"AAPL": market})

    score, detail = intraday_evidence_score("AAPL", item.campaign, store, StrategySettings())

    assert score >= 4
    assert detail["above_avwap"] == 1
    assert detail["two_closes"] == 1
    assert detail["sq_good"] == 1
    trigger = entry_a_trigger("AAPL", item.campaign, store)
    assert trigger is not None
    assert trigger > item.campaign.box.high
    assert _bars_30m()[-1].low <= trigger < _bars_30m()[-1].close


def test_position_size_uses_midpoint_stop_for_high_quality_retest():
    settings = StrategySettings()
    item = _item()
    stop_price = choose_stop(
        entry_type=EntryType.A_AVWAP_RETEST,
        direction=Direction.LONG,
        item=item,
        campaign=item.campaign,
        settings=settings,
    )

    plan = position_size(
        item=item,
        entry_price=100.0,
        stop_price=stop_price,
        direction=Direction.LONG,
        campaign=item.campaign,
        intraday_score=4,
        portfolio=PortfolioState(
            account_equity=100_000.0,
            base_risk_fraction=settings.base_risk_fraction,
        ),
        items={"AAPL": item},
        settings=settings,
    )

    assert stop_price < item.campaign.box.mid
    assert plan is not None
    assert plan.quantity > 0
    assert plan.tp1_price > plan.entry_price


def test_aggregate_bars_resets_at_session_boundaries():
    bars: list[Bar] = []
    day_one = date(2026, 3, 12)
    for idx in range(13):
        start = datetime(2026, 3, 12, 9, 30, tzinfo=ET) + timedelta(minutes=30 * idx)
        end = start + timedelta(minutes=30)
        bars.append(Bar("AAPL", start.astimezone(timezone.utc), end.astimezone(timezone.utc), 100 + idx, 101 + idx, 99 + idx, 100.5 + idx, 1000))
    for idx in range(3):
        start = datetime(2026, 3, 13, 9, 30, tzinfo=ET) + timedelta(minutes=30 * idx)
        end = start + timedelta(minutes=30)
        bars.append(Bar("AAPL", start.astimezone(timezone.utc), end.astimezone(timezone.utc), 120 + idx, 121 + idx, 119 + idx, 120.5 + idx, 1000))

    aggregated = aggregate_bars(bars, 8)

    assert len(aggregated) == 2
    assert aggregated[0].start_time.astimezone(ET).date() == day_one
    assert aggregated[0].end_time.astimezone(ET).time().hour == 13
    assert aggregated[1].start_time.astimezone(ET).date() == day_one
    assert aggregated[1].end_time.astimezone(ET).time().hour == 16


def test_market_regime_uses_proxy_bars_instead_of_nightly_field():
    item = _item()
    proxy_bars = [
        Bar(
            "SPY",
            _ts(1 + idx, 9, 30),
            _ts(1 + idx, 13, 30),
            100.0 + (idx * 2.0),
            101.8 + (idx * 2.0),
            100.7 + (idx * 2.0),
            101.6 + (idx * 2.0),
            500_000,
        )
        for idx in range(20)
    ]
    store = StrategyDataStore(
        {"AAPL": item},
        {
            "AAPL": type("Market", (), {"symbol": "AAPL", "bars_4h": _bars_4h(), "bars_30m": [], "daily_bars": []})(),
            "SPY": type("Market", (), {"symbol": "SPY", "bars_4h": proxy_bars, "bars_30m": [], "daily_bars": []})(),
            "QQQ": type("Market", (), {"symbol": "QQQ", "bars_4h": [Bar("QQQ", bar.start_time, bar.end_time, bar.open, bar.high, bar.low, bar.close, bar.volume) for bar in proxy_bars], "bars_30m": [], "daily_bars": []})(),
        },
    )

    regime, detail = market_regime_from_proxies(store)

    assert regime == Regime.BULL
    assert detail == {"SPY": "BULL", "QQQ": "BULL"}


def test_entry_b_uses_marketable_limit_not_blind_market():
    item = _item()
    plan = position_size(
        item=item,
        entry_price=101.0,
        stop_price=97.5,
        direction=Direction.LONG,
        campaign=item.campaign,
        intraday_score=4,
        portfolio=PortfolioState(account_equity=100_000.0, base_risk_fraction=StrategySettings().base_risk_fraction),
        items={"AAPL": item},
        settings=StrategySettings(),
    )

    assert plan is not None
    plan.entry_type = EntryType.B_SWEEP_RECLAIM
    order = build_entry_order(item, "DU1", plan)

    assert order.order_type.value == "LIMIT"
    assert order.limit_price is not None
    assert order.limit_price > plan.entry_price


def test_minute_accumulator_does_not_backfill_full_session_volume_on_first_tick():
    accumulator = _MinuteAccumulator()

    assert accumulator.update("AAPL", _ts(12, 10, 0), 100.0, 1_250_000.0) is None

    closed = accumulator.update("AAPL", _ts(12, 11, 0), 100.2, 1_251_000.0)

    assert closed is not None
    assert closed.volume == 0.0


def test_dirty_reset_waits_for_half_box_length():
    campaign = _campaign()
    campaign.state = CampaignState.DIRTY
    campaign.reentry_block_same_direction = True
    campaign.dirty_since = "2026-03-12"

    update_dirty_state(campaign, latest_close=98.5, settings=StrategySettings(), as_of=date(2026, 3, 18))

    assert campaign.state == CampaignState.DIRTY
    assert campaign.reentry_block_same_direction is True


def test_universe_selector_has_no_web_vendor_dependencies():
    root = Path(__file__).resolve().parents[2]
    banned = {
        "requests",
        "httpx",
        "aiohttp",
        "bs4",
        "yfinance",
        "pandas_datareader",
        "selenium",
        "lxml",
        "wikipedia",
    }
    for relative in ("strategy_alcb/research_generator.py", "strategy_alcb/universe_constituents.py"):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & banned)


def test_scanner_supplement_uses_all_configured_scan_codes():
    class _FakeIB:
        async def reqScannerDataAsync(self, sub):
            payload = {
                "TOP_PERC_GAIN": ["NVDA", "TSLA"],
                "TOP_PERC_LOSE": ["TSLA", "SMCI"],
                "HOT_BY_VOLUME": ["PLTR"],
            }[sub.scanCode]
            return [
                SimpleNamespace(contractDetails=SimpleNamespace(contract=SimpleNamespace(symbol=symbol)))
                for symbol in payload
            ]

    async def _run():
        symbols = await _fetch_scanner_symbols(
            _FakeIB(),
            rate=type("Rate", (), {"wait_for": staticmethod(lambda cost=1.0: asyncio.sleep(0))})(),
            scanner=ScannerSettings(scan_codes=("TOP_PERC_GAIN", "TOP_PERC_LOSE", "HOT_BY_VOLUME")),
        )
        assert symbols == ["NVDA", "TSLA", "SMCI", "PLTR"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Audit fix #1: stale_exit_needed runner awareness
# ---------------------------------------------------------------------------

def _make_position(*, partial_taken: bool = False, entry_price: float = 100.0, risk: float = 2.0, direction: Direction = Direction.LONG) -> PositionState:
    sign = 1 if direction == Direction.LONG else -1
    return PositionState(
        direction=direction,
        entry_price=entry_price,
        qty_entry=100,
        qty_open=40 if partial_taken else 100,
        final_stop=entry_price - sign * risk,
        current_stop=entry_price - sign * risk,
        entry_time=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        initial_risk_per_share=risk,
        max_favorable_price=entry_price + sign * 4.0,
        max_adverse_price=entry_price - sign * 1.0,
        tp1_price=entry_price + sign * 2.5,
        tp2_price=entry_price + sign * 5.0,
        partial_taken=partial_taken,
    )


def test_stale_exit_skips_runner_at_low_r():
    """Runner at 0.4R should NOT be stale-exited (default runner threshold is 0.0)."""
    settings = StrategySettings()
    pos = _make_position(partial_taken=True, entry_price=100.0, risk=2.0)
    # price at 100.8 → unrealized_r = 0.8 / 2.0 = 0.4
    now = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)  # 15+ biz days later
    assert not stale_exit_needed(pos, now, 100.8, settings)


def test_stale_exit_triggers_for_full_position_at_low_r():
    """Full position at 0.4R SHOULD be stale-exited."""
    settings = StrategySettings()
    pos = _make_position(partial_taken=False, entry_price=100.0, risk=2.0)
    now = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
    assert stale_exit_needed(pos, now, 100.8, settings)


def test_stale_exit_triggers_for_runner_below_zero_r():
    """Runner below 0R should still be stale-exited (default threshold 0.0)."""
    settings = StrategySettings()
    pos = _make_position(partial_taken=True, entry_price=100.0, risk=2.0)
    now = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
    # price at 99.5 → unrealized_r = -0.5 / 2.0 = -0.25
    assert stale_exit_needed(pos, now, 99.5, settings)


# ---------------------------------------------------------------------------
# Audit fix #2: position_size rejection logging (regime_mult / quality_mult)
# ---------------------------------------------------------------------------

def test_regime_mult_zero_for_long_in_bear_stock():
    """LONG + BEAR stock regime → regime_mult returns 0.0 regardless of market."""
    assert regime_mult(Direction.LONG, Regime.BEAR, Regime.BULL) == 0.0
    assert regime_mult(Direction.LONG, Regime.BEAR, Regime.TRANSITIONAL) == 0.0
    assert regime_mult(Direction.LONG, Regime.BEAR, Regime.BEAR) == 0.0


def test_quality_mult_zero_for_low_displacement():
    """Insufficient displacement ratio → quality_mult returns 0.0."""
    campaign = _campaign()
    campaign.breakout.disp_value = 0.5  # ratio = 0.5/0.8 = 0.625 < 1.0
    assert quality_mult(campaign, intraday_score=5) == 0.0


def test_quality_mult_zero_for_low_score():
    """Intraday score of 1 → quality_mult returns 0.0."""
    campaign = _campaign()
    assert quality_mult(campaign, intraday_score=1) == 0.0


# ---------------------------------------------------------------------------
# Audit fix #4: ratchet_runner_stop returns trail component detail
# ---------------------------------------------------------------------------

def test_ratchet_runner_stop_returns_trail_detail():
    """ratchet_runner_stop should return (new_stop, detail_dict) with binding info."""
    settings = StrategySettings()
    pos = _make_position(partial_taken=True, entry_price=90.0, risk=2.0)
    bars = _bars_4h()
    new_stop, detail = ratchet_runner_stop(pos, bars, settings)
    assert isinstance(new_stop, float)
    assert isinstance(detail, dict)
    assert "binding" in detail
    assert "structure" in detail
    assert detail["binding"] in ("structure", "ema50", "expansion", "avwap", "profit_ratchet")


def test_ratchet_runner_stop_empty_bars_returns_unchanged():
    """Empty bars → returns current stop with 'unchanged' binding."""
    settings = StrategySettings()
    pos = _make_position(partial_taken=True, entry_price=90.0, risk=2.0)
    new_stop, detail = ratchet_runner_stop(pos, [], settings)
    assert new_stop == pos.current_stop
    assert detail["binding"] == "unchanged"
