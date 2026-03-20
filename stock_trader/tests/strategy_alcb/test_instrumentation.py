import asyncio
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService
from instrumentation.src.missed_opportunity import MissedOpportunityLogger
from instrumentation.src.strategy_data_providers import ALCBInstrumentationDataProvider
from instrumentation.src.trade_logger import TradeLogger
from shared.oms.models.order import OrderRole
from strategy_alcb.config import ET, StrategySettings
from strategy_alcb.engine import ALCBEngine
from strategy_alcb.models import (
    Bar,
    Box,
    BreakoutQualification,
    Campaign,
    CampaignState,
    CandidateArtifact,
    CandidateItem,
    CompressionTier,
    Direction,
    EntryType,
    PendingOrderState,
    PositionPlan,
    PositionState,
    QuoteSnapshot,
    Regime,
    RegimeSnapshot,
    ResearchDailyBar,
)


def _ts(day: int, hour: int, minute: int) -> datetime:
    return datetime(2026, 3, day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _artifact() -> CandidateArtifact:
    daily = [
        ResearchDailyBar(
            trade_date=datetime(2026, 2, 1, tzinfo=ET).date() + timedelta(days=idx),
            open=90.0 + idx,
            high=91.0 + idx,
            low=89.0 + idx,
            close=90.5 + idx,
            volume=1_000_000,
        )
        for idx in range(20)
    ]
    bars_30m = [
        Bar("AAPL", _ts(12, 9, 30), _ts(12, 10, 0), 100.0, 101.0, 99.8, 100.5, 100_000),
        Bar("AAPL", _ts(12, 10, 0), _ts(12, 10, 30), 100.5, 101.4, 100.2, 101.1, 160_000),
    ]
    bars_4h = [
        Bar("AAPL", _ts(2, 9, 30) + timedelta(hours=4 * idx), _ts(2, 13, 30) + timedelta(hours=4 * idx), 90 + idx, 91 + idx, 89.5 + idx, 90.8 + idx, 500_000)
        for idx in range(12)
    ]
    campaign = Campaign(
        symbol="AAPL",
        state=CampaignState.POSITION_OPEN,
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
            disp_value=1.2,
            disp_threshold=0.8,
            breakout_rejected=False,
            rvol_d=1.7,
        ),
        avwap_anchor_ts="2026-03-03T09:30:00-05:00",
    )
    item = CandidateItem(
        symbol="AAPL",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        adv20_usd=100_000_000.0,
        median_spread_pct=0.001,
        selection_score=8,
        selection_detail={},
        stock_regime="BULL",
        market_regime="BULL",
        sector_regime="BULL",
        daily_trend_sign=1,
        relative_strength_percentile=0.9,
        accumulation_score=0.8,
        ttm_squeeze_bonus=0,
        average_30m_volume=150_000.0,
        median_30m_volume=140_000.0,
        tradable_flag=True,
        direction_bias="LONG",
        price=101.0,
        earnings_risk_flag=False,
        campaign=campaign,
        daily_bars=daily,
        bars_30m=bars_30m,
    )
    artifact = CandidateArtifact(
        trade_date=_ts(12, 9, 30).date(),
        generated_at=_ts(12, 8, 0),
        regime=RegimeSnapshot(0.8, "A", 1.0, True, True, True, True, "BULL"),
        items=[item],
        tradable=[item],
        overflow=[],
    )
    artifact.items[0].bars_30m = bars_30m
    return artifact


class _DummyOMS:
    def __init__(self) -> None:
        self.intents = []
        self._seq = 0

    async def submit_intent(self, intent):
        self.intents.append(intent)
        self._seq += 1
        return SimpleNamespace(oms_order_id=f"oid-{self._seq}")


class _Instrumentation:
    def __init__(self) -> None:
        self.missed = []
        self.orders = []
        self.indicators = []
        self.orderbooks = []
        self.errors = []

    def log_missed(self, **kwargs) -> None:
        self.missed.append(kwargs)

    def on_order_event(self, **kwargs) -> None:
        self.orders.append(kwargs)

    def on_indicator_snapshot(self, **kwargs) -> None:
        self.indicators.append(kwargs)

    def on_orderbook_context(self, **kwargs) -> None:
        self.orderbooks.append(kwargs)

    def log_error(self, **kwargs) -> None:
        self.errors.append(kwargs)


class _Recorder:
    def __init__(self) -> None:
        self.entries = []
        self.exits = []

    async def record_entry(self, **kwargs):
        self.entries.append(kwargs)
        return "trade-123"

    async def record_exit(self, **kwargs):
        self.exits.append(kwargs)


class _FailingOMS:
    async def submit_intent(self, intent):
        raise RuntimeError("boom")


class _MissingOrderIdOMS:
    async def submit_intent(self, intent):
        return SimpleNamespace(oms_order_id=None)


def _minute_bars(symbol: str, start: datetime, count: int, *, price: float = 100.0) -> list[Bar]:
    bars: list[Bar] = []
    for idx in range(count):
        base = price + (idx * 0.1)
        bars.append(
            Bar(
                symbol=symbol,
                start_time=start + timedelta(minutes=idx),
                end_time=start + timedelta(minutes=idx + 1),
                open=base,
                high=base + 0.2,
                low=base - 0.1,
                close=base + 0.05,
                volume=10_000 + idx,
            )
        )
    return bars


def _bars_30m(symbol: str, start: datetime, count: int, *, price: float = 100.0) -> list[Bar]:
    bars: list[Bar] = []
    for idx in range(count):
        base = price + idx
        bar_start = start + timedelta(minutes=30 * idx)
        bars.append(
            Bar(
                symbol=symbol,
                start_time=bar_start,
                end_time=bar_start + timedelta(minutes=30),
                open=base,
                high=base + 1.0,
                low=base - 0.5,
                close=base + 0.6,
                volume=100_000 + (idx * 1_000),
            )
        )
    return bars


def _snapshot_service(symbol: str = "AAPL", price: float = 101.0) -> MagicMock:
    service = MagicMock(spec=MarketSnapshotService)
    service.capture_now.return_value = MarketSnapshot(
        snapshot_id="snap-1",
        symbol=symbol,
        timestamp=datetime.now(timezone.utc).isoformat(),
        bid=price - 0.05,
        ask=price + 0.05,
        mid=price,
        spread_bps=9.5,
        last_trade_price=price,
        atr_14=2.0,
    )
    return service


def test_alcb_provider_exposes_5m_30m_1h_4h_and_daily_rows_with_since_and_limit():
    engine = ALCBEngine(
        oms_service=None,
        artifact=_artifact(),
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )
    start = _ts(12, 9, 30)
    engine._markets["AAPL"].minute_bars = _minute_bars("AAPL", start, 10, price=100.0)
    engine._markets["AAPL"].bars_30m = _bars_30m("AAPL", start, 4, price=100.0)
    engine._markets["AAPL"].bars_4h = [
        Bar("AAPL", _ts(2, 9, 30) + timedelta(hours=4 * idx), _ts(2, 13, 30) + timedelta(hours=4 * idx), 90 + idx, 91 + idx, 89.5 + idx, 90.8 + idx, 500_000)
        for idx in range(12)
    ]
    provider = ALCBInstrumentationDataProvider(engine)

    candles_5m = provider.get_ohlcv("AAPL", timeframe="5m", limit=10)
    assert len(candles_5m) == 2
    assert candles_5m[0][1] == 100.0
    assert candles_5m[0][2] == pytest.approx(100.6)
    assert candles_5m[0][3] == pytest.approx(99.9)
    assert candles_5m[0][4] == pytest.approx(100.45)
    assert provider.get_ohlcv("AAPL", timeframe="5m", since=candles_5m[1][0], limit=10) == [candles_5m[1]]

    candles_1h = provider.get_ohlcv("AAPL", timeframe="1h", limit=10)
    assert len(candles_1h) == 2
    assert candles_1h[0][1] == 100.0
    assert candles_1h[0][4] == pytest.approx(101.6)

    assert len(provider.get_ohlcv("AAPL", timeframe="30m", limit=10)) == 4
    assert len(provider.get_ohlcv("AAPL", timeframe="4h", limit=20)) == 12
    assert len(provider.get_ohlcv("AAPL", timeframe="1d", limit=20)) == 20
    assert provider.get_atr("AAPL") is not None


def test_alcb_on_quote_tracks_session_vwap_from_live_quote():
    engine = ALCBEngine(
        oms_service=None,
        artifact=_artifact(),
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )

    engine.on_quote(
        "AAPL",
        QuoteSnapshot(
            ts=_ts(12, 9, 45),
            bid=100.9,
            ask=101.0,
            last=100.95,
            cumulative_volume=10_000,
            cumulative_value=1_009_500,
            vwap=100.95,
            spread_pct=0.001,
        ),
    )
    assert engine._markets["AAPL"].session_vwap == pytest.approx(100.95)


def test_alcb_market_exit_submit_tracks_expected_price_and_depth():
    async def _run():
        instr = _Instrumentation()
        engine = ALCBEngine(
            oms_service=_DummyOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            instrumentation=instr,
        )
        state = engine._symbols["AAPL"]
        state.position = PositionState(
            direction=Direction.LONG,
            entry_price=100.0,
            qty_entry=25,
            qty_open=25,
            final_stop=97.5,
            current_stop=97.5,
            entry_time=_ts(12, 10, 0),
            initial_risk_per_share=2.5,
            max_favorable_price=100.0,
            max_adverse_price=100.0,
            tp1_price=102.0,
            tp2_price=104.0,
            trade_id="trade-1",
        )
        engine._markets["AAPL"].bid = 101.2
        engine._markets["AAPL"].ask = 101.25
        engine._markets["AAPL"].last_price = 101.22
        engine._markets["AAPL"].last_quote = QuoteSnapshot(
            ts=_ts(12, 10, 5),
            bid=101.2,
            ask=101.25,
            last=101.22,
            bid_size=250.0,
            ask_size=140.0,
            spread_pct=0.0005,
        )

        await engine._submit_market_exit("AAPL", 25, OrderRole.EXIT)

        assert instr.orders[-1]["requested_price"] == 101.2
        assert instr.orderbooks[-1]["bid_depth_10bps"] == 250.0
        assert instr.orderbooks[-1]["ask_depth_10bps"] == 140.0

    asyncio.run(_run())


def test_alcb_trade_logger_backfill_uses_provider_5m_aggregation():
    tmpdir = tempfile.mkdtemp()
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    exit_time = now - timedelta(hours=5)
    entry_time = exit_time - timedelta(minutes=30)

    engine = ALCBEngine(
        oms_service=None,
        artifact=_artifact(),
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )
    engine._markets["AAPL"].minute_bars = _minute_bars("AAPL", exit_time, 300, price=101.0)
    provider = ALCBInstrumentationDataProvider(engine)
    logger = TradeLogger(
        {
            "bot_id": "stock_trader",
            "strategy_id": "alcb",
            "data_dir": tmpdir,
            "data_source_id": "ibkr_us_equities",
        },
        _snapshot_service(),
        strategy_type="strategy_alcb",
    )

    logger.log_entry(
        trade_id="trade-1",
        pair="AAPL",
        side="LONG",
        entry_price=100.5,
        position_size=10,
        position_size_quote=1005.0,
        entry_signal="A_AVWAP_RETEST",
        entry_signal_id="sig-1",
        entry_signal_strength=8.0,
        active_filters=[],
        passed_filters=[],
        strategy_params={"stop0": 99.0},
        exchange_timestamp=entry_time,
    )
    logger.log_exit(
        trade_id="trade-1",
        exit_price=101.0,
        exit_reason="EXIT",
        exchange_timestamp=exit_time,
    )
    logger.run_post_exit_backfill(provider)

    files = list((Path(tmpdir) / "trades").glob("trades_*.jsonl"))
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    exit_event = json.loads(lines[-1])
    assert exit_event["post_exit_backfill_status"] == "complete"
    assert exit_event["post_exit_1h_price"] is not None
    assert exit_event["post_exit_4h_price"] is not None


def test_alcb_missed_backfill_uses_provider_5m_aggregation():
    tmpdir = tempfile.mkdtemp()
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    signal_time = now - timedelta(hours=5)

    engine = ALCBEngine(
        oms_service=None,
        artifact=_artifact(),
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )
    engine._markets["AAPL"].minute_bars = _minute_bars("AAPL", signal_time, 300, price=100.0)
    provider = ALCBInstrumentationDataProvider(engine)
    logger = MissedOpportunityLogger(
        {
            "bot_id": "stock_trader",
            "strategy_id": "alcb",
            "data_dir": tmpdir,
            "data_source_id": "ibkr_us_equities",
        },
        _snapshot_service(price=100.0),
    )

    logger.log_missed(
        pair="AAPL",
        side="LONG",
        signal="alcb_breakout",
        signal_id="missed-1",
        signal_strength=7.0,
        blocked_by="entry_gate",
        exchange_timestamp=signal_time,
        strategy_type="strategy_alcb",
    )
    logger.run_backfill(provider)

    files = list((Path(tmpdir) / "missed").glob("*.jsonl"))
    payload = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert payload["backfill_status"] == "partial"
    assert payload["outcome_1h"] is not None
    assert payload["outcome_4h"] is not None


def test_alcb_passes_entry_and_exit_latency_meta_into_trade_recorder():
    async def _run():
        recorder = _Recorder()
        engine = ALCBEngine(
            oms_service=_DummyOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            trade_recorder=recorder,
            instrumentation=_Instrumentation(),
        )
        state = engine._symbols["AAPL"]
        state.intraday_score = 9
        state.mode = "NORMAL"
        state.last_30m_bar_time = _ts(12, 10, 0)
        engine._portfolio.total_pnl_pct = -1.25
        state.entry_order = PendingOrderState(
            oms_order_id="entry-1",
            submitted_at=_ts(12, 9, 59),
            role="ENTRY",
            requested_qty=10,
            limit_price=101.0,
            direction=Direction.LONG,
            entry_type=EntryType.A_AVWAP_RETEST,
            entry_price=101.0,
            planned_stop_price=99.0,
            planned_tp1_price=103.0,
            planned_tp2_price=105.0,
            risk_per_share=2.0,
            risk_dollars=20.0,
        )
        engine._order_index["entry-1"] = ("AAPL", "ENTRY")

        await engine._handle_fill(
            SimpleNamespace(
                oms_order_id="entry-1",
                payload={"qty": 10, "price": 101.2, "commission": 1.0},
                timestamp=_ts(12, 10, 0),
            )
        )

        entry_meta = recorder.entries[0]["meta"]
        assert entry_meta["concurrent_positions"] == 1
        assert entry_meta["drawdown_pct"] == -1.25
        assert entry_meta["bar_id"] == engine._decision_bar_id("AAPL", _ts(12, 10, 0))
        assert entry_meta["entry_latency_ms"] == 60000
        assert entry_meta["execution_timestamps"]["order_submitted_at"] == _ts(12, 9, 59).isoformat()
        assert entry_meta["execution_timestamps"]["fill_received_at"] == _ts(12, 10, 0).isoformat()

        stop_order_id = state.position.stop_order_id
        engine._order_submission_times[stop_order_id] = datetime(
            2026, 3, 12, 10, 59, 30, tzinfo=ET
        ).astimezone(timezone.utc)
        await engine._handle_fill(
            SimpleNamespace(
                oms_order_id=stop_order_id,
                payload={"qty": 10, "price": 99.2, "commission": 1.5},
                timestamp=_ts(12, 11, 0),
            )
        )

        exit_meta = recorder.exits[0]["meta"]
        assert exit_meta["exit_latency_ms"] == 30000
        assert exit_meta["fees_paid"] == 2.5

    asyncio.run(_run())


def test_alcb_restores_stop_submission_time_for_exit_latency_and_expected_price():
    async def _run():
        recorder = _Recorder()
        engine = ALCBEngine(
            oms_service=None,
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            trade_recorder=recorder,
        )
        state = engine._symbols["AAPL"]
        state.position = PositionState(
            direction=Direction.LONG,
            entry_price=100.0,
            qty_entry=10,
            qty_open=10,
            final_stop=98.5,
            current_stop=98.5,
            entry_time=_ts(12, 10, 0),
            initial_risk_per_share=1.5,
            max_favorable_price=100.0,
            max_adverse_price=100.0,
            tp1_price=102.0,
            tp2_price=104.0,
            trade_id="trade-restore",
            stop_order_id="stop-restore",
            stop_submitted_at=datetime(2026, 3, 12, 10, 59, 30, tzinfo=ET).astimezone(timezone.utc),
        )
        engine._portfolio.open_positions["AAPL"] = state.position
        engine._restore_order_state("AAPL", state)

        await engine._handle_fill(
            SimpleNamespace(
                oms_order_id="stop-restore",
                payload={"qty": 10, "price": 98.2, "commission": 1.25},
                timestamp=_ts(12, 11, 0),
            )
        )

        exit_meta = recorder.exits[0]["meta"]
        assert exit_meta["expected_exit_price"] == 98.5
        assert exit_meta["exit_latency_ms"] == 30000

    asyncio.run(_run())


def test_alcb_emits_indicator_snapshot_on_entry_submission(monkeypatch):
    async def _run():
        instr = _Instrumentation()
        engine = ALCBEngine(
            oms_service=_DummyOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            instrumentation=instr,
        )
        state = engine._symbols["AAPL"]
        state.campaign.state = CampaignState.BREAKOUT
        state.last_30m_bar_time = _ts(12, 10, 0)
        market = engine._markets["AAPL"]
        market.last_price = 101.8
        market.spread_pct = 0.0012
        market.session_vwap = 101.0
        market.avwap_live = 100.8

        monkeypatch.setattr("strategy_alcb.engine.event_block", lambda item: False)
        monkeypatch.setattr("strategy_alcb.engine.in_entry_window", lambda now, settings: True)
        monkeypatch.setattr("strategy_alcb.engine.classify_4h_regime", lambda bars: Regime.BULL)
        monkeypatch.setattr("strategy_alcb.engine.directional_regime_pass", lambda *args, **kwargs: (True, {}))
        monkeypatch.setattr(
            "strategy_alcb.engine.intraday_evidence_score",
            lambda *args, **kwargs: (9, {"volume_alignment": 3, "tightness": 2}),
        )
        monkeypatch.setattr("strategy_alcb.engine.determine_intraday_mode", lambda *args, **kwargs: "NORMAL")
        monkeypatch.setattr(engine, "_intraday_threshold", lambda mode: 5)
        monkeypatch.setattr("strategy_alcb.engine.choose_stop", lambda *args, **kwargs: 99.5)
        monkeypatch.setattr(
            "strategy_alcb.engine.position_size",
            lambda *args, **kwargs: PositionPlan(
                symbol="AAPL",
                direction=Direction.LONG,
                entry_type=EntryType.A_AVWAP_RETEST,
                entry_price=101.5,
                stop_price=99.5,
                tp1_price=103.0,
                tp2_price=105.0,
                quantity=10,
                risk_per_share=2.0,
                risk_dollars=20.0,
                quality_mult=1.0,
                regime_mult=1.0,
                corr_mult=1.0,
            ),
        )
        monkeypatch.setattr("strategy_alcb.engine.portfolio_heat_after", lambda *args, **kwargs: 0.01)
        monkeypatch.setattr("strategy_alcb.engine.friction_gate_pass", lambda *args, **kwargs: True)
        monkeypatch.setattr("strategy_alcb.engine.max_positions_pass", lambda *args, **kwargs: True)
        monkeypatch.setattr("strategy_alcb.engine.sector_limit_pass", lambda *args, **kwargs: True)
        monkeypatch.setattr(engine, "_live_market_regime", lambda store: (Regime.BULL, {"SPY": "BULL"}))
        monkeypatch.setattr(engine, "_choose_entry", lambda symbol: (EntryType.A_AVWAP_RETEST, 101.5))
        submit_entry = AsyncMock(return_value=True)
        monkeypatch.setattr(engine, "_submit_entry", submit_entry)

        await engine._advance_symbol("AAPL", _ts(12, 10, 30))

        submit_entry.assert_awaited_once()
        assert instr.indicators
        snapshot = instr.indicators[-1]
        assert snapshot["decision"] == "entry_submitted"
        assert snapshot["signal_name"] == "alcb_intraday_decision"
        assert snapshot["context"]["candidate_entry_type"] == "A_AVWAP_RETEST"
        assert snapshot["context"]["market_regime"] == "BULL"
        assert "last_vs_session_vwap_pct" in snapshot["indicators"]
        assert "last_vs_avwap_pct" in snapshot["indicators"]

    asyncio.run(_run())


def test_alcb_logs_structured_engine_errors_for_oms_failures():
    async def _run():
        instr = _Instrumentation()
        engine = ALCBEngine(
            oms_service=_FailingOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            instrumentation=instr,
        )
        state = engine._symbols["AAPL"]
        state.position = PositionState(
            direction=Direction.LONG,
            entry_price=100.0,
            qty_entry=10,
            qty_open=10,
            final_stop=98.0,
            current_stop=98.0,
            entry_time=_ts(12, 10, 0),
            initial_risk_per_share=2.0,
            max_favorable_price=100.0,
            max_adverse_price=100.0,
            tp1_price=102.0,
            tp2_price=104.0,
        )
        state.position.stop_order_id = "stop-1"
        engine._markets["AAPL"].bid = 100.1
        engine._markets["AAPL"].ask = 100.2
        engine._markets["AAPL"].last_price = 100.15

        entry_plan = PositionPlan(
            symbol="AAPL",
            direction=Direction.LONG,
            entry_type=EntryType.A_AVWAP_RETEST,
            entry_price=100.5,
            stop_price=98.5,
            tp1_price=102.5,
            tp2_price=104.5,
            quantity=10,
            risk_per_share=2.0,
            risk_dollars=20.0,
            quality_mult=1.0,
            regime_mult=1.0,
            corr_mult=1.0,
        )

        assert await engine._submit_entry("AAPL", entry_plan, _ts(12, 10, 0)) is False
        state.position.stop_order_id = ""
        await engine._submit_stop("AAPL")
        state.position.stop_order_id = "stop-1"
        await engine._replace_stop("AAPL")
        await engine._submit_market_exit("AAPL", 10, OrderRole.EXIT)

        assert [error["error_type"] for error in instr.errors] == [
            "submit_entry_failed",
            "submit_stop_failed",
            "replace_stop_failed",
            "submit_market_exit_failed",
        ]

    asyncio.run(_run())


def test_alcb_logs_structured_errors_when_oms_receipt_has_no_order_id():
    async def _run():
        instr = _Instrumentation()
        engine = ALCBEngine(
            oms_service=_MissingOrderIdOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            instrumentation=instr,
        )
        state = engine._symbols["AAPL"]
        state.position = PositionState(
            direction=Direction.LONG,
            entry_price=100.0,
            qty_entry=10,
            qty_open=10,
            final_stop=98.0,
            current_stop=98.0,
            entry_time=_ts(12, 10, 0),
            initial_risk_per_share=2.0,
            max_favorable_price=100.0,
            max_adverse_price=100.0,
            tp1_price=102.0,
            tp2_price=104.0,
        )
        engine._markets["AAPL"].bid = 100.1
        engine._markets["AAPL"].ask = 100.2
        engine._markets["AAPL"].last_price = 100.15

        plan = PositionPlan(
            symbol="AAPL",
            direction=Direction.LONG,
            entry_type=EntryType.A_AVWAP_RETEST,
            entry_price=100.5,
            stop_price=98.5,
            tp1_price=102.5,
            tp2_price=104.5,
            quantity=10,
            risk_per_share=2.0,
            risk_dollars=20.0,
            quality_mult=1.0,
            regime_mult=1.0,
            corr_mult=1.0,
        )

        assert await engine._submit_entry("AAPL", plan, _ts(12, 10, 0)) is False
        state.position.stop_order_id = ""
        await engine._submit_stop("AAPL")
        await engine._submit_market_exit("AAPL", 10, OrderRole.EXIT)

        assert [error["error_type"] for error in instr.errors] == [
            "submit_entry_failed",
            "submit_stop_failed",
            "submit_market_exit_failed",
        ]

    asyncio.run(_run())
