import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy_iaric.config import ET, StrategySettings
from strategy_iaric.engine import IARICEngine
from strategy_iaric.models import PendingOrderState, RegimeSnapshot, WatchlistArtifact, WatchlistItem


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 12, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _artifact() -> WatchlistArtifact:
    item = WatchlistItem(
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
        sector_score=1.4,
        sector_rank_weight=1.0,
        sponsorship_score=1.6,
        sponsorship_state="STRONG",
        persistence=0.9,
        intensity_z=1.2,
        accel_z=1.1,
        rs_percentile=95.0,
        leader_pass=True,
        trend_pass=True,
        trend_strength=0.2,
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
        daily_rank=1.7,
        tradable_flag=True,
        conviction_bucket="TOP",
        conviction_multiplier=1.5,
        recommended_risk_r=1.5,
    )
    return WatchlistArtifact(
        trade_date=_ts(9, 30).date(),
        generated_at=_ts(8, 15),
        regime=RegimeSnapshot(0.8, "A", 1.0, True, True, True, True),
        items=[item],
        tradable=[item],
        overflow=[],
        held_positions=[],
    )


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

    def log_missed(self, **kwargs) -> None:
        self.missed.append(kwargs)

    def on_order_event(self, **kwargs) -> None:
        self.orders.append(kwargs)

    def on_indicator_snapshot(self, **kwargs) -> None:
        self.indicators.append(kwargs)

    def on_orderbook_context(self, **kwargs) -> None:
        self.orderbooks.append(kwargs)


class _Recorder:
    def __init__(self) -> None:
        self.entries = []
        self.exits = []

    async def record_entry(self, **kwargs):
        self.entries.append(kwargs)
        return "trade-123"

    async def record_exit(self, **kwargs):
        self.exits.append(kwargs)


def test_iaric_logs_missed_opportunity_on_spread_block():
    async def _run():
        instr = _Instrumentation()
        engine = IARICEngine(
            oms_service=_DummyOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            instrumentation=instr,
        )
        state = engine._symbols["AAPL"]
        market = engine._markets["AAPL"]
        state.fsm_state = "ACCEPTING"
        state.setup_type = "reclaim"
        state.confidence = "GREEN"
        state.stop_level = 99.0
        state.active_order_id = "SUBMITTING_ENTRY"
        market.last_price = 100.0
        market.ask = 100.05
        market.spread_pct = 0.02

        await engine._submit_entry("AAPL", _ts(9, 50))

        assert instr.missed
        assert instr.missed[-1]["blocked_by"] == "spread_guard"

    asyncio.run(_run())


def test_iaric_passes_instrumentation_meta_into_trade_recorder():
    async def _run():
        recorder = _Recorder()
        engine = IARICEngine(
            oms_service=_DummyOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            trade_recorder=recorder,
            instrumentation=_Instrumentation(),
        )
        state = engine._symbols["AAPL"]
        market = engine._markets["AAPL"]
        state.stop_level = 99.0
        state.setup_type = "reclaim"
        state.location_grade = "A"
        state.confidence = "GREEN"
        state.setup_tag = "TEST"
        state.entry_order = PendingOrderState(
            oms_order_id="entry-1",
            submitted_at=_ts(9, 54),
            role="ENTRY",
            requested_qty=10,
            limit_price=101.0,
        )
        engine._order_index["entry-1"] = ("AAPL", "ENTRY")
        market.last_price = 101.2

        await engine._handle_fill(
            SimpleNamespace(
                oms_order_id="entry-1",
                payload={"qty": 10, "price": 101.2},
                timestamp=_ts(9, 55),
            )
        )

        state.last_transition_reason = "time_stop"
        state.exit_order = PendingOrderState(
            oms_order_id="exit-1",
            submitted_at=_ts(10, 0),
            role="EXIT",
            requested_qty=10,
        )
        engine._order_index["exit-1"] = ("AAPL", "EXIT")

        await engine._handle_fill(
            SimpleNamespace(
                oms_order_id="exit-1",
                payload={"qty": 10, "price": 102.0},
                timestamp=_ts(10, 1),
            )
        )

        assert recorder.entries
        assert recorder.entries[0]["meta"]["entry_signal"] == "reclaim"
        assert recorder.entries[0]["meta"]["expected_entry_price"] == 101.0
        assert recorder.entries[0]["meta"]["signal_factors"]
        assert recorder.exits
        assert recorder.exits[0]["meta"]["expected_exit_price"] == 102.0

    asyncio.run(_run())


def test_iaric_market_exit_submit_captures_expected_price_and_depth():
    async def _run():
        instr = _Instrumentation()
        engine = IARICEngine(
            oms_service=_DummyOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
            instrumentation=instr,
        )
        state = engine._symbols["AAPL"]
        market = engine._markets["AAPL"]
        state.position = SimpleNamespace(
            qty_open=10,
            trade_id="trade-1",
        )
        state.in_position = True
        market.bid = 101.45
        market.ask = 101.5
        market.last_price = 101.48
        market.last_quote = SimpleNamespace(bid_size=300.0, ask_size=180.0)

        await engine._submit_market_exit("AAPL", 10, SimpleNamespace(value="EXIT"))

        assert state.exit_order is not None
        assert state.exit_order.limit_price == 101.45
        assert instr.orders[-1]["requested_price"] == 101.45
        assert instr.orderbooks[-1]["bid_depth_10bps"] == 300.0
        assert instr.orderbooks[-1]["ask_depth_10bps"] == 180.0

    asyncio.run(_run())
