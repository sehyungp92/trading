import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.ibkr_core.models.types import BrokerOrderRef, IBContractSpec
from shared.ibkr_core.state.cache import IBCache
from shared.oms.services.factory import build_oms_service
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
    IntradayStateSnapshot,
    PendingOrderState,
    PositionState,
    QuoteSnapshot,
    RegimeSnapshot,
    ResearchDailyBar,
    SymbolRuntimeState,
)


def _ts(day: int, hour: int, minute: int) -> datetime:
    return datetime(2026, 3, day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _enum_name(value):
    return getattr(value, "value", value)


class _FakeAdapter:
    def __init__(self):
        self.is_congested = False
        self.cache = IBCache()
        self.submissions = []
        self.cancellations = []
        self.replacements = []
        self._seq = 100
        self.on_ack = lambda *args, **kwargs: None
        self.on_reject = lambda *args, **kwargs: None
        self.on_fill = lambda *args, **kwargs: None
        self.on_status = lambda *args, **kwargs: None

    async def submit_order(self, **kwargs):
        self._seq += 1
        broker_ref = BrokerOrderRef(self._seq, self._seq + 1000, self._seq + 2000)
        self.submissions.append({"oms_order_id": kwargs["oms_order_id"], **kwargs})
        self.cache.register_order(kwargs["oms_order_id"], broker_ref.broker_order_id, broker_ref.perm_id)
        self.cache.contracts[broker_ref.con_id] = IBContractSpec(
            con_id=broker_ref.con_id,
            symbol=kwargs["instrument"].symbol,
            sec_type=kwargs["instrument"].sec_type,
            exchange=kwargs["instrument"].exchange,
            currency=kwargs["instrument"].currency,
            multiplier=kwargs["instrument"].multiplier,
            tick_size=kwargs["instrument"].tick_size,
            trading_class=kwargs["instrument"].trading_class,
            primary_exchange=kwargs["instrument"].primary_exchange,
        )
        self.on_ack(kwargs["oms_order_id"], broker_ref)
        self.on_status(kwargs["oms_order_id"], "Submitted", 0)
        return broker_ref

    async def cancel_order(self, broker_order_id: int, perm_id: int = 0):
        self.cancellations.append((broker_order_id, perm_id))

    async def replace_order(self, broker_order_id: int, new_qty=None, new_limit_price=None, new_stop_price=None):
        self.replacements.append((broker_order_id, new_qty, new_limit_price, new_stop_price))
        return BrokerOrderRef(broker_order_id, broker_order_id + 1000, broker_order_id + 2000)

    async def request_open_orders(self):
        return []

    async def request_positions(self):
        return []

    async def request_executions(self, since_ts=None):
        return []

    async def rebuild_cache(self, resolver):
        return None


def _daily_bars() -> list[ResearchDailyBar]:
    bars = []
    start = datetime(2026, 2, 10, tzinfo=ET).date()
    for idx in range(25):
        close = 90.0 + (0.35 * idx)
        bars.append(
            ResearchDailyBar(
                trade_date=start + timedelta(days=idx),
                open=close - 0.4,
                high=close + 1.1,
                low=close - 1.0,
                close=close,
                volume=1_000_000 + (idx * 10_000),
            )
        )
    bars[-2] = ResearchDailyBar(bars[-2].trade_date, 99.6, 100.7, 99.2, 100.4, 1_200_000)
    bars[-1] = ResearchDailyBar(bars[-1].trade_date, 100.2, 102.0, 99.9, 101.3, 1_800_000)
    return bars


def _bars_30m() -> list[Bar]:
    return [
        Bar("AAPL", _ts(12, 9, 30), _ts(12, 10, 0), 100.2, 100.8, 100.0, 100.5, 100_000),
        Bar("AAPL", _ts(12, 10, 0), _ts(12, 10, 30), 100.6, 101.2, 99.9, 101.0, 180_000),
    ]


def _bars_4h() -> list[Bar]:
    bars = []
    base = _ts(3, 9, 30)
    for idx in range(12):
        open_price = 90.0 + (idx * 0.8)
        bars.append(
            Bar(
                "AAPL",
                base + timedelta(hours=4 * idx),
                base + timedelta(hours=(4 * idx) + 4),
                open_price,
                open_price + 1.3,
                open_price - 0.5,
                open_price + 1.0,
                500_000,
            )
        )
    return bars


def _artifact() -> CandidateArtifact:
    campaign = Campaign(
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
        state=CampaignState.BREAKOUT,
    )
    item = CandidateItem(
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
        stock_regime="BULL",
        market_regime="BULL",
        sector_regime="BULL",
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
        campaign=campaign,
        daily_bars=_daily_bars(),
        bars_30m=_bars_30m(),
    )
    return CandidateArtifact(
        trade_date=_ts(12, 9, 30).date(),
        generated_at=_ts(12, 8, 0),
        regime=RegimeSnapshot(0.8, "A", 1.0, True, True, True, True, "BULL"),
        items=[item],
        tradable=[item],
        overflow=[],
        long_candidates=[item],
        short_candidates=[],
    )


def test_engine_submits_entry_and_handles_tp1_fill():
    async def _run():
        adapter = _FakeAdapter()
        oms = await build_oms_service(
            adapter=adapter,
            strategy_id="ALCB_v1",
            unit_risk_dollars=300.0,
            daily_stop_R=5.0,
            heat_cap_R=4.0,
            portfolio_daily_stop_R=5.0,
            db_pool=None,
            portfolio_rules_config=None,
        )
        await oms.start()

        engine = ALCBEngine(
            oms_service=oms,
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
        )
        engine._markets["AAPL"].bars_4h = _bars_4h()
        engine._markets["AAPL"].last_quote = QuoteSnapshot(
            ts=_ts(12, 10, 30),
            bid=100.95,
            ask=101.0,
            last=100.98,
            bid_size=200.0,
            ask_size=150.0,
            spread_pct=0.0005,
        )
        await engine.start()

        await engine._advance_symbol("AAPL", _ts(12, 10, 30))
        await asyncio.sleep(0.3)

        entry_order = adapter.submissions[0]
        assert entry_order["instrument"].symbol == "AAPL"

        adapter.on_fill(entry_order["oms_order_id"], "exec-entry", 100.0, entry_order["qty"], _ts(12, 10, 31), 0.0)
        await asyncio.sleep(0.3)

        state = engine._symbols["AAPL"]
        assert state.position is not None
        assert state.campaign.state.value == "POSITION_OPEN"
        assert any(_enum_name(sub["order_type"]) == "STOP" for sub in adapter.submissions)
        assert state.tp1_order is not None
        assert state.tp2_order is not None

        tp1_qty = state.tp1_order.requested_qty
        adapter.on_fill(state.tp1_order.oms_order_id, "exec-tp1", state.position.tp1_price, tp1_qty, _ts(12, 11, 0), 0.0)
        await asyncio.sleep(0.3)

        assert state.position.partial_taken is True
        assert state.position.profit_funded is True
        assert state.campaign.profit_funded is True
        assert state.stop_order is not None
        assert state.position.stop_order_id
        assert state.tp2_order is not None

        await engine.stop()
        await oms.stop()

    asyncio.run(_run())


def test_engine_restores_pending_order_indexes_from_state():
    artifact = _artifact()
    engine = ALCBEngine(
        oms_service=None,
        artifact=artifact,
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )
    snapshot = IntradayStateSnapshot(
        trade_date=_ts(12, 9, 30).date(),
        saved_at=_ts(12, 11, 0),
        symbols=[
            SymbolRuntimeState(
                symbol="AAPL",
                campaign=artifact.items[0].campaign,
                entry_order=PendingOrderState(
                    oms_order_id="entry-1",
                    submitted_at=_ts(12, 10, 30),
                    role="ENTRY",
                    requested_qty=100,
                    limit_price=100.0,
                    direction=Direction.LONG,
                    entry_type=EntryType.A_AVWAP_RETEST,
                    entry_price=100.0,
                    planned_stop_price=97.5,
                    planned_tp1_price=101.8,
                    planned_tp2_price=103.5,
                    risk_per_share=2.5,
                    risk_dollars=250.0,
                ),
                tp1_order=PendingOrderState(
                    oms_order_id="tp1-1",
                    submitted_at=_ts(12, 10, 32),
                    role="TP",
                    requested_qty=30,
                    limit_price=101.8,
                ),
                tp2_order=PendingOrderState(
                    oms_order_id="tp2-1",
                    submitted_at=_ts(12, 10, 32),
                    role="TP",
                    requested_qty=30,
                    limit_price=103.5,
                ),
                position=PositionState(
                    direction=Direction.LONG,
                    entry_price=100.0,
                    qty_entry=100,
                    qty_open=100,
                    final_stop=97.5,
                    current_stop=97.5,
                    entry_time=_ts(12, 10, 31),
                    initial_risk_per_share=2.5,
                    max_favorable_price=100.0,
                    max_adverse_price=100.0,
                    tp1_price=101.8,
                    tp2_price=103.5,
                    stop_order_id="stop-1",
                ),
            )
        ],
        markets=[],
    )

    engine.hydrate_state(snapshot)

    assert engine._order_index["entry-1"] == ("AAPL", "ENTRY")
    assert engine._order_index["tp1-1"] == ("AAPL", "TP1")
    assert engine._order_index["tp2-1"] == ("AAPL", "TP2")
    assert engine._order_index["stop-1"] == ("AAPL", "STOP")
    assert engine._portfolio.pending_entry_risk["AAPL"] == 250.0
    assert engine._pending_plans["entry-1"].stop_price == 97.5


def test_continuation_mode_disables_entry_a_and_b():
    artifact = _artifact()
    artifact.items[0].campaign.continuation_enabled = True
    artifact.items[0].campaign.state = CampaignState.CONTINUATION
    engine = ALCBEngine(
        oms_service=None,
        artifact=artifact,
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )
    engine._markets["AAPL"].bars_30m[-1].low = 100.1

    entry_type, _ = engine._choose_entry("AAPL")

    assert entry_type == EntryType.C_CONTINUATION


def test_hydrate_state_prefers_rebuilt_campaign_over_stale_dirty_state():
    artifact = _artifact()
    artifact.items[0].campaign.box.start_date = "2026-03-04"
    artifact.items[0].campaign.box.end_date = "2026-03-12"
    artifact.items[0].campaign.box.high = 100.5
    artifact.items[0].campaign.box.low = 97.3
    engine = ALCBEngine(
        oms_service=None,
        artifact=artifact,
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )
    stale_campaign = Campaign(
        symbol="AAPL",
        state=CampaignState.DIRTY,
        reentry_block_same_direction=True,
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
    )
    snapshot = IntradayStateSnapshot(
        trade_date=_ts(12, 9, 30).date(),
        saved_at=_ts(12, 11, 0),
        symbols=[SymbolRuntimeState(symbol="AAPL", campaign=stale_campaign)],
        markets=[],
    )

    engine.hydrate_state(snapshot)

    assert engine._symbols["AAPL"].campaign.state == CampaignState.BREAKOUT
    assert engine._symbols["AAPL"].campaign.box.start_date == "2026-03-04"


def test_engine_keeps_partial_entry_order_live_and_links_exit_orders_with_oca():
    async def _run():
        adapter = _FakeAdapter()
        oms = await build_oms_service(
            adapter=adapter,
            strategy_id="ALCB_v1",
            unit_risk_dollars=300.0,
            daily_stop_R=5.0,
            heat_cap_R=4.0,
            portfolio_daily_stop_R=5.0,
            db_pool=None,
            portfolio_rules_config=None,
        )
        await oms.start()

        engine = ALCBEngine(
            oms_service=oms,
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
        )
        engine._markets["AAPL"].bars_4h = _bars_4h()
        engine._markets["AAPL"].last_quote = QuoteSnapshot(
            ts=_ts(12, 10, 30),
            bid=100.95,
            ask=101.0,
            last=100.98,
            bid_size=200.0,
            ask_size=150.0,
            spread_pct=0.0005,
        )
        await engine.start()

        await engine._advance_symbol("AAPL", _ts(12, 10, 30))
        await asyncio.sleep(0.3)

        entry_order = adapter.submissions[0]
        adapter.on_fill(entry_order["oms_order_id"], "exec-entry-1", 100.0, 4, _ts(12, 10, 31), 0.1)
        await asyncio.sleep(0.3)

        state = engine._symbols["AAPL"]
        assert state.position is not None
        assert state.position.qty_open == 4
        assert state.entry_order is not None
        assert state.entry_order.filled_qty == 4
        assert state.stop_order is not None
        assert state.tp1_order is not None
        assert state.tp2_order is not None
        assert engine._portfolio.pending_entry_risk["AAPL"] == pytest.approx(
            state.entry_order.risk_per_share * (state.entry_order.requested_qty - state.entry_order.filled_qty)
        )

        exit_submissions = adapter.submissions[1:4]
        assert len(exit_submissions) == 3
        assert all(submission["oca_group"] for submission in exit_submissions)
        assert len({submission["oca_group"] for submission in exit_submissions}) == 1
        assert {submission["oca_type"] for submission in exit_submissions} == {1}

        adapter.on_fill(entry_order["oms_order_id"], "exec-entry-2", 100.1, entry_order["qty"] - 4, _ts(12, 10, 32), 0.2)
        await asyncio.sleep(0.3)

        tp1_qty, tp2_qty = engine._tp_quantities(state.position.qty_entry)
        assert state.entry_order is None
        assert state.position.qty_open == entry_order["qty"]
        assert state.stop_order is not None
        assert state.stop_order.requested_qty == state.position.qty_open
        assert state.tp1_order is not None
        assert state.tp1_order.requested_qty == tp1_qty
        assert state.tp2_order is not None
        assert state.tp2_order.requested_qty == tp2_qty
        assert any(replacement[2] is not None for replacement in adapter.replacements)

        await engine.stop()
        await oms.stop()

    asyncio.run(_run())


def test_engine_keeps_partial_exit_state_until_terminal():
    async def _run():
        class _LocalOMS:
            def __init__(self) -> None:
                self._seq = 0

            async def submit_intent(self, intent):
                self._seq += 1
                return SimpleNamespace(oms_order_id=f"oid-{self._seq}")

        engine = ALCBEngine(
            oms_service=_LocalOMS(),
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
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
            trade_id="trade-1",
            stop_order_id="stop-1",
            tp1_order_id="tp1-1",
            tp2_order_id="tp2-1",
            exit_oca_group="oca-1",
        )
        state.stop_order = PendingOrderState(
            oms_order_id="stop-1",
            submitted_at=_ts(12, 10, 1),
            role="STOP",
            requested_qty=10,
            stop_price=98.5,
        )
        state.tp1_order = PendingOrderState(
            oms_order_id="tp1-1",
            submitted_at=_ts(12, 10, 1),
            role="TP",
            requested_qty=3,
            limit_price=102.0,
        )
        state.tp2_order = PendingOrderState(
            oms_order_id="tp2-1",
            submitted_at=_ts(12, 10, 1),
            role="TP",
            requested_qty=3,
            limit_price=104.0,
        )
        engine._portfolio.open_positions["AAPL"] = state.position
        engine._restore_order_state("AAPL", state)

        await engine._handle_fill(
            SimpleNamespace(
                oms_order_id="tp1-1",
                payload={"qty": 1, "price": 102.0, "commission": 0.25, "requested_qty": 3},
                timestamp=_ts(12, 10, 30),
            )
        )

        assert state.position.qty_open == 9
        assert state.position.partial_taken is True
        assert state.tp1_order is not None
        assert state.tp1_order.filled_qty == 1
        assert state.position.tp1_order_id == "tp1-1"
        assert state.stop_order is not None
        assert state.stop_order.requested_qty == 9

        await engine._handle_fill(
            SimpleNamespace(
                oms_order_id="stop-1",
                payload={"qty": 4, "price": 98.5, "commission": 0.5, "requested_qty": 9},
                timestamp=_ts(12, 10, 31),
            )
        )

        assert state.position.qty_open == 5
        assert state.stop_order is not None
        assert state.stop_order.filled_qty == 4
        assert state.position.stop_order_id == "stop-1"
        assert state.exit_order is None

    asyncio.run(_run())


def test_event_loop_continues_after_handler_exception():
    async def _run():
        engine = ALCBEngine(
            oms_service=None,
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
        )
        processed: list[str] = []

        async def _handle_event(event):
            processed.append(event.payload["kind"])
            if event.payload["kind"] == "boom":
                raise RuntimeError("boom")
            engine._running = False

        engine._handle_event = _handle_event  # type: ignore[method-assign]
        engine._event_queue = asyncio.Queue()
        engine._running = True
        await engine._event_queue.put(SimpleNamespace(payload={"kind": "boom"}))
        await engine._event_queue.put(SimpleNamespace(payload={"kind": "after"}))

        await asyncio.wait_for(engine._event_loop(), timeout=1.0)

        assert processed == ["boom", "after"]

    asyncio.run(_run())
