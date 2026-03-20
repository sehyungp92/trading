import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.ibkr_core.models.types import BrokerOrderRef, IBContractSpec
from shared.ibkr_core.state.cache import IBCache
from shared.oms.services.factory import build_oms_service
from strategy_iaric.config import ET, StrategySettings
from strategy_iaric.engine import IARICEngine
from strategy_iaric.models import Bar, IntradayStateSnapshot, PendingOrderState, PositionState, QuoteSnapshot, RegimeSnapshot, SymbolIntradayState, WatchlistArtifact, WatchlistItem


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 12, hour, minute, tzinfo=ET).astimezone(timezone.utc)


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


def _enum_name(value):
    return getattr(value, "value", value)


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
    regime = RegimeSnapshot(0.8, "A", 1.0, True, True, True, True)
    return WatchlistArtifact(
        trade_date=_ts(9, 30).date(),
        generated_at=_ts(8, 15),
        regime=regime,
        items=[item],
        tradable=[item],
        overflow=[],
        held_positions=[],
    )


def test_engine_submits_entry_and_manages_position():
    async def _run():
        adapter = _FakeAdapter()
        oms = await build_oms_service(
            adapter=adapter,
            strategy_id="IARIC_v1",
            unit_risk_dollars=250.0,
            daily_stop_R=5.0,
            heat_cap_R=4.0,
            portfolio_daily_stop_R=5.0,
            db_pool=None,
            portfolio_rules_config=None,
        )
        await oms.start()

        engine = IARICEngine(
            oms_service=oms,
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
        )
        await engine.start()

        closes = {
            35: 104.0,
            36: 103.0,
            37: 102.0,
            38: 101.0,
            39: 100.2,
            40: 100.6,
            41: 100.7,
            42: 100.8,
            43: 100.9,
            44: 101.0,
            45: 101.0,
            46: 101.0,
            47: 101.1,
            48: 101.1,
            49: 101.2,
            50: 101.1,
            51: 101.2,
            52: 101.2,
            53: 101.3,
            54: 101.3,
        }

        for minute, close in closes.items():
            engine.on_quote(
                "AAPL",
                QuoteSnapshot(
                    ts=_ts(9, minute),
                    bid=close - 0.01,
                    ask=close,
                    last=close,
                    cumulative_volume=100_000 + (minute - 35) * 10_000,
                    spread_pct=0.0002,
                ),
            )
            engine.on_bar(
                "AAPL",
                Bar(
                    "AAPL",
                    _ts(9, minute),
                    _ts(9, minute + 1),
                    close + 0.1,
                    max(close + 0.2, 104.2 if minute == 35 else close + 0.2),
                    99.9 if minute == 39 else close - 0.2,
                    close,
                    120_000,
                ),
            )

        await asyncio.sleep(0.3)
        entry_order = next(sub for sub in adapter.submissions if _enum_name(sub["order_type"]) == "LIMIT")
        assert entry_order["instrument"].symbol == "AAPL"

        entry_qty = entry_order["qty"]
        adapter.on_fill(entry_order["oms_order_id"], "exec-entry", 101.3, entry_qty, _ts(9, 55), 0.0)
        await asyncio.sleep(0.3)

        state = engine._symbols["AAPL"]
        assert state.position is not None
        assert state.fsm_state == "IN_POSITION"
        assert any(_enum_name(sub["order_type"]) == "STOP" for sub in adapter.submissions)

        engine.on_quote(
            "AAPL",
            QuoteSnapshot(
                ts=_ts(10, 0),
                bid=106.98,
                ask=107.0,
                last=107.0,
                cumulative_volume=500_000,
                spread_pct=0.0001,
            ),
        )
        engine.on_bar(
            "AAPL",
            Bar(
                "AAPL",
                _ts(10, 0),
                _ts(10, 1),
                106.8,
                107.2,
                106.7,
                107.0,
                150_000,
            ),
        )
        await asyncio.sleep(0.3)

        assert any(_enum_name(sub["order_type"]) == "MARKET" for sub in adapter.submissions)

        await engine.stop()
        await oms.stop()

    asyncio.run(_run())


def test_engine_dedupes_repeated_flatten_requests():
    async def _run():
        adapter = _FakeAdapter()
        oms = await build_oms_service(
            adapter=adapter,
            strategy_id="IARIC_v1",
            unit_risk_dollars=250.0,
            daily_stop_R=5.0,
            heat_cap_R=4.0,
            portfolio_daily_stop_R=5.0,
            db_pool=None,
            portfolio_rules_config=None,
        )
        await oms.start()

        engine = IARICEngine(
            oms_service=oms,
            artifact=_artifact(),
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
        )
        state = engine._symbols["AAPL"]
        position = PositionState(
            entry_price=101.0,
            qty_entry=100,
            qty_open=100,
            final_stop=99.5,
            current_stop=99.5,
            entry_time=_ts(9, 35),
            initial_risk_per_share=1.5,
            max_favorable_price=101.0,
            max_adverse_price=101.0,
            time_stop_deadline=_ts(10, 0),
        )
        state.position = position
        state.in_position = True
        engine._portfolio.open_positions["AAPL"] = position
        engine._markets["AAPL"].last_price = 100.8

        engine._manage_position("AAPL", _ts(10, 5))
        engine._manage_position("AAPL", _ts(10, 5))
        await asyncio.sleep(0.3)

        market_orders = [sub for sub in adapter.submissions if _enum_name(sub["order_type"]) == "MARKET"]
        assert len(market_orders) == 1

        await engine.stop()
        await oms.stop()

    asyncio.run(_run())


def test_engine_restores_pending_order_indexes_from_state():
    engine = IARICEngine(
        oms_service=None,
        artifact=_artifact(),
        account_id="DU1",
        nav=100_000.0,
        settings=StrategySettings(),
    )
    snapshot = IntradayStateSnapshot(
        trade_date=_ts(9, 30).date(),
        saved_at=_ts(10, 0),
        symbols=[
            SymbolIntradayState(
                symbol="AAPL",
                tier="HOT",
                in_position=True,
                active_order_id="entry-1",
                entry_order=PendingOrderState(
                    oms_order_id="entry-1",
                    submitted_at=_ts(9, 55),
                    role="ENTRY",
                    requested_qty=100,
                    limit_price=101.0,
                ),
                position=PositionState(
                    entry_price=101.0,
                    qty_entry=100,
                    qty_open=100,
                    final_stop=99.5,
                    current_stop=99.5,
                    entry_time=_ts(9, 56),
                    initial_risk_per_share=1.5,
                    max_favorable_price=101.0,
                    max_adverse_price=101.0,
                    stop_order_id="stop-1",
                ),
                exit_order=PendingOrderState(
                    oms_order_id="exit-1",
                    submitted_at=_ts(10, 5),
                    role="EXIT",
                    requested_qty=100,
                ),
                pending_hard_exit=True,
            )
        ],
        meta={"active_symbols": ["AAPL"]},
    )

    engine.hydrate_state(snapshot)

    assert engine._order_index["entry-1"] == ("AAPL", "ENTRY")
    assert engine._order_index["stop-1"] == ("AAPL", "STOP")
    assert engine._order_index["exit-1"] == ("AAPL", "EXIT")
