import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.ibkr_core.models.types import BrokerOrderRef, IBContractSpec
from shared.ibkr_core.state.cache import IBCache
from shared.oms.services.factory import build_oms_service
from strategy_orb.config import ET, StrategySettings
from strategy_orb.data import load_universe_cache
from strategy_orb.diagnostics import JsonlDiagnostics
from strategy_orb.engine import USORBEngine
from strategy_orb.models import MinuteBar, QuoteSnapshot, State


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 12, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _cache_file(tmp_path: Path) -> Path:
    row = {
        "symbol": "AAPL",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "currency": "USD",
        "tick_size": 0.01,
        "point_value": 1.0,
        "adv20": 125_000_000.0,
        "prior_close": 100.0,
        "sma20": 99.0,
        "sma60": 95.0,
        "sma20_slope": 0.25,
        "atr1m14": 0.4,
        "opening_value15_baseline_0935_0950": 1_000_000.0,
        "minute_volume_baseline_0935_1115": {
            "09:35": 100_000.0,
            "09:50": 80_000.0,
            "10:00": 60_000.0,
            "11:15": 25_000.0,
        },
        "sector": "Technology",
        "float_shares": 500_000_000.0,
        "catalyst_tag": "earnings",
        "luld_tier": "tier_1",
        "tech_tag": True,
    }
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


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


def test_mocked_runtime_submits_entry_and_forced_flatten(tmp_path):
    async def _run():
        adapter = _FakeAdapter()
        oms = await build_oms_service(
            adapter=adapter,
            strategy_id="US_ORB_v1",
            unit_risk_dollars=400.0,
            daily_stop_R=5.0,
            heat_cap_R=4.0,
            portfolio_daily_stop_R=5.0,
            db_pool=None,
            portfolio_rules_config=None,
        )
        await oms.start()

        cache = load_universe_cache(_cache_file(tmp_path))
        engine = USORBEngine(
            oms_service=oms,
            cache=cache,
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(leader_breadth_min=1),
            diagnostics=JsonlDiagnostics(tmp_path / "diag", enabled=False),
        )
        await engine.start()

        engine.update_scanner_symbols(["AAPL"], _ts(9, 50))
        ctx = engine._ensure_context("AAPL")
        ctx.state = State.READY
        ctx.or_high = 101.0
        ctx.or_low = 100.0
        ctx.or_pct = 0.01
        ctx.surge = 4.0
        ctx.rvol_1m = 3.0
        ctx.imbalance_90s = 0.35
        ctx.vwap = 100.8
        ctx.spread_pct = 0.001
        ctx.last5m_value = 10_000_000.0
        ctx.acceptance.pulled_back = True
        ctx.acceptance.held_support = True
        ctx.acceptance.reclaimed = True
        ctx.quote = QuoteSnapshot(
            ts=_ts(9, 51),
            bid=101.39,
            ask=101.40,
            last=101.40,
            ask_size=50_000,
            spread_pct=0.001,
            vwap=100.8,
        )
        for minute in range(46, 51):
            ctx.bars.append(
                MinuteBar(
                    ts=_ts(9, minute),
                    open=100.8,
                    high=101.2,
                    low=100.7,
                    close=101.0,
                    volume=250_000,
                    dollar_value=25_000_000,
                )
            )
        engine._regime.regime_ok = True
        engine._regime.leader_breadth_ok = True
        engine._regime.leader_count = 1

        await engine._advance_symbol(ctx, _ts(9, 51))
        await asyncio.sleep(0.2)

        assert adapter.submissions
        assert adapter.submissions[0]["instrument"].symbol == "AAPL"
        assert adapter.submissions[0]["instrument"].sec_type == "STK"
        assert adapter.submissions[0]["order_type"] == "STOP_LIMIT"

        entry_order_id = adapter.submissions[0]["oms_order_id"]
        adapter.on_fill(entry_order_id, "exec-entry", 101.25, ctx.qty, _ts(9, 52), 0.0)
        await asyncio.sleep(0.2)

        assert ctx.state == State.IN_POSITION
        assert ctx.position is not None
        assert any(submission["order_type"] == "STOP" for submission in adapter.submissions)

        exit_qty = ctx.position.qty_open
        await engine._flatten_all("test_flatten")
        await asyncio.sleep(0.2)

        market_orders = [submission for submission in adapter.submissions if submission["order_type"] == "MARKET"]
        assert market_orders

        exit_order_id = market_orders[-1]["oms_order_id"]
        adapter.on_fill(exit_order_id, "exec-exit", 101.10, exit_qty, _ts(9, 53), 0.0)
        await asyncio.sleep(0.2)

        assert ctx.state == State.DONE
        assert ctx.position is None

        await engine.stop()
        await oms.stop()

    asyncio.run(_run())
