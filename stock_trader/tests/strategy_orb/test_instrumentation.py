import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy_orb.config import ET, StrategySettings
from strategy_orb.data import load_universe_cache
from strategy_orb.engine import USORBEngine
from strategy_orb.models import QuoteSnapshot, State


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
            "11:15": 25_000.0,
        },
        "sector": "Technology",
        "float_shares": 500_000_000.0,
        "catalyst_tag": "earnings",
        "luld_tier": "tier_1",
        "tech_tag": True,
        "secondary_universe": False,
    }
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


class _DummyOMS:
    def __init__(self) -> None:
        self.intents = []
        self._seq = 0
        self.risk = SimpleNamespace(daily_realized_pnl=0.0, open_risk_R=0.0, daily_realized_R=0.0)

    def stream_events(self, strategy_id: str):
        return asyncio.Queue()

    async def submit_intent(self, intent):
        self.intents.append(intent)
        self._seq += 1
        order = getattr(intent, "order", None)
        return SimpleNamespace(oms_order_id=(order.oms_order_id if order else f"oid-{self._seq}"))

    async def get_strategy_risk(self, strategy_id: str):
        return self.risk


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


def test_us_orb_logs_order_submit_and_ttl_missed(tmp_path):
    async def _run():
        oms = _DummyOMS()
        instr = _Instrumentation()
        cache = load_universe_cache(_cache_file(tmp_path))
        engine = USORBEngine(
            oms_service=oms,
            cache=cache,
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(leader_breadth_min=1),
            instrumentation=instr,
        )
        ctx = engine._ensure_context("AAPL")
        ctx.state = State.ORDER_SENT
        ctx.qty = 50
        ctx.planned_entry = 101.0
        ctx.planned_limit = 101.1
        ctx.final_stop = 100.4
        ctx.surge = 4.0
        ctx.pre_score = 85.0
        ctx.quality_score = 90.0
        ctx.rvol_1m = 3.0
        ctx.relative_strength_5m = 0.02
        ctx.spread_pct = 0.001
        ctx.quote = QuoteSnapshot(ts=_ts(9, 52), bid=101.0, ask=101.1, last=101.05, spread_pct=0.001)

        await engine._submit_entry(ctx, _ts(9, 51))
        assert instr.orders
        assert instr.orders[-1]["status"] == "SUBMITTED"

        ctx.state = State.ORDER_SENT
        ctx.entry_order.submitted_at = _ts(9, 51) - timedelta(seconds=26)
        await engine._advance_symbol(ctx, _ts(9, 52))

        assert instr.missed
        assert instr.missed[-1]["blocked_by"] == "entry_ttl"

    asyncio.run(_run())


def test_us_orb_market_exit_submit_tracks_expected_price_and_depth(tmp_path):
    async def _run():
        oms = _DummyOMS()
        instr = _Instrumentation()
        cache = load_universe_cache(_cache_file(tmp_path))
        engine = USORBEngine(
            oms_service=oms,
            cache=cache,
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(leader_breadth_min=1),
            instrumentation=instr,
        )
        ctx = engine._ensure_context("AAPL")
        ctx.quote = QuoteSnapshot(
            ts=_ts(10, 5),
            bid=101.2,
            ask=101.25,
            last=101.22,
            bid_size=250.0,
            ask_size=140.0,
            spread_pct=0.0005,
        )
        ctx.position = SimpleNamespace(qty_open=25, trade_id="trade-1")

        await engine._submit_market_exit(ctx, 25, SimpleNamespace(value="EXIT"))

        assert instr.orders[-1]["requested_price"] == 101.2
        assert instr.orderbooks[-1]["bid_depth_10bps"] == 250.0
        assert instr.orderbooks[-1]["ask_depth_10bps"] == 140.0
        assert next(iter(engine._exit_price_hints.values())) == 101.2

    asyncio.run(_run())
