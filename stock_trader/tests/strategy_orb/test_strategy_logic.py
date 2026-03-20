import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.oms.models.events import OMSEventType
from shared.oms.models.intent import IntentReceipt, IntentResult
from shared.oms.models.order import OrderRole
from strategy_orb.config import ET, StrategySettings
from strategy_orb.data import CacheValidationError, load_universe_cache
from strategy_orb.engine import USORBEngine
from strategy_orb.models import DangerSnapshot, DangerState, MinuteBar, PendingOrderState, PortfolioSnapshot, PositionState, QuoteSnapshot, RegimeSnapshot, State
from strategy_orb.signals import apply_gap_policy, compute_order_plan, exit_signal, live_gate_pass, quality_score, volatility_danger_snapshot


def _cache_row(symbol: str = "AAPL") -> dict:
    return {
        "symbol": symbol,
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
        "secondary_universe": False,
    }


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 12, hour, minute, tzinfo=ET).astimezone(timezone.utc)


class _DummyOMS:
    def __init__(self):
        self.events = asyncio.Queue()
        self.intents = []
        self.risk = SimpleNamespace(daily_realized_pnl=0.0, open_risk_R=0.0, daily_realized_R=0.0)

    def stream_events(self, strategy_id: str):
        return self.events

    async def submit_intent(self, intent):
        self.intents.append(intent)
        oms_order_id = intent.order.oms_order_id if intent.order else intent.target_oms_order_id
        return IntentReceipt(IntentResult.ACCEPTED, "intent", oms_order_id=oms_order_id)

    async def get_strategy_risk(self, strategy_id: str):
        return self.risk


def test_load_universe_cache_validates_required_fields(tmp_path):
    good_path = tmp_path / "good.jsonl"
    good_path.write_text(json.dumps(_cache_row()) + "\n", encoding="utf-8")
    rows = load_universe_cache(good_path)
    assert "AAPL" in rows
    assert rows["AAPL"].trend_ok is True

    bad_row = _cache_row()
    del bad_row["prior_close"]
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    try:
        load_universe_cache(bad_path)
    except CacheValidationError as exc:
        assert "prior_close" in str(exc)
    else:
        raise AssertionError("expected cache validation error")


def test_gap_policy_and_time_decay_gate():
    allowed, penalty, reason = apply_gap_policy(0.09, 82.0, 0.002)
    assert allowed is True
    assert penalty == 0.65
    assert reason == "gap_8_to_12pct"

    settings = StrategySettings()
    cache = {"AAPL": load_universe_cache(_write_cache_file([_cache_row()])).get("AAPL")}
    oms = _DummyOMS()
    engine = USORBEngine(oms_service=oms, cache=cache, account_id="DU1", nav=100_000.0, settings=settings)
    ctx = engine._ensure_context("AAPL")
    ctx.surge = 3.2
    ctx.or_pct = 0.015
    ctx.rvol_1m = 3.0
    ctx.vwap = 100.0
    ctx.spread_pct = 0.001
    ctx.quote = QuoteSnapshot(ts=_ts(10, 30), bid=100.0, ask=100.1, last=100.1, spread_pct=0.001)
    market = RegimeSnapshot(regime_ok=True, leader_breadth_ok=True)
    assert live_gate_pass(ctx, market, _ts(9, 51), settings) is True
    assert live_gate_pass(ctx, market, _ts(10, 31), settings) is False


def test_quality_score_adds_relative_strength_component():
    row = load_universe_cache(_write_cache_file([_cache_row()]))["AAPL"]
    market = RegimeSnapshot(regime_ok=True)

    good = SimpleNamespace(
        cached=row,
        surge=4.0,
        rvol_1m=3.0,
        imbalance_90s=0.35,
        acceptance=SimpleNamespace(pulled_back=True, held_support=True, reclaimed=True),
        spread_pct=0.001,
        relative_strength_5m=0.02,
        planned_entry=101.0,
        quote=None,
        vdm=DangerSnapshot(DangerState.SAFE),
    )
    weak = SimpleNamespace(
        cached=row,
        surge=4.0,
        rvol_1m=3.0,
        imbalance_90s=0.35,
        acceptance=SimpleNamespace(pulled_back=True, held_support=True, reclaimed=True),
        spread_pct=0.001,
        relative_strength_5m=-0.02,
        planned_entry=101.0,
        quote=None,
        vdm=DangerSnapshot(DangerState.SAFE),
    )

    assert quality_score(good, market) - quality_score(weak, market) == 8.0


def test_vdm_blocks_direct_danger_events_and_cautions_on_spread():
    row = load_universe_cache(_write_cache_file([_cache_row()]))["AAPL"]
    settings = StrategySettings()

    blocked_ctx = USORBEngine(
        oms_service=_DummyOMS(),
        cache={"AAPL": row},
        account_id="DU1",
        nav=100_000.0,
        settings=settings,
    )._ensure_context("AAPL")
    blocked_ctx.vwap = 100.0
    blocked_ctx.spread_pct = 0.001
    blocked_ctx.quote = QuoteSnapshot(ts=_ts(9, 50), bid=100.0, ask=100.1, last=100.1, past_limit=True, spread_pct=0.001)
    blocked_ctx.past_limit_events.append(_ts(9, 50))

    blocked = volatility_danger_snapshot(blocked_ctx, _ts(9, 50), settings)
    assert blocked.state == DangerState.BLOCKED
    assert blocked.cooldown_seconds == settings.halt_cooldown_s

    caution_ctx = USORBEngine(
        oms_service=_DummyOMS(),
        cache={"AAPL": row},
        account_id="DU1",
        nav=100_000.0,
        settings=settings,
    )._ensure_context("AAPL")
    caution_ctx.vwap = 100.0
    caution_ctx.spread_pct = 0.0038
    caution_ctx.quote = QuoteSnapshot(ts=_ts(9, 50), bid=100.0, ask=100.38, last=100.19, spread_pct=0.0038)

    caution = volatility_danger_snapshot(caution_ctx, _ts(9, 50), settings)
    assert caution.state == DangerState.CAUTION


def test_exit_signal_handles_partial_and_scratch():
    row = load_universe_cache(_write_cache_file([_cache_row()]))["AAPL"]
    market = RegimeSnapshot(flow_regime="mixed")

    partial_ctx = SimpleNamespace(
        cached=row,
        or_high=101.0,
        vwap=100.8,
        spread_pct=0.001,
        imbalance_90s=0.2,
        last_price=101.6,
        position=SimpleNamespace(
            entry_price=101.0,
            qty_open=100,
            final_stop=100.4,
            current_stop=100.4,
            entry_time=_ts(9, 52),
            initial_risk_per_share=0.6,
            max_favorable_price=101.2,
            partial_taken=False,
        ),
        quote=None,
        vdm=DangerSnapshot(DangerState.SAFE),
    )
    action, _ = exit_signal(partial_ctx, market, _ts(9, 55), StrategySettings())
    assert action == "partial"

    scratch_ctx = SimpleNamespace(
        cached=row,
        or_high=101.0,
        vwap=100.8,
        spread_pct=0.001,
        imbalance_90s=0.2,
        last_price=101.15,
        position=SimpleNamespace(
            entry_price=101.0,
            qty_open=100,
            final_stop=100.4,
            current_stop=100.4,
            entry_time=_ts(9, 52),
            initial_risk_per_share=0.6,
            max_favorable_price=101.25,
            partial_taken=True,
        ),
        quote=None,
        vdm=DangerSnapshot(DangerState.SAFE),
    )
    action, _ = exit_signal(scratch_ctx, market, _ts(10, 2), StrategySettings())
    assert action == "exit"


def test_compute_order_plan_scales_nav_based_notional_cap():
    row = load_universe_cache(_write_cache_file([_cache_row()]))["AAPL"]
    settings = StrategySettings()
    market = RegimeSnapshot(flow_regime="mixed")
    ctx = SimpleNamespace(
        or_high=100.0,
        last_price=100.2,
        vdm=DangerSnapshot(DangerState.SAFE),
        quality_score=80.0,
        spread=0.01,
        spread_pct=0.001,
        cached=row,
        vwap=99.8,
        acceptance=SimpleNamespace(retest_low=99.9),
        last5m_value=100_000_000.0,
        size_penalty=1.0,
        quote=SimpleNamespace(ask_size=1_000_000),
    )

    _, _, _, qty_small = compute_order_plan(
        ctx,
        PortfolioSnapshot(nav=50_000.0),
        market,
        _ts(9, 55),
        settings,
    )
    _, _, _, qty_large = compute_order_plan(
        ctx,
        PortfolioSnapshot(nav=100_000.0),
        market,
        _ts(9, 55),
        settings,
    )

    assert qty_small > 0
    assert qty_large > qty_small


def test_engine_state_machine_and_rearm():
    async def _run():
        cache = {"AAPL": load_universe_cache(_write_cache_file([_cache_row()]))["AAPL"]}
        oms = _DummyOMS()
        engine = USORBEngine(
            oms_service=oms,
            cache=cache,
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(leader_breadth_min=1),
        )
        ctx = engine._ensure_context("AAPL")
        ctx.state = State.CANDIDATE
        ctx.or_high = 100.0
        ctx.or_low = 99.0
        ctx.or_pct = 0.01005
        ctx.surge = 4.0
        ctx.rvol_1m = 3.1
        ctx.imbalance_90s = 0.35
        ctx.vwap = 99.8
        ctx.spread_pct = 0.001
        ctx.last5m_value = 5_000_000.0
        ctx.quote = QuoteSnapshot(ts=_ts(9, 51), bid=100.0, ask=100.01, last=100.01, ask_size=50_000, spread_pct=0.001)
        for hour, minute in ((9, 46), (9, 47), (9, 48), (9, 49), (9, 50)):
            ctx.bars.append(MinuteBar(ts=_ts(hour, minute), open=99.5, high=99.9, low=99.6, close=99.7, volume=250_000, dollar_value=25_000_000))
        engine._regime = RegimeSnapshot(regime_ok=True, leader_count=1, leader_breadth_ok=True, flow_regime="mixed")
        engine._portfolio = PortfolioSnapshot(nav=100_000.0)

        await engine._advance_symbol(ctx, _ts(9, 51))
        assert ctx.state == State.ARMED

        await engine._advance_symbol(ctx, _ts(9, 51))
        assert ctx.state == State.WAIT_BREAK

        ctx.quote.last = 100.2
        await engine._advance_symbol(ctx, _ts(9, 52))
        assert ctx.state == State.WAIT_ACCEPTANCE

        ctx.quote.last = 99.99
        await engine._advance_symbol(ctx, _ts(9, 53))
        ctx.quote.last = 100.05
        await engine._advance_symbol(ctx, _ts(9, 54))
        assert ctx.state == State.READY

        await engine._advance_symbol(ctx, _ts(9, 54))
        assert ctx.state == State.ORDER_SENT
        assert oms.intents[-1].intent_type.name == "NEW_ORDER"

        ctx.entry_order = PendingOrderState("oid-1", _ts(9, 54) - timedelta(seconds=26), "ENTRY", ctx.qty)
        await engine._advance_symbol(ctx, _ts(9, 55))
        assert oms.intents[-1].intent_type.name == "CANCEL_ORDER"

        cancel_event = SimpleNamespace(
            event_type=OMSEventType.ORDER_CANCELLED,
            oms_order_id="oid-1",
            payload={"symbol": "AAPL", "role": "ENTRY"},
            timestamp=_ts(9, 55),
        )
        await engine._handle_terminal(cancel_event)
        assert ctx.state == State.WAIT_BREAK
        assert ctx.rearms_used == 1

        ctx.state = State.ORDER_SENT
        ctx.entry_order = PendingOrderState("oid-2", _ts(9, 56), "ENTRY", ctx.qty)
        second_event = SimpleNamespace(
            event_type=OMSEventType.ORDER_CANCELLED,
            oms_order_id="oid-2",
            payload={"symbol": "AAPL", "role": "ENTRY"},
            timestamp=_ts(9, 56),
        )
        await engine._handle_terminal(second_event)
        assert ctx.state == State.DONE

    asyncio.run(_run())


def test_market_exit_requests_are_deduplicated_and_cleared_on_fill():
    async def _run():
        cache = {"AAPL": load_universe_cache(_write_cache_file([_cache_row()]))["AAPL"]}
        oms = _DummyOMS()
        engine = USORBEngine(
            oms_service=oms,
            cache=cache,
            account_id="DU1",
            nav=100_000.0,
            settings=StrategySettings(),
        )
        ctx = engine._ensure_context("AAPL")
        ctx.state = State.IN_POSITION
        ctx.quote = QuoteSnapshot(ts=_ts(10, 5), bid=101.2, ask=101.3, last=101.25, spread_pct=0.001)
        ctx.position = PositionState(
            entry_price=100.0,
            qty_entry=20,
            qty_open=20,
            final_stop=99.0,
            current_stop=99.0,
            entry_time=_ts(9, 55),
            initial_risk_per_share=1.0,
            max_favorable_price=100.0,
            max_adverse_price=100.0,
        )
        ctx.position.stop_order_id = "stop-1"

        await engine._submit_market_exit(ctx, 20, OrderRole.EXIT)
        first_exit = ctx.exit_order.oms_order_id
        exit_intents = [
            intent for intent in oms.intents
            if intent.order is not None and intent.order.role == OrderRole.EXIT
        ]
        assert len(exit_intents) == 1
        assert engine.open_order_count() == 1

        await engine._submit_market_exit(ctx, 20, OrderRole.EXIT)
        exit_intents = [
            intent for intent in oms.intents
            if intent.order is not None and intent.order.role == OrderRole.EXIT
        ]
        assert len(exit_intents) == 1

        fill_event = SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id=first_exit,
            payload={"symbol": "AAPL", "role": "EXIT", "qty": 20, "price": 101.2},
            timestamp=_ts(10, 6),
        )
        await engine._handle_fill(fill_event)

        assert engine.open_order_count() == 0
        assert ctx.exit_order is None
        assert ctx.position is None

    asyncio.run(_run())


def _write_cache_file(rows: list[dict]) -> Path:
    path = Path(__file__).with_name("tmp_cache.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path
