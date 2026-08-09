from __future__ import annotations

from datetime import datetime, timezone

from backtests.momentum.auto.downturn.plugin import DownturnPlugin
from backtests.momentum.config_downturn import DownturnBacktestConfig
from backtests.momentum.engine.downturn_engine import (
    DownturnEngine as ReplayDownturnEngine,
    _ActivePosition,
)
from backtests.momentum.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimOrder,
)
from strategies.core.actions import CancelAction
from strategies.momentum.downturn.core.logic import on_bar, on_fill
from strategies.momentum.downturn.core.state import (
    DownturnCoreState,
    DownturnEntryRequest,
    DownturnFill,
)
from strategies.momentum.downturn.models import (
    ActivePosition,
    CompositeRegime,
    EngineTag,
    VolState,
    WorkingEntry,
)

UTC = timezone.utc


def _ts(minute: int = 0) -> datetime:
    return datetime(2026, 4, 25, 10, minute, tzinfo=UTC)


def _entry_request(order_id: str = "entry-1", *, submitted_bar_idx: int = 10, ttl_bars: int = 72):
    return DownturnEntryRequest(
        client_order_id=order_id,
        symbol="NQ",
        engine_tag=EngineTag.FADE,
        signal_class="vwap_rejection",
        qty=2,
        entry_price=18_990.0,
        stop0=19_010.0,
        price=18_990.0,
        limit_price=18_988.0,
        stop_price=18_990.0,
        submitted_bar_idx=submitted_bar_idx,
        ttl_bars=ttl_bars,
        composite_regime=CompositeRegime.EMERGING_BEAR,
        vol_state=VolState.NORMAL,
    )


def _working_entry(order_id: str = "OMS-1", *, submitted_bar_idx: int = 10, ttl_bars: int = 72):
    return WorkingEntry(
        oms_order_id=order_id,
        engine_tag=EngineTag.FADE,
        signal_class="vwap_rejection",
        entry_price=18_990.0,
        stop0=19_010.0,
        qty=2,
        submitted_bar_idx=submitted_bar_idx,
        ttl_bars=ttl_bars,
        composite_regime=CompositeRegime.EMERGING_BEAR,
        vol_state=VolState.NORMAL,
    )


def test_core_refuses_a_second_owned_entry() -> None:
    state = DownturnCoreState(symbol="NQ", working_entries=[_working_entry()])

    next_state, actions, events = on_bar(
        state,
        bar_count_5m=11,
        bar_ts=_ts(5),
        entry_request=_entry_request("entry-2", submitted_bar_idx=11),
    )

    assert actions == []
    assert len(next_state.working_entries) == 1
    assert events[-1].code == "ENTRY_BLOCKED_OWNED_ORDER"


def test_core_bar_ttl_expires_on_observed_bar_count_not_clock_time() -> None:
    state = DownturnCoreState(
        symbol="NQ",
        bar_count_5m=10,
        working_entries=[_working_entry(ttl_bars=72)],
    )

    before, actions, _ = on_bar(
        state,
        bar_count_5m=81,
        bar_ts=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        expire_entries=True,
    )
    assert len(before.working_entries) == 1
    assert actions == []

    expired, actions, events = on_bar(
        before,
        bar_count_5m=82,
        bar_ts=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        expire_entries=True,
    )
    assert expired.working_entries == []
    assert isinstance(actions[0], CancelAction)
    assert events[-1].code == "ENTRY_EXPIRED"


def test_replay_expiry_removes_exact_broker_order() -> None:
    engine = ReplayDownturnEngine("NQ", DownturnBacktestConfig())
    engine._core_state = DownturnCoreState(
        symbol="NQ",
        bar_count_5m=10,
        working_entries=[_working_entry()],
    )
    engine.broker.submit_order(
        SimOrder(
            order_id="OMS-1",
            symbol="NQ",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            qty=2,
            stop_price=18_990.0,
            limit_price=18_988.0,
            submit_time=_ts(),
            ttl_hours=0,
            tag="entry",
        )
    )

    engine._expire_working_entries(81, datetime(2026, 5, 1, tzinfo=UTC))
    assert engine._has_working_entry()
    engine._expire_working_entries(82, datetime(2026, 8, 1, tzinfo=UTC))
    assert not engine._has_working_entry()
    assert engine.broker.pending_orders == []


def test_explicit_market_exit_fill_closes_once() -> None:
    engine = ReplayDownturnEngine("NQ", DownturnBacktestConfig())
    engine._position = _ActivePosition(
        engine_tag=EngineTag.FADE,
        signal_class="vwap_rejection",
        entry_price=19_000.0,
        stop0=19_010.0,
        qty=2,
        entry_time=_ts(),
        entry_bar_idx=10,
        composite_regime=CompositeRegime.EMERGING_BEAR,
        vol_state=VolState.NORMAL,
        in_correction=True,
        predator=False,
        tp_schedule=[],
    )
    engine._core_state = DownturnCoreState(
        symbol="NQ",
        position=ActivePosition(
            engine_tag=EngineTag.FADE,
            signal_class="vwap_rejection",
            trade_id="OMS-ENTRY",
            entry_price=19_000.0,
            stop0=19_010.0,
            qty=2,
            remaining_qty=2,
            entry_oms_order_id="OMS-ENTRY",
            stop_oms_order_id="OMS-STOP",
        ),
    )
    order = SimOrder(
        order_id="OMS-EXIT",
        symbol="NQ",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=2,
        tag="vwap_failure",
    )
    fill = FillResult(
        order=order,
        status=FillStatus.FILLED,
        fill_price=18_990.0,
        fill_time=_ts(5),
        commission=1.24,
    )

    engine._handle_fill(fill, _ts(5), 18_990.0, 10_000.0, [])
    engine._handle_fill(fill, _ts(5), 18_990.0, 10_000.0, [])

    assert engine._position is None
    assert engine._core_state.position is None
    assert len(engine._trades) == 1
    assert engine._trades[0].exit_type == "vwap_failure"


def test_core_partial_exit_decrements_then_final_exit_clears() -> None:
    state = DownturnCoreState(
        symbol="NQ",
        position=ActivePosition(
            engine_tag=EngineTag.FADE,
            signal_class="vwap_rejection",
            trade_id="OMS-ENTRY",
            entry_price=19_000.0,
            stop0=19_010.0,
            qty=3,
            remaining_qty=3,
            entry_oms_order_id="OMS-ENTRY",
            stop_oms_order_id="OMS-STOP",
        ),
    )
    partial, _, events = on_fill(
        state,
        DownturnFill("OMS-TP1", 18_990.0, 1, exit_type="tp1"),
    )
    assert partial.position is not None
    assert partial.position.remaining_qty == 2
    assert events[-1].code == "PARTIAL_EXIT_FILLED"

    closed, _, events = on_fill(
        partial,
        DownturnFill("OMS-EXIT", 18_995.0, 2, exit_type="vwap_failure"),
    )
    assert closed.position is None
    assert events[-1].code == "EXIT_FILLED"


def test_downturn_provenance_covers_live_and_shared_core(tmp_path) -> None:
    provenance = DownturnPlugin(tmp_path, max_workers=1).build_provenance()
    paths = {path for item in provenance.items for path in item.paths}

    assert "strategies/momentum/downturn/engine.py" in paths
    assert "strategies/momentum/downturn/core/logic.py" in paths
