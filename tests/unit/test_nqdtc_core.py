from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from backtests.momentum.engine import nqdtc_engine as nqdtc_backtest_engine
from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.engine.nqdtc_engine import NQDTCEngine as BacktestNQDTCEngine
from backtests.momentum.engine.sim_broker import (
    FillStatus as SimFillStatus,
    OrderSide as SimOrderSide,
    OrderType as SimOrderType,
    SimBroker,
    SimOrder,
)
from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from libs.oms.models.events import OMSEventType
from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitProtectiveStop,
)
from strategies.momentum.nqdtc import config as nqdtc_config
from strategies.momentum.nqdtc.core import entry_decision as nqdtc_entry_decision
from strategies.momentum.nqdtc.engine import NQDTCEngine
from strategies.momentum.nqdtc import signals as nqdtc_signals
from strategies.momentum.nqdtc import stops as nqdtc_stops
from strategies.momentum.nqdtc.core.logic import on_authorization, on_bar, on_fill, on_order_update
from strategies.momentum.nqdtc.core.serializers import restore_state, snapshot_state
from strategies.momentum.nqdtc.core.state import (
    NQDTCCoreState,
    NQDTCAuthorization,
    NQDTCEntryFillContext,
    NQDTCEntryRequest,
    NQDTCFill,
    NQDTCOrderUpdate,
    NQDTCSimpleRequest,
)
from strategies.momentum.nqdtc.models import (
    CompositeRegime,
    Direction,
    EntrySubtype,
    ExitTier,
    PositionState,
    Session,
    SessionEngineState,
    TPLevel,
    WorkingOrder,
)

UTC = timezone.utc


def test_nqdtc_contextual_score_filter_blocks_weak_wide_low_rvol(monkeypatch) -> None:
    monkeypatch.setattr(nqdtc_config, "WEAK_SCORE_BAND_FILTER_ENABLED", True)
    monkeypatch.setattr(nqdtc_config, "WEAK_SCORE_BAND_MAX_BOX_WIDTH", 225.0)
    monkeypatch.setattr(nqdtc_config, "WEAK_SCORE_BAND_MIN_RVOL", 1.75)

    ok, reason = nqdtc_signals.contextual_score_filter_pass(
        score=2.75,
        box_width=260.0,
        rvol=1.30,
    )

    assert ok is False
    assert reason == "weak_score_context"


def test_nqdtc_b_entry_regime_permission_is_config_driven(monkeypatch) -> None:
    monkeypatch.setattr(nqdtc_config, "B_ALLOW_ALIGNED", True)
    monkeypatch.setattr(nqdtc_config, "B_ALLOW_RANGE", False)
    assert nqdtc_signals.b_entry_regime_allowed("Aligned") is True
    assert nqdtc_signals.b_entry_regime_allowed("Range") is False

    monkeypatch.setattr(nqdtc_config, "B_ALLOW_RANGE", True)
    assert nqdtc_signals.b_entry_regime_allowed("Range") is True


def test_nqdtc_a_entry_context_gate_is_config_driven(monkeypatch) -> None:
    monkeypatch.setattr(nqdtc_config, "A_MAX_BOX_WIDTH", 225.0)
    monkeypatch.setattr(nqdtc_config, "A_MIN_SCORE", 0.0)
    monkeypatch.setattr(nqdtc_config, "A_BLOCK_WEAK_SCORE_BAND", False)

    ok, reason = nqdtc_signals.a_entry_context_allowed(score=2.5, box_width=226.0)
    assert ok is False
    assert reason == "a_box_width"

    monkeypatch.setattr(nqdtc_config, "A_MAX_BOX_WIDTH", 0.0)
    monkeypatch.setattr(nqdtc_config, "A_MIN_SCORE", 3.0)
    ok, reason = nqdtc_signals.a_entry_context_allowed(score=2.75, box_width=200.0)
    assert ok is False
    assert reason == "a_min_score"

    monkeypatch.setattr(nqdtc_config, "A_MIN_SCORE", 0.0)
    monkeypatch.setattr(nqdtc_config, "A_BLOCK_WEAK_SCORE_BAND", True)
    ok, reason = nqdtc_signals.a_entry_context_allowed(score=2.75, box_width=200.0)
    assert ok is False
    assert reason == "a_weak_score_band"

    ok, reason = nqdtc_signals.a_entry_context_allowed(score=3.0, box_width=200.0)
    assert ok is True
    assert reason == ""


def test_nqdtc_tp1_only_cap_mode_is_configurable(monkeypatch) -> None:
    monkeypatch.setattr(nqdtc_config, "TP1_ONLY_CAP_MODE", "range_degraded")
    assert nqdtc_stops.should_cap_tp1_only("NORMAL", "RANGE") is True
    assert nqdtc_stops.should_cap_tp1_only("DEGRADED", "TRENDING") is True

    monkeypatch.setattr(nqdtc_config, "TP1_ONLY_CAP_MODE", "degraded_only")
    assert nqdtc_stops.should_cap_tp1_only("NORMAL", "RANGE") is False
    assert nqdtc_stops.should_cap_tp1_only("DEGRADED", "TRENDING") is True

    monkeypatch.setattr(nqdtc_config, "TP1_ONLY_CAP_MODE", "off")
    assert nqdtc_stops.should_cap_tp1_only("DEGRADED", "RANGE") is False


def test_nqdtc_mfe_ratcheted_stop_locks_configured_tiers(monkeypatch) -> None:
    monkeypatch.setattr(nqdtc_config, "MFE_RATCHET_TIERS_ENABLED", True)
    monkeypatch.setattr(nqdtc_config, "MFE_RATCHET_T1_R", 2.0)
    monkeypatch.setattr(nqdtc_config, "MFE_RATCHET_T1_LOCK_R", 0.80)
    monkeypatch.setattr(nqdtc_config, "MFE_RATCHET_T2_R", 3.0)
    monkeypatch.setattr(nqdtc_config, "MFE_RATCHET_T2_LOCK_R", 1.35)
    monkeypatch.setattr(nqdtc_config, "MFE_RATCHET_T3_R", 4.0)
    monkeypatch.setattr(nqdtc_config, "MFE_RATCHET_T3_LOCK_R", 2.0)

    long_stop = nqdtc_stops.compute_mfe_ratcheted_stop(
        Direction.LONG,
        entry_price=100.0,
        initial_r_points=10.0,
        peak_r_initial=3.25,
        tick_size=0.25,
    )
    short_stop = nqdtc_stops.compute_mfe_ratcheted_stop(
        Direction.SHORT,
        entry_price=100.0,
        initial_r_points=10.0,
        peak_r_initial=4.25,
        tick_size=0.25,
    )

    assert long_stop == pytest.approx(113.50)
    assert short_stop == pytest.approx(80.00)


@pytest.mark.parity_smoke
def test_marketable_ioc_limit_fill_model_is_immediate_and_never_queued() -> None:
    broker = SimBroker()
    ts = datetime(2026, 4, 27, 14, 0, tzinfo=UTC)
    fill = broker.fill_marketable_ioc_limit(
        SimOrder(
            order_id="B-fill",
            symbol="MNQ",
            side=SimOrderSide.BUY,
            order_type=SimOrderType.LIMIT,
            qty=2,
            limit_price=100.50,
        ),
        ts,
        O=100.0,
        H=110.0,
        L=99.0,
        C=100.0,
        tick_size=0.25,
    )
    reject = broker.fill_marketable_ioc_limit(
        SimOrder(
            order_id="B-reject",
            symbol="MNQ",
            side=SimOrderSide.BUY,
            order_type=SimOrderType.LIMIT,
            qty=2,
            limit_price=100.00,
        ),
        ts,
        O=100.0,
        H=110.0,
        L=99.0,
        C=100.0,
        tick_size=0.25,
    )

    assert fill.status is SimFillStatus.FILLED
    assert fill.fill_price == 100.25
    assert fill.slippage_ticks == 1
    assert fill.filled_at_open is False
    assert reject.status is SimFillStatus.REJECTED
    assert broker.pending_orders == []


@pytest.mark.parity_smoke
def test_nqdtc_b_sweep_ioc_reject_clears_working_order_without_queueing() -> None:
    engine = BacktestNQDTCEngine(
        "MNQ",
        NQDTCBacktestConfig(symbols=["MNQ"], fixed_qty=1, track_shadows=False),
    )
    ts = datetime(2026, 4, 27, 14, 0, tzinfo=UTC)

    engine._submit_entry(
        direction=Direction.LONG,
        qty=1,
        order_type=SimOrderType.LIMIT,
        subtype=EntrySubtype.B_SWEEP,
        stop_for_risk=90.0,
        quality_mult=1.0,
        limit_price=100.0,
        bar_time=ts,
        sess=engine.rth,
        ioc_bar=(100.0, 110.0, 99.0, 100.0),
    )

    assert engine._entries_placed == 1
    assert engine._entries_filled == 0
    assert engine._working == {}
    assert engine.broker.pending_orders == []


@pytest.mark.parity_smoke
def test_nqdtc_b_sweep_ioc_fill_uses_entry_fill_path_without_queueing_entry() -> None:
    engine = BacktestNQDTCEngine(
        "MNQ",
        NQDTCBacktestConfig(symbols=["MNQ"], fixed_qty=1, track_shadows=False),
    )
    ts = datetime(2026, 4, 27, 14, 0, tzinfo=UTC)

    engine._submit_entry(
        direction=Direction.LONG,
        qty=1,
        order_type=SimOrderType.LIMIT,
        subtype=EntrySubtype.B_SWEEP,
        stop_for_risk=90.0,
        quality_mult=1.0,
        limit_price=100.50,
        bar_time=ts,
        sess=engine.rth,
        ioc_bar=(100.0, 110.0, 99.0, 100.0),
    )

    assert engine._entries_placed == 1
    assert engine._entries_filled == 1
    assert engine._active is not None
    assert engine._active.pos.entry_subtype is EntrySubtype.B_SWEEP
    assert engine._active.pos.entry_price == 100.25
    assert all(order.tag != EntrySubtype.B_SWEEP.value for order in engine.broker.pending_orders)


@pytest.mark.parity_smoke
def test_nqdtc_b_sweep_backtest_submits_ioc_intent(monkeypatch) -> None:
    engine = BacktestNQDTCEngine(
        "MNQ",
        NQDTCBacktestConfig(symbols=["MNQ"], fixed_qty=1, track_shadows=False),
    )
    ts = datetime(2026, 4, 27, 14, 0, tzinfo=UTC)
    captured: list[NQDTCEntryRequest] = []
    real_on_bar = nqdtc_backtest_engine.nqdtc_core_logic.on_bar

    def _capture_on_bar(state, **payload):
        if payload.get("entry_request") is not None:
            captured.append(payload["entry_request"])
        return real_on_bar(state, **payload)

    monkeypatch.setattr(nqdtc_backtest_engine.nqdtc_core_logic, "on_bar", _capture_on_bar)

    engine._submit_entry(
        direction=Direction.LONG,
        qty=1,
        order_type=SimOrderType.LIMIT,
        subtype=EntrySubtype.B_SWEEP,
        stop_for_risk=90.0,
        quality_mult=1.0,
        limit_price=100.50,
        bar_time=ts,
        sess=engine.rth,
        ioc_bar=(100.0, 110.0, 99.0, 100.0),
    )

    assert captured[0].tif == "IOC"
    assert captured[0].order_type == "LIMIT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (
            EntrySubtype.A_RETEST,
            [EntrySubtype.A_RETEST, EntrySubtype.A_LATCH],
        ),
        (EntrySubtype.B_SWEEP, [EntrySubtype.B_SWEEP]),
        (EntrySubtype.C_STANDARD, [EntrySubtype.C_STANDARD]),
    ],
)
async def test_live_nqdtc_shared_entry_core_preserves_drawdown_throttle_products(
    monkeypatch,
    requested: EntrySubtype,
    expected: list[EntrySubtype],
) -> None:
    engine = _live_nqdtc_engine_with_half_dd()
    session = _live_nqdtc_session()
    captured: list[dict] = []

    async def _capture_submit(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(oms_order_id=f"OMS-{len(captured)}")

    monkeypatch.setattr(engine, "_submit_order", _capture_submit)
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "classify_daily_support",
        lambda *_args, **_kwargs: (False, False),
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "compute_composite_regime",
        lambda *_args, **_kwargs: CompositeRegime.NEUTRAL,
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "regime_hard_block",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "a_entry_context_allowed",
        lambda **_kwargs: (True, ""),
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "entry_a_trigger",
        lambda *_args, **_kwargs: (100.0, 101.0),
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "b_entry_regime_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "entry_b_trigger",
        lambda *_args, **_kwargs: requested is EntrySubtype.B_SWEEP,
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.signals,
        "entry_c_hold_check",
        lambda *_args, **_kwargs: (
            requested is EntrySubtype.C_STANDARD,
            100.0,
        ),
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.stops,
        "compute_initial_stop",
        lambda _subtype, _direction, price, *_args, **_kwargs: price - 10.0,
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.sizing,
        "compute_contracts",
        lambda *_args, **_kwargs: 5,
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.sizing,
        "friction_ok",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        nqdtc_entry_decision.sizing,
        "tp1_viable",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(nqdtc_config, "A_ENTRY_RETEST_ENABLED", requested is EntrySubtype.A_RETEST)
    monkeypatch.setattr(nqdtc_config, "A_ENTRY_LATCH_ENABLED", requested is EntrySubtype.A_RETEST)
    monkeypatch.setattr(nqdtc_config, "C_HOLD_BARS", 3)

    engine._bars_5m = {
        "close": np.array([99.0, 100.0, 100.0]),
        "high": np.array([101.0, 101.0, 101.0]),
        "low": np.array([98.0, 99.0, 99.0]),
    }
    engine._bars_15m = {"close": np.array([])}
    engine._bars_daily = {"ema50": np.array([]), "atr14": np.array([])}
    session.disp_hist.data = [1.0] * 12
    session.last_disp_metric = 2.0
    session.last_disp_threshold = 1.0

    await engine._evaluate_entries(
        session,
        Session.RTH,
        datetime(2026, 4, 27, 14, 0, tzinfo=UTC),
    )

    assert [order["subtype"] for order in captured] == expected
    assert [order["qty"] for order in captured] == [2] * len(expected)
    if requested is EntrySubtype.B_SWEEP:
        assert captured[0]["tif"] == "IOC"


def _entry_request() -> NQDTCEntryRequest:
    return NQDTCEntryRequest(
        client_order_id="entry-1",
        symbol="NQ",
        subtype=EntrySubtype.A_RETEST,
        direction=Direction.LONG,
        qty=2,
        stop_for_risk=19950.0,
        order_type="STOP_LIMIT",
        price=19975.0,
        limit_price=19975.0,
        stop_price=19976.0,
        oca_group="OCA-1",
        quality_mult=1.2,
        submitted_bar_idx=15,
        ttl_bars=4,
    )


def _live_nqdtc_engine_with_half_dd() -> NQDTCEngine:
    engine = NQDTCEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={
            "NQ": SimpleNamespace(symbol="NQ", tick_size=0.25, point_value=20.0),
        },
    )
    engine._throttle.dd_pct = 0.10
    return engine


def _live_nqdtc_session() -> SessionEngineState:
    session = SessionEngineState(session=Session.RTH)
    session.atr14_30m = 2.0
    session.box.box_high = 110.0
    session.box.box_low = 90.0
    session.box.box_mid = 100.0
    session.breakout.breakout_bar_high = 112.0
    session.breakout.breakout_bar_low = 88.0
    session.breakout.active = True
    session.breakout.direction = Direction.LONG
    return session


def test_nqdtc_degenerate_authorization_matches_compatibility_proposal() -> None:
    request = _entry_request()
    event_time = datetime(2026, 4, 25, 11, 0, tzinfo=UTC)
    compatibility_state, compatibility_actions, compatibility_events = on_bar(
        NQDTCCoreState(symbol="NQ"),
        bar_count_5m=15,
        bar_ts=event_time,
        entry_request=request,
    )
    causal_state, causal_actions, causal_events = on_bar(
        NQDTCCoreState(symbol="NQ"),
        bar_count_5m=15,
        bar_ts=event_time,
        entry_request=request,
        authorization_required=True,
    )

    assert causal_actions == compatibility_actions
    assert causal_events == compatibility_events
    assert compatibility_state.pending_entries == {}
    assert causal_state.pending_entries == {request.client_order_id: request}

    causal_state, authorized_actions, events = on_authorization(
        causal_state,
        NQDTCAuthorization(
            client_order_id=request.client_order_id,
            approved=True,
            approved_qty=request.qty,
            requested_qty=request.qty,
            timestamp=event_time,
            symbol=request.symbol,
        ),
    )

    assert authorized_actions == compatibility_actions
    assert events[-1].code == "ENTRY_AUTHORIZED"
    assert causal_state.working_orders == compatibility_state.working_orders == []


@pytest.mark.parametrize("approved", [True, False], ids=["approved", "denied"])
def test_nqdtc_authorization_feedback_precedes_working_order_state(approved: bool) -> None:
    request = _entry_request()
    event_time = datetime(2026, 4, 25, 11, 0, tzinfo=UTC)
    state, proposals, events = on_bar(
        NQDTCCoreState(symbol="NQ"),
        bar_count_5m=15,
        bar_ts=event_time,
        entry_request=request,
        authorization_required=True,
    )

    assert [event.code for event in events] == ["ENTRY_REQUESTED"]
    assert proposals[0].qty == request.qty
    assert state.working_orders == []
    assert state.pending_entries[request.client_order_id] == request

    state, submissions, events = on_authorization(
        state,
        NQDTCAuthorization(
            client_order_id=request.client_order_id,
            approved=approved,
            approved_qty=1 if approved else 0,
            requested_qty=request.qty,
            timestamp=event_time,
            symbol="NQ",
            denial_reason="fixture_denial" if not approved else "",
            portfolio_decision_ref="portfolio-1",
        ),
    )

    if not approved:
        assert submissions == []
        assert state.pending_entries == {}
        assert state.working_orders == []
        assert [event.code for event in events] == ["ENTRY_DENIED"]
        return

    assert submissions[0].qty == 1
    assert state.pending_entries[request.client_order_id].qty == 1
    state, _, events = on_order_update(
        state,
        NQDTCOrderUpdate(
            oms_order_id="OMS-1",
            status="accepted",
            timestamp=event_time,
            order_role="entry",
            accepted_entry=replace(request, qty=1),
        ),
    )
    assert state.pending_entries == {}
    assert state.working_orders[0].qty == 1
    assert [event.code for event in events] == ["ENTRY_SUBMITTED"]


def test_nqdtc_core_entry_and_exit_roundtrip() -> None:
    state = NQDTCCoreState(symbol="NQ")

    state, actions, events = on_bar(
        state,
        bar_count_5m=15,
        bar_ts=datetime(2026, 4, 25, 11, 0, tzinfo=UTC),
        entry_request=_entry_request(),
    )
    assert isinstance(actions[0], SubmitEntry)
    assert events[-1].code == "ENTRY_REQUESTED"

    state, _, events = on_order_update(
        state,
        NQDTCOrderUpdate(
            oms_order_id="OMS-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 11, 1, tzinfo=UTC),
            order_role="entry",
            accepted_entry=_entry_request(),
        ),
    )
    assert state.working_orders[0].oms_order_id == "OMS-1"
    assert events[-1].code == "ENTRY_SUBMITTED"

    fill_context = NQDTCEntryFillContext(
        exit_tier=ExitTier.ALIGNED,
        tp_levels=[TPLevel(r_target=1.0, pct=0.5, qty=1)],
        mm_level=20020.0,
        mm_reached=False,
        box_high_at_entry=19980.0,
        box_low_at_entry=19940.0,
        box_mid_at_entry=19960.0,
        entry_session=Session.RTH,
        tp1_only_cap=False,
    )
    state, actions, events = on_fill(
        state,
        NQDTCFill(
            oms_order_id="OMS-1",
            fill_price=19977.0,
            fill_qty=2,
            fill_time=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
            entry_context=fill_context,
        ),
    )
    assert state.position.open is True
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].side == "SELL"
    assert events[-1].code == "ENTRY_FILLED"

    state, _, _ = on_order_update(
        state,
        NQDTCOrderUpdate(
            oms_order_id="STOP-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 11, 3, tzinfo=UTC),
            order_role="stop",
        ),
    )
    assert state.position.stop_oms_order_id == "STOP-1"

    snapshot = snapshot_state(state)
    restored = restore_state(snapshot)
    assert restored.position.open is True
    assert restored.position.entry_price == 19977.0
    assert restored.working_orders == []

    state, _, events = on_fill(
        state,
        NQDTCFill(
            oms_order_id="STOP-1",
            fill_price=19950.0,
            fill_qty=2,
            fill_time=datetime(2026, 4, 25, 11, 4, tzinfo=UTC),
            exit_type="stop",
        ),
    )
    assert state.position.open is False
    assert events[-1].code == "EXIT_FILLED"


def test_nqdtc_core_stop_cancel_and_flatten_transitions() -> None:
    state = NQDTCCoreState(
        symbol="NQ",
        position=PositionState(
            open=True,
            symbol="NQ",
            direction=Direction.LONG,
            entry_subtype=EntrySubtype.A_RETEST,
            entry_price=19977.0,
            stop_price=19950.0,
            initial_stop_price=19950.0,
            qty=2,
            qty_open=2,
            stop_oms_order_id="STOP-1",
        ),
        working_orders=[
            WorkingOrder(
                oms_order_id="OMS-2",
                subtype=EntrySubtype.C_STANDARD,
                direction=Direction.LONG,
                price=19990.0,
                qty=1,
                submitted_bar_idx=14,
                ttl_bars=6,
            )
        ],
    )

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 11, 5, tzinfo=UTC),
        stop_update=NQDTCSimpleRequest(reason="trail", price=19960.0, qty=2),
        cancel_order_ids=["OMS-2"],
        flatten_request=NQDTCSimpleRequest(reason="risk_off"),
    )

    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert isinstance(actions[1], CancelAction)
    assert isinstance(actions[2], FlattenPosition)
    assert [event.code for event in events] == [
        "STOP_REPLACEMENT_REQUESTED",
        "ORDER_CANCEL_REQUESTED",
        "FLATTEN_REQUESTED",
    ]


def test_nqdtc_core_ignores_unmatched_fill_without_exit_context() -> None:
    state = NQDTCCoreState(
        symbol="NQ",
        position=PositionState(
            open=True,
            symbol="NQ",
            direction=Direction.LONG,
            entry_subtype=EntrySubtype.A_RETEST,
            entry_price=19977.0,
            stop_price=19950.0,
            initial_stop_price=19950.0,
            qty=2,
            qty_open=2,
            stop_oms_order_id="STOP-1",
        ),
    )

    next_state, actions, events = on_fill(
        state,
        NQDTCFill(
            oms_order_id="UNRELATED",
            fill_price=19990.0,
            fill_qty=1,
            fill_time=datetime(2026, 4, 25, 11, 6, tzinfo=UTC),
        ),
    )

    assert next_state.position.open is True
    assert next_state.position.stop_oms_order_id == "STOP-1"
    assert actions == []
    assert events == []


def test_nqdtc_replay_driver_produces_normalized_decision_stream() -> None:
    fill_context = NQDTCEntryFillContext(
        exit_tier=ExitTier.ALIGNED,
        tp_levels=[TPLevel(r_target=1.0, pct=0.5, qty=1)],
        mm_level=20020.0,
        mm_reached=False,
        box_high_at_entry=19980.0,
        box_low_at_entry=19940.0,
        box_mid_at_entry=19960.0,
        entry_session=Session.RTH,
        tp1_only_cap=False,
    )
    steps = [
        ReplayStep(
            bar_input={
                "bar_count_5m": 15,
                "bar_ts": datetime(2026, 4, 25, 11, 0, tzinfo=UTC),
                "entry_request": _entry_request(),
            }
        ),
        ReplayStep(
            order_updates=[
                NQDTCOrderUpdate(
                    oms_order_id="OMS-1",
                    status="accepted",
                    timestamp=datetime(2026, 4, 25, 11, 1, tzinfo=UTC),
                    order_role="entry",
                    accepted_entry=_entry_request(),
                )
            ]
        ),
        ReplayStep(
            fills=[
                NQDTCFill(
                    oms_order_id="OMS-1",
                    fill_price=19977.0,
                    fill_qty=2,
                    fill_time=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
                    entry_context=fill_context,
                )
            ]
        ),
    ]

    result = run_replay(
        NQDTCCoreState(symbol="NQ"),
        steps=steps,
        on_bar=lambda state, payload: on_bar(state, **payload),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    codes = [event["code"] for event in normalize_decision_stream(result.events)]
    assert codes == ["ENTRY_REQUESTED", "ENTRY_SUBMITTED", "ENTRY_FILLED"]


class _DummyIB:
    pass


class _DummyOMS:
    pass


@pytest.mark.asyncio
async def test_nqdtc_engine_snapshot_and_hydrate_preserve_wrapper_contract(tmp_path) -> None:
    engine = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._position = PositionState(
        open=True,
        symbol="NQ",
        direction=Direction.LONG,
        entry_subtype=EntrySubtype.A_RETEST,
        entry_price=19977.0,
        stop_price=19950.0,
        initial_stop_price=19950.0,
        qty=2,
        qty_open=2,
        stop_oms_order_id="STOP-1",
        tp_levels=[TPLevel(r_target=1.0, pct=0.5, qty=1)],
    )
    engine._working_orders = [
        WorkingOrder(
            oms_order_id="OMS-2",
            subtype=EntrySubtype.C_STANDARD,
            direction=Direction.LONG,
            price=19990.0,
            qty=1,
            submitted_bar_idx=14,
            ttl_bars=6,
            oca_group="OCA-1",
        )
    ]
    engine._bar_count_5m = 15
    engine._last_decision_code = "PROTECTIVE_STOP_SUBMITTED"
    engine._last_decision_details = {"stop_oms_order_id": "STOP-1"}
    engine._last_bar_ts = datetime(2026, 4, 25, 11, 3, tzinfo=UTC)

    snapshot = engine.snapshot_state()

    restored = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    await restored.hydrate(snapshot)

    assert restored.health_status()["last_decision_code"] == "PROTECTIVE_STOP_SUBMITTED"
    assert restored._position.open is True
    assert restored._position.stop_oms_order_id == "STOP-1"
    assert restored._working_orders[0].oms_order_id == "OMS-2"
    assert restored._bar_count_5m == 15
    assert restored._symbol == "NQ"


@pytest.mark.asyncio
async def test_nqdtc_engine_entry_fill_routes_through_shared_core(tmp_path) -> None:
    engine = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._working_orders = [
        WorkingOrder(
            oms_order_id="OMS-1",
            subtype=EntrySubtype.A_RETEST,
            direction=Direction.LONG,
            price=19975.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            is_limit=True,
            quality_mult=1.2,
            stop_for_risk=19950.0,
            expected_fill_price=19975.0,
        ),
    ]
    engine._bar_count_5m = 15
    engine._bars_daily = {"ema50": [], "atr14": []}

    await engine._on_fill(
        SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id="OMS-1",
            payload={"price": 19977.0, "qty": 2},
            timestamp=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
        )
    )

    assert engine._position.open is True
    assert engine._position.entry_price == 19977.0
    assert engine._position.entry_subtype == EntrySubtype.A_RETEST
    assert [order.oms_order_id for order in engine._working_orders] == []
    assert engine.health_status()["last_decision_code"] == "ENTRY_FILLED"


@pytest.mark.asyncio
async def test_nqdtc_engine_rejected_filled_entry_clears_working_orders(tmp_path) -> None:
    engine = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._working_orders = [
        WorkingOrder(
            oms_order_id="OMS-1",
            subtype=EntrySubtype.A_RETEST,
            direction=Direction.LONG,
            price=19975.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            is_limit=True,
            quality_mult=1.2,
            stop_for_risk=19976.5,
            expected_fill_price=19975.0,
        ),
        WorkingOrder(
            oms_order_id="OMS-2",
            subtype=EntrySubtype.A_LATCH,
            direction=Direction.LONG,
            price=19976.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            quality_mult=1.2,
            stop_for_risk=19976.5,
            expected_fill_price=19976.0,
        ),
    ]
    engine._bar_count_5m = 15
    engine._bars_daily = {"ema50": [], "atr14": []}

    await engine._on_fill(
        SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id="OMS-1",
            payload={"price": 19977.0, "qty": 2},
            timestamp=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
        )
    )

    assert engine._position.open is False
    assert engine._working_orders == []
    assert engine.health_status()["last_decision_code"] == "ENTRY_FILL_REJECTED"
    assert engine.health_status()["last_decision_details"]["reason"] == "MIN_STOP_DISTANCE"


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_nqdtc_live_wrapper_entry_fill_matches_replay_core_state(tmp_path, monkeypatch) -> None:
    engine = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._working_orders = [
        WorkingOrder(
            oms_order_id="OMS-1",
            subtype=EntrySubtype.A_RETEST,
            direction=Direction.LONG,
            price=19975.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            is_limit=True,
            quality_mult=1.2,
            stop_for_risk=19950.0,
            expected_fill_price=19975.0,
        ),
    ]
    engine._bar_count_5m = 15
    engine._bars_daily = {"ema50": [], "atr14": []}

    async def _fake_place_stop(
        _stop_price: float,
        _qty: int,
        _direction,
        *,
        client_order_id: str = "",
        event_time: datetime | None = None,
    ) -> None:
        return None

    async def _fake_cancel_order(_oms_order_id: str) -> None:
        return None

    monkeypatch.setattr(engine, "_place_protective_stop", _fake_place_stop)
    monkeypatch.setattr(engine, "_cancel_order", _fake_cancel_order)
    monkeypatch.setattr(engine, "_log_telemetry", lambda *_args, **_kwargs: None)

    initial_state = restore_state(snapshot_state(engine._build_core_state()))
    fill_time = datetime(2026, 4, 25, 11, 2, tzinfo=UTC)

    await engine._on_fill(
        SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id="OMS-1",
            payload={"price": 19977.0, "qty": 2},
            timestamp=fill_time,
        )
    )

    wrapper_snapshot = snapshot_state(engine._build_core_state())
    position = engine._position
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    NQDTCFill(
                        oms_order_id="OMS-1",
                        fill_price=19977.0,
                        fill_qty=2,
                        fill_time=fill_time,
                        entry_context=NQDTCEntryFillContext(
                            exit_tier=position.exit_tier,
                            tp_levels=position.tp_levels,
                            mm_level=position.mm_level,
                            mm_reached=position.mm_reached,
                            box_high_at_entry=position.box_high_at_entry,
                            box_low_at_entry=position.box_low_at_entry,
                            box_mid_at_entry=position.box_mid_at_entry,
                            entry_session=position.entry_session,
                            tp1_only_cap=position.tp1_only_cap,
                            r_dollars=position.R_dollars,
                        ),
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: on_bar(state, **payload),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert snapshot_state(replay.state) == wrapper_snapshot
