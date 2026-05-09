from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from strategies.momentum.vdub import config as vdub_config
from strategies.momentum.vdub.core.logic import build_core_state as build_vdub_runtime_state
from strategies.momentum.vdub.core.logic import on_bar as on_vdub_bar
from strategies.momentum.vdub.core.logic import on_fill as on_vdub_fill
from strategies.momentum.vdub.core.logic import on_order_update as on_vdub_order_update
from strategies.momentum.vdub.core.serializers import (
    restore_state as restore_vdub_state,
    snapshot_state as snapshot_vdub_state,
)
from strategies.momentum.vdub.core.state import VdubCoreState, VdubEntryFillContext, VdubFill
from strategies.momentum.vdub.engine import VdubNQv4Engine
from strategies.momentum.vdub.models import (
    DayCounters,
    Direction as VdubDirection,
    EntryType as VdubEntryType,
    EventBlockState,
    RegimeState,
    VolState,
    WorkingEntry,
)
from strategies.stock.iaric.core.serializers import (
    restore_state as restore_iaric_state,
    snapshot_state as snapshot_iaric_state,
)
from strategies.stock.iaric.models import IntradayStateSnapshot, PBSymbolState
from strategies.swing.akc_helix.core.serializers import (
    restore_state as restore_akc_helix_state,
    snapshot_state as snapshot_akc_helix_state,
)
from strategies.swing.akc_helix.core.state import AKCHelixCoreState
from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.akc_helix.models import (
    CircuitBreakerState as AKCHelixCircuitBreakerState,
    PivotStore,
    Regime,
    TFState as AKCHelixTFState,
)
from strategies.swing.atrss.core.serializers import (
    restore_state as restore_atrss_state,
    snapshot_state as snapshot_atrss_state,
)
from strategies.swing.atrss.core.state import ATRSSCoreState
from strategies.swing.atrss.engine import ATRSSEngine
from strategies.swing.atrss.models import (
    BreakoutArmState,
    Candidate,
    CandidateType,
    Direction as ATRSSDirection,
    HaltState,
)
UTC = timezone.utc


def test_atrss_core_serializer_roundtrip_preserves_typed_state() -> None:
    state = ATRSSCoreState(
        pending_reverses=[
            Candidate(
                symbol="QQQ",
                type=CandidateType.REVERSE,
                direction=ATRSSDirection.LONG,
                trigger_price=512.5,
                time=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
            )
        ],
        halt_states={
            "QQQ": HaltState(
                is_halted=True,
                halt_detected_at=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
                queued_stop_updates=[("STOP-1", 505.0)],
            )
        },
        breakout_arm_states={
            "QQQ": BreakoutArmState(
                breakout_armed_dir=ATRSSDirection.SHORT,
                breakout_arm_low=503.25,
            )
        },
        last_decision_code="ATRSS",
    )

    restored = restore_atrss_state(snapshot_atrss_state(state))

    assert isinstance(restored.pending_reverses[0], Candidate)
    assert restored.pending_reverses[0].type is CandidateType.REVERSE
    assert restored.pending_reverses[0].direction is ATRSSDirection.LONG
    assert isinstance(restored.halt_states["QQQ"], HaltState)
    assert restored.halt_states["QQQ"].queued_stop_updates == [("STOP-1", 505.0)]
    assert isinstance(restored.breakout_arm_states["QQQ"], BreakoutArmState)
    assert restored.breakout_arm_states["QQQ"].breakout_armed_dir is ATRSSDirection.SHORT


def test_akc_helix_core_serializer_roundtrip_preserves_nested_runtime_models() -> None:
    state = AKCHelixCoreState(
        tf_states={"CL": {"1H": AKCHelixTFState(tf_label="1H", atr=3.2)}},
        pivots={"CL": {"1H": PivotStore(max_size=25)}},
        circuit_breakers={"CL": AKCHelixCircuitBreakerState(consecutive_stops=2)},
        regime_4h={"CL": Regime.BULL},
        prev_regimes={"CL": Regime.CHOP},
        last_decision_code="HELIX",
    )

    restored = restore_akc_helix_state(snapshot_akc_helix_state(state))

    assert isinstance(restored.tf_states["CL"]["1H"], AKCHelixTFState)
    assert restored.tf_states["CL"]["1H"].atr == pytest.approx(3.2)
    assert isinstance(restored.pivots["CL"]["1H"], PivotStore)
    assert restored.pivots["CL"]["1H"].max_size == 25
    assert isinstance(restored.circuit_breakers["CL"], AKCHelixCircuitBreakerState)
    assert restored.regime_4h["CL"] is Regime.BULL
    assert restored.prev_regimes["CL"] is Regime.CHOP


def test_vdub_core_serializer_roundtrip_preserves_typed_runtime_state() -> None:
    state = VdubCoreState(
        regime=RegimeState(daily_trend=1, vol_state=VolState.HIGH),
        counters=DayCounters(long_fills=2, breaker_hit=True),
        working_entries={
            "ENTRY-1": WorkingEntry(
                oms_order_id="ENTRY-1",
                direction=VdubDirection.LONG,
                qty=2,
                filter_decisions=[{"gate": "trend", "passed": True}],
            )
        },
        event_state=EventBlockState(
            blocked=True,
            block_end_ts=datetime(2026, 4, 25, 13, 15, tzinfo=UTC),
            cooldown_remaining=3,
        ),
        recent_wins=[True, False],
        last_decision_code="VDUB",
    )

    restored = restore_vdub_state(snapshot_vdub_state(state))

    assert isinstance(restored.regime, RegimeState)
    assert restored.regime.vol_state is VolState.HIGH
    assert isinstance(restored.counters, DayCounters)
    assert restored.counters.long_fills == 2
    assert isinstance(restored.working_entries["ENTRY-1"], WorkingEntry)
    assert restored.working_entries["ENTRY-1"].direction is VdubDirection.LONG
    assert restored.event_state.blocked is True
    assert restored.recent_wins == [True, False]




@pytest.mark.asyncio
async def test_atrss_engine_snapshot_and_hydrate_preserve_runtime_types() -> None:
    engine = ATRSSEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    engine.breakout_arm_states["QQQ"] = BreakoutArmState(breakout_arm_high=512.0)
    engine.halt_states["QQQ"] = HaltState(
        is_halted=True,
        halt_detected_at=datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
    )
    engine._last_decision_code = "HALT_GUARDED"

    restored = ATRSSEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    await restored.hydrate(engine.snapshot_state())

    assert isinstance(restored.breakout_arm_states["QQQ"], BreakoutArmState)
    assert restored.breakout_arm_states["QQQ"].breakout_arm_high == pytest.approx(512.0)
    assert isinstance(restored.halt_states["QQQ"], HaltState)
    assert restored.halt_states["QQQ"].is_halted is True
    assert restored.health_status()["last_decision_code"] == "HALT_GUARDED"


@pytest.mark.asyncio
async def test_akc_helix_engine_snapshot_and_hydrate_preserve_tf_state() -> None:
    engine = HelixEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    engine.tf_states["CL"] = {"1H": AKCHelixTFState(tf_label="1H", atr=3.2)}
    engine.circuit_breakers["CL"] = AKCHelixCircuitBreakerState(consecutive_stops=2)
    engine._last_decision_code = "SETUP_QUEUED"

    restored = HelixEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    await restored.hydrate(engine.snapshot_state())

    assert isinstance(restored.tf_states["CL"]["1H"], AKCHelixTFState)
    assert restored.tf_states["CL"]["1H"].atr == pytest.approx(3.2)
    assert isinstance(restored.circuit_breakers["CL"], AKCHelixCircuitBreakerState)
    assert restored.circuit_breakers["CL"].consecutive_stops == 2
    assert restored.health_status()["last_decision_code"] == "SETUP_QUEUED"


@pytest.mark.asyncio
async def test_vdub_engine_snapshot_and_hydrate_preserve_regime_state() -> None:
    instrument = SimpleNamespace(symbol="NQ")
    engine = VdubNQv4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=[instrument],
    )
    engine.regime.daily_trend = 1
    engine.counters.long_fills = 2
    engine._recent_wins = [True, False]
    engine._last_decision_code = "ENTRY_ARMED"

    restored = VdubNQv4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=[instrument],
    )
    await restored.hydrate(engine.snapshot_state())

    assert type(restored.regime) is type(engine.regime)
    assert restored.regime.daily_trend == 1
    assert type(restored.counters) is type(engine.counters)
    assert restored.counters.long_fills == 2
    assert restored._recent_wins == [True, False]
    assert restored.health_status()["last_decision_code"] == "ENTRY_ARMED"



def test_iaric_core_serializer_roundtrip_preserves_intraday_snapshot_shape() -> None:
    snapshot = IntradayStateSnapshot(
        trade_date=date(2026, 4, 25),
        saved_at=datetime(2026, 4, 25, 15, 30, tzinfo=UTC),
        symbols=[PBSymbolState(symbol="MSFT", stage="SCANNING")],
        last_decision_code="SNAPSHOT",
        meta={"active_symbols": ["MSFT"]},
    )

    restored = restore_iaric_state(snapshot_iaric_state(snapshot))

    assert isinstance(restored, IntradayStateSnapshot)
    assert isinstance(restored.symbols[0], PBSymbolState)
    assert restored.symbols[0].symbol == "MSFT"
    assert restored.meta["active_symbols"] == ["MSFT"]




@pytest.mark.asyncio
async def test_vdub_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch) -> None:
    instrument = SimpleNamespace(symbol="NQ", point_value=vdub_config.NQ_SPEC["point_value"])

    async def _fake_submit_intent(*_args, **_kwargs):
        return SimpleNamespace(oms_order_id=None)

    engine = VdubNQv4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(
            stream_events=lambda *_args, **_kwargs: None,
            submit_intent=_fake_submit_intent,
        ),
        instruments=[instrument],
    )
    working_entry = WorkingEntry(
        oms_order_id="OMS-V1",
        entry_type=VdubEntryType.TYPE_A,
        direction=VdubDirection.LONG,
        stop_entry=20010.0,
        qty=2,
        initial_stop=19980.0,
    )
    engine.working_entries["OMS-V1"] = working_entry

    initial_state = restore_vdub_state(snapshot_vdub_state(build_vdub_runtime_state(engine)))

    await engine._on_fill("OMS-V1", {"price": 20010.0, "qty": 2, "commission": 1.25})

    wrapper_snapshot = snapshot_vdub_state(build_vdub_runtime_state(engine))
    fill_time = engine.positions[0].entry_time
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    VdubFill(
                        oms_order_id="OMS-V1",
                        fill_price=20010.0,
                        fill_qty=2,
                        fill_time=fill_time,
                        point_value=vdub_config.NQ_SPEC["point_value"],
                        commission=1.25,
                        entry_context=VdubEntryFillContext(working_entry=working_entry),
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: on_vdub_bar(state, **payload),
        on_order_update=on_vdub_order_update,
        on_fill=on_vdub_fill,
    )

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert snapshot_vdub_state(replay.state) == wrapper_snapshot
