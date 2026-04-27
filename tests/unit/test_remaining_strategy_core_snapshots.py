from __future__ import annotations

from collections import deque
from datetime import date, datetime, timezone
import json
from types import SimpleNamespace

import pytest

from strategies.momentum.helix_v40.config import (
    PositionState as HelixV40PositionState,
    Setup as HelixV40Setup,
    SetupClass as HelixV40SetupClass,
    TF as HelixV40TF,
)
from strategies.momentum.helix_v40.core.serializers import (
    restore_state as restore_helix_v40_state,
    snapshot_state as snapshot_helix_v40_state,
)
from strategies.momentum.helix_v40.core.state import HelixV40CoreState
from strategies.momentum.helix_v40.engine import Helix4Engine
from strategies.momentum.vdub.core.serializers import (
    restore_state as restore_vdub_state,
    snapshot_state as snapshot_vdub_state,
)
from strategies.momentum.vdub.core.state import VdubCoreState
from strategies.momentum.vdub.engine import VdubNQv4Engine
from strategies.momentum.vdub.models import (
    DayCounters,
    Direction as VdubDirection,
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
from strategies.swing.breakout.core.serializers import (
    restore_state as restore_breakout_state,
    snapshot_state as snapshot_breakout_state,
)
from strategies.swing.breakout.core.state import BreakoutCoreState
from strategies.swing.breakout.engine import BreakoutEngine
from strategies.swing.breakout.models import CircuitBreakerState, SymbolCampaign

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


def test_breakout_core_serializer_roundtrip_preserves_tuple_key_maps() -> None:
    state = BreakoutCoreState(
        campaigns={"QQQ": SymbolCampaign(campaign_id=7, box_high=501.25)},
        circuit_breaker=CircuitBreakerState(halted=True),
        correlation_map={("QQQ", "GLD"): 0.75},
        order_requested_qty={"ENTRY-1": 3},
        last_decision_code="BREAKOUT",
    )

    snapshot = snapshot_breakout_state(state)
    restored = restore_breakout_state(json.loads(json.dumps(snapshot)))

    assert isinstance(restored.campaigns["QQQ"], SymbolCampaign)
    assert restored.campaigns["QQQ"].campaign_id == 7
    assert isinstance(restored.circuit_breaker, CircuitBreakerState)
    assert restored.circuit_breaker.halted is True
    assert restored.correlation_map == {("QQQ", "GLD"): 0.75}
    assert restored.order_requested_qty["ENTRY-1"] == 3


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


def test_helix_v40_core_serializer_roundtrip_preserves_set_and_tuple_state() -> None:
    detected_ts = datetime(2026, 4, 25, 13, 20, tzinfo=UTC)
    setup = HelixV40Setup(
        setup_id="SETUP-1",
        cls=HelixV40SetupClass.M,
        direction=1,
        tf_origin=HelixV40TF.H1,
        detected_ts=detected_ts,
        entry_stop=20260.0,
        stop0=20200.0,
    )
    state = HelixV40CoreState(
        positions=[
            HelixV40PositionState(
                pos_id="POS-1",
                direction=1,
                avg_entry=20250.0,
                contracts=1,
                unit1_risk_usd=120.0,
                origin_class=HelixV40SetupClass.M,
                origin_setup_id="SETUP-1",
                entry_ts=detected_ts,
                stop_price=20200.0,
            )
        ],
        pending_setups=[setup],
        placed_signatures={("M", 1)},
        sig_expiry={("M", 1): detected_ts},
        last_m_bar={1: 7},
        spread_recheck=[(setup, 2)],
        last_decision_code="HELIX_V40",
    )

    snapshot = snapshot_helix_v40_state(state)
    restored = restore_helix_v40_state(json.loads(json.dumps(snapshot)))

    assert isinstance(restored.positions[0], HelixV40PositionState)
    assert restored.positions[0].origin_class is HelixV40SetupClass.M
    assert isinstance(restored.pending_setups[0], HelixV40Setup)
    assert restored.pending_setups[0].tf_origin is HelixV40TF.H1
    assert restored.placed_signatures == {("M", 1)}
    assert restored.sig_expiry == {("M", 1): detected_ts}
    assert restored.last_m_bar == {1: 7}
    assert isinstance(restored.spread_recheck[0][0], HelixV40Setup)
    assert restored.spread_recheck[0][1] == 2


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
async def test_breakout_engine_snapshot_and_hydrate_preserve_campaign_state() -> None:
    engine = BreakoutEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    engine.campaigns["QQQ"] = SymbolCampaign(campaign_id=7, box_high=501.25)
    engine.circuit_breaker = CircuitBreakerState(halted=True)
    engine._last_decision_code = "CAMPAIGN_ACTIVE"

    restored = BreakoutEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    await restored.hydrate(engine.snapshot_state())

    assert isinstance(restored.campaigns["QQQ"], SymbolCampaign)
    assert restored.campaigns["QQQ"].campaign_id == 7
    assert restored.campaigns["QQQ"].box_high == pytest.approx(501.25)
    assert isinstance(restored.circuit_breaker, CircuitBreakerState)
    assert restored.circuit_breaker.halted is True
    assert restored.health_status()["last_decision_code"] == "CAMPAIGN_ACTIVE"


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


@pytest.mark.asyncio
async def test_helix_v40_engine_snapshot_and_hydrate_preserve_dataclass_state() -> None:
    instrument = SimpleNamespace(symbol="MNQ", point_value=2.0)
    engine = Helix4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=[instrument],
    )
    detected_ts = datetime(2026, 4, 25, 14, 0, tzinfo=UTC)
    engine.positions.positions = [
        HelixV40PositionState(
            pos_id="POS-1",
            direction=1,
            avg_entry=20250.0,
            contracts=1,
            unit1_risk_usd=120.0,
            origin_class=HelixV40SetupClass.M,
            origin_setup_id="SETUP-1",
            entry_ts=detected_ts,
            stop_price=20200.0,
        )
    ]
    engine.exec.pending_setups = [
        HelixV40Setup(
            setup_id="SETUP-1",
            cls=HelixV40SetupClass.M,
            direction=1,
            tf_origin=HelixV40TF.H1,
            detected_ts=detected_ts,
            entry_stop=20260.0,
            stop0=20200.0,
        )
    ]
    engine.risk.open_risk_r = 0.75
    engine.vol.vol_pct = 42.0
    engine._ts_history.extend([0.25, 0.5])
    engine._placed_signatures = {("M", 1)}
    engine._last_decision_code = "ARMED"

    restored = Helix4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=[instrument],
    )
    await restored.hydrate(engine.snapshot_state())

    assert isinstance(restored.positions.positions[0], HelixV40PositionState)
    assert restored.positions.positions[0].pos_id == "POS-1"
    assert isinstance(restored.exec.pending_setups[0], HelixV40Setup)
    assert restored.exec.pending_setups[0].setup_id == "SETUP-1"
    assert isinstance(restored._ts_history, deque)
    assert restored._ts_history.maxlen == 10
    assert list(restored._ts_history) == [0.25, 0.5]
    assert restored._placed_signatures == {("M", 1)}
    assert restored.health_status()["last_decision_code"] == "ARMED"


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
