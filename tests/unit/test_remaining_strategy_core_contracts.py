from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from backtests.shared.parity.decision_capture import normalize_decision_stream
from strategies.momentum.helix_v40.core import logic as helix_v40_logic
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitExit, SubmitProtectiveStop
from strategies.momentum.helix_v40.config import PositionState, Setup, SetupClass, SetupState
from strategies.momentum.helix_v40.core.state import (
    HelixV40CoreState,
    HelixV40EntryArmed,
    HelixV40EntryFillContext,
    HelixV40Fill,
    HelixV40OrderUpdate,
    HelixV40StopUpdateRequest,
)
from strategies.momentum.vdub.core import logic as vdub_logic
from strategies.momentum.vdub.core.state import (
    VdubCoreState,
    VdubEntryFillContext,
    VdubEntrySubmitted,
    VdubFill,
    VdubOrderUpdate,
)
from strategies.momentum.vdub.models import Direction, EntryType, SessionWindow, WorkingEntry
from strategies.stock.iaric.core import logic as iaric_logic
from strategies.stock.iaric.core.state import IARICBarInput, IARICFill, IARICOrderUpdate
from strategies.stock.iaric.models import IntradayStateSnapshot
from strategies.swing.akc_helix.core import logic as akc_helix_logic
from strategies.swing.akc_helix.core.state import (
    AKCHelixBarInput,
    AKCHelixCoreState,
    AKCHelixEntryRequest,
    AKCHelixFill,
    AKCHelixOrderUpdate,
)
from strategies.swing.akc_helix.models import (
    Direction as AKCHelixDirection,
    SetupClass as AKCHelixSetupClass,
    SetupInstance as AKCHelixSetup,
    SetupState as AKCHelixSetupState,
)
from strategies.swing.atrss.core import logic as atrss_logic
from strategies.swing.atrss.core.state import ATRSSBarInput, ATRSSCoreState, ATRSSEntryRequest, ATRSSFill, ATRSSOrderUpdate
from strategies.swing.atrss.models import Candidate, CandidateType, Direction as ATRSSDirection, HourlyState
from strategies.swing.breakout.core import logic as breakout_logic
from strategies.swing.breakout.core.state import BreakoutBarInput, BreakoutCoreState, BreakoutEntryRequest, BreakoutFill, BreakoutOrderUpdate
from strategies.swing.breakout.models import Direction as BreakoutDirection, EntryType as BreakoutEntryType, ExitTier, SetupInstance as BreakoutSetup, SetupState as BreakoutSetupState

UTC = timezone.utc


def _iaric_state() -> IntradayStateSnapshot:
    return IntradayStateSnapshot(
        trade_date=date(2026, 4, 25),
        saved_at=datetime(2026, 4, 25, 15, 30, tzinfo=UTC),
        symbols=[],
        last_decision_code="IDLE",
    )


def _last_bar_marker(state) -> str | datetime | None:
    if hasattr(state, "last_bar_ts"):
        return state.last_bar_ts
    if hasattr(state, "meta") and isinstance(state.meta, dict):
        return state.meta.get("last_bar_ts")
    return None


def _atrss_candidate() -> Candidate:
    return Candidate(
        symbol="QQQ",
        type=CandidateType.PULLBACK,
        direction=ATRSSDirection.LONG,
        trigger_price=510.25,
        initial_stop=503.5,
        qty=3,
        signal_bar=HourlyState(time=datetime(2026, 4, 25, 13, 0, tzinfo=UTC)),
    )


def _breakout_setup() -> BreakoutSetup:
    return BreakoutSetup(
        setup_id="BK-1",
        symbol="SPY",
        direction=BreakoutDirection.LONG,
        entry_type=BreakoutEntryType.A_AVWAP_RETEST,
        state=BreakoutSetupState.NEW,
        created_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        campaign_id=7,
        box_version=2,
        entry_price=510.5,
        stop0=504.0,
        current_stop=504.0,
        final_risk_dollars=130.0,
        shares_planned=5,
        oca_group="BKO-1",
        exit_tier=ExitTier.ALIGNED,
        regime_at_entry="BULL_TREND",
    )


def _akc_helix_setup() -> AKCHelixSetup:
    return AKCHelixSetup(
        setup_id="HELIX-1",
        symbol="QQQ",
        setup_class=AKCHelixSetupClass.CLASS_A,
        direction=AKCHelixDirection.LONG,
        origin_tf="4H",
        state=AKCHelixSetupState.NEW,
        created_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        bos_level=505.5,
        stop0=499.0,
        current_stop=499.0,
        qty_planned=3,
        oca_group="HELIX-OCA",
    )


def test_atrss_core_realistic_flow_preserves_shared_lifecycle_invariants() -> None:
    state, actions, events = atrss_logic.on_bar(
        ATRSSCoreState(),
        bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        entry_request=ATRSSEntryRequest(
            client_order_id="ENTRY-1",
            symbol="QQQ",
            candidate=_atrss_candidate(),
            limit_price=510.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert events[0].code == "ENTRY_REQUESTED"
    bar_marker = _last_bar_marker(state)

    state.pending_orders["ENTRY-1"] = {
        "symbol": "QQQ",
        "type": CandidateType.PULLBACK,
        "direction": ATRSSDirection.LONG,
        "trigger_price": 510.25,
        "initial_stop": 503.5,
        "qty": 3,
    }
    state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="ENTRY-1",
            fill_price=510.5,
            fill_qty=3,
            fill_time=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert events[0].code == "ENTRY_FILLED"
    assert _last_bar_marker(state) == bar_marker


def test_breakout_core_realistic_flow_preserves_shared_lifecycle_invariants() -> None:
    state, actions, events = breakout_logic.on_bar(
        BreakoutCoreState(),
        bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        entry_request=BreakoutEntryRequest(client_order_id="ENTRY-1", setup=_breakout_setup()),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert events[0].code == "ENTRY_REQUESTED"
    bar_marker = _last_bar_marker(state)

    state.order_to_setup["ENTRY-1"] = "BK-1"
    state.order_kind["ENTRY-1"] = "primary_entry"
    state.order_requested_qty["ENTRY-1"] = 5
    state, actions, events = breakout_logic.on_fill(
        state,
        BreakoutFill(
            oms_order_id="ENTRY-1",
            fill_price=510.75,
            fill_qty=5,
            fill_time=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert events[0].code == "ENTRY_FILLED"
    assert _last_bar_marker(state) == bar_marker


def test_akc_helix_core_realistic_flow_preserves_shared_lifecycle_invariants() -> None:
    state, actions, events = akc_helix_logic.on_bar(
        AKCHelixCoreState(),
        bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        entry_request=AKCHelixEntryRequest(
            client_order_id="ENTRY-1",
            setup=_akc_helix_setup(),
            order_type="STOP_LIMIT",
            limit_price=505.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert events[0].code == "ENTRY_REQUESTED"
    bar_marker = _last_bar_marker(state)

    state.order_to_setup["ENTRY-1"] = "HELIX-1"
    state, actions, events = akc_helix_logic.on_fill(
        state,
        AKCHelixFill(
            oms_order_id="ENTRY-1",
            fill_price=505.75,
            fill_qty=3,
            fill_time=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
            order_role="entry",
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert events[0].code == "ENTRY_FILLED"
    assert _last_bar_marker(state) == bar_marker


@pytest.mark.parametrize(
    ("state", "logic", "bar_payload", "order_payload", "fill_payload"),
    [
        (
            ATRSSCoreState(),
            atrss_logic,
            ATRSSBarInput(
                symbol="QQQ",
                timeframe="1h",
                bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                decision_code="ATRSS_BAR",
                decision_details={"symbol": "QQQ"},
            ),
            ATRSSOrderUpdate(
                oms_order_id="OMS-1",
                symbol="QQQ",
                timeframe="1h",
                timestamp=datetime(2026, 4, 25, 13, 1, tzinfo=UTC),
                decision_code="ATRSS_ORDER",
            ),
            ATRSSFill(
                oms_order_id="OMS-1",
                symbol="QQQ",
                timeframe="1h",
                fill_time=datetime(2026, 4, 25, 13, 2, tzinfo=UTC),
                decision_code="ATRSS_FILL",
            ),
        ),
        (
            BreakoutCoreState(),
            breakout_logic,
            BreakoutBarInput(
                symbol="QQQ",
                timeframe="1h",
                bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                decision_code="BREAKOUT_BAR",
            ),
            BreakoutOrderUpdate(
                oms_order_id="OMS-1",
                symbol="QQQ",
                timeframe="1h",
                timestamp=datetime(2026, 4, 25, 13, 1, tzinfo=UTC),
                decision_code="BREAKOUT_ORDER",
            ),
            BreakoutFill(
                oms_order_id="OMS-1",
                symbol="QQQ",
                timeframe="1h",
                fill_time=datetime(2026, 4, 25, 13, 2, tzinfo=UTC),
                decision_code="BREAKOUT_FILL",
            ),
        ),
        (
            AKCHelixCoreState(),
            akc_helix_logic,
            AKCHelixBarInput(
                symbol="CL",
                timeframe="1h",
                bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                decision_code="HELIX_BAR",
            ),
            AKCHelixOrderUpdate(
                oms_order_id="OMS-1",
                symbol="CL",
                timeframe="1h",
                timestamp=datetime(2026, 4, 25, 13, 1, tzinfo=UTC),
                decision_code="HELIX_ORDER",
            ),
            AKCHelixFill(
                oms_order_id="OMS-1",
                symbol="CL",
                timeframe="1h",
                fill_time=datetime(2026, 4, 25, 13, 2, tzinfo=UTC),
                decision_code="HELIX_FILL",
            ),
        ),
        # These strategies now have real shared cores; keep a generic lifecycle
        # contract smoke test here alongside their dedicated core suites.
        (
            _iaric_state(),
            iaric_logic,
            IARICBarInput(
                symbol="MSFT",
                timeframe="5m",
                bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                decision_code="IARIC_BAR",
            ),
            IARICOrderUpdate(
                oms_order_id="OMS-1",
                symbol="MSFT",
                timeframe="5m",
                timestamp=datetime(2026, 4, 25, 13, 1, tzinfo=UTC),
                decision_code="IARIC_ORDER",
            ),
            IARICFill(
                oms_order_id="OMS-1",
                symbol="MSFT",
                timeframe="5m",
                fill_time=datetime(2026, 4, 25, 13, 2, tzinfo=UTC),
                decision_code="IARIC_FILL",
            ),
        ),
    ],
)
def test_remaining_strategy_cores_preserve_shared_lifecycle_contract(
    state,
    logic,
    bar_payload,
    order_payload,
    fill_payload,
) -> None:
    state, actions, events = logic.on_bar(state, bar_payload)
    assert actions == []
    assert normalize_decision_stream(events)[0]["code"].endswith("_BAR")
    assert state.last_decision_code.endswith("_BAR")
    bar_marker = _last_bar_marker(state)
    assert bar_marker is not None

    state, actions, events = logic.on_order_update(state, order_payload)
    assert actions == []
    assert normalize_decision_stream(events)[0]["code"].endswith("_ORDER")
    assert state.last_decision_code.endswith("_ORDER")
    assert _last_bar_marker(state) == bar_marker

    state, actions, events = logic.on_fill(state, fill_payload)
    assert actions == []
    assert normalize_decision_stream(events)[0]["code"].endswith("_FILL")
    assert state.last_decision_code.endswith("_FILL")
    assert _last_bar_marker(state) == bar_marker


# ── Helix_v40 real core tests ─────────────────────────────────────


def _make_setup(**overrides) -> Setup:
    """Minimal Setup for testing core logic."""
    defaults = dict(
        setup_id="S-001",
        direction=1,
        cls=SetupClass.M,
        tf_origin="1h",
        detected_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        state=SetupState.PENDING,
        stop0=19980.0,
        entry_stop=19980.0,
        armed_risk_r=1.0,
        unit1_risk_usd=100.0,
        alignment_score=0.75,
        entry_oms_id="",
        catchup_oms_id="",
    )
    defaults.update(overrides)
    return Setup(**defaults)


def test_helix_v40_on_bar_armed_entry_emits_submit_event():
    """Armed entry via on_bar records signature and emits ENTRY_SUBMITTED."""
    state = HelixV40CoreState()
    bar_ts = datetime(2026, 4, 25, 14, 0, tzinfo=UTC)
    setup = _make_setup()

    armed = HelixV40EntryArmed(
        setup=setup,
        contracts=2,
        risk_r=1.0,
        signature=("MNQ", 1, "M", "20260425"),
        sig_expiry_ts=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        bar_idx_1h=100,
    )

    next_state, actions, events = helix_v40_logic.on_bar(
        state, bar_ts=bar_ts, armed_entries=[armed],
    )

    # Signature tracked
    assert armed.signature in next_state.placed_signatures
    # Pending risk incremented
    assert next_state.pending_risk_r == pytest.approx(1.0)
    # M bar index tracked
    assert next_state.last_m_bar.get(1) == 100
    # Event emitted
    assert len(events) == 1
    assert events[0].code == "ENTRY_ARMED"
    assert events[0].details["cls"] == "M"
    # Bar timestamp updated
    assert next_state.last_bar_ts == bar_ts


def test_helix_v40_on_bar_stop_update_emits_replace_action():
    """Stop update via on_bar emits ReplaceProtectiveStop action."""
    pos = PositionState(
        pos_id="S-001", direction=1, avg_entry=20000.0, contracts=2,
        stop_price=19980.0, stop_oms_id="STOP-001",
        unit1_risk_usd=100.0, origin_class=SetupClass.M,
        origin_setup_id="S-001", entry_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        entry_contracts=2,
    )
    state = HelixV40CoreState(positions=[pos])

    su = HelixV40StopUpdateRequest(
        pos_id="S-001", stop_price=19990.0, reason="TRAIL",
    )

    next_state, actions, events = helix_v40_logic.on_bar(
        state, bar_ts=datetime(2026, 4, 25, 14, 0, tzinfo=UTC), stop_updates=[su],
    )

    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].stop_price == 19990.0
    assert next_state.positions[0].stop_price == 19990.0
    assert len(events) == 1
    assert events[0].code == "STOP_REPLACEMENT_REQUESTED"


def test_helix_v40_on_fill_entry_creates_position_and_emits_stop():
    """Entry fill creates position, promotes risk, emits SubmitExit for stop."""
    setup = _make_setup(state=SetupState.PENDING, entry_oms_id="ENT-001")
    state = HelixV40CoreState(
        pending_setups=[setup],
        pending_risk_r=1.0,
        dir_risk_r={1: 1.0},
    )

    fill = HelixV40Fill(
        oms_order_id="ENT-001",
        fill_price=20000.0,
        fill_qty=2,
        fill_time=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
        point_value=2.0,
        entry_context=HelixV40EntryFillContext(setup=setup),
    )

    next_state, actions, events = helix_v40_logic.on_fill(state, fill)

    # Position created
    assert len(next_state.positions) == 1
    assert next_state.positions[0].pos_id == "S-001"
    assert next_state.positions[0].avg_entry == 20000.0
    # Pending setup consumed
    assert len(next_state.pending_setups) == 0
    # Risk promoted: pending released, open incremented
    assert next_state.pending_risk_r == pytest.approx(0.0)
    assert next_state.open_risk_r > 0.0
    # SubmitExit for protective stop
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitExit)
    assert actions[0].order_type == "STOP"
    # Event
    assert events[0].code == "ENTRY_FILLED"


def test_helix_v40_on_fill_catastrophic_emits_flatten():
    """Catastrophic entry fill emits FlattenPosition and releases risk."""
    setup = _make_setup(state=SetupState.PENDING, entry_oms_id="ENT-002")
    state = HelixV40CoreState(
        pending_setups=[setup],
        pending_risk_r=1.0,
        dir_risk_r={1: 1.0},
    )

    fill = HelixV40Fill(
        oms_order_id="ENT-002",
        fill_price=19950.0,
        fill_qty=2,
        fill_time=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
        point_value=2.0,
        entry_context=HelixV40EntryFillContext(setup=setup, is_catastrophic=True),
    )

    next_state, actions, events = helix_v40_logic.on_fill(state, fill)

    assert len(actions) == 1
    assert isinstance(actions[0], FlattenPosition)
    assert actions[0].reason == "CATASTROPHIC_FILL"
    assert next_state.pending_risk_r == pytest.approx(0.0)
    assert len(next_state.positions) == 0
    assert events[0].code == "CATASTROPHIC_FILL"


def test_helix_v40_on_order_update_terminal_releases_pending_risk():
    """Terminal order update for entry releases pending risk and removes setup."""
    setup = _make_setup(state=SetupState.PENDING, entry_oms_id="ENT-003")
    state = HelixV40CoreState(
        pending_setups=[setup],
        pending_risk_r=1.0,
        dir_risk_r={1: 1.0},
    )

    update = HelixV40OrderUpdate(
        oms_order_id="ENT-003",
        status="cancelled",
        timestamp=datetime(2026, 4, 25, 14, 5, tzinfo=UTC),
    )

    next_state, actions, events = helix_v40_logic.on_order_update(state, update)

    assert len(next_state.pending_setups) == 0
    assert next_state.pending_risk_r == pytest.approx(0.0)
    assert next_state.dir_risk_r[1] == pytest.approx(0.0)
    assert events[0].code == "ENTRY_CANCELLED"


# ── Vdub real core tests ─────────────────────────────────────────


def _make_working_entry(**overrides) -> WorkingEntry:
    """Minimal WorkingEntry for testing Vdub core logic."""
    defaults = dict(
        oms_order_id="WE-001",
        entry_type=EntryType.TYPE_A,
        direction=Direction.LONG,
        stop_entry=20010.0,
        limit_entry=0.0,
        qty=2,
        initial_stop=19980.0,
        session=SessionWindow.RTH,
    )
    defaults.update(overrides)
    return WorkingEntry(**defaults)


def test_vdub_on_bar_entry_submitted_registers_working_entry():
    """Entry submitted via on_bar registers in working_entries."""
    state = VdubCoreState()
    bar_ts = datetime(2026, 4, 25, 14, 0, tzinfo=UTC)
    we = _make_working_entry()

    submitted = VdubEntrySubmitted(
        working_entry=we, oms_order_id="OMS-V1", bar_idx=42,
    )

    next_state, actions, events = vdub_logic.on_bar(
        state, bar_ts=bar_ts, entry_submitted=submitted,
    )

    assert "OMS-V1" in next_state.working_entries
    assert next_state.working_entries["OMS-V1"].direction == Direction.LONG
    assert len(events) == 1
    assert events[0].code == "ENTRY_SUBMITTED"
    assert next_state.last_bar_ts == bar_ts


def test_vdub_on_fill_entry_creates_position_and_emits_stop():
    """Entry fill creates position and requests protective stop."""
    we = _make_working_entry(oms_order_id="OMS-V1")
    state = VdubCoreState(
        working_entries={"OMS-V1": we},
    )

    fill = VdubFill(
        oms_order_id="OMS-V1",
        fill_price=20010.0,
        fill_qty=2,
        fill_time=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
        point_value=2.0,
        entry_context=VdubEntryFillContext(working_entry=we),
    )

    next_state, actions, events = vdub_logic.on_fill(state, fill)

    # Position created
    assert len(next_state.positions) == 1
    assert next_state.positions[0].entry_price == 20010.0
    # Working entry consumed
    assert "OMS-V1" not in next_state.working_entries
    # SubmitExit for protective stop
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitExit)
    assert actions[0].order_type == "STOP"
    # Event
    assert events[0].code == "ENTRY_FILLED"


def test_vdub_on_order_update_terminal_removes_working_entry():
    """Terminal order update for entry removes working entry."""
    we = _make_working_entry(oms_order_id="OMS-V2")
    state = VdubCoreState(
        working_entries={"OMS-V2": we},
    )

    update = VdubOrderUpdate(
        oms_order_id="OMS-V2",
        status="cancelled",
        timestamp=datetime(2026, 4, 25, 14, 5, tzinfo=UTC),
    )

    next_state, actions, events = vdub_logic.on_order_update(state, update)

    assert "OMS-V2" not in next_state.working_entries
    assert events[0].code == "ENTRY_CANCELLED"
