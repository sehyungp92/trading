from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from libs.oms.models.events import OMSEventType

from strategies.core.actions import SubmitEntry, SubmitMarketExit
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core.daily_residual import (
    DailyResidualFill,
    apply_daily_residual_fill,
    build_daily_residual_execution_state,
    plan_daily_residual_session_orders,
    plan_daily_residual_forced_exit,
)
from strategies.stock.iaric.diagnostics import JsonlDiagnostics
from strategies.stock.iaric.models import (
    HeldPositionDirective,
    RegimeSnapshot,
    WatchlistArtifact,
    WatchlistItem,
)
from strategies.stock.iaric.residual_engine import IARICDailyResidualEngine
from strategies.stock.iaric.artifact_store import load_intraday_state
from strategies.stock.iaric import research_generator as residual_research_generator


def _item() -> WatchlistItem:
    return WatchlistItem(
        symbol="MSFT",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        regime_score=0.5,
        regime_tier="B",
        regime_risk_multiplier=1.0,
        sector_score=0.0,
        sector_rank_weight=1.0,
        sponsorship_score=0.0,
        sponsorship_state="RESIDUAL",
        persistence=1.0,
        intensity_z=2.0,
        accel_z=0.0,
        rs_percentile=1.0,
        leader_pass=True,
        trend_pass=True,
        trend_strength=0.0,
        earnings_risk_flag=False,
        blacklist_flag=False,
        anchor_date=date(2025, 1, 2),
        anchor_type="FROZEN_FACTOR_RESIDUAL",
        acceptance_pass=True,
        avwap_ref=100.0,
        avwap_band_lower=100.0,
        avwap_band_upper=100.0,
        daily_atr_estimate=2.0,
        intraday_atr_seed=0.02,
        daily_rank=1.0,
        tradable_flag=True,
        conviction_bucket="RESIDUAL_RANKED",
        conviction_multiplier=1.0,
        recommended_risk_r=0.0035,
        daily_signal_score=80.0,
        previous_close=100.0,
        sleeve_id="daily_residual_reversion",
        residual_factor_model="market_only",
        residual_formation_sessions=3,
        residual_z=-2.0,
        residual_volatility=0.02,
        residual_initial_dislocation_r=3.464101615,
        residual_anchor_price=106.0,
        residual_remaining_room_r=1.0,
        residual_score_components={"residual_extremeness": 0.8},
        residual_lane_id="test_peer_lane",
        residual_model_contract_version="frozen_residual_model_v2",
        residual_factor_names=("market",),
        residual_factor_betas=(1.0,),
        residual_model_estimation_session=date(2025, 1, 2),
        entry_clock="next_session_open",
    )


def _artifact() -> WatchlistArtifact:
    item = _item()
    return WatchlistArtifact(
        trade_date=date(2025, 1, 3),
        generated_at=datetime(2025, 1, 2, 22, tzinfo=timezone.utc),
        regime=RegimeSnapshot(0.5, "B", 1.0, True, True, True, True),
        items=[item],
        tradable=[item],
        overflow=[],
        strategy_mode="daily_residual_reversion",
        selection_contract_version="daily_residual_shared_selector_v5",
        strategy_parameters={
            "factor_model": "market_only",
            "formation_sessions": 3,
            "minimum_z": 1.0,
            "score_components": ["residual_extremeness"],
            "max_positions": 10,
            "max_positions_per_sector": 2,
            "risk_fraction": 0.0035,
            "maximum_notional_fraction": 0.10,
            "partial_normalization_fraction": 0.5,
            "full_normalization_fraction": 1.0,
            "structural_failure_extension_fraction": 0.5,
            "maximum_holding_sessions": 7,
            "partial_exit_fraction": 0.5,
            "entry_clock": "next_session_open",
            "universe_contract": "frozen_98_intraday_symbols_only",
        },
    )


def _artifact_with_full_exit() -> WatchlistArtifact:
    held = HeldPositionDirective(
        symbol="AAPL",
        entry_time=datetime(2024, 12, 20, 14, 30, tzinfo=timezone.utc),
        entry_price=90.0,
        size=100,
        stop=80.0,
        initial_r=2.0,
        setup_tag="daily_residual_reversion",
        time_stop_deadline=None,
        carry_eligible_flag=False,
        flow_reversal_flag=False,
        issuer="AAPL",
        sector="Technology",
        primary_exchange="NASDAQ",
        sleeve_id="daily_residual_reversion",
        residual_factor_model="market_only",
        residual_formation_sessions=3,
        residual_volatility=0.02,
        residual_initial_dislocation_r=3.0,
        residual_held_sessions=7,
        residual_last_processed_session=date(2025, 1, 2),
        residual_pending_action="full_exit",
        residual_pending_reason="residual_half_life_time_stop",
        residual_pending_exit_fraction=1.0,
        residual_qty_entry=100,
        residual_entry_score=40.0,
        residual_lane_id="test_peer_lane",
        residual_model_contract_version="frozen_residual_model_v2",
        residual_factor_names=("market",),
        residual_factor_betas=(1.0,),
        residual_model_estimation_session=date(2024, 12, 19),
    )
    return replace(_artifact(), held_positions=[held])


def test_daily_residual_execution_persists_canonical_issuer_identity() -> None:
    item = replace(_item(), symbol="GOOGL")
    artifact = replace(_artifact(), items=[item], tradable=[item])

    state = build_daily_residual_execution_state(
        artifact,
        nav=100_000.0,
        catastrophic_stop_atr=2.5,
        catastrophic_stop_residual_r=6.0,
    )

    assert state.symbols["GOOGL"].issuer == "ALPHABET"


def test_shared_core_releases_staged_entry_only_after_full_exit_is_flat() -> None:
    state = build_daily_residual_execution_state(
        _artifact_with_full_exit(),
        nav=100_000.0,
        catastrophic_stop_atr=2.5,
    )
    preopen = datetime(2025, 1, 3, 13, 0, tzinfo=timezone.utc)

    state, actions, events = plan_daily_residual_session_orders(
        state,
        ts=preopen,
        allow_entries=True,
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitMarketExit)
    assert [event.code for event in events] == [
        "RESIDUAL_MANAGEMENT_EXIT",
        "RESIDUAL_ENTRY_DEFERRED",
    ]
    assert state.entry_orders_staged is True
    assert state.entry_orders_planned is False
    assert state.session_orders_planned is False

    state, followups, _events = apply_daily_residual_fill(
        state,
        DailyResidualFill(
            client_order_id=actions[0].client_order_id,
            symbol="AAPL",
            role="EXIT",
            qty=40,
            price=91.0,
            ts=datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc),
        ),
    )
    assert followups == ()
    assert state.symbols["AAPL"].position.qty_open == 60
    assert state.entry_orders_planned is False

    state, followups, events = apply_daily_residual_fill(
        state,
        DailyResidualFill(
            client_order_id=actions[0].client_order_id,
            symbol="AAPL",
            role="EXIT",
            qty=60,
            price=91.0,
            ts=datetime(2025, 1, 3, 14, 30, 1, tzinfo=timezone.utc),
        ),
    )
    assert len(followups) == 1
    assert isinstance(followups[0], SubmitEntry)
    assert followups[0].symbol == "MSFT"
    assert [event.code for event in events] == [
        "RESIDUAL_EXIT_FILLED",
        "RESIDUAL_ENTRY_SELECTED",
    ]
    assert state.entry_orders_planned is True
    assert state.session_orders_planned is True


class _OMS:
    def __init__(self) -> None:
        self.orders = []
        self._counter = 0

    async def get_strategy_risk(self, _strategy_id):
        return SimpleNamespace(halted=False)

    async def get_portfolio_risk(self):
        return SimpleNamespace(halted=False)

    async def submit_intent(self, intent):
        if intent.order is None:
            return SimpleNamespace(oms_order_id=None, denial_reason=None)
        self._counter += 1
        oms_id = f"OMS-{self._counter}"
        self.orders.append((oms_id, intent.order))
        return SimpleNamespace(oms_order_id=oms_id, denial_reason=None)

    def stream_events(self, _strategy_id):
        raise AssertionError("background event stream is disabled in this test")


@pytest.mark.asyncio
async def test_live_adapter_keeps_replacement_deferred_through_partial_cancel(
    tmp_path: Path,
) -> None:
    oms = _OMS()
    settings = replace(StrategySettings(), state_dir=tmp_path / "state")
    live = IARICDailyResidualEngine(
        oms_service=oms,
        artifact=_artifact_with_full_exit(),
        account_id="TEST",
        nav=100_000.0,
        settings=settings,
        disable_background_tasks=True,
    )
    await live.advance(datetime(2025, 1, 3, 13, 0, tzinfo=timezone.utc))
    assert len(oms.orders) == 1
    exit_oms_id, exit_order = oms.orders[0]
    assert exit_order.client_order_id.endswith("AAPL-EXIT")
    persisted = load_intraday_state(date(2025, 1, 3), settings=settings)
    assert persisted.meta["entry_orders_staged"] is True
    restored = IARICDailyResidualEngine(
        oms_service=oms,
        artifact=_artifact_with_full_exit(),
        account_id="TEST",
        nav=100_000.0,
        settings=settings,
        disable_background_tasks=True,
    )
    restored.hydrate_state(persisted)
    live = restored

    await live._handle_fill(
        SimpleNamespace(
            oms_order_id=exit_oms_id,
            payload={"qty": 40, "price": 91.0, "commission": 0.5},
            timestamp=datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc),
        )
    )
    assert len(oms.orders) == 1

    await live._handle_terminal(
        SimpleNamespace(
            oms_order_id=exit_oms_id,
            event_type=OMSEventType.ORDER_CANCELLED,
            timestamp=datetime(2025, 1, 3, 14, 30, 1, tzinfo=timezone.utc),
        )
    )
    assert len(oms.orders) == 2
    emergency_oms_id, emergency_order = oms.orders[-1]
    assert emergency_order.client_order_id.endswith("AAPL-FORCED-EXIT")
    assert all(
        not order.client_order_id.endswith("MSFT-ENTRY")
        for _oms_id, order in oms.orders
    )

    await live._handle_fill(
        SimpleNamespace(
            oms_order_id=emergency_oms_id,
            payload={"qty": 60, "price": 90.5, "commission": 0.5},
            timestamp=datetime(2025, 1, 3, 14, 30, 2, tzinfo=timezone.utc),
        )
    )
    assert len(oms.orders) == 3
    assert oms.orders[-1][1].client_order_id.endswith("MSFT-ENTRY")


@pytest.mark.asyncio
async def test_live_daily_cache_requests_the_declared_research_price_basis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: dict[str, str] = {}

    async def _request(_ib, _contract, *, duration, bar_size, what):
        requested.update(
            {"duration": duration, "bar_size": bar_size, "what": what}
        )
        return [
            SimpleNamespace(
                date=date(2025, 1, 2),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1_000,
            )
        ]

    class _Rate:
        async def wait_for(self):
            return None

    monkeypatch.setattr(
        residual_research_generator,
        "_request_historical_bars",
        _request,
    )
    rows = await residual_research_generator._fetch_daily_bars_cached(
        object(),
        "MSFT",
        {"con_id": None, "primary_exchange": "NASDAQ"},
        tmp_path,
        "2025-01-03",
        _Rate(),
        asyncio.Semaphore(1),
        price_basis="TRADES",
    )
    assert rows[0]["close"] == 101.0
    assert requested == {
        "duration": "1 Y",
        "bar_size": "1 day",
        "what": "TRADES",
    }
    assert (tmp_path / "daily_bars" / "MSFT.json").is_file()


@pytest.mark.asyncio
async def test_live_adapter_and_replay_core_emit_equivalent_entry_and_fill_state(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    settings = replace(
        StrategySettings(),
        state_dir=tmp_path / "state",
        diagnostics_dir=tmp_path / "diagnostics",
    )
    oms = _OMS()
    live = IARICDailyResidualEngine(
        oms_service=oms,
        artifact=artifact,
        account_id="TEST",
        nav=100_000.0,
        settings=settings,
        diagnostics=JsonlDiagnostics(tmp_path / "diagnostics", enabled=False),
        disable_background_tasks=True,
    )
    decision_ts = datetime(2025, 1, 3, 13, 0, tzinfo=timezone.utc)
    await live.advance(decision_ts)
    assert len(oms.orders) == 1
    oms_id, entry_order = oms.orders[0]
    assert entry_order.order_type.value == "MARKET"
    assert entry_order.client_order_id == "IARIC-RES-2025-01-03-MSFT-ENTRY"

    event = SimpleNamespace(
        oms_order_id=oms_id,
        payload={"qty": entry_order.qty, "price": 101.0, "commission": 1.0},
        timestamp=datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc),
    )
    await live._handle_fill(event)
    assert len(oms.orders) == 2
    assert oms.orders[1][1].role.value == "STOP"

    replay_state = build_daily_residual_execution_state(
        artifact, nav=100_000.0, catastrophic_stop_atr=2.5
    )
    planned = replay_state.symbols["MSFT"]
    assert planned.planned_initial_risk_per_share == pytest.approx(
        planned.planned_entry_price * _item().residual_volatility * math.sqrt(7.0)
    )
    assert planned.planned_initial_risk_per_share < (
        planned.planned_entry_price - planned.planned_stop_price
    )
    replay_state, actions, selected_events = plan_daily_residual_session_orders(
        replay_state, ts=decision_ts, allow_entries=True
    )
    entry_action = actions[0]
    replay_state, stop_actions, fill_events = apply_daily_residual_fill(
        replay_state,
        DailyResidualFill(
            client_order_id=entry_action.client_order_id,
            symbol="MSFT",
            role="ENTRY",
            qty=entry_action.qty,
            price=101.0,
            ts=event.timestamp,
            commission=1.0,
        ),
    )
    live_symbol = live.snapshot_state().symbols[0]
    replay_symbol = replay_state.symbols["MSFT"]
    assert live_symbol.position.qty_open == replay_symbol.position.qty_open
    assert live_symbol.position.entry_price == replay_symbol.position.entry_price
    assert live_symbol.position.entry_score == pytest.approx(80.0)
    assert replay_symbol.position.entry_score == pytest.approx(80.0)
    assert live_symbol.protective_stop_price == replay_symbol.protective_stop_price
    assert type(stop_actions[0]).__name__ == "SubmitProtectiveStop"
    assert [event.code for event in live.decision_events] == [
        selected_events[0].code,
        fill_events[0].code,
    ]
    restored = load_intraday_state(artifact.trade_date, settings=settings)
    restored_symbol = restored.symbols[0]
    assert restored.meta["strategy_mode"] == "daily_residual_reversion"
    assert restored_symbol.position.entry_price == 101.0
    assert restored_symbol.position.entry_score == pytest.approx(80.0)
    assert restored_symbol.protective_stop_qty == entry_order.qty


@pytest.mark.asyncio
async def test_live_adapter_fails_closed_when_next_open_staging_is_missed(
    tmp_path: Path,
) -> None:
    oms = _OMS()
    settings = replace(StrategySettings(), state_dir=tmp_path / "state")
    live = IARICDailyResidualEngine(
        oms_service=oms,
        artifact=_artifact(),
        account_id="TEST",
        nav=100_000.0,
        settings=settings,
        disable_background_tasks=True,
    )
    await live.advance(datetime(2025, 1, 3, 14, 31, tzinfo=timezone.utc))
    assert oms.orders == []
    assert live.snapshot_state().symbols[0].entry_skipped_reason == (
        "missed_live_next_open_staging_cutoff"
    )


def test_forced_fold_exit_is_reduced_by_shared_core() -> None:
    artifact = _artifact()
    state = build_daily_residual_execution_state(
        artifact, nav=100_000.0, catastrophic_stop_atr=2.5
    )
    ts = datetime(2025, 1, 3, 13, 0, tzinfo=timezone.utc)
    state, actions, _events = plan_daily_residual_session_orders(
        state, ts=ts, allow_entries=True
    )
    state, _stops, _fills = apply_daily_residual_fill(
        state,
        DailyResidualFill(
            client_order_id=actions[0].client_order_id,
            symbol="MSFT",
            role="ENTRY",
            qty=actions[0].qty,
            price=101.0,
            ts=ts,
        ),
    )
    state, action, event = plan_daily_residual_forced_exit(
        state,
        symbol="MSFT",
        ts=ts,
        reason="fold_end_marked_liquidation",
    )
    assert event.code == "RESIDUAL_FORCED_EXIT"
    state, _actions, events = apply_daily_residual_fill(
        state,
        DailyResidualFill(
            client_order_id=action.client_order_id,
            symbol="MSFT",
            role="EXIT",
            qty=action.qty,
            price=102.0,
            ts=ts,
        ),
    )
    assert events[0].code == "RESIDUAL_EXIT_FILLED"
    assert state.symbols["MSFT"].position.qty_open == 0


def test_item_level_residual_risk_reduction_changes_shared_core_quantity() -> None:
    standard_artifact = _artifact()
    reduced_item = replace(_item(), recommended_risk_r=0.00175)
    reduced_artifact = _artifact()
    reduced_artifact.items = [reduced_item]
    reduced_artifact.tradable = [reduced_item]

    standard = build_daily_residual_execution_state(
        standard_artifact, nav=100_000.0, catastrophic_stop_atr=2.5
    )
    reduced = build_daily_residual_execution_state(
        reduced_artifact, nav=100_000.0, catastrophic_stop_atr=2.5
    )

    assert reduced.symbols["MSFT"].planned_qty == pytest.approx(
        standard.symbols["MSFT"].planned_qty / 2.0,
        abs=1,
    )


@pytest.mark.asyncio
async def test_rejected_protective_stop_emergency_flattens(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    settings = replace(StrategySettings(), state_dir=tmp_path / "state")
    oms = _OMS()
    live = IARICDailyResidualEngine(
        oms_service=oms,
        artifact=artifact,
        account_id="TEST",
        nav=100_000.0,
        settings=settings,
        disable_background_tasks=True,
    )
    await live.advance(datetime(2025, 1, 3, 13, 0, tzinfo=timezone.utc))
    entry_oms, entry_order = oms.orders[0]
    await live._handle_fill(
        SimpleNamespace(
            oms_order_id=entry_oms,
            payload={"qty": entry_order.qty, "price": 101.0, "commission": 1.0},
            timestamp=datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc),
        )
    )
    stop_oms, _stop_order = oms.orders[1]
    await live._handle_terminal(
        SimpleNamespace(
            oms_order_id=stop_oms,
            event_type=OMSEventType.ORDER_REJECTED,
            timestamp=datetime(2025, 1, 3, 14, 31, tzinfo=timezone.utc),
        )
    )
    assert len(oms.orders) == 3
    assert oms.orders[-1][1].client_order_id.endswith("FORCED-EXIT")
    assert live.snapshot_state().symbols[0].pending_role == "EXIT"


@pytest.mark.asyncio
async def test_expired_protective_stop_emergency_flattens(
    tmp_path: Path,
) -> None:
    settings = replace(StrategySettings(), state_dir=tmp_path / "state")
    oms = _OMS()
    live = IARICDailyResidualEngine(
        oms_service=oms,
        artifact=_artifact(),
        account_id="TEST",
        nav=100_000.0,
        settings=settings,
        disable_background_tasks=True,
    )
    await live.advance(datetime(2025, 1, 3, 13, 0, tzinfo=timezone.utc))
    entry_oms, entry_order = oms.orders[0]
    await live._handle_fill(
        SimpleNamespace(
            oms_order_id=entry_oms,
            payload={"qty": entry_order.qty, "price": 101.0, "commission": 1.0},
            timestamp=datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc),
        )
    )
    stop_oms, _stop_order = oms.orders[1]
    await live._handle_terminal(
        SimpleNamespace(
            oms_order_id=stop_oms,
            event_type=OMSEventType.ORDER_EXPIRED,
            timestamp=datetime(2025, 1, 3, 21, 0, tzinfo=timezone.utc),
        )
    )
    assert oms.orders[-1][1].client_order_id.endswith("FORCED-EXIT")
    assert live.health_status()["last_error"].startswith("order_expired")
