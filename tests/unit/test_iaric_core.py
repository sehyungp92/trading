from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
import pytest

from strategies.core.actions import CancelAction, FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitMarketExit, SubmitProtectiveStop
from strategies.stock.iaric.bar_policy import apply_completed_5m_bar
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core import logic as iaric_logic
from strategies.stock.iaric.core.logic import build_core_state as build_iaric_runtime_state
from strategies.stock.iaric.core.serializers import restore_state as restore_iaric_state
from strategies.stock.iaric.core.serializers import snapshot_state as snapshot_iaric_state
from strategies.stock.iaric.core.state import (
    IARICCoreState,
    IARICEntryRequest,
    IARICFill,
    IARICFlattenRequest,
    IARICOrderUpdate,
    IARICPartialExitRequest,
)
from strategies.stock.iaric.exits import (
    check_eod_flatten,
    check_v2_partial,
    compute_overnight_stop,
    partial_remainder_stop_after_fill,
    partial_exit_quantity,
    run_exit_chain,
    should_carry_overnight,
)
from strategies.stock.iaric.signals import score_daily_pullback_context
from backtests.stock.engine.iaric_pullback_intraday_hybrid_engine import _PBHybridState
from strategies.stock.iaric.diagnostics import JsonlDiagnostics
from strategies.stock.iaric.engine import IARICEngine
from strategies.stock.iaric.entry_request import build_ready_entry_request
from strategies.stock.iaric.execution import build_entry_order
from strategies.stock.iaric.models import Bar, MarketSnapshot, PBSymbolState, PendingOrderState, PortfolioState, PositionState, RegimeSnapshot, WatchlistArtifact
from strategies.stock.iaric.risk import route_sizing_multiplier

UTC = timezone.utc


def test_v2_partial_requires_positive_profit_trigger() -> None:
    assert check_v2_partial(0.5, already_taken=False, trigger_r=0.5)
    assert not check_v2_partial(0.5, already_taken=False, trigger_r=0.0)
    assert not check_v2_partial(0.5, already_taken=False, trigger_r=-0.1)
    assert not check_v2_partial(1.0, already_taken=True, trigger_r=0.5)


def test_v2_partial_quantity_is_shared_and_preserves_runner() -> None:
    assert partial_exit_quantity(
        current_qty=100,
        original_qty=100,
        fraction=0.25,
        minimum_remaining_size_pct=0.10,
    ) == 25
    assert partial_exit_quantity(
        current_qty=3,
        original_qty=3,
        fraction=0.50,
        minimum_remaining_size_pct=0.50,
    ) == 1
    assert partial_exit_quantity(
        current_qty=1,
        original_qty=1,
        fraction=0.50,
        minimum_remaining_size_pct=0.10,
    ) == 0


def test_partial_remainder_stop_is_capped_below_observed_fill() -> None:
    assert partial_remainder_stop_after_fill(
        current_stop=99.0,
        requested_stop=103.0,
        fill_price=101.25,
        execution_buffer=0.01,
    ) == pytest.approx(101.24)


def test_shared_v2_carry_switch_controls_live_and_replay_eod_behavior() -> None:
    now = datetime(2026, 4, 26, 19, 55, tzinfo=UTC)
    state = PBSymbolState(symbol="MSFT", route_family="OPEN_SCORED_ENTRY")
    disabled = StrategySettings(pb_carry_enabled=False)
    enabled = StrategySettings(pb_carry_enabled=True)

    assert check_eod_flatten(now, disabled) == (True, "EOD_FLATTEN")
    assert check_eod_flatten(now, enabled) == (False, "")
    assert should_carry_overnight(state, 0.2, 0.8, "A", None, 0, disabled) == (
        False,
        "carry_disabled",
    )
    assert should_carry_overnight(state, 0.2, 0.8, "A", None, 0, enabled) == (
        True,
        "carry",
    )


def test_secondary_route_sizing_multiplier_is_shared_and_primary_neutral() -> None:
    settings = StrategySettings(
        pb_v2_secondary_route_sizing_mult=0.65,
        pb_v2_afternoon_retest_sizing_mult=0.80,
    )
    assert route_sizing_multiplier("OPEN_SCORED_ENTRY", settings) == pytest.approx(1.0)
    assert route_sizing_multiplier("VWAP_BOUNCE", settings) == pytest.approx(0.65)
    assert route_sizing_multiplier("AFTERNOON_RETEST", settings) == pytest.approx(0.52)


def test_aperture_family_daily_counts_survive_snapshot_hydration() -> None:
    portfolio = SimpleNamespace(
        pending_entry_risk={},
        account_equity=100_000.0,
        base_risk_fraction=0.005,
        regime_allows_no_new_entries=False,
        open_positions={},
    )
    engine = SimpleNamespace(
        _artifact=SimpleNamespace(trade_date=date(2026, 4, 26)),
        _symbols={},
        _last_decision_code="IDLE",
        _active_symbols=set(),
        _order_index={},
        _portfolio=portfolio,
        _expected_stop_cancels=set(),
        _last_decision_details={},
        _last_bar_ts=None,
        _aperture_family_counts={"OPENING_FLUSH_RECLAIM": 2},
        _daily_entry_symbols=["GOOG"],
        _rescue_entry_count=1,
        _lane_entry_counts={"RESCUE_EVENT": 1},
    )
    snapshot = restore_iaric_state(snapshot_iaric_state(build_iaric_runtime_state(engine)))
    restored = SimpleNamespace(
        _symbols={},
        _markets={},
        _session_vwap={},
        _active_symbols=set(),
        _order_index={},
        _portfolio=SimpleNamespace(
            pending_entry_risk={},
            account_equity=0.0,
            base_risk_fraction=0.0,
            regime_allows_no_new_entries=False,
            open_positions={},
        ),
        _expected_stop_cancels=set(),
        _last_decision_code="",
        _last_decision_details={},
        _last_bar_ts=None,
        _aperture_family_counts={},
        _daily_entry_symbols=[],
        _rescue_entry_count=0,
        _lane_entry_counts={},
    )

    iaric_logic.apply_core_state(restored, snapshot)

    assert restored._aperture_family_counts == {"OPENING_FLUSH_RECLAIM": 2}
    assert restored._daily_entry_symbols == ["GOOG"]
    assert restored._rescue_entry_count == 1
    assert restored._lane_entry_counts == {"RESCUE_EVENT": 1}


@pytest.mark.parametrize("unrealized_r", [-0.20, 0.0, 0.40, 0.75])
def test_overnight_stop_does_not_force_breakeven_below_profit_lock(unrealized_r: float) -> None:
    settings = StrategySettings(pb_v2_carry_profit_lock_r=0.75)
    assert compute_overnight_stop(100.0, 96.0, 4.0, unrealized_r, settings) == pytest.approx(96.0)


def test_overnight_stop_locks_only_profit_above_retrace_allowance() -> None:
    settings = StrategySettings(pb_v2_carry_profit_lock_r=0.75)
    assert compute_overnight_stop(100.0, 96.0, 4.0, 1.25, settings) == pytest.approx(102.0)
    assert compute_overnight_stop(100.0, 103.0, 4.0, 1.25, settings) == pytest.approx(103.0)


def test_open_scored_rejects_rescue_by_default_and_obeys_shared_slot_cap() -> None:
    settings = StrategySettings(
        pb_v2_enabled=True,
        pb_v2_open_scored_enabled=True,
        pb_v2_open_scored_min_score=50.0,
        pb_v2_open_scored_rank_pct_max=80.0,
        pb_v2_open_scored_max_slots=4,
        pb_intraday_priority_reserve_slots=2,
    )
    payload = {
        "daily_signal_score": 80.0,
        "daily_signal_rank_pct": 25.0,
        "rescue_flow_candidate": True,
    }
    assert not iaric_logic.open_scored_eligible(settings, payload)
    payload["rescue_flow_candidate"] = False
    assert iaric_logic.open_scored_eligible(settings, payload)
    assert iaric_logic.open_scored_slot_cap(settings, 10, has_intraday_candidates=True) == 4
    assert iaric_logic.open_scored_slot_cap(settings, 5, has_intraday_candidates=True) == 3


def test_open_scored_trigger_policies_reject_rs_only_without_curve_fitted_thresholds() -> None:
    base = StrategySettings(
        pb_v2_enabled=True,
        pb_v2_open_scored_enabled=True,
        pb_v2_open_scored_min_score=45.0,
        pb_v2_open_scored_trigger_policy="dislocation",
    )
    payload = {
        "daily_signal_score": 70.0,
        "daily_signal_rank_pct": 20.0,
        "rescue_flow_candidate": False,
        "trigger_types": ["RS_STRONG"],
    }
    assert not iaric_logic.open_scored_eligible(base, payload)
    payload["trigger_types"] = ["RS_STRONG", "ROC5_DROP"]
    assert iaric_logic.open_scored_eligible(base, payload)
    multi = replace(base, pb_v2_open_scored_trigger_policy="multi_dislocation")
    payload["trigger_types"] = ["ROC5_DROP"]
    assert not iaric_logic.open_scored_eligible(multi, payload)
    payload["trigger_types"] = ["ROC5_DROP", "RS_STRONG"]
    assert not iaric_logic.open_scored_eligible(multi, payload)
    payload["trigger_types"] = ["ROC5_DROP", "DEPTH"]
    assert iaric_logic.open_scored_eligible(multi, payload)


def test_open_scored_oversold_or_multi_is_a_true_union_not_a_blanket_widening() -> None:
    settings = StrategySettings(
        pb_v2_enabled=True,
        pb_v2_open_scored_enabled=True,
        pb_v2_open_scored_min_score=45.0,
        pb_v2_open_scored_trigger_policy="oversold_or_multi",
    )
    payload = {
        "daily_signal_score": 70.0,
        "daily_signal_rank_pct": 20.0,
        "rescue_flow_candidate": False,
        "trigger_types": ["RSI2"],
    }
    assert iaric_logic.open_scored_eligible(settings, payload)
    payload["trigger_types"] = ["ROC5_DROP", "DEPTH"]
    assert iaric_logic.open_scored_eligible(settings, payload)
    payload["trigger_types"] = ["ROC5_DROP"]
    assert not iaric_logic.open_scored_eligible(settings, payload)
    payload["trigger_types"] = ["RS_STRONG", "ROC5_DROP"]
    assert not iaric_logic.open_scored_eligible(settings, payload)


def test_open_scored_optional_upper_bounds_reject_chase_tail_only_when_enabled() -> None:
    settings = StrategySettings(
        pb_v2_enabled=True,
        pb_v2_open_scored_enabled=True,
        pb_v2_open_scored_min_score=45.0,
        pb_v2_open_scored_max_score=75.0,
        pb_v2_open_scored_max_entry_score=65.0,
    )
    payload = {
        "daily_signal_score": 76.0,
        "daily_signal_rank_pct": 20.0,
        "rescue_flow_candidate": False,
    }
    assert not iaric_logic.open_scored_eligible(settings, payload)
    payload["daily_signal_score"] = 70.0
    assert iaric_logic.open_scored_eligible(settings, payload)
    assert iaric_logic.open_scored_entry_score_eligible(settings, 65.0)
    assert not iaric_logic.open_scored_entry_score_eligible(settings, 65.01)


def test_open_scored_completed_bar_confirmation_is_shared_and_causal() -> None:
    start = datetime(2026, 7, 15, 13, 30, tzinfo=UTC)
    bearish = _bar(start, open_=100.0, high=100.2, low=99.0, close=99.5)
    bullish = _bar(start, open_=100.0, high=101.0, low=99.5, close=100.8)
    market = MarketSnapshot(symbol="MSFT", session_vwap=100.4)
    bullish_policy = StrategySettings(pb_v2_open_scored_confirmation_policy="bullish_close")
    vwap_policy = StrategySettings(pb_v2_open_scored_confirmation_policy="bullish_vwap")

    assert not iaric_logic.open_scored_bar_confirmed(bullish_policy, bearish, market)
    assert iaric_logic.open_scored_bar_confirmed(bullish_policy, bullish, market)
    assert iaric_logic.open_scored_bar_confirmed(vwap_policy, bullish, market)


def test_open_scored_vwap_reclaim_requires_prior_below_then_completed_cross() -> None:
    start = datetime(2026, 7, 15, 13, 30, tzinfo=UTC)
    below = _bar(start, open_=100.0, high=100.1, low=99.0, close=99.2)
    reclaim = _bar(
        start + timedelta(minutes=5),
        open_=99.2,
        high=100.5,
        low=99.1,
        close=100.4,
    )
    settings = StrategySettings(
        pb_v2_open_scored_confirmation_policy="vwap_reclaim"
    )
    market = MarketSnapshot(symbol="MSFT")
    apply_completed_5m_bar(market, below, aggregation_bar_index=0)
    assert not iaric_logic.open_scored_bar_confirmed(settings, below, market)
    apply_completed_5m_bar(market, reclaim, aggregation_bar_index=1)
    assert iaric_logic.open_scored_bar_confirmed(settings, reclaim, market)


def test_exit_chain_does_not_apply_new_completed_bar_stop_to_prior_bar_low() -> None:
    start = datetime(2026, 7, 15, 15, 0, tzinfo=UTC)
    bar = _bar(start, open_=100.8, high=102.0, low=100.5, close=101.2)
    state = SimpleNamespace(
        stop_level=101.0,
        route_family="OPEN_SCORED_ENTRY",
        hold_bars=3,
    )
    settings = StrategySettings(
        pb_v2_ema_reversion_exit=False,
        pb_carry_enabled=True,
        pb_open_scored_max_hold_days=2,
    )

    hit_without_snapshot, reason = run_exit_chain(
        state,
        bar,
        bar.end_time,
        unrealized_r=0.2,
        max_mfe_r=0.4,
        ema10_value=None,
        rsi_value=None,
        session_vwap=None,
        hold_days=0,
        flow_history=None,
        recent_5m_bars=[bar],
        quick_exit_loss_r=0.0,
        config=settings,
    )
    hit_with_snapshot, _ = run_exit_chain(
        state,
        bar,
        bar.end_time,
        unrealized_r=0.2,
        max_mfe_r=0.4,
        ema10_value=None,
        rsi_value=None,
        session_vwap=None,
        hold_days=0,
        flow_history=None,
        recent_5m_bars=[bar],
        quick_exit_loss_r=0.0,
        config=settings,
        active_stop_level=99.0,
    )

    assert hit_without_snapshot and reason == "STOP_HIT"
    assert not hit_with_snapshot


def test_daily_pullback_score_has_seven_bounded_sweet_spot_components() -> None:
    sweet = score_daily_pullback_context(
        trend_tier="STRONG", rsi2=8.0, rsi5=25.0, cdd=3, depth_atr=2.0,
        bb_pctb=0.02, volume_climax=2.0, is_down_day=True, rs_ratio=1.02, roc5=-3.0,
    )
    extreme = score_daily_pullback_context(
        trend_tier="STRONG", rsi2=0.0, rsi5=5.0, cdd=9, depth_atr=5.0,
        bb_pctb=-0.50, volume_climax=5.0, is_down_day=True, rs_ratio=0.70, roc5=-15.0,
        sma_slope_positive=False, sma_dist_pct=-8.0, trigger_tier="HIGH", n_triggers=4,
        regime_tier="C", persistence=0.0,
    )
    assert len([key for key in sweet if key != "score"]) == 7
    assert sweet["score"] > extreme["score"]
    assert iaric_logic.opening_gap_eligible(StrategySettings(pb_v2_enabled=True), 100.0, 102.0)
    assert not iaric_logic.opening_gap_eligible(StrategySettings(pb_v2_enabled=True), 100.0, 104.0)


def _state(*symbols: PBSymbolState) -> IARICCoreState:
    return IARICCoreState(
        trade_date=date(2026, 4, 26),
        saved_at=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        symbols=list(symbols),
        last_decision_code="IDLE",
        meta={
            "active_symbols": [symbol.symbol for symbol in symbols],
            "pending_entry_risk": {},
            "order_index": {},
        },
    )


def _item():
    return SimpleNamespace(expected_5m_volume=1_000.0, average_30m_volume=6_000.0)


def _bar(base: datetime, *, open_: float, high: float, low: float, close: float, volume: float = 1_200.0) -> Bar:
    return Bar(
        symbol="MSFT",
        start_time=base,
        end_time=base + timedelta(minutes=5),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _market(symbol: str, bars: list[Bar], *, vwap: float) -> MarketSnapshot:
    market = MarketSnapshot(symbol=symbol)
    market.session_vwap = vwap
    market.session_low = min(bar.low for bar in bars)
    market.session_high = max(bar.high for bar in bars)
    market.last_5m_bar = bars[-1]
    for bar in bars:
        market.bars_5m.append(bar)
    return market


def _backtest_route_state(item, *, daily_signal_score: float = 72.0, daily_atr: float = 1.6) -> _PBHybridState:
    return _PBHybridState(
        symbol="MSFT",
        item=item,
        record=None,
        trigger_type="RSI2",
        entry_rsi=12.0,
        entry_gap_pct=-1.0,
        entry_sma_dist_pct=3.0,
        entry_cdd=2,
        entry_rank=1,
        entry_rank_pct=25.0,
        n_candidates=1,
        prev_iloc=0,
        sector="Tech",
        daily_atr=daily_atr,
        daily_signal_score=daily_signal_score,
    )


def test_route_entry_score_has_exactly_seven_decision_components() -> None:
    base = datetime(2026, 4, 26, 14, 0, tzinfo=UTC)
    bars = [
        _bar(base, open_=100.0, high=100.4, low=99.8, close=100.2),
        _bar(base + timedelta(minutes=5), open_=100.2, high=100.8, low=100.1, close=100.7),
    ]
    item = _item()
    state = _backtest_route_state(item)
    state.stop_level = 99.5
    state.reclaim_level = 100.0
    state.flush_bar_idx = 0
    bundle = iaric_logic.compute_route_entry_score_bundle(
        StrategySettings(), state, item, bars[-1], _market("MSFT", bars, vwap=100.1), 1,
        bars=bars,
    )
    components = {key for key in bundle if key not in {"score", "route_family"}}
    assert components == {
        "daily_signal", "reclaim", "volume", "vwap_hold", "cpr", "speed", "quality_adjustment",
    }


def test_route_entry_filters_block_distribution_and_insufficient_reversion_room() -> None:
    base = datetime(2026, 4, 26, 14, 0, tzinfo=UTC)
    bars = [
        _bar(base, open_=100.0, high=100.4, low=99.8, close=100.2),
        _bar(base + timedelta(minutes=5), open_=100.2, high=100.8, low=100.1, close=100.7),
    ]
    item = _item()
    state = _backtest_route_state(item, daily_atr=2.0)
    state.prev_close = 103.0
    state.stop_level = 99.5
    state.reclaim_level = 100.0
    state.flush_bar_idx = 0
    market = _market("MSFT", bars, vwap=100.1)

    distribution_block = iaric_logic.compute_route_entry_score_bundle(
        StrategySettings(pb_entry_micropressure_policy="block_distribute"),
        state,
        item,
        bars[-1],
        market,
        1,
        bars=bars,
        micropressure="DISTRIBUTE",
    )
    assert distribution_block["score"] == 0.0

    room_block = iaric_logic.compute_route_entry_score_bundle(
        StrategySettings(pb_entry_min_reversion_room_atr=1.25),
        state,
        item,
        bars[-1],
        market,
        1,
        bars=bars,
        micropressure="NEUTRAL",
    )
    assert room_block["score"] == 0.0

    allowed = iaric_logic.compute_route_entry_score_bundle(
        StrategySettings(
            pb_entry_micropressure_policy="block_distribute",
            pb_entry_min_reversion_room_atr=0.50,
        ),
        state,
        item,
        bars[-1],
        market,
        1,
        bars=bars,
        micropressure="NEUTRAL",
    )
    assert allowed["score"] > 0.0
    components = {key for key in allowed if key not in {"score", "route_family"}}
    assert len(components) == 7


def test_session_atr_uses_seed_then_completed_intraday_ranges_for_live_replay_parity() -> None:
    base = datetime(2026, 4, 26, 14, 0, tzinfo=UTC)
    item = SimpleNamespace(intraday_atr_seed=0.012, daily_atr_estimate=4.0, avwap_ref=100.0)
    bars = [
        _bar(base, open_=100.0, high=100.5, low=99.5, close=100.0),
        _bar(base + timedelta(minutes=5), open_=100.0, high=101.0, low=99.0, close=100.5),
        _bar(base + timedelta(minutes=10), open_=100.5, high=102.0, low=100.0, close=101.5),
    ]
    assert iaric_logic.estimate_session_atr(item, bars[:1], 4.0) == pytest.approx(1.2)
    assert iaric_logic.estimate_session_atr(item, bars, 4.0) == pytest.approx(2.0)


def test_session_atr_daily_fallback_is_intraday_scaled() -> None:
    item = SimpleNamespace(intraday_atr_seed=0.0, daily_atr_estimate=4.0, avwap_ref=100.0)
    assert iaric_logic.estimate_session_atr(item, [], 4.0) == pytest.approx(1.0)


def test_iaric_on_bar_entry_request_emits_submit_entry() -> None:
    state = _state(PBSymbolState(symbol="MSFT", route_family="OPENING_RECLAIM", stop_level=404.5))

    next_state, actions, events = iaric_logic.on_bar(
        state,
        bar_ts=datetime(2026, 4, 26, 14, 30, tzinfo=UTC),
        entry_request=IARICEntryRequest(
            client_order_id="ENTRY-1",
            symbol="MSFT",
            route="OPENING_RECLAIM",
            qty=25,
            limit_price=410.25,
            stop_price=404.5,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert actions[0].qty == 25
    assert actions[0].limit_price == 410.25
    assert actions[0].risk_context["stop_for_risk"] == 404.5
    assert events[0].code == "ENTRY_REQUESTED"
    assert next_state.last_decision_code == "ENTRY_REQUESTED"


def test_iaric_on_fill_entry_creates_position_and_stop_action() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            route_family="VWAP_BOUNCE",
            stop_level=404.5,
            entry_order=PendingOrderState(
                oms_order_id="ENTRY-1",
                submitted_at=datetime(2026, 4, 26, 14, 29, tzinfo=UTC),
                role="ENTRY",
                requested_qty=25,
                limit_price=410.25,
            ),
            active_order_id="ENTRY-1",
        )
    )
    state.meta["pending_entry_risk"] = {"MSFT": 143.75}

    next_state, actions, events = iaric_logic.on_fill(
        state,
        IARICFill(
            oms_order_id="ENTRY-1",
            symbol="MSFT",
            order_role="ENTRY",
            fill_price=410.5,
            fill_qty=25,
            fill_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
            commission=1.25,
        ),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.in_position is True
    assert symbol_state.position is not None
    assert symbol_state.position.qty_open == 25
    assert symbol_state.entry_order is None
    assert next_state.meta["pending_entry_risk"] == {}
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].qty == 25
    assert events[0].code == "ENTRY_FILLED"


def test_iaric_on_bar_flatten_with_pending_tp_emits_cancel() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            in_position=True,
            stage="IN_POSITION",
            position=PositionState(
                entry_price=410.5,
                qty_entry=25,
                qty_open=12,
                final_stop=404.5,
                current_stop=407.0,
                entry_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
                initial_risk_per_share=6.0,
                max_favorable_price=414.0,
                max_adverse_price=409.0,
                stop_order_id="STOP-1",
            ),
            exit_order=PendingOrderState(
                oms_order_id="TP-1",
                submitted_at=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
                role="TP",
                requested_qty=13,
            ),
        )
    )

    next_state, actions, events = iaric_logic.on_bar(
        state,
        bar_ts=datetime(2026, 4, 26, 15, 1, tzinfo=UTC),
        flatten_request=IARICFlattenRequest(symbol="MSFT", reason="FLOW_REVERSAL", qty=12),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.pending_hard_exit is True
    assert symbol_state.exit_order is not None
    assert symbol_state.exit_order.cancel_requested is True
    assert len(actions) == 1
    assert isinstance(actions[0], CancelAction)
    assert actions[0].target_order_id == "TP-1"
    assert events[0].code == "FLATTEN_QUEUED_AFTER_CANCEL"


def test_iaric_on_fill_partial_exit_resizes_stop() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            in_position=True,
            stage="IN_POSITION",
            v2_partial_taken=False,
            position=PositionState(
                entry_price=410.5,
                qty_entry=25,
                qty_open=25,
                final_stop=404.5,
                current_stop=408.0,
                entry_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
                initial_risk_per_share=6.0,
                max_favorable_price=414.0,
                max_adverse_price=409.0,
                stop_order_id="STOP-1",
                pending_partial_stop=414.7,
                pending_partial_stop_buffer=0.01,
            ),
            exit_order=PendingOrderState(
                oms_order_id="TP-1",
                submitted_at=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
                role="TP",
                requested_qty=12,
            ),
        )
    )

    next_state, actions, events = iaric_logic.on_fill(
        state,
        IARICFill(
            oms_order_id="TP-1",
            symbol="MSFT",
            order_role="TP",
            fill_price=413.0,
            fill_qty=12,
            fill_time=datetime(2026, 4, 26, 15, 2, tzinfo=UTC),
            commission=0.75,
        ),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.position is not None
    assert symbol_state.position.qty_open == 13
    assert symbol_state.v2_partial_taken is True
    assert symbol_state.position.current_stop == pytest.approx(412.99)
    assert symbol_state.position.pending_partial_stop == 0.0
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 13
    assert events[0].code == "PARTIAL_EXIT_FILLED"


def test_iaric_partial_request_does_not_move_stop_before_fill() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            in_position=True,
            stage="IN_POSITION",
            position=PositionState(
                entry_price=410.5,
                qty_entry=25,
                qty_open=25,
                final_stop=404.5,
                current_stop=408.0,
                entry_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
                initial_risk_per_share=6.0,
                max_favorable_price=414.0,
                max_adverse_price=409.0,
                stop_order_id="STOP-1",
            ),
        )
    )

    next_state, actions, events = iaric_logic.on_bar(
        state,
        bar_ts=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
        partial_exit_request=IARICPartialExitRequest(
            client_order_id="TP-1",
            symbol="MSFT",
            qty=12,
            remainder_stop_price=414.7,
            execution_buffer=0.01,
        ),
    )

    position = next_state.symbols[0].position
    assert position is not None
    assert position.current_stop == 408.0
    assert position.pending_partial_stop == 414.7
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitMarketExit)
    assert events[0].code == "PARTIAL_EXIT_REQUESTED"


def test_iaric_on_order_update_unexpected_stop_terminal_flattens() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            in_position=True,
            stage="IN_POSITION",
            position=PositionState(
                entry_price=410.5,
                qty_entry=25,
                qty_open=10,
                final_stop=404.5,
                current_stop=408.0,
                entry_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
                initial_risk_per_share=6.0,
                max_favorable_price=414.0,
                max_adverse_price=409.0,
                stop_order_id="STOP-1",
            ),
        )
    )

    next_state, actions, events = iaric_logic.on_order_update(
        state,
        IARICOrderUpdate(
            oms_order_id="STOP-1",
            symbol="MSFT",
            order_role="STOP",
            status="cancelled",
            timestamp=datetime(2026, 4, 26, 15, 5, tzinfo=UTC),
        ),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.position is not None
    assert symbol_state.position.stop_order_id == ""
    assert len(actions) == 1
    assert isinstance(actions[0], FlattenPosition)
    assert actions[0].qty == 10
    assert events[0].code == "STOP_TERMINAL"


def test_iaric_shared_opening_reclaim_progression_matches_live_and_backtest_state_shapes() -> None:
    settings = StrategySettings(
        pb_opening_reclaim_enabled=True,
        pb_opening_reclaim_min_daily_signal_score=0.0,
        pb_flush_window_bars=3,
        pb_ready_acceptance_bars=1,
        pb_ready_min_volume_ratio=0.5,
        pb_ready_min_cpr=0.5,
    )
    item = _item()
    start = datetime(2026, 4, 26, 14, 30, tzinfo=UTC)
    bars = [
        _bar(start, open_=100.0, high=100.2, low=99.0, close=99.2, volume=1_000.0),
        _bar(start + timedelta(minutes=5), open_=99.2, high=100.4, low=99.1, close=100.1, volume=1_300.0),
        _bar(start + timedelta(minutes=10), open_=100.1, high=100.8, low=100.0, close=100.7, volume=1_500.0),
    ]

    live_state = PBSymbolState(symbol="MSFT", daily_signal_score=72.0, daily_atr=1.6)
    backtest_state = _backtest_route_state(item)

    live_steps = []
    backtest_steps = []
    for idx, bar in enumerate(bars):
        market = _market("MSFT", bars[: idx + 1], vwap=99.8)
        live_state.session_low = market.session_low or 0.0
        live_steps.append(
            iaric_logic.advance_opening_reclaim_route(
                settings, live_state, item, bar, market, idx, 1.0, bars=bars[: idx + 1]
            )
        )
        backtest_steps.append(
            iaric_logic.advance_opening_reclaim_route(
                settings, backtest_state, item, bar, market, idx, 1.0, bars=bars[: idx + 1]
            )
        )

    assert [step.stage if step is not None else None for step in live_steps] == ["FLUSH_LOCKED", "RECLAIMING", "READY"]
    assert [step.stage if step is not None else None for step in backtest_steps] == ["FLUSH_LOCKED", "RECLAIMING", "READY"]
    assert live_state.intraday_score == backtest_state.intraday_score
    assert live_state.target_entry_price == backtest_state.target_entry_price
    assert live_state.ready_bar_idx == backtest_state.ready_bar_idx == 2


def test_iaric_shared_delayed_confirm_and_ready_acceptance_match_live_and_backtest_state_shapes() -> None:
    settings = StrategySettings(
        pb_v2_enabled=False,
        pb_delayed_confirm_after_bar=5,
        pb_delayed_confirm_score_min=40.0,
        pb_entry_score_min=55.0,
        pb_ready_min_volume_ratio=0.5,
    )
    item = _item()
    start = datetime(2026, 4, 26, 14, 30, tzinfo=UTC)
    bars = [
        _bar(start + timedelta(minutes=5 * idx), open_=100.0 + idx * 0.05, high=100.2 + idx * 0.1, low=99.3, close=99.8 + idx * 0.08, volume=1_050.0)
        for idx in range(5)
    ]
    bars.append(_bar(start + timedelta(minutes=25), open_=100.2, high=101.2, low=99.4, close=100.95, volume=1_600.0))
    bars.append(_bar(start + timedelta(minutes=30), open_=100.9, high=101.1, low=100.6, close=100.98, volume=1_250.0))

    live_state = PBSymbolState(symbol="MSFT", daily_signal_score=82.0, daily_atr=1.6)
    backtest_state = _backtest_route_state(item, daily_signal_score=82.0)
    live_market = _market("MSFT", bars[:6], vwap=100.2)
    backtest_market = _market("MSFT", bars[:6], vwap=100.2)
    live_state.session_low = live_market.session_low or 0.0

    live_step = iaric_logic.activate_delayed_confirm_route(
        settings, live_state, item, bars[5], live_market, 5, 1.0, bars=bars[:6]
    )
    backtest_step = iaric_logic.activate_delayed_confirm_route(
        settings, backtest_state, item, bars[5], backtest_market, 5, 1.0, bars=bars[:6]
    )

    assert live_step is not None and live_step.stage == "READY"
    assert backtest_step is not None and backtest_step.stage == "READY"
    assert live_state.ready_bar_idx == backtest_state.ready_bar_idx == 5
    assert live_state.intraday_score == backtest_state.intraday_score

    live_market = _market("MSFT", bars[:7], vwap=100.25)
    backtest_market = _market("MSFT", bars[:7], vwap=100.25)
    live_state.session_low = live_market.session_low or 0.0
    live_accept = iaric_logic.evaluate_ready_entry(
        settings, live_state, item, bars[6], live_market, 6, 1.0, bars=bars[:7]
    )
    backtest_accept = iaric_logic.evaluate_ready_entry(
        settings, backtest_state, item, bars[6], backtest_market, 6, 1.0, bars=bars[:7]
    )

    assert live_accept is not None and live_accept.acceptance is not None
    assert backtest_accept is not None and backtest_accept.acceptance is not None
    assert live_accept.acceptance.accepted_bar_idx == backtest_accept.acceptance.accepted_bar_idx == 6
    assert live_accept.acceptance.accepted_entry_price == backtest_accept.acceptance.accepted_entry_price
    assert live_accept.acceptance.entry_trigger == backtest_accept.acceptance.entry_trigger == "DELAYED_CONFIRM"


def test_shared_open_scored_retest_matches_live_and_backtest_state_shapes() -> None:
    settings = StrategySettings(
        pb_v2_enabled=True,
        pb_v2_open_scored_enabled=True,
        pb_v2_open_scored_min_score=45.0,
        pb_v2_open_scored_rank_pct_max=100.0,
        pb_v2_open_scored_trigger_policy="multi_dislocation",
        pb_entry_score_min=40.0,
        pb_open_scored_transition="confirmed_retest",
        pb_open_scored_retest_window_bars=6,
        pb_open_scored_retest_retrace_frac=0.35,
        pb_open_scored_retest_min_close_pct=0.55,
        pb_open_scored_retest_min_impulse_atr=0.10,
        pb_open_scored_retest_max_extension_atr=0.50,
    )
    item = _item()
    start = datetime(2026, 4, 26, 14, 30, tzinfo=UTC)
    bars = [
        _bar(start, open_=100.0, high=101.1, low=99.0, close=100.8, volume=1_600.0),
        _bar(
            start + timedelta(minutes=5),
            open_=100.25,
            high=100.75,
            low=99.95,
            close=100.65,
            volume=1_300.0,
        ),
    ]
    live_state = PBSymbolState(
        symbol="MSFT",
        daily_signal_score=82.0,
        daily_atr=1.6,
        entry_rank_pct=25.0,
        trigger_types=["ROC5_DROP", "DEPTH"],
    )
    backtest_state = _backtest_route_state(item, daily_signal_score=82.0)
    backtest_state.trigger_types = ["ROC5_DROP", "DEPTH"]
    signal_market = _market("MSFT", bars[:1], vwap=100.0)
    live_state.session_low = signal_market.session_low or 0.0

    live_arm = iaric_logic.arm_open_scored_retest_route(
        settings,
        live_state,
        item,
        bars[0],
        signal_market,
        0,
        1.0,
        bars=bars[:1],
    )
    backtest_arm = iaric_logic.arm_open_scored_retest_route(
        settings,
        backtest_state,
        item,
        bars[0],
        signal_market,
        0,
        1.0,
        bars=bars[:1],
    )

    assert live_arm is not None and live_arm.stage == "RETEST_ARMED"
    assert backtest_arm is not None and backtest_arm.stage == "RETEST_ARMED"
    assert live_state.target_entry_price == backtest_state.target_entry_price
    assert live_state.stop_level == backtest_state.stop_level
    assert live_state.intraday_score == backtest_state.intraday_score

    confirm_market = _market("MSFT", bars, vwap=100.1)
    live_state.session_low = confirm_market.session_low or 0.0
    live_accept = iaric_logic.advance_open_scored_retest_route(
        settings,
        live_state,
        item,
        bars[1],
        confirm_market,
        1,
        1.0,
        bars=bars,
    )
    backtest_accept = iaric_logic.advance_open_scored_retest_route(
        settings,
        backtest_state,
        item,
        bars[1],
        confirm_market,
        1,
        1.0,
        bars=bars,
    )

    assert live_accept is not None and live_accept.acceptance is not None
    assert backtest_accept is not None and backtest_accept.acceptance is not None
    assert live_accept.acceptance.accepted_bar_idx == 1
    assert live_accept.acceptance.accepted_bar_idx == backtest_accept.acceptance.accepted_bar_idx
    assert live_accept.acceptance.route_family == "OPEN_SCORED_RETEST"
    assert live_accept.acceptance.accepted_entry_price == backtest_accept.acceptance.accepted_entry_price
    assert live_accept.acceptance.score_components == backtest_accept.acceptance.score_components


def test_shared_open_scored_retrace_limit_matches_live_and_backtest_state_shapes() -> None:
    settings = StrategySettings(
        pb_v2_enabled=True,
        pb_v2_open_scored_enabled=True,
        pb_v2_open_scored_min_score=45.0,
        pb_v2_open_scored_rank_pct_max=100.0,
        pb_entry_score_min=40.0,
        pb_open_scored_transition="resting_retrace",
        pb_open_scored_retrace_limit_fraction=0.35,
        pb_open_scored_retrace_limit_window_bars=12,
    )
    item = _item()
    start = datetime(2026, 4, 26, 14, 30, tzinfo=UTC)
    signal_bar = _bar(
        start,
        open_=100.0,
        high=101.1,
        low=99.0,
        close=100.8,
        volume=1_600.0,
    )
    market = _market("MSFT", [signal_bar], vwap=100.0)
    live_state = PBSymbolState(
        symbol="MSFT",
        daily_signal_score=82.0,
        daily_atr=1.6,
        entry_rank_pct=25.0,
    )
    backtest_state = _backtest_route_state(item, daily_signal_score=82.0)
    live_state.session_low = market.session_low or 0.0

    live_step = iaric_logic.arm_open_scored_retrace_limit_route(
        settings,
        live_state,
        item,
        signal_bar,
        market,
        0,
        1.0,
        bars=[signal_bar],
    )
    backtest_step = iaric_logic.arm_open_scored_retrace_limit_route(
        settings,
        backtest_state,
        item,
        signal_bar,
        market,
        0,
        1.0,
        bars=[signal_bar],
    )

    assert live_step is not None and live_step.acceptance is not None
    assert backtest_step is not None and backtest_step.acceptance is not None
    assert live_step.acceptance.accepted_bar_idx == 0
    assert live_step.acceptance.route_family == "OPEN_SCORED_RETRACE_LIMIT"
    assert live_step.acceptance.accepted_entry_price == backtest_step.acceptance.accepted_entry_price
    assert live_state.target_entry_price == backtest_state.target_entry_price
    assert live_state.stop_level == backtest_state.stop_level
    assert live_state.intraday_score == backtest_state.intraday_score
    assert live_step.acceptance.score_components == backtest_step.acceptance.score_components


def test_retrace_limit_live_request_uses_target_and_extended_ttl() -> None:
    settings = StrategySettings()
    state = PBSymbolState(
        symbol="MSFT",
        stop_level=98.0,
        target_entry_price=99.25,
        sizing_mult=1.0,
    )
    item = SimpleNamespace(
        symbol="MSFT",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        regime_tier="A",
        entry_gap_pct=0.0,
    )
    market = MarketSnapshot(symbol="MSFT", last_price=101.0, ask=101.05)
    portfolio = PortfolioState(account_equity=100_000.0, base_risk_fraction=0.001)

    built = build_ready_entry_request(
        symbol="MSFT",
        state=state,
        item=item,
        market=market,
        portfolio=portfolio,
        symbol_to_sector={"MSFT": "Technology"},
        settings=settings,
        now=datetime(2026, 4, 27, 13, 35, tzinfo=UTC),
        route="OPEN_SCORED_RETRACE_LIMIT",
    )

    assert built.entry_request is not None
    assert built.entry_price == 99.25
    assert built.entry_request.limit_price == 99.25
    order = build_entry_order(
        item,
        "paper",
        built.entry_request.qty,
        built.entry_request.limit_price,
        built.entry_request.stop_price,
        ttl_seconds=3600,
    )
    assert order.limit_price == 99.25
    assert order.entry_policy.ttl_seconds == 3600


def test_open_scored_priority_value_is_shared_and_explicit() -> None:
    high = StrategySettings(pb_open_scored_priority="high_score")
    low = StrategySettings(pb_open_scored_priority="low_score")

    assert iaric_logic.route_priority_value(high, "OPEN_SCORED_ENTRY", 60.0) == -60.0
    assert iaric_logic.route_priority_value(low, "OPEN_SCORED_ENTRY", 60.0) == 60.0
    assert iaric_logic.route_priority_value(low, "DELAYED_CONFIRM", 60.0) == -60.0
    with pytest.raises(ValueError, match="pb_open_scored_priority"):
        iaric_logic.route_priority_value(
            StrategySettings(pb_open_scored_priority="unknown"),
            "OPEN_SCORED_ENTRY",
            60.0,
        )


def test_iaric_shared_thirty_min_context_bonus_matches_legacy_flat_bar_semantics() -> None:
    market = MarketSnapshot(symbol="MSFT")
    market.last_30m_bar = Bar(
        symbol="MSFT",
        start_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        end_time=datetime(2026, 4, 26, 14, 30, tzinfo=UTC),
        open=192.0,
        high=192.0,
        low=192.0,
        close=192.0,
        volume=1_000.0,
    )

    assert iaric_logic.thirty_min_context_bonus(market, weight=4.0) == 2.0


def test_iaric_shared_volume_ratio_preserves_legacy_zero_expected_volume_behavior() -> None:
    bar = _bar(datetime(2026, 4, 26, 14, 30, tzinfo=UTC), open_=100.0, high=101.0, low=99.5, close=100.5, volume=480.0)
    item = SimpleNamespace(expected_5m_volume=0.0, average_30m_volume=0.0)

    assert iaric_logic.compute_volume_ratio(bar, item) == 480.0


def test_iaric_shared_reset_route_state_uses_strategy_specific_reset_defaults() -> None:
    state = _backtest_route_state(_item())
    state.stage = "READY"
    state.route_family = "DELAYED_CONFIRM"
    state.ready_bar_idx = 7
    state.accepted_bar_idx = 9
    state.accepted_entry_price = 101.25

    iaric_logic.reset_route_state(state)

    assert state.stage == "WATCHING"
    assert state.route_family == ""
    assert state.ready_bar_idx == 0
    assert state.accepted_bar_idx == -1
    assert state.accepted_entry_price == 0.0


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_iaric_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch, tmp_path) -> None:
    artifact = WatchlistArtifact(
        trade_date=date(2026, 4, 26),
        generated_at=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        regime=RegimeSnapshot(
            score=0.75,
            tier="B",
            risk_multiplier=1.0,
            price_ok=True,
            breadth_ok=True,
            vol_ok=True,
            credit_ok=True,
        ),
        items=[],
        tradable=[],
        overflow=[],
    )
    engine = IARICEngine(
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        artifact=artifact,
        account_id="ACCT-1",
        nav=100_000.0,
        settings=StrategySettings(diagnostics_dir=str(tmp_path)),
        diagnostics=JsonlDiagnostics(Path(tmp_path), enabled=False),
    )
    engine._items["MSFT"] = SimpleNamespace(tick_size=0.01)
    engine._markets["MSFT"] = MarketSnapshot(symbol="MSFT")
    engine._symbols["MSFT"] = PBSymbolState(
        symbol="MSFT",
        route_family="VWAP_BOUNCE",
        stop_level=404.5,
        entry_order=PendingOrderState(
            oms_order_id="ENTRY-1",
            submitted_at=datetime(2026, 4, 26, 14, 29, tzinfo=UTC),
            role="ENTRY",
            requested_qty=25,
            limit_price=410.25,
        ),
        active_order_id="ENTRY-1",
    )
    engine._portfolio.pending_entry_risk["MSFT"] = 143.75
    engine._order_index["ENTRY-1"] = ("MSFT", "ENTRY")

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(engine, "_submit_stop", _noop)
    monkeypatch.setattr(engine, "_replace_stop", _noop)
    monkeypatch.setattr(engine, "_cancel_stop", _noop)
    monkeypatch.setattr(engine, "_submit_market_exit", _noop)
    monkeypatch.setattr(engine, "_record_entry_instrumentation", _noop)
    monkeypatch.setattr(engine, "_record_exit_instrumentation", _noop)

    initial_state = restore_iaric_state(snapshot_iaric_state(build_iaric_runtime_state(engine)))
    fill_time = datetime(2026, 4, 26, 14, 31, tzinfo=UTC)

    await engine._handle_fill(
        SimpleNamespace(
            oms_order_id="ENTRY-1",
            payload={"price": 410.5, "qty": 25, "commission": 1.25},
            timestamp=fill_time,
        )
    )

    wrapper_snapshot = snapshot_iaric_state(build_iaric_runtime_state(engine))
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    IARICFill(
                        oms_order_id="ENTRY-1",
                        fill_price=410.5,
                        fill_qty=25,
                        fill_time=fill_time,
                        commission=1.25,
                        symbol="MSFT",
                        order_role="ENTRY",
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: iaric_logic.on_bar(state, **payload),
        on_order_update=iaric_logic.on_order_update,
        on_fill=iaric_logic.on_fill,
    )

    replay_snapshot = snapshot_iaric_state(replay.state)
    replay_snapshot.pop("saved_at", None)
    wrapper_snapshot.pop("saved_at", None)

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert replay_snapshot == wrapper_snapshot
