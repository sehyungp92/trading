from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from strategies.stock.iaric.core.lanes import (
    IssuerEntryCandidate,
    SCORE_COMPONENTS,
    SCORE_PROFILES,
    anchor_exit_enabled,
    event_id,
    family_event_caps,
    issuer_exposure_decision,
    issuer_batch_arbitration,
    issuer_key,
    is_aperture_only_item,
    lane_counter_key,
    lane_daily_cap,
    lane_id_for_route,
    management_override,
    parse_mapping,
    score_from_components,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core import logic as iaric_logic
from strategies.stock.iaric.exits import _route_param, check_rsi_exit
from strategies.stock.iaric.models import Bar, MarketSnapshot, PBSymbolState
from strategies.stock.iaric.engine import IARICEngine


def test_every_score_profile_uses_exactly_seven_fixed_components() -> None:
    assert len(SCORE_COMPONENTS) == 7
    for weights in SCORE_PROFILES.values():
        assert tuple(weights) == SCORE_COMPONENTS
        assert sum(weights.values()) == pytest.approx(1.0)


def test_issuer_caps_combine_share_classes_and_fail_closed() -> None:
    settings = SimpleNamespace(
        pb_issuer_aliases="",
        pb_issuer_position_cap=1,
        pb_issuer_daily_entry_cap=1,
    )
    assert issuer_key("GOOG") == issuer_key("GOOGL") == "ALPHABET"
    decision = issuer_exposure_decision(
        settings,
        "GOOGL",
        active_symbols=["GOOG"],
        daily_entry_symbols=[],
    )
    assert decision.allowed is False
    assert decision.reason == "issuer_position_cap"

    decision = issuer_exposure_decision(
        settings,
        "GOOGL",
        active_symbols=[],
        daily_entry_symbols=["GOOG"],
    )
    assert decision.allowed is False
    assert decision.reason == "issuer_daily_entry_cap"


def test_zero_issuer_caps_preserve_historical_behavior() -> None:
    settings = SimpleNamespace(
        pb_issuer_aliases="",
        pb_issuer_position_cap=0,
        pb_issuer_daily_entry_cap=0,
    )
    assert issuer_exposure_decision(
        settings,
        "GOOGL",
        active_symbols=["GOOG"],
        daily_entry_symbols=["GOOG"],
    ).allowed is True


def test_same_batch_share_class_events_keep_only_best_causal_score() -> None:
    settings = SimpleNamespace(
        pb_issuer_aliases="",
        pb_issuer_event_dedupe_enabled=True,
    )
    decision = issuer_batch_arbitration(
        settings,
        [
            IssuerEntryCandidate(
                "GOOG", "APERTURE_GAP_EXHAUSTION_RECLAIM_ENTRY", 61.0, 2
            ),
            IssuerEntryCandidate(
                "GOOGL", "APERTURE_GAP_EXHAUSTION_RECLAIM_ENTRY", 67.0, 1
            ),
            IssuerEntryCandidate("MSFT", "OPEN_SCORED_ENTRY", 55.0, 3),
        ],
    )
    assert decision.selected_symbols == frozenset({"GOOGL", "MSFT"})
    assert decision.rejected_by_winner == {"GOOG": "GOOGL"}


def test_dual_eligible_incumbent_is_not_reclassified_as_aperture_only() -> None:
    pure = SimpleNamespace(aperture_candidate=True, trigger_tier="APERTURE")
    dual = SimpleNamespace(aperture_candidate=True, trigger_tier="HIGH")
    legacy = SimpleNamespace(aperture_candidate=False, trigger_tier="HIGH")
    assert is_aperture_only_item(pure) is True
    assert is_aperture_only_item(dual) is False
    assert is_aperture_only_item(legacy) is False


def test_lane_ids_are_economic_cohorts_not_symbols() -> None:
    assert lane_id_for_route("APERTURE_PRIOR_DAY_LOW_RECLAIM_ENTRY") == "APERTURE_LEVEL_RECLAIM"
    assert lane_id_for_route("APERTURE_UPTREND_PULLBACK_RECLAIM_ENTRY") == "APERTURE_TREND_PULLBACK"
    assert lane_id_for_route("OPEN_SCORED_ENTRY") == "OPEN_SCORED_ANCHOR"
    assert lane_id_for_route("OPEN_SCORED_RESCUE_ENTRY") == "RESCUE_EVENT"
    assert lane_counter_key("RESCUE_EVENT", "score rejected") == "lane__rescue_event__score_rejected"
    settings = SimpleNamespace(
        pb_reversion_lane_daily_caps="APERTURE_LEVEL_RECLAIM:2,RESCUE_EVENT:1"
    )
    assert lane_daily_cap(settings, "APERTURE_LEVEL_RECLAIM") == 2
    assert lane_daily_cap(settings, "RESCUE_EVENT") == 1
    assert lane_daily_cap(settings, "OPEN_SCORED_ANCHOR") is None


def test_profile_scoring_reuses_registered_components_only() -> None:
    components = {name: 0.5 for name in SCORE_COMPONENTS}
    assert score_from_components(components, "balanced") == pytest.approx(50.0)
    assert score_from_components(components, "level_reclaim") == pytest.approx(50.0)
    with pytest.raises(ValueError, match="exactly seven"):
        score_from_components({"future_leak": 1.0}, "balanced")


def test_family_management_profiles_are_opt_in() -> None:
    baseline = SimpleNamespace(pb_aperture_family_management_profiles="")
    assert management_override(
        baseline,
        "APERTURE_FAILED_BREAKDOWN_RECLAIM_ENTRY",
        "max_hold_days",
    ) is None

    enabled = SimpleNamespace(
        pb_aperture_family_management_profiles="FAILED_BREAKDOWN_RECLAIM:tail_capture"
    )
    assert management_override(
        enabled,
        "APERTURE_FAILED_BREAKDOWN_RECLAIM_ENTRY",
        "max_hold_days",
    ) == 5.0
    assert management_override(
        enabled,
        "APERTURE_MULTIDAY_HIGHER_LOW_RECLAIM_ENTRY",
        "max_hold_days",
    ) is None


def test_anchor_exit_requires_master_switch_and_fast_snapback_family_profile() -> None:
    baseline = SimpleNamespace(
        pb_aperture_anchor_exit_enabled=False,
        pb_aperture_family_management_profiles="",
    )
    route = "APERTURE_UPTREND_PULLBACK_RECLAIM_ENTRY"
    assert anchor_exit_enabled(baseline, route) is False

    tail = SimpleNamespace(
        pb_aperture_anchor_exit_enabled=True,
        pb_aperture_family_management_profiles="UPTREND_PULLBACK_RECLAIM:tail_capture",
    )
    assert anchor_exit_enabled(tail, route) is False

    snapback = SimpleNamespace(
        pb_aperture_anchor_exit_enabled=True,
        pb_aperture_family_management_profiles="GAP_EXHAUSTION_RECLAIM:fast_snapback",
    )
    assert anchor_exit_enabled(
        snapback,
        "APERTURE_GAP_EXHAUSTION_RECLAIM_ENTRY",
    ) is True

def test_registered_mapping_and_event_ids_are_deterministic() -> None:
    assert parse_mapping("GOOG:alphabet,BRK.B=berkshire", setting="aliases") == {
        "GOOG": "alphabet",
        "BRK.B": "berkshire",
    }
    assert event_id("prior_day_low_reclaim", 12) == "PRIOR_DAY_LOW_RECLAIM@12"
    with pytest.raises(ValueError, match="duplicate"):
        parse_mapping("GOOG:a,GOOG:b", setting="aliases")


def test_only_resettable_families_can_enable_a_second_episode() -> None:
    settings = SimpleNamespace(
        pb_aperture_family_max_events=(
            "PRIOR_DAY_LOW_RECLAIM:2,GAP_EXHAUSTION_RECLAIM:2"
        )
    )
    caps = family_event_caps(
        settings,
        {"PRIOR_DAY_LOW_RECLAIM", "GAP_EXHAUSTION_RECLAIM"},
    )
    assert caps == {
        "PRIOR_DAY_LOW_RECLAIM": 2,
        "GAP_EXHAUSTION_RECLAIM": 1,
    }


def test_raw_economic_filters_do_not_compare_atr_thresholds_to_score_components() -> None:
    settings = SimpleNamespace(
        pb_aperture_event_score_min=40.0,
        pb_aperture_family_score_floors=(
            "MARKET_SECTOR_RESIDUAL_RECLAIM:40"
        ),
        pb_aperture_family_filters=(
            "MARKET_SECTOR_RESIDUAL_RECLAIM:relative_exhaustion"
        ),
        pb_aperture_min_remaining_room_atr=0.10,
        pb_aperture_min_prospective_rr=0.60,
    )
    event = SimpleNamespace(
        family="MARKET_SECTOR_RESIDUAL_RECLAIM",
        score=65.0,
        score_components={name: 0.5 for name in SCORE_COMPONENTS},
        dislocation_atr=0.8,
        reclaim_atr=0.20,
        close_in_range=0.70,
        relative_volume=1.2,
        residual_dislocation_atr=-0.80,
        reversion_room_atr=0.30,
        prospective_reward_risk=1.0,
    )
    assert iaric_logic.aperture_event_admitted(settings, event) is True
    event.reversion_room_atr = 0.20
    assert iaric_logic.aperture_event_admitted(settings, event) is False


def test_rescue_event_lane_is_explicit_causal_and_selective() -> None:
    settings = StrategySettings(
        pb_v2_enabled=True,
        pb_v2_open_scored_enabled=True,
        pb_v2_open_scored_min_score=0.0,
        pb_open_scored_transition="next_bar",
        pb_open_scored_fill_timing="next_5m_open",
        pb_rescue_event_lane_enabled=True,
        pb_rescue_event_daily_score_min=60.0,
        pb_rescue_event_entry_score_min=0.0,
        pb_rescue_event_trigger_policy="oversold_or_multi",
        pb_entry_score_min=0.0,
        pb_v2_open_scored_confirmation_policy="any",
    )
    payload = {
        "daily_signal_score": 70.0,
        "daily_signal_rank_pct": 20.0,
        "rescue_flow_candidate": True,
        "trigger_types": ["RSI2"],
    }
    assert iaric_logic.open_scored_eligible(settings, payload)
    payload["daily_signal_score"] = 59.99
    assert not iaric_logic.open_scored_eligible(settings, payload)
    payload["daily_signal_score"] = 70.0
    payload["trigger_types"] = ["RS_STRONG"]
    assert not iaric_logic.open_scored_eligible(settings, payload)

    payload["trigger_types"] = ["RSI2"]
    state = PBSymbolState(
        symbol="MSFT",
        daily_signal_score=70.0,
        daily_atr=2.0,
        entry_rank_pct=20.0,
        rescue_flow_candidate=True,
        trigger_types=["RSI2"],
    )
    timestamp = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    prior_bar = Bar(
        symbol="MSFT",
        start_time=timestamp,
        end_time=timestamp.replace(minute=35),
        open=99.2,
        high=99.4,
        low=99.0,
        close=99.1,
        volume=1_000.0,
    )
    bar = Bar(
        symbol="MSFT",
        start_time=timestamp.replace(minute=35),
        end_time=timestamp.replace(minute=40),
        open=99.0,
        high=100.2,
        low=98.5,
        close=100.0,
        volume=2_000.0,
    )
    market = MarketSnapshot(symbol="MSFT", session_vwap=99.5, session_low=98.5)
    market.bars_5m.extend([prior_bar, bar])
    item = SimpleNamespace(expected_5m_volume=1_000.0, average_30m_volume=6_000.0)
    step = iaric_logic.activate_open_scored_direct_route(
        settings,
        state,
        item,
        bar,
        market,
        1,
        1.0,
        bars=[prior_bar, bar],
    )
    assert step is not None and step.acceptance is not None
    assert step.acceptance.route_family == "OPEN_SCORED_RESCUE_ENTRY"
    assert step.acceptance.lane_id == "RESCUE_EVENT"
    assert step.acceptance.accepted_bar_idx == 1

    same_open = SimpleNamespace(**vars(settings))
    same_open.pb_open_scored_fill_timing = "same_open"
    assert not iaric_logic.open_scored_eligible(same_open, payload)


def test_family_management_profile_flows_through_shared_exit_lookups() -> None:
    route = "APERTURE_PRIOR_DAY_LOW_RECLAIM_ENTRY"
    settings = StrategySettings(
        pb_aperture_rsi_exit=58.0,
        pb_aperture_family_management_profiles=(
            "PRIOR_DAY_LOW_RECLAIM:tail_capture"
        ),
    )
    assert iaric_logic.route_setting(settings, route, "max_hold_days") == 5.0
    assert _route_param(route, "max_hold_days", settings) == 5.0
    assert check_rsi_exit(60.0, route, settings) == (False, "")


def test_aperture_emits_lane_level_causal_audit_on_the_signal_bar() -> None:
    settings = StrategySettings(
        pb_aperture_enabled=True,
        pb_aperture_families="PRIOR_DAY_LOW_RECLAIM",
        pb_aperture_event_score_min=0.0,
        pb_aperture_prior_low_transition="next_bar",
        pb_aperture_family_score_profiles="PRIOR_DAY_LOW_RECLAIM:level_reclaim",
    )
    timestamp = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    prior_bar = Bar(
        symbol="MSFT",
        start_time=timestamp,
        end_time=timestamp.replace(minute=35),
        open=99.2,
        high=99.4,
        low=99.0,
        close=99.1,
        volume=1_000.0,
    )
    bar = Bar(
        symbol="MSFT",
        start_time=timestamp.replace(minute=35),
        end_time=timestamp.replace(minute=40),
        open=98.4,
        high=99.1,
        low=98.2,
        close=99.0,
        volume=1_500.0,
    )
    item = SimpleNamespace(
        aperture_candidate=True,
        previous_close=100.0,
        previous_high=101.0,
        previous_low=98.5,
        daily_atr_estimate=2.0,
        cdd_value=2,
        expected_5m_volume=1_000.0,
        five_day_return=-0.03,
        sma20_slope_atr=0.0,
        tick_size=0.01,
    )
    state = PBSymbolState(symbol="MSFT", daily_atr=2.0)
    market = MarketSnapshot(symbol="MSFT", session_low=98.2, session_vwap=98.7)
    market.bars_5m.extend([prior_bar, bar])
    step = iaric_logic.advance_aperture_route(
        settings,
        state,
        item,
        bar,
        market,
        1,
        1.0,
        bars=[prior_bar, bar],
    )
    assert step is not None and step.acceptance is not None
    assert step.acceptance.accepted_bar_idx == 1
    assert state.opportunity_audit_bar_idx == 1
    assert len(state.opportunity_audit_events) == 1
    audit = state.opportunity_audit_events[0]
    assert audit["event_id"] == "PRIOR_DAY_LOW_RECLAIM@1"
    assert audit["family"] == "PRIOR_DAY_LOW_RECLAIM"
    assert audit["lane_id"] == "APERTURE_LEVEL_RECLAIM"
    assert audit["score"] == pytest.approx(step.score)
    assert audit["reason"] == "next_bar_ready"
    assert audit["reversion_anchor"] == pytest.approx(100.0)
    assert audit["prospective_reward_risk"] > 0.0
    assert audit["episode_sequence"] == 1


def test_live_dispatch_capacity_uses_shared_issuer_and_rescue_caps() -> None:
    engine = object.__new__(IARICEngine)
    engine._settings = StrategySettings(
        pb_issuer_position_cap=1,
        pb_issuer_daily_entry_cap=1,
        pb_rescue_max_per_day=1,
    )
    engine._portfolio = SimpleNamespace(
        open_positions={"GOOG": object()},
        pending_entry_risk={},
    )
    engine._symbols = {
        "GOOGL": SimpleNamespace(
            active_order_id=None,
            entry_order=None,
            rescue_flow_candidate=False,
        )
    }
    engine._daily_entry_symbols = []
    engine._rescue_entry_count = 0
    engine._lane_entry_counts = {}
    candidate = SimpleNamespace(symbol="GOOGL", route="OPEN_SCORED_ENTRY")
    assert engine._entry_lane_capacity_reason(candidate) == "issuer_position_cap"

    engine._portfolio.open_positions = {}
    engine._daily_entry_symbols = ["GOOG"]
    assert engine._entry_lane_capacity_reason(candidate) == "issuer_daily_entry_cap"

    engine._settings = StrategySettings(
        pb_issuer_position_cap=1,
        pb_issuer_daily_entry_cap=0,
        pb_rescue_max_per_day=1,
    )
    engine._symbols = {
        "MSFT": SimpleNamespace(
            active_order_id=None,
            entry_order=None,
            rescue_flow_candidate=True,
        )
    }
    engine._daily_entry_symbols = []
    engine._rescue_entry_count = 1
    assert (
        engine._entry_lane_capacity_reason(
            SimpleNamespace(symbol="MSFT", route="OPEN_SCORED_RESCUE_ENTRY")
        )
        == "rescue_daily_cap"
    )
