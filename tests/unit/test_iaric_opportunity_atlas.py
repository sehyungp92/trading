from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backtests.stock.auto.runners.run_stock_opportunity_atlas import _score_integrity
from strategies.stock.iaric.core.lanes import score_from_components
from strategies.stock.iaric.core.opportunity import (
    DailyOpportunityContext,
    OPPORTUNITY_SCORE_WEIGHTS,
    detect_completed_bar_opportunities,
    evaluate_standardized_entry_variants,
    evaluate_standardized_opportunity,
    opportunity_score_components,
)
from strategies.stock.iaric.models import Bar


UTC = timezone.utc


def _bar(
    index: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1_000.0,
) -> Bar:
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC) + timedelta(minutes=5 * index)
    return Bar(
        symbol="MSFT",
        start_time=start,
        end_time=start + timedelta(minutes=5),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _context(**changes) -> DailyOpportunityContext:
    values = {
        "prev_close": 100.0,
        "prev_high": 101.0,
        "prev_low": 98.5,
        "daily_atr": 2.0,
        "consecutive_down_days": 0,
        "expected_5m_volume": 1_000.0,
    }
    values.update(changes)
    return DailyOpportunityContext(**values)


def test_gap_reclaim_is_completed_bar_signal_with_next_bar_entry() -> None:
    bars = [
        _bar(0, open_=99.0, high=99.2, low=98.8, close=99.15),
        _bar(1, open_=99.2, high=99.5, low=99.1, close=99.4),
    ]
    events = detect_completed_bar_opportunities(bars, _context())
    gap = next(event for event in events if event.family == "GAP_EXHAUSTION_RECLAIM")
    assert gap.signal_bar_index == 0
    assert gap.entry_bar_index == 1
    assert len(gap.score_components) == 7
    assert set(gap.score_components) == set(OPPORTUNITY_SCORE_WEIGHTS)


def test_partial_gap_reversion_signals_before_target_and_preserves_payoff_room() -> None:
    bars = [
        _bar(0, open_=99.0, high=99.45, low=98.8, close=99.35),
        _bar(1, open_=99.4, high=99.7, low=99.3, close=99.6),
    ]
    events = detect_completed_bar_opportunities(bars, _context())
    event = next(item for item in events if item.family == "GAP_PARTIAL_RECLAIM")
    assert event.reversion_anchor == pytest.approx(100.0)
    assert event.reversion_room_atr > 0.0
    assert event.prospective_reward_risk > 0.0
    assert event.stop_anchor < bars[0].low
    assert event.anchor_kind == "previous_close"


def test_completed_gap_is_not_mislabeled_as_a_new_reversion_entry() -> None:
    bars = [
        _bar(0, open_=99.0, high=100.2, low=98.8, close=100.1),
        _bar(1, open_=100.1, high=100.3, low=100.0, close=100.2),
    ]
    events = detect_completed_bar_opportunities(bars, _context())
    assert all(event.family != "GAP_PARTIAL_RECLAIM" for event in events)
    assert all(event.family != "GAP_FILL_RECLAIM" for event in events)


def test_score_transforms_preserve_order_across_ordinary_event_geometry() -> None:
    def components(depth: float, volume: float) -> dict[str, float]:
        return opportunity_score_components(
            dislocation_atr=depth,
            reclaim_atr=0.4,
            close_in_range=0.65,
            relative_volume=volume,
            residual_dislocation_atr=-depth,
            consecutive_down_days=2,
            reversion_room_atr=depth,
        )

    shallow = components(0.5, 1.0)
    medium = components(1.0, 2.0)
    deep = components(1.5, 2.8)
    for name in ("dislocation", "relative_volume", "residual_dislocation", "reversion_room"):
        assert shallow[name] < medium[name] < deep[name]

    extreme = components(1.5, 8.0)
    assert extreme["relative_volume"] < deep["relative_volume"]


def test_atlas_score_integrity_floor_is_outcome_blind_and_activation_selective() -> None:
    records = []
    for index in range(20):
        value = 0.10 + 0.04 * index
        components = {
            name: min(value + offset * 0.01, 0.95)
            for offset, name in enumerate(OPPORTUNITY_SCORE_WEIGHTS)
        }
        records.append({
            "date": f"2025-01-{(index % 10) + 1:02d}",
            "symbol": f"S{index:02d}",
            "score": score_from_components(components, "balanced"),
            "score_components": components,
            "reversion_room_atr": 0.05 if index < 5 else 0.50,
            "prospective_reward_risk": 0.30 if index < 5 else 1.20,
            "residual_dislocation_atr": -0.10 - 0.01 * index,
            "future_outcome_that_must_not_affect_floor": 999.0 if index % 2 else -999.0,
        })
    audit = _score_integrity(records)
    assert audit["activation_ready"] is True
    assert audit["room_rr_passes"] == 15
    assert audit["room_rr_rejects"] == 5
    assert audit["profile_score_quantiles"]["balanced"]["p90"] > 0.0


def test_residual_reclaim_requires_observed_dislocation_then_recovery() -> None:
    bars = [
        _bar(0, open_=100.0, high=100.1, low=99.5, close=99.6),
        _bar(1, open_=99.6, high=99.9, low=99.4, close=99.85),
        _bar(2, open_=99.85, high=100.0, low=99.8, close=99.95),
    ]
    events = detect_completed_bar_opportunities(
        bars,
        _context(),
        relative_dislocation_atr=[-0.60, -0.38, -0.25],
    )
    event = next(event for event in events if event.family == "MARKET_SECTOR_RESIDUAL_RECLAIM")
    assert event.signal_bar_index == 1
    assert event.entry_bar_index == 2
    assert event.residual_dislocation_atr == pytest.approx(-0.60)


def test_multiday_higher_low_is_disabled_without_prior_down_sequence() -> None:
    bars = [
        _bar(0, open_=100.0, high=100.1, low=99.2, close=99.4),
        _bar(1, open_=99.4, high=99.6, low=99.1, close=99.3),
        _bar(2, open_=99.3, high=99.8, low=99.25, close=99.7),
        _bar(3, open_=99.7, high=99.9, low=99.6, close=99.8),
    ]
    disabled = detect_completed_bar_opportunities(bars, _context(consecutive_down_days=1))
    enabled = detect_completed_bar_opportunities(bars, _context(consecutive_down_days=3))
    assert all(event.family != "MULTIDAY_HIGHER_LOW_RECLAIM" for event in disabled)
    assert any(event.family == "MULTIDAY_HIGHER_LOW_RECLAIM" for event in enabled)


def test_appending_future_bars_does_not_change_existing_completed_signal() -> None:
    prefix = [
        _bar(0, open_=99.0, high=99.2, low=98.8, close=99.15),
        _bar(1, open_=99.2, high=99.5, low=99.1, close=99.4),
    ]
    extended = prefix + [_bar(2, open_=105.0, high=110.0, low=90.0, close=91.0)]
    prefix_event = next(
        event for event in detect_completed_bar_opportunities(prefix, _context())
        if event.family == "GAP_EXHAUSTION_RECLAIM"
    )
    extended_event = next(
        event for event in detect_completed_bar_opportunities(extended, _context())
        if event.family == "GAP_EXHAUSTION_RECLAIM"
    )
    assert prefix_event == extended_event


def test_standardized_outcome_uses_next_open_and_conservative_same_bar_ordering() -> None:
    bars = [
        _bar(0, open_=99.0, high=99.2, low=98.8, close=99.15),
        _bar(1, open_=99.2, high=100.5, low=98.0, close=100.0),
    ]
    event = next(
        event for event in detect_completed_bar_opportunities(bars, _context())
        if event.family == "GAP_EXHAUSTION_RECLAIM"
    )
    outcome = evaluate_standardized_opportunity(
        event,
        bars,
        _context(),
        risk_atr=0.50,
        stop_r=1.0,
        target_r=1.0,
        roundtrip_bps=0.0,
    )
    assert outcome.entry_price == pytest.approx(99.2)
    assert outcome.risk_per_share == pytest.approx(1.0)
    assert outcome.stop_target_r == pytest.approx(-1.0)


def test_entry_variants_are_causal_and_use_only_post_signal_fills() -> None:
    bars = [
        _bar(0, open_=99.0, high=99.2, low=98.8, close=99.15),
        _bar(1, open_=99.2, high=99.35, low=99.05, close=99.30),
        _bar(2, open_=99.4, high=99.8, low=99.2, close=99.7),
    ]
    event = next(
        item for item in detect_completed_bar_opportunities(bars, _context())
        if item.family == "GAP_EXHAUSTION_RECLAIM"
    )

    variants = evaluate_standardized_entry_variants(event, bars, _context(), roundtrip_bps=0.0)

    assert set(variants) == {
        "next_bar_open", "one_bar_confirmation", "resting_25pct_retrace",
    }
    assert variants["next_bar_open"].entry_price == pytest.approx(bars[1].open)
    assert variants["one_bar_confirmation"].entry_price == pytest.approx(bars[2].open)
    assert variants["resting_25pct_retrace"].entry_price < bars[0].close


def test_rearmed_prior_low_requires_explicit_cap_and_completed_bar_separation() -> None:
    bars = [
        _bar(0, open_=98.4, high=99.0, low=98.2, close=98.9),
        _bar(1, open_=99.0, high=99.3, low=98.9, close=99.2),
        _bar(2, open_=99.2, high=99.4, low=99.0, close=99.1),
        _bar(3, open_=99.1, high=99.3, low=99.0, close=99.2),
        _bar(4, open_=99.2, high=99.4, low=99.1, close=99.3),
        _bar(5, open_=99.3, high=99.4, low=99.1, close=99.2),
        _bar(6, open_=98.3, high=99.1, low=98.1, close=99.0),
    ]
    default = detect_completed_bar_opportunities(
        bars,
        _context(),
        require_entry_bar=False,
    )
    rearmed = detect_completed_bar_opportunities(
        bars,
        _context(),
        require_entry_bar=False,
        max_events_per_family={"PRIOR_DAY_LOW_RECLAIM": 2},
        min_event_separation_bars=6,
    )
    default_prior_low = [event for event in default if event.family == "PRIOR_DAY_LOW_RECLAIM"]
    rearmed_prior_low = [event for event in rearmed if event.family == "PRIOR_DAY_LOW_RECLAIM"]
    assert [event.signal_bar_index for event in default_prior_low] == [0]
    assert [event.signal_bar_index for event in rearmed_prior_low] == [0, 6]
    assert [event.event_id for event in rearmed_prior_low] == [
        "PRIOR_DAY_LOW_RECLAIM@0",
        "PRIOR_DAY_LOW_RECLAIM@6",
    ]
