"""Tests for the IARIC reversion-event repairs.

Covers the four defects fixed in round 2:
  1. live/replay trigger vocabulary divergence (identity gates silently no-oped)
  2. saturated volume component (constant on the route that used it)
  3. constant speed component (flush_bar_idx pinned to 0)
  4. score-as-trigger (no discrete entry event)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from strategies.stock.iaric.core import logic
from strategies.stock.iaric.core.opportunity import prior_session_volume_expectations
from strategies.stock.iaric.core.opportunity import (
    DailyOpportunityContext,
    detect_completed_bar_opportunities,
)
from strategies.stock.iaric.models import Bar


# --------------------------------------------------------------------------
# 1. Trigger vocabulary parity
# --------------------------------------------------------------------------
def test_replay_and_live_trigger_names_normalize_to_the_same_canonical_set():
    """The replay and live emitters spell the same conditions differently.

    Before the repair the shared dislocation gate matched only the live
    spelling, so `dislocation`/`oversold`/`multi_dislocation` admitted nothing
    on the replay path while binding fully in live.
    """
    replay_names = ["DEEP_RSI", "MOD_RSI", "ATR_DEPTH", "BB_EXTREME",
                    "VOL_CAPITULATION", "RS_DIP"]
    live_names = ["RSI2", "RSI5_CDD", "DEPTH", "BB_PCTB",
                  "VOL_CLIMAX", "ROC5_DROP"]
    assert logic.normalize_trigger_types(replay_names) == logic.normalize_trigger_types(live_names)


def test_replay_trigger_names_are_recognised_as_dislocations():
    canonical = logic.normalize_trigger_types(["DEEP_RSI", "ATR_DEPTH"])
    assert canonical & logic.DISLOCATION_TRIGGERS == {"RSI2", "DEPTH"}


def test_unknown_trigger_name_fails_loudly():
    """A trigger added on one side only must not silently disable the gates."""
    with pytest.raises(ValueError, match="unknown IARIC trigger"):
        logic.assert_trigger_vocabulary(["NOT_A_REAL_TRIGGER"])


def test_canonical_names_pass_through_unchanged():
    assert logic.normalize_trigger_types(["RSI2", "GAP_FILL"]) == {"RSI2", "GAP_FILL"}


# --------------------------------------------------------------------------
# 2. RVOL is time-of-day matched and no longer saturates
# --------------------------------------------------------------------------
_T0 = datetime(2025, 6, 2, 13, 30, tzinfo=timezone.utc)


def _bar(volume: float = 1000.0, *, o=100.0, h=101.0, l=99.0, c=100.5) -> Bar:
    return Bar(
        symbol="TEST",
        start_time=_T0,
        end_time=_T0 + timedelta(minutes=5),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume,
    )


def test_rvol_uses_the_bar_index_baseline_not_a_flat_average():
    """A flat session average makes the 09:30 bar an extreme multiple always."""

    class _Item:
        expected_5m_volume = 1000.0
        average_30m_volume = 6000.0
        # Prior session was U-shaped: heavy open, quiet midday.
        expected_5m_profile = (8000.0, 3000.0, 2000.0, 1500.0)

    item = _Item()
    opening = _bar(volume=8000.0)
    # Against the flat average this is 8.0x; against the time-of-day baseline
    # it is an ordinary open.
    assert logic.compute_volume_ratio(opening, item) == pytest.approx(8.0)
    assert logic.compute_rvol(opening, item, 0) == pytest.approx(1.0)


def test_rvol_score_does_not_saturate_across_the_realistic_range():
    """The legacy transform clipped everything above 0.875x to the same value."""
    values = [logic.rvol_score(x) for x in (1.0, 1.5, 3.0, 6.0)]
    assert values == sorted(values), "rvol_score must be monotone increasing"
    assert len(set(values)) == 4, "each level must be distinguishable"
    assert 0.0 <= values[0] < values[-1] <= 1.0


def test_rvol_falls_back_to_the_flat_estimate_without_a_profile():
    class _Item:
        expected_5m_volume = 1000.0
        average_30m_volume = 6000.0
        expected_5m_profile = ()

    item = _Item()
    assert logic.compute_rvol(_bar(volume=2000.0), item, 0) == pytest.approx(2.0)


def test_shared_prior_session_volume_expectation_matches_time_of_day_and_fallback():
    bars = [_bar(volume=value) for value in (8000.0, 3000.0, 2000.0, 1500.0)]
    expected, profile = prior_session_volume_expectations(
        bars,
        fallback_daily_volume=780000.0,
    )
    assert profile == (8000.0, 3000.0, 2000.0, 1500.0)
    assert expected == pytest.approx(3625.0)

    fallback_expected, fallback_profile = prior_session_volume_expectations(
        [],
        fallback_daily_volume=780000.0,
    )
    assert fallback_profile == ()
    assert fallback_expected == pytest.approx(10000.0)


def test_rejected_first_observation_does_not_monopolize_a_one_shot_episode():
    bars = [
        _bar(o=98.0, h=99.0, l=97.8, c=98.8, volume=1000.0),
        _bar(o=98.7, h=99.5, l=98.6, c=99.3, volume=1000.0),
    ]
    context = DailyOpportunityContext(
        prev_close=100.0,
        prev_high=101.0,
        prev_low=98.5,
        daily_atr=2.0,
        expected_5m_volume=1000.0,
        expected_5m_profile=(1000.0, 1000.0),
    )
    first_only = [
        event for event in detect_completed_bar_opportunities(
            bars,
            context,
            require_entry_bar=False,
        )
        if event.family == "GAP_EXHAUSTION_RECLAIM"
    ]
    observations = [
        event for event in detect_completed_bar_opportunities(
            bars,
            context,
            require_entry_bar=False,
            allow_episode_updates=True,
        )
        if event.family == "GAP_EXHAUSTION_RECLAIM"
    ]
    assert len(first_only) == 1
    assert [event.signal_bar_index for event in observations] == [0, 1]
    assert {event.episode_sequence for event in observations} == {1}


# --------------------------------------------------------------------------
# 3 & 4. Dislocation band and the completed-bar reclaim event
# --------------------------------------------------------------------------
@dataclass
class _Settings:
    pb_v2_dislocation_band_atr: float = 0.35
    pb_v2_dislocation_use_prev_low: bool = True
    pb_v2_open_scored_rvol_min: float = 0.0
    pb_v2_event_stop_anchor: str = "session_low"
    pb_v2_event_stop_min_atr: float = 0.25


@dataclass
class _State:
    prev_close: float = 100.0
    prev_low: float = 98.0
    daily_atr: float = 2.0
    item: object = None


@dataclass
class _Market:
    bars_5m: list = field(default_factory=list)
    session_vwap: float | None = None


def test_dislocation_band_is_daily_anchored_and_takes_the_shallower_anchor():
    settings, state = _Settings(), _State()
    # prev_close - 0.35*2.0 = 99.3 ; prev_low = 98.0 -> shallower is 99.3
    assert logic.dislocation_band(settings, state) == pytest.approx(99.3)


def test_dislocation_band_is_zero_without_prior_session_anchors():
    settings = _Settings()
    state = _State(prev_close=0.0, prev_low=0.0, daily_atr=0.0)
    assert logic.dislocation_band(settings, state) == 0.0


def test_band_reclaim_requires_a_prior_bar_below_the_band():
    """Dislocation must be established on a strictly earlier completed bar."""
    settings, state = _Settings(), _State()
    # Single wide bar that dips below and closes above must NOT confirm:
    # otherwise one bar would satisfy both halves of the event.
    lone = _bar(o=99.5, h=100.5, l=99.0, c=100.2)
    market = _Market(bars_5m=[lone])
    assert logic.band_reclaim_confirmed(settings, lone, market, state=state) is False


def test_band_reclaim_confirms_on_dislocation_then_reclaim():
    settings, state = _Settings(), _State()
    dislocated = _bar(o=99.4, h=99.5, l=99.0, c=99.1)   # low 99.0 <= band 99.3
    reclaim = _bar(o=99.2, h=99.8, l=99.15, c=99.7)     # closes above 99.3, green
    market = _Market(bars_5m=[dislocated, reclaim])
    assert logic.band_reclaim_confirmed(settings, reclaim, market, state=state) is True


def test_band_reclaim_rejects_a_red_reclaim_bar():
    """A gap-through with a weak close is not a reclaim."""
    settings, state = _Settings(), _State()
    dislocated = _bar(o=99.4, h=99.5, l=99.0, c=99.1)
    weak = _bar(o=99.9, h=100.0, l=99.35, c=99.5)  # above band but red
    market = _Market(bars_5m=[dislocated, weak])
    assert logic.band_reclaim_confirmed(settings, weak, market, state=state) is False


def test_band_reclaim_rejects_a_close_still_below_the_band():
    settings, state = _Settings(), _State()
    dislocated = _bar(o=99.4, h=99.5, l=99.0, c=99.1)
    still_low = _bar(o=99.0, h=99.25, l=98.9, c=99.2)  # green but below 99.3
    market = _Market(bars_5m=[dislocated, still_low])
    assert logic.band_reclaim_confirmed(settings, still_low, market, state=state) is False


def test_band_reclaim_enforces_the_rvol_floor():
    class _Item:
        expected_5m_volume = 1000.0
        average_30m_volume = 6000.0
        expected_5m_profile = (1000.0, 1000.0)

    settings = _Settings(pb_v2_open_scored_rvol_min=1.50)
    state = _State(item=_Item())
    dislocated = _bar(o=99.4, h=99.5, l=99.0, c=99.1, volume=1000.0)
    thin = _bar(o=99.2, h=99.8, l=99.15, c=99.7, volume=900.0)   # 0.9x
    heavy = _bar(o=99.2, h=99.8, l=99.15, c=99.7, volume=2000.0)  # 2.0x
    assert logic.band_reclaim_confirmed(
        settings, thin, _Market(bars_5m=[dislocated, thin]), state=state
    ) is False
    assert logic.band_reclaim_confirmed(
        settings, heavy, _Market(bars_5m=[dislocated, heavy]), state=state
    ) is True


def test_dislocation_bar_index_is_not_pinned_to_zero():
    """`speed` was arithmetically constant because flush_bar_idx was hardcoded 0."""
    settings, state = _Settings(), _State()
    series = [
        _bar(o=100.0, h=100.2, l=99.8, c=100.0),   # 0: above band
        _bar(o=100.0, h=100.1, l=99.1, c=99.4),    # 1: below band 99.3
        _bar(o=99.4, h=99.6, l=99.35, c=99.5),     # 2: back above
        _bar(o=99.5, h=99.9, l=99.45, c=99.8),     # 3: reclaim
    ]
    assert logic._dislocation_bar_index(settings, state, series, 3) == 1


def test_event_stop_anchors_on_the_reclaim_bar_when_requested():
    settings = _Settings(pb_v2_event_stop_anchor="reclaim_bar")
    state = _State()
    reclaim = _bar(o=99.2, h=99.8, l=99.15, c=99.7)
    # Session low is far below after the dislocation; the reclaim anchor must
    # be tighter than it but never tighter than the ATR floor.
    anchor = logic.event_stop_anchor(settings, state, reclaim, session_low=95.0, daily_atr=2.0)
    assert 95.0 <= anchor <= 99.2


def test_event_stop_defaults_to_session_low():
    settings, state = _Settings(), _State()
    reclaim = _bar(o=99.2, h=99.8, l=99.15, c=99.7)
    assert logic.event_stop_anchor(settings, state, reclaim, 95.0, 2.0) == 95.0


def test_reclaim_or_limit_is_an_accepted_transition():
    class _S:
        pb_open_scored_transition = "reclaim_or_limit"

    assert logic.open_scored_transition(_S()) == "reclaim_or_limit"


def test_unknown_transition_is_rejected():
    class _S:
        pb_open_scored_transition = "teleport"

    with pytest.raises(ValueError, match="pb_open_scored_transition"):
        logic.open_scored_transition(_S())
