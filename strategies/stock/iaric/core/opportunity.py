"""Causal, portfolio-free equity opportunity definitions for IARIC research.

This module deliberately sits below nightly selection.  It identifies completed-
bar events on a common symbol universe so opportunity availability can be
measured before ranking, capacity, sizing, or exits obscure the result.  Event
detectors never inspect the nominated entry bar; all entries are the next bar
open.  Future bars are used only by the explicitly observer-only outcome helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log1p, sqrt
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from strategies.stock.iaric.models import Bar
from .lanes import REARMABLE_FAMILIES


REVERSION_FAMILIES = frozenset({
    "GAP_EXHAUSTION_RECLAIM",
    "GAP_FILL_RECLAIM",
    "GAP_PARTIAL_RECLAIM",
    "OPENING_FLUSH_RECLAIM",
    "OPENING_RANGE_LOW_RECLAIM",
    "PRIOR_DAY_LOW_RECLAIM",
    "VWAP_DEVIATION_RECLAIM",
    "FAILED_BREAKDOWN_RECLAIM",
    "MARKET_SECTOR_RESIDUAL_RECLAIM",
    "MULTIDAY_HIGHER_LOW_RECLAIM",
    "UPTREND_PULLBACK_RECLAIM",
    "VOLUME_CLIMAX_RECLAIM",
})
BREAKOUT_REFERENCE_FAMILIES = frozenset({
    "OR_BREAKOUT_REFERENCE",
    "PDH_BREAKOUT_REFERENCE",
})

# Exactly seven fixed, economically signed components.  Weights and scales are
# hypotheses, not sample-fitted estimates.
OPPORTUNITY_SCORE_WEIGHTS: dict[str, float] = {
    "dislocation": 0.17,
    "reclaim": 0.18,
    "close_quality": 0.13,
    "relative_volume": 0.10,
    "residual_dislocation": 0.14,
    "prior_down_sequence": 0.10,
    "reversion_room": 0.18,
}


@dataclass(frozen=True, slots=True)
class DailyOpportunityContext:
    prev_close: float
    prev_high: float
    prev_low: float
    daily_atr: float
    consecutive_down_days: int = 0
    expected_5m_volume: float = 0.0
    expected_5m_profile: tuple[float, ...] = ()
    five_day_return: float = 0.0
    sma20_slope_atr: float = 0.0


@dataclass(frozen=True, slots=True)
class OpportunityEvent:
    family: str
    signal_bar_index: int
    entry_bar_index: int
    signal_time: object
    score: float
    score_components: dict[str, float]
    dislocation_atr: float
    reclaim_atr: float
    close_in_range: float
    relative_volume: float
    residual_dislocation_atr: float
    reversion_room_atr: float
    reversion_anchor: float
    stop_anchor: float
    prospective_reward_risk: float
    episode_start_bar_index: int
    episode_sequence: int
    anchor_kind: str

    @property
    def theme(self) -> str:
        return "reversion" if self.family in REVERSION_FAMILIES else "breakout_reference"

    @property
    def event_id(self) -> str:
        return f"{self.family}@{self.signal_bar_index}"


@dataclass(frozen=True, slots=True)
class StandardizedOpportunityOutcome:
    entry_price: float
    risk_per_share: float
    cost_r: float
    stop_target_r: float
    bars_to_terminal: int
    mfe_r: float
    mae_r: float
    horizon_r: dict[str, float]


def prior_session_volume_expectations(
    prior_bars: Sequence[Bar],
    *,
    fallback_daily_volume: float = 0.0,
) -> tuple[float, tuple[float, ...]]:
    """Return the shared causal time-of-day volume baseline.

    The first-hour mean is retained as a fallback for incomplete profiles,
    while each available prior-session bar supplies the matched expectation
    for the same bar index.  Both live/replay fallback construction and the
    portfolio-free atlas use this function so score thresholds cannot drift
    merely because their relative-volume denominators differ.
    """

    return volume_expectations_from_profile(
        (float(bar.volume) for bar in prior_bars),
        fallback_daily_volume=fallback_daily_volume,
    )


def volume_expectations_from_profile(
    profile_values: Iterable[float],
    *,
    fallback_daily_volume: float = 0.0,
) -> tuple[float, tuple[float, ...]]:
    """Normalize raw prior-session volume values for every adapter."""

    profile = tuple(max(float(value), 1.0) for value in profile_values)
    if profile:
        expected_5m = fmean(profile[: min(len(profile), 12)])
    else:
        expected_5m = max(float(fallback_daily_volume), 0.0) / 78.0
    return max(float(expected_5m), 1.0), profile


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _close_in_range(bar: Bar) -> float:
    width = max(float(bar.high) - float(bar.low), 1e-9)
    return _clip01((float(bar.close) - float(bar.low)) / width)


def opportunity_score_components(
    *,
    dislocation_atr: float,
    reclaim_atr: float,
    close_in_range: float,
    relative_volume: float,
    residual_dislocation_atr: float,
    consecutive_down_days: int,
    reversion_room_atr: float,
) -> dict[str, float]:
    """Return the immutable seven-component event-quality score inputs."""

    depth = abs(float(dislocation_atr))
    participation = max(float(relative_volume), 0.0)
    participation_quality = (
        log1p(participation) / log1p(3.0)
        if participation <= 3.0
        else 1.0 / (1.0 + 0.20 * (participation - 3.0))
    )
    # Scales are fixed economic ranges, not sample quantiles.  They preserve
    # ordering across ordinary 0.3-1.5 ATR events and avoid the old saturation
    # where almost every shock received 1.0 for depth and participation.
    components = {
        "dislocation": _clip01(sqrt(depth / 2.0)),
        "reclaim": _clip01(max(float(reclaim_atr), 0.0) / max(depth, 0.20)),
        "close_quality": _clip01(float(close_in_range)),
        # Participation is hump-shaped: volume confirms capitulation up to
        # roughly 3x normal, while extreme prints progressively lose quality
        # because they are more likely to represent new information.
        "relative_volume": _clip01(participation_quality),
        "residual_dislocation": _clip01(
            sqrt(abs(min(float(residual_dislocation_atr), 0.0)) / 2.0)
        ),
        "prior_down_sequence": _clip01(float(consecutive_down_days) / 4.0),
        "reversion_room": _clip01(sqrt(max(float(reversion_room_atr), 0.0) / 2.0)),
    }
    if set(components) != set(OPPORTUNITY_SCORE_WEIGHTS):
        raise AssertionError("opportunity score must contain exactly seven registered components")
    return components


def opportunity_score(components: dict[str, float]) -> float:
    if set(components) != set(OPPORTUNITY_SCORE_WEIGHTS):
        raise ValueError("opportunity score components do not match the immutable specification")
    return 100.0 * sum(
        OPPORTUNITY_SCORE_WEIGHTS[name] * _clip01(components[name])
        for name in OPPORTUNITY_SCORE_WEIGHTS
    )


def detect_completed_bar_opportunities(
    bars: Sequence[Bar],
    context: DailyOpportunityContext,
    *,
    relative_dislocation_atr: Sequence[float] | None = None,
    opening_range_bars: int = 8,
    require_entry_bar: bool = True,
    max_events_per_family: int | Mapping[str, int] = 1,
    min_event_separation_bars: int = 0,
    allow_episode_updates: bool = False,
) -> list[OpportunityEvent]:
    """Detect capped causal episodes using completed bars and prior daily data.

    ``relative_dislocation_atr`` must be computed from returns observable at
    each corresponding completed bar.  The detector may nominate bar ``N+1``
    as the entry bar but never reads its OHLCV.
    """

    if context.daily_atr <= 0 or context.prev_close <= 0 or len(bars) < 2:
        return []
    if relative_dislocation_atr is not None and len(relative_dislocation_atr) < len(bars):
        raise ValueError("relative dislocation context must cover every supplied completed bar")
    atr = float(context.daily_atr)
    first_open = float(bars[0].open)
    gap_atr = (first_open - float(context.prev_close)) / atr
    expected_volume = max(float(context.expected_5m_volume), 0.0)

    def expected_volume_at(index: int) -> float:
        profile = tuple(float(value) for value in context.expected_5m_profile)
        if 0 <= int(index) < len(profile) and profile[int(index)] > 0:
            return profile[int(index)]
        return expected_volume
    episode_sequences: dict[str, dict[int, int]] = {}
    last_event_bar: dict[str, int] = {}
    events: list[OpportunityEvent] = []
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    running_low = float(bars[0].low)
    running_low_index = 0
    vwap_dislocated = False
    vwap_episode_low = running_low
    vwap_episode_start = 0
    residual_trough = 0.0
    residual_dislocated = False
    residual_episode_start = 0
    climax_high: float | None = None
    climax_low: float | None = None
    climax_index = 0
    prior_low_episode_low = float("inf")
    prior_low_episode_start = 0
    prior_low_armed = False
    prior_low_requires_reset = False
    failed_breakdown_reference = running_low
    failed_breakdown_requires_reset = False
    opening_low_requires_reset = False

    def family_cap(family: str) -> int:
        raw = (
            max_events_per_family.get(family, 1)
            if isinstance(max_events_per_family, Mapping)
            else max_events_per_family
        )
        cap = max(int(raw), 1)
        if family not in REARMABLE_FAMILIES:
            return 1
        return min(cap, 2)

    def emit(
        family: str,
        bar_index: int,
        *,
        anchor: float,
        stop_low: float,
        episode_start: int,
        anchor_kind: str,
        residual_value: float = 0.0,
    ) -> bool:
        # Offline outcome evaluation needs the nominated entry bar to be
        # present.  Live decisioning deliberately does not: after completed
        # signal bar N it may nominate N+1 without reading any field from it.
        family_episodes = episode_sequences.setdefault(family, {})
        episode_key = int(episode_start)
        is_new_episode = episode_key not in family_episodes
        if is_new_episode and len(family_episodes) >= family_cap(family):
            return False
        if not is_new_episode and not allow_episode_updates:
            return False
        prior_event_bar = last_event_bar.get(family)
        if (
            is_new_episode
            and
            prior_event_bar is not None
            and bar_index - prior_event_bar < max(int(min_event_separation_bars), 0)
        ):
            return False
        if require_entry_bar and bar_index + 1 >= len(bars):
            return False
        bar = bars[bar_index]
        close_quality = _close_in_range(bar)
        dislocation = (float(context.prev_close) - running_low) / atr
        reclaim = (float(bar.close) - running_low) / atr
        matched_volume = expected_volume_at(bar_index)
        relative_volume = float(bar.volume) / matched_volume if matched_volume > 0 else 1.0
        room = (float(anchor) - float(bar.close)) / atr
        stop_anchor = min(float(stop_low), float(bar.low)) - 0.05 * atr
        prospective_risk = max(float(bar.close) - stop_anchor, 1e-9)
        prospective_reward_risk = max(float(anchor) - float(bar.close), 0.0) / prospective_risk
        components = opportunity_score_components(
            dislocation_atr=dislocation,
            reclaim_atr=reclaim,
            close_in_range=close_quality,
            relative_volume=relative_volume,
            residual_dislocation_atr=residual_value,
            consecutive_down_days=context.consecutive_down_days,
            reversion_room_atr=room,
        )
        if is_new_episode:
            family_episodes[episode_key] = len(family_episodes) + 1
        events.append(OpportunityEvent(
            family=family,
            signal_bar_index=bar_index,
            entry_bar_index=bar_index + 1,
            signal_time=bar.end_time,
            score=opportunity_score(components),
            score_components=components,
            dislocation_atr=float(dislocation),
            reclaim_atr=float(reclaim),
            close_in_range=float(close_quality),
            relative_volume=float(relative_volume),
            residual_dislocation_atr=float(residual_value),
            reversion_room_atr=float(room),
            reversion_anchor=float(anchor),
            stop_anchor=float(stop_anchor),
            prospective_reward_risk=float(prospective_reward_risk),
            episode_start_bar_index=int(episode_start),
            episode_sequence=family_episodes[episode_key],
            anchor_kind=str(anchor_kind),
        ))
        last_event_bar[family] = bar_index
        return True

    for index, bar in enumerate(bars):
        typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        cumulative_pv += typical * max(float(bar.volume), 0.0)
        cumulative_volume += max(float(bar.volume), 0.0)
        vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else float(bar.close)
        prior_running_low = running_low
        if float(bar.low) < running_low:
            running_low = float(bar.low)
            running_low_index = index
        close_quality = _close_in_range(bar)
        bullish = float(bar.close) > float(bar.open)
        matched_volume = expected_volume_at(index)
        relative_volume = float(bar.volume) / matched_volume if matched_volume > 0 else 1.0
        residual = (
            float(relative_dislocation_atr[index])
            if relative_dislocation_atr is not None
            and isfinite(float(relative_dislocation_atr[index]))
            else 0.0
        )

        if index < 12 and gap_atr <= -0.35 and bullish and close_quality >= 0.60:
            if index == 0 or float(bar.close) > float(bars[index - 1].close):
                emit(
                    "GAP_EXHAUSTION_RECLAIM", index,
                    anchor=context.prev_close, stop_low=running_low,
                    episode_start=0, anchor_kind="previous_close",
                    residual_value=residual,
                )

        gap_size = max(float(context.prev_close) - first_open, 0.0)
        gap_recovery = (
            (float(bar.close) - first_open) / gap_size if gap_size > 0 else 0.0
        )
        if (
            index < 40 and gap_atr <= -0.25 and bullish
            and 0.25 <= gap_recovery < 0.90
            and float(bar.close) < context.prev_close
        ):
            emit(
                "GAP_PARTIAL_RECLAIM", index,
                anchor=context.prev_close, stop_low=running_low,
                episode_start=0, anchor_kind="previous_close",
                residual_value=residual,
            )

        opening_flush = (first_open - running_low) / atr
        if (
            index < 24
            and opening_flush >= 0.45
            and bullish
            and close_quality >= 0.65
            and index >= 1
            and float(bar.close) > float(bars[index - 1].close)
        ):
            emit(
                "OPENING_FLUSH_RECLAIM", index,
                anchor=max(first_open, vwap), stop_low=running_low,
                episode_start=0, anchor_kind="opening_price_or_vwap",
                residual_value=residual,
            )

        if prior_low_requires_reset and (
            float(bar.close) >= context.prev_low + 0.20 * atr
            and float(bar.low) > context.prev_low
        ):
            prior_low_requires_reset = False
        if not prior_low_requires_reset and float(bar.low) <= context.prev_low - 0.03 * atr:
            if not prior_low_armed:
                prior_low_episode_start = index
                prior_low_episode_low = float(bar.low)
            prior_low_armed = True
            prior_low_episode_low = min(prior_low_episode_low, float(bar.low))
        if (
            prior_low_armed and not prior_low_requires_reset
            and float(bar.close) > context.prev_low and bullish and close_quality >= 0.60
        ):
            if emit(
                "PRIOR_DAY_LOW_RECLAIM", index,
                anchor=context.prev_close, stop_low=prior_low_episode_low,
                episode_start=prior_low_episode_start, anchor_kind="previous_close",
                residual_value=residual,
            ):
                prior_low_armed = False
                prior_low_requires_reset = True

        if float(bar.low) <= vwap - 0.35 * atr:
            if not vwap_dislocated:
                vwap_episode_start = index
                vwap_episode_low = float(bar.low)
            vwap_dislocated = True
            vwap_episode_low = min(vwap_episode_low, float(bar.low))
        if (
            vwap_dislocated and bullish and close_quality >= 0.55
            and vwap - 0.30 * atr <= float(bar.close) <= vwap - 0.10 * atr
        ):
            if emit(
                "VWAP_DEVIATION_RECLAIM", index,
                anchor=vwap, stop_low=vwap_episode_low,
                episode_start=vwap_episode_start, anchor_kind="session_vwap",
                residual_value=residual,
            ):
                vwap_dislocated = False

        if failed_breakdown_requires_reset and (
            float(bar.close) >= vwap + 0.15 * atr and float(bar.low) > failed_breakdown_reference
        ):
            failed_breakdown_requires_reset = False
            failed_breakdown_reference = running_low
        if index >= 2 and not failed_breakdown_requires_reset:
            swept = float(bar.low) <= failed_breakdown_reference - 0.08 * atr
            reclaimed = float(bar.close) > failed_breakdown_reference
            if swept and reclaimed and bullish and close_quality >= 0.65:
                if emit(
                    "FAILED_BREAKDOWN_RECLAIM", index,
                    anchor=vwap, stop_low=float(bar.low), episode_start=index,
                    anchor_kind="session_vwap", residual_value=residual,
                ):
                    failed_breakdown_requires_reset = True

        if residual <= -0.50:
            if not residual_dislocated:
                residual_episode_start = index
            residual_dislocated = True
            residual_trough = min(residual_trough, residual)
        residual_recovery = residual - residual_trough
        if (
            residual_dislocated
            and index >= 1
            and residual_recovery >= 0.15
            and bullish
            and close_quality >= 0.55
            and float(bar.close) > float(bars[index - 1].close)
        ):
            residual_anchor = float(bar.close) - min(residual, 0.0) * atr
            if emit(
                "MARKET_SECTOR_RESIDUAL_RECLAIM", index,
                anchor=residual_anchor, stop_low=running_low,
                episode_start=residual_episode_start, anchor_kind="residual_normalization",
                residual_value=residual_trough,
            ):
                residual_dislocated = False
                residual_trough = 0.0

        if (
            context.consecutive_down_days >= 2
            and index >= 2
            and float(bar.low) > float(bars[index - 1].low)
            and float(bars[index - 1].low) <= prior_running_low + 0.10 * atr
            and float(bar.close) > float(bars[index - 1].high)
            and bullish
            and close_quality >= 0.55
        ):
            emit(
                "MULTIDAY_HIGHER_LOW_RECLAIM", index,
                anchor=context.prev_close, stop_low=running_low,
                episode_start=0, anchor_kind="previous_close",
                residual_value=residual,
            )

        if (
            context.sma20_slope_atr >= 0.25
            and -0.12 <= context.five_day_return <= 0.03
            and (context.prev_close - running_low) / atr >= 0.35
            and bullish
            and close_quality >= 0.60
            and float(bar.close) >= vwap
        ):
            emit(
                "UPTREND_PULLBACK_RECLAIM", index,
                anchor=context.prev_close, stop_low=running_low,
                episode_start=0, anchor_kind="previous_close",
                residual_value=residual,
            )

        bar_range = float(bar.high) - float(bar.low)
        if not bullish and bar_range >= 0.35 * atr and relative_volume >= 1.80:
            climax_high = float(bar.high)
            climax_low = float(bar.low)
            climax_index = index
        elif (
            climax_high is not None
            and bullish
            and close_quality >= 0.60
            and float(bar.close) >= (float(climax_high) + float(climax_low or climax_high)) / 2.0
            and float(bar.close) < climax_high
        ):
            if emit(
                "VOLUME_CLIMAX_RECLAIM", index,
                anchor=climax_high, stop_low=float(climax_low or bar.low),
                episode_start=climax_index, anchor_kind="climax_high",
                residual_value=residual,
            ):
                climax_high = None
                climax_low = None

        if index >= max(opening_range_bars, 1):
            opening_range_high = max(float(sample.high) for sample in bars[:opening_range_bars])
            opening_range_low = min(float(sample.low) for sample in bars[:opening_range_bars])
            prior_close = float(bars[index - 1].close)
            if prior_close <= opening_range_high < float(bar.close):
                emit(
                    "OR_BREAKOUT_REFERENCE", index,
                    anchor=context.prev_high, stop_low=opening_range_low,
                    episode_start=0, anchor_kind="previous_high",
                    residual_value=residual,
                )
            if opening_low_requires_reset and float(bar.close) >= opening_range_low + 0.20 * atr:
                opening_low_requires_reset = False
            if (
                not opening_low_requires_reset
                and
                float(bar.low) <= opening_range_low - 0.03 * atr
                and float(bar.close) > opening_range_low
                and bullish
                and close_quality >= 0.60
            ):
                if emit(
                    "OPENING_RANGE_LOW_RECLAIM", index,
                    anchor=max(first_open, vwap), stop_low=float(bar.low),
                    episode_start=index, anchor_kind="opening_price_or_vwap",
                    residual_value=residual,
                ):
                    opening_low_requires_reset = True
        if index >= 1 and float(bars[index - 1].close) <= context.prev_high < float(bar.close):
            emit(
                "PDH_BREAKOUT_REFERENCE", index,
                anchor=context.prev_high, stop_low=running_low,
                episode_start=running_low_index, anchor_kind="previous_high",
                residual_value=residual,
            )

    return events


def evaluate_standardized_opportunity(
    event: OpportunityEvent,
    bars: Sequence[Bar],
    context: DailyOpportunityContext,
    *,
    risk_atr: float = 0.50,
    stop_r: float = 1.0,
    target_r: float = 1.0,
    roundtrip_bps: float = 10.0,
) -> StandardizedOpportunityOutcome:
    """Observer-only equal-risk outcome with conservative OHLC ordering."""

    if event.entry_bar_index <= event.signal_bar_index:
        raise ValueError("opportunity entry must occur after its completed signal bar")
    if event.entry_bar_index >= len(bars):
        raise ValueError("opportunity entry bar is unavailable")
    entry = float(bars[event.entry_bar_index].open)
    return _evaluate_from_entry(
        bars,
        context,
        entry_bar_index=event.entry_bar_index,
        entry_price=entry,
        risk_atr=risk_atr,
        stop_r=stop_r,
        target_r=target_r,
        roundtrip_bps=roundtrip_bps,
        allow_entry_bar_target=True,
    )


def _evaluate_from_entry(
    bars: Sequence[Bar],
    context: DailyOpportunityContext,
    *,
    entry_bar_index: int,
    entry_price: float,
    risk_atr: float,
    stop_r: float,
    target_r: float,
    roundtrip_bps: float,
    allow_entry_bar_target: bool,
) -> StandardizedOpportunityOutcome:
    entry = float(entry_price)
    risk = max(float(context.daily_atr) * max(float(risk_atr), 1e-6), 0.01)
    cost_r = entry * max(float(roundtrip_bps), 0.0) / 10_000.0 / risk
    future = list(bars[entry_bar_index:])
    stop_price = entry - float(stop_r) * risk
    target_price = entry + float(target_r) * risk
    terminal_r: float | None = None
    bars_to_terminal = len(future)
    for offset, bar in enumerate(future, 1):
        stop_hit = float(bar.low) <= stop_price
        target_hit = float(bar.high) >= target_price
        if stop_hit:
            terminal_r = -float(stop_r) - cost_r
            bars_to_terminal = offset
            break
        if target_hit and (offset > 1 or allow_entry_bar_target):
            terminal_r = float(target_r) - cost_r
            bars_to_terminal = offset
            break
    if terminal_r is None:
        terminal_r = (float(future[-1].close) - entry) / risk - cost_r
    mfe_r = (max(float(bar.high) for bar in future) - entry) / risk - cost_r
    mae_r = (min(float(bar.low) for bar in future) - entry) / risk - cost_r
    horizon_r: dict[str, float] = {}
    for horizon in (1, 3, 6, 12, 24, 48):
        end_index = min(horizon - 1, len(future) - 1)
        horizon_r[f"bar_{horizon}"] = (float(future[end_index].close) - entry) / risk - cost_r
    horizon_r["eod"] = (float(future[-1].close) - entry) / risk - cost_r
    return StandardizedOpportunityOutcome(
        entry_price=entry,
        risk_per_share=risk,
        cost_r=cost_r,
        stop_target_r=float(terminal_r),
        bars_to_terminal=int(bars_to_terminal),
        mfe_r=float(mfe_r),
        mae_r=float(mae_r),
        horizon_r=horizon_r,
    )


def evaluate_standardized_entry_variants(
    event: OpportunityEvent,
    bars: Sequence[Bar],
    context: DailyOpportunityContext,
    *,
    risk_atr: float = 0.50,
    stop_r: float = 1.0,
    target_r: float = 1.0,
    roundtrip_bps: float = 10.0,
) -> dict[str, StandardizedOpportunityOutcome]:
    """Return three pre-registered causal entry mechanisms when filled.

    The confirmation route observes bar N+1 and enters at N+2 open.  The
    retrace limit is resting before any eligible fill bar; a target on its fill
    bar is ignored because OHLC cannot prove that the target followed the fill.
    """

    variants = {
        "next_bar_open": evaluate_standardized_opportunity(
            event,
            bars,
            context,
            risk_atr=risk_atr,
            stop_r=stop_r,
            target_r=target_r,
            roundtrip_bps=roundtrip_bps,
        ),
    }
    confirmation_index = event.signal_bar_index + 1
    confirmed_entry_index = confirmation_index + 1
    if confirmed_entry_index < len(bars):
        confirmation = bars[confirmation_index]
        signal = bars[event.signal_bar_index]
        if (
            float(confirmation.close) > float(confirmation.open)
            and float(confirmation.close) > float(signal.close)
            and _close_in_range(confirmation) >= 0.55
        ):
            variants["one_bar_confirmation"] = _evaluate_from_entry(
                bars,
                context,
                entry_bar_index=confirmed_entry_index,
                entry_price=float(bars[confirmed_entry_index].open),
                risk_atr=risk_atr,
                stop_r=stop_r,
                target_r=target_r,
                roundtrip_bps=roundtrip_bps,
                allow_entry_bar_target=True,
            )
    signal_bar = bars[event.signal_bar_index]
    retrace_distance = max(float(event.reclaim_atr) * context.daily_atr * 0.25, 0.02)
    limit_price = float(signal_bar.close) - retrace_distance
    last_limit_index = min(event.signal_bar_index + 3, len(bars) - 1)
    for entry_index in range(event.signal_bar_index + 1, last_limit_index + 1):
        fill_bar = bars[entry_index]
        if float(fill_bar.low) <= limit_price:
            variants["resting_25pct_retrace"] = _evaluate_from_entry(
                bars,
                context,
                entry_bar_index=entry_index,
                entry_price=limit_price,
                risk_atr=risk_atr,
                stop_r=stop_r,
                target_r=target_r,
                roundtrip_bps=roundtrip_bps,
                allow_entry_bar_target=False,
            )
            break
    return variants
