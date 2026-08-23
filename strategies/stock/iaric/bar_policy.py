"""Authoritative completed-five-minute bar contract for IARIC.

The optimized replay and the live engine both call ``apply_completed_5m_bar``.
Keeping the market-state transition here prevents input granularity or wrapper
code from silently changing the strategy that was backtested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import math
from typing import Any

from .config import ET
from .models import Bar, MarketSnapshot


FIVE_MINUTES = timedelta(minutes=5)
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
THIRTY_MINUTE_BAR_COUNT = 6


class Completed5mContractError(ValueError):
    """A bar cannot enter IARIC's action-generating five-minute clock."""


class Completed5mGapError(Completed5mContractError):
    """A completed five-minute interval is missing from a symbol stream."""


def exchange_timestamp(timestamp: datetime) -> datetime:
    """Return one timestamp on the authoritative New York session clock.

    Historical replay timestamps are occasionally timezone-naive.  The legacy
    stock store records those values as UTC, so that conversion is explicit
    here instead of being reimplemented by strategy adapters.
    """

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(ET)


def is_completed_rth_5m_bar(bar: Bar) -> bool:
    """Whether ``bar`` belongs to IARIC's action-generating RTH clock."""

    start_et = exchange_timestamp(bar.start_time)
    end_et = exchange_timestamp(bar.end_time)
    return bool(
        bar.end_time - bar.start_time == FIVE_MINUTES
        and start_et.date() == end_et.date()
        and start_et.second == 0
        and start_et.microsecond == 0
        and start_et.minute % 5 == 0
        and end_et.second == 0
        and end_et.microsecond == 0
        and end_et.minute % 5 == 0
        and RTH_OPEN <= start_et.time() < RTH_CLOSE
        and RTH_OPEN < end_et.time() <= RTH_CLOSE
    )


def completed_rth_5m_bars(bars: list[Bar]) -> list[Bar]:
    """Filter replay input onto the same session-relative clock used live."""

    return [bar for bar in bars if is_completed_rth_5m_bar(bar)]


@dataclass(frozen=True, slots=True)
class Completed5mTransition:
    """Observable result of applying one canonical five-minute bar."""

    bar_index: int
    session_vwap: float | None
    completed_30m_bar: Bar | None = None


def validate_completed_5m_bar(
    bar: Bar,
    *,
    received_at: datetime,
    expected_symbol: str | None = None,
) -> None:
    """Enforce the live-core completed-five-minute input contract.

    Timestamps are start/end timestamps for the half-open interval
    ``[start_time, end_time)``. Only completed RTH ``TRADES`` bars aligned to
    the New York five-minute clock are accepted.
    """

    if bar.start_time.tzinfo is None or bar.end_time.tzinfo is None:
        raise Completed5mContractError("IARIC 5m bars require timezone-aware timestamps")
    if received_at.tzinfo is None:
        raise Completed5mContractError("IARIC 5m receipt timestamps must be timezone-aware")
    if expected_symbol is not None and bar.symbol.upper() != expected_symbol.upper():
        raise Completed5mContractError(
            f"IARIC 5m symbol mismatch: expected {expected_symbol.upper()}, got {bar.symbol.upper()}"
        )
    if bar.end_time - bar.start_time != FIVE_MINUTES:
        raise Completed5mContractError(
            "IARIC action-generating input must be exactly one completed 5-minute bar"
        )
    start_et = exchange_timestamp(bar.start_time)
    end_et = exchange_timestamp(bar.end_time)
    if start_et.date() != end_et.date():
        raise Completed5mContractError("IARIC 5m bars cannot cross an ET session date")
    if start_et.second or start_et.microsecond or start_et.minute % 5:
        raise Completed5mContractError("IARIC 5m bar start is not aligned to the ET 5-minute clock")
    if end_et.second or end_et.microsecond or end_et.minute % 5:
        raise Completed5mContractError("IARIC 5m bar end is not aligned to the ET 5-minute clock")
    if not (RTH_OPEN <= start_et.time() < RTH_CLOSE) or not (RTH_OPEN < end_et.time() <= RTH_CLOSE):
        raise Completed5mContractError("IARIC 5m bars must be inside the 09:30-16:00 ET RTH session")
    if bar.end_time > received_at:
        raise Completed5mContractError("IARIC cannot evaluate an incomplete 5-minute bar")

    values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
    if not all(math.isfinite(float(value)) for value in values):
        raise Completed5mContractError("IARIC 5m OHLCV values must be finite")
    if min(bar.open, bar.high, bar.low, bar.close) <= 0 or bar.volume < 0:
        raise Completed5mContractError("IARIC 5m prices must be positive and volume non-negative")
    if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close) or bar.high < bar.low:
        raise Completed5mContractError("IARIC 5m OHLC values are inconsistent")


def validate_next_completed_5m_bar(previous: Bar | None, current: Bar) -> None:
    """Reject gaps and out-of-order bars within an RTH session."""

    if previous is None:
        return
    if current.start_time <= previous.start_time:
        raise Completed5mContractError("IARIC 5m bars must be strictly monotonic")
    previous_date = previous.start_time.astimezone(ET).date()
    current_date = current.start_time.astimezone(ET).date()
    if previous_date == current_date and current.start_time != previous.end_time:
        raise Completed5mGapError(
            f"IARIC 5m gap for {current.symbol}: expected {previous.end_time.isoformat()}, "
            f"got {current.start_time.isoformat()}"
        )


def aggregate_completed_30m_bars(symbol: str, bars: list[Bar]) -> Bar:
    """Aggregate exactly six contiguous completed five-minute bars."""

    if len(bars) != THIRTY_MINUTE_BAR_COUNT:
        raise Completed5mContractError("IARIC 30m context requires exactly six completed 5m bars")
    for previous, current in zip(bars, bars[1:]):
        if previous.end_time != current.start_time:
            raise Completed5mGapError("IARIC cannot derive 30m context across a missing 5m interval")
    return Bar(
        symbol=symbol,
        start_time=bars[0].start_time,
        end_time=bars[-1].end_time,
        open=bars[0].open,
        high=max(bar.high for bar in bars),
        low=min(bar.low for bar in bars),
        close=bars[-1].close,
        volume=sum(bar.volume for bar in bars),
    )


def apply_completed_5m_bar(
    market: MarketSnapshot,
    bar: Bar,
    *,
    state: Any | None = None,
    aggregation_bar_index: int | None = None,
    aggregation_bar_count: int = THIRTY_MINUTE_BAR_COUNT,
) -> Completed5mTransition:
    """Apply the optimized replay's canonical market-state transition.

    Contract validation and duplicate/gap policy belong to the live ingress.
    Replay data is already completed and validated by its data pipeline, so it
    calls this transition directly.
    """

    market.last_price = bar.close
    market.last_5m_bar = bar
    market.bars_5m.append(bar)
    market.session_high = bar.high if market.session_high is None else max(market.session_high, bar.high)
    market.session_low = bar.low if market.session_low is None else min(market.session_low, bar.low)
    market._cum_pv += bar.typical_price * bar.volume
    market._cum_vol += bar.volume
    market.session_vwap = market._cum_pv / max(market._cum_vol, 1.0)

    bar_index = len(market.bars_5m) - 1
    if state is not None:
        state.bars_seen_today = int(getattr(state, "bars_seen_today", 0)) + 1
        state.last_5m_bar_time = bar.end_time
        state.session_high = max(float(getattr(state, "session_high", 0.0)), bar.high)
        session_low = float(getattr(state, "session_low", 0.0))
        state.session_low = bar.low if session_low <= 0 else min(session_low, bar.low)

    completed_30m_bar = None
    clock_index = bar_index if aggregation_bar_index is None else int(aggregation_bar_index)
    window_count = max(1, int(aggregation_bar_count))
    if (clock_index + 1) % window_count == 0 and len(market.bars_5m) >= window_count:
        completed_30m_bar = aggregate_completed_30m_bars(
            bar.symbol,
            list(market.bars_5m)[-window_count:],
        )
        market.last_30m_bar = completed_30m_bar
        market.bars_30m.append(completed_30m_bar)

    return Completed5mTransition(
        bar_index=bar_index,
        session_vwap=market.session_vwap,
        completed_30m_bar=completed_30m_bar,
    )
