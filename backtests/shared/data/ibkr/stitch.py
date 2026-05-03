"""Continuous futures stitching helpers."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from .store import ensure_utc_index, merge_frames


def round_to_tick(value: float, tick_size: float) -> float:
    if tick_size <= 0:
        return value
    return round(round(value / tick_size) * tick_size, 10)


def stitch_panama(
    contract_data: dict[str, pd.DataFrame],
    rolls: list[tuple[date, str, str]],
    *,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """Stitch physical contracts with backward Panama adjustment."""
    normalized = {key: ensure_utc_index(df) for key, df in contract_data.items() if df is not None and not df.empty}
    if not normalized:
        return pd.DataFrame()

    all_rolls = sorted(rolls, key=lambda item: item[0])
    valid_rolls = [
        (roll_date, old, new)
        for roll_date, old, new in all_rolls
        if old in normalized and new in normalized
    ]
    if not all_rolls:
        return merge_frames(*normalized.values())

    ordered_months: list[str] = []
    for _roll_date, old, new in all_rolls:
        if old in normalized and old not in ordered_months:
            ordered_months.append(old)
        if new in normalized and new not in ordered_months:
            ordered_months.append(new)
    for month in sorted(normalized):
        if month not in ordered_months:
            ordered_months.append(month)

    segments: list[tuple[str, pd.DataFrame]] = []
    for month in ordered_months:
        frame = normalized.get(month)
        if frame is None or frame.empty:
            continue
        lower = _lower_roll(all_rolls, month)
        upper = _upper_roll(all_rolls, month)
        segment = frame
        if lower is not None:
            segment = segment[segment.index >= lower]
        if upper is not None:
            segment = segment[segment.index < upper]
        if not segment.empty:
            segments.append((month, segment))

    adjustments: dict[str, float] = {}
    cumulative = 0.0
    if ordered_months:
        adjustments[ordered_months[-1]] = 0.0

    for roll_date, old_month, new_month in reversed(all_rolls):
        if old_month not in normalized:
            continue
        if new_month not in normalized:
            adjustments[old_month] = cumulative
            continue
        roll_ts = pd.Timestamp(datetime.combine(roll_date, datetime.min.time()), tz="UTC")
        old_frame = normalized[old_month]
        new_frame = normalized[new_month]
        old_before = old_frame[old_frame.index < roll_ts]
        new_after = new_frame[new_frame.index >= roll_ts]
        if old_before.empty or new_after.empty:
            adjustments[old_month] = cumulative
            continue
        gap = round_to_tick(float(new_after.iloc[0]["open"]) - float(old_before.iloc[-1]["close"]), tick_size)
        cumulative += gap
        adjustments[old_month] = cumulative

    adjusted: list[pd.DataFrame] = []
    for month, segment in segments:
        adjustment = adjustments.get(month, 0.0)
        if adjustment:
            segment = segment.copy()
            for column in ("open", "high", "low", "close"):
                if column in segment.columns:
                    segment[column] = segment[column] - adjustment
        adjusted.append(segment)

    return merge_frames(*adjusted)


def _lower_roll(rolls: list[tuple[date, str, str]], month: str) -> pd.Timestamp | None:
    for roll_date, _old, new in rolls:
        if new == month:
            return pd.Timestamp(datetime.combine(roll_date, datetime.min.time()), tz="UTC")
    return None


def _upper_roll(rolls: list[tuple[date, str, str]], month: str) -> pd.Timestamp | None:
    for roll_date, old, _new in rolls:
        if old == month:
            return pd.Timestamp(datetime.combine(roll_date, datetime.min.time()), tz="UTC")
    return None
