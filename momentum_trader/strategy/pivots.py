"""Non-repainting 5-bar confirmed pivot detection."""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Optional

from .config import PIVOT_BARS, TF, Pivot, Bar
from .indicators import BarSeries


class PivotDetector:
    """Detects 5-bar confirmed pivots on a given timeframe.

    Pivot High at bar[t-2] confirmed at bar[t] if High[t-2] == max(High[t-4..t]).
    Pivot Low  at bar[t-2] confirmed at bar[t] if Low[t-2]  == min(Low[t-4..t]).
    """

    def __init__(self, tf: TF):
        self.tf = tf
        self._window: deque[Bar] = deque(maxlen=PIVOT_BARS)
        self.pivot_highs: list[Pivot] = []
        self.pivot_lows: list[Pivot] = []

    def on_bar(self, bar: Bar, series: BarSeries) -> list[Pivot]:
        """Process new bar, return any newly confirmed pivots."""
        self._window.append(bar)
        if len(self._window) < PIVOT_BARS:
            return []

        confirmed = []
        bars = list(self._window)
        mid = bars[2]  # pivot candidate at position 2 (t-2)
        idx = len(series.bars) - 3  # index in series for indicator lookup

        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        if mid.high == max(highs):
            p = Pivot(
                ts=mid.ts,
                kind="PH",
                price=mid.high,
                macd=series.macd_line_at(max(0, idx)),
                atr=series.atr_at(max(0, idx)),
            )
            self.pivot_highs.append(p)
            confirmed.append(p)

        if mid.low == min(lows):
            p = Pivot(
                ts=mid.ts,
                kind="PL",
                price=mid.low,
                macd=series.macd_line_at(max(0, idx)),
                atr=series.atr_at(max(0, idx)),
            )
            self.pivot_lows.append(p)
            confirmed.append(p)

        return confirmed

    def last_two_lows(self) -> tuple[Optional[Pivot], Optional[Pivot]]:
        """Returns (L1, L2) where L2 is more recent."""
        if len(self.pivot_lows) < 2:
            return None, None
        return self.pivot_lows[-2], self.pivot_lows[-1]

    def last_two_highs(self) -> tuple[Optional[Pivot], Optional[Pivot]]:
        """Returns (H1, H2) where H2 is more recent."""
        if len(self.pivot_highs) < 2:
            return None, None
        return self.pivot_highs[-2], self.pivot_highs[-1]

    def pivot_high_between(self, t1: datetime, t2: datetime) -> Optional[Pivot]:
        """Most recent pivot high with timestamp between t1 and t2."""
        for p in reversed(self.pivot_highs):
            if t1 < p.ts < t2:
                return p
        return None

    def pivot_low_between(self, t1: datetime, t2: datetime) -> Optional[Pivot]:
        """Most recent pivot low with timestamp between t1 and t2."""
        for p in reversed(self.pivot_lows):
            if t1 < p.ts < t2:
                return p
        return None

    def most_recent_pivot_high(self) -> Optional[Pivot]:
        return self.pivot_highs[-1] if self.pivot_highs else None

    def most_recent_pivot_low(self) -> Optional[Pivot]:
        return self.pivot_lows[-1] if self.pivot_lows else None
