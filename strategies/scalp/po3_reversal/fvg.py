from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .config import TradeDirection
from .models import PriceBar


class FvgState(Enum):
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    INVERTED = "INVERTED"
    RETESTED = "RETESTED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


@dataclass(slots=True)
class FairValueGap:
    id: str
    symbol: str
    timeframe: str
    created_at: datetime
    direction: TradeDirection
    low: float
    high: float
    state: FvgState = FvgState.OPEN
    first_touch_at: datetime | None = None
    last_touch_at: datetime | None = None
    inverted_at: datetime | None = None
    invalidated_at: datetime | None = None
    retested_at: datetime | None = None

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0


class FvgStateMachine:
    def __init__(
        self,
        *,
        symbol: str = "NQ",
        timeframe: str = "1m",
        max_age_bars: int = 20,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_age_bars = max_age_bars
        self._bars: list[PriceBar] = []
        self._gaps: list[FairValueGap] = []
        self._next_id = 0

    @property
    def gaps(self) -> tuple[FairValueGap, ...]:
        return tuple(self._gaps)

    def update(self, bar: PriceBar) -> list[FairValueGap]:
        self._bars.append(bar)
        changed: list[FairValueGap] = []
        if len(self._bars) >= 3:
            changed.extend(self._detect_new_gap())
        for gap in self._gaps:
            if gap in changed:
                continue
            if self._update_gap_state(gap, bar):
                changed.append(gap)
        self._expire_old_gaps()
        return changed

    def get_active_fvgs(self, direction: TradeDirection | int) -> list[FairValueGap]:
        direction = TradeDirection(int(direction))
        return [
            gap
            for gap in self._gaps
            if gap.direction is direction
            and gap.state in {FvgState.OPEN, FvgState.PARTIALLY_FILLED}
        ]

    def get_ifvg_entry(self, direction: TradeDirection | int) -> FairValueGap | None:
        direction = TradeDirection(int(direction))
        for gap in reversed(self._gaps):
            if gap.direction is direction and gap.state in {FvgState.INVERTED, FvgState.RETESTED}:
                return gap
        return None

    def _detect_new_gap(self) -> list[FairValueGap]:
        first, _, third = self._bars[-3:]
        new_gaps: list[FairValueGap] = []
        if first.high < third.low:
            new_gaps.append(self._new_gap(TradeDirection.LONG, first.high, third.low, third.ts))
        if first.low > third.high:
            new_gaps.append(self._new_gap(TradeDirection.SHORT, third.high, first.low, third.ts))
        self._gaps.extend(new_gaps)
        return new_gaps

    def _new_gap(self, direction: TradeDirection, low: float, high: float, ts: datetime) -> FairValueGap:
        self._next_id += 1
        return FairValueGap(
            id=f"{self.symbol}-{self.timeframe}-FVG-{self._next_id}",
            symbol=self.symbol,
            timeframe=self.timeframe,
            created_at=ts,
            direction=direction,
            low=float(low),
            high=float(high),
        )

    def _update_gap_state(self, gap: FairValueGap, bar: PriceBar) -> bool:
        original = gap.state
        touches = bar.low <= gap.high and bar.high >= gap.low
        if touches:
            gap.first_touch_at = gap.first_touch_at or bar.ts
            gap.last_touch_at = bar.ts
            if gap.state is FvgState.OPEN:
                gap.state = FvgState.PARTIALLY_FILLED

        if gap.direction is TradeDirection.LONG and bar.close < gap.low:
            gap.direction = TradeDirection.SHORT
            gap.state = FvgState.INVERTED
            gap.inverted_at = bar.ts
        elif gap.direction is TradeDirection.SHORT and bar.close > gap.high:
            gap.direction = TradeDirection.LONG
            gap.state = FvgState.INVERTED
            gap.inverted_at = bar.ts
        elif gap.state is FvgState.INVERTED and touches:
            gap.state = FvgState.RETESTED
            gap.retested_at = bar.ts
        return gap.state is not original

    def _expire_old_gaps(self) -> None:
        if len(self._bars) <= self.max_age_bars:
            return
        cutoff = self._bars[-self.max_age_bars].ts
        for gap in self._gaps:
            if gap.created_at < cutoff and gap.state in {
                FvgState.OPEN,
                FvgState.PARTIALLY_FILLED,
                FvgState.INVERTED,
            }:
                gap.state = FvgState.EXPIRED

