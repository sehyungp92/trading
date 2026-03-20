"""Setup detection: Class M (1H pullback) + Class F (fast re-entry after trailing exit).

v4.0 changes:
- Removed Class A (17% WR, disabled) and Class R (0 fills)
- Removed DivergenceHistory, _class_r_gate
- Added Class F: fast re-entry after profitable trailing exit
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from .config import (
    TF, SetupClass, SetupState, Setup,
    CLASS_M_STOP_ATR_MULT, CLASS_M_PB_MIN, CLASS_M_PB_MAX,
    CLASS_M_MOMENTUM_LOOKBACK,
    CLASS_F_WINDOW_BARS, CLASS_F_MIN_PB_ATR, CLASS_F_MAX_PB_ATR,
    CLASS_F_STOP_ATR_MULT, CLASS_F_SIZE_MULT,
    CLASS_T_TREND_STRENGTH_MIN, CLASS_T_HIST_LOOKBACK,
    CLASS_T_MACD_1H_LOOKBACK, CLASS_T_PIVOT_RECENCY_H,
    CLASS_T_COMPRESSION_MAX, CLASS_T_STOP_ATR_MULT,
    CLASS_T_MAX_STOP_DIST_ATR,
    STRONG_TREND_THRESHOLD, EXTENSION_ATR_MULT,
    REENTRY_LOOKBACK_1H,
)
from .indicators import BarSeries
from .pivots import PivotDetector


def _buffer(atr_val: float) -> float:
    return max(0.25, 0.05 * atr_val)


# ── Alignment & trend helpers ───────────────────────────────────────

def alignment_score(direction: int, daily: BarSeries, h4: BarSeries) -> int:
    s = 0
    if direction == 1:
        if daily.ema_fast() > daily.ema_slow() and daily.last_close > daily.ema_fast():
            s += 1
        if h4.ema_fast() > h4.ema_slow() and h4.last_close > h4.ema_fast():
            s += 1
    else:
        if daily.ema_fast() < daily.ema_slow() and daily.last_close < daily.ema_fast():
            s += 1
        if h4.ema_fast() < h4.ema_slow() and h4.last_close < h4.ema_fast():
            s += 1
    return s


def trend_strength(daily: BarSeries) -> float:
    atr_d = daily.current_atr()
    if atr_d <= 0:
        return 0.0
    return abs(daily.ema_fast() - daily.ema_slow()) / atr_d


def strong_trend_flag(score: int, daily: BarSeries) -> bool:
    return trend_strength(daily) > STRONG_TREND_THRESHOLD and score == 2


def extension_blocked(direction: int, daily: BarSeries) -> bool:
    atr_d = daily.current_atr()
    if direction == 1:
        return daily.last_close > daily.ema_fast() + EXTENSION_ATR_MULT * atr_d
    return daily.last_close < daily.ema_fast() - EXTENSION_ATR_MULT * atr_d


# ── Signal engine ───────────────────────────────────────────────────

class SignalEngine:
    """Detects setups from pivot and indicator state."""

    def __init__(self):
        self._last_exit_dir: dict[int, int] = {}
        self._reentry_used: dict[int, bool] = {}
        self._trend_strength_3d: Optional[float] = None
        # Class F state: trailing exit tracking
        self._trailing_exits: list[_TrailingExitRecord] = []

    def update_trend_strength_3d(self, val: Optional[float]) -> None:
        self._trend_strength_3d = val

    def record_exit(self, direction: int, exit_reason: str = "", bars_held: int = 0) -> None:
        self._last_exit_dir[direction] = 0
        # Track trailing exits for Class F
        if exit_reason == "TRAILING_STOP":
            self._trailing_exits.append(_TrailingExitRecord(
                direction=direction, bars_since=0,
            ))

    def tick_bars(self) -> None:
        for d in list(self._last_exit_dir):
            self._last_exit_dir[d] += 1
            if self._last_exit_dir[d] > REENTRY_LOOKBACK_1H:
                del self._last_exit_dir[d]
        # Age trailing exit records
        for rec in list(self._trailing_exits):
            rec.bars_since += 1
            if rec.bars_since > CLASS_F_WINDOW_BARS:
                self._trailing_exits.remove(rec)

    # ── Class M ─────────────────────────────────────────────────────

    def detect_class_M(
        self, pivots_1h: PivotDetector, h1: BarSeries, h4: BarSeries, daily: BarSeries,
        now_et: datetime,
    ) -> list[Setup]:
        setups: list[Setup] = []
        for direction in (1, -1):
            if direction == 1:
                L1, L2 = pivots_1h.last_two_lows()
                if not (L1 and L2 and L2.price > L1.price):
                    continue
            else:
                H1, H2 = pivots_1h.last_two_highs()
                if not (H1 and H2 and H2.price < H1.price):
                    continue
                L1, L2 = H1, H2

            score = alignment_score(direction, daily, h4)
            if extension_blocked(direction, daily):
                continue

            atr1 = h1.current_atr()
            if atr1 <= 0:
                continue

            # Pullback depth
            if direction == 1:
                recent_high = pivots_1h.most_recent_pivot_high()
                if not recent_high:
                    continue
                pb_depth = recent_high.price - L2.price
            else:
                recent_low = pivots_1h.most_recent_pivot_low()
                if not recent_low:
                    continue
                pb_depth = L2.price - recent_low.price

            if not (CLASS_M_PB_MIN * atr1 <= pb_depth <= CLASS_M_PB_MAX * atr1):
                continue

            # Momentum reclaim
            macd_now = h1.macd_line_now()
            macd_3ago = h1.macd_line_n_ago(CLASS_M_MOMENTUM_LOOKBACK)
            if direction == 1:
                if not (macd_now > macd_3ago or macd_now > L2.macd):
                    continue
            else:
                if not (macd_now < macd_3ago or macd_now < L2.macd):
                    continue

            # Breakout pivot
            if direction == 1:
                H_last = pivots_1h.pivot_high_between(L1.ts, L2.ts)
                if not H_last:
                    continue
                buf = _buffer(atr1)
                entry = H_last.price + buf
                stop0 = L2.price - CLASS_M_STOP_ATR_MULT * atr1
            else:
                L_last = pivots_1h.pivot_low_between(L1.ts, L2.ts)
                if not L_last:
                    continue
                buf = _buffer(atr1)
                entry = L_last.price - buf
                stop0 = L2.price + CLASS_M_STOP_ATR_MULT * atr1
                H_last = L_last

            s = Setup(
                setup_id=str(uuid.uuid4())[:8],
                cls=SetupClass.M,
                direction=direction,
                tf_origin=TF.H1,
                detected_ts=now_et,
                P1=L1, P2=L2,
                breakout_pivot=H_last,
                entry_stop=entry,
                stop0=stop0,
                buffer=buf,
                alignment_score=score,
                strong_trend=strong_trend_flag(score, daily),
                is_extended=False,
            )

            # Re-entry tag
            if (direction in self._last_exit_dir
                    and self._last_exit_dir[direction] <= REENTRY_LOOKBACK_1H
                    and score >= 1
                    and not self._reentry_used.get(direction, False)):
                s.is_reentry = True
                self._reentry_used[direction] = True

            setups.append(s)
        return setups

    # ── Class F (Fast Re-entry) ──────────────────────────────────────

    def detect_class_F(
        self, h1: BarSeries, h4: BarSeries, daily: BarSeries, now_et: datetime,
    ) -> list[Setup]:
        """Detect fast re-entry after a profitable trailing exit.

        Conditions:
        1. A trailing stop exit occurred within CLASS_F_WINDOW_BARS (8) bars
        2. Pullback from exit price is 0.25-1.0 ATR
        3. MACD still directional (same sign as trade direction)
        4. Half-size (0.50x)
        """
        setups: list[Setup] = []
        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        for rec in self._trailing_exits:
            direction = rec.direction
            last_close = h1.last_close

            # MACD must be directional
            macd_now = h1.macd_line_now()
            if direction == 1 and macd_now <= 0:
                continue
            if direction == -1 and macd_now >= 0:
                continue

            # Pullback depth: use highest/lowest since exit as reference
            lookback = min(rec.bars_since + 1, len(h1.bars))
            if lookback < 2:
                continue

            if direction == 1:
                ref_high = h1.highest_high(lookback)
                pb = ref_high - last_close
            else:
                ref_low = h1.lowest_low(lookback)
                pb = last_close - ref_low

            if not (CLASS_F_MIN_PB_ATR * atr1 <= pb <= CLASS_F_MAX_PB_ATR * atr1):
                continue

            score = alignment_score(direction, daily, h4)

            # Entry: current close + buffer, stop: tighter than M
            buf = _buffer(atr1)
            if direction == 1:
                entry = last_close + buf
                stop0 = last_close - CLASS_F_STOP_ATR_MULT * atr1
            else:
                entry = last_close - buf
                stop0 = last_close + CLASS_F_STOP_ATR_MULT * atr1

            s = Setup(
                setup_id=str(uuid.uuid4())[:8],
                cls=SetupClass.F,
                direction=direction,
                tf_origin=TF.H1,
                detected_ts=now_et,
                entry_stop=entry,
                stop0=stop0,
                buffer=buf,
                alignment_score=score,
                strong_trend=strong_trend_flag(score, daily),
                is_extended=False,
                parent_exit_bar=rec.bars_since,
            )
            setups.append(s)

        return setups

    # ── Class T (4H Trend Continuation) ────────────────────────────────

    def detect_class_T(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
    ) -> list[Setup]:
        """Detect 4H trend continuation setups (longs only).

        Fires during strong persistent trends where Class M stays silent
        because clean 1H pullback structure never forms.
        """
        setups: list[Setup] = []

        # Longs only — shorts are hostile
        direction = 1

        # 1. Daily: ema_fast > ema_slow
        if not (daily.ema_fast() > daily.ema_slow()):
            return setups

        # 2. 4H: ema_fast > ema_slow
        if not (h4.ema_fast() > h4.ema_slow()):
            return setups

        # 3. 4H: MACD histogram expanding and positive
        h4_hist_now = h4.macd_hist_now()
        h4_hist_prev = h4.macd_hist_n_ago(CLASS_T_HIST_LOOKBACK)
        if not (h4_hist_now > h4_hist_prev and h4_hist_now > 0):
            return setups

        # 4. Daily: trend_strength > CLASS_T_TREND_STRENGTH_MIN
        ts = trend_strength(daily)
        if ts <= CLASS_T_TREND_STRENGTH_MIN:
            return setups

        # 5. 1H: ema_fast > ema_slow
        if not (h1.ema_fast() > h1.ema_slow()):
            return setups

        # 6. 1H: MACD line > 0 and rising over lookback
        macd_1h_now = h1.macd_line_now()
        macd_1h_prev = h1.macd_line_n_ago(CLASS_T_MACD_1H_LOOKBACK)
        if not (macd_1h_now > 0 and macd_1h_now > macd_1h_prev):
            return setups

        # 7. Most recent pivot high within recency window
        recent_ph = pivots_1h.most_recent_pivot_high()
        if not recent_ph:
            return setups
        hours_since = (now_et - recent_ph.ts).total_seconds() / 3600
        if hours_since > CLASS_T_PIVOT_RECENCY_H:
            return setups

        # 8. No spike/expansion: atr_rolling(6) / atr_rolling(20) < threshold
        atr_short = h1.atr_rolling(6)
        atr_long = h1.atr_rolling(20)
        if atr_long <= 0:
            return setups
        if atr_short / atr_long >= CLASS_T_COMPRESSION_MAX:
            return setups

        # 9. Not extension blocked
        if extension_blocked(direction, daily):
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        # Entry: break above most recent pivot high + buffer
        buf = _buffer(atr1)
        entry = recent_ph.price + buf

        # Stop: most recent pivot low - 0.40 * ATR
        recent_pl = pivots_1h.most_recent_pivot_low()
        if not recent_pl:
            return setups
        stop0 = recent_pl.price - CLASS_T_STOP_ATR_MULT * atr1

        # Max stop distance check
        stop_dist = entry - stop0
        if stop_dist <= 0 or stop_dist > CLASS_T_MAX_STOP_DIST_ATR * atr1:
            return setups

        score = alignment_score(direction, daily, h4)

        s = Setup(
            setup_id=str(uuid.uuid4())[:8],
            cls=SetupClass.T,
            direction=direction,
            tf_origin=TF.H4,
            detected_ts=now_et,
            P1=None,
            P2=None,
            breakout_pivot=recent_ph,
            entry_stop=entry,
            stop0=stop0,
            buffer=buf,
            alignment_score=score,
            strong_trend=strong_trend_flag(score, daily),
            is_extended=False,
        )
        setups.append(s)
        return setups


class _TrailingExitRecord:
    __slots__ = ("direction", "bars_since")

    def __init__(self, direction: int, bars_since: int = 0):
        self.direction = direction
        self.bars_since = bars_since
