"""BRS rolling bar series with lazy indicator recomputation.

Maintains per-timeframe rolling deques of OHLCV bars. Indicators are
recomputed vectorially (same functions as backtest) only when the series
is marked dirty (new bar appended).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from .config import TF
from .indicators import (
    adx_suite,
    atr,
    compute_ema_slope,
    donchian_low,
    ema,
    swing_high_confirmed,
    volume_sma,
)


@dataclass(slots=True)
class Bar:
    """Single OHLCV bar."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class BRSBarSeries:
    """Rolling window of bars with lazy vectorized indicator recomputation.

    On ``add_bar`` the series is marked dirty.  On any indicator property
    access the full indicator suite is recomputed from the deque arrays.
    Cost is microseconds for 200-500 bar windows.
    """

    def __init__(self, tf: TF, maxlen: int = 500) -> None:
        self.tf = tf
        self.bars: deque[Bar] = deque(maxlen=maxlen)
        self._dirty = True

        # Raw numpy caches
        self._opens: Optional[np.ndarray] = None
        self._highs: Optional[np.ndarray] = None
        self._lows: Optional[np.ndarray] = None
        self._closes: Optional[np.ndarray] = None
        self._volumes: Optional[np.ndarray] = None

        # Indicator caches -- populated per timeframe
        # Daily (D1)
        self._ema_fast: Optional[np.ndarray] = None     # ema(15)
        self._ema_slow: Optional[np.ndarray] = None     # ema(40)
        self._ema_fast_slope: Optional[np.ndarray] = None
        self._ema_slow_slope: Optional[np.ndarray] = None
        self._atr14: Optional[np.ndarray] = None
        self._atr50: Optional[np.ndarray] = None
        self._adx: Optional[np.ndarray] = None
        self._plus_di: Optional[np.ndarray] = None
        self._minus_di: Optional[np.ndarray] = None

        # 4H (H4) -- ema(50), atr(14), adx(14)
        self._ema50: Optional[np.ndarray] = None

        # Hourly (H1) -- ema(20), ema(34), ema(50), atr(14),
        #   donchian_low(26), donchian_low(10), swing_high(5), volume_sma(20)
        self._ema20: Optional[np.ndarray] = None
        self._ema34: Optional[np.ndarray] = None
        self._donch_low_26: Optional[np.ndarray] = None
        self._donch_low_10: Optional[np.ndarray] = None
        self._swing_highs: Optional[np.ndarray] = None
        self._vol_sma20: Optional[np.ndarray] = None

    # ── mutation ──────────────────────────────────────────────────────

    def add_bar(self, bar: Bar) -> None:
        self.bars.append(bar)
        self._dirty = True

    def __len__(self) -> int:
        return len(self.bars)

    # ── internal recompute ────────────────────────────────────────────

    def _extract_arrays(self) -> None:
        n = len(self.bars)
        if n == 0:
            return
        self._opens = np.array([b.open for b in self.bars], dtype=np.float64)
        self._highs = np.array([b.high for b in self.bars], dtype=np.float64)
        self._lows = np.array([b.low for b in self.bars], dtype=np.float64)
        self._closes = np.array([b.close for b in self.bars], dtype=np.float64)
        self._volumes = np.array([b.volume for b in self.bars], dtype=np.float64)

    def _recompute(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._extract_arrays()
        n = len(self.bars)
        if n < 2:
            return

        h = self._highs
        l = self._lows
        c = self._closes

        # Always compute ATR14 and ADX suite
        self._atr14 = atr(h, l, c, 14)
        self._adx, self._plus_di, self._minus_di = adx_suite(h, l, c, 14)

        if self.tf == TF.D1:
            self._ema_fast = ema(c, 15)
            self._ema_slow = ema(c, 40)
            self._ema_fast_slope = compute_ema_slope(self._ema_fast, 5)
            self._ema_slow_slope = compute_ema_slope(self._ema_slow, 5)
            self._atr50 = atr(h, l, c, 50)

        elif self.tf == TF.H4:
            self._ema50 = ema(c, 50)

        elif self.tf == TF.H1:
            self._ema20 = ema(c, 20)
            self._ema34 = ema(c, 34)
            self._ema50 = ema(c, 50)
            self._donch_low_26 = donchian_low(l, 26)
            self._donch_low_10 = donchian_low(l, 10)
            self._swing_highs = swing_high_confirmed(h, 5)
            if self._volumes is not None:
                self._vol_sma20 = volume_sma(self._volumes, 20)

    # ── safe accessor helpers ─────────────────────────────────────────

    def _last(self, arr: Optional[np.ndarray]) -> float:
        if arr is None or len(arr) == 0:
            return 0.0
        return float(arr[-1])

    def _prior(self, arr: Optional[np.ndarray], n: int = 1) -> float:
        if arr is None or len(arr) <= n:
            return 0.0
        return float(arr[-1 - n])

    # ── public properties ─────────────────────────────────────────────

    def ensure_computed(self) -> None:
        self._recompute()

    # -- Raw bar access --
    @property
    def last_bar(self) -> Optional[Bar]:
        return self.bars[-1] if self.bars else None

    @property
    def prior_bar(self) -> Optional[Bar]:
        return self.bars[-2] if len(self.bars) >= 2 else None

    @property
    def closes(self) -> np.ndarray:
        self._recompute()
        return self._closes if self._closes is not None else np.array([], dtype=np.float64)

    @property
    def highs(self) -> np.ndarray:
        self._recompute()
        return self._highs if self._highs is not None else np.array([], dtype=np.float64)

    @property
    def lows(self) -> np.ndarray:
        self._recompute()
        return self._lows if self._lows is not None else np.array([], dtype=np.float64)

    @property
    def volumes(self) -> np.ndarray:
        self._recompute()
        return self._volumes if self._volumes is not None else np.array([], dtype=np.float64)

    # -- Daily indicators --
    @property
    def latest_ema_fast(self) -> float:
        self._recompute()
        return self._last(self._ema_fast)

    @property
    def latest_ema_slow(self) -> float:
        self._recompute()
        return self._last(self._ema_slow)

    @property
    def latest_ema_fast_slope(self) -> float:
        self._recompute()
        return self._last(self._ema_fast_slope)

    @property
    def latest_ema_slow_slope(self) -> float:
        self._recompute()
        return self._last(self._ema_slow_slope)

    @property
    def latest_atr14(self) -> float:
        self._recompute()
        return self._last(self._atr14)

    @property
    def latest_atr50(self) -> float:
        self._recompute()
        return self._last(self._atr50)

    @property
    def atr14_history(self) -> np.ndarray:
        self._recompute()
        return self._atr14 if self._atr14 is not None else np.array([], dtype=np.float64)

    @property
    def latest_adx(self) -> float:
        self._recompute()
        return self._last(self._adx)

    @property
    def latest_plus_di(self) -> float:
        self._recompute()
        return self._last(self._plus_di)

    @property
    def latest_minus_di(self) -> float:
        self._recompute()
        return self._last(self._minus_di)

    # -- 4H indicators --
    @property
    def latest_ema50(self) -> float:
        self._recompute()
        return self._last(self._ema50)

    @property
    def ema50_arr(self) -> np.ndarray:
        self._recompute()
        return self._ema50 if self._ema50 is not None else np.array([], dtype=np.float64)

    @property
    def adx_arr(self) -> np.ndarray:
        self._recompute()
        return self._adx if self._adx is not None else np.array([], dtype=np.float64)

    @property
    def atr14_arr(self) -> np.ndarray:
        self._recompute()
        return self._atr14 if self._atr14 is not None else np.array([], dtype=np.float64)

    # -- Hourly indicators --
    @property
    def latest_ema20(self) -> float:
        self._recompute()
        return self._last(self._ema20)

    @property
    def latest_ema34(self) -> float:
        self._recompute()
        return self._last(self._ema34)

    @property
    def latest_ema50_h(self) -> float:
        self._recompute()
        return self._last(self._ema50)

    @property
    def latest_donchian_low_26(self) -> float:
        self._recompute()
        return self._last(self._donch_low_26)

    @property
    def prior_donchian_low_26(self) -> float:
        """Prior bar's Donchian low (S3 period=26)."""
        self._recompute()
        return self._prior(self._donch_low_26, 1)

    @property
    def prior_donchian_low_bd(self) -> float:
        """Prior bar's Donchian low (BD period=10)."""
        self._recompute()
        return self._prior(self._donch_low_10, 1)

    @property
    def latest_swing_high(self) -> float:
        self._recompute()
        return self._last(self._swing_highs)

    @property
    def latest_volume_sma20(self) -> float:
        self._recompute()
        return self._last(self._vol_sma20)
