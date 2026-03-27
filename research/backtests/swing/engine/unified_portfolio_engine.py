"""Unified 5-strategy portfolio backtesting engine.

Runs ATRSS, AKC-Helix, Swing Breakout v3.3, S5-PB (Keltner Pullback),
and S5-Dual (Keltner Dual) under a single shared OMS simulation with
portfolio-level heat caps, priority-aware reservation, per-strategy
ceilings, and cross-strategy coordination.

Mirrors the production setup in main_multi.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, date

import numpy as np
import pandas as pd

from strategy.config import SYMBOL_CONFIGS as ATRSS_SYMBOL_CONFIGS
from strategy.models import Direction as AtrssDirection

from strategy_2.config import SYMBOL_CONFIGS as HELIX_SYMBOL_CONFIGS
from strategy_2.models import Direction as HelixDirection

from strategy_3.config import SYMBOL_CONFIGS as BREAKOUT_SYMBOL_CONFIGS
from strategy_3.config import _ETF_CONFIGS as BREAKOUT_ETF_CONFIGS
from strategy_3.models import (
    CircuitBreakerState as BreakoutCBState,
    Direction as BreakoutDirection,
    PositionState as BreakoutPositionState,
)

from strategy_4.config import SYMBOL_CONFIGS as S5_SYMBOL_CONFIGS
from backtest.engine.s5_engine import S5Engine

from backtest.config_unified import UnifiedBacktestConfig, StrategySlot
from backtest.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    build_numpy_arrays,
    filter_rth,
    normalize_timezone,
    resample_1h_to_4h,
)
from backtest.engine.backtest_engine import BacktestEngine, _AblationPatch
from backtest.engine.helix_engine import HelixEngine
from backtest.engine.breakout_engine import BreakoutEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class UnifiedPortfolioData:
    """Pre-loaded data for all symbols across all strategies.

    ATRSS requires RTH-filtered hourly data (matching its individual runner).
    Helix/Breakout use full hourly data (matching their individual runners).
    For shared symbols (e.g. QQQ), both versions are stored separately.
    """

    daily: dict[str, NumpyBars] = field(default_factory=dict)
    # RTH-filtered hourly data for ATRSS (UTC, RTH only)
    atrss_hourly: dict[str, NumpyBars] = field(default_factory=dict)
    atrss_daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    # RTH-filtered hourly data for Breakout (ET timezone, RTH only)
    breakout_hourly: dict[str, NumpyBars] = field(default_factory=dict)
    breakout_four_hour: dict[str, NumpyBars] = field(default_factory=dict)
    breakout_daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    breakout_four_hour_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    # Full hourly data for Helix (no RTH filter)
    hourly: dict[str, NumpyBars] = field(default_factory=dict)
    four_hour: dict[str, NumpyBars] = field(default_factory=dict)
    daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    four_hour_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)


def load_unified_data(config: UnifiedBacktestConfig) -> UnifiedPortfolioData:
    """Load parquet data for all symbols used by any strategy.

    Each strategy's individual runner loads data differently:
    - ATRSS: normalize_timezone(UTC) + filter_rth
    - Helix: raw parquet (no filter)
    - Breakout: tz_convert(ET) + filter_rth, 4H resampled from RTH-filtered

    We replicate each strategy's exact data pipeline to ensure trade alignment.
    """
    from pathlib import Path
    from zoneinfo import ZoneInfo
    data_dir = Path(config.data_dir)
    portfolio = UnifiedPortfolioData()

    atrss_set = set(config.atrss_symbols)
    breakout_set = set(config.breakout_symbols)
    overlay_syms = config.overlay_symbols if config.overlay_enabled else []
    all_symbols = sorted(set(
        config.atrss_symbols + config.helix_symbols + config.breakout_symbols
        + config.s5_pb_symbols + config.s5_dual_symbols
        + overlay_syms
    ))

    for sym in all_symbols:
        hourly_path = data_dir / f"{sym}_1h.parquet"
        daily_path = data_dir / f"{sym}_1d.parquet"

        if not hourly_path.exists() or not daily_path.exists():
            logger.warning("Missing data for %s, skipping", sym)
            continue

        hourly_df = pd.read_parquet(hourly_path)
        daily_df = pd.read_parquet(daily_path)

        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            hourly_df.index = pd.DatetimeIndex(hourly_df.index)
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            daily_df.index = pd.DatetimeIndex(daily_df.index)

        # Date range filtering (reduces bar count → faster engine run)
        if config.start_date:
            hourly_df = hourly_df[hourly_df.index >= config.start_date]
            daily_df = daily_df[daily_df.index >= config.start_date]
        if config.end_date:
            hourly_df = hourly_df[hourly_df.index <= config.end_date]
            daily_df = daily_df[daily_df.index <= config.end_date]

        # Standardize column names
        hourly_df.columns = hourly_df.columns.str.lower()
        daily_df.columns = daily_df.columns.str.lower()

        # --- Helix: raw unfiltered hourly (always built) ---
        portfolio.daily[sym] = build_numpy_arrays(daily_df)
        portfolio.hourly[sym] = build_numpy_arrays(hourly_df)
        four_hour_df = resample_1h_to_4h(hourly_df)
        portfolio.four_hour[sym] = build_numpy_arrays(four_hour_df)
        portfolio.daily_idx_maps[sym] = align_daily_to_hourly(hourly_df, daily_df)
        portfolio.four_hour_idx_maps[sym] = align_4h_to_hourly(hourly_df, four_hour_df)

        # --- ATRSS: normalize_timezone(UTC) + filter_rth ---
        if sym in atrss_set:
            atrss_h = normalize_timezone(hourly_df.copy())
            atrss_d = normalize_timezone(daily_df.copy())
            atrss_h = filter_rth(atrss_h)
            portfolio.atrss_hourly[sym] = build_numpy_arrays(atrss_h)
            portfolio.atrss_daily_idx_maps[sym] = align_daily_to_hourly(atrss_h, atrss_d)

        # --- Breakout: tz_convert(ET) + filter_rth, 4H from RTH-filtered ---
        if sym in breakout_set:
            et = ZoneInfo("America/New_York")
            bo_h = hourly_df.copy()
            if bo_h.index.tz is not None:
                bo_h = bo_h.tz_convert(et)
            else:
                bo_h = bo_h.tz_localize("UTC").tz_convert(et)
            bo_h = filter_rth(bo_h)
            bo_4h = resample_1h_to_4h(bo_h)
            portfolio.breakout_hourly[sym] = build_numpy_arrays(bo_h)
            portfolio.breakout_four_hour[sym] = build_numpy_arrays(bo_4h)
            portfolio.breakout_daily_idx_maps[sym] = align_daily_to_hourly(bo_h, daily_df)
            portfolio.breakout_four_hour_idx_maps[sym] = align_4h_to_hourly(bo_h, bo_4h)

    logger.info("Loaded data for %d symbols: %s", len(portfolio.hourly), list(portfolio.hourly))
    return portfolio


# ---------------------------------------------------------------------------
# Heat tracking
# ---------------------------------------------------------------------------

@dataclass
class StrategyHeatState:
    """Per-strategy heat tracking."""

    strategy_id: str = ""
    priority: int = 0
    unit_risk_dollars: float = 0.0
    max_heat_R: float = 1.0
    daily_stop_R: float = 2.0
    open_risk_dollars: float = 0.0
    daily_realized_dollars: float = 0.0

    @property
    def daily_realized_R(self) -> float:
        if self.unit_risk_dollars <= 0:
            return 0.0
        return self.daily_realized_dollars / self.unit_risk_dollars

    @property
    def open_risk_R(self) -> float:
        if self.unit_risk_dollars <= 0:
            return 0.0
        return self.open_risk_dollars / self.unit_risk_dollars


class PortfolioHeatTracker:
    """Portfolio-level heat cap enforcement.

    Mirrors the risk gateway logic in shared/oms/risk/gateway.py.
    """

    def __init__(
        self,
        heat_cap_R: float,
        portfolio_daily_stop_R: float,
        strategy_slots: list[StrategySlot],
        equity: float,
        simulate_live_r_normalization: bool = False,
    ):
        self._heat_cap_R = heat_cap_R
        self._portfolio_daily_stop_R = portfolio_daily_stop_R
        self._strats: dict[str, StrategyHeatState] = {}
        self._portfolio_halted = False
        self._simulate_live_r = simulate_live_r_normalization

        # Use highest-priority strategy's unit_risk_pct as the portfolio
        # normalization base so that portfolio-level R comparisons are
        # consistent regardless of which strategy requests entry.
        sorted_slots = sorted(strategy_slots, key=lambda s: s.priority)
        self._portfolio_unit_risk_pct = sorted_slots[0].unit_risk_pct if sorted_slots else 0.01
        self._portfolio_unit_risk = equity * self._portfolio_unit_risk_pct

        for slot in strategy_slots:
            urd = equity * slot.unit_risk_pct
            self._strats[slot.strategy_id] = StrategyHeatState(
                strategy_id=slot.strategy_id,
                priority=slot.priority,
                unit_risk_dollars=urd,
                max_heat_R=slot.max_heat_R,
                daily_stop_R=slot.daily_stop_R,
            )

    def update_unit_risk(self, equity: float, slots: list[StrategySlot]) -> None:
        """Recalibrate unit risk dollars from current equity."""
        self._portfolio_unit_risk = equity * self._portfolio_unit_risk_pct
        for slot in slots:
            if slot.strategy_id in self._strats:
                self._strats[slot.strategy_id].unit_risk_dollars = equity * slot.unit_risk_pct

    def can_enter(self, strategy_id: str, risk_dollars: float) -> tuple[bool, str]:
        """Check whether a new entry is allowed."""
        strat = self._strats.get(strategy_id)
        if strat is None:
            return False, f"Unknown strategy {strategy_id}"

        # Portfolio daily halt
        if self._portfolio_halted:
            return False, "Portfolio daily stop hit"

        # Use consistent portfolio-level normalization base so that the same
        # dollar amount of risk maps to the same R regardless of which
        # strategy is requesting entry.
        pu = self._portfolio_unit_risk

        total_daily_dollars = sum(s.daily_realized_dollars for s in self._strats.values())
        if pu > 0:
            portfolio_daily_R = total_daily_dollars / pu
            if portfolio_daily_R <= -self._portfolio_daily_stop_R:
                self._portfolio_halted = True
                return False, "Portfolio daily stop hit"

        # Strategy daily halt (uses strategy's own unit risk — correct)
        if strat.daily_realized_R <= -strat.daily_stop_R:
            return False, f"{strategy_id} daily stop hit"

        # Per-strategy heat ceiling (uses strategy's own unit risk — correct)
        if strat.unit_risk_dollars > 0:
            new_risk_R = risk_dollars / strat.unit_risk_dollars
            if strat.open_risk_R + new_risk_R > strat.max_heat_R:
                return False, f"{strategy_id} heat ceiling ({strat.open_risk_R:.2f}R + {new_risk_R:.2f}R > {strat.max_heat_R}R)"

        # Portfolio heat cap
        total_open_dollars = sum(s.open_risk_dollars for s in self._strats.values())
        if self._simulate_live_r:
            # Live OMS bug simulation: each strategy's open risk is converted
            # to R using its own URD, so 1R_atrss != 1R_helix in the sum.
            total_open_R = sum(s.open_risk_R for s in self._strats.values())
            new_risk_R = risk_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else float("inf")
        elif pu > 0:
            # Correct: use consistent portfolio-level unit risk
            total_open_R = total_open_dollars / pu
            new_risk_R = risk_dollars / pu
        else:
            total_open_R = 0.0
            new_risk_R = 0.0
        if total_open_R + new_risk_R > self._heat_cap_R:
            return False, f"Portfolio heat cap ({total_open_R:.2f}R + {new_risk_R:.2f}R > {self._heat_cap_R}R)"

        # Priority reservation: if remaining capacity is tight, reserve for higher priority
        if self._simulate_live_r:
            remaining_R = self._heat_cap_R - total_open_R
            remaining_dollars = remaining_R * strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0
        else:
            remaining_dollars = (self._heat_cap_R * pu) - total_open_dollars if pu > 0 else 0
        if remaining_dollars < 2 * risk_dollars:
            for other in self._strats.values():
                if other.priority < strat.priority and other.open_risk_dollars == 0:
                    return False, f"Heat reserved for {other.strategy_id}"

        return True, ""

    def update_open_risk(self, risk_by_strategy: dict[str, float]) -> None:
        """Refresh open risk from current positions."""
        for sid, dollars in risk_by_strategy.items():
            if sid in self._strats:
                self._strats[sid].open_risk_dollars = dollars

    def record_trade_close(self, strategy_id: str, pnl_dollars: float) -> None:
        """Update daily realized PnL after a trade close."""
        if strategy_id in self._strats:
            self._strats[strategy_id].daily_realized_dollars += pnl_dollars

    def on_new_day(self) -> None:
        """Reset daily counters."""
        for s in self._strats.values():
            s.daily_realized_dollars = 0.0
        self._portfolio_halted = False

    @property
    def total_open_risk_dollars(self) -> float:
        return sum(s.open_risk_dollars for s in self._strats.values())


# ---------------------------------------------------------------------------
# Cross-strategy coordination (simplified backtest version)
# ---------------------------------------------------------------------------

class BacktestCoordinator:
    """Simplified coordinator for backtest.

    Rule 1: ATRSS entry on symbol X → tighten Helix stop to BE on X.
    Rule 2: ATRSS has active position on symbol X → Helix 1.25x size boost.
    """

    def __init__(self, enable_tighten: bool = True, enable_size_boost: bool = True):
        self._enable_tighten = enable_tighten
        self._enable_size_boost = enable_size_boost
        # symbol → (direction, entry_price)
        self._atrss_positions: dict[str, tuple[int, float]] = {}
        self._pending_tighten: list[str] = []  # symbols needing Helix stop tighten
        self.tighten_count = 0
        self.boost_count = 0

    def on_atrss_position_change(
        self, symbol: str, direction: int, entry_price: float,
    ) -> None:
        """Track ATRSS position changes."""
        if direction != 0:  # FLAT
            was_flat = symbol not in self._atrss_positions
            self._atrss_positions[symbol] = (direction, entry_price)
            if was_flat and self._enable_tighten:
                self._pending_tighten.append(symbol)
        else:
            self._atrss_positions.pop(symbol, None)

    def consume_tighten_events(self) -> list[str]:
        """Return and clear pending tighten events."""
        events = list(self._pending_tighten)
        self._pending_tighten.clear()
        return events

    def has_atrss_position(self, symbol: str, direction: int) -> bool:
        """Check if ATRSS has an active position on symbol in the given direction."""
        if not self._enable_size_boost:
            return False
        pos = self._atrss_positions.get(symbol)
        if pos is None:
            return False
        return pos[0] == direction


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class UnifiedHeatStats:
    """Portfolio heat utilization statistics."""

    avg_heat_pct: float = 0.0
    max_heat_pct: float = 0.0
    pct_time_at_cap: float = 0.0


@dataclass
class StrategyResult:
    """Aggregated results for one strategy."""

    strategy_id: str = ""
    total_trades: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    total_r: float = 0.0
    max_drawdown_dollars: float = 0.0
    entries_blocked_by_heat: int = 0


@dataclass
class UnifiedPortfolioResult:
    """Combined results from the unified backtest."""

    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_stats: UnifiedHeatStats = field(default_factory=UnifiedHeatStats)
    strategy_results: dict[str, StrategyResult] = field(default_factory=dict)
    coordination_tighten_count: int = 0
    coordination_boost_count: int = 0
    portfolio_daily_stop_activations: int = 0
    overlay_pnl: float = 0.0
    overlay_per_symbol_pnl: dict[str, float] = field(default_factory=dict)
    # Raw per-engine results for detailed analysis
    atrss_trades: list = field(default_factory=list)
    helix_trades: list = field(default_factory=list)
    breakout_trades: list = field(default_factory=list)
    s5_pb_trades: list = field(default_factory=list)
    s5_dual_trades: list = field(default_factory=list)
    # Diagnostic event logs for portfolio diagnostics
    heat_rejections: list = field(default_factory=list)
    coordination_events: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: get point value from strategy configs
# ---------------------------------------------------------------------------

def _get_point_value(symbol: str) -> float:
    """Get point value from whichever strategy config has it."""
    for cfgs in (ATRSS_SYMBOL_CONFIGS, HELIX_SYMBOL_CONFIGS, BREAKOUT_SYMBOL_CONFIGS, BREAKOUT_ETF_CONFIGS):
        cfg = cfgs.get(symbol)
        if cfg is not None:
            return cfg.multiplier
    return 1.0


# ---------------------------------------------------------------------------
# Compute open risk across all engines
# ---------------------------------------------------------------------------

def _compute_open_risk(
    atrss_engines: dict[str, BacktestEngine],
    helix_engines: dict[str, HelixEngine],
    breakout_engines: dict[str, BreakoutEngine],
    s5_pb_engines: dict[str, S5Engine] | None = None,
    s5_dual_engines: dict[str, S5Engine] | None = None,
) -> dict[str, float]:
    """Returns {strategy_id: total_risk_dollars}."""
    risk: dict[str, float] = {
        "ATRSS": 0.0, "AKC_HELIX": 0.0, "SWING_BREAKOUT_V3": 0.0,
        "S5_PB": 0.0, "S5_DUAL": 0.0,
    }

    for sym, eng in atrss_engines.items():
        pos = eng.position
        if pos.direction != AtrssDirection.FLAT and pos.base_leg is not None:
            r = abs(pos.base_leg.entry_price - pos.current_stop) * eng.point_value * pos.total_qty
            risk["ATRSS"] += r

    for sym, eng in helix_engines.items():
        pos = eng.active_position
        if pos is not None and pos.qty_open > 0:
            r = abs(pos.fill_price - pos.current_stop) * eng.point_value * pos.qty_open
            risk["AKC_HELIX"] += r

    for sym, eng in breakout_engines.items():
        pos = eng.active_position
        if pos is not None and pos.qty_open > 0:
            r = abs(pos.fill_price - pos.current_stop) * eng.point_value * pos.qty_open
            risk["SWING_BREAKOUT_V3"] += r

    for strat_id, s5_dict in [("S5_PB", s5_pb_engines), ("S5_DUAL", s5_dual_engines)]:
        if s5_dict is None:
            continue
        for sym, eng in s5_dict.items():
            pos = eng.active_position
            if pos is not None:
                r = abs(pos.fill_price - pos.current_stop) * eng.point_value * pos.qty
                risk[strat_id] += r

    return risk


# ---------------------------------------------------------------------------
# Idle-capital overlay helpers
# ---------------------------------------------------------------------------

def _overlay_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed (matches idle_capital_study.py)."""
    out = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return out
    out[period - 1] = np.mean(series[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def _overlay_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed RSI for overlay (self-contained, no cross-strategy imports)."""
    n = len(closes)
    out = np.full(n, 50.0)
    if n < 2:
        return out

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.empty(len(deltas), dtype=float)
    avg_loss = np.empty(len(deltas), dtype=float)

    seed = min(period, len(deltas))
    avg_gain[0] = float(np.mean(gains[:seed]))
    avg_loss[0] = float(np.mean(losses[:seed]))

    alpha = 1.0 / period
    for i in range(1, len(deltas)):
        avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gains[i] * alpha
        avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + losses[i] * alpha

    for i in range(len(deltas)):
        ag = avg_gain[i]
        al = avg_loss[i]
        if al == 0:
            out[i + 1] = 100.0 if ag > 0 else 50.0
        else:
            rs = ag / al
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


def _overlay_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD for overlay -> (line, signal_line, histogram). Self-contained.

    The MACD line is valid from index (slow-1) onward.  The signal line EMA
    is computed only over the valid portion of the MACD line, then placed back
    into a full-length array so indices align with the input closes.
    """
    n = len(closes)
    ema_fast = _overlay_ema(closes, fast)
    ema_slow = _overlay_ema(closes, slow)
    line = ema_fast - ema_slow  # valid from index slow-1

    # Compute signal EMA over the valid tail of the MACD line
    valid_start = slow - 1  # first index where line is not NaN
    valid_line = line[valid_start:]
    valid_sig = _overlay_ema(valid_line, signal)

    # Place back into full-length arrays
    sig = np.full(n, np.nan, dtype=float)
    sig[valid_start:] = valid_sig
    hist = line - sig
    return line, sig, hist


def _precompute_overlay_rsi(
    daily: dict[str, NumpyBars],
    config: UnifiedBacktestConfig,
) -> dict[str, np.ndarray]:
    """Precompute RSI arrays per overlay symbol."""
    rsi_arrays: dict[str, np.ndarray] = {}
    for sym in config.overlay_symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        overrides = config.overlay_rsi_overrides.get(sym, {})
        period = overrides.get("period", config.overlay_rsi_period)
        rsi_arrays[sym] = _overlay_rsi(bars.closes, period)
    return rsi_arrays


def _precompute_overlay_macd(
    daily: dict[str, NumpyBars],
    config: UnifiedBacktestConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute MACD arrays per overlay symbol."""
    macd_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym in config.overlay_symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        overrides = config.overlay_macd_overrides.get(sym, {})
        fast = overrides.get("fast", config.overlay_macd_fast)
        slow = overrides.get("slow", config.overlay_macd_slow)
        sig = overrides.get("signal", config.overlay_macd_signal)
        macd_arrays[sym] = _overlay_macd(bars.closes, fast, slow, sig)
    return macd_arrays


def _compute_overlay_score(
    sym: str,
    d_idx: int,
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]],
    overlay_rsi: dict[str, np.ndarray],
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: UnifiedBacktestConfig,
) -> float:
    """Compute composite overlay score [0, 1] for a symbol at a daily bar index.

    Components:
    - EMA score (weight 0.40): 0-1 based on spread magnitude
    - RSI score (weight 0.30): 1.0 in sweet spot [50-65], 0.0 if overbought or < bull_min
    - MACD score (weight 0.30): 1.0 if histogram positive+rising, 0.0 if negative+falling
    """
    W_EMA, W_RSI, W_MACD = config.overlay_score_weights

    # --- EMA score ---
    ema_score = 0.0
    ema_pair = overlay_emas.get(sym)
    if ema_pair is not None:
        ema_f, ema_s = ema_pair
        if not (np.isnan(ema_f[d_idx]) or np.isnan(ema_s[d_idx])):
            spread = (ema_f[d_idx] - ema_s[d_idx]) / ema_s[d_idx] if ema_s[d_idx] != 0 else 0.0
            if spread > 0:
                ema_score = min(spread / config.overlay_ema_spread_norm, 1.0)

    # --- RSI score ---
    rsi_score = 0.0
    rsi_arr = overlay_rsi.get(sym)
    if rsi_arr is not None:
        rsi_overrides = config.overlay_rsi_overrides.get(sym, {})
        overbought = rsi_overrides.get("overbought", config.overlay_rsi_overbought)
        bull_min = rsi_overrides.get("bull_min", config.overlay_rsi_bull_min)
        rsi_val = rsi_arr[d_idx]
        if rsi_val >= overbought:
            rsi_score = 0.0
        elif rsi_val < bull_min:
            rsi_score = 0.0
        elif 50 <= rsi_val <= 65:
            rsi_score = 1.0
        elif rsi_val < 50:
            # Ramp from bull_min to 50
            rsi_score = (rsi_val - bull_min) / (50 - bull_min) if (50 - bull_min) > 0 else 0.0
        else:
            # Ramp down from 65 to overbought
            rsi_score = (overbought - rsi_val) / (overbought - 65) if (overbought - 65) > 0 else 0.0

    # --- MACD score ---
    macd_score = 0.0
    macd_data = overlay_macd.get(sym)
    if macd_data is not None:
        _line, _sig, hist = macd_data
        if not np.isnan(hist[d_idx]):
            hist_val = hist[d_idx]
            hist_prev = hist[d_idx - 1] if d_idx > 0 and not np.isnan(hist[d_idx - 1]) else hist_val
            rising = hist_val > hist_prev
            s_pr, s_pf, s_nf, s_nr = config.overlay_macd_scores
            if hist_val > 0 and rising:
                macd_score = s_pr
            elif hist_val > 0:
                macd_score = s_pf  # positive but falling
            elif hist_val < 0 and not rising:
                macd_score = s_nf  # negative and falling
            else:
                macd_score = s_nr  # negative but rising (recovery)

    return W_EMA * ema_score + W_RSI * rsi_score + W_MACD * macd_score


def _precompute_overlay_emas(
    daily: dict[str, NumpyBars],
    config: UnifiedBacktestConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute fast/slow EMAs on daily closes for overlay symbols."""
    emas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym in config.overlay_symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        fast, slow = config.overlay_ema_overrides.get(
            sym, (config.overlay_ema_fast, config.overlay_ema_slow),
        )
        ema_fast = _overlay_ema(bars.closes, fast)
        ema_slow = _overlay_ema(bars.closes, slow)
        emas[sym] = (ema_fast, ema_slow)
    return emas


def _build_daily_date_index(
    daily: dict[str, NumpyBars],
    symbols: list[str],
) -> dict[str, dict[date, int]]:
    """Build {symbol: {date: bar_index}} for daily data lookup."""
    idx: dict[str, dict[date, int]] = {}
    for sym in symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        mapping: dict[date, int] = {}
        for i in range(len(bars.times)):
            t = bars.times[i]
            if hasattr(t, 'item'):
                t = t.item()
            dt = pd.Timestamp(t).date() if not isinstance(t, date) else t
            if isinstance(dt, datetime):
                dt = dt.date()
            mapping[dt] = i
        idx[sym] = mapping
    return idx


def _rebalance_overlay(
    config: UnifiedBacktestConfig,
    daily: dict[str, NumpyBars],
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]],
    daily_date_idx: dict[str, dict[date, int]],
    current_date: date,
    portfolio_equity: float,
    overlay_shares: dict[str, float],
    overlay_rsi: dict[str, np.ndarray] | None = None,
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    overlay_in_position: dict[str, bool] | None = None,
) -> None:
    """Rebalance overlay positions at day boundary.

    Legacy "ema" mode: equal/weighted across bullish symbols (EMA fast > slow).
    "multi" mode: EMA+RSI+MACD composite scoring with hysteresis and adaptive sizing.
    """
    if config.overlay_mode == "multi" and overlay_rsi is not None and overlay_macd is not None:
        _rebalance_overlay_multi(
            config, daily, overlay_emas, daily_date_idx, current_date,
            portfolio_equity, overlay_shares, overlay_rsi, overlay_macd,
            overlay_in_position or {},
        )
        return

    # --- Legacy "ema" mode (unchanged) ---
    available = max(portfolio_equity * config.overlay_max_pct, 0.0)
    signals: dict[str, bool] = {}
    n_bullish = 0

    for sym in config.overlay_symbols:
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            signals[sym] = False
            continue
        d_idx = sym_idx.get(current_date)
        fast, slow = config.overlay_ema_overrides.get(
            sym, (config.overlay_ema_fast, config.overlay_ema_slow),
        )
        if d_idx is None or d_idx < slow:
            signals[sym] = False
            continue
        ema_f, ema_s = overlay_emas[sym]
        signals[sym] = bool(ema_f[d_idx] > ema_s[d_idx])
        if signals[sym]:
            n_bullish += 1

    # Compute per-symbol allocation (weighted or equal-weight)
    if config.overlay_weights is None:
        bullish_w = {s: 1.0 for s in config.overlay_symbols if signals.get(s)}
    else:
        bullish_w = {s: config.overlay_weights.get(s, 1.0)
                     for s in config.overlay_symbols if signals.get(s)}
    total_w = sum(bullish_w.values())

    for sym in config.overlay_symbols:
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            overlay_shares[sym] = 0.0
            continue
        d_idx = sym_idx.get(current_date)
        if d_idx is None:
            overlay_shares[sym] = 0.0
            continue
        price = daily[sym].closes[d_idx]
        if signals.get(sym, False) and price > 0 and total_w > 0:
            alloc = available * bullish_w[sym] / total_w
            overlay_shares[sym] = int(alloc / price)
        else:
            overlay_shares[sym] = 0.0


def _rebalance_overlay_multi(
    config: UnifiedBacktestConfig,
    daily: dict[str, NumpyBars],
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]],
    daily_date_idx: dict[str, dict[date, int]],
    current_date: date,
    portfolio_equity: float,
    overlay_shares: dict[str, float],
    overlay_rsi: dict[str, np.ndarray],
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    overlay_in_position: dict[str, bool],
) -> None:
    """Multi-indicator overlay rebalance with scoring, hysteresis, and adaptive sizing."""
    available = max(portfolio_equity * config.overlay_max_pct, 0.0)
    scores: dict[str, float] = {}
    active_syms: dict[str, float] = {}  # sym -> score for syms that pass hysteresis

    for sym in config.overlay_symbols:
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            overlay_shares[sym] = 0.0
            continue
        d_idx = sym_idx.get(current_date)
        fast, slow = config.overlay_ema_overrides.get(
            sym, (config.overlay_ema_fast, config.overlay_ema_slow),
        )
        if d_idx is None or d_idx < slow:
            overlay_shares[sym] = 0.0
            overlay_in_position[sym] = False
            continue

        score = _compute_overlay_score(
            sym, d_idx, overlay_emas, overlay_rsi, overlay_macd, config,
        )
        scores[sym] = score

        currently_in = overlay_in_position.get(sym, False)
        if currently_in:
            # Hysteresis: stay in unless score drops below exit threshold
            if score < config.overlay_exit_score_max:
                overlay_in_position[sym] = False
            else:
                active_syms[sym] = score
        else:
            # Entry: require score >= entry threshold
            if score >= config.overlay_entry_score_min:
                overlay_in_position[sym] = True
                active_syms[sym] = score

    # Allocate capital across active symbols
    if not active_syms:
        for sym in config.overlay_symbols:
            overlay_shares[sym] = 0.0
        return

    # Weighted allocation: base weight * adaptive scaling
    if config.overlay_weights is None:
        base_weights = {s: 1.0 for s in active_syms}
    else:
        base_weights = {s: config.overlay_weights.get(s, 1.0) for s in active_syms}
    total_base_w = sum(base_weights.values())

    for sym in config.overlay_symbols:
        if sym not in active_syms:
            overlay_shares[sym] = 0.0
            continue

        sym_idx = daily_date_idx.get(sym, {})
        d_idx = sym_idx.get(current_date)
        if d_idx is None:
            overlay_shares[sym] = 0.0
            continue
        price = daily[sym].closes[d_idx]
        if price <= 0 or total_base_w <= 0:
            overlay_shares[sym] = 0.0
            continue

        # Base allocation from weights
        alloc_pct = base_weights[sym] / total_base_w

        # Adaptive sizing: scale allocation by signal strength
        if config.overlay_adaptive_sizing:
            score = active_syms[sym]
            # Map score to [min_alloc_pct, max_alloc_pct]
            alloc_range = config.overlay_max_alloc_pct - config.overlay_min_alloc_pct
            size_factor = config.overlay_min_alloc_pct + alloc_range * score
        else:
            size_factor = 1.0

        alloc = available * alloc_pct * size_factor
        overlay_shares[sym] = int(alloc / price)


def _compute_overlay_value(
    overlay_shares: dict[str, float],
    daily: dict[str, NumpyBars],
    daily_date_idx: dict[str, dict[date, int]],
    current_date: date,
) -> float:
    """MTM value of overlay positions using daily close prices."""
    value = 0.0
    for sym, shares in overlay_shares.items():
        if shares == 0.0:
            continue
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            continue
        d_idx = sym_idx.get(current_date)
        if d_idx is None:
            continue
        value += shares * daily[sym].closes[d_idx]
    return value


# ---------------------------------------------------------------------------
# Main entry: run_unified
# ---------------------------------------------------------------------------

def run_unified(
    data: UnifiedPortfolioData,
    config: UnifiedBacktestConfig,
) -> UnifiedPortfolioResult:
    """Run all strategies stepping through time with shared portfolio constraints."""

    # When fixed_qty is used, heat enforcement is disabled because position
    # sizes don't scale with equity — the R-based thresholds become meaningless.
    skip_heat = config.fixed_qty is not None

    atrss_bt = config.build_atrss_config()
    helix_bt = config.build_helix_config()
    breakout_bt = config.build_breakout_config()

    # --- Instantiate engines ---
    atrss_engines: dict[str, BacktestEngine] = {}
    for sym in config.atrss_symbols:
        cfg = ATRSS_SYMBOL_CONFIGS.get(sym)
        if cfg is None or sym not in data.atrss_hourly:
            continue
        mult = config.symbol_risk_multipliers.get(f"ATRSS:{sym}", 1.0)
        if mult != 1.0:
            cfg = _dc_replace(cfg, base_risk_pct=cfg.base_risk_pct * mult)
        atrss_engines[sym] = BacktestEngine(
            symbol=sym, cfg=cfg, bt_config=atrss_bt,
            point_value=cfg.multiplier,
        )

    helix_engines: dict[str, HelixEngine] = {}
    for sym in config.helix_symbols:
        cfg = HELIX_SYMBOL_CONFIGS.get(sym)
        if cfg is None or sym not in data.hourly:
            continue
        mult = config.symbol_risk_multipliers.get(f"AKC_HELIX:{sym}", 1.0)
        if mult != 1.0:
            cfg = _dc_replace(cfg, base_risk_pct=cfg.base_risk_pct * mult)
        eng = HelixEngine(
            symbol=sym, cfg=cfg, bt_config=helix_bt,
            point_value=cfg.multiplier,
        )
        # Precompute indicator arrays once (avoids per-bar recomputation)
        eng._precompute_indicators(data.hourly[sym], data.four_hour[sym])
        helix_engines[sym] = eng

    # Breakout engines: run isolated in fixed_qty mode (matches run_breakout_independent),
    # share state in risk-based mode (matches run_breakout_synchronized / production).
    shared_breakout_positions: dict[str, BreakoutPositionState] | None = None if skip_heat else {}
    shared_breakout_cb: BreakoutCBState | None = None if skip_heat else BreakoutCBState()
    breakout_engines: dict[str, BreakoutEngine] = {}
    for sym in config.breakout_symbols:
        cfg = BREAKOUT_SYMBOL_CONFIGS.get(sym) or BREAKOUT_ETF_CONFIGS.get(sym)
        if cfg is None or sym not in data.breakout_hourly:
            continue
        mult = config.symbol_risk_multipliers.get(f"SWING_BREAKOUT_V3:{sym}", 1.0)
        if mult != 1.0:
            cfg = _dc_replace(cfg, base_risk_pct=cfg.base_risk_pct * mult)
        kw: dict = dict(symbol=sym, cfg=cfg, bt_config=breakout_bt,
                        point_value=cfg.multiplier)
        if shared_breakout_positions is not None:
            kw["external_positions"] = shared_breakout_positions
            kw["external_circuit_breaker"] = shared_breakout_cb
        eng = BreakoutEngine(**kw)
        # Precompute indicator arrays once (avoids per-bar recomputation)
        eng._precompute_indicators(data.daily[sym], data.breakout_hourly[sym], data.breakout_four_hour[sym])
        # Initialize rolling histories and slot medians (matching BreakoutEngine.run())
        eng._init_histories(data.daily[sym], config.warmup_daily)
        eng._init_slot_medians(data.breakout_hourly[sym], config.warmup_hourly)
        breakout_engines[sym] = eng

    # --- S5 engines ---
    s5_pb_bt = config.build_s5_pb_config()
    s5_dual_bt = config.build_s5_dual_config()

    s5_pb_engines: dict[str, S5Engine] = {}
    for sym in config.s5_pb_symbols:
        s5cfg = S5_SYMBOL_CONFIGS.get(sym)
        if s5cfg is None or sym not in data.daily:
            continue
        eng = S5Engine(
            symbol=sym, cfg=s5cfg, bt_config=s5_pb_bt,
            point_value=s5cfg.multiplier,
        )
        eng._precompute_indicators(data.daily[sym])
        s5_pb_engines[sym] = eng

    s5_dual_engines: dict[str, S5Engine] = {}
    for sym in config.s5_dual_symbols:
        s5cfg = S5_SYMBOL_CONFIGS.get(sym)
        if s5cfg is None or sym not in data.daily:
            continue
        eng = S5Engine(
            symbol=sym, cfg=s5cfg, bt_config=s5_dual_bt,
            point_value=s5cfg.multiplier,
        )
        eng._precompute_indicators(data.daily[sym])
        s5_dual_engines[sym] = eng

    if not atrss_engines and not helix_engines and not breakout_engines:
        logger.warning("No engines created — check symbols and data")
        return UnifiedPortfolioResult()

    # --- Build unified timestamp index ---
    # ATRSS uses RTH-filtered hourly; Helix/Breakout use full hourly
    time_sets: dict[str, dict] = {}
    all_times_set: set = set()
    all_engines_flat: dict[str, object] = {}

    for sym, eng in atrss_engines.items():
        key = f"atrss_{sym}"
        all_engines_flat[key] = eng
        times = data.atrss_hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            t = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[t] = i
        time_sets[key] = mapping
        all_times_set.update(mapping.keys())

    for sym, eng in helix_engines.items():
        key = f"helix_{sym}"
        all_engines_flat[key] = eng
        times = data.hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            t = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[t] = i
        time_sets[key] = mapping
        all_times_set.update(mapping.keys())

    for sym, eng in breakout_engines.items():
        key = f"breakout_{sym}"
        all_engines_flat[key] = eng
        times = data.breakout_hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            t = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[t] = i
        time_sets[key] = mapping
        all_times_set.update(mapping.keys())

    unified_ts = sorted(all_times_set)
    logger.info("Unified timeline: %d bars", len(unified_ts))

    # --- Pre-cache _to_datetime conversions for engine hot-paths ---
    # The Helix and Breakout engines convert numpy datetime64 → Python
    # datetime thousands of times per bar (lookback-window scans).
    # Pre-converting all unique timestamps into a cache and monkey-patching
    # the engines' _to_datetime eliminates redundant pd.Timestamp() calls.
    from datetime import timezone as _tz
    _dt_cache: dict = {}
    def _fast_to_datetime(ts, _c=_dt_cache) -> datetime:
        r = _c.get(ts)
        if r is not None:
            return r
        if isinstance(ts, datetime):
            r = ts if ts.tzinfo else ts.replace(tzinfo=_tz.utc)
        else:
            r = pd.Timestamp(ts).to_pydatetime()
        _c[ts] = r
        return r
    # Pre-populate from all timestamp arrays
    for _bars in (*data.hourly.values(), *data.atrss_hourly.values(),
                  *data.breakout_hourly.values(), *data.four_hour.values(),
                  *data.breakout_four_hour.values(), *data.daily.values()):
        for _t in _bars.times:
            _fast_to_datetime(_t)
    # Patch engine static methods to use the cache
    HelixEngine._to_datetime = staticmethod(_fast_to_datetime)
    BreakoutEngine._to_datetime = staticmethod(_fast_to_datetime)

    # --- Initialize trackers ---
    init_eq = config.initial_equity
    portfolio_equity = init_eq

    heat_tracker = PortfolioHeatTracker(
        heat_cap_R=config.heat_cap_R,
        portfolio_daily_stop_R=config.portfolio_daily_stop_R,
        strategy_slots=[config.atrss, config.helix, config.breakout,
                        config.s5_pb, config.s5_dual],
        equity=init_eq,
        simulate_live_r_normalization=config.simulate_live_r_normalization,
    )
    coordinator = BacktestCoordinator(
        enable_tighten=config.enable_atrss_helix_tighten,
        enable_size_boost=config.enable_atrss_helix_size_boost,
    )

    # --- Overlay initialization ---
    overlay_shares: dict[str, float] = {}
    overlay_prev_value: float = 0.0
    overlay_cumulative_pnl: float = 0.0
    overlay_per_sym_pnl: dict[str, float] = {}
    overlay_prev_sym_value: dict[str, float] = {}
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    overlay_rsi: dict[str, np.ndarray] = {}
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    overlay_in_position: dict[str, bool] = {}
    # Always build daily_date_idx (needed for S5 and optionally for overlay)
    s5_all_syms = sorted(set(config.s5_pb_symbols + config.s5_dual_symbols))
    daily_date_idx_syms = list(set(
        (config.overlay_symbols if config.overlay_enabled else []) + s5_all_syms
    ))
    daily_date_idx: dict[str, dict[date, int]] = _build_daily_date_index(data.daily, daily_date_idx_syms)
    if config.overlay_enabled:
        overlay_emas = _precompute_overlay_emas(data.daily, config)
        if config.overlay_mode == "multi":
            overlay_rsi = _precompute_overlay_rsi(data.daily, config)
            overlay_macd = _precompute_overlay_macd(data.daily, config)
        for sym in config.overlay_symbols:
            overlay_shares[sym] = 0.0
            overlay_per_sym_pnl[sym] = 0.0
            overlay_prev_sym_value[sym] = 0.0
            overlay_in_position[sym] = False

    # Per-engine previous equity for delta tracking
    # ATRSS and Helix: engine.equity starts at initial_equity, modified only by fills
    # Breakout: engine.equity is NOT set to portfolio_equity (avoid double-counting)
    prev_equity: dict[str, float] = {}
    for label, engines in [("atrss", atrss_engines), ("helix", helix_engines), ("breakout", breakout_engines),
                           ("s5_pb", s5_pb_engines), ("s5_dual", s5_dual_engines)]:
        for sym in engines:
            prev_equity[f"{label}_{sym}"] = init_eq

    # Track previous trade counts to detect actual trade closes
    prev_trade_counts: dict[str, int] = {}
    for label, engines in [("atrss", atrss_engines), ("helix", helix_engines), ("breakout", breakout_engines),
                           ("s5_pb", s5_pb_engines), ("s5_dual", s5_dual_engines)]:
        for sym in engines:
            prev_trade_counts[f"{label}_{sym}"] = 0

    equity_curve: list[float] = []
    timestamps: list = []
    heat_samples: list[float] = []
    prev_date: date | None = None
    blocked_entries: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "SWING_BREAKOUT_V3": 0, "S5_PB": 0, "S5_DUAL": 0}
    heat_rejection_log: list[dict] = []
    coordination_event_log: list[dict] = []
    daily_stop_activations = 0

    warmup_d = config.warmup_daily
    warmup_h = config.warmup_hourly
    warmup_4h = config.warmup_4h

    # Track ATRSS position state for coordination
    atrss_prev_positions: dict[str, bool] = {sym: False for sym in atrss_engines}

    # --- Main bar loop ---
    # Apply ablation patches for ATRSS module-level constants (matches run_independent)
    _patch = _AblationPatch(atrss_bt.flags, atrss_bt.param_overrides)
    _patch.__enter__()

    for ts in unified_ts:

        # Detect daily boundary
        ts_dt = pd.Timestamp(ts).to_pydatetime() if not isinstance(ts, datetime) else ts
        current_date = ts_dt.date() if hasattr(ts_dt, 'date') else pd.Timestamp(ts).date()
        if prev_date is not None and current_date != prev_date:
            heat_tracker.on_new_day()

            # --- Overlay: MTM delta from previous day, then rebalance ---
            # Overlay PnL is tracked separately so active engines don't see it
            # in their sizing equity.  Use overlay_last_daily (most recent
            # trading day with a daily bar) for MTM/rebalance.  Skip weekends
            # and holidays where the hourly data has bars (e.g. IBIT crypto)
            # but daily ETF data does not.
            if config.overlay_enabled and prev_date in daily_date_idx.get(config.overlay_symbols[0], {}):
                new_value = _compute_overlay_value(
                    overlay_shares, data.daily, daily_date_idx, prev_date,
                )
                delta = new_value - overlay_prev_value
                overlay_cumulative_pnl += delta

                # Per-symbol overlay PnL tracking
                for _osym in config.overlay_symbols:
                    _sh = overlay_shares.get(_osym, 0.0)
                    _sv = 0.0
                    if _sh != 0.0:
                        _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                        if _sidx is not None:
                            _sv = _sh * data.daily[_osym].closes[_sidx]
                    _sdelta = _sv - overlay_prev_sym_value.get(_osym, 0.0)
                    overlay_per_sym_pnl[_osym] = overlay_per_sym_pnl.get(_osym, 0.0) + _sdelta

                total_equity = portfolio_equity + overlay_cumulative_pnl
                _rebalance_overlay(
                    config, data.daily, overlay_emas, daily_date_idx,
                    prev_date, total_equity, overlay_shares,
                    overlay_rsi=overlay_rsi or None,
                    overlay_macd=overlay_macd or None,
                    overlay_in_position=overlay_in_position,
                )
                overlay_prev_value = _compute_overlay_value(
                    overlay_shares, data.daily, daily_date_idx, prev_date,
                )
                # Update per-symbol previous values after rebalance
                for _osym in config.overlay_symbols:
                    _sh = overlay_shares.get(_osym, 0.0)
                    _sv = 0.0
                    if _sh != 0.0:
                        _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                        if _sidx is not None:
                            _sv = _sh * data.daily[_osym].closes[_sidx]
                    overlay_prev_sym_value[_osym] = _sv

            # === S5 daily-bar engines (step on daily boundary) ===
            for strat_id, s5_engines_dict in [("S5_PB", s5_pb_engines), ("S5_DUAL", s5_dual_engines)]:
                for sym, engine in s5_engines_dict.items():
                    d_idx = daily_date_idx.get(sym, {}).get(prev_date)
                    if d_idx is None:
                        continue
                    engine.sizing_equity = portfolio_equity
                    had_position = engine.active_position is not None
                    try:
                        engine._step_bar(data.daily[sym], d_idx, warmup_d)
                    except (OverflowError, ValueError):
                        continue
                    has_position = engine.active_position is not None
                    if has_position and not had_position:
                        pos = engine.active_position
                        if skip_heat:
                            ok, reason = True, ""
                        else:
                            risk_dollars = abs(pos.fill_price - pos.current_stop) * engine.point_value * pos.qty
                            ok, reason = heat_tracker.can_enter(strat_id, risk_dollars)
                        if not ok:
                            blocked_entries[strat_id] += 1
                            engine.force_flatten()
                            if "daily stop" in reason.lower():
                                daily_stop_activations += 1

        prev_date = current_date

        # === Step 1: ATRSS engines (priority 0) ===
        for sym, engine in atrss_engines.items():
            key = f"atrss_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            engine.sizing_equity = portfolio_equity
            try:
                candidates = engine.step_bar(
                    data.daily[sym], data.atrss_hourly[sym],
                    data.atrss_daily_idx_maps[sym], bar_idx,
                    warmup_d, warmup_h,
                )
            except (OverflowError, ValueError):
                # Edge case: trigger ≈ stop → infinite qty; skip this bar
                candidates = []

            # Heat-gated candidate submission
            if candidates:
                bar_time = _to_datetime(ts)
                for cand in candidates:
                    if skip_heat:
                        engine.submit_candidate(cand, bar_time)
                    else:
                        risk_dollars = abs(cand.trigger_price - cand.initial_stop) * engine.point_value * cand.qty
                        ok, reason = heat_tracker.can_enter("ATRSS", risk_dollars)
                        if ok:
                            engine.submit_candidate(cand, bar_time)
                        else:
                            blocked_entries["ATRSS"] += 1
                            heat_rejection_log.append({"time": bar_time, "strategy": "ATRSS", "reason": reason})
                            if "daily stop" in reason.lower():
                                daily_stop_activations += 1

            # Track ATRSS position changes for coordination
            has_pos = engine.position.direction != AtrssDirection.FLAT
            had_pos = atrss_prev_positions.get(sym, False)
            if has_pos and not had_pos:
                entry_price = engine.position.base_leg.entry_price if engine.position.base_leg else 0.0
                coordinator.on_atrss_position_change(sym, engine.position.direction, entry_price)
            elif not has_pos and had_pos:
                coordinator.on_atrss_position_change(sym, 0, 0.0)
            atrss_prev_positions[sym] = has_pos

        # === Step 2: Coordination Rule 1 — tighten Helix stops ===
        for tighten_sym in coordinator.consume_tighten_events():
            if tighten_sym in helix_engines:
                h_eng = helix_engines[tighten_sym]
                pos = h_eng.active_position
                if pos is not None and pos.qty_open > 0:
                    # Tighten to breakeven (fill_price)
                    be_price = pos.fill_price
                    if pos.setup.direction == HelixDirection.LONG:
                        if be_price > pos.current_stop:
                            pos.current_stop = be_price
                            coordinator.tighten_count += 1
                            coordination_event_log.append({"time": _to_datetime(ts), "type": "tighten", "trigger_strategy": "ATRSS", "target_strategy": "AKC_HELIX", "symbol": tighten_sym})
                    elif pos.setup.direction == HelixDirection.SHORT:
                        if be_price < pos.current_stop:
                            pos.current_stop = be_price
                            coordinator.tighten_count += 1
                            coordination_event_log.append({"time": _to_datetime(ts), "type": "tighten", "trigger_strategy": "ATRSS", "target_strategy": "AKC_HELIX", "symbol": tighten_sym})

        # === Step 3: Helix engines (priority 1) ===
        for sym, engine in helix_engines.items():
            key = f"helix_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            engine.sizing_equity = portfolio_equity

            # Snapshot position state before step
            had_position = engine.active_position is not None and engine.active_position.qty_open > 0

            try:
                engine._step_bar(
                    data.daily[sym], data.hourly[sym], data.four_hour[sym],
                    data.daily_idx_maps[sym], data.four_hour_idx_maps[sym],
                    bar_idx, warmup_d, warmup_h,
                )
            except (OverflowError, ValueError):
                continue

            # Detect new entry
            has_position = engine.active_position is not None and engine.active_position.qty_open > 0
            if has_position and not had_position:
                pos = engine.active_position
                if skip_heat:
                    ok, reason = True, ""
                else:
                    risk_dollars = abs(pos.fill_price - pos.current_stop) * engine.point_value * pos.qty_open
                    ok, reason = heat_tracker.can_enter("AKC_HELIX", risk_dollars)
                if not ok:
                    # Force flatten: reverse the entry
                    blocked_entries["AKC_HELIX"] += 1
                    heat_rejection_log.append({"time": _to_datetime(ts), "strategy": "AKC_HELIX", "reason": reason})
                    _force_flatten_helix(engine, pos)
                    if "daily stop" in reason.lower():
                        daily_stop_activations += 1
                else:
                    # Check coordination Rule 2: size boost
                    if pos.setup and hasattr(pos.setup, 'direction'):
                        direction = pos.setup.direction
                        if coordinator.has_atrss_position(sym, direction):
                            coordinator.boost_count += 1
                            coordination_event_log.append({"time": _to_datetime(ts), "type": "boost", "trigger_strategy": "ATRSS", "target_strategy": "AKC_HELIX", "symbol": sym})

        # === Step 4: Breakout engines (priority 2) ===
        for sym, engine in breakout_engines.items():
            key = f"breakout_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            # Breakout uses self.equity for both sizing and PnL tracking
            # (no separate sizing_equity like ATRSS/Helix).  In risk-based
            # mode we inject portfolio equity for sizing, then extract the
            # fill-caused PnL delta and restore independent tracking.
            if not skip_heat:
                saved_equity = engine.equity
                engine.equity = portfolio_equity

            had_position = engine.active_position is not None and engine.active_position.qty_open > 0

            try:
                engine._step_bar(
                    data.daily[sym], data.breakout_hourly[sym],
                    data.breakout_four_hour[sym],
                    data.breakout_daily_idx_maps[sym],
                    data.breakout_four_hour_idx_maps[sym],
                    bar_idx, warmup_d, warmup_h, warmup_4h,
                )
            except (OverflowError, ValueError):
                if not skip_heat:
                    engine.equity = saved_equity
                continue

            # Restore independent equity: saved + fill PnL from this bar
            if not skip_heat:
                fill_pnl = engine.equity - portfolio_equity
                engine.equity = saved_equity + fill_pnl

            has_position = engine.active_position is not None and engine.active_position.qty_open > 0
            if has_position and not had_position:
                pos = engine.active_position
                if skip_heat:
                    ok, reason = True, ""
                else:
                    risk_dollars = abs(pos.fill_price - pos.current_stop) * engine.point_value * pos.qty_open
                    ok, reason = heat_tracker.can_enter("SWING_BREAKOUT_V3", risk_dollars)
                if not ok:
                    blocked_entries["SWING_BREAKOUT_V3"] += 1
                    heat_rejection_log.append({"time": _to_datetime(ts), "strategy": "SWING_BREAKOUT_V3", "reason": reason})
                    _force_flatten_breakout(engine, pos)
                    if "daily stop" in reason.lower():
                        daily_stop_activations += 1

        # === Step 5: Update portfolio equity from per-engine PnL deltas ===
        # Each engine's equity = initial_equity + cumulative realized PnL
        # Portfolio equity = initial_equity + sum of all engines' realized PnL
        for label, engines, strat_id in [
            ("atrss", atrss_engines, "ATRSS"),
            ("helix", helix_engines, "AKC_HELIX"),
            ("breakout", breakout_engines, "SWING_BREAKOUT_V3"),
            ("s5_pb", s5_pb_engines, "S5_PB"),
            ("s5_dual", s5_dual_engines, "S5_DUAL"),
        ]:
            for sym, eng in engines.items():
                key = f"{label}_{sym}"
                delta = eng.equity - prev_equity[key]
                portfolio_equity += delta
                prev_equity[key] = eng.equity

                # Record realized PnL only when a trade actually closes
                n_trades = len(eng.trades)
                if n_trades > prev_trade_counts[key]:
                    # New trade(s) closed — record the PnL from the most recent trade
                    for t_idx in range(prev_trade_counts[key], n_trades):
                        trade_pnl = eng.trades[t_idx].pnl_dollars
                        heat_tracker.record_trade_close(strat_id, trade_pnl)
                    prev_trade_counts[key] = n_trades

        # === Step 6: Update heat tracker ===
        risk_by_strat = _compute_open_risk(atrss_engines, helix_engines, breakout_engines,
                                           s5_pb_engines, s5_dual_engines)
        heat_tracker.update_open_risk(risk_by_strat)

        equity_curve.append(portfolio_equity + overlay_cumulative_pnl)
        timestamps.append(ts)

        # Track heat utilization (normalized to ATRSS R-units for consistency)
        total_risk = sum(risk_by_strat.values())
        # Use ATRSS unit_risk_dollars as the normalization base (1R = 1% of equity)
        norm_base = config.initial_equity * config.atrss.unit_risk_pct
        heat_R = total_risk / norm_base if norm_base > 0 else 0.0
        heat_samples.append(heat_R)

    _patch.__exit__(None, None, None)

    # --- Flatten open S5 positions at end of data ---
    for strat_id, s5_dict in [("S5_PB", s5_pb_engines), ("S5_DUAL", s5_dual_engines)]:
        for sym, engine in s5_dict.items():
            if engine.active_position is not None:
                bars = data.daily[sym]
                engine._flatten_position(
                    engine.active_position,
                    bars.closes[-1],
                    engine._to_datetime(bars.times[-1]),
                    "END_OF_DATA",
                )

    # --- Final overlay settlement: MTM the last day's positions ---
    if config.overlay_enabled and prev_date is not None:
        # Find last valid daily date for settlement
        ref_sym = config.overlay_symbols[0]
        if ref_sym in daily_date_idx and prev_date in daily_date_idx[ref_sym]:
            final_value = _compute_overlay_value(
                overlay_shares, data.daily, daily_date_idx, prev_date,
            )
            delta = final_value - overlay_prev_value
            overlay_cumulative_pnl += delta
            # Final per-symbol settlement
            for _osym in config.overlay_symbols:
                _sh = overlay_shares.get(_osym, 0.0)
                _sv = 0.0
                if _sh != 0.0:
                    _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                    if _sidx is not None:
                        _sv = _sh * data.daily[_osym].closes[_sidx]
                _sdelta = _sv - overlay_prev_sym_value.get(_osym, 0.0)
                overlay_per_sym_pnl[_osym] = overlay_per_sym_pnl.get(_osym, 0.0) + _sdelta
        # Update last equity curve point to include final overlay PnL
        if equity_curve:
            equity_curve[-1] = portfolio_equity + overlay_cumulative_pnl

    # --- Build results ---
    heat_arr = np.array(heat_samples) if heat_samples else np.array([0.0])
    heat_stats = UnifiedHeatStats(
        avg_heat_pct=float(np.mean(heat_arr)),
        max_heat_pct=float(np.max(heat_arr)),
        pct_time_at_cap=float(np.mean(heat_arr >= config.heat_cap_R)) * 100,
    )

    # Collect trades per strategy
    all_atrss_trades = []
    for eng in atrss_engines.values():
        all_atrss_trades.extend(eng.trades)

    all_helix_trades = []
    for eng in helix_engines.values():
        all_helix_trades.extend(eng.trades)

    all_breakout_trades = []
    for eng in breakout_engines.values():
        all_breakout_trades.extend(eng.trades)

    all_s5_pb_trades = []
    for eng in s5_pb_engines.values():
        all_s5_pb_trades.extend(eng.trades)

    all_s5_dual_trades = []
    for eng in s5_dual_engines.values():
        all_s5_dual_trades.extend(eng.trades)

    strategy_results = {}
    for sid, trades, blocked in [
        ("ATRSS", all_atrss_trades, blocked_entries["ATRSS"]),
        ("AKC_HELIX", all_helix_trades, blocked_entries["AKC_HELIX"]),
        ("SWING_BREAKOUT_V3", all_breakout_trades, blocked_entries["SWING_BREAKOUT_V3"]),
        ("S5_PB", all_s5_pb_trades, blocked_entries["S5_PB"]),
        ("S5_DUAL", all_s5_dual_trades, blocked_entries["S5_DUAL"]),
    ]:
        total_pnl = sum(t.pnl_dollars for t in trades)
        wins = sum(1 for t in trades if t.pnl_dollars > 0)
        losses = sum(1 for t in trades if t.pnl_dollars <= 0)
        total_r = sum(t.r_multiple for t in trades)
        strategy_results[sid] = StrategyResult(
            strategy_id=sid,
            total_trades=len(trades),
            total_pnl=total_pnl,
            winning_trades=wins,
            losing_trades=losses,
            total_r=total_r,
            entries_blocked_by_heat=blocked,
        )

    return UnifiedPortfolioResult(
        combined_equity=np.array(equity_curve),
        combined_timestamps=np.array(timestamps),
        heat_stats=heat_stats,
        strategy_results=strategy_results,
        coordination_tighten_count=coordinator.tighten_count,
        coordination_boost_count=coordinator.boost_count,
        portfolio_daily_stop_activations=daily_stop_activations,
        overlay_pnl=overlay_cumulative_pnl,
        overlay_per_symbol_pnl=dict(overlay_per_sym_pnl),
        atrss_trades=all_atrss_trades,
        helix_trades=all_helix_trades,
        breakout_trades=all_breakout_trades,
        s5_pb_trades=all_s5_pb_trades,
        s5_dual_trades=all_s5_dual_trades,
        heat_rejections=heat_rejection_log,
        coordination_events=coordination_event_log,
    )


# ---------------------------------------------------------------------------
# Force-flatten helpers (when heat check fails post-entry)
# ---------------------------------------------------------------------------

def _force_flatten_helix(engine: HelixEngine, pos) -> None:
    """Reverse a Helix entry that breached the heat cap."""
    engine.broker.cancel_all(engine.symbol)
    # Reverse entry commission leaked by the phantom fill
    entry_comm = getattr(pos, "commission", 0.0)
    if entry_comm > 0:
        engine.equity += entry_comm
        engine.total_commission -= entry_comm
    engine.setups_filled = max(0, engine.setups_filled - 1)
    engine.active_position = None


def _force_flatten_breakout(engine: BreakoutEngine, pos) -> None:
    """Reverse a Breakout entry that breached the heat cap."""
    engine.broker.cancel_all(engine.symbol)
    # Reverse entry commission leaked by the phantom fill
    entry_comm = getattr(pos, "commission", 0.0)
    if entry_comm > 0:
        engine.equity += entry_comm
        engine.total_commission -= entry_comm
    engine.entries_filled = max(0, engine.entries_filled - 1)
    # Reset campaign state back from POSITION_OPEN to BREAKOUT
    if hasattr(engine, "campaign") and engine.campaign is not None:
        from backtest.engine.breakout_engine import CampaignState
        if engine.campaign.state == CampaignState.POSITION_OPEN:
            engine.campaign.state = CampaignState.BREAKOUT
    engine.active_position = None


def _to_datetime(ts) -> datetime:
    """Convert numpy datetime64 or pandas Timestamp to UTC-aware datetime.

    Must match BacktestEngine._to_datetime which returns UTC-aware datetimes,
    otherwise SimBroker comparisons fail with naive/aware mismatch.
    """
    from datetime import timezone
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if hasattr(ts, 'astype'):
        # numpy datetime64
        unix_epoch = np.datetime64(0, 'ns')
        one_second = np.timedelta64(1, 's')
        seconds = (ts - unix_epoch) / one_second
        return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
    if hasattr(ts, 'to_pydatetime'):
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_unified_report(result: UnifiedPortfolioResult, config: UnifiedBacktestConfig) -> None:
    """Print a summary of the unified backtest results."""
    eq = result.combined_equity
    if len(eq) == 0:
        print("No results to report.")
        return

    init_eq = config.initial_equity
    final_eq = eq[-1]
    total_return = (final_eq - init_eq) / init_eq * 100

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = float(np.min(dd))
    max_dd_dollars = float(np.min(eq - peak))

    # Sharpe (annualized from hourly returns)
    returns = np.diff(eq) / eq[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 7)  # ~7 trading hours/day
    else:
        sharpe = 0.0

    total_trades = sum(sr.total_trades for sr in result.strategy_results.values())
    active_pnl = sum(sr.total_pnl for sr in result.strategy_results.values())
    total_pnl = active_pnl + result.overlay_pnl

    print("\n" + "=" * 70)
    print("UNIFIED 5-STRATEGY PORTFOLIO BACKTEST RESULTS")
    print("=" * 70)
    print(f"Initial Equity:  ${init_eq:,.2f}")
    print(f"Final Equity:    ${final_eq:,.2f}")
    print(f"Total Return:    {total_return:+.2f}%")
    print(f"Total PnL:       ${total_pnl:+,.2f}")
    print(f"Max Drawdown:    {max_dd:.2f}% (${max_dd_dollars:,.2f})")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Total Trades:    {total_trades}")
    print(f"Timeline:        {len(eq):,} bars")

    print(f"\n{'Heat Statistics':}")
    print(f"  Avg Heat (R):        {result.heat_stats.avg_heat_pct:.3f}")
    print(f"  Max Heat (R):        {result.heat_stats.max_heat_pct:.3f}")
    print(f"  % Time at Cap:       {result.heat_stats.pct_time_at_cap:.1f}%")

    print(f"\n{'Cross-Strategy Coordination':}")
    print(f"  Helix Stop Tightens: {result.coordination_tighten_count}")
    print(f"  Helix Size Boosts:   {result.coordination_boost_count}")
    print(f"  Portfolio Daily Stops: {result.portfolio_daily_stop_activations}")

    if config.overlay_enabled:
        ovl_pct = result.overlay_pnl / init_eq * 100 if init_eq > 0 else 0
        act_pct = active_pnl / init_eq * 100 if init_eq > 0 else 0
        if config.overlay_mode == "multi":
            mode_label = f"Multi (EMA+RSI+MACD) entry>={config.overlay_entry_score_min} exit<={config.overlay_exit_score_max}"
            print(f"\n{'Idle-Capital Overlay (' + mode_label + ')':}")
            print(f"  Symbols:             {', '.join(config.overlay_symbols)}")
            print(f"  Adaptive Sizing:     {'ON' if config.overlay_adaptive_sizing else 'OFF'} [{config.overlay_min_alloc_pct:.0%}-{config.overlay_max_alloc_pct:.0%}]")
        else:
            print(f"\n{'Idle-Capital Overlay (EMA {0}/{1})'.format(config.overlay_ema_fast, config.overlay_ema_slow):}")
        print(f"  Overlay PnL:         ${result.overlay_pnl:+,.2f} ({ovl_pct:+.1f}%)")
        print(f"  Active Strategy PnL: ${active_pnl:+,.2f} ({act_pct:+.1f}%)")
        print(f"  Overlay Max Pct:     {config.overlay_max_pct:.0%}")
        if result.overlay_per_symbol_pnl:
            for osym, opnl in sorted(result.overlay_per_symbol_pnl.items()):
                opnl_pct = opnl / init_eq * 100 if init_eq > 0 else 0
                print(f"    {osym:<6} overlay PnL: ${opnl:+,.2f} ({opnl_pct:+.1f}%)")

    print(f"\n{'Per-Strategy Breakdown':}")
    print(f"{'Strategy':<22} {'Trades':>7} {'Win%':>6} {'PnL':>10} {'Total R':>8} {'Blocked':>8}")
    print("-" * 65)
    for sid in ["ATRSS", "AKC_HELIX", "SWING_BREAKOUT_V3", "S5_PB", "S5_DUAL"]:
        sr = result.strategy_results.get(sid)
        if sr is None:
            continue
        win_pct = sr.winning_trades / sr.total_trades * 100 if sr.total_trades > 0 else 0
        print(f"{sid:<22} {sr.total_trades:>7} {win_pct:>5.1f}% ${sr.total_pnl:>9,.2f} {sr.total_r:>7.1f}R {sr.entries_blocked_by_heat:>8}")

    print("=" * 70)
