"""Unified swing portfolio backtesting engine.

Runs ATRSS, AKC-Helix, BRS R9, Swing Breakout v3.3, and the idle-capital
overlay under one shared replay with portfolio-level heat caps,
priority-aware reservation, per-strategy ceilings, and cross-strategy
coordination.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, date

import numpy as np
import pandas as pd

from strategies.swing.atrss.config import SYMBOL_CONFIGS as ATRSS_SYMBOL_CONFIGS
from strategies.swing.atrss.models import Direction as AtrssDirection

from strategies.swing.akc_helix.config import SYMBOL_CONFIGS as HELIX_SYMBOL_CONFIGS
from strategies.swing.akc_helix.models import Direction as HelixDirection

from strategies.swing.breakout.config import SYMBOL_CONFIGS as BREAKOUT_SYMBOL_CONFIGS
from strategies.swing.breakout.config import _ETF_CONFIGS as BREAKOUT_ETF_CONFIGS
from strategies.swing.breakout.models import (
    CircuitBreakerState as BreakoutCBState,
    Direction as BreakoutDirection,
    PositionState as BreakoutPositionState,
)
from strategies.swing.brs.models import BRSRegime
from strategies.swing.overlay.shared import allocate_weighted_targets, compute_ema

from libs.oms.risk.swing_portfolio_adapter import SwingPortfolioHeatAdapter as PortfolioHeatTracker

from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    build_numpy_arrays,
    filter_rth,
    normalize_timezone,
    resample_1h_to_4h,
)
from backtests.swing.engine.backtest_engine import BacktestEngine, _AblationPatch
from backtests.swing.engine.brs_engine import BRSEngine
from backtests.swing.engine.helix_engine import HelixEngine
from backtests.swing.engine.breakout_engine import BreakoutEngine

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
    # RTH-filtered hourly data for BRS (UTC, RTH only)
    brs_daily: dict[str, NumpyBars] = field(default_factory=dict)
    brs_hourly: dict[str, NumpyBars] = field(default_factory=dict)
    brs_four_hour: dict[str, NumpyBars] = field(default_factory=dict)
    brs_daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    brs_four_hour_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
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
    brs_set = set(config.brs_symbols)
    breakout_set = set(config.breakout_symbols)
    overlay_syms = config.overlay_symbols if config.overlay_enabled else []
    all_symbols = sorted(set(
        config.atrss_symbols + config.helix_symbols + config.brs_symbols + config.breakout_symbols
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

        # --- BRS: normalize_timezone(UTC) + filter_rth + 4H from RTH-filtered ---
        if sym in brs_set:
            brs_h = normalize_timezone(hourly_df.copy())
            brs_d = normalize_timezone(daily_df.copy())
            brs_h = filter_rth(brs_h)
            brs_4h = resample_1h_to_4h(brs_h)
            portfolio.brs_daily[sym] = build_numpy_arrays(brs_d)
            portfolio.brs_hourly[sym] = build_numpy_arrays(brs_h)
            portfolio.brs_four_hour[sym] = build_numpy_arrays(brs_4h)
            portfolio.brs_daily_idx_maps[sym] = align_daily_to_hourly(brs_h, brs_d)
            portfolio.brs_four_hour_idx_maps[sym] = align_4h_to_hourly(brs_h, brs_4h)

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
    entry_signals_fired: int = 0
    entries_accepted_by_portfolio: int = 0
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
    combined_equity_mtm: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_equity_realized: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_stats: UnifiedHeatStats = field(default_factory=UnifiedHeatStats)
    strategy_results: dict[str, StrategyResult] = field(default_factory=dict)
    coordination_tighten_count: int = 0
    coordination_boost_count: int = 0
    portfolio_daily_stop_activations: int = 0
    overlay_pnl: float = 0.0
    overlay_commission: float = 0.0
    overlay_per_symbol_pnl: dict[str, float] = field(default_factory=dict)
    # Raw per-engine results for detailed analysis
    atrss_trades: list = field(default_factory=list)
    helix_trades: list = field(default_factory=list)
    brs_trades: list = field(default_factory=list)
    breakout_trades: list = field(default_factory=list)
    # Diagnostic event logs for portfolio diagnostics
    entry_events: list = field(default_factory=list)
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
    brs_engines: dict[str, BRSEngine],
    breakout_engines: dict[str, BreakoutEngine],
) -> dict[str, float]:
    """Returns {strategy_id: total_risk_dollars}."""
    risk: dict[str, float] = {
        "ATRSS": 0.0, "AKC_HELIX": 0.0, "BRS_R9": 0.0, "SWING_BREAKOUT_V3": 0.0,
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

    for eng in brs_engines.values():
        risk["BRS_R9"] += eng.reserved_risk_dollars()

    for sym, eng in breakout_engines.items():
        pos = eng.active_position
        if pos is not None and pos.qty_open > 0:
            r = abs(pos.fill_price - pos.current_stop) * eng.point_value * pos.qty_open
            risk["SWING_BREAKOUT_V3"] += r

    return risk


def _latest_engine_mark(engine, default_equity: float) -> float:
    """Return the latest per-engine MTM equity mark when available."""
    eq_arr = getattr(engine, "_eq_arr", None)
    bar_idx = int(getattr(engine, "_bar_idx", 0) or 0)
    if eq_arr is not None and bar_idx > 0:
        try:
            value = float(eq_arr[bar_idx - 1])
            if np.isfinite(value):
                return value
        except (IndexError, TypeError, ValueError):
            pass

    curve = getattr(engine, "equity_curve", None)
    if curve is not None and len(curve) > 0:
        try:
            value = float(curve[-1])
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass

    value = float(getattr(engine, "equity", default_equity) or default_equity)
    return value if np.isfinite(value) else float(default_equity)


def _combined_child_mtm_equity(
    initial_equity: float,
    engine_groups: list[tuple[str, dict[str, object]]],
    overlay_pnl: float,
) -> float:
    combined = float(initial_equity) + float(overlay_pnl)
    for label, engines in engine_groups:
        basis = 0.0 if label == "brs" else float(initial_equity)
        for engine in engines.values():
            combined += _latest_engine_mark(engine, basis) - basis
    return combined


def _overlay_transaction_cost(shares_delta: float, raw_price: float, slippage) -> float:
    shares = abs(float(shares_delta))
    price = float(raw_price)
    if shares <= 0 or price <= 0:
        return 0.0
    per_share_commission = shares * float(getattr(slippage, "commission_per_share_etf", 0.0) or 0.0)
    min_commission = float(getattr(slippage, "commission_min_etf_order", 0.0) or 0.0)
    commission = max(per_share_commission, min_commission) if min_commission > 0 else per_share_commission
    slip = price * float(getattr(slippage, "overlay_slip_bps", 0.0) or 0.0) / 10_000
    return commission + slip * shares


# ---------------------------------------------------------------------------
# Idle-capital overlay helpers
# ---------------------------------------------------------------------------

def _overlay_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed (matches idle_capital_study.py)."""
    return compute_ema(series, period)


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
    bear_regime_active: bool = False,
) -> None:
    """Rebalance overlay positions at day boundary.

    Legacy "ema" mode: equal/weighted across bullish symbols (EMA fast > slow).
    "multi" mode: EMA+RSI+MACD composite scoring with hysteresis and adaptive sizing.
    """
    if config.overlay_mode == "multi" and overlay_rsi is not None and overlay_macd is not None:
        _rebalance_overlay_multi(
            config, daily, overlay_emas, daily_date_idx, current_date,
            portfolio_equity, overlay_shares, overlay_rsi, overlay_macd,
            overlay_in_position or {}, bear_regime_active=bear_regime_active,
        )
        return

    # --- Legacy "ema" mode (shared decision/allocation semantics) ---
    signals: dict[str, bool] = {}

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

    execution_prices: dict[str, float] = {}
    for sym in config.overlay_symbols:
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            continue
        d_idx = sym_idx.get(current_date)
        if d_idx is None:
            continue
        if d_idx + 1 < len(daily[sym].opens):
            execution_prices[sym] = float(daily[sym].opens[d_idx + 1])
        else:
            execution_prices[sym] = float(daily[sym].closes[d_idx])

    target_shares = allocate_weighted_targets(
        config.overlay_symbols,
        signals=signals,
        prices=execution_prices,
        portfolio_equity=portfolio_equity,
        max_equity_pct=config.overlay_max_pct,
        weights=config.overlay_weights,
    )
    if bear_regime_active:
        for sym in config.overlay_symbols:
            if overlay_shares.get(sym, 0.0) == 0.0 and target_shares.get(sym, 0) > 0:
                target_shares[sym] = 0
    for sym in config.overlay_symbols:
        overlay_shares[sym] = float(target_shares.get(sym, 0))


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
    bear_regime_active: bool = False,
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
                if bear_regime_active and overlay_shares.get(sym, 0.0) == 0.0:
                    overlay_in_position[sym] = False
                    continue
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
        # Execute at next day's open (signal at close, fill at next open)
        if d_idx + 1 < len(daily[sym].opens):
            price = daily[sym].opens[d_idx + 1]
        else:
            price = daily[sym].closes[d_idx]  # last day fallback
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
    brs_bt = config.build_brs_config()
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

    brs_engines: dict[str, BRSEngine] = {}
    for sym in _brs_run_order(config.brs_symbols):
        if sym not in data.brs_hourly:
            continue
        sym_cfg = brs_bt.get_symbol_config(sym)
        mult = config.symbol_risk_multipliers.get(f"BRS_R9:{sym}", 1.0)
        if mult != 1.0:
            sym_cfg = _dc_replace(sym_cfg, base_risk_pct=sym_cfg.base_risk_pct * mult)
        eng = BRSEngine(
            symbol=sym,
            sym_cfg=sym_cfg,
            cfg=brs_bt,
            starting_equity=0.0,
        )
        if sym == "GLD" and "QQQ" in brs_engines:
            eng._qqq_regimes = brs_engines["QQQ"]._regime_history
        eng._prepare_run_context(
            daily=data.brs_daily[sym],
            hourly=data.brs_hourly[sym],
            four_hour=data.brs_four_hour[sym],
            daily_idx_map=data.brs_daily_idx_maps[sym],
            four_hour_idx_map=data.brs_four_hour_idx_maps[sym],
        )
        brs_engines[sym] = eng

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
        eng._init_equity_arrays(len(data.breakout_hourly[sym]), data.breakout_hourly[sym].times.dtype)
        # Precompute indicator arrays once (avoids per-bar recomputation)
        eng._precompute_indicators(data.daily[sym], data.breakout_hourly[sym], data.breakout_four_hour[sym])
        # Initialize rolling histories and slot medians (matching BreakoutEngine.run())
        eng._init_histories(data.daily[sym], config.warmup_daily)
        eng._init_slot_medians(data.breakout_hourly[sym], config.warmup_hourly)
        breakout_engines[sym] = eng

    if not atrss_engines and not helix_engines and not brs_engines and not breakout_engines:
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

    for sym, eng in brs_engines.items():
        key = f"brs_{sym}"
        all_engines_flat[key] = eng
        times = data.brs_hourly[sym].times
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
                  *data.brs_hourly.values(), *data.brs_four_hour.values(),
                  *data.breakout_hourly.values(), *data.four_hour.values(),
                  *data.breakout_four_hour.values(), *data.brs_daily.values(),
                  *data.daily.values()):
        for _t in _bars.times:
            _fast_to_datetime(_t)
    # Patch engine static methods to use the cache
    HelixEngine._to_datetime = staticmethod(_fast_to_datetime)
    BreakoutEngine._to_datetime = staticmethod(_fast_to_datetime)

    # --- Initialize trackers ---
    init_eq = config.initial_equity
    portfolio_equity = init_eq
    peak_risk_equity = init_eq
    risk_sizing_equity = init_eq
    risk_scale = 1.0

    heat_tracker = PortfolioHeatTracker(
        heat_cap_R=config.heat_cap_R,
        portfolio_daily_stop_R=config.portfolio_daily_stop_R,
        strategy_slots=[config.atrss, config.helix, config.brs, config.breakout],
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
    overlay_total_commission: float = 0.0
    overlay_per_sym_pnl: dict[str, float] = {}
    overlay_prev_sym_value: dict[str, float] = {}
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    overlay_rsi: dict[str, np.ndarray] = {}
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    overlay_in_position: dict[str, bool] = {}
    daily_date_idx_syms = list(set(config.overlay_symbols if config.overlay_enabled else []))
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
    for label, engines in [("atrss", atrss_engines), ("helix", helix_engines), ("breakout", breakout_engines)]:
        for sym in engines:
            prev_equity[f"{label}_{sym}"] = init_eq
    for sym in brs_engines:
        prev_equity[f"brs_{sym}"] = 0.0

    # Track previous trade counts to detect actual trade closes
    prev_trade_counts: dict[str, int] = {}
    for label, engines in [("atrss", atrss_engines), ("helix", helix_engines), ("brs", brs_engines), ("breakout", breakout_engines)]:
        for sym in engines:
            prev_trade_counts[f"{label}_{sym}"] = 0

    equity_curve_mtm: list[float] = []
    equity_curve_realized: list[float] = []
    timestamps: list = []
    heat_samples: list[float] = []
    prev_date: date | None = None
    entry_attempts: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "BRS_R9": 0, "SWING_BREAKOUT_V3": 0}
    accepted_entries: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "BRS_R9": 0, "SWING_BREAKOUT_V3": 0}
    blocked_entries: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "BRS_R9": 0, "SWING_BREAKOUT_V3": 0}
    entry_event_log: list[dict] = []
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
            # and holidays where the hourly data has bars but daily ETF data
            # does not.
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
                old_overlay_shares = dict(overlay_shares)
                _rebalance_overlay(
                    config, data.daily, overlay_emas, daily_date_idx,
                    prev_date, total_equity, overlay_shares,
                    overlay_rsi=overlay_rsi or None,
                    overlay_macd=overlay_macd or None,
                    overlay_in_position=overlay_in_position,
                    bear_regime_active=_brs_bear_strong_active(brs_engines),
                )
                # Overlay transaction costs: commission + slippage on share deltas
                for _osym in config.overlay_symbols:
                    old_sh = old_overlay_shares.get(_osym, 0.0)
                    new_sh = overlay_shares.get(_osym, 0.0)
                    delta_sh = abs(new_sh - old_sh)
                    _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                    if delta_sh > 0 and _sidx is not None and _sidx + 1 < len(data.daily[_osym].opens):
                        cost = _overlay_transaction_cost(
                            delta_sh,
                            data.daily[_osym].opens[_sidx + 1],
                            config.slippage,
                        )
                        overlay_total_commission += cost
                        overlay_cumulative_pnl -= cost

                # Baseline for next MTM: next-open for changed positions
                # (execution price), close for held (preserves overnight gap).
                overlay_prev_value = 0.0
                for _osym in config.overlay_symbols:
                    old_sh = old_overlay_shares.get(_osym, 0.0)
                    new_sh = overlay_shares.get(_osym, 0.0)
                    _sv = 0.0
                    if new_sh != 0.0:
                        _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                        if _sidx is not None:
                            if old_sh != new_sh and _sidx + 1 < len(data.daily[_osym].opens):
                                _sv = new_sh * data.daily[_osym].opens[_sidx + 1]
                            else:
                                _sv = new_sh * data.daily[_osym].closes[_sidx]
                    overlay_prev_sym_value[_osym] = _sv
                    overlay_prev_value += _sv

        if config.dynamic_risk_enabled:
            peak_risk_equity = max(peak_risk_equity, portfolio_equity)
            drawdown_pct = (peak_risk_equity - portfolio_equity) / peak_risk_equity if peak_risk_equity > 0 else 0.0
            risk_scale = _drawdown_risk_multiplier(drawdown_pct, config.drawdown_risk_tiers)
            risk_sizing_equity = max(portfolio_equity * risk_scale, 0.0)
            heat_tracker.update_unit_risk(
                risk_sizing_equity,
                [config.atrss, config.helix, config.brs, config.breakout],
            )
        else:
            risk_sizing_equity = portfolio_equity
            risk_scale = 1.0

        prev_date = current_date

        # === Step 1: ATRSS engines (priority 0) ===
        for sym, engine in atrss_engines.items():
            key = f"atrss_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            engine.sizing_equity = risk_sizing_equity
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
                    entry_attempts["ATRSS"] += 1
                    risk_dollars = abs(cand.trigger_price - cand.initial_stop) * engine.point_value * cand.qty
                    if skip_heat:
                        engine.submit_candidate(cand, bar_time)
                        accepted_entries["ATRSS"] += 1
                        _log_entry_event(
                            entry_event_log, heat_tracker, bar_time, "ATRSS", sym, "entry",
                            "accepted", "", risk_dollars,
                            direction=int(cand.direction),
                            entry_price=cand.trigger_price,
                            stop_price=cand.initial_stop,
                            quantity=cand.qty,
                            signal_type=_enum_label(cand.type),
                            quality_score=getattr(cand, "rank_score", None),
                            entry_already_filled=False,
                            metadata={"risk_scale": risk_scale, "atrh": getattr(cand, "atrh", 0.0)},
                        )
                    else:
                        ok, reason = heat_tracker.can_enter("ATRSS", risk_dollars)
                        if ok:
                            engine.submit_candidate(cand, bar_time)
                            accepted_entries["ATRSS"] += 1
                            _log_entry_event(
                                entry_event_log, heat_tracker, bar_time, "ATRSS", sym, "entry",
                                "accepted", "", risk_dollars,
                                direction=int(cand.direction),
                                entry_price=cand.trigger_price,
                                stop_price=cand.initial_stop,
                                quantity=cand.qty,
                                signal_type=_enum_label(cand.type),
                                quality_score=getattr(cand, "rank_score", None),
                                entry_already_filled=False,
                                metadata={"risk_scale": risk_scale, "atrh": getattr(cand, "atrh", 0.0)},
                            )
                        else:
                            blocked_entries["ATRSS"] += 1
                            event = _log_entry_event(
                                entry_event_log, heat_tracker, bar_time, "ATRSS", sym, "entry",
                                "blocked", reason, risk_dollars,
                                direction=int(cand.direction),
                                entry_price=cand.trigger_price,
                                stop_price=cand.initial_stop,
                                quantity=cand.qty,
                                signal_type=_enum_label(cand.type),
                                quality_score=getattr(cand, "rank_score", None),
                                entry_already_filled=False,
                                metadata={"risk_scale": risk_scale, "atrh": getattr(cand, "atrh", 0.0)},
                            )
                            heat_rejection_log.append(event)
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

        # === Step 3: BRS engines (priority 2) ===
        for sym in _brs_run_order(list(brs_engines)):
            engine = brs_engines[sym]
            key = f"brs_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            had_pending_entry = engine._pending_entry is not None
            had_pending_add = engine._pending_add is not None

            try:
                engine._step_bar(
                    bar_idx,
                    sizing_equity=risk_sizing_equity,
                    shared_reserved_risk_dollars=_brs_reserved_risk_dollars(brs_engines),
                    shared_concurrent_positions=_brs_reserved_position_slots(brs_engines),
                )
            except (OverflowError, ValueError):
                continue

            if not skip_heat and engine._pending_entry is not None and not had_pending_entry:
                pending = engine._pending_entry
                signal = pending.signal
                risk_dollars = pending.signal.risk_per_unit * pending.qty
                entry_attempts["BRS_R9"] += 1
                ok, reason = heat_tracker.can_enter("BRS_R9", risk_dollars)
                if not ok:
                    blocked_entries["BRS_R9"] += 1
                    event = _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "BRS_R9", sym, "entry",
                        "blocked", reason, risk_dollars,
                        direction=int(signal.direction),
                        entry_price=signal.signal_price,
                        stop_price=signal.stop_price,
                        quantity=pending.qty,
                        signal_type=_enum_label(signal.entry_type),
                        quality_score=getattr(signal, "quality_score", None),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "regime": _enum_label(getattr(signal, "regime_at_entry", "")),
                            "bear_conviction": float(getattr(signal, "bear_conviction", 0.0) or 0.0),
                            "vol_factor": float(getattr(signal, "vol_factor", 0.0) or 0.0),
                        },
                    )
                    heat_rejection_log.append(event)
                    _cancel_brs_pending_order(engine, "entry")
                    if "daily stop" in reason.lower():
                        daily_stop_activations += 1
                else:
                    accepted_entries["BRS_R9"] += 1
                    _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "BRS_R9", sym, "entry",
                        "accepted", "", risk_dollars,
                        direction=int(signal.direction),
                        entry_price=signal.signal_price,
                        stop_price=signal.stop_price,
                        quantity=pending.qty,
                        signal_type=_enum_label(signal.entry_type),
                        quality_score=getattr(signal, "quality_score", None),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "regime": _enum_label(getattr(signal, "regime_at_entry", "")),
                            "bear_conviction": float(getattr(signal, "bear_conviction", 0.0) or 0.0),
                            "vol_factor": float(getattr(signal, "vol_factor", 0.0) or 0.0),
                        },
                    )

            if not skip_heat and engine._pending_add is not None and not had_pending_add:
                pending_add = engine._pending_add
                signal = pending_add.signal
                risk_dollars = engine._pos_risk_per_unit * pending_add.qty
                entry_attempts["BRS_R9"] += 1
                ok, reason = heat_tracker.can_enter("BRS_R9", risk_dollars)
                if not ok:
                    blocked_entries["BRS_R9"] += 1
                    event = _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "BRS_R9", sym, "add_on",
                        "blocked", reason, risk_dollars,
                        direction=int(signal.direction),
                        entry_price=signal.signal_price,
                        stop_price=signal.stop_price,
                        quantity=pending_add.qty,
                        signal_type=_enum_label(signal.entry_type),
                        quality_score=getattr(signal, "quality_score", None),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "regime": _enum_label(getattr(signal, "regime_at_entry", "")),
                            "bear_conviction": float(getattr(signal, "bear_conviction", 0.0) or 0.0),
                            "vol_factor": float(getattr(signal, "vol_factor", 0.0) or 0.0),
                        },
                    )
                    heat_rejection_log.append(event)
                    _cancel_brs_pending_order(engine, "add_on")
                    if "daily stop" in reason.lower():
                        daily_stop_activations += 1
                else:
                    accepted_entries["BRS_R9"] += 1
                    _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "BRS_R9", sym, "add_on",
                        "accepted", "", risk_dollars,
                        direction=int(signal.direction),
                        entry_price=signal.signal_price,
                        stop_price=signal.stop_price,
                        quantity=pending_add.qty,
                        signal_type=_enum_label(signal.entry_type),
                        quality_score=getattr(signal, "quality_score", None),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "regime": _enum_label(getattr(signal, "regime_at_entry", "")),
                            "bear_conviction": float(getattr(signal, "bear_conviction", 0.0) or 0.0),
                            "vol_factor": float(getattr(signal, "vol_factor", 0.0) or 0.0),
                        },
                    )

        # === Step 4: Breakout engines (priority 3) ===
        for sym, engine in breakout_engines.items():
            key = f"breakout_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            # Breakout uses self.equity for both sizing and PnL tracking
            # (no separate sizing_equity like ATRSS/Helix). In risk-based
            # mode we inject portfolio equity for sizing, then extract the
            # fill-caused PnL delta and restore independent tracking.
            if not skip_heat:
                saved_equity = engine.equity
                engine.equity = risk_sizing_equity

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
                fill_pnl = engine.equity - risk_sizing_equity
                engine.equity = saved_equity + fill_pnl

            has_position = engine.active_position is not None and engine.active_position.qty_open > 0
            if has_position and not had_position:
                pos = engine.active_position
                risk_dollars = abs(pos.fill_price - pos.current_stop) * engine.point_value * pos.qty_open
                entry_attempts["SWING_BREAKOUT_V3"] += 1
                if skip_heat:
                    ok, reason = True, ""
                else:
                    ok, reason = heat_tracker.can_enter("SWING_BREAKOUT_V3", risk_dollars)
                if not ok:
                    blocked_entries["SWING_BREAKOUT_V3"] += 1
                    event = _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "SWING_BREAKOUT_V3", sym, "entry",
                        "blocked", reason, risk_dollars,
                        direction=int(pos.setup.direction),
                        entry_price=pos.fill_price,
                        stop_price=pos.current_stop,
                        quantity=pos.qty_open,
                        signal_type=_enum_label(pos.entry_type or getattr(pos.setup, "entry_type", "")),
                        quality_score=getattr(pos, "quality_mult", None),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "regime": str(getattr(pos, "regime_at_entry", "") or ""),
                            "disp_at_entry": float(getattr(pos, "disp_at_entry", 0.0) or 0.0),
                            "rvol_d_at_entry": float(getattr(pos, "rvol_d_at_entry", 0.0) or 0.0),
                        },
                    )
                    heat_rejection_log.append(event)
                    _force_flatten_breakout(engine, pos)
                    if "daily stop" in reason.lower():
                        daily_stop_activations += 1
                else:
                    accepted_entries["SWING_BREAKOUT_V3"] += 1
                    _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "SWING_BREAKOUT_V3", sym, "entry",
                        "accepted", "", risk_dollars,
                        direction=int(pos.setup.direction),
                        entry_price=pos.fill_price,
                        stop_price=pos.current_stop,
                        quantity=pos.qty_open,
                        signal_type=_enum_label(pos.entry_type or getattr(pos.setup, "entry_type", "")),
                        quality_score=getattr(pos, "quality_mult", None),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "regime": str(getattr(pos, "regime_at_entry", "") or ""),
                            "disp_at_entry": float(getattr(pos, "disp_at_entry", 0.0) or 0.0),
                            "rvol_d_at_entry": float(getattr(pos, "rvol_d_at_entry", 0.0) or 0.0),
                        },
                    )

        # === Step 5: Helix engines (priority 4) ===
        for sym, engine in helix_engines.items():
            key = f"helix_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            engine.sizing_equity = risk_sizing_equity

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
                risk_dollars = abs(pos.fill_price - pos.current_stop) * engine.point_value * pos.qty_open
                entry_attempts["AKC_HELIX"] += 1
                if skip_heat:
                    ok, reason = True, ""
                else:
                    ok, reason = heat_tracker.can_enter("AKC_HELIX", risk_dollars)
                if not ok:
                    # Force flatten: reverse the entry
                    blocked_entries["AKC_HELIX"] += 1
                    event = _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "AKC_HELIX", sym, "entry",
                        "blocked", reason, risk_dollars,
                        direction=int(pos.setup.direction),
                        entry_price=pos.fill_price,
                        stop_price=pos.current_stop,
                        quantity=pos.qty_open,
                        signal_type=_enum_label(pos.setup.setup_class),
                        quality_score=float(getattr(pos.setup, "adx_at_entry", 0.0) or 0.0),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "origin_tf": str(getattr(pos.setup, "origin_tf", "") or ""),
                            "regime": str(getattr(pos, "regime_at_entry", "") or ""),
                            "regime_4h": str(getattr(pos.setup, "regime_4h_at_entry", "") or ""),
                            "div_mag_norm": float(getattr(pos.setup, "div_mag_norm", 0.0) or 0.0),
                        },
                    )
                    heat_rejection_log.append(event)
                    _force_flatten_helix(engine, pos)
                    if "daily stop" in reason.lower():
                        daily_stop_activations += 1
                else:
                    accepted_entries["AKC_HELIX"] += 1
                    _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "AKC_HELIX", sym, "entry",
                        "accepted", "", risk_dollars,
                        direction=int(pos.setup.direction),
                        entry_price=pos.fill_price,
                        stop_price=pos.current_stop,
                        quantity=pos.qty_open,
                        signal_type=_enum_label(pos.setup.setup_class),
                        quality_score=float(getattr(pos.setup, "adx_at_entry", 0.0) or 0.0),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "origin_tf": str(getattr(pos.setup, "origin_tf", "") or ""),
                            "regime": str(getattr(pos, "regime_at_entry", "") or ""),
                            "regime_4h": str(getattr(pos.setup, "regime_4h_at_entry", "") or ""),
                            "div_mag_norm": float(getattr(pos.setup, "div_mag_norm", 0.0) or 0.0),
                        },
                    )
                    # Check coordination Rule 2: size boost
                    if pos.setup and hasattr(pos.setup, 'direction'):
                        direction = pos.setup.direction
                        if coordinator.has_atrss_position(sym, direction):
                            coordinator.boost_count += 1
                            coordination_event_log.append({"time": _to_datetime(ts), "type": "boost", "trigger_strategy": "ATRSS", "target_strategy": "AKC_HELIX", "symbol": sym})

        # === Step 6: Update portfolio equity from per-engine PnL deltas ===
        # Each engine's equity = initial_equity + cumulative realized PnL
        # Portfolio equity = initial_equity + sum of all engines' realized PnL
        for label, engines, strat_id in [
            ("atrss", atrss_engines, "ATRSS"),
            ("brs", brs_engines, "BRS_R9"),
            ("breakout", breakout_engines, "SWING_BREAKOUT_V3"),
            ("helix", helix_engines, "AKC_HELIX"),
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

        # === Step 7: Update heat tracker ===
        risk_by_strat = _compute_open_risk(atrss_engines, helix_engines, brs_engines, breakout_engines)
        heat_tracker.update_open_risk(risk_by_strat)

        engine_groups = [
            ("atrss", atrss_engines),
            ("brs", brs_engines),
            ("breakout", breakout_engines),
            ("helix", helix_engines),
        ]
        equity_curve_realized.append(portfolio_equity + overlay_cumulative_pnl)
        equity_curve_mtm.append(_combined_child_mtm_equity(init_eq, engine_groups, overlay_cumulative_pnl))
        timestamps.append(ts)

        # Track heat utilization (normalized to ATRSS R-units for consistency)
        total_risk = sum(risk_by_strat.values())
        # Use the same portfolio unit risk as the gate, including dynamic NAV throttles.
        norm_base = heat_tracker.portfolio_unit_risk_dollars
        heat_R = total_risk / norm_base if norm_base > 0 else 0.0
        heat_samples.append(heat_R)

    _patch.__exit__(None, None, None)

    for sym, eng in brs_engines.items():
        key = f"brs_{sym}"
        eng._finalize_end_of_data()
        if eng.equity_curve:
            eng.equity_curve[-1] = eng._mtm_equity()
        delta = eng.equity - prev_equity[key]
        portfolio_equity += delta
        prev_equity[key] = eng.equity
        n_trades = len(eng.trades)
        if n_trades > prev_trade_counts[key]:
            for t_idx in range(prev_trade_counts[key], n_trades):
                heat_tracker.record_trade_close("BRS_R9", eng.trades[t_idx].pnl_dollars)
            prev_trade_counts[key] = n_trades
        if equity_curve_realized:
            equity_curve_realized[-1] = portfolio_equity + overlay_cumulative_pnl
        if equity_curve_mtm:
            equity_curve_mtm[-1] = _combined_child_mtm_equity(
                init_eq,
                [
                    ("atrss", atrss_engines),
                    ("brs", brs_engines),
                    ("breakout", breakout_engines),
                    ("helix", helix_engines),
                ],
                overlay_cumulative_pnl,
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
        if equity_curve_realized:
            equity_curve_realized[-1] = portfolio_equity + overlay_cumulative_pnl
        if equity_curve_mtm:
            equity_curve_mtm[-1] = _combined_child_mtm_equity(
                init_eq,
                [
                    ("atrss", atrss_engines),
                    ("brs", brs_engines),
                    ("breakout", breakout_engines),
                    ("helix", helix_engines),
                ],
                overlay_cumulative_pnl,
            )

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

    all_brs_trades = []
    for eng in brs_engines.values():
        all_brs_trades.extend(eng.trades)

    all_breakout_trades = []
    for eng in breakout_engines.values():
        all_breakout_trades.extend(eng.trades)

    strategy_results = {}
    for sid, trades, blocked in [
        ("ATRSS", all_atrss_trades, blocked_entries["ATRSS"]),
        ("BRS_R9", all_brs_trades, blocked_entries["BRS_R9"]),
        ("SWING_BREAKOUT_V3", all_breakout_trades, blocked_entries["SWING_BREAKOUT_V3"]),
        ("AKC_HELIX", all_helix_trades, blocked_entries["AKC_HELIX"]),
    ]:
        total_pnl = sum(t.pnl_dollars for t in trades)
        wins = sum(1 for t in trades if t.pnl_dollars > 0)
        losses = sum(1 for t in trades if t.pnl_dollars <= 0)
        total_r = sum(t.r_multiple for t in trades)
        strategy_results[sid] = StrategyResult(
            strategy_id=sid,
            entry_signals_fired=entry_attempts[sid],
            entries_accepted_by_portfolio=accepted_entries[sid],
            total_trades=len(trades),
            total_pnl=total_pnl,
            winning_trades=wins,
            losing_trades=losses,
            total_r=total_r,
            entries_blocked_by_heat=blocked,
        )

    return UnifiedPortfolioResult(
        combined_equity=np.array(equity_curve_mtm),
        combined_equity_mtm=np.array(equity_curve_mtm),
        combined_equity_realized=np.array(equity_curve_realized),
        combined_timestamps=np.array(timestamps),
        heat_stats=heat_stats,
        strategy_results=strategy_results,
        coordination_tighten_count=coordinator.tighten_count,
        coordination_boost_count=coordinator.boost_count,
        portfolio_daily_stop_activations=daily_stop_activations,
        overlay_pnl=overlay_cumulative_pnl,
        overlay_commission=overlay_total_commission,
        overlay_per_symbol_pnl=dict(overlay_per_sym_pnl),
        atrss_trades=all_atrss_trades,
        helix_trades=all_helix_trades,
        brs_trades=all_brs_trades,
        breakout_trades=all_breakout_trades,
        entry_events=entry_event_log,
        heat_rejections=heat_rejection_log,
        coordination_events=coordination_event_log,
    )


# ---------------------------------------------------------------------------
# Force-flatten helpers (when heat check fails post-entry)
# ---------------------------------------------------------------------------

def _brs_run_order(symbols) -> list[str]:
    return sorted(symbols, key=lambda sym: (0 if sym == "QQQ" else 1, sym))


def _brs_reserved_risk_dollars(engines: dict[str, BRSEngine]) -> float:
    return sum(engine.reserved_risk_dollars() for engine in engines.values())


def _brs_reserved_position_slots(engines: dict[str, BRSEngine]) -> int:
    return sum(engine.reserved_position_slots() for engine in engines.values())


def _brs_bear_strong_active(engines: dict[str, BRSEngine]) -> bool:
    return any(getattr(engine.daily_ctx, "regime", None) == BRSRegime.BEAR_STRONG for engine in engines.values())


def _drawdown_risk_multiplier(drawdown_pct: float, tiers: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> float:
    multiplier = 1.0
    for threshold, tier_multiplier in sorted((float(t), float(m)) for t, m in tiers):
        if drawdown_pct >= threshold:
            multiplier = tier_multiplier
    return max(0.0, multiplier)


def _enum_label(value) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", getattr(value, "name", value)))


def _cancel_brs_pending_order(engine: BRSEngine, tag: str) -> None:
    cancelled = engine.broker.cancel_orders(engine.symbol, tag=tag)
    cancelled_ids = {order.order_id for order in cancelled}
    if tag == "entry" and engine._pending_entry is not None:
        cancelled_ids.add(engine._pending_entry.order_id)
        engine._pending_entry = None
    elif tag == "add_on" and engine._pending_add is not None:
        cancelled_ids.add(engine._pending_add.order_id)
        engine._pending_add = None
    for order_id in cancelled_ids:
        engine._core_state.pending_orders.pop(order_id, None)


def _log_entry_event(
    events: list[dict],
    heat_tracker: PortfolioHeatTracker,
    event_time: datetime,
    strategy_id: str,
    symbol: str,
    stage: str,
    status: str,
    reason: str,
    risk_dollars: float,
    *,
    direction: int | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    quantity: float | None = None,
    signal_type: str | None = None,
    quality_score: float | None = None,
    entry_already_filled: bool | None = None,
    metadata: dict | None = None,
) -> dict:
    event = {
        "time": event_time,
        "strategy": strategy_id,
        "symbol": symbol,
        "stage": stage,
        "status": status,
        "reason": reason,
        "risk_dollars": float(risk_dollars or 0.0),
        "risk_context": _entry_risk_context(heat_tracker, strategy_id, risk_dollars),
    }
    if direction is not None:
        event["direction"] = int(direction)
    if entry_price is not None:
        event["entry_price"] = float(entry_price)
    if stop_price is not None:
        event["stop_price"] = float(stop_price)
    if quantity is not None:
        event["quantity"] = float(quantity)
    if signal_type:
        event["signal_type"] = str(signal_type)
    if quality_score is not None:
        event["quality_score"] = float(quality_score)
    if entry_already_filled is not None:
        event["entry_already_filled"] = bool(entry_already_filled)
    if metadata:
        event["metadata"] = dict(metadata)
    events.append(event)
    return event


def _entry_risk_context(
    heat_tracker: PortfolioHeatTracker,
    strategy_id: str,
    risk_dollars: float,
) -> dict[str, float]:
    if hasattr(heat_tracker, "entry_risk_context"):
        return heat_tracker.entry_risk_context(strategy_id, risk_dollars)
    strat = heat_tracker._strats.get(strategy_id)
    portfolio_unit = heat_tracker._portfolio_unit_risk
    total_open_dollars = heat_tracker.total_open_risk_dollars
    portfolio_open_r = total_open_dollars / portfolio_unit if portfolio_unit > 0 else 0.0
    portfolio_request_r = risk_dollars / portfolio_unit if portfolio_unit > 0 else 0.0
    context = {
        "portfolio_open_risk_R": float(portfolio_open_r),
        "portfolio_request_risk_R": float(portfolio_request_r),
        "portfolio_after_request_R": float(portfolio_open_r + portfolio_request_r),
        "portfolio_heat_cap_R": float(heat_tracker._heat_cap_R),
    }
    if strat is not None:
        strategy_request_r = risk_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0.0
        context.update(
            {
                "strategy_open_risk_R": float(strat.open_risk_R),
                "strategy_request_risk_R": float(strategy_request_r),
                "strategy_after_request_R": float(strat.open_risk_R + strategy_request_r),
                "strategy_heat_cap_R": float(strat.max_heat_R),
                "strategy_daily_realized_R": float(strat.daily_realized_R),
                "strategy_daily_stop_R": float(strat.daily_stop_R),
            }
        )
    return context


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
        from backtests.swing.engine.breakout_engine import CampaignState
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
    print("UNIFIED SWING PORTFOLIO BACKTEST RESULTS")
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
        print(f"  Overlay Costs:       ${result.overlay_commission:,.2f}")
        print(f"  Active Strategy PnL: ${active_pnl:+,.2f} ({act_pct:+.1f}%)")
        print(f"  Overlay Max Pct:     {config.overlay_max_pct:.0%}")
        if result.overlay_per_symbol_pnl:
            for osym, opnl in sorted(result.overlay_per_symbol_pnl.items()):
                opnl_pct = opnl / init_eq * 100 if init_eq > 0 else 0
                print(f"    {osym:<6} overlay PnL: ${opnl:+,.2f} ({opnl_pct:+.1f}%)")

    print(f"\n{'Per-Strategy Breakdown':}")
    print(f"{'Strategy':<22} {'Trades':>7} {'Win%':>6} {'PnL':>10} {'Total R':>8} {'Blocked':>8}")
    print("-" * 65)
    for sid in ["ATRSS", "AKC_HELIX", "BRS_R9", "SWING_BREAKOUT_V3"]:
        sr = result.strategy_results.get(sid)
        if sr is None:
            continue
        win_pct = sr.winning_trades / sr.total_trades * 100 if sr.total_trades > 0 else 0
        print(f"{sid:<22} {sr.total_trades:>7} {win_pct:>5.1f}% ${sr.total_pnl:>9,.2f} {sr.total_r:>7.1f}R {sr.entries_blocked_by_heat:>8}")

    print("=" * 70)
