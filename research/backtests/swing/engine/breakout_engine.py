"""Core single-symbol bar-by-bar Breakout v3.3-ETF backtesting engine.

Translates strategy_3/engine.py's async event-driven architecture into a
synchronous bar-by-bar loop.  Uses SimBroker for fill simulation (shared
with the ATRSS and Helix engines).

All strategy logic is called via pure functions in strategy_3/*.py.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone, time

import numpy as np
import pandas as pd

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from strategy_3 import allocator, gates, signals, stops
from strategy_3.config import (
    ADD_RISK_MULT,
    ADX_PERIOD,
    ATR_DAILY_LONG_PERIOD,
    ATR_DAILY_PERIOD,
    ATR_HOURLY_PERIOD,
    BE_BUFFER_ATR_MULT,
    CAMPAIGN_RISK_BUDGET_MULT,
    CHOP_DEGRADED_SIZE_MULT,
    CHOP_DEGRADED_STALE_ADJ,
    CORR_LOOKBACK_BARS,
    DISP_LOOKBACK_MAX,
    EMA_1H_PERIOD,
    EMA_4H_PERIOD,
    EMA_DAILY_PERIOD,
    ENTRY_A_TTL_RTH_HOURS,
    ENTRY_C_TTL_RTH_HOURS,
    INSIDE_CLOSE_INVALIDATION_COUNT,
    MAX_ADDS_PER_CAMPAIGN,
    PENDING_MAX_RTH_HOURS,
    REGIME_CHOP_BLOCK,
    REGIME_CHOP_SCORE_OVERRIDE,
    SCORE_THRESHOLD_DEGRADED,
    SCORE_THRESHOLD_NORMAL,
    SCORE_THRESHOLD_RANGE,
    TP1_R_ALIGNED,
    TP1_R_CAUTION,
    TP1_R_NEUTRAL,
    TP2_R_ALIGNED,
    TP2_R_CAUTION,
    TP2_R_NEUTRAL,
    SYMBOL_CONFIGS,
    SymbolConfig,
)
from strategy_3.indicators import (
    adx,
    atr,
    compute_avwap,
    compute_daily_slope,
    compute_regime_4h,
    compute_rvol_d,
    compute_rvol_h,
    compute_wvwap,
    ema,
    get_slot_key,
    highest,
    lowest,
    past_only_quantile,
    pullback_ref,
    sma,
    update_slot_medians,
)
from strategy_3.models import (
    CampaignState,
    ChopMode,
    CircuitBreakerState,
    DailyContext,
    Direction,
    EntryType,
    ExitTier,
    HourlyState,
    PendingEntry,
    PositionState,
    Regime4H,
    RollingHistories,
    SetupInstance,
    SetupState,
    SymbolCampaign,
    TradeRegime,
)

from backtest.config import SlippageConfig
from backtest.config_breakout import BreakoutAblationFlags, BreakoutBacktestConfig
from backtest.data.preprocessing import NumpyBars
from backtest.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimBroker,
    SimOrder,
)

logger = logging.getLogger(__name__)

try:
    import zoneinfo
    _ET = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    _ET = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# Param override patching (matches ATRSS _AblationPatch pattern)
# ---------------------------------------------------------------------------

class _AblationPatch:
    """Temporarily patch strategy_3 module constants for param_overrides.

    Monkeypatches constants in strategy_3.config (and this engine module's
    own top-level bindings) within a context manager. Restores originals
    on exit. Safe in single-threaded / per-process execution.
    """

    def __init__(self, flags: BreakoutAblationFlags, param_overrides: dict[str, float] | None = None):
        self.flags = flags
        self.overrides = param_overrides or {}
        self._patches: list[tuple[object, str, object]] = []

    def __enter__(self):
        import sys
        import strategy_3.config as scfg

        engine_mod = sys.modules[__name__]

        for key, val in self.overrides.items():
            upper_key = key.upper()
            # Patch source module
            if hasattr(scfg, upper_key):
                self._patch(scfg, upper_key, val)
            # Patch engine module's own top-level imported binding
            if hasattr(engine_mod, upper_key):
                self._patch(engine_mod, upper_key, val)
        return self

    def __exit__(self, *exc):
        for obj, attr, orig in reversed(self._patches):
            setattr(obj, attr, orig)
        self._patches.clear()

    def _patch(self, obj, attr: str, value):
        self._patches.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)


# ---------------------------------------------------------------------------
# Trade record for post-hoc analysis (Breakout-specific fields)
# ---------------------------------------------------------------------------

@dataclass
class BreakoutTradeRecord:
    """One completed Breakout trade (entry + exit)."""

    symbol: str = ""
    direction: int = 0
    entry_type: str = ""         # A, B, C_standard, C_continuation, ADD
    exit_tier: str = ""          # ALIGNED, NEUTRAL, CAUTION
    campaign_id: int = 0
    box_version: int = 0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0     # initial stop
    qty: int = 0
    exit_reason: str = ""       # STOP, STALE, GAP_STOP, TRAILING_STOP, FLATTEN, END_OF_DATA
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    r_multiple_per_share: float = 0.0  # pnl_points / r_price (sizing-independent)
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    days_held: int = 0
    commission: float = 0.0
    # Partial fills
    tp1_done: bool = False
    tp2_done: bool = False
    runner_active: bool = False
    qty_tp1: int = 0
    qty_tp2: int = 0
    # Adds
    add_count: int = 0
    add_contribution_r: float = 0.0
    # Gap events
    gap_stop_event: bool = False
    gap_stop_fill_price: float = 0.0
    # Context at entry
    disp_at_entry: float = 0.0
    disp_th_at_entry: float = 0.0
    rvol_d_at_entry: float = 0.0
    rvol_h_at_entry: float = 0.0
    regime_at_entry: str = ""
    sq_good_at_entry: bool = False
    quality_mult_at_entry: float = 1.0
    score_at_entry: int = 0
    # Score sub-components (Phase 2)
    score_vol_at_entry: int = 0
    score_squeeze_at_entry: int = 0
    score_regime_at_entry: int = 0
    score_consec_at_entry: int = 0
    score_atr_at_entry: int = 0
    # Quality mult sub-components (Phase 2)
    regime_mult_at_entry: float = 1.0
    disp_mult_at_entry: float = 1.0
    squeeze_mult_at_entry: float = 1.0
    corr_mult_at_entry: float = 1.0
    chop_mode_at_entry: str = ""
    continuation_at_entry: bool = False
    # Partial fill timing (Phase 2)
    tp1_bar: int = 0
    tp2_bar: int = 0
    be_bar: int = 0
    # Stop evolution (Phase 2)
    final_stop_at_exit: float = 0.0
    stop_tightens: int = 0
    # Regime at exit (Phase 2)
    regime_at_exit: str = ""
    # Friction
    friction_dollars: float = 0.0


# ---------------------------------------------------------------------------
# Signal event for filter attribution
# ---------------------------------------------------------------------------

@dataclass
class SignalEvent:
    """Every hourly evaluation logged for filter attribution analysis."""

    timestamp: datetime | None = None
    symbol: str = ""
    campaign_state: str = ""
    direction: int = 0
    entry_type_selected: str = ""     # A, B, C_standard, C_continuation, or ""
    allowed: bool = False
    blocked_reason: str = ""          # gate that blocked, or ""
    # Multipliers at evaluation time
    quality_mult: float = 0.0
    expiry_mult: float = 0.0
    disp_mult: float = 0.0
    # Context
    rvol_h: float = 0.0
    r_state: float = 0.0
    regime_4h: str = ""
    score_total: int = 0
    # Score sub-components (Phase 2)
    score_vol: int = 0
    score_squeeze: int = 0
    score_regime: int = 0
    score_consec: int = 0
    score_atr: int = 0
    chop_mode: str = ""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BreakoutSymbolResult:
    """Result of backtesting a single symbol with Breakout."""

    symbol: str
    trades: list[BreakoutTradeRecord] = field(default_factory=list)
    signal_events: list[SignalEvent] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    total_commission: float = 0.0
    # Diagnostic counters
    entries_placed: int = 0
    entries_filled: int = 0
    entries_expired: int = 0
    entries_rejected: int = 0
    entries_blocked: int = 0
    adds_placed: int = 0
    adds_filled: int = 0
    # Campaign stats
    campaigns_activated: int = 0
    breakouts_qualified: int = 0
    dirty_episodes: int = 0
    continuations_entered: int = 0
    # Regime distribution
    regime_bars_bull: int = 0
    regime_bars_bear: int = 0
    regime_bars_chop: int = 0


# ---------------------------------------------------------------------------
# Active position tracking during backtest
# ---------------------------------------------------------------------------

@dataclass
class _ActivePosition:
    """Internal position tracking during the bar-by-bar loop."""

    setup: SetupInstance
    fill_price: float = 0.0
    qty_open: int = 0
    qty_initial: int = 0
    initial_stop: float = 0.0
    current_stop: float = 0.0
    r_price: float = 0.0          # |entry - stop0| for R calc
    entry_time: datetime | None = None
    bars_held: int = 0
    days_held: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0
    # Partials
    tp1_done: bool = False
    tp2_done: bool = False
    runner_active: bool = False
    trail_mult: float = 4.5
    qty_tp1: int = 0
    qty_tp2: int = 0
    # Adds
    add_count: int = 0
    add_realized_pnl: float = 0.0
    # PnL
    realized_pnl: float = 0.0
    commission: float = 0.0
    # Context
    regime_at_entry: str = ""
    exit_tier: str = ""
    entry_type: str = ""
    # Cached daily context at entry
    disp_at_entry: float = 0.0
    disp_th_at_entry: float = 0.0
    rvol_d_at_entry: float = 0.0
    sq_good_at_entry: bool = False
    quality_mult_at_entry: float = 1.0
    score_at_entry: int = 0
    # Score sub-components (Phase 2)
    score_vol_at_entry: int = 0
    score_squeeze_at_entry: int = 0
    score_regime_at_entry: int = 0
    score_consec_at_entry: int = 0
    score_atr_at_entry: int = 0
    # Quality mult sub-components (Phase 2)
    regime_mult_at_entry: float = 1.0
    disp_mult_at_entry: float = 1.0
    squeeze_mult_at_entry: float = 1.0
    corr_mult_at_entry: float = 1.0
    chop_mode_at_entry: str = ""
    continuation_at_entry: bool = False
    # Partial fill timing (Phase 2)
    tp1_bar: int = 0
    tp2_bar: int = 0
    be_bar: int = 0
    # Stop evolution (Phase 2)
    stop_tightens: int = 0
    # Gap
    gap_stop_event: bool = False
    gap_stop_fill_price: float = 0.0
    # Friction
    friction_dollars: float = 0.0
    # Regime shift tracking (Change #7)
    regime_shift_bars: int = 0
    # Last daily date tracked for days_held
    _last_daily_date: date | None = None


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class BreakoutEngine:
    """Single-symbol bar-by-bar Breakout v3.3-ETF backtest engine.

    Processes 1H bars sequentially, with daily campaign management at
    daily boundaries and hourly entry/exit logic on each bar.
    Uses SimBroker for fill simulation.
    """

    def __init__(
        self,
        symbol: str,
        cfg: SymbolConfig,
        bt_config: BreakoutBacktestConfig,
        *,
        point_value: float = 1.0,
        external_positions: dict[str, PositionState] | None = None,
        external_circuit_breaker: CircuitBreakerState | None = None,
        external_correlation_map: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self.bt_config = bt_config
        self.flags = bt_config.flags
        self.point_value = point_value

        # Use ETF commission rate when trading ETFs
        from dataclasses import replace as _replace
        slippage = bt_config.slippage
        if cfg.is_etf:
            slippage = _replace(slippage,
                                commission_per_contract=slippage.commission_per_share_etf)
        self.broker = SimBroker(slippage_config=slippage)
        self.equity = bt_config.initial_equity

        # Campaign state
        self.campaign = SymbolCampaign()
        self.histories = RollingHistories()
        self.hourly_state = HourlyState()

        # External state for portfolio mode
        self.positions: dict[str, PositionState] = external_positions or {}
        self.circuit_breaker = external_circuit_breaker or CircuitBreakerState()
        self.correlation_map: dict[tuple[str, str], float] = external_correlation_map or {}

        # Position tracking
        self.active_position: _ActivePosition | None = None
        self._entry_a_outstanding: bool = False

        # Daily context cache
        self._daily_ctx: dict | None = None
        self._daily_close_d: float = 0.0

        # Bar index tracking
        self._prev_daily_idx: int = -1
        self._prev_4h_idx: int = -1

        # Slot volume medians for RVOL_H
        self._slot_volumes: dict[tuple[int, int], list[float]] = {}

        # 4H state cache
        self._cached_regime_4h: Regime4H = Regime4H.RANGE_CHOP
        self._cached_atr14_4h: float = 0.0
        self._cached_ema50_4h: float = 0.0
        self._cached_4h_highs: list[float] = []
        self._cached_4h_lows: list[float] = []
        self._regime_history: list[Regime4H] = []  # recent 4H regime labels for stability check

        # Pending setup (between order submission and fill)
        self._pending_setup: SetupInstance | None = None

        # Results
        self.trades: list[BreakoutTradeRecord] = []
        self.signal_events: list[SignalEvent] = []
        self.equity_curve: list[float] = []
        self.timestamps: list = []
        self.total_commission: float = 0.0

        # Diagnostic counters
        self.entries_placed: int = 0
        self.entries_filled: int = 0
        self.entries_expired: int = 0
        self.entries_rejected: int = 0
        self.entries_blocked: int = 0
        self.adds_placed: int = 0
        self.adds_filled: int = 0
        self.campaigns_activated: int = 0
        self.breakouts_qualified: int = 0
        self.dirty_episodes: int = 0
        self.continuations_entered: int = 0
        self.regime_bars_bull: int = 0
        self.regime_bars_bear: int = 0
        self.regime_bars_chop: int = 0

    def run(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
    ) -> BreakoutSymbolResult:
        """Run the full backtest over the hourly bar series."""
        warmup_d = self.bt_config.warmup_daily
        warmup_h = self.bt_config.warmup_hourly
        warmup_4h = self.bt_config.warmup_4h

        with _AblationPatch(self.flags, self.bt_config.param_overrides):
            # Precompute full indicator arrays once (avoids recomputing
            # bounded windows on every bar)
            self._precompute_indicators(daily, hourly, four_hour)

            # Initialize rolling histories from daily warmup period
            self._init_histories(daily, warmup_d)

            # Initialize slot medians from hourly warmup period
            self._init_slot_medians(hourly, warmup_h)

            for t in range(len(hourly)):
                self._step_bar(
                    daily, hourly, four_hour,
                    daily_idx_map, four_hour_idx_map, t,
                    warmup_d, warmup_h, warmup_4h,
                )

            # Close any remaining position at last bar's close
            if self.active_position is not None:
                self._flatten_position(
                    self.active_position,
                    hourly.closes[-1],
                    self._to_datetime(hourly.times[-1]),
                    "END_OF_DATA",
                )

        return BreakoutSymbolResult(
            symbol=self.symbol,
            trades=self.trades,
            signal_events=self.signal_events,
            equity_curve=np.array(self.equity_curve),
            timestamps=np.array(self.timestamps),
            total_commission=self.total_commission,
            entries_placed=self.entries_placed,
            entries_filled=self.entries_filled,
            entries_expired=self.entries_expired,
            entries_rejected=self.entries_rejected,
            entries_blocked=self.entries_blocked,
            adds_placed=self.adds_placed,
            adds_filled=self.adds_filled,
            campaigns_activated=self.campaigns_activated,
            breakouts_qualified=self.breakouts_qualified,
            dirty_episodes=self.dirty_episodes,
            continuations_entered=self.continuations_entered,
            regime_bars_bull=self.regime_bars_bull,
            regime_bars_bear=self.regime_bars_bear,
            regime_bars_chop=self.regime_bars_chop,
        )

    # ------------------------------------------------------------------
    # Indicator precomputation
    # ------------------------------------------------------------------

    def _precompute_indicators(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
    ) -> None:
        """Precompute full-length indicator arrays once before the main loop.

        Avoids recomputing bounded-window ATR/EMA on every bar.
        """
        # Daily indicators
        self._pre_d_atr14 = atr(daily.highs, daily.lows, daily.closes, ATR_DAILY_PERIOD)
        self._pre_d_atr50 = atr(daily.highs, daily.lows, daily.closes, ATR_DAILY_LONG_PERIOD)
        self._pre_d_ema20 = ema(daily.closes, 20)

        # Daily datetime conversions
        self._pre_d_datetimes = [self._to_datetime(daily.times[i]) for i in range(len(daily.times))]

        # Hourly indicators
        self._pre_h_atr14 = atr(hourly.highs, hourly.lows, hourly.closes, ATR_HOURLY_PERIOD)
        self._pre_h_ema20 = ema(hourly.closes, EMA_1H_PERIOD)

        # Hourly datetime conversions (biggest bottleneck: 4ms/bar)
        self._pre_h_datetimes = [self._to_datetime(hourly.times[i]) for i in range(len(hourly.times))]

        # 4H indicators
        if len(four_hour.closes) > 0:
            self._pre_4h_atr14 = atr(four_hour.highs, four_hour.lows, four_hour.closes, ATR_HOURLY_PERIOD)
            self._pre_4h_ema50 = ema(four_hour.closes, EMA_4H_PERIOD)
            self._pre_4h_adx = adx(four_hour.highs, four_hour.lows, four_hour.closes, ADX_PERIOD)
        else:
            self._pre_4h_atr14 = np.array([])
            self._pre_4h_ema50 = np.array([])
            self._pre_4h_adx = np.array([])

    # ------------------------------------------------------------------
    # History initialization
    # ------------------------------------------------------------------

    def _init_histories(self, daily: NumpyBars, warmup_d: int) -> None:
        """Initialize rolling histories from daily warmup bars."""
        if len(daily) < warmup_d:
            return

        closes = daily.closes[:warmup_d]
        highs = daily.highs[:warmup_d]
        lows = daily.lows[:warmup_d]

        # Use precomputed arrays if available, otherwise compute (called externally before run())
        if hasattr(self, '_pre_d_atr14'):
            atr14 = self._pre_d_atr14[:warmup_d]
            atr50 = self._pre_d_atr50[:warmup_d]
        else:
            atr14 = atr(highs, lows, closes, ATR_DAILY_PERIOD)
            atr50 = atr(highs, lows, closes, ATR_DAILY_LONG_PERIOD)

        for i in range(max(ATR_DAILY_LONG_PERIOD, 50), len(closes)):
            L = 12
            rh = float(np.max(highs[max(0, i - L + 1):i + 1]))
            rl = float(np.min(lows[max(0, i - L + 1):i + 1]))
            bh = rh - rl
            sq = bh / float(atr50[i]) if atr50[i] > 0 else 1.0
            self.histories.add_squeeze(sq)

            sma_val = float(np.mean(closes[max(0, i - 20):i + 1]))
            disp = abs(float(closes[i]) - sma_val) / float(atr14[i]) if atr14[i] > 0 else 0.0
            self.histories.add_disp(disp)
            self.histories.add_atr(float(atr14[i]))

    def _init_slot_medians(self, hourly: NumpyBars, warmup_h: int) -> None:
        """Initialize slot volume medians from hourly warmup bars."""
        n = min(warmup_h, len(hourly))
        for i in range(n):
            dt = self._pre_h_datetimes[i] if hasattr(self, '_pre_h_datetimes') else self._to_datetime(hourly.times[i])
            key = get_slot_key(dt)
            self._slot_volumes.setdefault(key, []).append(float(hourly.volumes[i]))
        self.histories.slot_medians = update_slot_medians(self._slot_volumes)

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def _step_bar(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
        t: int,
        warmup_d: int,
        warmup_h: int,
        warmup_4h: int,
    ) -> None:
        """Process one hourly bar."""
        bar_time = self._pre_h_datetimes[t] if hasattr(self, '_pre_h_datetimes') else self._to_datetime(hourly.times[t])
        O = hourly.opens[t]
        H = hourly.highs[t]
        L = hourly.lows[t]
        C = hourly.closes[t]
        V = hourly.volumes[t]

        # Skip NaN bars (carry forward last MTM — no valid price to re-mark)
        if np.isnan(O) or np.isnan(C):
            self.equity_curve.append(self.equity_curve[-1] if self.equity_curve else self.equity)
            self.timestamps.append(hourly.times[t])
            return

        # 1. Process fills from SimBroker
        fills = self.broker.process_bar(
            self.symbol, bar_time, O, H, L, C, self.cfg.tick_size,
        )
        for fill in fills:
            self._handle_fill(fill, bar_time, C)

        # 2. Update daily state on boundary
        d_idx = int(daily_idx_map[t])
        is_daily_boundary = d_idx != self._prev_daily_idx
        self._prev_daily_idx = d_idx

        if is_daily_boundary and d_idx >= warmup_d:
            self._on_daily_close(daily, d_idx, warmup_d)

        # 3. Update 4H state on boundary
        fh_idx = int(four_hour_idx_map[t])
        is_4h_boundary = fh_idx != self._prev_4h_idx
        self._prev_4h_idx = fh_idx

        if is_4h_boundary and fh_idx >= warmup_4h:
            self._update_4h_state(four_hour, fh_idx)

        # 4. Update hourly state
        if t >= warmup_h:
            self._update_hourly_state(hourly, t, bar_time, V)

        # 5. Update active position tracking
        if self.active_position is not None:
            self._manage_active_position(
                bar_time, O, H, L, C, four_hour, fh_idx, is_4h_boundary,
            )

        # Skip signal detection during warmup
        if t < warmup_h or d_idx < warmup_d:
            self.equity_curve.append(self.equity)
            self.timestamps.append(hourly.times[t])
            return

        # 6. Pending mechanism
        if self.campaign.pending and self._daily_ctx:
            self.campaign.pending.rth_hours_elapsed += 1
            if gates.pending_expired(self.campaign.pending, bar_time):
                self.campaign.pending = None
            elif gates.pending_block_cleared(
                self.campaign.pending.reason,
                self.positions,
                self.campaign.pending.direction,
                self._daily_ctx.get("final_risk_dollars", 0.0),
                self.equity,
            ):
                # Re-validate and place
                self._place_entry(
                    self.campaign.pending.entry_type_requested,
                    self.campaign.pending.direction,
                    bar_time,
                )
                self.campaign.pending = None

        # 7. Detect new entries (only if no position and daily context ready)
        elif (self.active_position is None
              and self._daily_ctx
              and self.campaign.state not in (
                  CampaignState.INACTIVE, CampaignState.INVALIDATED,
                  CampaignState.EXPIRED)):
            have_pos = (
                self.symbol in self.positions
                and self.positions[self.symbol].qty > 0
            )
            if have_pos:
                self._handle_adds(bar_time)
            else:
                self._detect_and_place_entry(bar_time)

        # Record mark-to-market equity (realized + unrealized)
        pv = self.point_value
        mtm_equity = self.equity
        if self.active_position is not None:
            pos = self.active_position
            if pos.setup.direction == Direction.LONG:
                mtm_equity += (C - pos.fill_price) * pv * pos.qty_open
            else:
                mtm_equity += (pos.fill_price - C) * pv * pos.qty_open
        self.equity_curve.append(mtm_equity)
        self.timestamps.append(hourly.times[t])

    # ------------------------------------------------------------------
    # Daily close routine (campaign state machine)
    # ------------------------------------------------------------------

    def _on_daily_close(
        self,
        daily: NumpyBars,
        d_idx: int,
        warmup_d: int,
    ) -> None:
        """Per-symbol daily close: campaign state machine, breakout qualification."""
        campaign = self.campaign
        hist = self.histories
        cfg = self.cfg

        end = d_idx + 1
        start = max(0, end - warmup_d)
        closes = daily.closes[start:end]
        highs = daily.highs[start:end]
        lows = daily.lows[start:end]
        volumes = daily.volumes[start:end]

        if len(closes) < ATR_DAILY_LONG_PERIOD + 10:
            return

        # --- 1) Compute daily indicators on bounded slice (preserves EMA seed parity) ---
        atr14_arr = atr(highs, lows, closes, ATR_DAILY_PERIOD)
        atr50_arr = atr(highs, lows, closes, ATR_DAILY_LONG_PERIOD)
        atr14_d = float(atr14_arr[-1])
        atr50_d = float(atr50_arr[-1])
        sma_atr = float(np.mean(atr14_arr[-50:]))
        risk_regime = atr14_d / sma_atr if sma_atr > 0 else 1.0
        atr_expanding = atr14_d > sma_atr

        # Adaptive L with hysteresis
        atr_ratio = atr14_d / atr50_d if atr50_d > 0 else 1.0
        campaign.L = signals.choose_L_with_hysteresis(atr_ratio, campaign)
        L = campaign.L

        # Candidate rolling box
        range_high_roll = float(highest(highs, L)[-1])
        range_low_roll = float(lowest(lows, L)[-1])
        box_height = range_high_roll - range_low_roll
        squeeze_metric = box_height / atr50_d if atr50_d > 0 else 1.0
        containment = float(np.sum(
            (closes[-L:] >= range_low_roll) & (closes[-L:] <= range_high_roll)
        ) / L) if L > 0 else 0.0

        # Update histories
        hist.add_squeeze(squeeze_metric)
        hist.add_atr(atr14_d)
        sq_good, sq_loose = signals.classify_squeeze_tier(
            squeeze_metric, hist.squeeze_hist[:-1]
        )

        # --- Chop mode ---
        ema20_d = ema(closes, 20)
        chop_score = signals.compute_chop_score(atr14_d, hist.atr_hist, closes, ema20_d)
        chop_mode = signals.classify_chop_mode(chop_score)
        if not self.flags.disable_chop_mode:
            campaign.chop_mode = chop_mode.value
        else:
            campaign.chop_mode = ChopMode.NORMAL.value

        # --- 2) Campaign activation / state management ---
        box_active = signals.detect_compression(containment, squeeze_metric, hist.squeeze_hist[:-1])

        # Reset terminal states so new campaigns can start
        if campaign.state in (CampaignState.EXPIRED, CampaignState.INVALIDATED):
            campaign.state = CampaignState.INACTIVE

        if campaign.state == CampaignState.INACTIVE:
            if box_active:
                campaign.state = CampaignState.COMPRESSION
                campaign.campaign_id += 1
                campaign.box_version = 0
                campaign.box_high = range_high_roll
                campaign.box_low = range_low_roll
                campaign.box_height = box_height
                campaign.box_mid = (range_high_roll + range_low_roll) / 2.0
                campaign.anchor_time = gates.next_rth_open(
                    self._to_datetime(daily.times[d_idx])
                )
                campaign.box_age_days = 0
                campaign.reentry_count = {"LONG": {0: 0}, "SHORT": {0: 0}}
                campaign.add_count = 0
                campaign.campaign_risk_used = 0.0
                campaign.pending = None
                self.campaigns_activated += 1
            self._daily_ctx = None
            return

        campaign.box_age_days += 1
        close_today = float(closes[-1])
        self._daily_close_d = close_today

        # --- 3) AVWAP_D ---
        bar_times = self._pre_d_datetimes[start:end] if hasattr(self, '_pre_d_datetimes') else [self._to_datetime(daily.times[start + i]) for i in range(len(closes))]
        avwap_d_arr = compute_avwap(highs, lows, closes, volumes, bar_times, campaign.anchor_time)
        avwap_d = float(avwap_d_arr[-1]) if not np.isnan(avwap_d_arr[-1]) else close_today

        # --- 4) Breakout qualification ---
        direction = signals.check_structural_breakout(close_today, campaign.box_high, campaign.box_low)

        # 4H regime (from cached state)
        regime_4h = self._cached_regime_4h
        daily_slope = compute_daily_slope(closes)

        # Hard block
        if direction and not self.flags.disable_hard_block:
            if gates.hard_block(direction, regime_4h, daily_slope, atr14_d):
                direction = None

        # Chop HALT blocks new breakout qualification
        if (not self.flags.disable_chop_mode
                and chop_mode == ChopMode.HALT
                and direction is not None):
            direction = None

        # Directional filter: block directions that are negative-EV for this symbol
        if direction is not None:
            dir_str = "LONG" if direction == Direction.LONG else "SHORT"
            if dir_str not in cfg.allowed_directions:
                direction = None

        # --- Inside close invalidation ---
        if (campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN)
                and campaign.bars_since_breakout > 0):
            inside = campaign.box_low <= close_today <= campaign.box_high
            if inside:
                campaign.inside_close_count += 1
                if campaign.inside_close_count >= INSIDE_CLOSE_INVALIDATION_COUNT:
                    campaign.state = CampaignState.INVALIDATED
                    self._daily_ctx = None
                    return
            else:
                campaign.inside_close_count = 0

        # --- 5) DIRTY handling ---
        if campaign.state == CampaignState.DIRTY:
            current_date = bar_times[-1].date() if hasattr(bar_times[-1], 'date') else None
            if campaign.dirty_start_date and current_date:
                dirty_duration = (current_date - campaign.dirty_start_date).days
            else:
                dirty_duration = 0

            if not self.flags.disable_dirty:
                should_reset, box_shifted = signals.check_dirty_reset(
                    range_high_roll, range_low_roll,
                    campaign.dirty_high or 0, campaign.dirty_low or 0,
                    atr14_d, squeeze_metric, hist.squeeze_hist[:-1],
                    dirty_duration, campaign.L,
                )
                if should_reset:
                    campaign.state = CampaignState.COMPRESSION
                    campaign.box_high = range_high_roll
                    campaign.box_low = range_low_roll
                    campaign.box_height = box_height
                    campaign.box_mid = (range_high_roll + range_low_roll) / 2.0
                    campaign.anchor_time = gates.next_rth_open(bar_times[-1])
                    if box_shifted:
                        campaign.box_version += 1
                        dir_key = "LONG" if direction == Direction.LONG else "SHORT"
                        campaign.reentry_count.setdefault(dir_key, {})[campaign.box_version] = 0
                    self._daily_ctx = None
                    return

            # Opposite direction while DIRTY
            if direction and direction != campaign.breakout_direction:
                disp_pass, disp, disp_th = signals.check_displacement(
                    close_today, avwap_d, atr14_d, hist.disp_hist[:-1], atr_expanding
                )
                if not self.flags.disable_displacement:
                    if not (disp_pass and signals.dirty_opposite_allowed(disp, disp_th)):
                        self._daily_ctx = None
                        return
            elif direction and direction == campaign.breakout_direction:
                self._daily_ctx = None
                return
            else:
                self._daily_ctx = None
                return

        # Handle breakout failure → DIRTY trigger
        if (not self.flags.disable_dirty
                and campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN)
                and campaign.breakout_direction
                and campaign.bars_since_breakout > 0):
            if signals.check_dirty_trigger(
                close_today, campaign.box_high, campaign.box_low,
                campaign.breakout_direction, campaign.bars_since_breakout, cfg.m_break,
            ):
                campaign.state = CampaignState.DIRTY
                campaign.dirty_start_date = bar_times[-1].date() if hasattr(bar_times[-1], 'date') else None
                campaign.dirty_high = range_high_roll
                campaign.dirty_low = range_low_roll
                campaign.L_at_dirty = campaign.L
                self.dirty_episodes += 1
                self._daily_ctx = None
                return

        if direction is None:
            if campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN,
                                  CampaignState.CONTINUATION):
                campaign.bars_since_breakout += 1
                if campaign.bars_since_breakout > campaign.hard_expiry_bars > 0:
                    campaign.state = CampaignState.EXPIRED
                    self._daily_ctx = None
            return

        # --- 6) Displacement pass ---
        disp_pass, disp, disp_th = signals.check_displacement(
            close_today, avwap_d, atr14_d, hist.disp_hist[:-1], atr_expanding
        )
        hist.add_disp(disp)
        if not self.flags.disable_displacement and not disp_pass:
            self._daily_ctx = None
            return

        # --- 7) Breakout quality reject ---
        open_today = float(daily.opens[d_idx]) if hasattr(daily, 'opens') else close_today
        if not self.flags.disable_breakout_quality_reject:
            if signals.check_breakout_quality_reject(
                float(highs[-1]), float(lows[-1]), open_today, close_today, atr14_d, direction
            ):
                self._daily_ctx = None
                return

        # --- 8) Evidence score ---
        rvol_d = compute_rvol_d(volumes)
        score_detail = signals.compute_evidence_score_detailed(
            rvol_d, disp, disp_th, disp_pass, sq_good, sq_loose,
            regime_4h, direction, atr_expanding, closes,
            campaign.box_high, campaign.box_low,
        )
        score_total = score_detail["total"]
        vol_score = score_detail["vol"]

        # --- 8b) Score threshold gating (spec §9) ---
        score_threshold = SCORE_THRESHOLD_NORMAL
        if chop_mode == ChopMode.DEGRADED:
            score_threshold = SCORE_THRESHOLD_DEGRADED
        if regime_4h == Regime4H.RANGE_CHOP:
            score_threshold = max(score_threshold, SCORE_THRESHOLD_RANGE)
        if not self.flags.disable_score_threshold:
            if score_total < score_threshold:
                if campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN,
                                      CampaignState.CONTINUATION):
                    campaign.bars_since_breakout += 1
                    campaign.expiry_mult = signals.compute_expiry_mult(
                        campaign.bars_since_breakout, campaign.expiry_bars
                    )
                    if campaign.bars_since_breakout > campaign.hard_expiry_bars > 0:
                        campaign.state = CampaignState.EXPIRED
                self._daily_ctx = None
                return

        # --- 8c) Regime chop block ---
        if (not self.flags.disable_regime_chop_block
                and REGIME_CHOP_BLOCK
                and regime_4h == Regime4H.RANGE_CHOP
                and score_total < REGIME_CHOP_SCORE_OVERRIDE):
            if campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN,
                                  CampaignState.CONTINUATION):
                campaign.bars_since_breakout += 1
                campaign.expiry_mult = signals.compute_expiry_mult(
                    campaign.bars_since_breakout, campaign.expiry_bars
                )
                if campaign.bars_since_breakout > campaign.hard_expiry_bars > 0:
                    campaign.state = CampaignState.EXPIRED
            self._daily_ctx = None
            return

        # --- 9) Expiry ---
        expiry_bars, hard_expiry_bars = signals.compute_expiry_bars(atr14_d, hist.atr_hist)
        campaign.expiry_bars = expiry_bars
        campaign.hard_expiry_bars = hard_expiry_bars

        # --- 10) Continuation mode ---
        disp_mult = allocator.compute_disp_mult(disp, hist.disp_hist)
        if not self.flags.disable_continuation:
            campaign.continuation = signals.check_continuation_mode(
                close_today, campaign.box_high, campaign.box_low, campaign.box_height,
                atr14_d, direction, campaign.bars_since_breakout, regime_4h, disp_mult,
            )
        else:
            campaign.continuation = False

        # --- 11) Update breakout state ---
        if campaign.state in (CampaignState.COMPRESSION, CampaignState.BREAKOUT):
            campaign.state = CampaignState.BREAKOUT
            campaign.breakout_direction = direction
            campaign.breakout_date = bar_times[-1].date() if hasattr(bar_times[-1], 'date') else None
            campaign.bars_since_breakout = 0
            campaign.expiry_mult = 1.0
            campaign.inside_close_count = 0
            self.breakouts_qualified += 1
        else:
            campaign.bars_since_breakout += 1
            campaign.expiry_mult = signals.compute_expiry_mult(
                campaign.bars_since_breakout, campaign.expiry_bars
            )

        if (campaign.continuation
                and campaign.state == CampaignState.BREAKOUT
                and campaign.bars_since_breakout > 0):
            campaign.state = CampaignState.CONTINUATION
            self.continuations_entered += 1

        # --- 12) Cache daily context for hourly engine ---
        trade_regime = signals.determine_trade_regime(direction, regime_4h)
        quality_mult = allocator.compute_quality_mult(
            direction, regime_4h, disp, hist.disp_hist, sq_good, sq_loose,
            self.symbol, self.positions, self.correlation_map, score_total,
            regime_at_entry=regime_4h.value,
        )
        # Quality mult sub-components for diagnostics
        regime_mult = allocator.compute_regime_mult(direction, regime_4h)
        squeeze_mult = allocator.compute_squeeze_mult(sq_good, sq_loose)
        corr_mult = allocator.compute_corr_mult(
            direction, self.symbol, self.positions, self.correlation_map,
            regime_at_entry=regime_4h.value,
        )

        # Compute final risk
        expiry_mult = campaign.expiry_mult
        cb_ok, cb_reason, cb_mult = gates.circuit_breaker_check(self.circuit_breaker)
        if not cb_ok:
            self._daily_ctx = None
            return

        if campaign.chop_mode == "DEGRADED":
            cb_mult *= CHOP_DEGRADED_SIZE_MULT

        final_risk_dollars = allocator.compute_final_risk(
            self.equity, cfg.base_risk_pct, risk_regime,
            quality_mult, expiry_mult, cb_mult,
        )

        self._daily_ctx = {
            "direction": direction,
            "disp": disp,
            "disp_th": disp_th,
            "disp_mult": disp_mult,
            "displacement_pass": disp_pass,
            "rvol_d": rvol_d,
            "vol_score": vol_score,
            "sq_good": sq_good,
            "sq_loose": sq_loose,
            "atr_expanding": atr_expanding,
            "atr14_d": atr14_d,
            "atr50_d": atr50_d,
            "risk_regime": risk_regime,
            "regime_4h": regime_4h,
            "quality_mult": quality_mult,
            "score_total": score_total,
            "trade_regime": trade_regime,
            "chop_mode": chop_mode.value,
            "final_risk_dollars": final_risk_dollars,
            "cb_mult": cb_mult,
            "expiry_mult": expiry_mult,
            # Score sub-components (Phase 2)
            "score_vol": score_detail["vol"],
            "score_squeeze": score_detail["squeeze"],
            "score_regime": score_detail["regime"],
            "score_consec": score_detail["consec"],
            "score_atr": score_detail["atr"],
            # Quality mult sub-components (Phase 2)
            "regime_mult": regime_mult,
            "squeeze_mult": squeeze_mult,
            "corr_mult": corr_mult,
        }

        # --- Breakout-day entry: bypass hourly A/B/C ---
        if (self.bt_config.breakout_day_entry
                and campaign.state == CampaignState.BREAKOUT
                and campaign.bars_since_breakout == 0
                and self.active_position is None):
            self._place_breakout_day_entry(direction, bar_times[-1])

    # ------------------------------------------------------------------
    # 4H state update
    # ------------------------------------------------------------------

    def _update_4h_state(self, four_hour: NumpyBars, fh_idx: int) -> None:
        """Update 4H regime classification and cache for trailing stop."""
        lookback = min(fh_idx + 1, 200)
        start = fh_idx + 1 - lookback

        closes = four_hour.closes[start:fh_idx + 1]
        highs_arr = four_hour.highs[start:fh_idx + 1]
        lows_arr = four_hour.lows[start:fh_idx + 1]

        if len(closes) < EMA_4H_PERIOD + 4:
            return

        # Inline regime classification using precomputed arrays (saves ~31s from ADX)
        ema_val = float(self._pre_4h_ema50[fh_idx])
        atr_val = float(self._pre_4h_atr14[fh_idx])
        adx_val = float(self._pre_4h_adx[fh_idx])
        ema_prev3 = float(self._pre_4h_ema50[fh_idx - 3]) if fh_idx >= 3 else ema_val
        slope_4h = ema_val - ema_prev3
        slope_th = 0.10 * atr_val
        price = float(four_hour.closes[fh_idx])

        if adx_val < 20:
            regime_4h = Regime4H.RANGE_CHOP
        elif slope_4h > slope_th and price > ema_val:
            regime_4h = Regime4H.BULL_TREND
        elif slope_4h < -slope_th and price < ema_val:
            regime_4h = Regime4H.BEAR_TREND
        else:
            regime_4h = Regime4H.RANGE_CHOP

        self._cached_regime_4h = regime_4h
        self._regime_history.append(regime_4h)
        if len(self._regime_history) > 10:
            self._regime_history = self._regime_history[-10:]

        # Track regime distribution
        if regime_4h == Regime4H.BULL_TREND:
            self.regime_bars_bull += 1
        elif regime_4h == Regime4H.BEAR_TREND:
            self.regime_bars_bear += 1
        else:
            self.regime_bars_chop += 1

        # Use precomputed arrays for trailing stop cache
        self._cached_atr14_4h = float(self._pre_4h_atr14[fh_idx])
        self._cached_ema50_4h = float(self._pre_4h_ema50[fh_idx])
        self._cached_4h_highs = [float(h) for h in highs_arr[-20:]]
        self._cached_4h_lows = [float(l) for l in lows_arr[-20:]]

    # ------------------------------------------------------------------
    # Hourly state update
    # ------------------------------------------------------------------

    def _update_hourly_state(
        self,
        hourly: NumpyBars,
        t: int,
        bar_time: datetime,
        volume: float,
    ) -> None:
        """Update hourly indicators: ATR14_H, AVWAP_H, WVWAP, EMA20_H, RVOL_H."""
        hs = self.hourly_state
        lookback = min(t + 1, 200)
        start = t + 1 - lookback

        h_closes = hourly.closes[start:t + 1]
        h_highs = hourly.highs[start:t + 1]
        h_lows = hourly.lows[start:t + 1]
        h_volumes = hourly.volumes[start:t + 1]

        # ATR14_H — use precomputed
        if t >= ATR_HOURLY_PERIOD:
            hs.atr14_h = float(self._pre_h_atr14[t])

        # EMA20_H — use precomputed
        if t >= EMA_1H_PERIOD:
            hs.ema20_h = float(self._pre_h_ema20[t])

        # AVWAP_H (campaign-anchored) — cannot precompute, depends on anchor_time
        if self.campaign.anchor_time:
            h_times = self._pre_h_datetimes[start:t + 1] if hasattr(self, '_pre_h_datetimes') else [self._to_datetime(hourly.times[start + i]) for i in range(lookback)]
            avwap_arr = compute_avwap(h_highs, h_lows, h_closes, h_volumes,
                                       h_times, self.campaign.anchor_time)
            val = float(avwap_arr[-1])
            hs.avwap_h = val if not np.isnan(val) else float(h_closes[-1])

            # WVWAP
            wvwap_arr = compute_wvwap(h_highs, h_lows, h_closes, h_volumes, h_times)
            hs.wvwap = float(wvwap_arr[-1])

        hs.close = float(h_closes[-1])
        hs.bar_time = bar_time

        # Rolling data for signals
        hs.closes = [float(c) for c in h_closes[-200:]]
        hs.highs = [float(h) for h in h_highs[-200:]]
        hs.lows = [float(l) for l in h_lows[-200:]]

        # RVOL_H (slot-normalized)
        key = get_slot_key(bar_time)
        self._slot_volumes.setdefault(key, []).append(float(volume))
        # Periodically update medians (every 50 bars)
        if t % 50 == 0:
            self.histories.slot_medians = update_slot_medians(self._slot_volumes)
        hs.rvol_h = compute_rvol_h(float(volume), key, self.histories.slot_medians)

    # ------------------------------------------------------------------
    # Entry detection and placement
    # ------------------------------------------------------------------

    def _detect_and_place_entry(self, bar_time: datetime) -> None:
        """Detect entry signal (A→B→C priority) and place order."""
        if self.bt_config.breakout_day_entry:
            return  # initial entries handled in _on_daily_close; adds still work via _handle_adds

        ctx = self._daily_ctx
        if ctx is None:
            return

        campaign = self.campaign
        cfg = self.cfg
        hs = self.hourly_state

        direction: Direction = ctx["direction"]
        regime_4h: Regime4H = ctx["regime_4h"]
        disp_mult = ctx["disp_mult"]
        quality_mult = ctx["quality_mult"]
        final_risk_dollars = ctx["final_risk_dollars"]
        trade_regime: TradeRegime = ctx["trade_regime"]
        atr14_d = ctx["atr14_d"]
        atr14_h = hs.atr14_h
        avwap_h = hs.avwap_h
        rvol_h = hs.rvol_h
        expiry_mult = ctx["expiry_mult"]

        # Chop HALT blocks all new entries
        if not self.flags.disable_chop_mode and campaign.chop_mode == "HALT":
            return

        # Re-entry gate
        if not self.flags.disable_reentry_gate:
            if campaign.last_exit_date and campaign.last_exit_direction:
                current_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time
                days_since_exit = (current_date - campaign.last_exit_date).days
                if not gates.reentry_allowed(
                    direction, campaign, campaign.last_exit_realized_r, days_since_exit,
                ):
                    return

        # Entry selection
        # Ablation: disable C_continuation entry by temporarily clearing continuation
        saved_continuation = campaign.continuation
        if self.flags.disable_c_continuation_entry:
            campaign.continuation = False

        entry_type = signals.select_entry_type(
            direction=direction,
            campaign=campaign,
            hourly_close=hs.close,
            hourly_low=float(hs.lows[-1]) if hs.lows else hs.close,
            hourly_high=float(hs.highs[-1]) if hs.highs else hs.close,
            hourly_closes=hs.closes,
            hourly_highs=hs.highs,
            hourly_lows=hs.lows,
            avwap_h=avwap_h,
            atr14_h=atr14_h,
            atr14_d=atr14_d,
            rvol_h=rvol_h,
            disp_mult=disp_mult,
            quality_mult=quality_mult,
            regime_4h=regime_4h,
            entry_a_active=self._entry_a_outstanding,
            atr_expanding=ctx.get("atr_expanding", True),
            ema20_h=hs.ema20_h,
        )

        # Restore continuation state after entry selection
        campaign.continuation = saved_continuation

        if entry_type is None:
            return

        # Displacement ceiling: reject C_standard entries too far from AVWAP
        if entry_type in (EntryType.C_STANDARD, EntryType.C_CONTINUATION) and atr14_d > 0:
            from strategy_3.config import C_ENTRY_MAX_DISP_H
            disp_from_avwap = abs(hs.close - avwap_h) / atr14_d
            if disp_from_avwap > C_ENTRY_MAX_DISP_H:
                return

        # Full eligibility check
        ok, reason, _ = gates.full_eligibility_check(
            self.symbol, direction, campaign, cfg, bar_time,
            self.circuit_breaker, self.positions,
            final_risk_dollars, self.equity,
            [],  # No news calendar in backtest
            self.correlation_map,
        )

        if not ok:
            # Log signal event
            if self.bt_config.track_signals:
                self._log_signal_event(
                    bar_time, direction, entry_type, False, reason,
                    ctx, rvol_h,
                )
            # Pending mechanism
            if not self.flags.disable_pending:
                block_reason = gates.check_transient_blocks(
                    self.symbol, direction, self.positions,
                    final_risk_dollars, self.equity, self.correlation_map,
                )
                if block_reason:
                    campaign.pending = PendingEntry(
                        created_ts=bar_time,
                        direction=direction,
                        entry_type_requested=entry_type,
                        reason=block_reason,
                    )
            self.entries_blocked += 1
            return

        # Micro guard
        if not self.flags.disable_micro_guard:
            base_risk_adj = allocator.compute_risk_regime_adj(cfg.base_risk_pct, ctx["risk_regime"])
            final_risk_pct = final_risk_dollars / self.equity if self.equity > 0 else 0.0
            if not allocator.micro_guard_ok(expiry_mult, disp_mult, trade_regime, final_risk_pct, base_risk_adj):
                if self.bt_config.track_signals:
                    self._log_signal_event(
                        bar_time, direction, entry_type, False, "micro_guard",
                        ctx, rvol_h,
                    )
                self.entries_blocked += 1
                return

        # Friction gate
        if not self.flags.disable_friction_gate:
            stop_price = stops.compute_initial_stop(
                direction, entry_type.value,
                campaign.box_high, campaign.box_low, campaign.box_mid,
                atr14_d, cfg.atr_stop_mult, ctx["sq_good"], cfg.tick_size,
            )
            shares_est = allocator.compute_shares(final_risk_dollars, avwap_h, stop_price, cfg.fee_bps_est)
            if shares_est > 0 and not gates.friction_gate(self.symbol, cfg.fee_bps_est, avwap_h, shares_est, final_risk_dollars):
                if self.bt_config.track_signals:
                    self._log_signal_event(
                        bar_time, direction, entry_type, False, "friction_gate",
                        ctx, rvol_h,
                    )
                self.entries_blocked += 1
                return

        # Place entry
        if self.bt_config.track_signals:
            self._log_signal_event(
                bar_time, direction, entry_type, True, "",
                ctx, rvol_h,
            )
        self._place_entry(entry_type, direction, bar_time)

    def _place_entry(
        self,
        entry_type: EntryType,
        direction: Direction,
        bar_time: datetime,
    ) -> None:
        """Compute stop, size, and submit entry order via SimBroker."""
        ctx = self._daily_ctx
        if ctx is None:
            return

        campaign = self.campaign
        cfg = self.cfg
        hs = self.hourly_state

        atr14_d = ctx["atr14_d"]
        atr14_h = hs.atr14_h
        avwap_h = hs.avwap_h
        final_risk_dollars = ctx["final_risk_dollars"]
        quality_mult = ctx["quality_mult"]
        trade_regime: TradeRegime = ctx["trade_regime"]
        sq_good = ctx["sq_good"]

        # Stop
        stop_price = stops.compute_initial_stop(
            direction, entry_type.value,
            campaign.box_high, campaign.box_low, campaign.box_mid,
            atr14_d, cfg.atr_stop_mult, sq_good, cfg.tick_size,
        )

        # Entry level
        if entry_type == EntryType.A_AVWAP_RETEST:
            buffer = 0.03 * atr14_d
            entry_price = avwap_h + buffer if direction == Direction.LONG else avwap_h - buffer
        elif entry_type in (EntryType.C_STANDARD, EntryType.C_CONTINUATION):
            # C entries: signal confirmed via 2 consecutive closes — enter near market
            entry_price = hs.close
        else:
            entry_price = avwap_h

        entry_price = round_to_tick(entry_price, cfg.tick_size,
                                     "up" if direction == Direction.LONG else "down")

        # Shares
        shares = allocator.compute_shares(final_risk_dollars, entry_price, stop_price, cfg.fee_bps_est)
        if shares <= 0:
            return
        shares = min(shares, cfg.max_shares)

        if self.bt_config.fixed_qty is not None:
            shares = self.bt_config.fixed_qty

        # R per share
        r_per_share = abs(entry_price - stop_price)
        if r_per_share <= 0:
            return

        # Correct risk dollars to match actual position risk under fixed_qty
        if self.bt_config.fixed_qty is not None:
            final_risk_dollars = shares * r_per_share

        # TP levels
        tp1_r, tp2_r = self._get_tp_r_multiples(trade_regime, self.cfg.tp_scale)

        # Build setup instance
        setup = SetupInstance(
            symbol=self.symbol,
            direction=direction,
            entry_type=entry_type,
            state=SetupState.ARMED,
            created_ts=bar_time,
            armed_ts=bar_time,
            campaign_id=campaign.campaign_id,
            box_version=campaign.box_version,
            entry_price=entry_price,
            stop0=stop_price,
            buffer=0.03 * atr14_d,
            final_risk_dollars=final_risk_dollars,
            quality_mult=quality_mult,
            expiry_mult=campaign.expiry_mult,
            shares_planned=shares,
            exit_tier=ExitTier(trade_regime.value),
            regime_at_entry=ctx["regime_4h"].value if isinstance(ctx["regime_4h"], Regime4H) else str(ctx["regime_4h"]),
            r_price=r_per_share,
            current_stop=stop_price,
        )

        # Determine order type and TTL
        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL

        if entry_type == EntryType.A_AVWAP_RETEST:
            order_type = OrderType.MARKET
            ttl_hours = 2  # needs >1 so MARKET order survives to next bar
            self._entry_a_outstanding = True
        elif entry_type == EntryType.B_SWEEP_RECLAIM:
            order_type = OrderType.MARKET
            ttl_hours = 1  # immediate
        elif entry_type in (EntryType.C_STANDARD, EntryType.C_CONTINUATION):
            order_type = OrderType.LIMIT
            ttl_hours = ENTRY_C_TTL_RTH_HOURS
        else:
            order_type = OrderType.MARKET
            ttl_hours = 1

        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            qty=shares,
            stop_price=0.0,
            limit_price=entry_price if order_type == OrderType.LIMIT else 0.0,
            tick_size=cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=ttl_hours,
            tag="entry",
        )

        self.broker.submit_order(order)
        self._pending_setup = setup
        self.entries_placed += 1

    # ------------------------------------------------------------------
    # Breakout-day entry (bypass hourly A/B/C)
    # ------------------------------------------------------------------

    def _place_breakout_day_entry(self, direction: Direction, daily_bar_time: datetime) -> None:
        """Place entry at box boundary on breakout day, bypassing hourly signals."""
        ctx = self._daily_ctx
        if ctx is None:
            return

        campaign = self.campaign
        cfg = self.cfg

        atr14_d = ctx["atr14_d"]
        final_risk_dollars = ctx["final_risk_dollars"]
        trade_regime: TradeRegime = ctx["trade_regime"]
        sq_good = ctx["sq_good"]

        # Entry at box boundary (the breakout level)
        if direction == Direction.LONG:
            entry_price = campaign.box_high
        else:
            entry_price = campaign.box_low
        entry_price = round_to_tick(entry_price, cfg.tick_size,
                                     "up" if direction == Direction.LONG else "down")

        # Stop at box_mid - buffer (same as existing logic)
        stop_price = stops.compute_initial_stop(
            direction, "BREAKOUT_DAY",
            campaign.box_high, campaign.box_low, campaign.box_mid,
            atr14_d, cfg.atr_stop_mult, sq_good, cfg.tick_size,
        )

        # Shares
        shares = allocator.compute_shares(final_risk_dollars, entry_price, stop_price, cfg.fee_bps_est)
        if shares <= 0:
            return
        shares = min(shares, cfg.max_shares)
        if self.bt_config.fixed_qty is not None:
            shares = self.bt_config.fixed_qty

        r_per_share = abs(entry_price - stop_price)
        if r_per_share <= 0:
            return

        # Correct risk dollars to match actual position risk under fixed_qty
        if self.bt_config.fixed_qty is not None:
            final_risk_dollars = shares * r_per_share

        tp1_r, tp2_r = self._get_tp_r_multiples(trade_regime, cfg.tp_scale)

        setup = SetupInstance(
            symbol=self.symbol,
            direction=direction,
            entry_type=EntryType.BREAKOUT_DAY,
            state=SetupState.ARMED,
            created_ts=daily_bar_time,
            armed_ts=daily_bar_time,
            campaign_id=campaign.campaign_id,
            box_version=campaign.box_version,
            entry_price=entry_price,
            stop0=stop_price,
            buffer=0.0,
            final_risk_dollars=final_risk_dollars,
            quality_mult=ctx["quality_mult"],
            expiry_mult=campaign.expiry_mult,
            shares_planned=shares,
            exit_tier=ExitTier(trade_regime.value),
            regime_at_entry=ctx["regime_4h"].value if isinstance(ctx["regime_4h"], Regime4H) else str(ctx["regime_4h"]),
            r_price=r_per_share,
            current_stop=stop_price,
        )

        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,  # fills at next bar open
            qty=shares,
            tick_size=cfg.tick_size,
            submit_time=daily_bar_time,
            ttl_hours=24,
            tag="entry",
        )
        self.broker.submit_order(order)
        self._pending_setup = setup
        self.entries_placed += 1

    # ------------------------------------------------------------------
    # Add handling
    # ------------------------------------------------------------------

    def _handle_adds(self, bar_time: datetime) -> None:
        """Handle pullback add entries for active position."""
        pos = self.active_position
        if pos is None:
            return

        ctx = self._daily_ctx
        if ctx is None:
            return

        campaign = self.campaign
        cfg = self.cfg
        hs = self.hourly_state
        direction: Direction = ctx["direction"]
        atr14_d = ctx["atr14_d"]
        atr14_h = hs.atr14_h
        avwap_h = hs.avwap_h
        rvol_h = hs.rvol_h
        final_risk_dollars = ctx["final_risk_dollars"]
        regime_4h: Regime4H = ctx["regime_4h"]

        # Eligibility
        if campaign.add_count >= MAX_ADDS_PER_CAMPAIGN:
            return
        if not (pos.tp1_done or pos.runner_active):
            return
        if not signals.is_regime_aligned(direction, regime_4h):
            return

        initial_risk = pos.setup.final_risk_dollars if pos.setup.final_risk_dollars > 0 else final_risk_dollars
        add_risk = allocator.compute_add_risk_dollars(initial_risk, rvol_h)

        if not self.flags.disable_add_rvol_acceptance:
            if not allocator.add_budget_ok(campaign, initial_risk, add_risk):
                return

        # Pullback ref
        ref = pullback_ref(hs.close, hs.wvwap, avwap_h, hs.ema20_h, atr14_d)

        # Touch + resume check
        if len(hs.closes) < 2:
            return
        reclaim_buffer = max(0.12 * atr14_h, 0.03 * atr14_d)
        triggered = signals.check_add_trigger(
            direction, hs.close,
            float(hs.lows[-1]) if hs.lows else hs.close,
            float(hs.highs[-1]) if hs.highs else hs.close,
            ref, float(hs.closes[-2]),
            rvol_h,
            high_vol_regime=ctx.get("risk_regime", 1.0) > 1.2,
            reclaim_buffer=reclaim_buffer,
        )
        if not triggered:
            return

        # CLV acceptance for low-volume adds
        if rvol_h < 0.8:
            if not signals.strong_close_location(
                direction,
                float(hs.highs[-1]) if hs.highs else hs.close,
                float(hs.lows[-1]) if hs.lows else hs.close,
                hs.close,
            ):
                return

        # Add stop
        pullback_low = min(hs.lows[-3:]) if len(hs.lows) >= 3 else (hs.lows[-1] if hs.lows else hs.close)
        pullback_high = max(hs.highs[-3:]) if len(hs.highs) >= 3 else (hs.highs[-1] if hs.highs else hs.close)
        add_stop = stops.compute_add_stop(
            direction, pullback_low, pullback_high,
            ref, atr14_d, cfg.atr_stop_mult, cfg.tick_size,
        )

        # Size
        add_shares = allocator.compute_shares(add_risk, ref, add_stop, cfg.fee_bps_est)
        if add_shares <= 0:
            return

        # Friction gate
        if not self.flags.disable_friction_gate:
            if not gates.friction_gate(self.symbol, cfg.fee_bps_est, ref, add_shares, add_risk):
                return

        # Submit add order (LIMIT at pullback ref)
        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        limit_price = round_to_tick(ref, cfg.tick_size,
                                     "up" if direction == Direction.LONG else "down")

        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            qty=add_shares,
            limit_price=limit_price,
            tick_size=cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=ENTRY_A_TTL_RTH_HOURS,
            tag="add_on",
        )
        self.broker.submit_order(order)
        campaign.add_count += 1
        campaign.campaign_risk_used += add_risk
        self.adds_placed += 1

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _handle_fill(self, fill: FillResult, bar_time: datetime, close: float) -> None:
        """Route a fill result to the appropriate handler."""
        if fill.status == FillStatus.EXPIRED:
            if fill.order.tag == "entry":
                self._pending_setup = None
                self._entry_a_outstanding = False
                self.entries_expired += 1
            return

        if fill.status == FillStatus.REJECTED:
            if fill.order.tag == "entry":
                self._pending_setup = None
                self._entry_a_outstanding = False
                self.entries_rejected += 1
            return

        if fill.status != FillStatus.FILLED:
            return

        if fill.order.tag == "entry":
            self._on_entry_fill(fill, bar_time)
        elif fill.order.tag == "protective_stop":
            self._on_stop_fill(fill, bar_time)
        elif fill.order.tag == "partial":
            self._on_partial_fill(fill, bar_time)
        elif fill.order.tag == "add_on":
            self._on_add_fill(fill, bar_time)

    def _on_entry_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle entry fill: create position, place protective stop."""
        setup = self._pending_setup
        if setup is None:
            return

        self._pending_setup = None
        self._entry_a_outstanding = False
        self.entries_filled += 1
        self.total_commission += fill.commission
        self.equity -= fill.commission

        ctx = self._daily_ctx or {}

        pos = _ActivePosition(
            setup=setup,
            fill_price=fill.fill_price,
            qty_open=fill.order.qty,
            qty_initial=fill.order.qty,
            initial_stop=setup.stop0,
            current_stop=setup.stop0,
            r_price=setup.r_price,
            entry_time=bar_time,
            mfe_price=fill.fill_price,
            mae_price=fill.fill_price,
            regime_at_entry=setup.regime_at_entry or "",
            exit_tier=setup.exit_tier.value if isinstance(setup.exit_tier, ExitTier) else str(setup.exit_tier),
            entry_type=setup.entry_type.value if isinstance(setup.entry_type, EntryType) else str(setup.entry_type),
            disp_at_entry=ctx.get("disp", 0.0),
            disp_th_at_entry=ctx.get("disp_th", 0.0),
            rvol_d_at_entry=ctx.get("rvol_d", 1.0),
            sq_good_at_entry=ctx.get("sq_good", False),
            quality_mult_at_entry=ctx.get("quality_mult", 1.0),
            score_at_entry=ctx.get("score_total", 0),
            # Score sub-components (Phase 2)
            score_vol_at_entry=ctx.get("score_vol", 0),
            score_squeeze_at_entry=ctx.get("score_squeeze", 0),
            score_regime_at_entry=ctx.get("score_regime", 0),
            score_consec_at_entry=ctx.get("score_consec", 0),
            score_atr_at_entry=ctx.get("score_atr", 0),
            # Quality mult sub-components (Phase 2)
            regime_mult_at_entry=ctx.get("regime_mult", 1.0),
            disp_mult_at_entry=ctx.get("disp_mult", 1.0),
            squeeze_mult_at_entry=ctx.get("squeeze_mult", 1.0),
            corr_mult_at_entry=ctx.get("corr_mult", 1.0),
            chop_mode_at_entry=ctx.get("chop_mode", ""),
            continuation_at_entry=self.campaign.continuation,
            commission=fill.commission,
        )

        # Compute friction
        pos.friction_dollars = gates.compute_friction_dollars(
            self.cfg.fee_bps_est, fill.fill_price, fill.order.qty,
        )

        self.active_position = pos

        # Update campaign state
        if self.campaign.state == CampaignState.BREAKOUT:
            self.campaign.state = CampaignState.POSITION_OPEN

        # Track re-entry
        if self.campaign.last_exit_date is not None:
            dir_key = "LONG" if setup.direction == Direction.LONG else "SHORT"
            bv = self.campaign.box_version
            self.campaign.reentry_count.setdefault(dir_key, {})
            self.campaign.reentry_count[dir_key][bv] = self.campaign.reentry_count[dir_key].get(bv, 0) + 1
            self.campaign.last_exit_date = None

        # Update position state (for portfolio mode)
        ps = self.positions.setdefault(
            self.symbol,
            PositionState(
                symbol=self.symbol,
                direction=setup.direction,
                campaign_id=self.campaign.campaign_id,
                box_version=self.campaign.box_version,
            ),
        )
        ps.qty += fill.order.qty
        ps.avg_cost = fill.fill_price
        ps.current_stop = setup.stop0
        ps.total_risk_dollars += setup.final_risk_dollars
        ps.direction = setup.direction
        if ps.regime_at_entry is None:
            ps.regime_at_entry = setup.regime_at_entry

        # Place protective stop
        stop_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
        stop_order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=stop_side,
            order_type=OrderType.STOP,
            qty=fill.order.qty,
            stop_price=setup.stop0,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="protective_stop",
        )
        self.broker.submit_order(stop_order)

    def _on_stop_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle protective stop fill."""
        pos = self.active_position
        if pos is None:
            return

        self.total_commission += fill.commission
        self.equity -= fill.commission

        setup = pos.setup
        if setup.direction == Direction.LONG:
            pnl_points = fill.fill_price - pos.fill_price
        else:
            pnl_points = pos.fill_price - fill.fill_price

        pv = self.point_value
        pnl_dollars = pnl_points * pv * pos.qty_open + pos.realized_pnl
        r_multiple = pnl_dollars / setup.final_risk_dollars if setup.final_risk_dollars > 0 else 0.0

        r_base = pos.r_price
        if r_base > 0:
            if setup.direction == Direction.LONG:
                mfe_r = (pos.mfe_price - pos.fill_price) / r_base
                mae_r = (pos.fill_price - pos.mae_price) / r_base
            else:
                mfe_r = (pos.fill_price - pos.mfe_price) / r_base
                mae_r = (pos.mae_price - pos.fill_price) / r_base
        else:
            mfe_r = mae_r = 0.0

        self.equity += pnl_points * pv * pos.qty_open

        self._record_trade(pos, fill.fill_price, bar_time, "STOP", pnl_points,
                           pnl_dollars, r_multiple, mfe_r, mae_r,
                           fill.commission)

        self._update_circuit_breaker(r_multiple, bar_time)
        self.broker.cancel_all(self.symbol)
        self._clear_position()

    def _on_partial_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle partial exit fill (TP1/TP2)."""
        pos = self.active_position
        if pos is None:
            return

        self.total_commission += fill.commission
        self.equity -= fill.commission
        pos.commission += fill.commission

        pv = self.point_value
        if pos.setup.direction == Direction.LONG:
            partial_pnl = (fill.fill_price - pos.fill_price) * pv * fill.order.qty
        else:
            partial_pnl = (pos.fill_price - fill.fill_price) * pv * fill.order.qty

        pos.realized_pnl += partial_pnl
        self.equity += partial_pnl
        pos.qty_open -= fill.order.qty

        # If partial exit closed entire position, record trade and clear
        if pos.qty_open <= 0:
            setup = pos.setup
            if setup.direction == Direction.LONG:
                pnl_points = fill.fill_price - pos.fill_price
            else:
                pnl_points = pos.fill_price - fill.fill_price
            pnl_dollars = pos.realized_pnl
            r_multiple = pnl_dollars / setup.final_risk_dollars if setup.final_risk_dollars > 0 else 0.0
            r_base = pos.r_price
            if r_base > 0:
                if setup.direction == Direction.LONG:
                    mfe_r = (pos.mfe_price - pos.fill_price) / r_base
                    mae_r = (pos.fill_price - pos.mae_price) / r_base
                else:
                    mfe_r = (pos.fill_price - pos.mfe_price) / r_base
                    mae_r = (pos.mae_price - pos.fill_price) / r_base
            else:
                mfe_r = mae_r = 0.0
            exit_reason = "TP1" if not pos.tp2_done or pos.tp2_bar == pos.tp1_bar else "TP2"
            self._record_trade(pos, fill.fill_price, bar_time, exit_reason,
                               pnl_points, pnl_dollars, r_multiple, mfe_r, mae_r,
                               fill.commission)
            self._update_circuit_breaker(r_multiple, bar_time)
            self.broker.cancel_all(self.symbol)
            self._clear_position()
            return

        # Replace protective stop with updated qty
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        stop_side = OrderSide.SELL if pos.setup.direction == Direction.LONG else OrderSide.BUY
        stop_order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=stop_side,
            order_type=OrderType.STOP,
            qty=pos.qty_open,
            stop_price=pos.current_stop,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="protective_stop",
        )
        self.broker.submit_order(stop_order)

        # Update portfolio position state
        ps = self.positions.get(self.symbol)
        if ps:
            ps.qty = max(0, ps.qty - fill.order.qty)

    def _on_add_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle add-on entry fill."""
        pos = self.active_position
        if pos is None:
            return

        self.total_commission += fill.commission
        self.equity -= fill.commission
        pos.commission += fill.commission
        pos.qty_open += fill.order.qty
        pos.add_count += 1
        self.adds_filled += 1

        # Update protective stop to include add-on qty
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        stop_side = OrderSide.SELL if pos.setup.direction == Direction.LONG else OrderSide.BUY
        stop_order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=stop_side,
            order_type=OrderType.STOP,
            qty=pos.qty_open,
            stop_price=pos.current_stop,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="protective_stop",
        )
        self.broker.submit_order(stop_order)

        # Update portfolio position state
        ps = self.positions.get(self.symbol)
        if ps:
            ps.qty += fill.order.qty

    # ------------------------------------------------------------------
    # Active position management
    # ------------------------------------------------------------------

    def _manage_active_position(
        self,
        bar_time: datetime,
        O: float, H: float, L: float, C: float,
        four_hour: NumpyBars,
        fh_idx: int,
        is_4h_boundary: bool,
    ) -> None:
        """Per-bar management: gap stop, TP1/TP2, runner trailing, stale exit."""
        pos = self.active_position
        if pos is None:
            return

        setup = pos.setup
        pos.bars_held += 1

        # Track days held
        bar_date = bar_time.date() if hasattr(bar_time, 'date') else None
        if bar_date and bar_date != pos._last_daily_date:
            pos.days_held += 1
            pos._last_daily_date = bar_date

        # Update MFE/MAE
        if setup.direction == Direction.LONG:
            if H > pos.mfe_price:
                pos.mfe_price = H
            if L < pos.mae_price:
                pos.mae_price = L
        else:
            if L < pos.mfe_price:
                pos.mfe_price = L
            if H > pos.mae_price:
                pos.mae_price = H

        # Compute current R
        r_base = pos.r_price
        if r_base <= 0:
            return

        if setup.direction == Direction.LONG:
            r_now = (C - pos.fill_price) / r_base
        else:
            r_now = (pos.fill_price - C) / r_base

        # R state including realized PnL from partials
        pv = self.point_value
        unrealized = (C - pos.fill_price) * pv * pos.qty_open if setup.direction == Direction.LONG \
            else (pos.fill_price - C) * pv * pos.qty_open
        r_state = (pos.realized_pnl + unrealized) / setup.final_risk_dollars \
            if setup.final_risk_dollars > 0 else r_now

        # --- Gap-through-stop check at session open (09:30) ---
        bar_t = bar_time.time() if hasattr(bar_time, 'time') else None
        if bar_t and bar_t.hour == 9 and bar_t.minute <= 35:
            gap_triggered, gap_fill = stops.handle_gap_through_stop(
                setup.direction, pos.current_stop, O,
            )
            if gap_triggered:
                pos.gap_stop_event = True
                pos.gap_stop_fill_price = gap_fill
                self._flatten_position(pos, gap_fill, bar_time, "GAP_STOP")
                return

        # --- Compute stale flags (no exit yet — TP takes priority) ---
        stale_days = pos.days_held
        should_exit_stale = False
        if not self.flags.disable_stale_exit:
            if self.campaign.chop_mode == "DEGRADED":
                stale_days -= CHOP_DEGRADED_STALE_ADJ  # ADJ is -2, so adds 2
            should_warn, should_tighten, should_exit_stale = stops.check_stale_exit(
                stale_days, r_state,
            )
        else:
            should_warn = should_tighten = False

        # --- TP1 ---
        trade_regime = TradeRegime(pos.exit_tier) if pos.exit_tier else TradeRegime.NEUTRAL
        tp1_r, tp2_r = self._get_tp_r_multiples(trade_regime, self.cfg.tp_scale)

        new_stop = pos.current_stop
        tp1_hit_this_bar = False

        # --- Pre-runner trail: capture MFE as it develops, even before TP1 ---
        # Once r_state exceeds 0.10R, start ratcheting stop toward entry
        if not pos.runner_active and r_state > 0.10:
            # Lock in 30% of unrealized gain
            if setup.direction == Direction.LONG:
                candidate = pos.fill_price + 0.30 * (C - pos.fill_price)
                candidate = round_to_tick(candidate, self.cfg.tick_size, "down")
                if candidate > new_stop:
                    new_stop = candidate
            else:
                candidate = pos.fill_price - 0.30 * (pos.fill_price - C)
                candidate = round_to_tick(candidate, self.cfg.tick_size, "up")
                if candidate < new_stop:
                    new_stop = candidate

        if not pos.tp1_done and r_state >= tp1_r:
            pos.tp1_done = True
            pos.tp1_bar = pos.bars_held
            tp1_hit_this_bar = True
            # Two-leg exit: close 33% at TP1, runner on remaining 67%
            # Degraded/range: close 50% at TP1, runner on 50%
            chop_cap = self.campaign.chop_mode == "DEGRADED"
            range_cap = pos.exit_tier == ExitTier.NEUTRAL.value
            if chop_cap or range_cap:
                tp1_qty = max(1, pos.qty_open // 2)
            else:
                tp1_qty = max(1, pos.qty_open // 3)

            # Immediately activate runner on remainder (skip TP2 partial)
            pos.tp2_done = True
            pos.tp2_bar = pos.bars_held
            pos.runner_active = True

            pos.qty_tp1 = tp1_qty
            self._submit_partial_exit(pos, tp1_qty, bar_time)

            # Move stop to BE
            ctx = self._daily_ctx or {}
            be_stop = stops.compute_be_stop(
                setup.direction, pos.fill_price,
                ctx.get("atr14_d", 1.0),
                self.cfg.tick_size,
            )
            if setup.direction == Direction.LONG:
                new_stop = max(new_stop, be_stop)
            else:
                new_stop = min(new_stop, be_stop)
            pos.be_bar = pos.bars_held

            # Update portfolio position
            ps = self.positions.get(self.symbol)
            if ps:
                ps.tp1_done = True
                ps.tp2_done = True
                ps.runner_active = True

        # --- Stale exit (deferred — only if TP1 not hit this bar) ---
        if should_exit_stale and not tp1_hit_this_bar:
            self._flatten_position(pos, C, bar_time, "STALE")
            return

        # --- Runner trailing stop (4H ATR-based) ---
        if pos.runner_active and is_4h_boundary and self._cached_atr14_4h > 0:
            # Stale tighten
            if should_tighten:
                pos.trail_mult = stops.apply_stale_tighten(pos.trail_mult)

            # Compute trail mult
            continuation = self.campaign.continuation
            pos.trail_mult = stops.compute_trail_mult(r_state, r_now, continuation)

            trailing_stop = stops.compute_trailing_stop(
                setup.direction,
                self._cached_4h_highs,
                self._cached_4h_lows,
                self._cached_atr14_4h,
                pos.trail_mult,
                self._cached_ema50_4h,
                pos.current_stop,
                self.cfg.tick_size,
            )
            if setup.direction == Direction.LONG and trailing_stop > new_stop:
                new_stop = trailing_stop
            elif setup.direction == Direction.SHORT and trailing_stop < new_stop:
                new_stop = trailing_stop

        # --- Update stop if tightened ---
        if new_stop != pos.current_stop:
            safe = False
            if setup.direction == Direction.LONG and new_stop > pos.current_stop:
                safe = True
            elif setup.direction == Direction.SHORT and new_stop < pos.current_stop:
                safe = True
            if safe:
                pos.stop_tightens += 1
                pos.current_stop = new_stop
                self.broker.cancel_orders(self.symbol, tag="protective_stop")
                stop_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
                stop_order = SimOrder(
                    order_id=self.broker.next_order_id(),
                    symbol=self.symbol,
                    side=stop_side,
                    order_type=OrderType.STOP,
                    qty=pos.qty_open,
                    stop_price=new_stop,
                    tick_size=self.cfg.tick_size,
                    submit_time=bar_time,
                    ttl_hours=0,
                    tag="protective_stop",
                )
                self.broker.submit_order(stop_order)

                ps = self.positions.get(self.symbol)
                if ps:
                    ps.current_stop = new_stop

    def _submit_partial_exit(
        self, pos: _ActivePosition, qty: int, bar_time: datetime,
    ) -> None:
        """Submit a market order for partial exit."""
        if qty <= 0 or qty > pos.qty_open:
            return
        exit_side = OrderSide.SELL if pos.setup.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            qty=qty,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="partial",
        )
        self.broker.submit_order(order)

    # ------------------------------------------------------------------
    # Position flatten (market exit)
    # ------------------------------------------------------------------

    def _flatten_position(
        self,
        pos: _ActivePosition,
        exit_price: float,
        bar_time: datetime,
        reason: str,
    ) -> None:
        """Flatten entire position at the given price."""
        setup = pos.setup

        if setup.direction == Direction.LONG:
            pnl_points = exit_price - pos.fill_price
        else:
            pnl_points = pos.fill_price - exit_price

        pv = self.point_value
        pnl_dollars = pnl_points * pv * pos.qty_open + pos.realized_pnl
        r_multiple = pnl_dollars / setup.final_risk_dollars if setup.final_risk_dollars > 0 else 0.0

        r_base = pos.r_price
        if r_base > 0:
            if setup.direction == Direction.LONG:
                mfe_r = (pos.mfe_price - pos.fill_price) / r_base
                mae_r = (pos.fill_price - pos.mae_price) / r_base
            else:
                mfe_r = (pos.fill_price - pos.mfe_price) / r_base
                mae_r = (pos.mae_price - pos.fill_price) / r_base
        else:
            mfe_r = mae_r = 0.0

        self.equity += pnl_points * pv * pos.qty_open

        self._record_trade(pos, exit_price, bar_time, reason, pnl_points,
                           pnl_dollars, r_multiple, mfe_r, mae_r, 0.0)

        self._update_circuit_breaker(r_multiple, bar_time)
        self.broker.cancel_all(self.symbol)
        self._clear_position()

    def _clear_position(self) -> None:
        """Clear active position and portfolio state."""
        pos = self.active_position
        if pos:
            # Store re-entry info on campaign
            campaign = self.campaign
            bar_date = pos._last_daily_date
            if bar_date:
                campaign.last_exit_date = bar_date
                campaign.last_exit_direction = pos.setup.direction
                # Approximate realized R
                if pos.setup.final_risk_dollars > 0:
                    campaign.last_exit_realized_r = pos.realized_pnl / pos.setup.final_risk_dollars
                else:
                    campaign.last_exit_realized_r = 0.0

        # Clear portfolio position
        ps = self.positions.get(self.symbol)
        if ps:
            ps.qty = 0
            ps.tp1_done = False
            ps.tp2_done = False
            ps.runner_active = False
            ps.trail_active = False
            ps.total_risk_dollars = 0.0
            ps.regime_at_entry = None

        self.active_position = None
        self._pending_setup = None

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def _record_trade(
        self,
        pos: _ActivePosition,
        exit_price: float,
        bar_time: datetime,
        reason: str,
        pnl_points: float,
        pnl_dollars: float,
        r_multiple: float,
        mfe_r: float,
        mae_r: float,
        exit_commission: float,
    ) -> None:
        """Record a completed trade."""
        setup = pos.setup
        r_per_share = pos.r_price
        r_multiple_per_share = pnl_points / r_per_share if r_per_share > 0 else 0.0
        trade = BreakoutTradeRecord(
            symbol=self.symbol,
            direction=setup.direction,
            entry_type=pos.entry_type,
            exit_tier=pos.exit_tier,
            campaign_id=setup.campaign_id,
            box_version=setup.box_version,
            entry_time=pos.entry_time,
            exit_time=bar_time,
            entry_price=pos.fill_price,
            exit_price=exit_price,
            stop_price=pos.initial_stop,
            qty=pos.qty_initial,
            exit_reason=reason,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
            r_multiple_per_share=r_multiple_per_share,
            mfe_r=mfe_r,
            mae_r=mae_r,
            bars_held=pos.bars_held,
            days_held=pos.days_held,
            commission=pos.commission + exit_commission,
            tp1_done=pos.tp1_done,
            tp2_done=pos.tp2_done,
            runner_active=pos.runner_active,
            qty_tp1=pos.qty_tp1,
            qty_tp2=pos.qty_tp2,
            add_count=pos.add_count,
            gap_stop_event=pos.gap_stop_event,
            gap_stop_fill_price=pos.gap_stop_fill_price,
            disp_at_entry=pos.disp_at_entry,
            disp_th_at_entry=pos.disp_th_at_entry,
            rvol_d_at_entry=pos.rvol_d_at_entry,
            rvol_h_at_entry=self.hourly_state.rvol_h,
            regime_at_entry=pos.regime_at_entry,
            sq_good_at_entry=pos.sq_good_at_entry,
            quality_mult_at_entry=pos.quality_mult_at_entry,
            score_at_entry=pos.score_at_entry,
            # Score sub-components (Phase 2)
            score_vol_at_entry=pos.score_vol_at_entry,
            score_squeeze_at_entry=pos.score_squeeze_at_entry,
            score_regime_at_entry=pos.score_regime_at_entry,
            score_consec_at_entry=pos.score_consec_at_entry,
            score_atr_at_entry=pos.score_atr_at_entry,
            # Quality mult sub-components (Phase 2)
            regime_mult_at_entry=pos.regime_mult_at_entry,
            disp_mult_at_entry=pos.disp_mult_at_entry,
            squeeze_mult_at_entry=pos.squeeze_mult_at_entry,
            corr_mult_at_entry=pos.corr_mult_at_entry,
            chop_mode_at_entry=pos.chop_mode_at_entry,
            continuation_at_entry=pos.continuation_at_entry,
            # Partial fill timing (Phase 2)
            tp1_bar=pos.tp1_bar,
            tp2_bar=pos.tp2_bar,
            be_bar=pos.be_bar,
            # Stop evolution (Phase 2)
            final_stop_at_exit=pos.current_stop,
            stop_tightens=pos.stop_tightens,
            # Regime at exit (Phase 2)
            regime_at_exit=self._cached_regime_4h.value if self._cached_regime_4h else "",
            friction_dollars=pos.friction_dollars,
        )
        self.trades.append(trade)

    # ------------------------------------------------------------------
    # Signal event logging
    # ------------------------------------------------------------------

    def _log_signal_event(
        self,
        bar_time: datetime,
        direction: Direction,
        entry_type: EntryType | None,
        allowed: bool,
        blocked_reason: str,
        ctx: dict,
        rvol_h: float,
    ) -> None:
        """Log a signal event for filter attribution."""
        event = SignalEvent(
            timestamp=bar_time,
            symbol=self.symbol,
            campaign_state=self.campaign.state.value,
            direction=direction,
            entry_type_selected=entry_type.value if entry_type else "",
            allowed=allowed,
            blocked_reason=blocked_reason,
            quality_mult=ctx.get("quality_mult", 0.0),
            expiry_mult=ctx.get("expiry_mult", 0.0),
            disp_mult=ctx.get("disp_mult", 0.0),
            rvol_h=rvol_h,
            regime_4h=ctx.get("regime_4h", Regime4H.RANGE_CHOP).value if isinstance(ctx.get("regime_4h"), Regime4H) else str(ctx.get("regime_4h", "")),
            score_total=ctx.get("score_total", 0),
            # Score sub-components (Phase 2)
            score_vol=ctx.get("score_vol", 0),
            score_squeeze=ctx.get("score_squeeze", 0),
            score_regime=ctx.get("score_regime", 0),
            score_consec=ctx.get("score_consec", 0),
            score_atr=ctx.get("score_atr", 0),
            chop_mode=ctx.get("chop_mode", ""),
        )
        self.signal_events.append(event)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _update_circuit_breaker(self, r_multiple: float, bar_time: datetime) -> None:
        """Update circuit breaker state after a trade close."""
        cb = self.circuit_breaker
        cb.weekly_realized_r += r_multiple
        cb.monthly_realized_r += r_multiple

        from strategy_3.config import MONTHLY_HALT_R, MONTHLY_HALT_DAYS
        if cb.monthly_realized_r <= MONTHLY_HALT_R:
            cb.halted = True

    # ------------------------------------------------------------------
    # TP R-multiples
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tp_r_multiples(trade_regime: TradeRegime, tp_scale: float = 1.0) -> tuple[float, float]:
        """Return (TP1_R, TP2_R) for the given trade regime, scaled per-symbol."""
        if trade_regime == TradeRegime.ALIGNED:
            return TP1_R_ALIGNED * tp_scale, TP2_R_ALIGNED * tp_scale
        if trade_regime == TradeRegime.CAUTION:
            return TP1_R_CAUTION * tp_scale, TP2_R_CAUTION * tp_scale
        return TP1_R_NEUTRAL * tp_scale, TP2_R_NEUTRAL * tp_scale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_datetime(ts) -> datetime:
        """Convert numpy datetime64 or pandas Timestamp to datetime."""
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, 'to_pydatetime'):
            return ts.to_pydatetime()
        return pd.Timestamp(ts).to_pydatetime()
