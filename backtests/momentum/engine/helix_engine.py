"""Helix v4.0 Apex Trail backtest engine — 5-minute primary loop.

Replicates strategy/engine.py orchestration in synchronous, bar-by-bar mode.
Imports pure functions from strategy/ (signals, indicators, gates, risk, session).

v4.0 changes vs helix_engine.py:
- Only Class M + Class F (no A/R, no halt queue)
- DOW_BLOCKED: Monday + Wednesday
- Hour-of-day sizing multiplier
- Single-phase R-based trailing (no bar phases)
- Momentum stall exit at bar 8
- Uniform partials (P1 +1.0R/30%, P2 +2.0R/20%, runner 50%)
- Tighter MFE ratchet (1.0R activation, 60% floor)
- Single time-decay BE at bar 12
- Tighter early adverse (4 bars, -0.80R)
- Faster stale exits (14/18 bars)
- Chop zone sizing penalty instead of hard gate
"""
from __future__ import annotations

import logging
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from backtest.analysis.helix_shadow_tracker import HelixShadowTracker
from backtest.config import round_to_tick, SlippageConfig
from backtest.config_helix import Helix4AblationFlags, Helix4BacktestConfig
from backtest.data.preprocessing import NumpyBars
from backtest.engine.sim_broker import (
    FillResult, FillStatus, OrderSide, OrderType, SimBroker, SimOrder,
)

from strategy import config as C
from strategy.config import (
    Setup, SetupClass, SetupState, PositionState, SessionBlock, TF, Bar, Pivot,
    MAX_CONCURRENT_POSITIONS, DUPLICATE_OVERRIDE_MIN_R,
    PARTIAL1_R, PARTIAL1_FRAC, PARTIAL2_R, PARTIAL2_FRAC,
    EARLY_STALE_BARS, EARLY_STALE_MAX_BARS, STALE_M_BARS, STALE_R_THRESHOLD,
    TRAIL_ACTIVATION_R, TRAIL_LOOKBACK_1H, TRAIL_MULT_MAX, TRAIL_MULT_FLOOR, TRAIL_R_DIVISOR,
    MFE_RATCHET_ACTIVATION_R, MFE_RATCHET_FLOOR_PCT,
    TIME_DECAY_BARS, TIME_DECAY_PROGRESS_R,
    SPIKE_FILTER_ATR_MULT, EXTENSION_ATR_MULT,
    CATASTROPHIC_FILL_MULT,
    TELEPORT_RTH_TICKS, TELEPORT_RTH_ATR_FRAC,
    TELEPORT_ETH_TICKS, TELEPORT_ETH_ATR_FRAC,
    SESSION_SIZE_MULT,
    CATCHUP_OVERSHOOT_ATR_FRAC, CATCHUP_OFFSET_PTS, TTL_CATCHUP_S,
    EARLY_ADVERSE_BARS, EARLY_ADVERSE_R,
    MOMENTUM_STALL_BAR, MOMENTUM_STALL_MIN_R, MOMENTUM_STALL_MIN_MFE,
    DOW_BLOCKED, HOUR_SIZE_MULT, DOW_SIZE_MULT,
    CLASS_T_MIN_BARS_SINCE_M,
)
from strategy.gates import check_gates, GateResult
from strategy.indicators import BarSeries, VolEngine
from strategy.pivots import PivotDetector
from strategy.positions import unrealized_r, r_state
from strategy.risk import RiskEngine
from strategy.session import (
    get_session_block, entries_allowed,
    is_halt, is_reopen_dead, is_pre_halt_cancel,
    class_allowed_in_session, max_spread_for_session, is_rth,
    session_size_mult, NewsCalendar,
)
from strategy.signals import (
    SignalEngine, alignment_score, trend_strength, strong_trend_flag,
)

logger = logging.getLogger(__name__)

_ET = None


def _get_et():
    global _ET
    if _ET is None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    return _ET


def _to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        from zoneinfo import ZoneInfo
        return dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(_get_et())
    return dt.astimezone(_get_et())


# ---------------------------------------------------------------------------
# Logging / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Helix4SetupEvent:
    """Every setup detected (even if not traded)."""
    setup_id: str
    timestamp: datetime
    setup_class: str
    direction: int
    tf_origin: str
    alignment_score: int
    strong_trend: bool
    vol_pct: float
    entry_stop: float
    stop0: float
    session_block: str


@dataclass
class Helix4GateDecision:
    """Every gate evaluation."""
    setup_id: str
    timestamp: datetime
    decision: str  # "placed" or "blocked"
    block_reasons: list[str] = field(default_factory=list)
    ablation_override: bool = False


@dataclass
class Helix4TradeRecord:
    """Completed trade with full context."""
    symbol: str = ""
    direction: int = 0
    setup_class: str = ""
    # Timing
    setup_time: datetime | None = None
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    bars_held_1h: int = 0
    # Prices
    avg_entry: float = 0.0
    exit_price: float = 0.0
    initial_stop: float = 0.0
    # PnL
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    unit1_risk_usd: float = 0.0
    dd_mult_at_entry: float = 1.0
    # Exit
    exit_reason: str = ""
    # Context
    session_at_entry: str = ""
    vol_pct_at_entry: float = 0.0
    alignment_at_entry: int = 0
    teleport: bool = False
    commission: float = 0.0
    # Milestones
    hit_1r: bool = False
    partial_done: bool = False
    partial2_done: bool = False
    entry_contracts: int = 0
    exit_contracts: int = 0


@dataclass
class Helix4EntryTracking:
    """Track every placed entry order for fill-rate diagnostics."""
    setup_id: str = ""
    setup_class: str = ""
    direction: int = 0
    entry_stop: float = 0.0
    stop0: float = 0.0
    arm_price: float = 0.0
    arm_time: datetime | None = None
    session_block: str = ""
    alignment_score: int = 0
    contracts: int = 0
    risk_r: float = 0.0
    unit1_risk: float = 0.0
    # Outcome
    filled: bool = False
    fill_price: float = 0.0
    fill_time: datetime | None = None
    expired: bool = False
    expire_time: datetime | None = None
    closest_price: float = 0.0
    # Heat state at arm time
    heat_total_r: float = 0.0
    heat_dir_r: float = 0.0


@dataclass
class Helix4SymbolResult:
    """Per-symbol backtest output."""
    symbol: str = ""
    trades: list[Helix4TradeRecord] = field(default_factory=list)
    setup_log: list[Helix4SetupEvent] = field(default_factory=list)
    gate_log: list[Helix4GateDecision] = field(default_factory=list)
    entry_tracking: list[Helix4EntryTracking] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    total_commission: float = 0.0
    # Counters
    setups_detected: int = 0
    entries_placed: int = 0
    entries_filled: int = 0
    gates_blocked: int = 0
    shadow_summary: str = ""
    shadow_tracker: HelixShadowTracker | None = None


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class _ActivePosition:
    """Internal mutable position state for backtest."""
    pos: PositionState
    record: Helix4TradeRecord
    stop_order_id: str = ""
    mfe_price: float = 0.0
    mae_price: float = 0.0
    setup_id: str = ""
    entry_contracts: int = 0
    commission_at_open: float = 0.0


@dataclass
class _PendingSetup:
    """Armed entry order awaiting fill."""
    setup: Setup
    order_id: str
    stop_order_price: float  # stop0 for protective stop after fill
    unit1_risk: float
    setup_size_mult: float
    session_mult: float
    contracts: int
    record_template: Helix4TradeRecord
    catchup_order_id: str = ""
    oca_group: str = ""
    armed_risk_r: float = 0.0


# ---------------------------------------------------------------------------
# Ablation patch
# ---------------------------------------------------------------------------

@contextmanager
def _ablation_patch(overrides: dict[str, float]):
    """Temporarily override strategy.config module-level constants."""
    import strategy.gates as _gates
    import strategy.risk as _risk_mod

    _submodules = [_gates, _risk_mod]
    originals = {}
    sub_originals: list[tuple] = []
    for key, value in overrides.items():
        upper = key.upper()
        if hasattr(C, upper):
            originals[upper] = getattr(C, upper)
            setattr(C, upper, value)
            for mod in _submodules:
                if hasattr(mod, upper):
                    sub_originals.append((mod, upper, getattr(mod, upper)))
                    setattr(mod, upper, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(C, key, value)
        for mod, attr, old_val in sub_originals:
            setattr(mod, attr, old_val)


# Gate reason -> ablation flag mapping
_GATE_ABLATION_MAP = {
    "extension_blocked": "use_extension_gate",
    "session_RTH_DEAD": "use_dead_zone_midday",
    "session_no_entries_RTH_DEAD": "use_dead_zone_midday",
    "session_REOPEN_DEAD": "use_reopen_dead_zone",
    "session_no_entries_REOPEN_DEAD": "use_reopen_dead_zone",
    "news_blocked": "use_news_guard",
    "spike_filter": "use_spike_filter",
    "min_stop": "use_min_stop_gate",
    "spread": "use_spread_gate",
    "eth_europe": "enable_ETH_Europe",
}


def _should_ablation_override(reason: str, flags: Helix4AblationFlags) -> bool:
    """Check if a gate block reason is disabled by ablation flags."""
    # When ETH_EUROPE regime is active, ALL eth_europe gates must be respected
    # (shorts blocked, score requirements, regime trend/vol filters)
    if reason.startswith("eth_europe"):
        if flags.use_eth_europe_regime:
            # Regime mode: regime gates are enforced, non-regime gates also enforced
            return False
        # No regime: override if blanket is also off (both disabled → engine blocks anyway)
        if not flags.enable_ETH_Europe:
            return True
        return False
    for prefix, flag_name in _GATE_ABLATION_MAP.items():
        if reason.startswith(prefix) and not getattr(flags, flag_name, True):
            return True
    if "ETH_EUROPE" in reason and not flags.enable_ETH_Europe:
        if not flags.use_eth_europe_regime:
            return True
    return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Helix4Engine:
    """Single-symbol bar-by-bar Helix v4.0 backtest on 5-minute bars."""

    def __init__(self, symbol: str, bt_config: Helix4BacktestConfig) -> None:
        self.symbol = symbol
        self.cfg = bt_config
        self.flags = bt_config.flags
        self.tick = bt_config.tick_size
        self.pv = bt_config.point_value

        # SimBroker
        self.broker = SimBroker(slippage_config=bt_config.slippage)

        # Account
        self.equity = bt_config.initial_equity

        # Strategy components (all synchronous / pure)
        self._h1 = BarSeries(TF.H1, maxlen=500)
        self._h4 = BarSeries(TF.H4, maxlen=200)
        self._daily = BarSeries(TF.D1, maxlen=200)
        self._pivots_1h = PivotDetector(TF.H1)
        self._pivots_4h = PivotDetector(TF.H4)
        self._vol = VolEngine()
        self._signals = SignalEngine()
        self._risk = RiskEngine(bt_config.initial_equity, self._vol, point_value=bt_config.point_value)
        self._news = NewsCalendar()

        # Drawdown throttle
        from libs.risk.drawdown_throttle import DrawdownThrottle
        self._throttle = DrawdownThrottle(bt_config.initial_equity)

        # Load news calendar if provided
        if bt_config.news_calendar_path:
            self._load_news_calendar(bt_config.news_calendar_path)

        # Shadow tracker
        self._shadow: HelixShadowTracker | None = (
            HelixShadowTracker() if bt_config.track_shadows else None
        )

        # Deduplication: track placed setup signatures
        self._placed_signatures: set[tuple] = set()
        self._sig_expiry: dict[tuple, datetime] = {}

        # Class T: track last M placement bar per direction for dedup
        self._last_m_bar: dict[int, int] = {}

        # Position state
        self._active_positions: list[_ActivePosition] = []
        self._pending: dict[str, _PendingSetup] = {}

        # Index tracking
        self._last_1h_idx = -1
        self._last_4h_idx = -1
        self._last_d_idx = -1
        self._bar_idx_1h = 0

        # Indicator caches
        self._vol_pct = 50.0
        self._trend_strength_3d: float | None = None
        self._ts_history: deque[float] = deque(maxlen=10)

        # Result accumulators
        self._trades: list[Helix4TradeRecord] = []
        self._setup_log: list[Helix4SetupEvent] = []
        self._gate_log: list[Helix4GateDecision] = []
        self._entry_tracking: list[Helix4EntryTracking] = []
        self._entry_tracking_by_setup: dict[str, Helix4EntryTracking] = {}
        self._equity_history: list[float] = []
        self._time_history: list = []
        self._total_commission = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        minute_bars: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily: NumpyBars,
        hourly_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
        daily_idx_map: np.ndarray,
    ) -> Helix4SymbolResult:
        n = len(minute_bars)
        warmup_d = self.cfg.warmup_daily
        warmup_h = self.cfg.warmup_1h
        warmup_4h = self.cfg.warmup_4h

        with _ablation_patch(dict(self.cfg.param_overrides)):
            for t in range(n):
                bar_time = self._bar_time(minute_bars.times[t])
                O = float(minute_bars.opens[t])
                H = float(minute_bars.highs[t])
                L = float(minute_bars.lows[t])
                Cl = float(minute_bars.closes[t])

                if np.isnan(O) or np.isnan(Cl):
                    continue

                h_idx = int(hourly_idx_map[t])
                fh_idx = int(four_hour_idx_map[t])
                d_idx = int(daily_idx_map[t])

                self._step_minute(
                    bar_time, O, H, L, Cl,
                    hourly, four_hour, daily,
                    h_idx, fh_idx, d_idx,
                    warmup_d, warmup_h, warmup_4h,
                )

            # Close any remaining positions
            last_close = float(minute_bars.closes[-1])
            last_time = self._bar_time(minute_bars.times[-1])
            for ap in list(self._active_positions):
                if ap.pos.contracts > 0:
                    self._close_position_market(ap, last_close, last_time, "END_OF_DATA")

        # Shadow simulation
        shadow_summary = ""
        if self._shadow and self._shadow.rejections:
            h_data = {self.symbol: (
                hourly.opens, hourly.highs, hourly.lows, hourly.closes, hourly.volumes,
            )}
            h_times = {self.symbol: hourly.times}
            m_data = {self.symbol: (
                minute_bars.opens, minute_bars.highs, minute_bars.lows, minute_bars.closes,
            )}
            m_times = {self.symbol: minute_bars.times}
            self._shadow.simulate_shadows(h_data, h_times, m_data, m_times)
            shadow_summary = self._shadow.format_summary()

        return Helix4SymbolResult(
            symbol=self.symbol,
            trades=self._trades,
            setup_log=self._setup_log,
            gate_log=self._gate_log,
            entry_tracking=self._entry_tracking,
            equity_curve=np.array(self._equity_history),
            timestamps=np.array(self._time_history),
            total_commission=self._total_commission,
            setups_detected=len(self._setup_log),
            entries_placed=sum(1 for p in self._gate_log if p.decision == "placed"),
            entries_filled=len(self._trades),
            gates_blocked=sum(1 for g in self._gate_log if g.decision == "blocked"),
            shadow_summary=shadow_summary,
            shadow_tracker=self._shadow if self._shadow and self._shadow.rejections else None,
        )

    # ------------------------------------------------------------------
    # Per-bar step (5-minute)
    # ------------------------------------------------------------------

    def _step_minute(
        self,
        bar_time: datetime,
        O: float, H: float, L: float, Cl: float,
        hourly: NumpyBars, four_hour: NumpyBars, daily: NumpyBars,
        h_idx: int, fh_idx: int, d_idx: int,
        warmup_d: int, warmup_h: int, warmup_4h: int,
    ) -> None:
        now_et = _to_et(bar_time)

        # Pre-halt cancel
        if is_pre_halt_cancel(now_et) and self._pending:
            for oid in list(self._pending):
                pend = self._pending.pop(oid, None)
                if pend is not None:
                    self._risk.release_pending_risk(pend.setup.direction, pend.armed_risk_r)
            self.broker.cancel_orders(self.symbol, tag="helix4_entry")
            self.broker.cancel_orders(self.symbol, tag="helix4_catchup")

        # 1. Process broker fills
        fills = self.broker.process_bar(self.symbol, bar_time, O, H, L, Cl, self.tick)
        for fill in fills:
            self._handle_fill(fill, bar_time)

        # 2. MFE/MAE + 5m catastrophic check
        for ap in list(self._active_positions):
            if ap.pos.contracts > 0:
                self._update_mfe_mae(ap, H, L)
                self._check_catastrophic_5m(ap, H, L, bar_time)

        # 2b. Track closest price approach for unfilled entries
        for oid, pend in list(self._pending.items()):
            et = self._entry_tracking_by_setup.get(pend.setup.setup_id)
            if et and not et.filled:
                if pend.setup.direction == 1:
                    if H > et.closest_price:
                        et.closest_price = H
                else:
                    if et.closest_price == 0 or L < et.closest_price:
                        et.closest_price = L

        # 3. Higher-TF boundary processing
        new_d = d_idx != self._last_d_idx and d_idx >= warmup_d
        new_4h = fh_idx != self._last_4h_idx and fh_idx >= warmup_4h
        new_1h = h_idx != self._last_1h_idx and h_idx >= warmup_h

        if new_d:
            self._on_daily_boundary(daily, d_idx)
            self._last_d_idx = d_idx

        if new_4h:
            self._on_4h_boundary(four_hour, fh_idx, bar_time)
            self._last_4h_idx = fh_idx

        if new_1h:
            self._on_1h_boundary(hourly, h_idx, bar_time)
            self._last_1h_idx = h_idx

        # 4. Equity snapshot (mark-to-market)
        if new_1h or len(self._equity_history) == 0:
            mtm = self.equity
            for ap in self._active_positions:
                p = ap.pos
                if p.contracts > 0:
                    mtm += (Cl - p.avg_entry) * p.direction * self.pv * p.contracts
            self._equity_history.append(mtm)
            self._time_history.append(bar_time)

    # ------------------------------------------------------------------
    # TF boundary handlers
    # ------------------------------------------------------------------

    def _on_daily_boundary(self, daily: NumpyBars, d_idx: int) -> None:
        bar = Bar(
            ts=self._bar_time(daily.times[d_idx]),
            open=float(daily.opens[d_idx]),
            high=float(daily.highs[d_idx]),
            low=float(daily.lows[d_idx]),
            close=float(daily.closes[d_idx]),
            volume=float(daily.volumes[d_idx]) if d_idx < len(daily.volumes) else 0,
        )
        self._daily.add_bar(bar)
        self._vol.update(self._daily)
        self._vol_pct = self._vol.vol_pct
        self._risk.update_equity(self.equity)
        self._throttle.update_equity(self.equity)
        self._throttle.daily_reset()

        # Track trend_strength for 3-day lookback
        ts_val = trend_strength(self._daily)
        self._ts_history.append(ts_val)
        if len(self._ts_history) >= 4:
            self._trend_strength_3d = self._ts_history[-4]
        self._signals.update_trend_strength_3d(self._trend_strength_3d)

    def _on_4h_boundary(self, four_hour: NumpyBars, fh_idx: int, bar_time: datetime) -> None:
        bar = Bar(
            ts=self._bar_time(four_hour.times[fh_idx]),
            open=float(four_hour.opens[fh_idx]),
            high=float(four_hour.highs[fh_idx]),
            low=float(four_hour.lows[fh_idx]),
            close=float(four_hour.closes[fh_idx]),
            volume=float(four_hour.volumes[fh_idx]) if fh_idx < len(four_hour.volumes) else 0,
        )
        self._h4.add_bar(bar)
        self._pivots_4h.on_bar(bar, self._h4)

    def _on_1h_boundary(self, hourly: NumpyBars, h_idx: int, bar_time: datetime) -> None:
        self._bar_idx_1h += 1
        bar = Bar(
            ts=self._bar_time(hourly.times[h_idx]),
            open=float(hourly.opens[h_idx]),
            high=float(hourly.highs[h_idx]),
            low=float(hourly.lows[h_idx]),
            close=float(hourly.closes[h_idx]),
            volume=float(hourly.volumes[h_idx]) if h_idx < len(hourly.volumes) else 0,
        )
        self._h1.add_bar(bar)
        self._pivots_1h.on_bar(bar, self._h1)
        self._signals.tick_bars()

        now_et = _to_et(bar_time)

        # Position management
        for ap in list(self._active_positions):
            if ap.pos.contracts > 0:
                self._manage_position(ap, now_et, float(hourly.closes[h_idx]))

        # Expire pending setups
        self._cleanup_expired_pending(bar_time)

        # Signal detection and entry
        if self._h1.current_atr() > 0 and self._daily.current_atr() > 0:
            self._detect_and_arm(now_et)

    # ------------------------------------------------------------------
    # Signal detection and entry
    # ------------------------------------------------------------------

    def _detect_and_arm(self, now_et: datetime) -> None:
        active_count = sum(1 for ap in self._active_positions if ap.pos.contracts > 0)

        block = get_session_block(now_et)

        # Halt / dead zone quick exit
        if is_halt(now_et) or is_reopen_dead(now_et):
            return

        # DOW block: Monday + Wednesday
        if self.flags.block_monday and now_et.weekday() == 0:
            return
        if self.flags.block_wednesday and now_et.weekday() == 2:
            return

        # ETH_EUROPE block (ablation flag + regime expansion)
        if block == SessionBlock.ETH_EUROPE:
            if not self.flags.enable_ETH_Europe and not self.flags.use_eth_europe_regime:
                return
            # If regime mode only (no blanket), let entries through to gate chain

        atr1 = self._h1.current_atr()

        # Detect setups: Class M on every 1H bar
        setups: list[Setup] = []
        m_setups = self._signals.detect_class_M(
            self._pivots_1h, self._h1, self._h4, self._daily, now_et)
        setups.extend(m_setups)

        # Class F: fast re-entry after trailing exit
        if self.flags.use_class_F:
            f_setups = self._signals.detect_class_F(
                self._h1, self._h4, self._daily, now_et)
            setups.extend(f_setups)

        # Class T: 4H trend continuation (longs only)
        if self.flags.use_class_T:
            t_setups = self._signals.detect_class_T(
                self._pivots_1h, self._h1, self._h4, self._daily, now_et)
            # Suppress if Class M fired same bar same direction
            m_dirs = {s.direction for s in m_setups}
            t_setups = [s for s in t_setups if s.direction not in m_dirs]
            # Suppress if M placed recently same direction
            t_setups = [s for s in t_setups
                        if (self._bar_idx_1h - self._last_m_bar.get(s.direction, -999))
                           >= CLASS_T_MIN_BARS_SINCE_M]
            setups.extend(t_setups)

        # Expire old dedup signatures
        self._expire_signatures(now_et)

        for setup in setups:
            session_block = get_session_block(now_et)

            # Dedup: skip if same pivot structure already placed
            # Class T uses breakout_pivot.ts instead of P1.ts (P1=None)
            sig = (
                setup.P1.ts if setup.P1 else (
                    setup.breakout_pivot.ts if setup.breakout_pivot else None),
                setup.P2.ts if setup.P2 else None,
                setup.direction,
                setup.cls.value,
            )
            if sig in self._placed_signatures:
                continue

            # Pending dedup: skip if same class+direction already pending (matches live engine.py:391-393)
            if any(p.setup.cls == setup.cls and p.setup.direction == setup.direction
                   for p in self._pending.values()):
                continue

            # Drawdown throttle: daily loss cap halt
            if self.flags.use_drawdown_throttle and self._throttle.daily_halted:
                self._throttle.entries_blocked_daily += 1
                continue

            # Log setup
            self._setup_log.append(Helix4SetupEvent(
                setup_id=setup.setup_id,
                timestamp=now_et,
                setup_class=setup.cls.value,
                direction=setup.direction,
                tf_origin=setup.tf_origin.value,
                alignment_score=setup.alignment_score,
                strong_trend=setup.strong_trend,
                vol_pct=self._vol_pct,
                entry_stop=setup.entry_stop,
                stop0=setup.stop0,
                session_block=session_block.value,
            ))

            # Block short entries below minimum alignment score
            if setup.direction == -1 and setup.alignment_score < self.flags.short_min_score:
                self._gate_log.append(Helix4GateDecision(
                    setup_id=setup.setup_id,
                    timestamp=now_et,
                    decision="blocked",
                    block_reasons=[f"short_min_score_{setup.alignment_score}<{self.flags.short_min_score}"],
                ))
                if self._shadow:
                    self._shadow.record_rejection(
                        symbol=self.symbol,
                        direction=setup.direction,
                        filter_names=[f"short_min_score_{setup.alignment_score}<{self.flags.short_min_score}"],
                        time=now_et,
                        entry_price=setup.entry_stop,
                        stop_price=setup.stop0,
                        setup_class=setup.cls.value,
                        session_block=session_block.value,
                        alignment_score=setup.alignment_score,
                    )
                continue

            # Mirror live engine: block shorts at ETH_QUALITY_AM unless alignment_score >= 2
            if setup.direction == -1 and session_block == SessionBlock.ETH_QUALITY_AM and setup.alignment_score < 2:
                self._gate_log.append(Helix4GateDecision(
                    setup_id=setup.setup_id,
                    timestamp=now_et,
                    decision="blocked",
                    block_reasons=["eth_quality_am_short_block"],
                ))
                continue

            # Gate check with ablation
            gate_result, all_reasons = self._check_gates_with_ablation(setup, now_et)
            gate_decision = Helix4GateDecision(
                setup_id=setup.setup_id,
                timestamp=now_et,
                decision="placed" if gate_result else "blocked",
                block_reasons=all_reasons,
            )

            if not gate_result:
                overridden = all(
                    _should_ablation_override(r, self.flags) for r in all_reasons
                ) if all_reasons else False

                if overridden:
                    gate_decision.decision = "placed"
                    gate_decision.ablation_override = True
                else:
                    self._gate_log.append(gate_decision)
                    if self._shadow:
                        self._shadow.record_rejection(
                            symbol=self.symbol,
                            direction=setup.direction,
                            filter_names=all_reasons,
                            time=now_et,
                            entry_price=setup.entry_stop,
                            stop_price=setup.stop0,
                            setup_class=setup.cls.value,
                            session_block=session_block.value,
                            alignment_score=setup.alignment_score,
                        )
                    continue

            self._gate_log.append(gate_decision)

            # Block arming if at concurrent position limit
            if active_count >= MAX_CONCURRENT_POSITIONS:
                self._gate_log.append(Helix4GateDecision(
                    setup_id=setup.setup_id, timestamp=now_et,
                    decision="blocked", block_reasons=["max_concurrent_positions"],
                ))
                continue

            # Block duplicate: same direction + same class already open
            last_price = self._h1.last_close
            dup_blocking = False
            for ap in self._active_positions:
                if ap.pos.contracts <= 0:
                    continue
                if ap.pos.direction == setup.direction and ap.pos.origin_class == setup.cls:
                    cur_r = unrealized_r(ap.pos, last_price, self.pv)
                    if cur_r < DUPLICATE_OVERRIDE_MIN_R:
                        dup_blocking = True
                        break
            if dup_blocking:
                self._gate_log.append(Helix4GateDecision(
                    setup_id=setup.setup_id, timestamp=now_et,
                    decision="blocked", block_reasons=["duplicate_class_direction"],
                ))
                continue

            # Sizing
            self._risk.update_equity(self.equity)
            unit1 = self._risk.compute_unit1_risk_usd(setup)
            sm = self._risk.setup_size_mult(setup)
            if sm <= 0:
                self._gate_log.append(Helix4GateDecision(
                    setup_id=setup.setup_id, timestamp=now_et,
                    decision="blocked", block_reasons=[f"size_mult_zero_{sm:.2f}"],
                ))
                continue

            sess_mult = session_size_mult(session_block)

            # RTH_DEAD enhanced sizing
            if self.flags.use_rtd_dead_enhanced:
                ts_daily = trend_strength(self._daily)
                rtd_mult = self._risk.rtd_dead_enhanced_mult(
                    session_block, setup.alignment_score, ts_daily)
                sess_mult *= rtd_mult

            # ETH_EUROPE regime sizing (reduced from base)
            if self.flags.use_eth_europe_regime and session_block == SessionBlock.ETH_EUROPE:
                sess_mult *= self._risk.eth_europe_regime_mult(session_block)

            # Hour-of-day sizing
            hour_mult = 1.0
            if self.flags.use_hour_sizing:
                hour_mult = HOUR_SIZE_MULT.get(now_et.hour, 1.0)

            # Day-of-week sizing (Thursday halving etc.)
            dow_mult = DOW_SIZE_MULT.get(now_et.weekday(), 1.0) if DOW_SIZE_MULT else 1.0

            stop_dist = abs(setup.entry_stop - setup.stop0)
            if stop_dist <= 0:
                self._gate_log.append(Helix4GateDecision(
                    setup_id=setup.setup_id, timestamp=now_et,
                    decision="blocked", block_reasons=[f"stop_dist_zero_{stop_dist:.2f}"],
                ))
                continue

            risk_based = int(unit1 * sm * sess_mult * hour_mult * dow_mult / (stop_dist * self.pv))
            if self.cfg.fixed_qty is not None:
                contracts = self.cfg.fixed_qty
                if dow_mult < 1.0:
                    contracts = max(1, int(contracts * dow_mult))
                unit1 = stop_dist * self.pv * contracts
            else:
                contracts = risk_based

            # Drawdown throttle: reduce sizing
            if self.flags.use_drawdown_throttle:
                dd_mult = self._throttle.dd_size_mult
                if dd_mult <= 0.0:
                    self._throttle.entries_blocked_dd += 1
                    self._gate_log.append(Helix4GateDecision(
                        setup_id=setup.setup_id, timestamp=now_et,
                        decision="blocked", block_reasons=["drawdown_pause"],
                    ))
                    continue
                if dd_mult < 1.0:
                    contracts = max(1, int(contracts * dd_mult))
                    unit1 = stop_dist * self.pv * contracts

            if self.flags.vol_50_80_sizing_mult < 1.0 and 50 < self._vol_pct < 80 and contracts > 1:
                contracts = max(1, int(contracts * self.flags.vol_50_80_sizing_mult))

            if contracts < 1:
                self._gate_log.append(Helix4GateDecision(
                    setup_id=setup.setup_id, timestamp=now_et,
                    decision="blocked",
                    block_reasons=[f"contracts_zero_u1={unit1:.0f}_sm={sm:.2f}_sess={sess_mult:.2f}_sd={stop_dist:.1f}"],
                ))
                continue

            # Place stop-market entry via SimBroker with OCA group
            side = OrderSide.BUY if setup.direction == 1 else OrderSide.SELL
            order_id = self.broker.next_order_id()
            ttl_h = C.TTL_1H_S // 3600
            oca_grp = f"helix4_entry_{setup.setup_id}"

            order = SimOrder(
                order_id=order_id,
                symbol=self.symbol,
                side=side,
                order_type=OrderType.STOP,
                qty=contracts,
                stop_price=round_to_tick(setup.entry_stop, self.tick),
                tick_size=self.tick,
                submit_time=now_et,
                ttl_hours=ttl_h,
                tag="helix4_entry",
                oca_group=oca_grp,
            )
            self.broker.submit_order(order)

            # Track pending risk for heat caps
            entry_risk_r_armed = self._risk.compute_risk_r(
                setup.entry_stop, setup.stop0, contracts, unit1)
            self._risk.add_pending_risk(setup.direction, entry_risk_r_armed)

            record = Helix4TradeRecord(
                symbol=self.symbol,
                direction=setup.direction,
                setup_class=setup.cls.value,
                setup_time=now_et,
                initial_stop=setup.stop0,
                session_at_entry=session_block.value,
                vol_pct_at_entry=self._vol_pct,
                alignment_at_entry=setup.alignment_score,
            )
            pending = _PendingSetup(
                setup=setup, order_id=order_id,
                stop_order_price=setup.stop0,
                unit1_risk=unit1, setup_size_mult=sm,
                session_mult=sess_mult, contracts=contracts,
                record_template=record,
                oca_group=oca_grp,
                armed_risk_r=entry_risk_r_armed,
            )
            self._pending[order_id] = pending

            # Record dedup signature with TTL-based expiry
            self._placed_signatures.add(sig)
            ttl_s = setup.ttl_seconds()
            self._sig_expiry[sig] = now_et + timedelta(seconds=ttl_s)

            # Track M placements for Class T suppression
            if setup.cls == SetupClass.M:
                self._last_m_bar[setup.direction] = self._bar_idx_1h

            # Entry tracking
            last_price = self._h1.last_close if self._h1.last_close > 0 else 0.0
            entry_risk_r = self._risk.compute_risk_r(
                setup.entry_stop, setup.stop0, contracts, unit1)
            et = Helix4EntryTracking(
                setup_id=setup.setup_id,
                setup_class=setup.cls.value,
                direction=setup.direction,
                entry_stop=setup.entry_stop,
                stop0=setup.stop0,
                arm_price=last_price,
                arm_time=now_et,
                session_block=session_block.value,
                alignment_score=setup.alignment_score,
                contracts=contracts,
                risk_r=entry_risk_r,
                unit1_risk=unit1,
                closest_price=last_price,
                heat_total_r=self._risk.open_risk_r + self._risk.pending_risk_r,
                heat_dir_r=self._risk.dir_risk_r.get(setup.direction, 0.0),
            )
            self._entry_tracking.append(et)
            self._entry_tracking_by_setup[setup.setup_id] = et

            # Catch-up
            catchup_id = self._place_catchup_if_overshot(
                setup, pending, side, last_price, atr1, now_et, oca_grp,
            )
            if catchup_id:
                pending.catchup_order_id = catchup_id
                self._pending[catchup_id] = pending

    def _place_catchup_if_overshot(
        self, setup: Setup, pending: _PendingSetup,
        side: OrderSide, last_price: float, atr1h: float,
        now_et: datetime, oca_grp: str,
    ) -> str:
        """Place catch-up marketable LIMIT if price overshot within cap."""
        if last_price <= 0 or atr1h <= 0:
            return ""

        overshoot_cap = CATCHUP_OVERSHOOT_ATR_FRAC * atr1h

        if setup.direction == 1:
            overshoot = last_price - setup.entry_stop
        else:
            overshoot = setup.entry_stop - last_price

        if overshoot <= 0 or overshoot > overshoot_cap:
            return ""

        if setup.direction == 1:
            limit = round_to_tick(last_price + CATCHUP_OFFSET_PTS, self.tick)
        else:
            limit = round_to_tick(last_price - CATCHUP_OFFSET_PTS, self.tick)

        catchup_id = self.broker.next_order_id()
        catchup_order = SimOrder(
            order_id=catchup_id,
            symbol=self.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            qty=pending.contracts,
            limit_price=limit,
            tick_size=self.tick,
            submit_time=now_et,
            ttl_minutes=int(TTL_CATCHUP_S / 60),
            tag="helix4_catchup",
            oca_group=oca_grp,
        )
        self.broker.submit_order(catchup_order)
        return catchup_id

    def _check_gates_with_ablation(
        self, setup: Setup, now_et: datetime,
    ) -> tuple[bool, list[str]]:
        """Run check_gates(collect_all=True), return (passed, all_reasons)."""
        unit1 = self._risk.compute_unit1_risk_usd(setup)
        stop_dist = abs(setup.entry_stop - setup.stop0)
        sm = self._risk.setup_size_mult(setup)
        contracts = int(unit1 * sm / (stop_dist * self.pv)) if stop_dist > 0 else 0
        new_risk_r = self._risk.compute_risk_r(
            setup.entry_stop, setup.stop0, max(contracts, 1), unit1)

        spread = self._estimate_spread(now_et)
        mid_price = self._h1.last_close if self._h1.last_close > 0 else setup.entry_stop
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2

        result = check_gates(
            setup=setup,
            now_et=now_et,
            h1=self._h1,
            daily=self._daily,
            vol=self._vol,
            news=self._news,
            bid=bid,
            ask=ask,
            open_risk_r=self._risk.open_risk_r,
            pending_risk_r=self._risk.pending_risk_r,
            dir_risk_r=self._risk.dir_risk_r.get(setup.direction, 0.0),
            heat_cap_r=self._risk.heat_cap_r(),
            heat_cap_dir_r=self._risk.heat_cap_dir_r(),
            collect_all=True,
        )

        if result.passed:
            return True, []

        return False, result.all_reasons

    def _estimate_spread(self, now_et: datetime) -> float:
        """Session-aware synthetic spread in points."""
        block = get_session_block(now_et)
        base = {
            SessionBlock.RTH_PRIME1: self.cfg.spread_model_rth,
            SessionBlock.RTH_PRIME2: self.cfg.spread_model_rth,
            SessionBlock.RTH_DEAD: self.cfg.spread_model_rth,
            SessionBlock.ETH_QUALITY_AM: self.cfg.spread_model_eth_quality,
            SessionBlock.ETH_QUALITY_PM: self.cfg.spread_model_eth_quality,
            SessionBlock.ETH_EUROPE: self.cfg.spread_model_eth_europe,
            SessionBlock.ETH_OVERNIGHT: self.cfg.spread_model_overnight,
            SessionBlock.REOPEN_DEAD: self.cfg.spread_model_overnight,
            SessionBlock.DAILY_HALT: self.cfg.spread_model_overnight,
        }.get(block, self.cfg.spread_model_rth)

        vol_widen = 1.0 + 0.5 * max(0.0, (self._vol_pct - 50) / 50)
        return base * vol_widen

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _handle_fill(self, fill: FillResult, bar_time: datetime) -> None:
        order = fill.order

        if fill.status == FillStatus.FILLED:
            pending = self._pending.pop(order.order_id, None)
            if pending is not None:
                sibling = (pending.catchup_order_id if order.order_id == pending.order_id
                           else pending.order_id)
                self._pending.pop(sibling, None)
                et = self._entry_tracking_by_setup.get(pending.setup.setup_id)
                if et:
                    et.filled = True
                    et.fill_price = fill.fill_price
                    et.fill_time = bar_time
                self._on_entry_fill(pending, fill, bar_time)
            else:
                for ap in self._active_positions:
                    if order.order_id == ap.stop_order_id:
                        self._on_stop_fill(ap, fill, bar_time)
                        break

        elif fill.status in (FillStatus.EXPIRED, FillStatus.CANCELLED):
            pending = self._pending.pop(order.order_id, None)
            if pending:
                self._risk.release_pending_risk(
                    pending.setup.direction, pending.armed_risk_r)
                et = self._entry_tracking_by_setup.get(pending.setup.setup_id)
                if et and not et.filled:
                    et.expired = True
                    et.expire_time = bar_time

    def _on_entry_fill(
        self, pending: _PendingSetup, fill: FillResult, bar_time: datetime,
    ) -> None:
        fp = fill.fill_price
        commission = fill.commission
        self._total_commission += commission
        self.equity -= commission
        self._throttle.update_equity(self.equity)

        setup = pending.setup
        now_et = _to_et(bar_time)

        # Teleport check
        atr1 = self._h1.current_atr()
        is_teleport = self._teleport_check(fp, setup.entry_stop, now_et, atr1)

        pos = PositionState(
            pos_id=setup.setup_id,
            direction=setup.direction,
            avg_entry=fp,
            contracts=pending.contracts,
            unit1_risk_usd=pending.unit1_risk,
            origin_class=setup.cls,
            origin_setup_id=setup.setup_id,
            entry_ts=bar_time,
            stop_price=setup.stop0,
            teleport_penalty=is_teleport and self.flags.use_teleport_penalty,
            alignment_score_at_entry=setup.alignment_score,
            current_alignment_score=setup.alignment_score,
            entry_contracts=pending.contracts,
        )

        record = pending.record_template
        record.entry_time = bar_time
        record.avg_entry = fp
        record.teleport = is_teleport
        record.unit1_risk_usd = pending.unit1_risk
        record.dd_mult_at_entry = self._throttle.dd_size_mult
        record.entry_contracts = pending.contracts

        ap = _ActivePosition(
            pos=pos, record=record,
            mfe_price=fp, mae_price=fp,
            setup_id=setup.setup_id,
            entry_contracts=pending.contracts,
            commission_at_open=self._total_commission,
        )
        self._active_positions.append(ap)

        # Place protective stop
        stop_side = OrderSide.SELL if setup.direction == 1 else OrderSide.BUY
        stop_id = self.broker.next_order_id()
        stop_order = SimOrder(
            order_id=stop_id, symbol=self.symbol, side=stop_side,
            order_type=OrderType.STOP, qty=pending.contracts,
            stop_price=round_to_tick(setup.stop0, self.tick),
            tick_size=self.tick, submit_time=bar_time, ttl_hours=0,
            tag="protective_stop",
        )
        self.broker.submit_order(stop_order)
        ap.stop_order_id = stop_id

        # Risk tracking
        risk_r = self._risk.compute_risk_r(fp, setup.stop0, pending.contracts, pending.unit1_risk)
        pos.current_risk_r = risk_r
        self._risk.promote_to_open(setup.direction, pending.armed_risk_r)
        delta = risk_r - pending.armed_risk_r
        if abs(delta) > 1e-9:
            self._risk.adjust_open_risk(setup.direction, delta)

    def _on_stop_fill(self, ap: _ActivePosition, fill: FillResult, bar_time: datetime) -> None:
        commission = fill.commission
        self._total_commission += commission
        self.equity -= commission
        exit_reason = "TRAILING_STOP" if ap.pos.trailing_active else "INITIAL_STOP"
        self._close_position(ap, fill.fill_price, bar_time, exit_reason)

    # ------------------------------------------------------------------
    # Position management (v4.0: simplified 9-step loop)
    # ------------------------------------------------------------------

    def _manage_position(self, ap: _ActivePosition, now_et: datetime, last_price: float) -> None:
        """v4.0 management: single-phase trail, momentum stall, uniform partials."""
        pos = ap.pos
        pos.bars_held_1h += 1

        R_total = r_state(pos, last_price, self.pv)
        pos.peak_mfe_r = max(pos.peak_mfe_r, max(0.0, R_total))

        # Update alignment score
        pos.current_alignment_score = alignment_score(pos.direction, self._daily, self._h4)

        # 1) Catastrophic floor
        if R_total < self.cfg.catastrophic_r:
            self._close_position_market(ap, last_price, now_et, "CATASTROPHIC_STOP")
            return

        # 1b) Early adverse: -0.80R within 4 bars (tighter than v3.1)
        if pos.bars_held_1h <= EARLY_ADVERSE_BARS and R_total <= EARLY_ADVERSE_R:
            self._close_position_market(ap, last_price, now_et, "EARLY_ADVERSE")
            return

        # 2) Momentum stall: bar 8, R < 0.30, MFE < 0.80, not trailing
        if (self.flags.use_momentum_stall
                and not pos.trailing_active
                and pos.bars_held_1h >= MOMENTUM_STALL_BAR
                and R_total < MOMENTUM_STALL_MIN_R
                and pos.peak_mfe_r < MOMENTUM_STALL_MIN_MFE):
            self._close_position_market(ap, last_price, now_et, "MOMENTUM_STALL")
            return

        # 3) Trail activation at TRAIL_ACTIVATION_R + BE stop
        if not pos.trailing_active and R_total >= TRAIL_ACTIVATION_R:
            pos.trailing_active = True
            be_stop = pos.avg_entry
            self._tighten_stop(ap, be_stop)

        # 3b) +1R milestone
        if not pos.did_1r and R_total >= 1.0:
            pos.did_1r = True
            ap.record.hit_1r = True

        # 3c) Uniform partials (no alignment-conditional logic)
        entry_qty = pos.entry_contracts if pos.entry_contracts > 0 else pos.contracts

        # Partial 1 at +1.0R (30%)
        if (not pos.partial_done
                and R_total >= PARTIAL1_R
                and pos.contracts >= 2):
            partial_qty = max(1, int(entry_qty * PARTIAL1_FRAC))
            partial_qty = min(partial_qty, pos.contracts - 1)
            self._do_partial_exit(ap, partial_qty, last_price)
            pos.partial_done = True
            ap.record.partial_done = True

        # Partial 2 at +1.5R (30%)
        if (pos.partial_done and not pos.partial2_done
                and R_total >= PARTIAL2_R
                and pos.contracts >= 2):
            partial_qty = max(1, int(entry_qty * PARTIAL2_FRAC))
            partial_qty = min(partial_qty, pos.contracts - 1)
            self._do_partial_exit(ap, partial_qty, last_price)
            pos.partial2_done = True
            ap.record.partial2_done = True

        # 4) Single-phase chandelier trail (no bar-based phases)
        if self.flags.use_trailing and pos.trailing_active and pos.contracts > 0:
            new_stop = self._compute_trail(ap, last_price)
            self._tighten_stop(ap, new_stop)

        # 4a) MFE ratchet floor
        if (self.flags.use_mfe_ratchet
                and pos.trailing_active
                and pos.peak_mfe_r >= MFE_RATCHET_ACTIVATION_R
                and pos.contracts > 0):
            min_exit_r = pos.peak_mfe_r * MFE_RATCHET_FLOOR_PCT
            r_points = pos.unit1_risk_usd / (self.pv * max(pos.contracts, 1))
            if r_points > 0:
                floor_price = pos.avg_entry + pos.direction * min_exit_r * r_points
                floor_price = round_to_tick(floor_price, self.tick)
                self._tighten_stop(ap, floor_price)

        # 4b) Single time-decay: BE force at bar 12
        if self.flags.use_time_decay and not pos.trailing_active and pos.contracts > 0:
            if pos.bars_held_1h >= TIME_DECAY_BARS and R_total < TIME_DECAY_PROGRESS_R:
                self._tighten_stop(ap, pos.avg_entry)

        # 5) Early stale
        if self.flags.use_early_stale and self._should_early_stale(pos, R_total):
            self._close_position_market(ap, last_price, now_et, "EARLY_STALE")
            return

        # 6) Stale exit
        if self.flags.use_stale_exit and self._should_stale(pos, R_total):
            self._close_position_market(ap, last_price, now_et, "STALE")
            return

    def _do_partial_exit(self, ap: _ActivePosition, partial_qty: int, last_price: float) -> None:
        """Execute a partial exit: update equity, PnL, contracts, risk, stop qty."""
        pos = ap.pos
        pts_gained = (last_price - pos.avg_entry) * pos.direction
        partial_pnl = pts_gained * self.pv * partial_qty
        commission = self.cfg.slippage.commission_per_contract * partial_qty
        self._total_commission += commission
        self.equity += partial_pnl - commission
        pos.realized_partial_usd += partial_pnl
        pos.contracts -= partial_qty
        # Adjust risk tracking
        delta_r = abs(pos.avg_entry - pos.stop_price) * self.pv * partial_qty / max(pos.unit1_risk_usd, 1e-9)
        self._risk.adjust_open_risk(pos.direction, -delta_r)
        pos.current_risk_r = max(0, pos.current_risk_r - delta_r)
        # Update stop order qty in broker
        for o in self.broker.pending_orders:
            if o.order_id == ap.stop_order_id:
                o.qty = pos.contracts
                break

    def _compute_trail(self, ap: _ActivePosition, last_price: float) -> float:
        """Single-phase R-based trail: mult = max(1.5, 3.0 - R/5)."""
        pos = ap.pos
        R = r_state(pos, last_price, self.pv)

        mult = max(TRAIL_MULT_FLOOR, TRAIL_MULT_MAX - (R / TRAIL_R_DIVISOR))
        pos.trail_mult = mult

        atr1 = self._h1.current_atr()
        if pos.direction == 1:
            hh = self._h1.highest_high(TRAIL_LOOKBACK_1H)
            return hh - mult * atr1
        else:
            ll = self._h1.lowest_low(TRAIL_LOOKBACK_1H)
            return ll + mult * atr1

    def _should_early_stale(self, pos: PositionState, R_total: float) -> bool:
        if pos.bars_held_1h < EARLY_STALE_BARS:
            return False
        if pos.trailing_active:
            return False
        if R_total >= 0:
            return False

        if pos.bars_held_1h >= EARLY_STALE_MAX_BARS:
            return True

        if len(self._h1.bars) >= 3:
            last_close = self._h1.last_close
            prev_close = self._h1.close_n_ago(1)
            prev2_close = self._h1.close_n_ago(2)
            if pos.direction == 1:
                two_adverse = last_close < prev_close and prev_close < prev2_close
            else:
                two_adverse = last_close > prev_close and prev_close > prev2_close

            if two_adverse and R_total < -0.3:
                return True
            return False

        return True

    def _should_stale(self, pos: PositionState, R_total: float) -> bool:
        if R_total >= STALE_R_THRESHOLD:
            return False
        return pos.bars_held_1h >= STALE_M_BARS

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tighten_stop(self, ap: _ActivePosition, new_stop: float) -> None:
        """Tighten stop (never loosen)."""
        pos = ap.pos
        new_stop = round_to_tick(new_stop, self.tick)
        if pos.direction == 1:
            if new_stop <= pos.stop_price:
                return
        else:
            if new_stop >= pos.stop_price:
                return
        pos.stop_price = new_stop
        for o in self.broker.pending_orders:
            if o.order_id == ap.stop_order_id:
                o.stop_price = new_stop
                break

    def _close_position(
        self, ap: _ActivePosition, exit_price: float, bar_time: datetime, reason: str,
    ) -> None:
        pos = ap.pos
        if pos.contracts <= 0:
            return

        if pos.direction == 1:
            pnl = (exit_price - pos.avg_entry) * self.pv * pos.contracts
        else:
            pnl = (pos.avg_entry - exit_price) * self.pv * pos.contracts

        total_pnl = pnl + pos.realized_partial_usd
        r_mult = total_pnl / max(pos.unit1_risk_usd, 1e-9)

        self.equity += pnl
        self._throttle.update_equity(self.equity)
        self._throttle.record_trade_close(r_mult)

        # MFE/MAE in R
        qty_for_mfe = ap.entry_contracts if ap.entry_contracts > 0 else pos.contracts
        if pos.unit1_risk_usd > 0:
            if pos.direction == 1:
                mfe_r = (ap.mfe_price - pos.avg_entry) * self.pv * qty_for_mfe / pos.unit1_risk_usd
                mae_r = (pos.avg_entry - ap.mae_price) * self.pv * qty_for_mfe / pos.unit1_risk_usd
            else:
                mfe_r = (pos.avg_entry - ap.mfe_price) * self.pv * qty_for_mfe / pos.unit1_risk_usd
                mae_r = (ap.mae_price - pos.avg_entry) * self.pv * qty_for_mfe / pos.unit1_risk_usd
        else:
            mfe_r = mae_r = 0.0

        record = ap.record
        record.exit_time = bar_time
        record.exit_price = exit_price
        record.pnl_dollars = total_pnl
        record.r_multiple = r_mult
        record.mfe_r = mfe_r
        record.mae_r = mae_r
        record.exit_reason = reason
        record.bars_held_1h = pos.bars_held_1h
        record.commission = self._total_commission - ap.commission_at_open
        record.exit_contracts = pos.contracts
        record.partial2_done = pos.partial2_done

        self._trades.append(record)

        # Release risk
        self._risk.open_risk_r = max(0, self._risk.open_risk_r - pos.current_risk_r)
        d = pos.direction
        self._risk.dir_risk_r[d] = max(0, self._risk.dir_risk_r.get(d, 0) - pos.current_risk_r)

        # Signal engine exit tracking (for Class F re-entry detection)
        self._signals.record_exit(pos.direction, exit_reason=reason, bars_held=pos.bars_held_1h)

        pos.contracts = 0
        if ap in self._active_positions:
            self._active_positions.remove(ap)

    def _close_position_market(
        self, ap: _ActivePosition, price: float, bar_time: datetime, reason: str,
    ) -> None:
        """Close position with market-order slippage and commission applied."""
        slip = self.cfg.slippage.slip_ticks_normal * self.tick
        if ap.pos.direction == 1:  # LONG
            exit_price = price - slip
        else:
            exit_price = price + slip
        commission = self.cfg.slippage.commission_per_contract * ap.pos.contracts
        self._total_commission += commission
        self.equity -= commission
        self._close_position(ap, exit_price, bar_time, reason)

    def _update_mfe_mae(self, ap: _ActivePosition, H: float, L: float) -> None:
        if ap.pos.direction == 1:
            ap.mfe_price = max(ap.mfe_price, H)
            ap.mae_price = min(ap.mae_price, L)
        else:
            ap.mfe_price = min(ap.mfe_price, L)
            ap.mae_price = max(ap.mae_price, H)

    def _check_catastrophic_5m(
        self, ap: _ActivePosition, H: float, L: float, bar_time: datetime,
    ) -> None:
        pos = ap.pos
        catastrophic_r = self.cfg.catastrophic_r

        if pos.direction == 1:
            worst_price = L
        else:
            worst_price = H

        R_at_worst = unrealized_r(pos, worst_price, self.pv)
        if R_at_worst > catastrophic_r:
            return

        exit_price = pos.avg_entry + (
            catastrophic_r * pos.unit1_risk_usd / (pos.direction * self.pv * pos.contracts)
        )
        exit_price = round_to_tick(exit_price, self.tick)

        # Apply exit commission (slippage already baked into catastrophic exit_price)
        commission = self.cfg.slippage.commission_per_contract * pos.contracts
        self._total_commission += commission
        self.equity -= commission
        self._close_position(ap, exit_price, bar_time, "CATASTROPHIC_STOP")

    def _teleport_check(
        self, fill_price: float, trigger_price: float, now_et: datetime, atr1h: float,
    ) -> bool:
        slip_pts = abs(fill_price - trigger_price)
        block = get_session_block(now_et)
        if is_rth(block):
            thresh = max(TELEPORT_RTH_TICKS * self.tick, TELEPORT_RTH_ATR_FRAC * atr1h)
        else:
            thresh = max(TELEPORT_ETH_TICKS * self.tick, TELEPORT_ETH_ATR_FRAC * atr1h)
        return slip_pts > thresh

    def _cleanup_expired_pending(self, bar_time: datetime) -> None:
        active_ids = {o.order_id for o in self.broker.pending_orders}
        expired = [oid for oid in self._pending if oid not in active_ids]
        for oid in expired:
            self._pending.pop(oid, None)

    def _expire_signatures(self, now_et: datetime) -> None:
        expired = [sig for sig, exp in self._sig_expiry.items() if now_et >= exp]
        for sig in expired:
            self._placed_signatures.discard(sig)
            del self._sig_expiry[sig]

    def _load_news_calendar(self, path) -> None:
        import csv
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            logger.warning("News calendar not found: %s", p)
            return
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = datetime.fromisoformat(row["datetime"])
                event_type = row.get("event_type", row.get("type", ""))
                self._news.add_event(ts, event_type)

    @staticmethod
    def _bar_time(raw_time) -> datetime:
        if isinstance(raw_time, datetime):
            return raw_time
        if isinstance(raw_time, np.datetime64):
            ts = (raw_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            from zoneinfo import ZoneInfo
            return datetime.fromtimestamp(float(ts), tz=ZoneInfo("UTC"))
        return datetime.utcfromtimestamp(float(raw_time))
