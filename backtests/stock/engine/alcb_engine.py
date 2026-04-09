"""ALCB T1 intraday momentum continuation backtest engine.

5m bar replay of momentum breakout logic:
- Replays 78 bars/day (09:30-16:00 ET) for each tradable symbol
- Opening range build (first N bars, default 6 = 30 min)
- Entry triggers: OR_BREAKOUT, PDH_BREAKOUT, COMBINED_BREAKOUT
- Exit cascade: CLOSE_STOP → FLOW_REVERSAL → PARTIAL_TAKE → EOD_FLATTEN/CARRY
- Full portfolio constraints with SimBroker order fill simulation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from backtests.stock.analysis.alcb_shadow_tracker import (
    ALCBShadowTracker,
    ShadowSetup,
)
from backtests.stock.config import SlippageConfig
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.engine.research_replay import ResearchReplayEngine
from backtests.stock.engine.sim_broker import SimBroker
from backtests.stock.models import Direction as BTDirection, TradeRecord

from strategies.stock.alcb.config import StrategySettings
from strategies.stock.alcb.exits import (
    carry_eligible_momentum,
    classify_momentum_trade,
    should_exit_for_reversal,
    should_take_partial,
)
from strategies.stock.alcb.models import (
    CandidateArtifact,
    CandidateItem,
    Direction,
    MomentumSetup,
)
from strategies.stock.alcb.risk import (
    momentum_regime_mult,
    momentum_size_mult,
    momentum_stop_price,
)
from strategies.stock.alcb.signals import (
    adx_from_bars,
    atr_from_bars,
    close_location_value,
    compute_bar_rvol,
    compute_momentum_score,
    compute_opening_range,
    compute_session_avwap,
)

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_IBKR_COMM_PER_SHARE = 0.005


# ---------------------------------------------------------------------------
# Internal position state
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    symbol: str
    direction: Direction
    entry_price: float
    entry_time: datetime
    quantity: int
    qty_original: int
    risk_per_share: float
    stop: float
    current_stop: float
    sector: str
    regime_tier: str
    entry_type: str
    momentum_score: int
    momentum_setup: MomentumSetup | None
    avwap_at_entry: float
    signal_time: datetime
    signal_bar_index: int
    fill_bar_index: int
    reentry_sequence: int
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    partial_taken: bool = False
    partial_qty_exited: int = 0
    realized_partial_pnl: float = 0.0
    realized_partial_commission: float = 0.0
    realized_partial_slippage: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    hold_bars: int = 0
    carry_days: int = 0
    opened_date: date | None = None
    setup_tag: str = ""

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        if self.direction == Direction.LONG:
            return (price - self.entry_price) / self.risk_per_share
        return (self.entry_price - price) / self.risk_per_share


@dataclass
class _PendingEntry:
    symbol: str
    item: CandidateItem
    entry_type: str
    signal_time: datetime
    signal_bar_index: int
    signal_low: float
    or_low: float
    daily_atr: float
    avwap_at_signal: float
    momentum_score: int
    score_detail: dict
    setup: MomentumSetup
    regime_tier: str
    opened_date: date
    reentry_sequence: int


@dataclass
class _BreakoutArmState:
    or_armed: bool = True
    pdh_armed: bool = True
    reentry_count: int = 0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ALCBIntradayResult:
    """Result from the ALCB T1 momentum backtest."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, CandidateArtifact]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ALCBIntradayEngine:
    """T1 ALCB momentum backtest engine using 5m bars.

    Per trading day:
    1. Load CandidateArtifact from research replay
    2. Handle overnight carry positions (gap stops, carry timeout)
    3. Replay 5m bars: build opening range, scan for momentum breakouts,
       manage positions with signal-based exits
    """

    def __init__(self, config: ALCBBacktestConfig, replay: ResearchReplayEngine) -> None:
        self.config = config
        self.replay = replay
        self.settings = StrategySettings(**config.param_overrides) if config.param_overrides else StrategySettings()
        self.ablation = config.ablation
        self.broker = SimBroker(slippage_config=config.slippage or SlippageConfig())
        self._shadow_tracker: ALCBShadowTracker | None = None

    @property
    def shadow_tracker(self) -> ALCBShadowTracker | None:
        return self._shadow_tracker

    @shadow_tracker.setter
    def shadow_tracker(self, tracker: ALCBShadowTracker | None) -> None:
        self._shadow_tracker = tracker

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> ALCBIntradayResult:
        settings = self.settings
        ablation = self.ablation
        shadow = self._shadow_tracker

        start = date.fromisoformat(self.config.start_date)
        end = date.fromisoformat(self.config.end_date)

        equity = self.config.initial_equity
        trades: list[TradeRecord] = []
        equity_curve: list[float] = [equity]
        timestamps: list[date] = [start]
        daily_selections: dict[date, CandidateArtifact] = {}

        positions: dict[str, _Position] = {}
        prior_day: dict[str, tuple[float, float, float]] = {}

        current = start
        while current <= end:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            # ----------------------------------------------------------
            # Phase 1: Daily Setup
            # ----------------------------------------------------------
            try:
                artifact = self.replay.alcb_selection_for_date(current, settings)
            except Exception:
                current += timedelta(days=1)
                continue

            daily_selections[current] = artifact
            regime_tier = artifact.regime.tier

            # Regime gate — flatten carried positions on C-tier days
            if ablation.use_regime_gate and regime_tier == "C":
                for sym in list(positions):
                    pos = positions[sym]
                    bars_5m = self.replay.get_5m_bar_objects_for_date(sym, current)
                    if bars_5m:
                        exit_price = bars_5m[0].open
                        exit_time = bars_5m[0].start_time
                    else:
                        exit_price = pos.entry_price
                        exit_time = datetime.combine(current, time(9, 30))
                    closed = self._close_position(
                        pos, exit_price, exit_time, "REGIME_GATE_FLATTEN", settings,
                    )
                    trades.append(closed)
                    equity += closed.pnl_net
                    del positions[sym]
                equity_curve.append(equity)
                timestamps.append(current)
                current += timedelta(days=1)
                continue

            sym_items = {item.symbol: item for item in artifact.tradable}
            pending_entries: dict[str, _PendingEntry] = {}
            arm_states: dict[str, _BreakoutArmState] = {
                sym: _BreakoutArmState() for sym in sym_items
            }

            # ----------------------------------------------------------
            # Phase 2: Overnight Carry Processing
            # ----------------------------------------------------------
            for sym in list(positions):
                pos = positions[sym]
                bars_5m = self.replay.get_5m_bar_objects_for_date(sym, current)
                if not bars_5m:
                    closed = self._close_position(
                        pos, pos.entry_price,
                        datetime.combine(current, time(9, 30)),
                        "DATA_GAP", settings,
                    )
                    trades.append(closed)
                    equity += closed.pnl_net
                    del positions[sym]
                    continue

                open_price = bars_5m[0].open
                pos.carry_days += 1

                # Gap through stop
                gap_stopped = (
                    (pos.direction == Direction.LONG and open_price <= pos.current_stop)
                    or (pos.direction == Direction.SHORT and open_price >= pos.current_stop)
                )
                if gap_stopped:
                    closed = self._close_position(
                        pos, open_price, bars_5m[0].start_time, "GAP_STOP", settings,
                    )
                    trades.append(closed)
                    equity += closed.pnl_net
                    del positions[sym]
                    continue

                # Max carry days exceeded
                if pos.carry_days > settings.max_carry_days:
                    closed = self._close_position(
                        pos, open_price, bars_5m[0].start_time, "CARRY_TIMEOUT",
                        settings,
                    )
                    trades.append(closed)
                    equity += closed.pnl_net
                    del positions[sym]
                    continue

            # ----------------------------------------------------------
            # Phase 3: 5m Bar Replay
            # ----------------------------------------------------------
            n_or = settings.opening_range_bars

            today_symbols = list(sym_items.keys())

            or_built: dict[str, bool] = {}
            or_bars: dict[str, list] = {}
            or_data: dict[str, tuple[float, float, float]] = {}
            session_bars: dict[str, list] = {}
            qe_phantom_slots = [0]  # QE no-recycle: mutable counter for freed slots

            for sym in today_symbols:
                or_built[sym] = False
                or_bars[sym] = []
                session_bars[sym] = []

            # Gather 5m bars per symbol
            all_bars: dict[str, list] = {}
            for sym in today_symbols:
                bars = self.replay.get_5m_bar_objects_for_date(sym, current)
                if bars:
                    all_bars[sym] = bars
            for sym in positions:
                if sym not in all_bars:
                    bars = self.replay.get_5m_bar_objects_for_date(sym, current)
                    if bars:
                        all_bars[sym] = bars

            max_bars = max((len(b) for b in all_bars.values()), default=0)

            for bar_idx in range(max_bars):
                for sym, bars in all_bars.items():
                    if bar_idx >= len(bars):
                        continue
                    bar = bars[bar_idx]

                    # Track session bars
                    if sym not in session_bars:
                        session_bars[sym] = []
                    session_bars[sym].append(bar)

                    # Update shadow tracker for ALL symbols on every bar
                    if shadow:
                        shadow.update_bar(sym, bar.high, bar.low, bar.close)

                    self._fill_pending_entry(
                        sym,
                        bar,
                        bar_idx,
                        pending_entries,
                        positions,
                        regime_tier,
                        equity,
                        settings,
                        ablation,
                        qe_phantom_slots,
                        shadow,
                    )

                    # --- Position management ---
                    if sym in positions:
                        pos = positions[sym]
                        pos.hold_bars += 1

                        # Update MFE/MAE
                        if pos.direction == Direction.LONG:
                            pos.max_favorable = max(pos.max_favorable, bar.high)
                            pos.max_adverse = min(pos.max_adverse, bar.low)
                        else:
                            pos.max_favorable = min(pos.max_favorable, bar.low)
                            pos.max_adverse = max(pos.max_adverse, bar.high)

                        # Breakeven stop after reaching MFE threshold
                        if settings.close_stop_be_after_r > 0 and pos.risk_per_share > 0:
                            mfe_r = pos.unrealized_r(pos.max_favorable)
                            if mfe_r >= settings.close_stop_be_after_r:
                                be_price = pos.entry_price + 0.01 if pos.direction == Direction.LONG else pos.entry_price - 0.01
                                if pos.direction == Direction.LONG and be_price > pos.current_stop:
                                    pos.current_stop = be_price
                                elif pos.direction == Direction.SHORT and be_price < pos.current_stop:
                                    pos.current_stop = be_price

                        exited = self._exit_cascade(
                            pos, sym, bar, session_bars, positions,
                            trades, regime_tier, settings, ablation,
                            qe_phantom_slots,
                        )
                        if exited:
                            equity += trades[-1].pnl_net
                        if sym in positions:
                            continue  # still open after this bar

                    # --- Opening Range Build ---
                    if sym in or_built and not or_built[sym]:
                        or_bars[sym].append(bar)
                        if len(or_bars[sym]) >= n_or:
                            oh, ol, ov = compute_opening_range(or_bars[sym], n_or)
                            or_data[sym] = (oh, ol, ov)
                            or_built[sym] = True
                        continue

                    self._update_rearm_state(
                        sym,
                        bar,
                        session_bars,
                        or_data,
                        prior_day,
                        sym_items,
                        arm_states,
                    )

                    # --- Entry Logic ---
                    self._try_entry(
                        sym, bar, current, session_bars, or_data, sym_items,
                        prior_day, positions, pending_entries, arm_states,
                        regime_tier, equity, settings, ablation, shadow,
                        qe_phantom_slots, bar_idx, bar_idx + 1 < len(bars),
                    )

            # --- Update prior day cache ---
            for sym, bars_list in all_bars.items():
                if bars_list:
                    prior_day[sym] = (
                        max(b.high for b in bars_list),
                        min(b.low for b in bars_list),
                        bars_list[-1].close,
                    )

            equity_curve.append(equity)
            timestamps.append(current)
            current += timedelta(days=1)

        # --- Close remaining positions ---
        for sym in list(positions):
            pos = positions[sym]
            closed = self._close_position(
                pos, pos.entry_price,
                datetime.combine(end, time(16, 0)),
                "BACKTEST_END", settings,
            )
            trades.append(closed)
            equity += closed.pnl_net

        if shadow:
            shadow.flush_stale()

        return ALCBIntradayResult(
            trades=trades,
            equity_curve=np.array(equity_curve, dtype=np.float64),
            timestamps=np.array(timestamps),
            daily_selections=daily_selections,
        )

    # ------------------------------------------------------------------
    # Exit cascade
    # ------------------------------------------------------------------

    def _exit_cascade(
        self,
        pos: _Position,
        sym: str,
        bar,
        session_bars: dict[str, list],
        positions: dict[str, _Position],
        trades: list[TradeRecord],
        regime_tier: str,
        settings: StrategySettings,
        ablation,
        qe_phantom_slots: list[int] | None = None,
    ) -> bool:
        """Run exit cascade. Returns True if position was closed (appends to trades)."""

        # 1. CLOSE_STOP
        stop_hit = (
            (pos.direction == Direction.LONG and bar.low <= pos.current_stop)
            or (pos.direction == Direction.SHORT and bar.high >= pos.current_stop)
        )
        if stop_hit:
            closed = self._close_position(
                pos, pos.current_stop, bar.start_time, "CLOSE_STOP", settings,
            )
            trades.append(closed)
            del positions[sym]
            return True

        # 1b-i. QUICK_EXIT STAGE 1 (cut deeply underwater trades early)
        if ablation.use_quick_exit_stage1 and settings.qe_stage1_bars > 0:
            if pos.hold_bars == settings.qe_stage1_bars:
                ur = pos.unrealized_r(bar.close)
                if ur < settings.qe_stage1_min_r:
                    closed = self._close_position(
                        pos, bar.close, bar.start_time, "QUICK_EXIT", settings,
                    )
                    trades.append(closed)
                    del positions[sym]
                    if qe_phantom_slots is not None and ablation.use_qe_no_recycle:
                        qe_phantom_slots[0] += 1
                    return True

        # 1b-ii. MFE CONVICTION EXIT (kill trades that never showed conviction)
        if ablation.use_mfe_conviction_exit and settings.mfe_conviction_check_bars > 0:
            if pos.hold_bars == settings.mfe_conviction_check_bars:
                mfe_r = pos.unrealized_r(pos.max_favorable) if pos.risk_per_share > 0 else 0.0
                if mfe_r < settings.mfe_conviction_min_r:
                    # Compound mode: also require current R below floor
                    if settings.mfe_conviction_floor_r != 0.0:
                        current_r = pos.unrealized_r(bar.close)
                        if current_r >= settings.mfe_conviction_floor_r:
                            pass  # Trade is recovering, keep it
                        else:
                            closed = self._close_position(
                                pos, bar.close, bar.start_time, "MFE_CONVICTION", settings,
                            )
                            trades.append(closed)
                            del positions[sym]
                            return True
                    else:
                        closed = self._close_position(
                            pos, bar.close, bar.start_time, "MFE_CONVICTION", settings,
                        )
                        trades.append(closed)
                        del positions[sym]
                        return True

        # 1b. TIME_BASED_QUICK_EXIT (cut short-hold losers before they bleed)
        if ablation.use_time_based_quick_exit and settings.quick_exit_max_bars > 0:
            if pos.hold_bars == settings.quick_exit_max_bars:
                ur = pos.unrealized_r(bar.close)
                if ur < settings.quick_exit_min_r:
                    closed = self._close_position(
                        pos, bar.close, bar.start_time, "QUICK_EXIT", settings,
                    )
                    trades.append(closed)
                    del positions[sym]
                    if qe_phantom_slots is not None and ablation.use_qe_no_recycle:
                        qe_phantom_slots[0] += 1
                    return True

        # 2. FLOW_REVERSAL
        if ablation.use_flow_reversal_exit:
            # Skip FR if position has reached significant MFE (profit protection)
            mfe_r = pos.unrealized_r(pos.max_favorable) if pos.risk_per_share > 0 else 0.0
            skip_fr = settings.fr_mfe_grace_r > 0 and mfe_r >= settings.fr_mfe_grace_r

            # FR conditional gating: only trigger under specific conditions
            if not skip_fr and settings.fr_max_hold_bars > 0:
                if pos.hold_bars > settings.fr_max_hold_bars:
                    skip_fr = True  # too late for FR, let trailing stop handle it
            if not skip_fr and settings.fr_cpr_threshold > 0:
                bar_cpr = close_location_value(bar)
                if bar_cpr >= settings.fr_cpr_threshold:
                    skip_fr = True  # bar still closing strong, skip FR

            if not skip_fr:
                sb = session_bars.get(sym, [])
                recent = sb[-8:]
                avwap = compute_session_avwap(sb, len(sb) - 1) if sb else 0.0
                if should_exit_for_reversal(
                    recent, pos.entry_price, avwap,
                    hold_bars=pos.hold_bars,
                    min_hold_bars=settings.flow_reversal_min_hold_bars,
                    require_below_entry=settings.flow_reversal_require_below_entry,
                ):
                    closed = self._close_position(
                        pos, bar.close, bar.start_time, "FLOW_REVERSAL", settings,
                    )
                    trades.append(closed)
                    del positions[sym]
                    return True

        # 2b. MFE-ACTIVATED TRAILING STOP (replaces/supplements FR)
        if settings.fr_trailing_activate_r > 0:
            mfe_r = pos.unrealized_r(pos.max_favorable) if pos.risk_per_share > 0 else 0.0
            if mfe_r >= settings.fr_trailing_activate_r:
                trail_r = mfe_r - settings.fr_trailing_distance_r
                if trail_r > 0:
                    trail_price = (
                        pos.entry_price + trail_r * pos.risk_per_share
                        if pos.direction == Direction.LONG
                        else pos.entry_price - trail_r * pos.risk_per_share
                    )
                    if pos.direction == Direction.LONG and trail_price > pos.current_stop:
                        pos.current_stop = trail_price
                    elif pos.direction == Direction.SHORT and trail_price < pos.current_stop:
                        pos.current_stop = trail_price

        # 2c. ADAPTIVE TRAILING STOP (time-phased: no trail -> wide -> tight)
        if ablation.use_adaptive_trail and settings.adaptive_trail_start_bars > 0:
            if pos.hold_bars >= settings.adaptive_trail_start_bars:
                mfe_r = pos.unrealized_r(pos.max_favorable) if pos.risk_per_share > 0 else 0.0
                if pos.hold_bars >= settings.adaptive_trail_tighten_bars:
                    activate_r = settings.adaptive_trail_late_activate_r
                    distance_r = settings.adaptive_trail_late_distance_r
                else:
                    activate_r = settings.adaptive_trail_mid_activate_r
                    distance_r = settings.adaptive_trail_mid_distance_r
                if mfe_r >= activate_r:
                    trail_r = mfe_r - distance_r
                    if trail_r > 0:
                        trail_price = (
                            pos.entry_price + trail_r * pos.risk_per_share
                            if pos.direction == Direction.LONG
                            else pos.entry_price - trail_r * pos.risk_per_share
                        )
                        if pos.direction == Direction.LONG and trail_price > pos.current_stop:
                            pos.current_stop = trail_price
                        elif pos.direction == Direction.SHORT and trail_price < pos.current_stop:
                            pos.current_stop = trail_price

        # 3. PARTIAL_TAKE
        if ablation.use_partial_takes:
            ur = pos.unrealized_r(bar.close)
            take, frac = should_take_partial(ur, pos.partial_taken, settings)
            if take:
                partial_qty = max(1, int(pos.quantity * frac))
                if partial_qty < pos.quantity:
                    partial_fill, partial_slip = self._market_fill_price(
                        bar.close,
                        pos.direction,
                        is_entry=False,
                    )
                    if pos.direction == Direction.LONG:
                        partial_pnl = (partial_fill - pos.entry_price) * partial_qty
                    else:
                        partial_pnl = (pos.entry_price - partial_fill) * partial_qty
                    comm = self.broker.slippage_config.commission_per_share * partial_qty
                    pos.partial_taken = True
                    pos.partial_qty_exited += partial_qty
                    pos.realized_partial_pnl += partial_pnl
                    pos.realized_partial_commission += comm
                    pos.realized_partial_slippage += partial_slip * partial_qty
                    pos.quantity -= partial_qty
                    if settings.move_stop_to_be:
                        pos.current_stop = pos.entry_price

        # 4. EOD check
        bar_time_et = bar.start_time.astimezone(_ET).time()
        if bar_time_et >= settings.eod_flatten_time:
            if ablation.use_carry_logic:
                sb = session_bars.get(sym, [])
                recent = sb[-8:]
                avwap = compute_session_avwap(sb, len(sb) - 1) if sb else 0.0
                trade_class = classify_momentum_trade(recent, pos.entry_price, avwap)
                ur = pos.unrealized_r(bar.close)
                eod_cpr = close_location_value(bar)
                eligible, _ = carry_eligible_momentum(
                    trade_class, ur, eod_cpr, regime_tier, settings,
                )
                if eligible:
                    return False  # hold overnight

            closed = self._close_position(
                pos, bar.close, bar.start_time, "EOD_FLATTEN", settings,
            )
            trades.append(closed)
            del positions[sym]
            return True

        return False

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _market_fill_price(
        self,
        price: float,
        direction: Direction,
        *,
        is_entry: bool,
    ) -> tuple[float, float]:
        """Return (slipped_fill_price, slippage_per_share)."""

        slip_per_share = price * (self.broker.slippage_config.slip_bps_normal / 10_000)
        if direction == Direction.LONG:
            fill_price = price + slip_per_share if is_entry else price - slip_per_share
        else:
            fill_price = price - slip_per_share if is_entry else price + slip_per_share
        return round(fill_price, 2), slip_per_share

    def _consume_arms(self, state: _BreakoutArmState, entry_type: str) -> None:
        if entry_type in {"OR_BREAKOUT", "COMBINED_BREAKOUT"}:
            state.or_armed = False
        if entry_type in {"PDH_BREAKOUT", "COMBINED_BREAKOUT"}:
            state.pdh_armed = False

    def _update_rearm_state(
        self,
        sym: str,
        bar,
        session_bars: dict[str, list],
        or_data: dict[str, tuple[float, float, float]],
        prior_day: dict[str, tuple[float, float, float]],
        sym_items: dict[str, CandidateItem],
        arm_states: dict[str, _BreakoutArmState],
    ) -> None:
        state = arm_states.get(sym)
        item = sym_items.get(sym)
        if state is None or item is None or sym not in or_data:
            return

        sb = session_bars.get(sym, [])
        if not sb:
            return
        avwap = compute_session_avwap(sb, len(sb) - 1)
        if avwap <= 0 or bar.close > avwap:
            return

        or_high, _, _ = or_data[sym]
        if bar.close <= or_high:
            state.or_armed = True

        pdh, _, _ = prior_day.get(sym, (0.0, 0.0, 0.0))
        if pdh == 0.0 and item.daily_bars:
            pdh = item.daily_bars[-1].high
        if pdh > 0 and bar.close <= pdh:
            state.pdh_armed = True

    def _fill_pending_entry(
        self,
        sym: str,
        bar,
        bar_idx: int,
        pending_entries: dict[str, _PendingEntry],
        positions: dict[str, _Position],
        regime_tier: str,
        equity: float,
        settings: StrategySettings,
        ablation,
        qe_phantom_slots: list[int] | None = None,
        shadow: ALCBShadowTracker | None = None,
    ) -> None:
        pending = pending_entries.get(sym)
        if pending is None:
            return
        if sym in positions:
            pending_entries.pop(sym, None)
            return

        entry_price, entry_slip = self._market_fill_price(
            bar.open,
            Direction.LONG,
            is_entry=True,
        )
        stop_price = momentum_stop_price(
            entry_price,
            pending.or_low,
            pending.signal_low,
            pending.daily_atr,
            settings,
        )
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            pending_entries.pop(sym, None)
            return

        item = pending.item
        n_open = len(positions)
        if qe_phantom_slots is not None:
            n_open += qe_phantom_slots[0]
        if n_open >= settings.max_positions:
            pending_entries.pop(sym, None)
            return

        if ablation.use_sector_limit:
            sector_count = sum(1 for p in positions.values() if p.sector == item.sector)
            if sector_count >= settings.max_positions_per_sector:
                pending_entries.pop(sym, None)
                return

        if ablation.use_heat_cap:
            open_risk = sum(p.risk_per_share * p.quantity for p in positions.values())
            if open_risk >= settings.heat_cap_r * (equity * settings.base_risk_fraction):
                pending_entries.pop(sym, None)
                return

        reg_mult = momentum_regime_mult(pending.regime_tier, settings)
        size_mult = momentum_size_mult(pending.momentum_score, settings)
        dow_mult = settings.thursday_sizing_mult if pending.opened_date.weekday() == 3 else settings.tuesday_sizing_mult if pending.opened_date.weekday() == 1 else 1.0

        sector_mult = 1.0
        if item.sector == "Financials":
            sector_mult = settings.sector_mult_financials
        elif item.sector == "Communication Services":
            sector_mult = settings.sector_mult_communication
        elif item.sector == "Industrials":
            sector_mult = settings.sector_mult_industrials
        elif item.sector == "Consumer Discretionary":
            sector_mult = settings.sector_mult_consumer_disc
        elif item.sector == "Healthcare":
            sector_mult = settings.sector_mult_healthcare

        risk_budget = (
            equity
            * settings.base_risk_fraction
            * reg_mult
            * size_mult
            * dow_mult
            * sector_mult
        )
        qty = int(risk_budget / risk_per_share)
        if qty <= 0:
            pending_entries.pop(sym, None)
            return

        if settings.intraday_leverage > 0:
            total_notional = sum(p.entry_price * p.quantity for p in positions.values())
            available_bp = equity * settings.intraday_leverage - total_notional
            max_qty_bp = int(available_bp / entry_price)
            qty = min(qty, max_qty_bp)
            if qty <= 0:
                pending_entries.pop(sym, None)
                return

        if item.average_30m_volume > 0:
            max_qty = int(item.average_30m_volume * settings.max_participation_30m)
            qty = min(qty, max(1, max_qty))
            if qty <= 0:
                pending_entries.pop(sym, None)
                return

        commission_entry = self.broker.slippage_config.commission_per_share * qty
        pos = _Position(
            symbol=sym,
            direction=Direction.LONG,
            entry_price=entry_price,
            entry_time=bar.start_time,
            quantity=qty,
            qty_original=qty,
            risk_per_share=risk_per_share,
            stop=stop_price,
            current_stop=stop_price,
            sector=item.sector,
            regime_tier=pending.regime_tier,
            entry_type=pending.entry_type,
            momentum_score=pending.momentum_score,
            momentum_setup=pending.setup,
            avwap_at_entry=pending.avwap_at_signal,
            signal_time=pending.signal_time,
            signal_bar_index=pending.signal_bar_index,
            fill_bar_index=bar_idx,
            reentry_sequence=pending.reentry_sequence,
            commission_entry=commission_entry,
            slippage_entry=entry_slip * qty,
            max_favorable=entry_price,
            max_adverse=entry_price,
            opened_date=pending.opened_date,
            setup_tag=f"T1_{pending.entry_type}",
        )
        positions[sym] = pos
        pending_entries.pop(sym, None)

        if shadow:
            shadow.record_funnel("entered")

    def _try_entry(
        self,
        sym: str,
        bar,
        current: date,
        session_bars: dict[str, list],
        or_data: dict[str, tuple[float, float, float]],
        sym_items: dict[str, CandidateItem],
        prior_day: dict[str, tuple[float, float, float]],
        positions: dict[str, _Position],
        pending_entries: dict[str, _PendingEntry],
        arm_states: dict[str, _BreakoutArmState],
        regime_tier: str,
        equity: float,
        settings: StrategySettings,
        ablation,
        shadow: ALCBShadowTracker | None,
        qe_phantom_slots: list[int] | None = None,
        bar_idx: int = -1,
        has_next_bar: bool = True,
    ) -> None:
        """Schedule a next-bar market entry from a fully closed signal bar."""
        if sym not in or_data:
            return
        if sym in positions or sym in pending_entries:
            return

        signal_time = getattr(bar, "end_time", None) or (bar.start_time + timedelta(minutes=5))
        bar_time_et = signal_time.astimezone(_ET).time()
        if bar_time_et < settings.entry_window_start or bar_time_et > settings.entry_window_end:
            return

        item = sym_items.get(sym)
        arm_state = arm_states.get(sym)
        if item is None or arm_state is None:
            return

        # --- Compute all signal values upfront ---
        pdh, pdl, pdc = prior_day.get(sym, (0.0, 0.0, 0.0))
        if pdh == 0.0 and item.daily_bars:
            last_daily = item.daily_bars[-1]
            pdh, pdl, pdc = last_daily.high, last_daily.low, last_daily.close

        or_high, or_low, or_vol = or_data[sym]
        # CandidateItem has average_30m_volume; 30m bar = 6 × 5m bars
        expected_vol = (item.average_30m_volume / 6.0) if item.average_30m_volume > 0 else 1.0
        bar_rvol = compute_bar_rvol(bar.volume, expected_vol)
        cpr = close_location_value(bar)

        sb = session_bars.get(sym, [])
        avwap = compute_session_avwap(sb, len(sb) - 1) if sb else 0.0

        daily_bars = item.daily_bars
        # adx_from_bars and atr_from_bars work with ResearchDailyBar via duck typing
        daily_atr = atr_from_bars(daily_bars, 14) if len(daily_bars) >= 2 else 0.0
        adx_val = adx_from_bars(daily_bars, 14) if len(daily_bars) >= 16 else 0.0

        # Sector flow not available in CandidateItem; max momentum score is 7/8
        sector_flow = 0.0

        signal_price = bar.close
        stop_price = momentum_stop_price(signal_price, or_low, bar.low, daily_atr, settings)
        risk_per_share = abs(signal_price - stop_price)

        if shadow:
            shadow.record_funnel("evaluated")

        if ablation.use_rvol_filter and bar_rvol < settings.rvol_threshold:
            return
        if ablation.use_cpr_filter and cpr < settings.cpr_threshold:
            return

        above_or = arm_state.or_armed and bar.close > or_high
        above_pdh = (
            ablation.use_prior_day_high_breakout
            and arm_state.pdh_armed
            and bar.close > pdh
        )
        if above_or and above_pdh and ablation.use_combined_breakout:
            entry_type_str = "COMBINED_BREAKOUT"
        elif above_or:
            entry_type_str = "OR_BREAKOUT"
        elif above_pdh:
            entry_type_str = "PDH_BREAKOUT"
        else:
            return

        # Block COMBINED_BREAKOUT in Tier B when configured
        if (entry_type_str == "COMBINED_BREAKOUT"
                and settings.block_combined_regime_b
                and regime_tier == "B"):
            if shadow:
                shadow.record_funnel("block_combined_regime_b")
            return

        if shadow:
            shadow.record_funnel("entry_signal")

        # --- Momentum score ---
        m_score, score_detail = compute_momentum_score(
            bar, sb, pdh, pdc, or_high, avwap, adx_val, sector_flow, settings,
            use_avwap_filter=ablation.use_avwap_filter,
        )

        def _reject(gate: str) -> None:
            if shadow and risk_per_share > 0:
                shadow.record_funnel(gate)
                shadow.record_rejection(ShadowSetup(
                    symbol=sym,
                    trade_date=current,
                    rejection_gate=gate,
                    direction=Direction.LONG,
                    entry_price=signal_price,
                    stop_price=stop_price,
                    momentum_score=m_score,
                    rvol_at_rejection=bar_rvol,
                    entry_type=entry_type_str,
                ))

        # --- Gate checks ---

        # AVWAP filter
        if ablation.use_avwap_filter and avwap > 0 and bar.close < avwap:
            _reject("avwap_filter")
            return

        # RVOL cap filter
        if settings.rvol_max < 999 and bar_rvol > settings.rvol_max:
            _reject("rvol_max_cap")
            return

        # Daily ATR dollar filter (quality: prefer high-ATR movers)
        if settings.min_daily_atr_usd > 0 and daily_atr < settings.min_daily_atr_usd:
            _reject("min_atr_usd")
            return

        # Selection score filter (quality: prefer high-ranked symbols)
        if settings.min_selection_score > 0 and item.selection_score < settings.min_selection_score:
            _reject("min_selection_score")
            return

        # Relative strength filter (quality: prefer RS leaders)
        if settings.min_rs_percentile > 0 and item.relative_strength_percentile < settings.min_rs_percentile:
            _reject("min_rs_percentile")
            return

        # Momentum score gate (with late-entry escalation)
        effective_score_min = settings.momentum_score_min
        if settings.late_entry_score_min > 0 and bar_time_et >= settings.late_entry_cutoff:
            effective_score_min = max(effective_score_min, settings.late_entry_score_min)
        if ablation.use_momentum_score_gate and m_score < effective_score_min:
            _reject("momentum_score_gate")
            return

        # COMBINED_BREAKOUT quality gate (require higher score / RVOL for weaker entry type)
        if ablation.use_combined_quality_gate and entry_type_str == "COMBINED_BREAKOUT":
            if settings.combined_breakout_score_min > 0 and m_score < settings.combined_breakout_score_min:
                _reject("combined_quality_score")
                return
            if settings.combined_breakout_min_rvol > 0 and bar_rvol < settings.combined_breakout_min_rvol:
                _reject("combined_quality_rvol")
                return
            # COMBINED-specific AVWAP distance cap
            if settings.combined_avwap_cap_pct > 0 and avwap > 0:
                avwap_dist_pct = (bar.close - avwap) / avwap
                if avwap_dist_pct > settings.combined_avwap_cap_pct:
                    _reject("combined_avwap_cap")
                    return
            # COMBINED-specific breakout distance cap
            if settings.combined_breakout_cap_r > 0 and risk_per_share > 0:
                breakout_dist_r = (bar.close - or_high) / risk_per_share
                if breakout_dist_r > settings.combined_breakout_cap_r:
                    _reject("combined_breakout_cap")
                    return

        # OR_BREAKOUT quality gate (require higher score / RVOL for OR entries)
        if ablation.use_or_quality_gate and entry_type_str == "OR_BREAKOUT":
            if settings.or_breakout_score_min > 0 and m_score < settings.or_breakout_score_min:
                _reject("or_quality_score")
                return
            if settings.or_breakout_min_rvol > 0 and bar_rvol < settings.or_breakout_min_rvol:
                _reject("or_quality_rvol")
                return

        # AVWAP distance cap (block extended entries too far above AVWAP)
        if ablation.use_avwap_distance_cap and settings.avwap_distance_cap_pct > 0 and avwap > 0:
            avwap_dist_pct = (bar.close - avwap) / avwap
            if avwap_dist_pct > settings.avwap_distance_cap_pct:
                _reject("avwap_distance_cap")
                return

        # OR width minimum (block tight opening ranges)
        if ablation.use_or_width_min and settings.or_width_min_pct > 0:
            or_width_pct = (or_high - or_low) / or_high if or_high > 0 else 0
            if or_width_pct < settings.or_width_min_pct:
                _reject("or_width_min")
                return

        # Breakout distance cap (block entries too far from OR high)
        if ablation.use_breakout_distance_cap and settings.breakout_distance_cap_r > 0 and risk_per_share > 0:
            breakout_dist_r = (bar.close - or_high) / risk_per_share
            if breakout_dist_r > settings.breakout_distance_cap_r:
                _reject("breakout_distance_cap")
                return

        # Long-only gate
        if ablation.use_long_only and item.direction_bias == "SHORT":
            _reject("long_only")
            return

        # Portfolio limits
        n_open = len(positions)
        if qe_phantom_slots is not None:
            n_open += qe_phantom_slots[0]
        if n_open >= settings.max_positions:
            _reject("max_positions")
            return

        if ablation.use_sector_limit:
            sector_count = sum(1 for p in positions.values() if p.sector == item.sector)
            if sector_count >= settings.max_positions_per_sector:
                _reject("sector_limit")
                return

        if ablation.use_heat_cap:
            open_risk = sum(p.risk_per_share * p.quantity for p in positions.values())
            if open_risk >= settings.heat_cap_r * (equity * settings.base_risk_fraction):
                _reject("heat_cap")
                return

        # --- Sizing ---
        if risk_per_share <= 0:
            return

        reg_mult = momentum_regime_mult(regime_tier, settings)
        size_mult = momentum_size_mult(m_score, settings)
        dow_mult = settings.thursday_sizing_mult if current.weekday() == 3 else settings.tuesday_sizing_mult if current.weekday() == 1 else 1.0
        # Sector-weighted sizing
        sector_mult = 1.0
        if item.sector == "Financials":
            sector_mult = settings.sector_mult_financials
        elif item.sector == "Communication Services":
            sector_mult = settings.sector_mult_communication
        elif item.sector == "Industrials":
            sector_mult = settings.sector_mult_industrials
        elif item.sector == "Consumer Discretionary":
            sector_mult = settings.sector_mult_consumer_disc
        elif item.sector == "Healthcare":
            sector_mult = settings.sector_mult_healthcare
        risk_budget = equity * settings.base_risk_fraction * reg_mult * size_mult * dow_mult * sector_mult
        qty = int(risk_budget / risk_per_share)
        if qty <= 0:
            return

        # Buying power constraint
        if settings.intraday_leverage > 0:
            total_notional = sum(p.entry_price * p.quantity for p in positions.values())
            available_bp = equity * settings.intraday_leverage - total_notional
            max_qty_bp = int(available_bp / signal_price)
            qty = min(qty, max_qty_bp)
            if qty <= 0:
                _reject("buying_power")
                return

        if item.average_30m_volume > 0:
            max_qty = int(item.average_30m_volume * settings.max_participation_30m)
            qty = min(qty, max(1, max_qty))

        if not has_next_bar:
            return

        setup = MomentumSetup(
            symbol=sym,
            or_high=or_high,
            or_low=or_low,
            or_volume=or_vol,
            prior_day_high=pdh,
            prior_day_low=pdl,
            prior_day_close=pdc,
            breakout_level=or_high if "OR" in entry_type_str else pdh,
            entry_type=entry_type_str,
            rvol_at_entry=bar_rvol,
            momentum_score=m_score,
            score_detail=score_detail,
            avwap_at_entry=avwap,
        )

        self._consume_arms(arm_state, entry_type_str)
        pending_entries[sym] = _PendingEntry(
            symbol=sym,
            item=item,
            entry_type=entry_type_str,
            signal_time=signal_time,
            signal_bar_index=bar_idx,
            signal_low=bar.low,
            or_low=or_low,
            daily_atr=daily_atr,
            avwap_at_signal=avwap,
            momentum_score=m_score,
            score_detail=score_detail,
            setup=setup,
            regime_tier=regime_tier,
            opened_date=current,
            reentry_sequence=arm_state.reentry_count,
        )
        arm_state.reentry_count += 1

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(
        self,
        pos: _Position,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        settings: StrategySettings,
    ) -> TradeRecord:
        """Close a position and produce a TradeRecord."""
        exit_fill, exit_slip = self._market_fill_price(
            exit_price,
            pos.direction,
            is_entry=False,
        )
        if pos.direction == Direction.LONG:
            pnl = (exit_fill - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_fill) * pos.quantity
        pnl += pos.realized_partial_pnl
        total_qty = pos.qty_original

        comm_rate = self.broker.slippage_config.commission_per_share
        commission = pos.commission_entry + comm_rate * pos.quantity + pos.realized_partial_commission
        slippage = pos.slippage_entry + exit_slip * pos.quantity + pos.realized_partial_slippage

        # R-multiple
        r_mult = 0.0
        if pos.risk_per_share > 0 and total_qty > 0:
            total_risk = pos.risk_per_share * total_qty
            r_mult = (pnl - commission) / total_risk

        # MFE/MAE in R
        mfe_r = 0.0
        mae_r = 0.0
        if pos.risk_per_share > 0:
            if pos.direction == Direction.LONG:
                mfe_r = (pos.max_favorable - pos.entry_price) / pos.risk_per_share
                mae_r = (pos.entry_price - pos.max_adverse) / pos.risk_per_share
            else:
                mfe_r = (pos.entry_price - pos.max_favorable) / pos.risk_per_share
                mae_r = (pos.max_adverse - pos.entry_price) / pos.risk_per_share

        metadata: dict = {
            "momentum_score": pos.momentum_score,
            "entry_type": pos.entry_type,
            "avwap_at_entry": pos.avwap_at_entry,
            "carry_days": pos.carry_days,
            "partial_taken": pos.partial_taken,
            "mfe_r": round(mfe_r, 4),
            "mae_r": round(mae_r, 4),
            "regime_tier": pos.regime_tier,
            "signal_time": pos.signal_time.isoformat(),
            "signal_bar_index": pos.signal_bar_index,
            "fill_time": pos.entry_time.isoformat(),
            "fill_bar_index": pos.fill_bar_index,
            "reentry_sequence": pos.reentry_sequence,
        }
        if pos.momentum_setup:
            metadata["or_high"] = pos.momentum_setup.or_high
            metadata["or_low"] = pos.momentum_setup.or_low
            metadata["rvol_at_entry"] = pos.momentum_setup.rvol_at_entry
            metadata["breakout_level"] = pos.momentum_setup.breakout_level

        bt_dir = BTDirection.LONG if pos.direction == Direction.LONG else BTDirection.SHORT

        return TradeRecord(
            strategy="alcb",
            symbol=pos.symbol,
            direction=bt_dir,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_fill,
            quantity=total_qty,
            pnl=pnl,
            r_multiple=r_mult,
            risk_per_share=pos.risk_per_share,
            commission=commission,
            slippage=slippage,
            entry_type=pos.entry_type,
            exit_reason=exit_reason,
            sector=pos.sector,
            regime_tier=pos.regime_tier,
            hold_bars=pos.hold_bars,
            max_favorable=pos.max_favorable,
            max_adverse=pos.max_adverse,
            metadata=metadata,
            signal_time=pos.signal_time,
            signal_bar_index=pos.signal_bar_index,
            fill_time=pos.entry_time,
            fill_bar_index=pos.fill_bar_index,
            reentry_sequence=pos.reentry_sequence,
        )
