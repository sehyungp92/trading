"""Tier 2 v3 IARIC intraday (5m bar) backtest engine.

FSM-driven 5m intraday engine with overnight carry layer.

With all v3_* flags OFF: identical behavior to T1 FSM engine (iaric_engine.py)
except exits are tagged consistently for comparison.

Architecture: T1 FSM core + carry layer + v3 enhancement hooks.

Entry:  FSM state machine (detect_setup → lock_setup → acceptance → entry)
Exit:   T1 cascade (stop → time_stop → avwap_breakdown → partial → eod)
Carry:  Binary carry_eligible + optional carry_score fallback
Hooks:  Adaptive trail, entry improvement, PM re-entry, staleness tighten
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from math import floor
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np

from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.models import Direction as BTDirection, TradeRecord

from strategies.stock.iaric.config import ET, StrategySettings
from strategies.stock.iaric.exits import (
    carry_eligible,
    classify_trade,
    should_exit_for_avwap_breakdown,
    should_exit_for_time_stop,
    should_take_partial,
)
from strategies.stock.iaric.models import (
    Bar,
    MarketSnapshot,
    PositionState,
    SymbolIntradayState,
    WatchlistArtifact,
    WatchlistItem,
)
from strategies.stock.iaric.risk import (
    compute_final_risk_unit,
    max_positions_for_regime,
    timing_gate_allows_entry,
    timing_multiplier,
)
from strategies.stock.iaric.signals import (
    compute_location_grade,
    compute_micropressure_proxy,
    compute_required_acceptance,
    cooldown_expired,
    lock_setup,
    reset_setup_state,
    resolve_confidence,
    update_acceptance,
)

if TYPE_CHECKING:
    from research.backtests.stock.analysis.iaric_shadow_tracker import IARICShadowTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MKT_OPEN = time(9, 30)
_MKT_CLOSE = time(16, 0)
_BARS_PER_DAY = 78
_MINUTES_PER_BAR = 5

_SPONSORSHIP_TO_SIGNAL = {
    "STRONG": "STRONG",
    "ACCUMULATE": "ACCUMULATE",
    "NEUTRAL": "NEUTRAL",
    "STALE": "STALE",
    "WEAK": "DISTRIBUTE",
    "BREAKDOWN": "DISTRIBUTE",
}


# ---------------------------------------------------------------------------
# Internal position tracking
# ---------------------------------------------------------------------------

@dataclass
class _V3Position:
    """Backtest-internal V3 position — T1 FSM fields + carry layer."""

    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    risk_per_share: float
    stop: float                     # structural stop from FSM lock_setup
    current_stop: float             # active stop (≥ structural; raised by trail)
    sector: str
    regime_tier: str
    conviction_multiplier: float
    sponsorship_state: str
    entry_trigger: str = "FSM_ENTRY"  # FSM_ENTRY | FSM_IMPROVED | PM_REENTRY

    qty_original: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    mfe_price: float = 0.0
    mfe_r: float = 0.0
    highest_close: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    hold_bars: int = 0
    partial_taken: bool = False
    partial_qty_exited: int = 0
    realized_partial_pnl: float = 0.0
    realized_partial_commission: float = 0.0
    trail_active: bool = False
    carry_days: int = 0
    entry_bar_idx: int = 0

    # Enriched diagnostic fields (from T1)
    confidence: str = "YELLOW"
    location_grade: str = "B"
    acceptance_count: int = 0
    required_acceptance_count: int = 0
    micropressure_signal: str = "NEUTRAL"
    timing_window: str = ""
    timing_mult: float = 1.0
    setup_type: str = ""
    drop_from_hod_pct: float = 0.0
    risk_unit_final: float = 0.0
    conviction_adders: tuple = ()
    time_stop_deadline: datetime | None = None
    setup_tag: str = "UNCLASSIFIED"

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (price - self.entry_price) / self.risk_per_share

    def to_position_state(self) -> PositionState:
        return PositionState(
            entry_price=self.entry_price,
            entry_time=self.entry_time,
            qty_entry=self.qty_original or self.quantity,
            qty_open=self.quantity + self.partial_qty_exited,
            final_stop=self.stop,
            current_stop=self.current_stop,
            initial_risk_per_share=self.risk_per_share,
            max_favorable_price=self.max_favorable,
            max_adverse_price=self.max_adverse if self.max_adverse > 0 else self.entry_price,
            partial_taken=self.partial_taken,
            setup_tag=self.setup_tag,
            time_stop_deadline=self.time_stop_deadline,
        )

    def build_metadata(self, item: WatchlistItem, regime_risk_mult: float) -> dict:
        """Build enriched metadata dict for TradeRecord."""
        rps = max(self.risk_per_share, 0.01)
        oq = self.qty_original or (self.quantity + self.partial_qty_exited)
        return {
            "conviction_bucket": item.conviction_bucket,
            "conviction_multiplier": self.conviction_multiplier,
            "confidence": self.confidence,
            "location_grade": self.location_grade,
            "acceptance_count": self.acceptance_count,
            "required_acceptance_count": self.required_acceptance_count,
            "conviction_adders": list(self.conviction_adders),
            "micropressure_signal": self.micropressure_signal,
            "sponsorship_state": self.sponsorship_state,
            "timing_window": self.timing_window,
            "timing_multiplier": self.timing_mult,
            "setup_type": self.setup_type,
            "drop_from_hod_pct": round(self.drop_from_hod_pct, 4),
            "risk_unit_final": round(self.risk_unit_final, 4),
            "partial_taken": self.partial_taken,
            "partial_qty_fraction": self.partial_qty_exited / oq if oq else 0,
            "regime_risk_multiplier": regime_risk_mult,
            "setup_tag": self.setup_tag,
            "entry_trigger": self.entry_trigger,
            "carry_days": self.carry_days,
            "trail_active": self.trail_active,
            "mfe_r": round(
                (self.max_favorable - self.entry_price) / rps, 4
            ) if self.max_favorable > 0 else 0,
            "mae_r": round(
                (self.entry_price - self.max_adverse) / rps, 4
            ) if self.max_adverse > 0 and self.max_adverse < self.entry_price else 0,
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class IARICIntradayV2Result:
    """Result from the IARIC v3 intraday backtest."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, WatchlistArtifact]
    fsm_log: list[dict] = field(default_factory=list)
    rejection_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _timing_window_label(now: datetime, settings: StrategySettings) -> str:
    et = now.astimezone(ET).time()
    for start, end, _ in settings.timing_sizing:
        if start <= et < end:
            return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    return "OUTSIDE"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IARICIntradayEngineV2:
    """Tier 2 v3 IARIC backtest engine: FSM core + carry layer.

    Per trading day:
    1. Process overnight carry positions (flow reversal, max days, re-eligible)
    2. Load WatchlistArtifact; apply entry gates (conviction, carry-only, etc.)
    3. Initialize SymbolIntradayState per symbol (FSM starts IDLE)
    4. Replay 5m bars: FSM transitions → entry on READY_TO_ENTER → exit cascade
    5. EOD: carry_eligible → CARRY or FLATTEN (gated by v3_carry_enabled)
    """

    def __init__(
        self,
        config: IARICBacktestConfig,
        replay: ResearchReplayEngine,
        settings: StrategySettings | None = None,
        shadow_tracker: IARICShadowTracker | None = None,
    ):
        self._config = config
        self._replay = replay
        self._settings = settings or StrategySettings()

        if config.param_overrides:
            self._settings = replace(self._settings, **config.param_overrides)

        self._slippage = config.slippage
        self._sector_map = {sym: sector for sym, sector, _ in replay._universe}
        self._shadow = shadow_tracker

    # ------------------------------------------------------------------
    # Shadow / logging helpers
    # ------------------------------------------------------------------

    def _log_fsm(
        self, fsm_log: list[dict], symbol: str, trade_date: date,
        timestamp: datetime, from_state: str, to_state: str, reason: str,
        acceptance_count: int = 0,
    ) -> None:
        fsm_log.append({
            "symbol": symbol, "date": trade_date, "timestamp": timestamp,
            "from_state": from_state, "to_state": to_state, "reason": reason,
            "acceptance_count": acceptance_count,
        })

    def _log_rejection(
        self, rejection_log: list[dict], symbol: str, trade_date: date,
        timestamp: datetime, gate: str, setup_type: str = "",
        acceptance_count: int = 0, entry_price: float = 0, stop_price: float = 0,
        confidence: str = "", location_grade: str = "",
    ) -> None:
        rejection_log.append({
            "symbol": symbol, "date": trade_date, "timestamp": timestamp,
            "gate": gate, "setup_type": setup_type,
            "acceptance_count": acceptance_count,
            "entry_price": entry_price, "stop_price": stop_price,
            "confidence": confidence, "location_grade": location_grade,
        })

    def _record_shadow_rejection(
        self, symbol: str, trade_date: date, gate: str,
        entry_price: float, stop_price: float,
        setup_type: str = "", sector: str = "", regime_tier: str = "",
        sponsorship_state: str = "", confidence: str = "",
        location_grade: str = "", acceptance_count: int = 0,
        conviction_multiplier: float = 0.0, risk_per_share: float = 0.0,
    ) -> None:
        if not self._shadow:
            return
        from research.backtests.stock.analysis.iaric_shadow_tracker import IARICShadowSetup
        self._shadow.record_rejection(IARICShadowSetup(
            symbol=symbol,
            trade_date=trade_date,
            rejection_gate=gate,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_per_share=risk_per_share or abs(entry_price - stop_price),
            setup_type=setup_type,
            sector=sector,
            regime_tier=regime_tier,
            sponsorship_state=sponsorship_state,
            confidence=confidence,
            location_grade=location_grade,
            acceptance_count=acceptance_count,
            conviction_multiplier=conviction_multiplier,
        ))

    # ------------------------------------------------------------------
    # Overnight carry processing
    # ------------------------------------------------------------------

    def _process_overnight_carries(
        self,
        carry_positions: dict[str, _V3Position],
        trade_date: date,
        equity: float,
        trades: list[TradeRecord],
        ts: datetime,
    ) -> float:
        """Process carry positions at the start of a new trading day.

        Returns updated equity.
        """
        settings = self._settings
        cfg = self._config
        closed: list[str] = []

        for sym, pos in list(carry_positions.items()):
            ohlc = self._replay.get_daily_ohlc(sym, trade_date)
            if ohlc is None:
                fallback = self._replay.get_daily_close(sym, trade_date)
                if fallback is None:
                    fallback = pos.entry_price
                trade, eq_delta = self._close_position(pos, fallback, ts, "DATA_GAP")
                equity += eq_delta
                trades.append(trade)
                closed.append(sym)
                continue

            O, H, L, C = ohlc
            pos.carry_days += 1
            pos.max_favorable = max(pos.max_favorable, H)
            pos.max_adverse = min(pos.max_adverse, L) if pos.max_adverse > 0 else L
            pos.mfe_price = max(pos.mfe_price, H)
            pos.highest_close = max(pos.highest_close, C)
            pos.mfe_r = max(pos.mfe_r, pos.unrealized_r(H))

            exit_price = None
            exit_reason = ""

            # Flow reversal check
            if cfg.ablation.use_flow_reversal_exit:
                lookback = settings.flow_reversal_lookback
                last_n = self._replay.get_flow_proxy_last_n(sym, trade_date, lookback)
                if last_n is not None and all(v < 0 for v in last_n):
                    exit_price = O
                    exit_reason = "FLOW_REVERSAL"

            # Max carry days
            if exit_price is None and pos.carry_days >= settings.max_carry_days:
                exit_price = C
                exit_reason = "CARRY_MAX_DAYS"

            # Still carry-eligible?
            if exit_price is None:
                cur_r = pos.unrealized_r(C)
                tier_ok = pos.regime_tier == "A" or (
                    pos.regime_tier == "B" and settings.regime_b_carry_mult > 0
                )
                still_eligible = (
                    cfg.ablation.use_carry_logic
                    and tier_ok
                    and pos.sponsorship_state == "STRONG"
                    and cur_r > settings.min_carry_r
                )
                if still_eligible:
                    continue  # Keep carrying
                exit_price = C
                exit_reason = "CARRY_EXIT"

            trade, eq_delta = self._close_position(pos, exit_price, ts, exit_reason)
            equity += eq_delta
            trades.append(trade)
            closed.append(sym)

        for sym in closed:
            carry_positions.pop(sym, None)

        return equity

    # ------------------------------------------------------------------
    # EOD carry scoring (from T2v2)
    # ------------------------------------------------------------------

    def _compute_carry_score(
        self,
        pos: _V3Position,
        bar: Bar,
        vwap: float,
        item: WatchlistItem,
        bars: list[Bar],
        bar_idx: int,
    ) -> float:
        """Compute carry score (0-100) from 6 intraday evidence factors."""
        score = 0.0

        # Factor 1: Current R-multiple (weight 25%)
        cur_r = pos.unrealized_r(bar.close)
        r_score = min(max(cur_r / 2.0, 0.0), 1.0) * 100
        score += r_score * 0.25

        # Factor 2: Price vs session VWAP (weight 20%)
        if vwap > 0:
            atr_est = item.daily_atr_estimate if item.daily_atr_estimate > 0 else bar.close * 0.01
            dist = (bar.close - vwap) / atr_est
            vwap_score = min(max((dist + 0.5) / 1.0, 0.0), 1.0) * 100
        else:
            vwap_score = 50.0
        score += vwap_score * 0.20

        # Factor 3: Close in daily range (weight 15%)
        if bars and bar_idx > 0:
            day_high = max(b.high for b in bars[:bar_idx + 1])
            day_low = min(b.low for b in bars[:bar_idx + 1])
            rng = max(day_high - day_low, 1e-9)
            range_score = ((bar.close - day_low) / rng) * 100
        else:
            range_score = 50.0
        score += range_score * 0.15

        # Factor 4: Micropressure proxy from last 6 bars (weight 15%)
        if bar_idx >= 6:
            bullish_bars = sum(1 for b in bars[bar_idx - 5:bar_idx + 1]
                               if b.close > b.open)
            micro_score = (bullish_bars / 6.0) * 100
        else:
            micro_score = 50.0
        score += micro_score * 0.15

        # Factor 5: Volume trend PM vs AM (weight 10%)
        # Frequency-aware: AM = first half of session, PM = second half
        midpoint = max(bar_idx // 2, 2)
        if bar_idx >= 6:
            am_bars = bars[1:midpoint]          # skip bar 0 (opening bar)
            pm_bars = bars[midpoint:bar_idx + 1]
            am_vol = sum(b.volume for b in am_bars) / max(len(am_bars), 1)
            pm_vol = sum(b.volume for b in pm_bars) / max(len(pm_bars), 1)
            if am_vol > 0:
                vol_ratio = pm_vol / am_vol
                vol_score = min(max(vol_ratio, 0.0), 2.0) / 2.0 * 100
            else:
                vol_score = 50.0
        else:
            vol_score = 50.0
        score += vol_score * 0.10

        # Factor 6: Sponsorship state (weight 15%)
        spon_map = {"STRONG": 100, "ACCUMULATE": 75, "NEUTRAL": 50,
                     "DISTRIBUTE": 25, "WEAK": 0, "BREAKDOWN": 0}
        spon_score = spon_map.get(item.sponsorship_state, 50)
        score += spon_score * 0.15

        return score

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def _close_position(
        self,
        pos: _V3Position,
        exit_price: float,
        ts: datetime,
        reason: str,
        item: WatchlistItem | None = None,
        regime_risk_mult: float = 1.0,
    ) -> tuple[TradeRecord, float]:
        """Close a position and return (TradeRecord, equity_delta).

        equity_delta is only the remaining runner's PnL minus its commission,
        excluding already-counted partial PnL.
        """
        slip = exit_price * self._slippage.slip_bps_normal / 10_000
        fill = round(exit_price - slip, 2)
        commission = self._slippage.commission_per_share * pos.quantity
        runner_pnl = (fill - pos.entry_price) * pos.quantity

        total_pnl = runner_pnl + pos.realized_partial_pnl
        total_commission = pos.commission_entry + commission + pos.realized_partial_commission
        total_qty = pos.quantity + pos.partial_qty_exited
        orig_risk = pos.risk_per_share * (pos.qty_original or total_qty)
        r_mult = total_pnl / orig_risk if orig_risk > 0 else 0.0

        meta = pos.build_metadata(item, regime_risk_mult) if item else {
            "conviction_multiplier": pos.conviction_multiplier,
            "sponsorship": pos.sponsorship_state,
            "entry_trigger": pos.entry_trigger,
            "carry_days": pos.carry_days,
            "mfe_r": round(pos.mfe_r, 3),
        }

        trade = TradeRecord(
            strategy="IARIC",
            symbol=pos.symbol,
            direction=BTDirection.LONG,
            entry_time=pos.entry_time,
            exit_time=ts,
            entry_price=pos.entry_price,
            exit_price=fill,
            quantity=total_qty,
            pnl=total_pnl,
            r_multiple=r_mult,
            risk_per_share=pos.risk_per_share,
            commission=total_commission,
            slippage=pos.slippage_entry + slip * pos.quantity,
            entry_type=pos.entry_trigger,
            exit_reason=reason,
            sector=pos.sector,
            regime_tier=pos.regime_tier,
            hold_bars=pos.hold_bars if pos.carry_days == 0 else pos.carry_days,
            max_favorable=pos.max_favorable,
            max_adverse=pos.max_adverse,
            metadata=meta,
        )
        # equity_delta: only the remaining runner piece (partials already in equity)
        equity_delta = runner_pnl - commission
        return trade, equity_delta

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------

    def run(self) -> IARICIntradayV2Result:
        """Execute the full 5m-bar intraday backtest with optional carry."""
        cfg = self._config
        settings = self._settings
        start = date.fromisoformat(cfg.start_date)
        end = date.fromisoformat(cfg.end_date)

        trading_dates = self._replay.tradable_dates(start, end)
        if not trading_dates:
            return IARICIntradayV2Result(
                trades=[], equity_curve=np.array([cfg.initial_equity]),
                timestamps=np.array([]), daily_selections={},
            )

        equity = cfg.initial_equity
        carry_positions: dict[str, _V3Position] = {}
        trades: list[TradeRecord] = []
        equity_history: list[float] = [equity]
        ts_history: list[datetime] = []
        daily_selections: dict[date, WatchlistArtifact] = {}
        fsm_log: list[dict] = []
        rejection_log: list[dict] = []

        # Pre-extract ablation flags
        abl = cfg.ablation
        use_regime_gate = abl.use_regime_gate
        use_time_stop = abl.use_time_stop
        use_avwap_breakdown = abl.use_avwap_breakdown_exit
        use_partial_take = abl.use_partial_take
        use_carry = abl.use_carry_logic
        use_sector_limit = abl.use_sector_limit
        use_sponsorship_filter = abl.use_sponsorship_filter

        # Pre-extract slippage params
        slip_bps = self._slippage.slip_bps_normal / 10_000
        comm_per_share = self._slippage.commission_per_share

        # Pre-extract settings params
        panic_drop = settings.panic_flush_drop_pct
        panic_minutes = settings.panic_flush_minutes
        drift_drop = settings.drift_exhaustion_drop_pct
        drift_minutes = settings.drift_exhaustion_minutes
        stale_minutes = settings.setup_stale_minutes
        base_risk_frac = settings.base_risk_fraction
        ts_minutes = settings.time_stop_minutes
        max_per_sector = cfg.max_per_sector

        # V3 flags
        v3_carry = settings.v3_carry_enabled
        v3_carry_score_fb = settings.v3_carry_score_fallback
        v3_carry_score_thresh = settings.v3_carry_score_threshold
        v3_trail = settings.v3_adaptive_trail
        v3_trail_activation_r = settings.v3_trail_activation_r
        v3_trail_atr_mult = settings.v3_trail_atr_mult
        v3_improve = settings.v3_entry_improvement
        v3_improve_bars = settings.v3_improvement_window_bars
        v3_improve_discount = settings.v3_improvement_discount_pct
        v3_pm_reentry = settings.v3_pm_reentry
        v3_pm_reentry_bar = settings.v3_pm_reentry_after_bar
        v3_stale_tighten = settings.v3_staleness_tighten
        v3_stale_hours = settings.v3_staleness_hours
        v3_stale_tighten_atr = settings.v3_staleness_tighten_atr

        # Open-entry mode: enter all tradable at bar 1 (near open), bypass FSM
        open_entry = settings.t2_open_entry
        open_entry_stop_atr = settings.t2_open_entry_stop_atr
        open_entry_size_mult = settings.t2_open_entry_size_mult
        stop_risk_cap_pct = settings.stop_risk_cap_pct
        use_close_stop = settings.use_close_stop
        daily_alpha_mode = settings.t2_daily_alpha_mode
        bar_minutes = settings.t2_bar_minutes

        has_shadow = self._shadow is not None
        shadow = self._shadow

        n_dates = len(trading_dates)
        for day_idx, trade_date in enumerate(trading_dates):
            if day_idx > 0 and day_idx % 50 == 0:
                logger.info("IARIC v3 progress: %d/%d dates (%.0f%%)",
                            day_idx, n_dates, 100 * day_idx / n_dates)

            ts = datetime(trade_date.year, trade_date.month, trade_date.day,
                          tzinfo=timezone.utc)
            ts_history.append(ts)

            # 1. Process overnight carry positions
            if carry_positions:
                equity = self._process_overnight_carries(
                    carry_positions, trade_date, equity, trades, ts,
                )

            # 2. Load daily selection
            artifact = self._replay.iaric_selection_for_date(trade_date, settings)
            daily_selections[trade_date] = artifact

            if use_regime_gate and artifact.regime.tier == "C":
                equity_history.append(equity)
                continue

            max_pos = max_positions_for_regime(artifact.regime.tier, settings)
            mwis = artifact.market_wide_institutional_selling
            regime_risk_mult = artifact.regime.risk_multiplier

            # 3. Build tradable map with entry gates
            #    open_entry mode: artifact.tradable (matches T1 daily engine)
            #    FSM mode: artifact.items (matches T1 FSM engine)
            tradable_map: dict[str, WatchlistItem] = {}
            items = artifact.tradable if open_entry else artifact.items
            if settings.entry_order_by_conviction:
                items = sorted(items,
                               key=lambda it: it.conviction_multiplier, reverse=True)
            for item in items:
                sym = item.symbol
                if sym in carry_positions:
                    continue
                if item.conviction_multiplier <= settings.min_conviction_multiplier:
                    continue
                if settings.carry_only_entry:
                    can_carry = (
                        item.sponsorship_state == "STRONG"
                        and (artifact.regime.tier == "A"
                             or (artifact.regime.tier == "B"
                                 and settings.regime_b_carry_mult > 0))
                    )
                    if not can_carry:
                        continue
                if settings.strong_only_entry and item.sponsorship_state != "STRONG":
                    continue
                tradable_map[sym] = item

            # 4. Per-day state
            positions: dict[str, _V3Position] = {}
            fsm_states: dict[str, SymbolIntradayState] = {}
            market_snapshots: dict[str, MarketSnapshot] = {}
            sector_counts: dict[str, int] = {}
            stopped_today: set[str] = set()
            pending_improvements: dict[str, dict] = {}  # sym → {price, stop, bar_idx, ...}

            # Carry positions count toward sector limits
            for cpos in carry_positions.values():
                sector_counts[cpos.sector] = sector_counts.get(cpos.sector, 0) + 1

            hod_bar_index: dict[str, int] = {}
            bar_30m_accum: dict[str, list[Bar]] = {}

            # Initialize FSM for each tradable symbol
            for sym, item in tradable_map.items():
                flow_signal = "UNAVAILABLE"
                flow_data = self._replay.get_flow_proxy_last_n(
                    sym, trade_date, settings.flow_reversal_lookback,
                )
                if flow_data is not None:
                    if all(v < 0 for v in flow_data):
                        flow_signal = "WEAK"
                    elif all(v > 0 for v in flow_data):
                        flow_signal = "STRONG"
                    else:
                        flow_signal = "NEUTRAL"

                fsm_states[sym] = SymbolIntradayState(
                    symbol=sym,
                    fsm_state="IDLE",
                    setup_time=None,
                    setup_low=0.0,
                    reclaim_level=0.0,
                    stop_level=0.0,
                    acceptance_count=0,
                    required_acceptance_count=0,
                    confidence="YELLOW",
                    micropressure_signal="NEUTRAL",
                    flowproxy_signal=flow_signal,
                    micropressure_mode="LIVE",
                    invalidated_at=None,
                    tier="WARM",
                    sponsorship_signal=_SPONSORSHIP_TO_SIGNAL.get(
                        item.sponsorship_state, "NEUTRAL",
                    ),
                )
                market_snapshots[sym] = MarketSnapshot(
                    symbol=sym,
                    last_price=item.avwap_ref,
                    session_high=0.0,
                    session_low=float("inf"),
                    session_vwap=0.0,
                    avwap_live=item.avwap_ref,
                )
                hod_bar_index[sym] = 0
                bar_30m_accum[sym] = []

            # 5. Replay bars per symbol (5m or 30m depending on bar_minutes)
            _td30 = timedelta(minutes=30)
            for sym in list(tradable_map.keys()):
                if bar_minutes == 30:
                    alcb_bars = self._replay.get_30m_bar_objects_for_date(sym, trade_date)
                    if not alcb_bars:
                        continue
                    bars = [
                        Bar(symbol=b.symbol, start_time=b.start_time,
                            end_time=b.start_time + _td30,
                            open=b.open, high=b.high, low=b.low,
                            close=b.close, volume=b.volume)
                        for b in alcb_bars
                    ]
                else:
                    bars = self._replay.get_5m_bar_objects_for_date(sym, trade_date)
                if not bars:
                    continue
                bars = [b for b in bars
                        if _MKT_OPEN <= b.start_time.astimezone(ET).time() < _MKT_CLOSE]
                if not bars:
                    continue

                item = tradable_map[sym]
                sym_state = fsm_states[sym]
                market = market_snapshots[sym]
                sec = item.sector

                atr_5m_pct = item.intraday_atr_seed if item.intraday_atr_seed > 0 else 0.01
                if settings.t2_avwap_band_mult != 1.0:
                    band_pct = settings.avwap_band_pct * settings.t2_avwap_band_mult
                    avwap_band_lo = item.avwap_ref * (1.0 - band_pct)
                    avwap_band_hi = item.avwap_ref * (1.0 + band_pct)
                else:
                    avwap_band_lo = item.avwap_band_lower
                    avwap_band_hi = item.avwap_band_upper
                expected_vol = item.expected_5m_volume if item.expected_5m_volume > 0 else 0.0
                median_vol = item.average_30m_volume if item.average_30m_volume > 0 else 0.0

                # Session ATR in dollars (for v3 adaptive trail)
                ref_price = bars[0].open if bars else item.avwap_ref
                session_atr = (
                    item.intraday_atr_seed * ref_price if item.intraday_atr_seed > 0
                    else (item.daily_atr_estimate * 0.3 if item.daily_atr_estimate > 0
                          else ref_price * 0.01)
                )

                sym_bar_idx = 0
                sym_hod_idx = 0
                cum_vol = 0.0
                cum_pv = 0.0
                accum_30m: list[Bar] = bar_30m_accum[sym]
                vwap = 0.0

                for bar in bars:
                    now = bar.end_time
                    sym_bar_idx += 1

                    # Update market snapshot
                    if bar.high > (market.session_high or 0):
                        market.session_high = bar.high
                        sym_hod_idx = sym_bar_idx
                    market.last_price = bar.close
                    if market.session_low == float("inf"):
                        market.session_low = bar.low
                    else:
                        market.session_low = min(market.session_low, bar.low)
                    cum_vol += bar.volume
                    cum_pv += bar.close * bar.volume
                    if cum_vol > 0:
                        market.session_vwap = cum_pv / cum_vol
                        vwap = market.session_vwap

                    market.bars_5m.append(bar)
                    market.last_5m_bar = bar

                    # Aggregate 30m bars
                    if bar_minutes == 30:
                        bar_30m = bar
                        market.last_30m_bar = bar_30m
                    else:
                        accum_30m.append(bar)
                        bar_30m = None
                        if len(accum_30m) >= 6:
                            bar_30m = Bar(
                                symbol=sym,
                                start_time=accum_30m[0].start_time,
                                end_time=accum_30m[-1].end_time,
                                open=accum_30m[0].open,
                                high=max(b.high for b in accum_30m),
                                low=min(b.low for b in accum_30m),
                                close=accum_30m[-1].close,
                                volume=sum(b.volume for b in accum_30m),
                            )
                            market.last_30m_bar = bar_30m
                            accum_30m.clear()

                    # Shadow tracker bar update
                    if has_shadow:
                        shadow.update_bar(sym, bar.high, bar.low, bar.close)

                    # ═══════════════════════════════════════════════
                    # MANAGE EXISTING POSITION
                    # ═══════════════════════════════════════════════
                    if sym in positions:
                        pos = positions[sym]
                        pos.hold_bars += 1

                        # Track MFE/MAE
                        pos.max_favorable = max(pos.max_favorable, bar.high)
                        pos.max_adverse = min(pos.max_adverse, bar.low) if pos.max_adverse > 0 else bar.low
                        pos.mfe_price = max(pos.mfe_price, bar.high)
                        pos.highest_close = max(pos.highest_close, bar.close)
                        cur_r = pos.unrealized_r(bar.high)
                        pos.mfe_r = max(pos.mfe_r, cur_r)

                        exit_price = None
                        exit_reason = ""

                        # [v3] Adaptive trail — update current_stop (ratchet UP only)
                        if v3_trail and pos.mfe_r >= v3_trail_activation_r:
                            trail_stop = pos.highest_close - v3_trail_atr_mult * session_atr
                            if trail_stop > pos.current_stop:
                                pos.current_stop = trail_stop
                                pos.trail_active = True

                        # 1. STOP_HIT (uses current_stop which may be trailed)
                        if bar.low <= pos.current_stop:
                            exit_price = pos.current_stop
                            exit_reason = "STOP_HIT"

                        # 2. TIME_STOP (skipped in daily_alpha_mode)
                        if exit_price is None and use_time_stop and not daily_alpha_mode:
                            pos_state = pos.to_position_state()
                            if should_exit_for_time_stop(pos_state, now, bar.close):
                                exit_price = bar.close
                                exit_reason = "TIME_STOP"

                        # 3. AVWAP_BREAKDOWN (30m) (skipped in daily_alpha_mode)
                        if (exit_price is None and use_avwap_breakdown
                                and not daily_alpha_mode
                                and bar_30m is not None and market.avwap_live):
                            if should_exit_for_avwap_breakdown(
                                bar_30m, market.avwap_live,
                                item.average_30m_volume, settings,
                            ):
                                exit_price = bar.close
                                exit_reason = "AVWAP_BREAKDOWN"

                        # 4. PARTIAL_TAKE (skipped in daily_alpha_mode)
                        if exit_price is None and not pos.partial_taken and use_partial_take and not daily_alpha_mode:
                            pos_state = pos.to_position_state()
                            take, fraction = should_take_partial(
                                pos_state, bar.close, settings,
                                regime_multiplier=regime_risk_mult,
                            )
                            if take:
                                partial_qty = max(1, int(pos.quantity * fraction))
                                pnl = (bar.close - pos.entry_price) * partial_qty
                                comm = comm_per_share * partial_qty
                                equity += pnl - comm
                                pos.quantity -= partial_qty
                                pos.partial_taken = True
                                pos.partial_qty_exited = partial_qty
                                pos.realized_partial_pnl += pnl
                                pos.realized_partial_commission += comm
                                if pos.quantity <= 0:
                                    exit_price = bar.close
                                    exit_reason = "PARTIAL_FULL"

                        # [v3] Staleness tighten (NOT exit, just tighten stop)
                        if (exit_price is None and v3_stale_tighten):
                            bars_held = sym_bar_idx - pos.entry_bar_idx
                            hours_held = bars_held * bar_minutes / 60.0
                            if hours_held >= v3_stale_hours and pos.mfe_r < 0.1:
                                tighten = pos.entry_price - v3_stale_tighten_atr * session_atr
                                pos.current_stop = max(pos.current_stop, tighten)

                        # 5. EOD decision at 15:55 ET
                        if exit_price is None:
                            bar_et = now.astimezone(ET) if now.tzinfo else now
                            if bar_et.time() >= time(15, 55):
                                pos_state = pos.to_position_state()
                                pos.setup_tag = classify_trade(market, pos_state)

                                # Close-stop: exit underwater at close (T1 behavior)
                                if use_close_stop and bar.close <= pos.entry_price:
                                    exit_price = bar.close
                                    exit_reason = "CLOSE_STOP"
                                elif v3_carry and use_carry:
                                    # T1-style inline carry check (simpler than
                                    # carry_eligible which requires setup_tag/AVWAP/
                                    # top-quartile gates designed for live trading).
                                    cur_r = pos.unrealized_r(bar.close)
                                    tier_ok = (
                                        pos.regime_tier == "A"
                                        or (pos.regime_tier == "B"
                                            and settings.regime_b_carry_mult > 0)
                                    )
                                    carry_ok = (
                                        tier_ok
                                        and pos.sponsorship_state == "STRONG"
                                        and cur_r > settings.min_carry_r
                                        and not item.earnings_risk_flag
                                        and not item.blacklist_flag
                                    )
                                    if carry_ok:
                                        pos.carry_days = 1
                                        carry_positions[sym] = pos
                                        positions.pop(sym, None)
                                        sector_counts[sec] = max(0, sector_counts.get(sec, 1) - 1)
                                    elif v3_carry_score_fb:
                                        score = self._compute_carry_score(
                                            pos, bar, vwap, item, bars, sym_bar_idx - 1,
                                        )
                                        if score >= v3_carry_score_thresh:
                                            pos.carry_days = 1
                                            carry_positions[sym] = pos
                                            positions.pop(sym, None)
                                            sector_counts[sec] = max(0, sector_counts.get(sec, 1) - 1)
                                        else:
                                            exit_price = bar.close
                                            exit_reason = "EOD_FLATTEN"
                                    else:
                                        exit_price = bar.close
                                        exit_reason = "EOD_FLATTEN"
                                else:
                                    exit_price = bar.close
                                    exit_reason = "EOD_FLATTEN"

                        # Process exit
                        if exit_price is not None and pos.quantity > 0:
                            slip = exit_price * slip_bps
                            fill = round(exit_price - slip, 2)
                            commission = comm_per_share * pos.quantity
                            runner_pnl = (fill - pos.entry_price) * pos.quantity
                            equity += runner_pnl - commission

                            total_pnl = runner_pnl + pos.realized_partial_pnl
                            total_commission = pos.commission_entry + commission + pos.realized_partial_commission
                            total_qty = pos.quantity + pos.partial_qty_exited
                            orig_risk = pos.risk_per_share * (pos.qty_original or total_qty)
                            r_mult = total_pnl / orig_risk if orig_risk > 0 else 0.0

                            trades.append(TradeRecord(
                                strategy="IARIC",
                                symbol=sym,
                                direction=BTDirection.LONG,
                                entry_time=pos.entry_time,
                                exit_time=now,
                                entry_price=pos.entry_price,
                                exit_price=fill,
                                quantity=total_qty,
                                pnl=total_pnl,
                                r_multiple=r_mult,
                                risk_per_share=pos.risk_per_share,
                                commission=total_commission,
                                slippage=pos.slippage_entry + slip * pos.quantity,
                                entry_type=pos.entry_trigger,
                                exit_reason=exit_reason,
                                sector=pos.sector,
                                regime_tier=pos.regime_tier,
                                hold_bars=pos.hold_bars,
                                max_favorable=pos.max_favorable,
                                max_adverse=pos.max_adverse,
                                metadata=pos.build_metadata(item, regime_risk_mult),
                            ))
                            positions.pop(sym, None)
                            sector_counts[sec] = max(0, sector_counts.get(sec, 1) - 1)
                            if exit_reason == "STOP_HIT":
                                stopped_today.add(sym)

                        continue  # Don't attempt entry while in position

                    # ═══════════════════════════════════════════════
                    # CHECK PENDING IMPROVEMENT WINDOW
                    # ═══════════════════════════════════════════════
                    if v3_improve and sym in pending_improvements:
                        pend = pending_improvements[sym]
                        target_price = pend["target_price"]
                        discount_price = target_price * (1.0 - v3_improve_discount)
                        expiry_bar = pend["expiry_bar"]

                        enter_now = False
                        improved_price = target_price

                        if bar.low <= discount_price:
                            # Got an improved price
                            improved_price = discount_price
                            enter_now = True
                        elif sym_bar_idx >= expiry_bar:
                            # Window expired — enter at original price
                            enter_now = True

                        if enter_now:
                            del pending_improvements[sym]
                            # Reconstruct entry from pending state
                            entry_price = improved_price
                            stop_level = pend["stop_level"]
                            risk_per_share = abs(entry_price - stop_level)
                            if risk_per_share > 0:
                                risk_unit = pend["risk_unit"]
                                risk_dollars = equity * base_risk_frac * risk_unit
                                qty = int(floor(risk_dollars / risk_per_share))
                                if qty >= 1:
                                    slip_e = entry_price * slip_bps
                                    fill_price = round(entry_price + slip_e, 2)
                                    commission = comm_per_share * qty
                                    trigger = "FSM_IMPROVED" if improved_price < target_price else "FSM_ENTRY"

                                    positions[sym] = _V3Position(
                                        symbol=sym,
                                        entry_price=fill_price,
                                        entry_time=now,
                                        quantity=qty,
                                        risk_per_share=risk_per_share,
                                        stop=stop_level,
                                        current_stop=stop_level,
                                        sector=sec,
                                        regime_tier=artifact.regime.tier,
                                        conviction_multiplier=item.conviction_multiplier,
                                        sponsorship_state=item.sponsorship_state,
                                        entry_trigger=trigger,
                                        qty_original=qty,
                                        commission_entry=commission,
                                        slippage_entry=slip_e * qty,
                                        max_favorable=bar.high,
                                        max_adverse=bar.low,
                                        mfe_price=bar.high,
                                        highest_close=bar.close,
                                        entry_bar_idx=sym_bar_idx,
                                        **pend["enrichment"],
                                    )
                                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                                    self._log_fsm(
                                        fsm_log, sym, trade_date, now,
                                        "READY_TO_ENTER", "IN_POSITION",
                                        f"entry|trigger={trigger}|qty={qty}",
                                    )
                            continue  # Either entered or failed — move on

                    # ═══════════════════════════════════════════════
                    # OPEN ENTRY (bar 1 = near open, bypasses FSM)
                    # ═══════════════════════════════════════════════
                    if open_entry and sym_bar_idx == 1 and sym not in positions:
                        # Check position/sector limits
                        active_count = len(positions) + len(carry_positions)
                        if active_count < max_pos and sector_counts.get(sec, 0) < max_per_sector:
                            oe_price = bar.close
                            daily_atr = item.daily_atr_estimate
                            if daily_atr <= 0:
                                daily_atr = abs(bar.high - bar.low) or oe_price * 0.01
                            stop_dist = min(daily_atr * open_entry_stop_atr,
                                            oe_price * stop_risk_cap_pct)
                            oe_stop = oe_price - stop_dist
                            oe_rps = abs(oe_price - oe_stop)
                            if oe_rps > 0:
                                risk_mult = item.conviction_multiplier * regime_risk_mult
                                oe_risk_dollars = equity * base_risk_frac * risk_mult * open_entry_size_mult
                                oe_qty = int(floor(oe_risk_dollars / oe_rps))
                                if oe_qty >= 1:
                                    oe_slip = oe_price * slip_bps
                                    oe_fill = round(oe_price + oe_slip, 2)
                                    oe_comm = comm_per_share * oe_qty
                                    # No time_stop for open entries (T1 daily has none)
                                    positions[sym] = _V3Position(
                                        symbol=sym,
                                        entry_price=oe_fill,
                                        entry_time=now,
                                        quantity=oe_qty,
                                        risk_per_share=oe_rps,
                                        stop=oe_stop,
                                        current_stop=oe_stop,
                                        sector=sec,
                                        regime_tier=artifact.regime.tier,
                                        conviction_multiplier=item.conviction_multiplier,
                                        sponsorship_state=item.sponsorship_state,
                                        entry_trigger="OPEN_ENTRY",
                                        qty_original=oe_qty,
                                        commission_entry=oe_comm,
                                        slippage_entry=oe_slip * oe_qty,
                                        max_favorable=bar.high,
                                        max_adverse=bar.low,
                                        mfe_price=bar.high,
                                        highest_close=bar.close,
                                        entry_bar_idx=sym_bar_idx,
                                        time_stop_deadline=None,
                                    )
                                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                                    self._log_fsm(
                                        fsm_log, sym, trade_date, now,
                                        "IDLE", "IN_POSITION",
                                        f"open_entry|risk_mult={risk_mult:.3f}|qty={oe_qty}",
                                    )
                        if sym in positions:
                            continue  # entered via open entry, skip FSM

                    # ═══════════════════════════════════════════════
                    # FSM STATE MACHINE
                    # ═══════════════════════════════════════════════

                    if has_shadow:
                        shadow.record_funnel("evaluated")

                    # Entry gates (only block READY_TO_ENTER → entry)
                    active_count = len(positions) + len(carry_positions)
                    entry_blocked = False
                    entry_block_reason = ""
                    if active_count >= max_pos:
                        entry_blocked = True
                        entry_block_reason = "position_cap"
                    elif not timing_gate_allows_entry(now, settings):
                        entry_blocked = True
                        entry_block_reason = "timing_blocked"
                    elif use_sector_limit and sector_counts.get(sec, 0) >= max_per_sector:
                        entry_blocked = True
                        entry_block_reason = "sector_limit"

                    # Approximate minutes_since_hod
                    approx_minutes_since_hod = (sym_bar_idx - sym_hod_idx) * 5
                    drop_from_hod = 0.0
                    session_high = market.session_high
                    if session_high and session_high > 0:
                        drop_from_hod = (session_high - bar.close) / session_high

                    # Global invalidation checks
                    if sym_state.fsm_state in ("SETUP_DETECTED", "ACCEPTING"):
                        if sym_state.stop_level is not None and bar.low <= sym_state.stop_level:
                            prev = sym_state.fsm_state
                            sym_state.fsm_state = "INVALIDATED"
                            sym_state.invalidated_at = now
                            self._log_fsm(
                                fsm_log, sym, trade_date, now,
                                prev, "INVALIDATED", "stop_breach",
                                acceptance_count=sym_state.acceptance_count,
                            )
                            self._log_rejection(
                                rejection_log, sym, trade_date, now, "price_invalidation",
                                setup_type=sym_state.setup_type or "",
                                acceptance_count=sym_state.acceptance_count,
                                entry_price=bar.close,
                                stop_price=sym_state.stop_level or 0,
                            )
                            self._record_shadow_rejection(
                                sym, trade_date, "price_invalidation",
                                bar.close, sym_state.stop_level or 0,
                                setup_type=sym_state.setup_type or "",
                                sector=sec, regime_tier=artifact.regime.tier,
                                acceptance_count=sym_state.acceptance_count,
                            )
                        elif sym_state.setup_time:
                            elapsed = (now - sym_state.setup_time).total_seconds() / 60
                            if elapsed > stale_minutes:
                                prev = sym_state.fsm_state
                                sym_state.fsm_state = "INVALIDATED"
                                sym_state.invalidated_at = now
                                self._log_fsm(
                                    fsm_log, sym, trade_date, now,
                                    prev, "INVALIDATED", "setup_stale",
                                    acceptance_count=sym_state.acceptance_count,
                                )
                                self._log_rejection(
                                    rejection_log, sym, trade_date, now, "stale_rejection",
                                    setup_type=sym_state.setup_type or "",
                                    acceptance_count=sym_state.acceptance_count,
                                    entry_price=bar.close,
                                    stop_price=sym_state.stop_level or 0,
                                )
                                self._record_shadow_rejection(
                                    sym, trade_date, "stale_rejection",
                                    bar.close, sym_state.stop_level or 0,
                                    setup_type=sym_state.setup_type or "",
                                    sector=sec, regime_tier=artifact.regime.tier,
                                    acceptance_count=sym_state.acceptance_count,
                                )

                    # State-specific logic
                    if sym_state.fsm_state == "IDLE":
                        if has_shadow:
                            shadow.record_funnel("idle_check")

                        in_avwap_band = (
                            avwap_band_lo <= bar.low <= avwap_band_hi
                            or avwap_band_lo <= bar.close <= avwap_band_hi
                        )
                        if not in_avwap_band:
                            continue

                        setup_type = None
                        if (drop_from_hod >= panic_drop
                                and approx_minutes_since_hod <= panic_minutes):
                            setup_type = "PANIC_FLUSH"
                        elif (drop_from_hod >= drift_drop
                              and approx_minutes_since_hod >= drift_minutes):
                            setup_type = "DRIFT_EXHAUSTION"

                        if setup_type:
                            if has_shadow:
                                shadow.record_funnel("setup_detected")

                            sym_state.setup_type = setup_type
                            lock_setup(sym_state, bar, atr_5m_pct, reason=setup_type)
                            sym_state.location_grade = compute_location_grade(item, market)

                            self._log_fsm(
                                fsm_log, sym, trade_date, now,
                                "IDLE", "SETUP_DETECTED", setup_type,
                            )

                    elif sym_state.fsm_state == "SETUP_DETECTED":
                        reclaim_touched = False
                        if sym_state.reclaim_level is not None:
                            if bar.high >= sym_state.reclaim_level:
                                reclaim_touched = True
                            elif market.last_price is not None and market.last_price >= sym_state.reclaim_level:
                                reclaim_touched = True

                        if reclaim_touched:
                            sym_state.fsm_state = "ACCEPTING"
                            required, adders = compute_required_acceptance(
                                item=item, sym=sym_state, now=now,
                                settings=settings,
                                market_wide_institutional_selling=mwis,
                            )
                            sym_state.required_acceptance_count = required
                            self._log_fsm(
                                fsm_log, sym, trade_date, now,
                                "SETUP_DETECTED", "ACCEPTING",
                                f"reclaim_hit|adders={'|'.join(adders)}",
                            )
                            if has_shadow:
                                shadow.record_funnel("accepting")

                    elif sym_state.fsm_state == "ACCEPTING":
                        update_acceptance(sym_state, bar)

                        vol_for_proxy = expected_vol if expected_vol > 0 else bar.volume
                        med_for_proxy = median_vol if median_vol > 0 else bar.volume
                        if sym_state.reclaim_level:
                            mp = compute_micropressure_proxy(
                                bar, vol_for_proxy, med_for_proxy,
                                sym_state.reclaim_level,
                            )
                            sym_state.micropressure_signal = mp

                        sym_state.confidence = resolve_confidence(sym_state)

                        if (sym_state.acceptance_count >= sym_state.required_acceptance_count
                                and sym_state.confidence != "RED"):
                            sym_state.fsm_state = "READY_TO_ENTER"
                            self._log_fsm(
                                fsm_log, sym, trade_date, now,
                                "ACCEPTING", "READY_TO_ENTER",
                                f"accepted|conf={sym_state.confidence}",
                                acceptance_count=sym_state.acceptance_count,
                            )
                            if has_shadow:
                                shadow.record_funnel("ready_to_enter")

                    elif sym_state.fsm_state == "INVALIDATED":
                        if cooldown_expired(sym_state, now, settings):
                            reset_setup_state(sym_state)
                            self._log_fsm(
                                fsm_log, sym, trade_date, now,
                                "INVALIDATED", "IDLE", "cooldown_expired",
                            )

                    # ═══════════════════════════════════════════════
                    # ENTRY ON READY_TO_ENTER
                    # ═══════════════════════════════════════════════
                    if sym_state.fsm_state == "READY_TO_ENTER":
                        # Determine if this is a PM re-entry for a stopped symbol
                        # When v3_pm_reentry is OFF, stopped symbols can still re-enter
                        # via normal FSM (matches T1 behavior); tag stays FSM_ENTRY.
                        is_pm_reentry = (
                            v3_pm_reentry
                            and sym in stopped_today
                            and sym_bar_idx >= v3_pm_reentry_bar
                        )

                        if entry_blocked:
                            self._log_rejection(
                                rejection_log, sym, trade_date, now, entry_block_reason,
                                setup_type=sym_state.setup_type or "",
                                acceptance_count=sym_state.acceptance_count,
                                entry_price=bar.close,
                                stop_price=sym_state.stop_level or 0,
                            )
                            self._record_shadow_rejection(
                                sym, trade_date, entry_block_reason,
                                bar.close, sym_state.stop_level or 0,
                                setup_type=sym_state.setup_type or "",
                                sector=sec, regime_tier=artifact.regime.tier,
                                sponsorship_state=item.sponsorship_state,
                                confidence=sym_state.confidence or "",
                                location_grade=sym_state.location_grade or "",
                                acceptance_count=sym_state.acceptance_count,
                                conviction_multiplier=item.conviction_multiplier,
                            )
                            continue

                        # Sponsorship filter
                        if (use_sponsorship_filter
                                and item.sponsorship_state in ("WEAK", "BREAKDOWN")):
                            self._log_rejection(
                                rejection_log, sym, trade_date, now, "sponsorship_filter",
                                setup_type=sym_state.setup_type or "",
                                entry_price=bar.close,
                                stop_price=sym_state.stop_level or 0,
                            )
                            self._record_shadow_rejection(
                                sym, trade_date, "sponsorship_filter",
                                bar.close, sym_state.stop_level or 0,
                                setup_type=sym_state.setup_type or "",
                                sector=sec, regime_tier=artifact.regime.tier,
                                sponsorship_state=item.sponsorship_state,
                            )
                            reset_setup_state(sym_state)
                            continue

                        entry_price = bar.close
                        stop_level = sym_state.stop_level
                        if not stop_level or stop_level <= 0:
                            stop_level = entry_price * 0.98

                        risk_per_share = abs(entry_price - stop_level)
                        if risk_per_share <= 0:
                            reset_setup_state(sym_state)
                            continue

                        # 6-factor sizing
                        risk_unit = compute_final_risk_unit(item, sym_state, now, settings)
                        if risk_unit <= 0:
                            reset_setup_state(sym_state)
                            continue

                        # Compute enrichment fields for pending/immediate entry
                        ts_deadline = now + timedelta(minutes=ts_minutes)
                        tw_label = _timing_window_label(now, settings)
                        tm = timing_multiplier(now, settings)
                        _, adders = compute_required_acceptance(
                            item=item, sym=sym_state, now=now,
                            settings=settings,
                            market_wide_institutional_selling=mwis,
                        )
                        enrichment = dict(
                            confidence=sym_state.confidence or "YELLOW",
                            location_grade=sym_state.location_grade or "B",
                            acceptance_count=sym_state.acceptance_count,
                            required_acceptance_count=sym_state.required_acceptance_count,
                            micropressure_signal=sym_state.micropressure_signal,
                            timing_window=tw_label,
                            timing_mult=tm,
                            setup_type=sym_state.setup_type or "",
                            drop_from_hod_pct=drop_from_hod,
                            risk_unit_final=risk_unit,
                            conviction_adders=tuple(adders),
                            time_stop_deadline=ts_deadline,
                        )

                        # [v3] Entry improvement window (not for PM re-entries)
                        if v3_improve and not is_pm_reentry:
                            pending_improvements[sym] = {
                                "target_price": entry_price,
                                "stop_level": stop_level,
                                "risk_unit": risk_unit,
                                "expiry_bar": sym_bar_idx + v3_improve_bars,
                                "enrichment": enrichment,
                            }
                            reset_setup_state(sym_state)
                            continue

                        # Immediate entry
                        risk_dollars = equity * base_risk_frac * risk_unit
                        qty = int(floor(risk_dollars / risk_per_share))
                        if qty < 1:
                            reset_setup_state(sym_state)
                            continue

                        if has_shadow:
                            shadow.record_funnel("entered")

                        trigger = "PM_REENTRY" if is_pm_reentry else "FSM_ENTRY"
                        slip_e = entry_price * slip_bps
                        fill_price = round(entry_price + slip_e, 2)
                        commission = comm_per_share * qty

                        positions[sym] = _V3Position(
                            symbol=sym,
                            entry_price=fill_price,
                            entry_time=now,
                            quantity=qty,
                            risk_per_share=risk_per_share,
                            stop=stop_level,
                            current_stop=stop_level,
                            sector=sec,
                            regime_tier=artifact.regime.tier,
                            conviction_multiplier=item.conviction_multiplier,
                            sponsorship_state=item.sponsorship_state,
                            entry_trigger=trigger,
                            qty_original=qty,
                            commission_entry=commission,
                            slippage_entry=slip_e * qty,
                            max_favorable=bar.high,
                            max_adverse=bar.low,
                            mfe_price=bar.high,
                            highest_close=bar.close,
                            entry_bar_idx=sym_bar_idx,
                            **enrichment,
                        )
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1

                        self._log_fsm(
                            fsm_log, sym, trade_date, now,
                            "READY_TO_ENTER", "IN_POSITION",
                            f"{'pm_reentry' if is_pm_reentry else 'entry'}|risk_unit={risk_unit:.3f}|qty={qty}",
                            acceptance_count=sym_state.acceptance_count,
                        )
                        reset_setup_state(sym_state)

                # End of bar loop for this symbol

            # 6. EOD: flatten remaining intraday positions
            for sym, pos in list(positions.items()):
                if sym in carry_positions:
                    continue  # already carried
                close_price = self._replay.get_daily_close(sym, trade_date)
                if close_price is None:
                    continue
                item = tradable_map.get(sym)
                ts_close = datetime(
                    trade_date.year, trade_date.month, trade_date.day,
                    16, 0, tzinfo=timezone.utc,
                )
                slip = close_price * slip_bps
                fill = round(close_price - slip, 2)
                commission = comm_per_share * pos.quantity
                runner_pnl = (fill - pos.entry_price) * pos.quantity
                equity += runner_pnl - commission

                total_pnl = runner_pnl + pos.realized_partial_pnl
                total_commission = pos.commission_entry + commission + pos.realized_partial_commission
                total_qty = pos.quantity + pos.partial_qty_exited
                orig_risk = pos.risk_per_share * (pos.qty_original or total_qty)
                r_mult = total_pnl / orig_risk if orig_risk > 0 else 0.0

                meta = pos.build_metadata(item, regime_risk_mult) if item else {"entry_trigger": pos.entry_trigger}

                trades.append(TradeRecord(
                    strategy="IARIC",
                    symbol=sym,
                    direction=BTDirection.LONG,
                    entry_time=pos.entry_time,
                    exit_time=ts_close,
                    entry_price=pos.entry_price,
                    exit_price=fill,
                    quantity=total_qty,
                    pnl=total_pnl,
                    r_multiple=r_mult,
                    risk_per_share=pos.risk_per_share,
                    commission=total_commission,
                    slippage=pos.slippage_entry + slip * pos.quantity,
                    entry_type=pos.entry_trigger,
                    exit_reason="EOD_FLATTEN",
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=pos.hold_bars,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=meta,
                ))

            # Flush shadow tracker
            if has_shadow:
                shadow.flush_stale()

            equity_history.append(equity)

        # Close remaining carry positions at end of backtest
        if carry_positions and trading_dates:
            last_date = trading_dates[-1]
            ts_final = datetime(last_date.year, last_date.month, last_date.day,
                                tzinfo=timezone.utc)
            for sym, pos in list(carry_positions.items()):
                close_price = self._replay.get_daily_close(sym, last_date)
                if close_price is None:
                    close_price = pos.entry_price  # fallback: flat exit
                trade, eq_delta = self._close_position(pos, close_price, ts_final, "END_OF_BACKTEST")
                equity += eq_delta
                trades.append(trade)

        logger.info(
            "IARIC v3 complete: %d trades, final equity $%.2f (%.1f%%)",
            len(trades), equity, (equity / cfg.initial_equity - 1) * 100,
        )

        return IARICIntradayV2Result(
            trades=trades,
            equity_curve=np.array(equity_history),
            timestamps=np.array([
                np.datetime64(ts.replace(tzinfo=None)) for ts in ts_history
            ]),
            daily_selections=daily_selections,
            fsm_log=fsm_log,
            rejection_log=rejection_log,
        )
