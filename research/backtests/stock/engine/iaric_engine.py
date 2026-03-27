"""Tier 2 IARIC intraday (5m bar) backtest engine.

Full 5m-bar replay of IARIC FSM logic using real signal functions:
- Replays 78 bars/day (09:30-16:00 ET) for each tradable symbol
- FSM: IDLE -> SETUP_DETECTED -> ACCEPTING -> READY_TO_ENTER -> IN_POSITION -> INVALIDATED
- Setup detection via detect_setup() with PANIC_FLUSH/DRIFT_EXHAUSTION types
- Acceptance counting with 7 adders via compute_required_acceptance()
- Confidence resolution (RED/YELLOW/GREEN) via resolve_confidence()
- Location grade (A/B/C) via compute_location_grade()
- 6-factor sizing via compute_final_risk_unit()
- Timing gate via timing_gate_allows_entry()
- Exit: time stop, partial R take, AVWAP breakdown (30m), EOD carry/flatten
- FSM event log, rejection log, shadow tracker integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from math import floor
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np

from research.backtests.stock.config import SlippageConfig
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
class _IARICPosition:
    """Backtest-internal IARIC Tier 2 position."""

    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    risk_per_share: float
    stop: float
    sector: str
    regime_tier: str
    conviction_multiplier: float
    sponsorship_state: str
    qty_original: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    hold_bars: int = 0
    partial_taken: bool = False
    partial_qty_exited: int = 0
    realized_partial_pnl: float = 0.0
    realized_partial_commission: float = 0.0
    # Enriched diagnostic fields
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
            qty_open=self.quantity - self.partial_qty_exited,
            final_stop=self.stop,
            current_stop=self.stop,
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
            "mfe_r": round(
                (self.max_favorable - self.entry_price) / rps, 4
            ) if self.max_favorable > 0 else 0,
            "mae_r": round(
                (self.entry_price - self.max_adverse) / rps, 4
            ) if self.max_adverse > 0 and self.max_adverse < self.entry_price else 0,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class IARICIntradayResult:
    """Result from the IARIC Tier 2 intraday backtest."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, WatchlistArtifact]
    fsm_log: list[dict] = field(default_factory=list)
    rejection_log: list[dict] = field(default_factory=list)


def _timing_window_label(now: datetime, settings: StrategySettings) -> str:
    """Return human-readable timing window label."""
    et = now.astimezone(ET).time()
    for start, end, _ in settings.timing_sizing:
        if start <= et < end:
            return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    return "OUTSIDE"


class IARICIntradayEngine:
    """Tier 2 IARIC backtest engine using 5m bars.

    Per trading day:
    1. Load WatchlistArtifact from research replay
    2. Initialize SymbolIntradayState per symbol (FSM starts IDLE)
    3. Replay pre-built 5m Bar objects using real signal functions for FSM transitions
    4. Entry on READY_TO_ENTER with 6-factor sizing from compute_final_risk_unit()
    5. Exit: time stop, partial, AVWAP breakdown, EOD carry/flatten
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

    def run(self) -> IARICIntradayResult:
        """Execute the full 5m-bar intraday backtest."""
        cfg = self._config
        settings = self._settings
        start = date.fromisoformat(cfg.start_date)
        end = date.fromisoformat(cfg.end_date)

        trading_dates = self._replay.tradable_dates(start, end)
        if not trading_dates:
            return IARICIntradayResult(
                trades=[], equity_curve=np.array([cfg.initial_equity]),
                timestamps=np.array([]), daily_selections={},
            )

        equity = cfg.initial_equity
        trades: list[TradeRecord] = []
        equity_history: list[float] = [equity]
        ts_history: list[datetime] = []
        daily_selections: dict[date, WatchlistArtifact] = {}
        fsm_log: list[dict] = []
        rejection_log: list[dict] = []

        # Pre-extract ablation flags to locals (hot-loop optimization)
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

        # Pre-extract settings params used in hot loop
        panic_drop = settings.panic_flush_drop_pct
        panic_minutes = settings.panic_flush_minutes
        drift_drop = settings.drift_exhaustion_drop_pct
        drift_minutes = settings.drift_exhaustion_minutes
        stale_minutes = settings.setup_stale_minutes
        base_risk_frac = settings.base_risk_fraction
        ts_minutes = settings.time_stop_minutes
        max_per_sector = cfg.max_per_sector

        has_shadow = self._shadow is not None
        shadow = self._shadow

        n_dates = len(trading_dates)
        for day_idx, trade_date in enumerate(trading_dates):
            if day_idx > 0 and day_idx % 50 == 0:
                logger.info("IARIC T2 progress: %d/%d dates (%.0f%%)",
                            day_idx, n_dates, 100 * day_idx / n_dates)

            ts = datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc)
            ts_history.append(ts)

            # Run nightly selection
            artifact = self._replay.iaric_selection_for_date(trade_date, settings)
            daily_selections[trade_date] = artifact

            if use_regime_gate and artifact.regime.tier == "C":
                equity_history.append(equity)
                continue

            max_pos = max_positions_for_regime(artifact.regime.tier, settings)
            tradable_map = {item.symbol: item for item in artifact.items}
            mwis = artifact.market_wide_institutional_selling  # for acceptance adder
            regime_risk_mult = artifact.regime.risk_multiplier

            # Per-day state
            positions: dict[str, _IARICPosition] = {}
            fsm_states: dict[str, SymbolIntradayState] = {}
            market_snapshots: dict[str, MarketSnapshot] = {}
            sector_counts: dict[str, int] = {}

            # Track bar index where session_high was set (for minutes_since_hod approx)
            hod_bar_index: dict[str, int] = {}
            bar_index: dict[str, int] = {}

            # 30m bar aggregation for AVWAP breakdown exit
            bar_30m_accum: dict[str, list[Bar]] = {}

            # VWAP accumulators
            vwap_cum_vol: dict[str, float] = {}
            vwap_cum_pv: dict[str, float] = {}

            # Initialize FSM for each tradable symbol
            for sym, item in tradable_map.items():
                # Wire up flow proxy signal from daily data (removes
                # permanent +1 acceptance adder for "flow_unavailable")
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
                    micropressure_mode="LIVE",  # backtest computes proxy
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
                bar_index[sym] = 0
                bar_30m_accum[sym] = []

            # Replay pre-built 5m Bar objects (no itertuples, no object creation)
            for sym in list(tradable_map.keys()):
                bars = self._replay.get_5m_bar_objects_for_date(sym, trade_date)
                if not bars:
                    continue
                # Filter to regular market hours (matches live engine)
                bars = [b for b in bars
                        if _MKT_OPEN <= b.start_time.astimezone(ET).time() < _MKT_CLOSE]
                if not bars:
                    continue

                item = tradable_map[sym]
                sym_state = fsm_states[sym]
                market = market_snapshots[sym]
                sec = item.sector

                # Pre-extract per-symbol constants
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

                sym_bar_idx = 0
                sym_hod_idx = 0
                cum_vol = 0.0
                cum_pv = 0.0
                accum_30m: list[Bar] = bar_30m_accum[sym]

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

                    # Track 5m bars on market for classify_trade
                    market.bars_5m.append(bar)
                    market.last_5m_bar = bar

                    # Aggregate 30m bars (every 6 5m bars)
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

                    # Shadow tracker: update bar for all active shadows
                    if has_shadow:
                        shadow.update_bar(sym, bar.high, bar.low, bar.close)

                    # ----- Manage existing position -----
                    if sym in positions:
                        pos = positions[sym]
                        pos.hold_bars += 1

                        # Track MFE/MAE
                        pos.max_favorable = max(pos.max_favorable, bar.high)
                        pos.max_adverse = min(pos.max_adverse, bar.low) if pos.max_adverse > 0 else bar.low

                        exit_price = None
                        exit_reason = ""

                        # Stop hit
                        if bar.low <= pos.stop:
                            exit_price = pos.stop
                            exit_reason = "STOP_HIT"

                        # Time stop
                        if exit_price is None and use_time_stop:
                            pos_state = pos.to_position_state()
                            if should_exit_for_time_stop(pos_state, now, bar.close):
                                exit_price = bar.close
                                exit_reason = "TIME_STOP"

                        # AVWAP breakdown (30m)
                        if (exit_price is None and use_avwap_breakdown
                                and bar_30m is not None and market.avwap_live):
                            if should_exit_for_avwap_breakdown(
                                bar_30m, market.avwap_live,
                                item.average_30m_volume, settings,
                            ):
                                exit_price = bar.close
                                exit_reason = "AVWAP_BREAKDOWN"

                        # Partial take
                        if exit_price is None and not pos.partial_taken and use_partial_take:
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

                        # EOD flatten check (15:55 ET)
                        if exit_price is None:
                            bar_et = now.astimezone(ET) if now.tzinfo else now
                            if bar_et.time() >= time(15, 55):
                                pos_state = pos.to_position_state()
                                pos.setup_tag = classify_trade(market, pos_state)

                                if use_carry:
                                    elig, reason = carry_eligible(
                                        item, market, pos_state,
                                        flow_reversal_flag=False,
                                    )
                                    if not elig:
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
                                entry_type="FSM_ENTRY",
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

                        continue  # Don't attempt entry while in position

                    # ----- FSM: attempt entry -----

                    # Funnel tracking
                    if has_shadow:
                        shadow.record_funnel("evaluated")

                    # Entry gates — only block READY_TO_ENTER → entry,
                    # NOT FSM progression (setup detection, acceptance, etc.)
                    entry_blocked = False
                    entry_block_reason = ""
                    if len(positions) >= max_pos:
                        entry_blocked = True
                        entry_block_reason = "position_cap"
                    elif not timing_gate_allows_entry(now, settings):
                        entry_blocked = True
                        entry_block_reason = "timing_blocked"
                    elif use_sector_limit and sector_counts.get(sec, 0) >= max_per_sector:
                        entry_blocked = True
                        entry_block_reason = "sector_limit"

                    # Compute approximate minutes_since_hod for detect_setup
                    approx_minutes_since_hod = (sym_bar_idx - sym_hod_idx) * 5

                    # Compute drop from HOD
                    drop_from_hod = 0.0
                    session_high = market.session_high
                    if session_high and session_high > 0:
                        drop_from_hod = (session_high - bar.close) / session_high

                    # --- FSM State Machine (matches live alpha_step) ---

                    # Global invalidation checks (before state-specific logic,
                    # matching live alpha_step order)
                    if sym_state.fsm_state in ("SETUP_DETECTED", "ACCEPTING"):
                        # Stop breach
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
                            # Fall through to INVALIDATED handler below

                        # Stale timeout
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
                                # Fall through to INVALIDATED handler below

                    # State-specific logic
                    if sym_state.fsm_state == "IDLE":
                        if has_shadow:
                            shadow.record_funnel("idle_check")

                        # Inline detect_setup() -- matches live exactly:
                        # only PANIC_FLUSH and DRIFT_EXHAUSTION
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
                        # Check reclaim touched -> transition to ACCEPTING
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
                        # Update acceptance count
                        update_acceptance(sym_state, bar)

                        # Update micropressure proxy
                        vol_for_proxy = expected_vol if expected_vol > 0 else bar.volume
                        med_for_proxy = median_vol if median_vol > 0 else bar.volume
                        if sym_state.reclaim_level:
                            mp = compute_micropressure_proxy(
                                bar, vol_for_proxy, med_for_proxy,
                                sym_state.reclaim_level,
                            )
                            sym_state.micropressure_signal = mp

                        # Resolve confidence
                        sym_state.confidence = resolve_confidence(sym_state)

                        # Check if ready (matches live: RED does NOT invalidate,
                        # setup stays in ACCEPTING and can recover)
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

                    # --- Entry on READY_TO_ENTER ---
                    if sym_state.fsm_state == "READY_TO_ENTER":
                        # Entry gate rejection (FSM already ran above)
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

                        # 6-factor sizing via compute_final_risk_unit
                        risk_unit = compute_final_risk_unit(item, sym_state, now, settings)
                        if risk_unit <= 0:
                            reset_setup_state(sym_state)
                            continue

                        risk_dollars = equity * base_risk_frac * risk_unit
                        qty = int(floor(risk_dollars / risk_per_share))
                        if qty < 1:
                            reset_setup_state(sym_state)
                            continue

                        # Buying power constraint
                        if settings.intraday_leverage > 0:
                            total_notional = sum(
                                p.entry_price * p.quantity for p in positions.values()
                            )
                            available_bp = equity * settings.intraday_leverage - total_notional
                            max_qty_bp = int(available_bp / entry_price)
                            qty = min(qty, max_qty_bp)
                            if qty < 1:
                                reset_setup_state(sym_state)
                                continue

                        if has_shadow:
                            shadow.record_funnel("entered")

                        slip = entry_price * slip_bps
                        fill_price = round(entry_price + slip, 2)
                        commission = comm_per_share * qty

                        # Compute time stop deadline
                        ts_deadline = now + timedelta(minutes=ts_minutes)

                        # Get timing window label and multiplier
                        tw_label = _timing_window_label(now, settings)
                        tm = timing_multiplier(now, settings)

                        # Get acceptance adders for metadata
                        _, adders = compute_required_acceptance(
                            item=item, sym=sym_state, now=now,
                            settings=settings,
                            market_wide_institutional_selling=mwis,
                        )

                        positions[sym] = _IARICPosition(
                            symbol=sym,
                            entry_price=fill_price,
                            entry_time=now,
                            quantity=qty,
                            risk_per_share=risk_per_share,
                            stop=stop_level,
                            sector=sec,
                            regime_tier=artifact.regime.tier,
                            conviction_multiplier=item.conviction_multiplier,
                            sponsorship_state=item.sponsorship_state,
                            qty_original=qty,
                            commission_entry=commission,
                            slippage_entry=slip * qty,
                            max_favorable=bar.high,
                            max_adverse=bar.low,
                            # Enriched fields
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
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1

                        self._log_fsm(
                            fsm_log, sym, trade_date, now,
                            "READY_TO_ENTER", "IN_POSITION",
                            f"entry|risk_unit={risk_unit:.3f}|qty={qty}",
                            acceptance_count=sym_state.acceptance_count,
                        )
                        reset_setup_state(sym_state)

            # EOD: flatten any remaining intraday positions
            for sym, pos in list(positions.items()):
                close_price = self._replay.get_daily_close(sym, trade_date)
                if close_price is None:
                    continue
                item = tradable_map.get(sym)
                ts_close = datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=timezone.utc)
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

                meta = pos.build_metadata(item, regime_risk_mult) if item else {"conviction": "N/A"}

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
                    entry_type="FSM_ENTRY",
                    exit_reason="EOD_FLATTEN",
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=pos.hold_bars,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=meta,
                ))

            # End of day: flush shadow tracker stale entries
            if has_shadow:
                shadow.flush_stale()

            equity_history.append(equity)

        logger.info(
            "IARIC Tier 2 complete: %d trades, final equity $%.2f (%.1f%%)",
            len(trades), equity, (equity / cfg.initial_equity - 1) * 100,
        )

        return IARICIntradayResult(
            trades=trades,
            equity_curve=np.array(equity_history),
            timestamps=np.array([np.datetime64(ts.replace(tzinfo=None)) for ts in ts_history]),
            daily_selections=daily_selections,
            fsm_log=fsm_log,
            rejection_log=rejection_log,
        )
