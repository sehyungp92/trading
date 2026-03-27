"""Tier 1 IARIC daily-bar backtest engine.

IARIC is primarily an intraday strategy (positions opened and closed same day,
with optional overnight carry). Tier 1 tests a simplified version:

- "Does being on the tradable watchlist predict positive returns?"
- For each tradable item with sufficient conviction:
  * Entry at same day's open (simulating successful intraday entry)
  * Default exit at same day's close (intraday strategy)
  * Carry eligible: hold overnight if regime Tier A + sponsorship STRONG + in profit
  * Flow reversal exit: 2 consecutive negative flow proxy days -> exit at open
- Position sizing uses conviction_multiplier × regime_risk_multiplier
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from math import floor

import numpy as np

from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.models import Direction as BTDirection, TradeRecord

from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.models import WatchlistArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal position tracking
# ---------------------------------------------------------------------------


@dataclass
class _IARICPosition:
    """Backtest-internal IARIC position."""

    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    risk_per_share: float
    sector: str
    regime_tier: str
    conviction_bucket: str
    conviction_multiplier: float
    sponsorship_state: str
    stop: float
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    carry_eligible: bool = False
    hold_days: int = 0
    partial_taken: bool = False
    partial_pnl: float = 0.0          # accumulated gross PnL from partial exits
    partial_qty_exited: int = 0       # total shares exited via partials
    partial_commission: float = 0.0   # accumulated commission from partial exits
    highest_carry_close: float = 0.0  # highest close during carry (for trailing stop)
    # Enriched fields for diagnostic metadata
    regime_risk_multiplier: float = 1.0
    risk_unit_final: float = 0.0

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.quantity

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (price - self.entry_price) / self.risk_per_share

    def build_metadata(self) -> dict:
        """Build enriched metadata dict matching T2 schema for diagnostics."""
        rps = max(self.risk_per_share, 0.01)
        return {
            "conviction_bucket": self.conviction_bucket,
            "conviction_multiplier": self.conviction_multiplier,
            "confidence": "DAILY",
            "location_grade": "DAILY",
            "acceptance_count": 0,
            "required_acceptance_count": 0,
            "conviction_adders": [],
            "micropressure_signal": "N/A",
            "sponsorship_state": self.sponsorship_state,
            "timing_window": "OPEN",
            "timing_multiplier": 1.0,
            "setup_type": "DAILY_ENTRY",
            "drop_from_hod_pct": round(
                (self.max_favorable - self.entry_price) / max(self.max_favorable, 0.01), 4
            ) if self.max_favorable > self.entry_price else 0.0,
            "risk_unit_final": round(self.risk_unit_final, 4),
            "partial_taken": self.partial_taken,
            "partial_qty_fraction": round(self.partial_qty_exited / max(self.quantity + self.partial_qty_exited, 1), 4),
            "regime_risk_multiplier": self.regime_risk_multiplier,
            "setup_tag": "DAILY_ENTRY",
            "mfe_r": round(
                (self.max_favorable - self.entry_price) / rps, 4
            ) if self.max_favorable > 0 else 0.0,
            "mae_r": round(
                (self.entry_price - self.max_adverse) / rps, 4
            ) if self.max_adverse > 0 and self.max_adverse < self.entry_price else 0.0,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class IARICDailyResult:
    """Result from running the IARIC daily backtest."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, WatchlistArtifact]


class IARICDailyEngine:
    """Tier 1 IARIC backtest engine using daily bars.

    For each trading day:
    1. Run research replay -> WatchlistArtifact with tradable items
    2. Enter at open for each tradable item with conviction >= CORE
    3. Exit at close (default) or carry overnight if eligible
    4. Flow reversal exits on next open
    """

    def __init__(
        self,
        config: IARICBacktestConfig,
        replay: ResearchReplayEngine,
        settings: StrategySettings | None = None,
    ):
        self._config = config
        self._replay = replay
        self._settings = settings or StrategySettings()

        if config.param_overrides:
            self._settings = replace(self._settings, **config.param_overrides)

        self._slippage = config.slippage
        self._sector_map = {sym: sector for sym, sector, _ in replay._universe}

    def run(self) -> IARICDailyResult:
        """Execute the full daily-bar backtest."""
        cfg = self._config
        settings = self._settings
        start = date.fromisoformat(cfg.start_date)
        end = date.fromisoformat(cfg.end_date)

        trading_dates = self._replay.tradable_dates(start, end)
        if not trading_dates:
            logger.warning("No trading dates in range %s to %s", start, end)
            return IARICDailyResult(
                trades=[], equity_curve=np.array([cfg.initial_equity]),
                timestamps=np.array([]), daily_selections={},
            )

        # State
        equity = cfg.initial_equity
        carry_positions: dict[str, _IARICPosition] = {}  # Overnight carries
        trades: list[TradeRecord] = []
        equity_history: list[float] = [equity]
        ts_history: list[datetime] = []
        daily_selections: dict[date, WatchlistArtifact] = {}

        for day_idx, trade_date in enumerate(trading_dates):
            ts = datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc)
            ts_history.append(ts)

            # ----- Process overnight carry positions -----
            closed_carry: list[str] = []
            for sym, pos in list(carry_positions.items()):
                ohlc = self._replay.get_daily_ohlc(sym, trade_date)
                if ohlc is None:
                    # Can't process -- close at yesterday's entry
                    closed_carry.append(sym)
                    continue

                O, H, L, C = ohlc
                pos.hold_days += 1

                # Track MFE/MAE
                pos.max_favorable = max(pos.max_favorable, H)
                pos.max_adverse = min(pos.max_adverse, L) if pos.max_adverse > 0 else L

                # Track highest close for trailing stop
                pos.highest_carry_close = max(pos.highest_carry_close, C)

                # Carry trailing stop: ratchet stop upward after activation period
                if (
                    settings.carry_trail_activate_days > 0
                    and pos.hold_days >= settings.carry_trail_activate_days
                    and pos.highest_carry_close > 0
                ):
                    if settings.carry_trail_atr_mult == 0.0:
                        trail_stop = pos.entry_price  # breakeven
                    else:
                        daily_atr = abs(H - L) if abs(H - L) > 0 else pos.risk_per_share
                        trail_stop = pos.highest_carry_close - settings.carry_trail_atr_mult * daily_atr
                    if trail_stop > pos.stop:
                        pos.stop = trail_stop

                # Check trailing stop hit (only when trailing stop is enabled)
                exit_price = None
                exit_reason = ""

                if settings.carry_trail_activate_days > 0 and L <= pos.stop:
                    exit_price = pos.stop
                    exit_reason = "CARRY_TRAIL_STOP"

                # Check flow reversal exit
                if exit_price is None and cfg.ablation.use_flow_reversal_exit:
                    # Check if flow reversed (2 consecutive negative flow proxy days)
                    # Direct array lookup — O(1) instead of full 415-symbol snapshot
                    lookback = self._settings.flow_reversal_lookback
                    last_n = self._replay.get_flow_proxy_last_n(sym, trade_date, lookback)
                    if last_n is not None and all(v < 0 for v in last_n):
                        exit_price = O
                        exit_reason = "FLOW_REVERSAL"

                # Default: exit at close (carry for one more day if still eligible)
                if exit_price is None:
                    # Check if still carry-eligible
                    cur_r = pos.unrealized_r(C)
                    tier_ok = pos.regime_tier == "A" or (
                        pos.regime_tier == "B" and settings.regime_b_carry_mult > 0
                    )
                    still_eligible = (
                        cfg.ablation.use_carry_logic
                        and tier_ok
                        and pos.sponsorship_state == "STRONG"
                        and cur_r > settings.min_carry_r
                        and pos.hold_days < settings.max_carry_days
                    )
                    if still_eligible and settings.carry_top_quartile:
                        daily_range = max(H - L, 1e-9)
                        close_pct = (C - L) / daily_range
                        if close_pct < settings.carry_close_pct_min:
                            still_eligible = False
                    if still_eligible:
                        continue  # Keep carrying
                    exit_price = C
                    exit_reason = "CARRY_EXIT"

                # Close position (remainder after any partials)
                slip = exit_price * self._slippage.slip_bps_normal / 10_000
                fill = round(exit_price - slip, 2)  # IARIC is long-only in daily sim
                commission = self._slippage.commission_per_share * pos.quantity
                remainder_pnl = (fill - pos.entry_price) * pos.quantity
                total_pnl = remainder_pnl + pos.partial_pnl
                total_qty = pos.quantity + pos.partial_qty_exited
                r_mult = pos.unrealized_r(fill)

                equity += remainder_pnl - commission
                trades.append(TradeRecord(
                    strategy="IARIC",
                    symbol=sym,
                    direction=BTDirection.LONG,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=fill,
                    quantity=total_qty,
                    pnl=total_pnl,
                    r_multiple=r_mult,
                    risk_per_share=pos.risk_per_share,
                    commission=pos.commission_entry + pos.partial_commission + commission,
                    slippage=pos.slippage_entry + slip * pos.quantity,
                    entry_type="SPONSORSHIP",
                    exit_reason=exit_reason,
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=pos.hold_days,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=pos.build_metadata(),
                ))

                if cfg.verbose:
                    logger.info(
                        "[%s] CARRY EXIT %s @ %.2f reason=%s PnL=%.2f R=%.2f",
                        trade_date, sym, fill, exit_reason, total_pnl, r_mult,
                    )

                closed_carry.append(sym)

            for sym in closed_carry:
                carry_positions.pop(sym, None)

            # ----- Run nightly selection -----
            artifact = self._replay.iaric_selection_for_date(trade_date, self._settings)
            daily_selections[trade_date] = artifact

            # Regime gate
            if cfg.ablation.use_regime_gate and artifact.regime.tier == "C":
                equity_history.append(equity)
                continue

            # Determine max positions for regime
            if artifact.regime.tier == "A":
                max_pos = cfg.max_positions_tier_a
            else:
                max_pos = cfg.max_positions_tier_b

            # ----- Enter new positions -----
            active_count = len(carry_positions)
            sector_counts: dict[str, int] = {}
            for pos in carry_positions.values():
                sector_counts[pos.sector] = sector_counts.get(pos.sector, 0) + 1

            intraday_positions: list[_IARICPosition] = []

            # Pre-compute per-day invariants (avoid repeated calls inside item loop)
            prev_date = (
                self._replay.get_prev_trading_date(trade_date)
                if settings.t1_gap_down_skip_pct > 0 else None
            )
            dow = trade_date.weekday()  # 0=Mon, 1=Tue, ..., 4=Fri

            tradable_items = artifact.tradable
            if settings.entry_order_by_conviction:
                tradable_items = sorted(
                    artifact.tradable,
                    key=lambda it: it.conviction_multiplier,
                    reverse=True,
                )

            for item in tradable_items:
                sym = item.symbol
                if sym in carry_positions:
                    continue  # Already carrying

                # Carry-only entry: skip if trade can never carry
                if settings.carry_only_entry:
                    can_carry = (
                        item.sponsorship_state == "STRONG"
                        and (artifact.regime.tier == "A"
                             or (artifact.regime.tier == "B"
                                 and settings.regime_b_carry_mult > 0))
                    )
                    if not can_carry:
                        continue

                # Strong-only entry: require STRONG sponsorship
                if settings.strong_only_entry and item.sponsorship_state != "STRONG":
                    continue

                # Conviction filter
                if item.conviction_multiplier <= settings.min_conviction_multiplier:
                    continue

                # Entry flow gate: skip if recent flow proxy is negative
                if settings.t1_entry_flow_gate:
                    last_n = self._replay.get_flow_proxy_last_n(
                        sym, trade_date, settings.t1_entry_flow_lookback
                    )
                    if last_n is not None and any(v < 0 for v in last_n):
                        continue

                # Position limit
                if active_count + len(intraday_positions) >= max_pos:
                    break

                # Sector limit
                sector = item.sector
                sec_count = sector_counts.get(sector, 0)
                if cfg.ablation.use_sector_limit and sec_count >= cfg.max_per_sector:
                    continue

                # Get today's OHLC
                ohlc = self._replay.get_daily_ohlc(sym, trade_date)
                if ohlc is None:
                    continue
                O, H, L, C = ohlc

                # Gap-down filter: skip if open gaps below previous close
                if prev_date is not None:
                    prev_close = self._replay.get_daily_close(sym, prev_date)
                    if prev_close is not None and prev_close > 0:
                        gap = (O - prev_close) / prev_close
                        if gap < -settings.t1_gap_down_skip_pct:
                            continue

                # Entry at open with slippage
                slip_bps = self._slippage.slip_bps_normal
                slip = O * slip_bps / 10_000
                fill_price = round(O + slip, 2)

                # Risk: daily ATR-based stop
                daily_atr = item.daily_atr_estimate
                if daily_atr <= 0:
                    daily_atr = abs(H - L)
                risk_offset = min(daily_atr, fill_price * settings.stop_risk_cap_pct)
                stop_price = fill_price - risk_offset
                risk_per_share = abs(fill_price - stop_price)
                if risk_per_share <= 0:
                    continue

                # Size: conviction × regime × base_risk
                risk_mult = item.conviction_multiplier * artifact.regime.risk_multiplier
                if not cfg.ablation.use_conviction_scaling:
                    risk_mult = artifact.regime.risk_multiplier
                risk_dollars = equity * settings.base_risk_fraction * risk_mult

                # Day-of-week sizing adjustment
                if dow == 1:  # Tuesday
                    risk_dollars *= settings.dow_tuesday_mult
                elif dow == 4:  # Friday
                    risk_dollars *= settings.dow_friday_mult
                qty = int(floor(risk_dollars / risk_per_share))
                if qty < 1:
                    continue

                # Buying power constraint
                if settings.intraday_leverage > 0:
                    carry_notional = sum(
                        p.entry_price * p.quantity for p in carry_positions.values()
                    )
                    intraday_notional = sum(
                        p.entry_price * p.quantity for p in intraday_positions
                    )
                    available_bp = equity * settings.intraday_leverage - carry_notional - intraday_notional
                    max_qty_bp = int(available_bp / fill_price)
                    qty = min(qty, max_qty_bp)
                    if qty < 1:
                        continue

                commission = self._slippage.commission_per_share * qty

                pos = _IARICPosition(
                    symbol=sym,
                    entry_price=fill_price,
                    entry_time=ts,
                    quantity=qty,
                    risk_per_share=risk_per_share,
                    sector=sector,
                    regime_tier=artifact.regime.tier,
                    conviction_bucket=item.conviction_bucket,
                    conviction_multiplier=item.conviction_multiplier,
                    sponsorship_state=item.sponsorship_state,
                    stop=stop_price,
                    commission_entry=commission,
                    slippage_entry=slip * qty,
                    max_favorable=H,
                    max_adverse=L,
                    regime_risk_multiplier=artifact.regime.risk_multiplier,
                    risk_unit_final=risk_mult,
                )

                intraday_positions.append(pos)
                sector_counts[sector] = sec_count + 1

                if cfg.verbose:
                    logger.info(
                        "[%s] ENTRY %s @ %.2f qty=%d conviction=%s regime=%s",
                        trade_date, sym, fill_price, qty,
                        item.conviction_bucket, artifact.regime.tier,
                    )

            # ----- Process intraday positions: exit at close or carry -----
            for pos in intraday_positions:
                ohlc = self._replay.get_daily_ohlc(pos.symbol, trade_date)
                if ohlc is None:
                    continue
                O, H, L, C = ohlc

                # Intraday partial take: if H reaches trigger R, take fraction off
                if (
                    settings.t1_partial_takes
                    and not pos.partial_taken
                    and pos.risk_per_share > 0
                ):
                    trigger_price = pos.entry_price + settings.t1_partial_r_trigger * pos.risk_per_share
                    if H >= trigger_price:
                        partial_qty = max(1, int(floor(pos.quantity * settings.t1_partial_fraction)))
                        partial_fill = round(trigger_price - trigger_price * self._slippage.slip_bps_normal / 10_000, 2)
                        partial_comm = self._slippage.commission_per_share * partial_qty
                        partial_gross = (partial_fill - pos.entry_price) * partial_qty
                        equity += partial_gross - partial_comm
                        pos.partial_taken = True
                        pos.partial_pnl = partial_gross
                        pos.partial_commission = partial_comm
                        pos.partial_qty_exited = partial_qty
                        pos.quantity -= partial_qty

                        if cfg.verbose:
                            logger.info(
                                "[%s] PARTIAL %s qty=%d @ %.2f (%.1fR trigger)",
                                trade_date, pos.symbol, partial_qty, partial_fill,
                                settings.t1_partial_r_trigger,
                            )

                        if pos.quantity < 1:
                            # Entire position exited via partial
                            trades.append(TradeRecord(
                                strategy="IARIC",
                                symbol=pos.symbol,
                                direction=BTDirection.LONG,
                                entry_time=pos.entry_time,
                                exit_time=ts,
                                entry_price=pos.entry_price,
                                exit_price=partial_fill,
                                quantity=pos.partial_qty_exited,
                                pnl=pos.partial_pnl,
                                r_multiple=pos.unrealized_r(partial_fill),
                                risk_per_share=pos.risk_per_share,
                                commission=pos.commission_entry + pos.partial_commission,
                                slippage=pos.slippage_entry,
                                entry_type="SPONSORSHIP",
                                exit_reason="PARTIAL_FULL",
                                sector=pos.sector,
                                regime_tier=pos.regime_tier,
                                hold_bars=1,
                                max_favorable=pos.max_favorable,
                                max_adverse=pos.max_adverse,
                                metadata=pos.build_metadata(),
                            ))
                            continue

                exit_price = None
                exit_reason = ""

                # Intraday flow check — exit at close if flow reversed
                if settings.intraday_flow_check and cfg.ablation.use_flow_reversal_exit:
                    lookback = settings.flow_reversal_lookback
                    last_n = self._replay.get_flow_proxy_last_n(pos.symbol, trade_date, lookback)
                    if last_n is not None and all(v < 0 for v in last_n):
                        exit_price = C
                        exit_reason = "FLOW_REVERSAL_INTRADAY"

                # Check stop
                if exit_price is None:
                    if settings.use_close_stop:
                        if C < pos.entry_price:
                            exit_price = C
                            exit_reason = "CLOSE_STOP"
                        # else: profitable, falls through to carry check
                    elif L <= pos.stop:
                        exit_price = pos.stop
                        exit_reason = "STOP_HIT"

                # Carry check (only if not already exiting)
                if exit_price is None:
                    cur_r = pos.unrealized_r(C)
                    carry = (
                        cfg.ablation.use_carry_logic
                        and pos.sponsorship_state == "STRONG"
                        and cur_r > settings.min_carry_r
                        and (artifact.regime.tier == "A"
                             or (artifact.regime.tier == "B"
                                 and settings.regime_b_carry_mult > 0))
                    )

                    # Top-quartile carry gate (applies to all carry types)
                    if carry and settings.carry_top_quartile:
                        daily_range = max(H - L, 1e-9)
                        close_pct = (C - L) / daily_range
                        if close_pct < settings.carry_close_pct_min:
                            carry = False

                    if carry:
                        pos.carry_eligible = True
                        pos.hold_days = 1
                        carry_positions[pos.symbol] = pos
                        continue  # Don't close -- carry overnight

                    exit_price = C
                    exit_reason = "EOD_FLATTEN"

                # Close (remainder after any partials)
                slip = exit_price * self._slippage.slip_bps_normal / 10_000
                fill = round(exit_price - slip, 2)
                commission = self._slippage.commission_per_share * pos.quantity
                remainder_pnl = (fill - pos.entry_price) * pos.quantity
                total_pnl = remainder_pnl + pos.partial_pnl
                total_qty = pos.quantity + pos.partial_qty_exited
                r_mult = pos.unrealized_r(fill)

                equity += remainder_pnl - commission
                trades.append(TradeRecord(
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
                    commission=pos.commission_entry + pos.partial_commission + commission,
                    slippage=pos.slippage_entry + slip * pos.quantity,
                    entry_type="SPONSORSHIP",
                    exit_reason=exit_reason,
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=1,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=pos.build_metadata(),
                ))

                if cfg.verbose:
                    logger.info(
                        "[%s] EXIT %s @ %.2f reason=%s PnL=%.2f R=%.2f",
                        trade_date, pos.symbol, fill, exit_reason, total_pnl, r_mult,
                    )

            equity_history.append(equity)

        # Close any remaining carry positions at last available price
        if carry_positions and trading_dates:
            last_date = trading_dates[-1]
            for sym, pos in list(carry_positions.items()):
                close_price = self._replay.get_daily_close(sym, last_date)
                if close_price is None:
                    continue
                ts = datetime(last_date.year, last_date.month, last_date.day, tzinfo=timezone.utc)
                commission = self._slippage.commission_per_share * pos.quantity
                remainder_pnl = (close_price - pos.entry_price) * pos.quantity
                total_pnl = remainder_pnl + pos.partial_pnl
                total_qty = pos.quantity + pos.partial_qty_exited
                r_mult = pos.unrealized_r(close_price)
                equity += remainder_pnl - commission

                trades.append(TradeRecord(
                    strategy="IARIC",
                    symbol=sym,
                    direction=BTDirection.LONG,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=close_price,
                    quantity=total_qty,
                    pnl=total_pnl,
                    r_multiple=r_mult,
                    risk_per_share=pos.risk_per_share,
                    commission=pos.commission_entry + pos.partial_commission + commission,
                    slippage=pos.slippage_entry,
                    entry_type="SPONSORSHIP",
                    exit_reason="END_OF_BACKTEST",
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=pos.hold_days,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=pos.build_metadata(),
                ))

        logger.info(
            "IARIC Tier 1 complete: %d trades, final equity: $%.2f (%.1f%%)",
            len(trades), equity, (equity / cfg.initial_equity - 1) * 100,
        )

        return IARICDailyResult(
            trades=trades,
            equity_curve=np.array(equity_history),
            timestamps=np.array([
                np.datetime64(ts.replace(tzinfo=None)) for ts in ts_history
            ]),
            daily_selections=daily_selections,
        )
