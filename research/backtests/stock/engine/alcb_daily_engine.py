"""Tier 1 ALCB daily-bar backtest engine.

Tests whether the ALCB research/scoring/ranking pipeline generates
selection alpha using simplified daily-bar entries and exits:
- Entry: next day's open after breakout qualifies on day T
- Stop: box.low - ATR_mult × ATR14
- TP1/TP2 partials + runner trailing (daily ATR based)
- Stale exit, dirty state management, continuation entries
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from math import floor
from pathlib import Path

import numpy as np

from research.backtests.stock.config import SlippageConfig
from research.backtests.stock.config_alcb import ALCBBacktestConfig
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.models import Direction as BTDirection, TradeRecord

from strategies.stock.alcb.config import StrategySettings
from strategies.stock.alcb.exits import (
    breakeven_plus_buffer,
    business_days_between,
    gap_through_stop,
    maybe_enable_continuation,
    stale_exit_needed,
    tp1_hit,
    tp2_hit,
    update_dirty_state,
)
from strategies.stock.alcb.models import (
    Box,
    BreakoutQualification,
    Campaign,
    CampaignState,
    CandidateArtifact,
    CandidateItem,
    Direction,
    EntryType,
    PortfolioState,
    PositionPlan,
    PositionState,
    Regime,
    RegimeSnapshot,
)
from strategies.stock.alcb.risk import (
    base_risk_fraction,
    choose_stop,
    choose_targets,
    correlation_mult as compute_corr_mult,
    estimate_cost_buffer_per_share,
    friction_gate_pass,
    max_positions_pass,
    quality_mult as compute_quality_mult,
    regime_mult as compute_regime_mult,
    sector_limit_pass,
)
from strategies.stock.alcb.signals import atr_from_bars, detect_compression_box

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal position tracking
# ---------------------------------------------------------------------------


@dataclass
class _OpenPosition:
    """Backtest-internal position state wrapper."""

    symbol: str
    direction: Direction
    entry_type: EntryType
    entry_price: float
    entry_time: datetime
    quantity: int
    stop: float
    tp1: float
    tp2: float
    risk_per_share: float
    sector: str
    regime_tier: str
    campaign: Campaign
    qty_original: int = 0  # Original entry quantity (before partials)
    partial_taken: bool = False
    tp2_taken: bool = False
    profit_funded: bool = False
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    hold_bars: int = 0
    realized_partial_pnl: float = 0.0  # Accumulated PnL from TP1/TP2 partials
    realized_partial_commission: float = 0.0  # Accumulated commission from partials
    realized_partial_slippage: float = 0.0  # Accumulated slippage from partials
    realized_partial_qty: int = 0  # Total shares closed via partials
    # Diagnostic metadata fields
    initial_stop: float = 0.0
    quality_mult: float = 1.0
    regime_mult_val: float = 1.0
    corr_mult: float = 1.0
    intraday_score: int = 0
    intraday_mode: str = ""
    risk_dollars: float = 0.0

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.quantity

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        if self.direction == Direction.LONG:
            return (price - self.entry_price) / self.risk_per_share
        return (self.entry_price - price) / self.risk_per_share

    def build_metadata(self) -> dict:
        """Build diagnostic metadata dict for TradeRecord."""
        rps = self.risk_per_share or 1.0
        if self.direction == Direction.LONG:
            mfe_r = (self.max_favorable - self.entry_price) / rps
            mae_r = (self.entry_price - self.max_adverse) / rps if self.max_adverse > 0 else 0.0
        else:
            mfe_r = (self.entry_price - self.max_favorable) / rps if self.max_favorable > 0 else 0.0
            mae_r = (self.max_adverse - self.entry_price) / rps

        meta: dict = {
            "initial_stop": self.initial_stop,
            "exit_stop": self.stop,
            "tp1_price": self.tp1,
            "tp2_price": self.tp2,
            "qty_original": self.qty_original,
            "partial_taken": self.partial_taken,
            "tp2_taken": self.tp2_taken,
            "realized_partial_pnl": self.realized_partial_pnl,
            "quality_mult": self.quality_mult,
            "regime_mult": self.regime_mult_val,
            "corr_mult": self.corr_mult,
            "risk_dollars": self.risk_dollars,
            "intraday_score": self.intraday_score,
            "mfe_r": round(mfe_r, 4),
            "mae_r": round(mae_r, 4),
        }

        cam = self.campaign
        if cam:
            meta["campaign_state"] = cam.state.value
            meta["dirty_recovery"] = cam.dirty_since is not None
            meta["continuation_enabled"] = cam.continuation_enabled
            if cam.box:
                meta["box_tier"] = cam.box.tier.value
                meta["box_containment"] = cam.box.containment
                meta["box_squeeze"] = cam.box.squeeze_metric
                meta["box_length"] = cam.box.L_used
            if cam.breakout:
                dt = cam.breakout.disp_threshold
                meta["disp_ratio"] = round(cam.breakout.disp_value / dt, 4) if dt > 0 else 0.0
                meta["breakout_rvol"] = cam.breakout.rvol_d

        return meta

    def to_position_state(self) -> PositionState:
        """Convert to strategy PositionState for reuse of exit functions."""
        return PositionState(
            direction=self.direction,
            entry_price=self.entry_price,
            qty_entry=self.quantity,
            qty_open=self.quantity,
            final_stop=self.stop,
            current_stop=self.stop,
            entry_time=self.entry_time,
            initial_risk_per_share=self.risk_per_share,
            max_favorable_price=self.max_favorable,
            max_adverse_price=self.max_adverse,
            tp1_price=self.tp1,
            tp2_price=self.tp2,
            partial_taken=self.partial_taken,
            tp2_taken=self.tp2_taken,
            profit_funded=self.profit_funded,
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class ALCBDailyResult:
    """Result from running the ALCB daily backtest."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, CandidateArtifact]


class ALCBDailyEngine:
    """Tier 1 ALCB backtest engine using daily bars.

    For each trading day:
    1. Run research replay → CandidateArtifact with tradable candidates
    2. Enter: next day's open for qualified breakouts
    3. Manage: gap-through-stop, stop hit, TP1/TP2 partials,
       runner trailing, stale exit, dirty state
    4. Portfolio: max 5 positions, max 2/sector, 6R heat cap, regime gate
    """

    def __init__(
        self,
        config: ALCBBacktestConfig,
        replay: ResearchReplayEngine,
        settings: StrategySettings | None = None,
    ):
        self._config = config
        self._replay = replay
        self._settings = settings or StrategySettings()

        # Apply param overrides
        if config.param_overrides:
            self._settings = replace(self._settings, **config.param_overrides)

        self._slippage = config.slippage
        self._sector_map = {sym: sector for sym, sector, _ in replay._universe}

    def run(self) -> ALCBDailyResult:
        """Execute the full daily-bar backtest."""
        cfg = self._config
        settings = self._settings
        start = date.fromisoformat(cfg.start_date)
        end = date.fromisoformat(cfg.end_date)

        trading_dates = self._replay.tradable_dates(start, end)
        if not trading_dates:
            logger.warning("No trading dates in range %s to %s", start, end)
            return ALCBDailyResult(
                trades=[], equity_curve=np.array([cfg.initial_equity]),
                timestamps=np.array([]), daily_selections={},
            )

        # State
        equity = cfg.initial_equity
        positions: dict[str, _OpenPosition] = {}
        campaigns: dict[str, Campaign] = {}  # Track campaigns across days
        trades: list[TradeRecord] = []
        equity_history: list[float] = [equity]
        ts_history: list[datetime] = []
        daily_selections: dict[date, CandidateArtifact] = {}

        # Pending entries: symbols to enter at next day's open
        pending_entries: list[tuple[CandidateItem, Direction, Campaign, RegimeSnapshot]] = []

        n_dates = len(trading_dates)
        for day_idx, trade_date in enumerate(trading_dates):
            if day_idx > 0 and day_idx % 50 == 0:
                logger.info("ALCB T1 progress: %d/%d dates (%.0f%%)",
                            day_idx, n_dates, 100 * day_idx / n_dates)
            ts = datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc)
            ts_history.append(ts)

            # ----- Process pending entries (fill at today's open) -----
            for item, direction, campaign, regime in pending_entries:
                open_price = self._replay.get_daily_close(item.symbol, trade_date)
                ohlc = self._replay.get_daily_ohlc(item.symbol, trade_date)
                if ohlc is None:
                    continue
                open_price = ohlc[0]

                # Apply slippage
                slip_bps = self._slippage.slip_bps_normal
                slip = open_price * slip_bps / 10_000
                if direction == Direction.LONG:
                    fill_price = round(open_price + slip, 2)
                else:
                    fill_price = round(open_price - slip, 2)

                # Re-compute stop from current item data
                stop_price = choose_stop(
                    EntryType.A_AVWAP_RETEST, direction, item, campaign, settings,
                )
                if abs(fill_price - stop_price) < 0.01:
                    continue

                risk_per_share = abs(fill_price - stop_price) + estimate_cost_buffer_per_share(item, fill_price)
                if risk_per_share <= 0:
                    continue

                # Size position
                risk_fraction = base_risk_fraction(item, settings)
                risk_dollars = equity * risk_fraction
                qty = int(floor(risk_dollars / risk_per_share))
                if qty < 1:
                    continue

                # Check portfolio constraints
                portfolio = self._build_portfolio(positions, equity, settings)
                if not max_positions_pass(portfolio, settings):
                    continue
                if not sector_limit_pass(item, portfolio, self._sector_map, settings):
                    continue

                # Heat cap check
                current_heat = portfolio.open_risk_dollars()
                proposed_heat = current_heat + qty * risk_per_share
                max_heat = equity * settings.heat_cap_r * settings.base_risk_fraction
                if proposed_heat > max_heat:
                    continue

                _stock_regime = Regime(item.stock_regime) if item.stock_regime in Regime._value2member_map_ else Regime.TRANSITIONAL
                _market_regime = Regime(item.market_regime) if item.market_regime in Regime._value2member_map_ else Regime.TRANSITIONAL

                tp1, tp2 = choose_targets(
                    direction, fill_price, stop_price,
                    _stock_regime, _market_regime, settings,
                )

                # Compute diagnostic multipliers (not used for sizing)
                _regime_m = compute_regime_mult(direction, _stock_regime, _market_regime)
                _quality_m = compute_quality_mult(campaign, 0, settings)
                _tradable_map = {c.symbol: c for c in artifact.tradable}
                _corr_m = compute_corr_mult(item.symbol, direction, portfolio, _tradable_map, settings)

                commission = self._slippage.commission_per_share * qty
                positions[item.symbol] = _OpenPosition(
                    symbol=item.symbol,
                    direction=direction,
                    entry_type=EntryType.A_AVWAP_RETEST,
                    entry_price=fill_price,
                    entry_time=ts,
                    quantity=qty,
                    stop=stop_price,
                    tp1=tp1,
                    tp2=tp2,
                    risk_per_share=risk_per_share,
                    sector=item.sector,
                    regime_tier=regime.tier,
                    campaign=campaign,
                    qty_original=qty,
                    commission_entry=commission,
                    slippage_entry=slip * qty,
                    initial_stop=stop_price,
                    quality_mult=_quality_m,
                    regime_mult_val=_regime_m,
                    corr_mult=_corr_m,
                    risk_dollars=risk_dollars,
                )

                if cfg.verbose:
                    logger.info(
                        "[%s] ENTRY %s %s @ %.2f stop=%.2f qty=%d R/share=%.2f",
                        trade_date, direction.value, item.symbol, fill_price,
                        stop_price, qty, risk_per_share,
                    )

            # ----- Manage existing positions -----
            closed_today: list[str] = []
            for sym, pos in list(positions.items()):
                ohlc = self._replay.get_daily_ohlc(sym, trade_date)
                if ohlc is None:
                    continue
                O, H, L, C = ohlc
                pos.hold_bars += 1

                # Track MFE/MAE
                if pos.direction == Direction.LONG:
                    pos.max_favorable = max(pos.max_favorable, H)
                    pos.max_adverse = min(pos.max_adverse, L) if pos.max_adverse > 0 else L
                else:
                    pos.max_favorable = min(pos.max_favorable, L) if pos.max_favorable > 0 else L
                    pos.max_adverse = max(pos.max_adverse, H)

                exit_price = None
                exit_reason = ""

                # 1. Gap through stop (open beyond stop)
                pos_state = pos.to_position_state()
                if gap_through_stop(pos_state, O):
                    exit_price = O
                    exit_reason = "GAP_THROUGH_STOP"

                # 2. Stop hit (intrabar)
                if exit_price is None:
                    if pos.direction == Direction.LONG and L <= pos.stop:
                        exit_price = pos.stop
                        exit_reason = "STOP_HIT"
                    elif pos.direction == Direction.SHORT and H >= pos.stop:
                        exit_price = pos.stop
                        exit_reason = "STOP_HIT"

                # 3. TP1 hit → partial exit (30%)
                if exit_price is None and not pos.partial_taken:
                    if pos.direction == Direction.LONG and H >= pos.tp1:
                        partial_qty = max(1, int(pos.quantity * settings.tp1_fraction))
                        partial_pnl = (pos.tp1 - pos.entry_price) * partial_qty
                        commission = self._slippage.commission_per_share * partial_qty
                        equity += partial_pnl - commission
                        pos.quantity -= partial_qty
                        pos.partial_taken = True
                        pos.profit_funded = True
                        pos.realized_partial_pnl += partial_pnl
                        pos.realized_partial_commission += commission
                        pos.realized_partial_qty += partial_qty

                        # Move stop to breakeven + buffer
                        daily_atr = self._daily_atr(sym, trade_date)
                        be_stop = breakeven_plus_buffer(pos_state, daily_atr, settings)
                        pos.stop = max(pos.stop, be_stop)

                        if cfg.log_trades:
                            logger.debug("[%s] TP1 partial %s: %d shares @ %.2f", trade_date, sym, partial_qty, pos.tp1)

                        if pos.quantity <= 0:
                            exit_price = pos.tp1
                            exit_reason = "TP1_FULL_EXIT"

                    elif pos.direction == Direction.SHORT and L <= pos.tp1:
                        partial_qty = max(1, int(pos.quantity * settings.tp1_fraction))
                        partial_pnl = (pos.entry_price - pos.tp1) * partial_qty
                        commission = self._slippage.commission_per_share * partial_qty
                        equity += partial_pnl - commission
                        pos.quantity -= partial_qty
                        pos.partial_taken = True
                        pos.profit_funded = True
                        pos.realized_partial_pnl += partial_pnl
                        pos.realized_partial_commission += commission
                        pos.realized_partial_qty += partial_qty

                        daily_atr = self._daily_atr(sym, trade_date)
                        be_stop = breakeven_plus_buffer(pos_state, daily_atr, settings)
                        pos.stop = min(pos.stop, be_stop)

                        if pos.quantity <= 0:
                            exit_price = pos.tp1
                            exit_reason = "TP1_FULL_EXIT"

                # 4. TP2 hit → partial exit (30%)
                if exit_price is None and pos.partial_taken and not pos.tp2_taken:
                    if pos.direction == Direction.LONG and H >= pos.tp2:
                        partial_qty = max(1, int(pos.quantity * settings.tp2_fraction))
                        partial_pnl = (pos.tp2 - pos.entry_price) * partial_qty
                        commission = self._slippage.commission_per_share * partial_qty
                        equity += partial_pnl - commission
                        pos.quantity -= partial_qty
                        pos.tp2_taken = True
                        pos.realized_partial_pnl += partial_pnl
                        pos.realized_partial_commission += commission
                        pos.realized_partial_qty += partial_qty

                        if pos.quantity <= 0:
                            exit_price = pos.tp2
                            exit_reason = "TP2_FULL_EXIT"

                    elif pos.direction == Direction.SHORT and L <= pos.tp2:
                        partial_qty = max(1, int(pos.quantity * settings.tp2_fraction))
                        partial_pnl = (pos.entry_price - pos.tp2) * partial_qty
                        commission = self._slippage.commission_per_share * partial_qty
                        equity += partial_pnl - commission
                        pos.quantity -= partial_qty
                        pos.tp2_taken = True
                        pos.realized_partial_pnl += partial_pnl
                        pos.realized_partial_commission += commission
                        pos.realized_partial_qty += partial_qty

                        if pos.quantity <= 0:
                            exit_price = pos.tp2
                            exit_reason = "TP2_FULL_EXIT"

                # 5. Runner trailing stop (simplified: daily ATR based)
                if exit_price is None and pos.partial_taken and cfg.ablation.use_trailing_stop:
                    daily_atr = self._daily_atr(sym, trade_date)
                    if daily_atr > 0:
                        if pos.direction == Direction.LONG:
                            trail = C - 1.5 * daily_atr
                            pos.stop = max(pos.stop, trail)
                        else:
                            trail = C + 1.5 * daily_atr
                            pos.stop = min(pos.stop, trail)

                # 6. Stale exit
                if exit_price is None and cfg.ablation.use_stale_exit:
                    if stale_exit_needed(pos_state, ts, C, settings):
                        exit_price = C
                        exit_reason = "STALE_EXIT"

                # Process full exit
                if exit_price is not None and pos.quantity > 0:
                    slip = exit_price * self._slippage.slip_bps_normal / 10_000
                    if pos.direction == Direction.LONG:
                        fill = round(exit_price - slip, 2)
                    else:
                        fill = round(exit_price + slip, 2)
                    commission = self._slippage.commission_per_share * pos.quantity

                    if pos.direction == Direction.LONG:
                        runner_pnl = (fill - pos.entry_price) * pos.quantity
                    else:
                        runner_pnl = (pos.entry_price - fill) * pos.quantity

                    # Combine runner + partial PnLs for complete trade record
                    total_pnl = runner_pnl + pos.realized_partial_pnl
                    total_commission = (
                        pos.commission_entry + commission + pos.realized_partial_commission
                    )
                    total_slippage = pos.slippage_entry + slip * pos.quantity + pos.realized_partial_slippage
                    total_qty = pos.quantity + pos.realized_partial_qty

                    # R-multiple based on total PnL vs original risk
                    orig_risk = pos.risk_per_share * (pos.qty_original or total_qty)
                    r_mult = total_pnl / orig_risk if orig_risk > 0 else 0.0

                    equity += runner_pnl - commission
                    trades.append(TradeRecord(
                        strategy="ALCB",
                        symbol=sym,
                        direction=BTDirection.LONG if pos.direction == Direction.LONG else BTDirection.SHORT,
                        entry_time=pos.entry_time,
                        exit_time=ts,
                        entry_price=pos.entry_price,
                        exit_price=fill,
                        quantity=total_qty,
                        pnl=total_pnl,
                        r_multiple=r_mult,
                        risk_per_share=pos.risk_per_share,
                        commission=total_commission,
                        slippage=total_slippage,
                        entry_type=pos.entry_type.value,
                        exit_reason=exit_reason,
                        sector=pos.sector,
                        regime_tier=pos.regime_tier,
                        hold_bars=pos.hold_bars,
                        max_favorable=pos.max_favorable,
                        max_adverse=pos.max_adverse,
                        metadata=pos.build_metadata(),
                    ))

                    if cfg.verbose:
                        logger.info(
                            "[%s] EXIT %s %s @ %.2f reason=%s PnL=%.2f R=%.2f",
                            trade_date, pos.direction.value, sym, fill,
                            exit_reason, total_pnl, r_mult,
                        )

                    closed_today.append(sym)

                    # Update dirty state on campaign
                    if cfg.ablation.use_dirty_state and sym in campaigns:
                        update_dirty_state(
                            campaigns[sym], C, settings, trade_date,
                            daily_bars=self._get_alcb_daily_bars(sym, trade_date),
                        )

                    # Check continuation enabling
                    if cfg.ablation.use_continuation_entries and sym in campaigns:
                        daily_atr = self._daily_atr(sym, trade_date)
                        maybe_enable_continuation(campaigns[sym], C, daily_atr, settings)

            # Remove closed positions
            for sym in closed_today:
                positions.pop(sym, None)

            # ----- Run nightly selection -----
            artifact = self._replay.alcb_selection_for_date(trade_date, self._settings)
            daily_selections[trade_date] = artifact

            # Regime gate
            if cfg.ablation.use_regime_gate and artifact.regime.tier == "C":
                equity_history.append(equity)
                continue

            # Identify new entry candidates via box detection + daily breakout
            pending_entries = []
            for candidate in artifact.tradable:
                sym = candidate.symbol
                if sym in positions:
                    continue  # Already in position

                daily_bars = candidate.daily_bars
                if not daily_bars or len(daily_bars) < 21:
                    continue

                # Detect compression box from PRIOR bars (exclude today).
                # Then check if today's close breaks out of that prior box.
                prior_bars = daily_bars[:-1]
                box = detect_compression_box(prior_bars, self._settings)
                if box is None:
                    continue

                # Check for daily breakout: today's close outside prior box
                todays_close = daily_bars[-1].close
                tol = box.height * self._settings.breakout_tolerance_pct
                direction: Direction | None = None
                if todays_close > box.high - tol:
                    direction = Direction.LONG
                elif todays_close < box.low + tol:
                    direction = Direction.SHORT
                else:
                    continue  # Still inside box, no breakout

                # Build campaign with BREAKOUT state
                campaign = campaigns.get(sym) or Campaign(symbol=sym)
                campaign.box = box
                campaign.state = CampaignState.BREAKOUT
                campaign.breakout = BreakoutQualification(
                    direction=direction,
                    breakout_date=str(trade_date),
                    structural_pass=True,
                    displacement_pass=True,  # skip for Tier 1 (needs 30m bars)
                    disp_value=0.0,
                    disp_threshold=0.0,
                    breakout_rejected=False,
                    rvol_d=1.0,
                )
                campaigns[sym] = campaign

                pending_entries.append((candidate, direction, campaign, artifact.regime))

            equity_history.append(equity)

        # Final: close any remaining open positions at last available price
        if positions and trading_dates:
            last_date = trading_dates[-1]
            for sym, pos in list(positions.items()):
                close_price = self._replay.get_daily_close(sym, last_date)
                if close_price is None:
                    continue
                ts = datetime(last_date.year, last_date.month, last_date.day, tzinfo=timezone.utc)

                # Apply exit slippage (consistent with main exit path)
                slip = close_price * self._slippage.slip_bps_normal / 10_000
                if pos.direction == Direction.LONG:
                    fill = round(close_price - slip, 2)
                    runner_pnl = (fill - pos.entry_price) * pos.quantity
                else:
                    fill = round(close_price + slip, 2)
                    runner_pnl = (pos.entry_price - fill) * pos.quantity

                commission = self._slippage.commission_per_share * pos.quantity

                # Combine runner + partial PnLs (partials already added to equity)
                total_pnl = runner_pnl + pos.realized_partial_pnl
                total_commission = pos.commission_entry + commission + pos.realized_partial_commission
                total_slippage = pos.slippage_entry + slip * pos.quantity + pos.realized_partial_slippage
                total_qty = pos.quantity + pos.realized_partial_qty

                orig_risk = pos.risk_per_share * (pos.qty_original or total_qty)
                r_mult = total_pnl / orig_risk if orig_risk > 0 else 0.0

                equity += runner_pnl - commission

                trades.append(TradeRecord(
                    strategy="ALCB",
                    symbol=sym,
                    direction=BTDirection.LONG if pos.direction == Direction.LONG else BTDirection.SHORT,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=fill,
                    quantity=total_qty,
                    pnl=total_pnl,
                    r_multiple=r_mult,
                    risk_per_share=pos.risk_per_share,
                    commission=total_commission,
                    slippage=total_slippage,
                    entry_type=pos.entry_type.value,
                    exit_reason="END_OF_BACKTEST",
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=pos.hold_bars,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=pos.build_metadata(),
                ))

        logger.info(
            "ALCB Tier 1 complete: %d trades, final equity: $%.2f (%.1f%%)",
            len(trades), equity, (equity / cfg.initial_equity - 1) * 100,
        )

        return ALCBDailyResult(
            trades=trades,
            equity_curve=np.array(equity_history),
            timestamps=np.array([
                np.datetime64(ts.replace(tzinfo=None)) for ts in ts_history
            ]),
            daily_selections=daily_selections,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_portfolio(
        self, positions: dict[str, _OpenPosition], equity: float, settings: StrategySettings,
    ) -> PortfolioState:
        """Build a PortfolioState from current backtest positions."""
        open_pos: dict[str, PositionState] = {}
        for sym, pos in positions.items():
            open_pos[sym] = pos.to_position_state()
        return PortfolioState(
            account_equity=equity,
            base_risk_fraction=settings.base_risk_fraction,
            open_positions=open_pos,
        )

    def _daily_atr(self, symbol: str, trade_date: date, period: int = 14) -> float:
        """Compute daily ATR for a symbol as of trade_date."""
        bars = self._get_alcb_daily_bars(symbol, trade_date)
        if not bars:
            return 0.0
        return atr_from_bars(bars, period)

    def _get_alcb_daily_bars(self, symbol: str, trade_date: date, lookback: int = 30):
        """Get ALCB ResearchDailyBar list for ATR computation."""
        from research.backtests.stock.engine.research_replay import _daily_bars_from_arrays
        didx = self._replay._daily_didx.get(symbol)
        arrs = self._replay._daily_arrs.get(symbol)
        if didx is None or arrs is None:
            return []
        return _daily_bars_from_arrays(arrs, didx[0], didx[1], trade_date, lookback=lookback)
