"""Multi-Asset Swing Breakout v3.3-ETF — async event-driven core engine.

Orchestrates daily campaign management and hourly entry/add execution.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Any, Optional

import numpy as np

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick
from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import (
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderType,
    RiskContext,
)
from libs.oms.risk.calculator import RiskCalculator
from libs.services.trade_recorder import TradeRecorder

from . import allocator, gates, signals, stops
from .config import (
    ADD_RISK_MULT,
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
    STRATEGY_ID,
    SYMBOL_CONFIGS,
    SYMBOLS,
    TP1_R_ALIGNED,
    TP1_R_CAUTION,
    TP1_R_NEUTRAL,
    TP2_R_ALIGNED,
    TP2_R_CAUTION,
    TP2_R_NEUTRAL,
    SymbolConfig,
)
from .indicators import (
    atr,
    compute_avwap,
    compute_regime_4h,
    compute_rvol_d,
    compute_rvol_h,
    compute_wvwap,
    construct_4h_bars,
    compute_daily_slope,
    ema,
    get_slot_key,
    highest,
    lowest,
    past_only_quantile,
    pullback_ref,
    rolling_correlation,
    sma,
    update_slot_medians,
)
from .models import (
    CampaignState,
    ChopMode,
    CircuitBreakerState,
    DailyContext,
    Direction,
    EntryType,
    ExitTier,
    HourlyState,
    PositionState,
    Regime4H,
    RollingHistories,
    SetupInstance,
    SetupState,
    SymbolCampaign,
    TradeRecord,
    TradeRegime,
)

logger = logging.getLogger(__name__)

from zoneinfo import ZoneInfo as _ZoneInfo
_ET = _ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BreakoutEngine:
    """Core v3.3-ETF campaign-based breakout engine."""

    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        config: dict[str, SymbolConfig] | None = None,
        trade_recorder: TradeRecorder | None = None,
        equity: float = 100_000.0,
        news_calendar: list[tuple[str, datetime]] | None = None,
        market_calendar: Any | None = None,
        instrumentation: Any | None = None,
        equity_offset: float = 0.0,
        equity_alloc_pct: float = 1.0,
    ) -> None:
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = instruments
        self._config = config or SYMBOL_CONFIGS
        self._recorder = trade_recorder
        self._equity = equity
        self._equity_offset = equity_offset
        self._equity_alloc_pct = equity_alloc_pct
        self._news_calendar: list[tuple[str, datetime]] = news_calendar or []
        self._market_cal = market_calendar
        self._kit = instrumentation

        # Wire drawdown tracker with initial equity
        if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
            self._kit.ctx.drawdown_tracker.update_equity(self._equity)

        # Per-symbol campaign state
        self.campaigns: dict[str, SymbolCampaign] = {
            sym: SymbolCampaign() for sym in self._config
        }
        self.histories: dict[str, RollingHistories] = {
            sym: RollingHistories() for sym in self._config
        }
        self.hourly_states: dict[str, HourlyState] = {
            sym: HourlyState() for sym in self._config
        }

        # Position tracking
        self.positions: dict[str, PositionState] = {}
        self.active_setups: dict[str, SetupInstance] = {}
        self.circuit_breaker = CircuitBreakerState()

        # Correlation map: {(sym_a, sym_b): corr}
        self.correlation_map: dict[tuple[str, str], float] = {}

        # Order tracking
        self._order_to_setup: dict[str, str] = {}
        self._order_kind: dict[str, str] = {}
        self._order_requested_qty: dict[str, int] = {}
        self._oca_counter: int = 0
        self._risk_halted = False
        self._risk_halt_reason = ""

        # Live market data
        self._tickers: dict[str, Any] = {}
        self.contracts: dict[str, Any] = {}

        # Async tasks
        self._event_task: asyncio.Task | None = None
        self._hourly_task: asyncio.Task | None = None
        self._daily_task: asyncio.Task | None = None
        self._running = False
        self._resubscribing = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize state, subscribe to events, start schedulers."""
        logger.info("Breakout v3.3 engine starting …")
        self._running = True

        # OMS event subscription
        event_queue = self._oms.stream_events(STRATEGY_ID)
        self._event_task = asyncio.create_task(self._process_events(event_queue))

        # Resolve ETF contracts
        cf = getattr(self._ib, "_contract_factory", None)
        for sym, cfg in self._config.items():
            try:
                if cf is not None:
                    contract, _ = await cf.resolve(
                        sym,
                        cfg.contract_expiry,
                        instrument=self._instruments.get(sym),
                    )
                    self.contracts[sym] = contract
                    continue
                from ib_async import Stock
                contract = Stock(sym, cfg.exchange, "USD")
                qualified = await self._ib.ib.qualifyContractsAsync(contract)
                if qualified:
                    self.contracts[sym] = qualified[0]
            except Exception as e:
                logger.warning("Could not resolve contract for %s: %s", sym, e)

        # Subscribe market data
        for sym, contract in self.contracts.items():
            try:
                self._tickers[sym] = self._ib.ib.reqMktData(contract, "", False, False)
            except Exception as e:
                logger.warning("Could not subscribe mkt data for %s: %s", sym, e)

        # Load initial bar history
        await self._load_initial_bars()

        # Start schedulers
        self._hourly_task = asyncio.create_task(self._hourly_scheduler())
        self._daily_task = asyncio.create_task(self._daily_scheduler())

        # Register farm-recovery handler for automatic market data resubscription
        self._ib.register_farm_recovery_callback("default", self._on_farm_recovery)

        logger.info("Breakout v3.3 engine started for %s", list(self._config.keys()))

    async def stop(self) -> None:
        """Shutdown engine gracefully."""
        logger.info("Breakout v3.3 engine stopping …")
        self._running = False

        for task in [self._event_task, self._hourly_task, self._daily_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        for sym in list(self._tickers):
            contract = self.contracts.get(sym)
            if contract:
                try:
                    self._ib.ib.cancelMktData(contract)
                except Exception:
                    pass
        self._tickers.clear()

        logger.info("Breakout v3.3 engine stopped")

    # ------------------------------------------------------------------
    # Farm recovery
    # ------------------------------------------------------------------

    def _on_farm_recovery(self, farm_name: str) -> None:
        """Synchronous callback from FarmMonitor — schedule async resubscription."""
        if not self._running:
            return
        logger.info("Farm %s recovered — scheduling market data resubscription", farm_name)
        asyncio.get_running_loop().call_soon(
            lambda: asyncio.create_task(self._resubscribe_market_data())
        )

    async def _resubscribe_market_data(self) -> None:
        """Cancel and re-request market data for all tracked symbols."""
        if not self._running:
            return
        if self._resubscribing:
            return
        self._resubscribing = True
        try:
            logger.info("Resubscribing market data for %d symbols", len(self._tickers))

            for sym in list(self._tickers):
                contract = self.contracts.get(sym)
                if contract:
                    try:
                        self._ib.ib.cancelMktData(contract)
                    except Exception:
                        pass
            self._tickers.clear()

            await asyncio.sleep(1.0)

            for sym, contract in self.contracts.items():
                try:
                    self._tickers[sym] = self._ib.ib.reqMktData(contract, "", False, False)
                except Exception as e:
                    logger.warning("Resubscribe failed for %s: %s", sym, e)

            logger.info("Market data resubscription complete: %d tickers", len(self._tickers))
        finally:
            self._resubscribing = False

    # ------------------------------------------------------------------
    # Schedulers
    # ------------------------------------------------------------------

    async def _hourly_scheduler(self) -> None:
        """Run hourly cycle at each RTH hour close."""
        while self._running:
            now = datetime.now(timezone.utc)
            # Next hour boundary + 10s buffer for bar finalization
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
            await asyncio.sleep((next_hour - now).total_seconds())
            if not self._running:
                break
            try:
                now_et = datetime.now(timezone.utc).astimezone(_ET)
                if gates.is_rth_time(now_et):
                    await self._hourly_cycle(now_et)
            except Exception:
                logger.exception("Error in hourly cycle")

    async def _daily_scheduler(self) -> None:
        """Run daily close routine at ~16:15 ET."""
        while self._running:
            now = datetime.now(timezone.utc)
            try:
                now_et = now.astimezone(_ET)
            except Exception:
                now_et = now
            # Schedule for 16:15 ET today (or tomorrow if past)
            target = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
            if target <= now_et:
                target += timedelta(days=1)
            while target.weekday() >= 5 or (
                self._market_cal
                and not self._market_cal.is_trading_day(target.date())
            ):
                target += timedelta(days=1)
            wait = (target - now_et).total_seconds()
            await asyncio.sleep(max(1, wait))
            if not self._running:
                break
            try:
                await self._daily_cycle()
            except Exception:
                logger.exception("Error in daily cycle")

    # ------------------------------------------------------------------
    # Initial bar loading
    # ------------------------------------------------------------------

    async def _load_initial_bars(self) -> None:
        """Fetch initial bar history and initialize indicators."""
        for sym in self._config:
            try:
                contract = self.contracts.get(sym)
                if not contract:
                    continue

                # Daily bars (~3y)
                daily_bars = await self._ib.ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr="3 Y",
                    barSizeSetting="1 day", whatToShow="TRADES",
                    useRTH=True, formatDate=1,
                )
                if daily_bars:
                    self._init_daily_indicators(sym, daily_bars)

                # Hourly bars (~2yr)
                hourly_bars = await self._ib.ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr="2 Y",
                    barSizeSetting="1 hour", whatToShow="TRADES",
                    useRTH=True, formatDate=1,
                )
                if hourly_bars:
                    self._init_hourly_indicators(sym, hourly_bars)

            except Exception:
                logger.exception("Error loading initial bars for %s", sym)

    def _init_daily_indicators(self, sym: str, bars: list[Any]) -> None:
        """Initialize daily rolling histories from historical bars."""
        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        volumes = np.array([b.volume for b in bars], dtype=float)

        hist = self.histories[sym]
        atr14 = atr(highs, lows, closes, ATR_DAILY_PERIOD)
        atr50 = atr(highs, lows, closes, ATR_DAILY_LONG_PERIOD)

        # Build squeeze and disp histories
        for i in range(max(ATR_DAILY_LONG_PERIOD, 50), len(closes)):
            L = 12  # use default for history init
            rh = float(np.max(highs[max(0, i - L + 1):i + 1]))
            rl = float(np.min(lows[max(0, i - L + 1):i + 1]))
            bh = rh - rl
            sq = bh / float(atr50[i]) if atr50[i] > 0 else 1.0
            hist.add_squeeze(sq)

            # Displacement requires AVWAP — use SMA as proxy for initialization
            sma_val = float(np.mean(closes[max(0, i - 20):i + 1]))
            disp = abs(float(closes[i]) - sma_val) / float(atr14[i]) if atr14[i] > 0 else 0.0
            hist.add_disp(disp)
            hist.add_atr(float(atr14[i]))

    def _init_hourly_indicators(self, sym: str, bars: list[Any]) -> None:
        """Initialize hourly state and slot medians from historical bars."""
        hs = self.hourly_states[sym]
        if not bars:
            return

        closes = [float(b.close) for b in bars]
        highs_l = [float(b.high) for b in bars]
        lows_l = [float(b.low) for b in bars]
        hs.closes = closes[-200:]
        hs.highs = highs_l[-200:]
        hs.lows = lows_l[-200:]
        hs.close = closes[-1]

        # Build slot volume medians
        volumes_by_slot: dict[tuple[int, int], list[float]] = {}
        for b in bars:
            dt = b.date if isinstance(b.date, datetime) else datetime.fromisoformat(str(b.date))
            key = get_slot_key(dt)
            volumes_by_slot.setdefault(key, []).append(float(b.volume))

        self.histories[sym].slot_medians = update_slot_medians(volumes_by_slot)

    # ------------------------------------------------------------------
    # Daily close routine (spec §5 daily campaign engine)
    # ------------------------------------------------------------------

    async def _daily_cycle(self) -> None:
        """Execute daily close routine for all symbols."""
        logger.info("=== Daily close cycle ===")
        await self._refresh_equity()

        for sym in self._config:
            try:
                await self._on_daily_close(sym)
            except Exception:
                logger.exception("Error in daily close for %s", sym)

        # Update correlations
        self._update_correlations()

    async def _on_daily_close(self, symbol: str) -> None:
        """Per-symbol daily close: update campaign, check breakout."""
        contract = self.contracts.get(symbol)
        if not contract:
            return

        cfg = self._config[symbol]
        campaign = self.campaigns[symbol]
        hist = self.histories[symbol]

        # Fetch fresh daily bars
        bars = await self._ib.ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr="200 D",
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if not bars or len(bars) < ATR_DAILY_LONG_PERIOD + 10:
            return

        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        volumes = np.array([b.volume for b in bars], dtype=float)
        bar_dates = [b.date for b in bars]
        bar_times = [
            b.date if isinstance(b.date, datetime)
            else datetime.fromisoformat(str(b.date))
            for b in bars
        ]

        if self._kit and len(closes) > 0:
            self._kit.record_close(symbol, float(closes[-1]))

        # --- 1) Compute daily indicators ---
        atr14_d_arr = atr(highs, lows, closes, ATR_DAILY_PERIOD)
        atr50_d_arr = atr(highs, lows, closes, ATR_DAILY_LONG_PERIOD)
        atr14_d = float(atr14_d_arr[-1])
        atr50_d = float(atr50_d_arr[-1])
        sma_atr = float(np.mean(atr14_d_arr[-50:])) if len(atr14_d_arr) >= 50 else atr14_d
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

        # --- Chop mode (spec §7.2) ---
        ema20_d = ema(closes, 20)
        chop_score = signals.compute_chop_score(atr14_d, hist.atr_hist, closes, ema20_d)
        chop_mode = signals.classify_chop_mode(chop_score)
        campaign.chop_mode = chop_mode.value

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
                campaign.anchor_time = gates.next_rth_open(bar_times[-1])
                campaign.box_age_days = 0
                campaign.reentry_count = {"LONG": {0: 0}, "SHORT": {0: 0}}
                campaign.add_count = 0
                campaign.campaign_risk_used = 0.0
                campaign.pending = None
                logger.info("%s: Campaign %d activated, box [%.2f, %.2f]",
                            symbol, campaign.campaign_id, campaign.box_low, campaign.box_high)
            return

        campaign.box_age_days += 1

        # --- 3) AVWAP_D ---
        avwap_d_arr = compute_avwap(highs, lows, closes, volumes, bar_times, campaign.anchor_time)
        avwap_d = float(avwap_d_arr[-1]) if not np.isnan(avwap_d_arr[-1]) else float(closes[-1])

        # --- 4) Breakout qualification ---
        close_today = float(closes[-1])
        direction = signals.check_structural_breakout(close_today, campaign.box_high, campaign.box_low)

        # 4H regime + hard block
        hourly_bars_raw = await self._fetch_hourly_bars(symbol)
        if hourly_bars_raw and len(hourly_bars_raw) > 50:
            h_closes = np.array([b.close for b in hourly_bars_raw], dtype=float)
            h_highs = np.array([b.high for b in hourly_bars_raw], dtype=float)
            h_lows = np.array([b.low for b in hourly_bars_raw], dtype=float)
            h_vols = np.array([b.volume for b in hourly_bars_raw], dtype=float)
            h4_h, h4_l, h4_c, h4_v, h4_t = construct_4h_bars(h_highs, h_lows, h_closes, h_vols,
                [b.date if isinstance(b.date, datetime) else datetime.fromisoformat(str(b.date))
                 for b in hourly_bars_raw])
            regime_4h, slope_4h, adx_4h = compute_regime_4h(h4_c, h4_h, h4_l) if len(h4_c) > 0 else (Regime4H.RANGE_CHOP, 0.0, 0.0)
        else:
            regime_4h, slope_4h, adx_4h = Regime4H.RANGE_CHOP, 0.0, 0.0

        daily_slope = compute_daily_slope(closes)

        # Emit indicator snapshot at breakout evaluation
        if self._kit and direction is not None:
            self._kit.on_indicator_snapshot(
                pair=symbol,
                indicators={
                    "atr14_d": atr14_d,
                    "atr50_d": atr50_d,
                    "squeeze_metric": squeeze_metric,
                    "containment": containment,
                    "chop_score": chop_score,
                    "box_high": campaign.box_high,
                    "box_low": campaign.box_low,
                    "box_height": box_height,
                    "close": close_today,
                    "adx_4h": adx_4h,
                    "regime_4h": regime_4h.value if hasattr(regime_4h, "value") else str(regime_4h),
                },
                signal_name="breakout_v3",
                signal_strength=0.0,
                decision="enter",
                strategy_id="SWING_BREAKOUT_V3",
            )
            _snap = self._kit.capture_snapshot(symbol)
            if _snap:
                self._kit.on_orderbook_context(
                    pair=symbol,
                    best_bid=_snap.get("bid", 0),
                    best_ask=_snap.get("ask", 0),
                    trade_context="signal_eval",
                )

        if direction and gates.hard_block(direction, regime_4h, daily_slope, atr14_d):
            logger.info("%s: hard block %s", symbol, direction)
            if self._kit:
                self._kit.on_filter_decision(
                    pair=symbol, filter_name="hard_block",
                    passed=False, threshold=0.0, actual_value=0.0,
                    signal_name="breakout_v3", strategy_id="SWING_BREAKOUT_V3",
                )
                self._kit.log_missed(
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    signal="breakout",
                    signal_id=f"{symbol}_hard_block_{bar_dates[-1]}",
                    signal_strength=0.0,
                    blocked_by="hard_block",
                    block_reason=f"4H regime {regime_4h.value} blocks {direction.name}",
                )
            direction = None

        # --- Chop HALT blocks new breakout qualification ---
        if chop_mode == ChopMode.HALT and direction is not None:
            logger.info("%s: CHOP HALT blocks %s breakout", symbol, direction.name)
            if self._kit:
                self._kit.on_filter_decision(
                    pair=symbol, filter_name="chop_halt",
                    passed=False, threshold=0.0, actual_value=chop_score,
                    signal_name="breakout_v3", strategy_id="SWING_BREAKOUT_V3",
                )
                self._kit.log_missed(
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    signal="breakout",
                    signal_id=f"{symbol}_chop_halt_{bar_dates[-1]}",
                    signal_strength=0.0,
                    blocked_by="chop_halt",
                    block_reason=f"Chop score too high, HALT mode active",
                )
            direction = None

        # Directional filter: block directions that are negative-EV for this symbol
        if direction is not None:
            dir_str = "LONG" if direction == Direction.LONG else "SHORT"
            if dir_str not in cfg.allowed_directions:
                if self._kit:
                    self._kit.on_filter_decision(
                        pair=symbol, filter_name="direction_filter",
                        passed=False, threshold=0.0, actual_value=0.0,
                        signal_name="breakout_v3", strategy_id="SWING_BREAKOUT_V3",
                    )
                    self._kit.log_missed(
                        pair=symbol,
                        side=dir_str,
                        signal="breakout",
                        signal_id=f"{symbol}_direction_filter_{bar_dates[-1]}",
                        signal_strength=0.0,
                        blocked_by="direction_filter",
                        block_reason=f"{dir_str} not in allowed_directions {cfg.allowed_directions}",
                    )
                direction = None

        # --- Inside close invalidation (spec §10) ---
        if (campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN)
                and campaign.bars_since_breakout > 0):
            inside = campaign.box_low <= close_today <= campaign.box_high
            if inside:
                campaign.inside_close_count += 1
                if campaign.inside_close_count >= INSIDE_CLOSE_INVALIDATION_COUNT:
                    campaign.state = CampaignState.INVALIDATED
                    logger.info("%s: Invalidated — %d consecutive inside closes",
                                symbol, campaign.inside_close_count)
                    return
            else:
                campaign.inside_close_count = 0

        # --- 5) DIRTY handling ---
        if campaign.state == CampaignState.DIRTY:
            # Compute dirty duration as calendar days since DIRTY was triggered
            current_date = bar_dates[-1] if isinstance(bar_dates[-1], date) else bar_dates[-1].date() if hasattr(bar_dates[-1], 'date') else None
            if campaign.dirty_start_date and current_date:
                dirty_start = campaign.dirty_start_date if isinstance(campaign.dirty_start_date, date) else campaign.dirty_start_date
                dirty_duration = (current_date - dirty_start).days
            else:
                dirty_duration = 0
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
                logger.info("%s: DIRTY reset (box_shifted=%s)", symbol, box_shifted)
                return

            # Opposite direction while DIRTY
            if direction and direction != campaign.breakout_direction:
                disp_pass, disp, disp_th = signals.check_displacement(
                    close_today, avwap_d, atr14_d, hist.disp_hist[:-1], atr_expanding
                )
                if disp_pass and signals.dirty_opposite_allowed(disp, disp_th):
                    pass  # Allow opposite — continue to qualification
                else:
                    return
            elif direction and direction == campaign.breakout_direction:
                return  # DIRTY blocks same-direction
            else:
                return

        # Handle breakout failure → DIRTY trigger
        if (campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN)
                and campaign.breakout_direction
                and campaign.bars_since_breakout > 0):
            if signals.check_dirty_trigger(
                close_today, campaign.box_high, campaign.box_low,
                campaign.breakout_direction, campaign.bars_since_breakout, cfg.m_break,
            ):
                campaign.state = CampaignState.DIRTY
                campaign.dirty_start_date = bar_dates[-1] if isinstance(bar_dates[-1], date) else None
                campaign.dirty_high = range_high_roll
                campaign.dirty_low = range_low_roll
                campaign.L_at_dirty = campaign.L
                logger.info("%s: DIRTY triggered", symbol)
                return

        if direction is None:
            # Increment bars since breakout if in breakout state
            if campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN,
                                  CampaignState.CONTINUATION):
                campaign.bars_since_breakout += 1
                # Check expiry
                if campaign.bars_since_breakout > campaign.hard_expiry_bars > 0:
                    campaign.state = CampaignState.EXPIRED
                    logger.info("%s: Campaign expired", symbol)
            return

        # --- 6) Displacement pass ---
        disp_pass, disp, disp_th = signals.check_displacement(
            close_today, avwap_d, atr14_d, hist.disp_hist[:-1], atr_expanding
        )
        hist.add_disp(disp)
        if not disp_pass:
            if self._kit and direction is not None:
                self._kit.log_missed(
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    signal="breakout",
                    signal_id=f"{symbol}_displacement_{bar_dates[-1]}",
                    signal_strength=round(disp / disp_th, 2) if disp_th > 0 else 0.0,
                    blocked_by="displacement",
                    block_reason=f"displacement {disp:.3f} < threshold {disp_th:.3f}",
                )
            return

        # --- 7) Breakout quality reject ---
        open_today = float(bars[-1].open) if hasattr(bars[-1], 'open') else close_today
        if signals.check_breakout_quality_reject(
            float(highs[-1]), float(lows[-1]), open_today, close_today, atr14_d, direction
        ):
            logger.info("%s: Breakout quality rejected", symbol)
            if self._kit and direction is not None:
                self._kit.log_missed(
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    signal="breakout",
                    signal_id=f"{symbol}_breakout_quality_{bar_dates[-1]}",
                    signal_strength=0.0,
                    blocked_by="breakout_quality",
                    block_reason="Breakout bar quality check failed",
                )
            return

        # --- 8) Evidence score (detailed) ---
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
        if score_total < score_threshold:
            logger.info("%s: Score %d below threshold %d",
                        symbol, score_total, score_threshold)
            if self._kit and direction is not None:
                self._kit.log_missed(
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    signal="breakout",
                    signal_id=f"{symbol}_score_threshold_{bar_dates[-1]}",
                    signal_strength=round(score_total / score_threshold, 2) if score_threshold > 0 else 0.0,
                    blocked_by="score_threshold",
                    block_reason=f"score {score_total} < threshold {score_threshold}",
                )
            if campaign.state in (CampaignState.BREAKOUT, CampaignState.POSITION_OPEN,
                                  CampaignState.CONTINUATION):
                campaign.bars_since_breakout += 1
                campaign.expiry_mult = signals.compute_expiry_mult(
                    campaign.bars_since_breakout, campaign.expiry_bars
                )
                if campaign.bars_since_breakout > campaign.hard_expiry_bars > 0:
                    campaign.state = CampaignState.EXPIRED
            return

        # --- 8c) Regime chop block ---
        if (REGIME_CHOP_BLOCK
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
            return

        # --- 9) Expiry ---
        expiry_bars, hard_expiry_bars = signals.compute_expiry_bars(atr14_d, hist.atr_hist)
        campaign.expiry_bars = expiry_bars
        campaign.hard_expiry_bars = hard_expiry_bars

        # --- 10) Continuation mode ---
        disp_mult = allocator.compute_disp_mult(disp, hist.disp_hist)
        campaign.continuation = signals.check_continuation_mode(
            close_today, campaign.box_high, campaign.box_low, campaign.box_height,
            atr14_d, direction, campaign.bars_since_breakout, regime_4h, disp_mult,
        )

        # --- 11) Update breakout state ---
        if campaign.state in (CampaignState.COMPRESSION, CampaignState.BREAKOUT):
            campaign.state = CampaignState.BREAKOUT
            campaign.breakout_direction = direction
            campaign.breakout_date = bar_dates[-1] if isinstance(bar_dates[-1], date) else None
            campaign.bars_since_breakout = 0
            campaign.expiry_mult = 1.0
            campaign.inside_close_count = 0
            logger.info("%s: Breakout %s qualified (Disp=%.2f, score=%d)",
                        symbol, direction.name, disp, score_total)
        else:
            campaign.bars_since_breakout += 1
            campaign.expiry_mult = signals.compute_expiry_mult(
                campaign.bars_since_breakout, campaign.expiry_bars
            )

        if (campaign.continuation
                and campaign.state == CampaignState.BREAKOUT
                and campaign.bars_since_breakout > 0):
            campaign.state = CampaignState.CONTINUATION
            logger.info("%s: Entered continuation mode", symbol)

        # --- 12) Cache daily context for hourly engine ---
        trade_regime = signals.determine_trade_regime(direction, regime_4h)
        quality_mult = allocator.compute_quality_mult(
            direction, regime_4h, disp, hist.disp_hist, sq_good, sq_loose,
            symbol, self.positions, self.correlation_map, score_total,
            regime_at_entry=regime_4h.value,
        )

        # Cache final_risk_dollars at daily level to prevent intra-day sizing drift (3F)
        expiry_mult_d = campaign.expiry_mult
        cb_ok_d, _, cb_mult_d = gates.circuit_breaker_check(self.circuit_breaker)
        if campaign.chop_mode == "DEGRADED":
            cb_mult_d *= CHOP_DEGRADED_SIZE_MULT
        final_risk_dollars_d = allocator.compute_final_risk(
            self._equity, cfg.base_risk_pct, risk_regime,
            quality_mult, expiry_mult_d, cb_mult_d,
        ) if cb_ok_d else 0.0

        campaign.daily_context = {
            "direction": direction,
            "disp": disp,
            "disp_th": disp_th,
            "disp_mult": disp_mult,
            "displacement_pass": disp_pass,
            "rvol_d": rvol_d,
            "vol_score": vol_score,
            "score_vol": score_detail.get("vol", 0),
            "score_squeeze": score_detail.get("squeeze", 0),
            "score_regime": score_detail.get("regime", 0),
            "score_consec": score_detail.get("consec", 0),
            "score_atr": score_detail.get("atr", 0),
            "sq_good": sq_good,
            "sq_loose": sq_loose,
            "atr_expanding": atr_expanding,
            "atr14_d": atr14_d,
            "risk_regime": risk_regime,
            "regime_4h": regime_4h,
            "quality_mult": quality_mult,
            "score_total": score_total,
            "trade_regime": trade_regime,
            "bar_date": bar_dates[-1],
            "chop_mode": chop_mode.value,
            "score_threshold": score_threshold,
            "final_risk_dollars": final_risk_dollars_d,
            "cb_mult": cb_mult_d,
            "expiry_mult": expiry_mult_d,
        }

    # ------------------------------------------------------------------
    # Hourly close routine (spec §12 entries, §13 adds, §19 pending)
    # ------------------------------------------------------------------

    async def _hourly_cycle(self, now_et: datetime) -> None:
        """Execute hourly cycle for all symbols."""
        logger.info("=== Hourly cycle %s ===", now_et.isoformat())

        for sym in self._config:
            try:
                await self._on_hourly_close(sym, now_et)
            except Exception:
                logger.exception("Error in hourly close for %s", sym)

        # Manage active setups (exits, trailing, stale)
        await self._manage_exits(now_et)

        # Hook 1: Market snapshot + regime classification (post-decision)
        if self._kit:
            for sym in self._config:
                self._kit.capture_snapshot(sym)
                self._kit.classify_regime(sym)

    async def _on_hourly_close(self, symbol: str, now_et: datetime) -> None:
        """Per-symbol hourly logic: entries, adds, pending re-check."""
        campaign = self.campaigns[symbol]
        cfg = self._config[symbol]

        # Skip if campaign not ready
        if campaign.state in (CampaignState.INACTIVE, CampaignState.INVALIDATED,
                              CampaignState.EXPIRED):
            return
        if campaign.daily_context is None:
            return

        ctx = campaign.daily_context
        direction: Direction = ctx["direction"]

        # Chop HALT blocks all new entries
        if campaign.chop_mode == "HALT":
            return

        # Fetch latest hourly bar
        hourly_bars_raw = await self._fetch_hourly_bars(symbol, duration="5 D")
        if not hourly_bars_raw:
            return

        h_closes = np.array([b.close for b in hourly_bars_raw], dtype=float)
        h_highs = np.array([b.high for b in hourly_bars_raw], dtype=float)
        h_lows = np.array([b.low for b in hourly_bars_raw], dtype=float)
        h_volumes = np.array([b.volume for b in hourly_bars_raw], dtype=float)
        h_times = [
            b.date if isinstance(b.date, datetime)
            else datetime.fromisoformat(str(b.date))
            for b in hourly_bars_raw
        ]

        # Update hourly state
        hs = self.hourly_states[symbol]
        hs.closes = [float(c) for c in h_closes[-200:]]
        hs.highs = [float(h) for h in h_highs[-200:]]
        hs.lows = [float(l) for l in h_lows[-200:]]
        hs.close = float(h_closes[-1])
        hs.bar_time = h_times[-1] if h_times else None

        # Compute hourly indicators
        atr14_h_arr = atr(h_highs, h_lows, h_closes, ATR_HOURLY_PERIOD)
        atr14_h = float(atr14_h_arr[-1])
        hs.atr14_h = atr14_h
        atr14_d = ctx["atr14_d"]

        # AVWAP_H
        avwap_h_arr = compute_avwap(h_highs, h_lows, h_closes, h_volumes,
                                     h_times, campaign.anchor_time)
        avwap_h = float(avwap_h_arr[-1]) if not np.isnan(avwap_h_arr[-1]) else float(h_closes[-1])
        hs.avwap_h = avwap_h

        # WVWAP
        wvwap_arr = compute_wvwap(h_highs, h_lows, h_closes, h_volumes, h_times)
        wvwap = float(wvwap_arr[-1])
        hs.wvwap = wvwap

        # EMA20_H
        ema20_h = float(ema(h_closes, EMA_1H_PERIOD)[-1])
        hs.ema20_h = ema20_h

        # RVOL_H (slot-normalized)
        slot_key = get_slot_key(now_et)
        rvol_h = compute_rvol_h(float(h_volumes[-1]), slot_key,
                                 self.histories[symbol].slot_medians)
        hs.rvol_h = rvol_h

        # Quality and risk
        disp_mult = ctx["disp_mult"]
        quality_mult = ctx["quality_mult"]
        regime_4h: Regime4H = ctx["regime_4h"]
        trade_regime: TradeRegime = ctx["trade_regime"]

        # Use daily-cached risk to prevent intra-day sizing drift (3F)
        final_risk_dollars = ctx.get("final_risk_dollars", 0.0)
        cb_mult = ctx.get("cb_mult", 1.0)
        expiry_mult = ctx.get("expiry_mult", 1.0)
        if final_risk_dollars <= 0:
            return

        # --- Pending mechanism ---
        if campaign.pending:
            campaign.pending.rth_hours_elapsed += 1
            if gates.pending_expired(campaign.pending, now_et):
                campaign.pending = None
            elif gates.pending_block_cleared(
                campaign.pending.reason, self.positions,
                campaign.pending.direction, final_risk_dollars, self._equity
            ):
                # Re-validate and place
                await self._place_entry(
                    symbol, campaign.pending.entry_type_requested.value,
                    direction, campaign, cfg, avwap_h, atr14_h, atr14_d,
                    final_risk_dollars, quality_mult, now_et,
                )
                campaign.pending = None
            return

        # --- Check if we already have a position (adds) ---
        have_position = symbol in self.positions and self.positions[symbol].qty > 0
        if have_position:
            await self._handle_adds(
                symbol, direction, rvol_h, avwap_h, wvwap, ema20_h,
                atr14_d, atr14_h, campaign, cfg, quality_mult,
                final_risk_dollars, now_et,
            )
            return

        # --- Re-entry gate (spec §21) ---
        if campaign.last_exit_date and campaign.last_exit_direction:
            current_date = now_et.date() if hasattr(now_et, 'date') else now_et
            days_since_exit = (current_date - campaign.last_exit_date).days
            if not gates.reentry_allowed(
                direction, campaign, campaign.last_exit_realized_r, days_since_exit,
            ):
                return
            # If re-entry is allowed, track it
            dir_key = "LONG" if direction == Direction.LONG else "SHORT"
            campaign.reentry_count.setdefault(dir_key, {}).setdefault(campaign.box_version, 0)

        # Check if Entry A is outstanding (spec §12.5 — C only after A resolves)
        entry_a_active = any(
            s.symbol == symbol
            and s.entry_type == EntryType.A_AVWAP_RETEST
            and s.state in (SetupState.ARMED, SetupState.TRIGGERED)
            for s in self.active_setups.values()
        )

        # --- Entry selection A → B → C ---
        entry_type = signals.select_entry_type(
            direction=direction,
            campaign=campaign,
            hourly_close=float(h_closes[-1]),
            hourly_low=float(h_lows[-1]),
            hourly_high=float(h_highs[-1]),
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
            entry_a_active=entry_a_active,
            atr_expanding=ctx.get("atr_expanding", True),
            ema20_h=ema20_h,
        )
        if entry_type is None:
            return

        # C entry displacement ceiling (3E): reject if price too far from AVWAP
        if entry_type in (EntryType.C_STANDARD, EntryType.C_CONTINUATION) and atr14_d > 0:
            from .config import C_ENTRY_MAX_DISP_H
            disp_from_avwap = abs(hs.close - avwap_h) / atr14_d
            if disp_from_avwap > C_ENTRY_MAX_DISP_H:
                return

        # Full eligibility check
        ok, reason, _ = gates.full_eligibility_check(
            symbol, direction, campaign, cfg, now_et,
            self.circuit_breaker, self.positions,
            final_risk_dollars, self._equity, self._news_calendar,
            self.correlation_map,
        )
        if not ok:
            # If blocked by transient condition, create pending
            block_reason = gates.check_transient_blocks(
                symbol, direction, self.positions, final_risk_dollars,
                self._equity, self.correlation_map,
            )
            if block_reason:
                from .models import PendingEntry
                campaign.pending = PendingEntry(
                    created_ts=now_et,
                    direction=direction,
                    entry_type_requested=entry_type,
                    reason=block_reason,
                )
                logger.info("%s: Pending created (%s) — %s", symbol, entry_type.value, block_reason)
            return

        # Micro guard (spec §14.4)
        base_risk_adj = allocator.compute_risk_regime_adj(cfg.base_risk_pct, ctx["risk_regime"])
        final_risk_pct = final_risk_dollars / self._equity if self._equity > 0 else 0.0
        if not allocator.micro_guard_ok(expiry_mult, disp_mult, trade_regime, final_risk_pct, base_risk_adj):
            logger.info("%s: Micro guard blocked", symbol)
            if self._kit and direction is not None:
                self._kit.log_missed(
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    signal="breakout",
                    signal_id=f"{symbol}_micro_guard_{datetime.now(timezone.utc).date()}",
                    signal_strength=0.0,
                    blocked_by="micro_guard",
                    block_reason="Micro guard risk check failed",
                )
            return

        # Friction gate
        entry_price = avwap_h  # approximate
        stop_price = stops.compute_initial_stop(
            direction, entry_type.value,
            campaign.box_high, campaign.box_low, campaign.box_mid,
            atr14_d, cfg.atr_stop_mult, ctx["sq_good"], cfg.tick_size,
        )
        shares = allocator.compute_shares(final_risk_dollars, entry_price, stop_price, cfg.fee_bps_est)
        if shares <= 0:
            return

        if not gates.friction_gate(symbol, cfg.fee_bps_est, entry_price, shares, final_risk_dollars):
            logger.info("%s: Friction gate blocked", symbol)
            return

        # Place the entry
        await self._place_entry(
            symbol, entry_type.value, direction, campaign, cfg,
            avwap_h, atr14_h, atr14_d, final_risk_dollars, quality_mult, now_et,
        )

    # ------------------------------------------------------------------
    # Entry placement
    # ------------------------------------------------------------------

    async def _place_entry(
        self,
        symbol: str,
        entry_type: str,
        direction: Direction,
        campaign: SymbolCampaign,
        cfg: SymbolConfig,
        avwap_h: float,
        atr14_h: float,
        atr14_d: float,
        final_risk_dollars: float,
        quality_mult: float,
        now: datetime,
    ) -> None:
        """Compute stop, size, and submit entry orders via OMS."""
        if self._risk_halted:
            logger.warning(
                "%s: entry suppressed while OMS risk halt is active: %s",
                symbol,
                self._risk_halt_reason or "unspecified",
            )
            return
        ctx = campaign.daily_context or {}
        sq_good = ctx.get("sq_good", False)
        trade_regime = ctx.get("trade_regime", TradeRegime.NEUTRAL)

        # Stop
        stop_price = stops.compute_initial_stop(
            direction, entry_type,
            campaign.box_high, campaign.box_low, campaign.box_mid,
            atr14_d, cfg.atr_stop_mult, sq_good, cfg.tick_size,
        )

        # Entry level
        if entry_type == "A":
            # Limit at AVWAP_H ± buffer
            buffer = 0.03 * atr14_d
            entry_price = avwap_h + buffer if direction == Direction.LONG else avwap_h - buffer
        elif entry_type in ("C_standard", "C_continuation"):
            # C entries: signal confirmed via 2 consecutive closes — enter near market
            hs = self.hourly_states.get(symbol)
            entry_price = hs.close if hs else avwap_h
        else:
            # Market / marketable limit near current
            entry_price = avwap_h

        entry_price = round_to_tick(entry_price, cfg.tick_size,
                                     "up" if direction == Direction.LONG else "down")

        # Shares
        shares = allocator.compute_shares(final_risk_dollars, entry_price, stop_price, cfg.fee_bps_est)
        if shares <= 0:
            return
        shares = min(shares, cfg.max_shares)

        # R per share
        r_per_share = abs(entry_price - stop_price)

        # TP levels
        tp1_r, tp2_r = self._get_tp_r_multiples(trade_regime, cfg.tp_scale)
        tp1, tp2 = stops.compute_tp_levels(direction, entry_price, r_per_share,
                                            tp1_r, tp2_r, cfg.tick_size)

        # OCA
        self._oca_counter += 1
        oca_group = f"BRK_{symbol}_{self._oca_counter}"

        # Build setup instance
        setup = SetupInstance(
            symbol=symbol,
            direction=direction,
            entry_type=EntryType(entry_type),
            state=SetupState.ARMED,
            created_ts=now,
            armed_ts=now,
            campaign_id=campaign.campaign_id,
            box_version=campaign.box_version,
            entry_price=entry_price,
            stop0=stop_price,
            buffer=0.03 * atr14_d,
            final_risk_dollars=final_risk_dollars,
            quality_mult=quality_mult,
            expiry_mult=campaign.expiry_mult,
            shares_planned=shares,
            oca_group=oca_group,
            exit_tier=ExitTier(trade_regime.value),
            regime_at_entry=ctx.get("regime_4h", Regime4H.RANGE_CHOP).value,
            r_price=r_per_share,
            current_stop=stop_price,
        )

        # Filter decision telemetry
        total_risk = sum(p.total_risk_dollars for p in self.positions.values() if p.qty > 0)
        heat_actual = total_risk / self._equity if self._equity > 0 else 0
        setup.filter_decisions = [
            {"filter_name": "score_threshold", "threshold": ctx.get("score_threshold", 2),
             "actual_value": ctx.get("score_total", 0), "passed": True,
             "margin_pct": round((ctx.get("score_total", 0) - ctx.get("score_threshold", 2)) / max(ctx.get("score_threshold", 2), 0.01) * 100, 1)},
            {"filter_name": "displacement", "threshold": ctx.get("disp_th", 0),
             "actual_value": ctx.get("disp", 0), "passed": ctx.get("displacement_pass", True),
             "margin_pct": round((ctx.get("disp", 0) - ctx.get("disp_th", 0)) / max(ctx.get("disp_th", 0), 0.01) * 100, 1) if ctx.get("disp_th", 0) > 0 else 0},
        ]

        # Submit orders
        inst = self._instruments.get(symbol)
        if inst is None:
            return

        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        risk_ctx = RiskContext(
            stop_for_risk=stop_price,
            planned_entry_price=entry_price,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                entry_price, stop_price, shares, inst.point_value if hasattr(inst, 'point_value') else 1.0,
            ),
        )

        # Entry A: market order with 2h TTL; B: market/MOC; C: limit
        if entry_type == "A":
            order_type = OrderType.MARKET
            ttl_secs = 2 * 3600
        elif entry_type == "B":
            order_type = OrderType.MARKET
            ttl_secs = 60  # immediate
        else:
            order_type = OrderType.LIMIT
            ttl_secs = ENTRY_C_TTL_RTH_HOURS * 3600

        entry_order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=shares,
            order_type=order_type,
            limit_price=entry_price if order_type == OrderType.LIMIT else None,
            tif="GTC",
            role=OrderRole.ENTRY,
            entry_policy=EntryPolicy(ttl_seconds=ttl_secs),
            risk_context=risk_ctx,
            oca_group=oca_group,
            oca_type=1,
        )

        receipt = await self._oms.submit_intent(
            Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=entry_order)
        )
        if receipt.oms_order_id:
            setup.primary_order_id = receipt.oms_order_id
            self._track_order(
                receipt.oms_order_id,
                setup.setup_id,
                "primary_entry",
                setup.shares_planned,
            )

            if self._kit:
                self._kit.on_order_event(
                    order_id=receipt.oms_order_id,
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    order_type=order_type.value if hasattr(order_type, 'value') else str(order_type),
                    status="SUBMITTED",
                    requested_qty=float(shares),
                    requested_price=entry_price,
                    strategy_id=STRATEGY_ID,
                )

        self.active_setups[setup.setup_id] = setup
        logger.info("%s: Entry %s %s placed — shares=%d, stop=%.2f, tp1=%.2f",
                    symbol, entry_type, direction.name, shares, stop_price, tp1)

    # ------------------------------------------------------------------
    # Adds (spec §13)
    # ------------------------------------------------------------------

    async def _handle_adds(
        self,
        symbol: str,
        direction: Direction,
        rvol_h: float,
        avwap_h: float,
        wvwap: float,
        ema20_h: float,
        atr14_d: float,
        atr14_h: float,
        campaign: SymbolCampaign,
        cfg: SymbolConfig,
        quality_mult: float,
        final_risk_dollars: float,
        now: datetime,
    ) -> None:
        """Handle pullback add entries."""
        if self._risk_halted:
            logger.warning(
                "%s: add suppressed while OMS risk halt is active: %s",
                symbol,
                self._risk_halt_reason or "unspecified",
            )
            return
        pos = self.positions.get(symbol)
        if not pos or pos.qty <= 0:
            return

        # Eligibility
        if campaign.add_count >= MAX_ADDS_PER_CAMPAIGN:
            return
        if not (pos.tp1_done or pos.runner_active):
            return
        if not signals.is_regime_aligned(direction, campaign.daily_context.get("regime_4h", Regime4H.RANGE_CHOP)):
            return

        initial_risk = pos.total_risk_dollars if pos.total_risk_dollars > 0 else final_risk_dollars
        add_risk = allocator.compute_add_risk_dollars(initial_risk, rvol_h)
        if not allocator.add_budget_ok(campaign, initial_risk, add_risk):
            return

        # Pullback ref
        ref = pullback_ref(float(self.hourly_states[symbol].close), wvwap, avwap_h, ema20_h, atr14_d)

        # Touch + resume
        hs = self.hourly_states[symbol]
        if len(hs.closes) < 2:
            return
        reclaim_buffer = max(0.12 * atr14_h, 0.03 * atr14_d)
        triggered = signals.check_add_trigger(
            direction, float(hs.closes[-1]), float(hs.lows[-1]), float(hs.highs[-1]),
            ref, float(hs.closes[-2]), rvol_h,
            high_vol_regime=campaign.daily_context.get("risk_regime", 1.0) > 1.2,
            reclaim_buffer=reclaim_buffer,
        )
        if not triggered:
            return

        # CLV acceptance for low-volume adds
        if rvol_h < 0.8:
            if not signals.strong_close_location(
                direction, float(hs.highs[-1]), float(hs.lows[-1]), float(hs.closes[-1])
            ):
                return

        # Add stop
        add_stop = stops.compute_add_stop(
            direction, min(hs.lows[-3:]) if len(hs.lows) >= 3 else hs.lows[-1],
            max(hs.highs[-3:]) if len(hs.highs) >= 3 else hs.highs[-1],
            ref, atr14_d, cfg.atr_stop_mult, cfg.tick_size,
        )

        # Size
        add_shares = allocator.compute_shares(add_risk, ref, add_stop, cfg.fee_bps_est)
        if add_shares <= 0:
            return

        # Friction gate
        if not gates.friction_gate(symbol, cfg.fee_bps_est, ref, add_shares, add_risk):
            return

        # Submit add order
        inst = self._instruments.get(symbol)
        if not inst:
            return

        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        add_order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=add_shares,
            order_type=OrderType.LIMIT,
            limit_price=round_to_tick(ref, cfg.tick_size,
                                       "up" if direction == Direction.LONG else "down"),
            tif="GTC",
            role=OrderRole.ENTRY,
            entry_policy=EntryPolicy(ttl_seconds=ENTRY_A_TTL_RTH_HOURS * 3600),
            risk_context=RiskContext(
                stop_for_risk=add_stop,
                planned_entry_price=ref,
                risk_dollars=add_risk,
            ),
        )

        receipt = await self._oms.submit_intent(
            Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=add_order)
        )

        if not receipt.oms_order_id:
            return
        setup_id = next(
            (
                sid
                for sid, candidate in self.active_setups.items()
                if candidate.symbol == symbol and candidate.qty_open > 0
            ),
            "",
        )
        if setup_id:
            self._track_order(
                receipt.oms_order_id,
                setup_id,
                "add_entry",
                add_shares,
            )
        campaign.add_count += 1
        campaign.campaign_risk_used += add_risk
        logger.info("%s: Add #%d placed — shares=%d, ref=%.2f", symbol, campaign.add_count, add_shares, ref)

    # ------------------------------------------------------------------
    # Exit management (spec §20)
    # ------------------------------------------------------------------

    async def _manage_exits(self, now_et: datetime) -> None:
        """Manage TP, trailing stops, stale exits for active positions."""
        for setup_id in list(self.active_setups):
            setup = self.active_setups[setup_id]
            if setup.state not in (SetupState.FILLED, SetupState.ACTIVE):
                continue
            if setup.qty_open <= 0:
                continue

            symbol = setup.symbol
            pos = self.positions.get(symbol)
            if not pos or pos.qty <= 0:
                continue

            setup.bars_held += 1
            hs = self.hourly_states.get(symbol)
            if not hs:
                continue

            current_price = hs.close

            # MFE/MAE price tracking
            bar_high = float(hs.highs[-1]) if hs.highs else current_price
            bar_low = float(hs.lows[-1]) if hs.lows else current_price
            if setup.direction == Direction.LONG:
                if bar_high > setup.mfe_price or setup.mfe_price == 0:
                    setup.mfe_price = bar_high
                if bar_low < setup.mae_price or setup.mae_price == 0:
                    setup.mae_price = bar_low
            else:
                if bar_low < setup.mfe_price or setup.mfe_price == 0:
                    setup.mfe_price = bar_low
                if bar_high > setup.mae_price or setup.mae_price == 0:
                    setup.mae_price = bar_high
            # MFE/MAE R computation
            if setup.fill_price and setup.final_risk_dollars and setup.fill_qty:
                risk_per_share = setup.final_risk_dollars / setup.fill_qty
                if risk_per_share > 0:
                    if setup.direction == Direction.LONG:
                        setup.mfe_r = (setup.mfe_price - setup.fill_price) / risk_per_share
                        setup.mae_r = (setup.fill_price - setup.mae_price) / risk_per_share
                    else:
                        setup.mfe_r = (setup.fill_price - setup.mfe_price) / risk_per_share
                        setup.mae_r = (setup.mae_price - setup.fill_price) / risk_per_share

            # Compute current R (match backtest: include realized PnL from partials)
            r_base = setup.r_price
            r_state = 0.0
            if r_base > 0:
                if setup.direction == Direction.LONG:
                    r_now = (current_price - setup.fill_price) / r_base
                else:
                    r_now = (setup.fill_price - current_price) / r_base
                unrealized = ((current_price - setup.fill_price) * setup.qty_open
                              if setup.direction == Direction.LONG
                              else (setup.fill_price - current_price) * setup.qty_open)
                r_state = ((setup.realized_pnl + unrealized) / setup.final_risk_dollars
                           if setup.final_risk_dollars > 0 else r_now)
            setup.r_state = r_state

            # Gap-through-stop check (at session open)
            if now_et.time().hour == 9 and now_et.time().minute <= 35:
                gap_triggered, gap_fill = stops.handle_gap_through_stop(
                    setup.direction, setup.current_stop, current_price,
                )
                if gap_triggered:
                    setup.gap_stop_event = True
                    setup.gap_stop_fill_price = gap_fill
                    logger.info("%s: Gap-through-stop at %.2f", symbol, gap_fill)
                    await self._close_position(setup, gap_fill, "gap_stop", now_et)
                    continue

            # Compute stale flags (no exit yet — TP takes priority)
            if setup.fill_ts:
                setup.days_held = (now_et - setup.fill_ts).days
            else:
                setup.days_held = setup.bars_held // 7  # fallback
            stale_days = setup.days_held
            _chop_camp = self.campaigns.get(symbol)
            if _chop_camp and _chop_camp.chop_mode == "DEGRADED":
                stale_days -= CHOP_DEGRADED_STALE_ADJ  # ADJ is -2, so adds 2
            should_warn, should_tighten, should_exit_stale = stops.check_stale_exit(
                stale_days, r_state,
            )

            # TP1
            _cfg = self._config[symbol]
            tp1_r, tp2_r = self._get_tp_r_multiples(
                TradeRegime(setup.exit_tier.value), _cfg.tp_scale
            )

            new_stop = setup.current_stop
            tp1_hit_this_bar = False

            # Pre-runner trail: capture MFE as it develops, even before TP1
            # Once r_state exceeds 0.10R, start ratcheting stop toward entry
            if not setup.runner_active and r_state > 0.10:
                if setup.direction == Direction.LONG:
                    candidate = setup.fill_price + 0.30 * (current_price - setup.fill_price)
                    candidate = round_to_tick(candidate, _cfg.tick_size, "down")
                    if candidate > new_stop:
                        new_stop = candidate
                else:
                    candidate = setup.fill_price - 0.30 * (setup.fill_price - current_price)
                    candidate = round_to_tick(candidate, _cfg.tick_size, "up")
                    if candidate < new_stop:
                        new_stop = candidate

            if not setup.tp1_done and not setup.tp1_order_id and r_state >= tp1_r:
                setup.tp1_done = True
                pos.tp1_done = True
                tp1_hit_this_bar = True
                # Two-leg exit: close 33% at TP1, runner on remaining 67%
                # Degraded/range: close 50% at TP1, runner on 50%
                _chop_cap = (self.campaigns.get(symbol) and
                             self.campaigns[symbol].chop_mode == "DEGRADED")
                _range_cap = setup.exit_tier == ExitTier.NEUTRAL
                if _chop_cap or _range_cap:
                    tp1_qty = max(1, setup.qty_open // 2)
                else:
                    tp1_qty = max(1, setup.qty_open // 3)

                # Immediately activate runner on remainder (skip TP2 partial)
                setup.tp2_done = True
                setup.runner_active = True
                pos.tp2_done = True
                pos.runner_active = True

                await self._submit_partial_close(setup, tp1_qty, current_price, "tp1", now_et)

                # Move stop to BE
                be_stop = stops.compute_be_stop(
                    setup.direction, setup.fill_price,
                    self.campaigns[symbol].daily_context.get("atr14_d", 1.0) if self.campaigns[symbol].daily_context else 1.0,
                    _cfg.tick_size,
                )
                if setup.direction == Direction.LONG:
                    new_stop = max(new_stop, be_stop)
                else:
                    new_stop = min(new_stop, be_stop)

                logger.info("%s: TP1 hit at R=%.2f, partial close %d shares, stop → BE=%.2f",
                            symbol, r_state, tp1_qty, new_stop)

            # Update stop if tightened
            if new_stop != setup.current_stop:
                safe = False
                if setup.direction == Direction.LONG and new_stop > setup.current_stop:
                    safe = True
                elif setup.direction == Direction.SHORT and new_stop < setup.current_stop:
                    safe = True
                if safe:
                    old_stop = setup.current_stop
                    setup.current_stop = new_stop
                    pos.current_stop = new_stop
                    if self._kit:
                        self._kit.log_stop_adjustment(
                            trade_id=setup.trade_id or f"BKOUT-{symbol}",
                            symbol=symbol, old_stop=old_stop, new_stop=new_stop,
                            adjustment_type="trailing", trigger="atr_trail",
                        )

            # Stop hit check — covers pre-runner trail and any stop ratchets
            # Skip if broker owns the stop (stop_order_id is set)
            if not tp1_hit_this_bar and not setup.stop_order_id:
                stopped = (
                    (setup.direction == Direction.LONG and current_price <= setup.current_stop)
                    or (setup.direction == Direction.SHORT and current_price >= setup.current_stop)
                )
                if stopped:
                    logger.info("%s: Stop hit at %.2f (stop=%.2f, R=%.2f)",
                                symbol, current_price, setup.current_stop, r_state)
                    await self._close_position(setup, current_price, "stop_hit", now_et)
                    continue

            # Stale exit (deferred — only if TP1 not hit this bar)
            if should_exit_stale and not tp1_hit_this_bar:
                logger.info("%s: Stale exit at R=%.2f", symbol, r_state)
                await self._close_position(setup, current_price, "stale_exit", now_et)
                continue

            # Trailing stop (runner) — spec §20.3
            if setup.runner_active:
                # Stale tighten applies on top of trailing
                if should_tighten:
                    setup.trail_mult = stops.apply_stale_tighten(setup.trail_mult)

                # Compute 4H trailing stop using hourly bar data
                await self._update_runner_trailing_stop(setup, pos, symbol, now_et)

    async def _submit_partial_close(
        self, setup: SetupInstance, qty: int, price: float, reason: str, now: datetime,
    ) -> None:
        """Submit a partial close order for TP1/TP2."""
        inst = self._instruments.get(setup.symbol)
        if not inst or qty <= 0:
            return

        exit_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
        partial_order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=exit_side,
            qty=qty,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.EXIT,
        )
        receipt = await self._oms.submit_intent(
            Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=partial_order)
        )

        # Track the partial close order — fill handler will update qty_open
        if receipt and receipt.oms_order_id:
            self._track_order(receipt.oms_order_id, setup.setup_id, "partial_close", qty)

    async def _update_runner_trailing_stop(
        self, setup: SetupInstance, pos: PositionState, symbol: str, now_et: datetime,
    ) -> None:
        """Fetch 4H data and ratchet the runner trailing stop (spec §20.3)."""
        hourly_bars_raw = await self._fetch_hourly_bars(symbol, duration="20 D")
        if not hourly_bars_raw or len(hourly_bars_raw) < 20:
            return

        h_closes = np.array([b.close for b in hourly_bars_raw], dtype=float)
        h_highs = np.array([b.high for b in hourly_bars_raw], dtype=float)
        h_lows = np.array([b.low for b in hourly_bars_raw], dtype=float)
        h_vols = np.array([b.volume for b in hourly_bars_raw], dtype=float)
        h_times = [
            b.date if isinstance(b.date, datetime)
            else datetime.fromisoformat(str(b.date))
            for b in hourly_bars_raw
        ]

        h4_h, h4_l, h4_c, h4_v, h4_t = construct_4h_bars(h_highs, h_lows, h_closes, h_vols, h_times)
        if len(h4_c) < 10:
            return

        from .indicators import atr as compute_atr, ema as compute_ema
        from .config import EMA_4H_PERIOD, ATR_HOURLY_PERIOD

        atr14_4h_arr = compute_atr(h4_h, h4_l, h4_c, ATR_HOURLY_PERIOD)
        atr14_4h = float(atr14_4h_arr[-1])
        ema50_4h = float(compute_ema(h4_c, EMA_4H_PERIOD)[-1])

        # Compute trail mult considering R state and continuation
        campaign = self.campaigns.get(symbol)
        continuation = campaign.continuation if campaign else False
        r_proxy = setup.r_state
        setup.trail_mult = stops.compute_trail_mult(setup.r_state, r_proxy, continuation)

        cfg = self._config[symbol]
        new_stop = stops.compute_trailing_stop(
            setup.direction,
            [float(h) for h in h4_h],
            [float(l) for l in h4_l],
            atr14_4h,
            setup.trail_mult,
            ema50_4h,
            setup.current_stop,
            cfg.tick_size,
        )

        if setup.direction == Direction.LONG:
            if new_stop > setup.current_stop:
                old_stop = setup.current_stop
                setup.current_stop = new_stop
                pos.current_stop = new_stop
                logger.info("%s: Runner trail ratcheted stop → %.2f", symbol, new_stop)
                if self._kit:
                    self._kit.log_stop_adjustment(
                        trade_id=setup.trade_id or f"BKOUT-{symbol}",
                        symbol=symbol, old_stop=old_stop, new_stop=new_stop,
                        adjustment_type="trailing", trigger="runner_4h_trail",
                    )
        else:
            if new_stop < setup.current_stop:
                old_stop = setup.current_stop
                setup.current_stop = new_stop
                pos.current_stop = new_stop
                logger.info("%s: Runner trail ratcheted stop → %.2f", symbol, new_stop)
                if self._kit:
                    self._kit.log_stop_adjustment(
                        trade_id=setup.trade_id or f"BKOUT-{symbol}",
                        symbol=symbol, old_stop=old_stop, new_stop=new_stop,
                        adjustment_type="trailing", trigger="runner_4h_trail",
                    )

        # Check if current price hit the trailing stop
        current_price = self.hourly_states[symbol].close if symbol in self.hourly_states else 0.0
        if current_price > 0:
            stopped = (setup.direction == Direction.LONG and current_price <= setup.current_stop) or \
                      (setup.direction == Direction.SHORT and current_price >= setup.current_stop)
            if stopped:
                logger.info("%s: Runner trailing stop hit at %.2f", symbol, current_price)
                await self._close_position(setup, current_price, "trailing_stop", now_et)

    async def _close_position(
        self, setup: SetupInstance, exit_price: float, reason: str, now: datetime,
    ) -> None:
        """Submit close order and record trade."""
        # Guard: no-op if already closed or broker already exiting
        if setup.qty_open <= 0:
            return
        if self._has_pending_exit(setup):
            logger.info("%s: Skipping engine close — exit order already in flight", setup.symbol)
            return

        inst = self._instruments.get(setup.symbol)
        if not inst:
            return

        side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
        close_order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=setup.qty_open,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.EXIT,
        )
        await self._oms.submit_intent(
            Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=close_order)
        )
        # Cancel any outstanding bracket orders before finalizing
        await self._cancel_remaining_exit_orders(setup, except_kind="engine_close")
        await self._finalize_closed_setup(
            setup,
            exit_price,
            reason,
            now,
            order_id=setup.stop_order_id or setup.primary_order_id or "",
        )

    def _has_pending_exit(self, setup: SetupInstance) -> bool:
        """True if any tracked order for this setup is an exit-type order."""
        for oms_order_id, setup_id in self._order_to_setup.items():
            if setup_id != setup.setup_id:
                continue
            kind = self._order_kind.get(oms_order_id, "")
            if kind in {"stop", "partial_close", "engine_close"}:
                return True
        return False

    # ------------------------------------------------------------------
    # Bracket orders (spec §16, §20.2) — protect positions on crash
    # ------------------------------------------------------------------

    async def _submit_bracket_orders(self, setup: SetupInstance) -> None:
        """Submit standing stop-loss, TP1, and TP2 orders via OMS after a fill."""
        inst = self._instruments.get(setup.symbol)
        if not inst:
            return

        cfg = self._config[setup.symbol]
        exit_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY

        # --- Stop-loss order ---
        stop_order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=exit_side,
            qty=setup.fill_qty,
            order_type=OrderType.STOP,
            stop_price=setup.current_stop,
            tif="GTC",
            role=OrderRole.EXIT,
            oca_group=setup.oca_group,
            oca_type=1,
        )
        stop_receipt = await self._oms.submit_intent(
            Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=stop_order)
        )
        if stop_receipt.oms_order_id:
            setup.stop_order_id = stop_receipt.oms_order_id
            self._track_order(
                stop_receipt.oms_order_id,
                setup.setup_id,
                "stop",
                setup.fill_qty,
            )

        # --- TP levels ---
        trade_regime = TradeRegime(setup.exit_tier.value)
        tp1_r, tp2_r = self._get_tp_r_multiples(trade_regime, cfg.tp_scale)
        tp1_price, tp2_price = stops.compute_tp_levels(
            setup.direction, setup.fill_price, setup.r_price,
            tp1_r, tp2_r, cfg.tick_size,
        )

        # TP1 order — regime-sensitive partial close (matches _manage_exits)
        _chop_cap = (self.campaigns.get(setup.symbol) and
                     self.campaigns[setup.symbol].chop_mode == "DEGRADED")
        _range_cap = setup.exit_tier == ExitTier.NEUTRAL
        if _chop_cap or _range_cap:
            tp1_qty = max(1, setup.fill_qty // 2)
        else:
            tp1_qty = max(1, setup.fill_qty // 3)
        tp1_order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=exit_side,
            qty=tp1_qty,
            order_type=OrderType.LIMIT,
            limit_price=tp1_price,
            tif="GTC",
            role=OrderRole.EXIT,
            oca_group=setup.oca_group,
            oca_type=1,
        )
        tp1_receipt = await self._oms.submit_intent(
            Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=tp1_order)
        )
        if tp1_receipt.oms_order_id:
            setup.tp1_order_id = tp1_receipt.oms_order_id
            self._track_order(
                tp1_receipt.oms_order_id,
                setup.setup_id,
                "tp1",
                tp1_qty,
            )

        # TP2 order — partial close (half of remaining)
        remaining_after_tp1 = setup.fill_qty - tp1_qty
        tp2_qty = max(1, remaining_after_tp1 // 2) if remaining_after_tp1 > 1 else remaining_after_tp1
        if tp2_qty > 0:
            tp2_order = OMSOrder(
                strategy_id=STRATEGY_ID,
                instrument=inst,
                side=exit_side,
                qty=tp2_qty,
                order_type=OrderType.LIMIT,
                limit_price=tp2_price,
                tif="GTC",
                role=OrderRole.EXIT,
                oca_group=setup.oca_group,
                oca_type=1,
            )
            tp2_receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=tp2_order)
            )
            if tp2_receipt.oms_order_id:
                setup.tp2_order_id = tp2_receipt.oms_order_id
                self._track_order(
                    tp2_receipt.oms_order_id,
                    setup.setup_id,
                    "tp2",
                    tp2_qty,
                )

        logger.info("%s: Bracket orders submitted — stop=%.2f, TP1=%.2f (%d), TP2=%.2f (%d)",
                    setup.symbol, setup.current_stop, tp1_price, tp1_qty, tp2_price, tp2_qty)

    # ------------------------------------------------------------------
    # OMS Event Processing
    # ------------------------------------------------------------------

    async def _cancel_remaining_exit_orders(
        self, setup: SetupInstance, except_kind: str,
    ) -> None:
        """Cancel outstanding bracket exit orders after a terminal fill.

        On stop fill: cancel TP orders.
        On TP fill that closes position: cancel stop + remaining TPs.
        """
        to_cancel: list[tuple[str, str]] = []  # (order_id, kind)
        if except_kind != "stop" and setup.stop_order_id:
            to_cancel.append((setup.stop_order_id, "stop"))
        if except_kind != "tp1" and setup.tp1_order_id:
            to_cancel.append((setup.tp1_order_id, "tp1"))
        if except_kind != "tp2" and setup.tp2_order_id:
            to_cancel.append((setup.tp2_order_id, "tp2"))

        for order_id, kind in to_cancel:
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=order_id,
                    )
                )
                logger.info("%s: Cancelled %s order %s after %s fill",
                            setup.symbol, kind, order_id, except_kind)
            except Exception as e:
                logger.warning("%s: Failed to cancel %s order %s: %s",
                               setup.symbol, kind, order_id, e)
            # Clear the reference regardless — broker will handle the cancel
            self._forget_order(order_id)

    async def _amend_stop_qty(self, setup: SetupInstance) -> None:
        """Replace the protective stop with updated qty after partial TP fill."""
        old_stop_id = setup.stop_order_id
        if not old_stop_id:
            return
        inst = self._instruments.get(setup.symbol)
        if not inst:
            return

        # Cancel old stop
        try:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.CANCEL_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=old_stop_id,
                )
            )
        except Exception as e:
            logger.warning("%s: Failed to cancel old stop for qty amendment: %s",
                           setup.symbol, e)
            return
        self._forget_order(old_stop_id)

        # Submit replacement stop with reduced qty
        exit_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
        new_stop = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=exit_side,
            qty=setup.qty_open,
            order_type=OrderType.STOP,
            stop_price=setup.current_stop,
            tif="GTC",
            role=OrderRole.EXIT,
            oca_group=setup.oca_group,
            oca_type=1,
        )
        receipt = await self._oms.submit_intent(
            Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=new_stop)
        )
        if receipt.oms_order_id:
            setup.stop_order_id = receipt.oms_order_id
            self._track_order(receipt.oms_order_id, setup.setup_id, "stop", setup.qty_open)
            logger.info("%s: Stop qty amended %d → %d (order %s)",
                        setup.symbol, self._order_requested_qty.get(old_stop_id, 0),
                        setup.qty_open, receipt.oms_order_id)

    def _track_order(
        self,
        oms_order_id: str,
        setup_id: str,
        order_kind: str,
        requested_qty: int = 0,
    ) -> None:
        """Track live order metadata needed to reconcile broker events."""
        self._order_to_setup[oms_order_id] = setup_id
        self._order_kind[oms_order_id] = order_kind
        if requested_qty > 0:
            self._order_requested_qty[oms_order_id] = requested_qty

    def _forget_order(self, oms_order_id: str | None) -> tuple[str, int]:
        """Remove tracked order metadata and clear any attached exit reference."""
        if not oms_order_id:
            return "", 0

        setup_id = self._order_to_setup.pop(oms_order_id, "")
        order_kind = self._order_kind.pop(oms_order_id, "")
        requested_qty = self._order_requested_qty.pop(oms_order_id, 0)

        setup = self.active_setups.get(setup_id) if setup_id else None
        if setup:
            if order_kind == "stop" and setup.stop_order_id == oms_order_id:
                setup.stop_order_id = ""
            elif order_kind == "tp1" and setup.tp1_order_id == oms_order_id:
                setup.tp1_order_id = ""
            elif order_kind == "tp2" and setup.tp2_order_id == oms_order_id:
                setup.tp2_order_id = ""

        return order_kind, requested_qty

    def _clear_setup_tracked_orders(self, setup: SetupInstance) -> None:
        """Drop all outstanding tracked orders for a setup once it is no longer live."""
        for oms_order_id, setup_id in list(self._order_to_setup.items()):
            if setup_id != setup.setup_id:
                continue
            self._forget_order(oms_order_id)

    def _extract_fill_details(
        self,
        event: OMSEvent,
        *,
        fallback_price: float,
        fallback_qty: int,
    ) -> tuple[float, int]:
        """Normalize OMS fill payloads across FILL and ORDER_FILLED events."""
        payload = event.payload or {}
        if event.event_type == OMSEventType.FILL:
            fill_price = float(payload.get("price", 0) or 0)
            fill_qty = int(payload.get("qty", 0) or 0)
        else:
            fill_price = float(payload.get("avg_fill_price", 0) or 0)
            fill_qty = int(payload.get("filled_qty", 0) or 0)

        if fill_price <= 0:
            fill_price = float(fallback_price)
        if fill_qty <= 0:
            fill_qty = int(fallback_qty)
        return fill_price, fill_qty

    def _apply_add_fill(
        self,
        setup: SetupInstance,
        fill_price: float,
        fill_qty: int,
    ) -> int:
        """Sync a filled add-on entry into local position state."""
        add_qty = max(0, fill_qty)
        if add_qty <= 0:
            return 0

        current_qty = max(0, setup.fill_qty)
        entry_anchor = setup.avg_entry if setup.avg_entry > 0 else setup.fill_price
        if current_qty > 0 and entry_anchor > 0:
            setup.avg_entry = ((entry_anchor * current_qty) + (fill_price * add_qty)) / (current_qty + add_qty)
        else:
            setup.avg_entry = fill_price

        setup.fill_qty = current_qty + add_qty
        setup.qty_open += add_qty
        setup.add_count += 1
        setup.state = SetupState.ACTIVE

        pos = self.positions.setdefault(
            setup.symbol,
            PositionState(
                symbol=setup.symbol,
                direction=setup.direction,
                campaign_id=setup.campaign_id,
                box_version=setup.box_version,
            ),
        )
        pos.qty += add_qty
        pos.avg_cost = setup.avg_entry
        pos.add_count += 1
        return add_qty

    def _apply_exit_fill(
        self,
        setup: SetupInstance,
        fill_price: float,
        fill_qty: int,
    ) -> int:
        """Apply a broker-driven exit fill to local quantity and realized PnL."""
        exit_qty = max(0, min(fill_qty, setup.qty_open if setup.qty_open > 0 else fill_qty))
        if exit_qty <= 0:
            return 0

        entry_anchor = setup.fill_price if setup.fill_price > 0 else setup.avg_entry
        if setup.direction == Direction.LONG:
            pnl_points = fill_price - entry_anchor
        else:
            pnl_points = entry_anchor - fill_price
        point_value = float(getattr(self._instruments.get(setup.symbol), "point_value", 1.0) or 1.0)
        setup.realized_pnl += pnl_points * point_value * exit_qty
        setup.qty_open = max(0, setup.qty_open - exit_qty)
        if setup.final_risk_dollars > 0:
            setup.r_state = setup.realized_pnl / setup.final_risk_dollars

        pos = self.positions.get(setup.symbol)
        if pos:
            pos.qty = max(0, pos.qty - exit_qty)
            pos.r_state = setup.r_state

        return exit_qty

    async def _finalize_closed_setup(
        self,
        setup: SetupInstance,
        exit_price: float,
        reason: str,
        now: datetime,
        *,
        order_id: str,
    ) -> None:
        """Finalize local strategy state after the position is fully closed."""
        self._clear_setup_tracked_orders(setup)

        setup.state = SetupState.CLOSED
        setup.qty_open = 0
        pos = self.positions.get(setup.symbol)
        if pos:
            pos.qty = 0

        # Remove from active_setups so hourly cycle no longer iterates it
        self.active_setups.pop(setup.setup_id, None)

        campaign = self.campaigns.get(setup.symbol)
        if campaign:
            exit_date = now.date() if isinstance(now, datetime) else now
            campaign.last_exit_date = exit_date
            campaign.last_exit_direction = setup.direction
            campaign.last_exit_realized_r = setup.r_state

        if self._recorder:
            ctx = campaign.daily_context if campaign and campaign.daily_context else {}
            record = TradeRecord(
                symbol=setup.symbol,
                direction=setup.direction,
                entry_type=setup.entry_type,
                exit_tier=setup.exit_tier,
                campaign_id=setup.campaign_id,
                box_version=setup.box_version,
                entry_ts=setup.fill_ts,
                exit_ts=now,
                bars_held=setup.bars_held,
                days_held=setup.days_held,
                entry_price=setup.fill_price,
                exit_price=exit_price,
                stop_price=setup.stop0,
                shares=setup.fill_qty,
                realized_pnl=setup.realized_pnl,
                r_multiple=setup.r_state,
                disp_at_entry=ctx.get("disp", 0.0),
                rvol_d_at_entry=ctx.get("rvol_d", 1.0),
                regime_at_entry=setup.regime_at_entry or "",
                quality_mult_at_entry=setup.quality_mult,
                gap_stop_event=setup.gap_stop_event,
            )
            await self._recorder.record(record.__dict__)

        if self._kit:
            if setup.direction == Direction.LONG:
                _mfe_pct = (setup.mfe_price - setup.fill_price) / setup.fill_price if setup.fill_price > 0 else None
                _mae_pct = (setup.fill_price - setup.mae_price) / setup.fill_price if setup.fill_price > 0 else None
                _pnl_pct = (exit_price - setup.fill_price) / setup.fill_price if setup.fill_price > 0 else None
            else:
                _mfe_pct = (setup.fill_price - setup.mfe_price) / setup.fill_price if setup.fill_price > 0 else None
                _mae_pct = (setup.mae_price - setup.fill_price) / setup.fill_price if setup.fill_price > 0 else None
                _pnl_pct = (setup.fill_price - exit_price) / setup.fill_price if setup.fill_price > 0 else None
            self._kit.log_exit(
                trade_id=setup.setup_id,
                exit_price=exit_price,
                exit_reason=reason,
                mfe_price=setup.mfe_price,
                mae_price=setup.mae_price,
                mfe_r=setup.mfe_r,
                mae_r=setup.mae_r,
                mfe_pct=_mfe_pct,
                mae_pct=_mae_pct,
                pnl_pct=_pnl_pct,
            )

            self._kit.on_order_event(
                order_id=order_id or setup.primary_order_id or "",
                pair=setup.symbol,
                side="SELL" if setup.direction == Direction.LONG else "BUY",
                order_type="STOP",
                status="FILLED",
                requested_qty=float(setup.fill_qty),
                filled_qty=float(setup.fill_qty),
                requested_price=setup.current_stop,
                fill_price=exit_price,
                related_trade_id=setup.setup_id,
                strategy_id=STRATEGY_ID,
            )

        logger.info("%s: Position closed ??reason=%s, R=%.2f", setup.symbol, reason, setup.r_state)

    async def _process_events(self, event_queue: asyncio.Queue) -> None:
        """Process OMS events (fills, cancels, etc.)."""
        while self._running:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                if event.event_type == OMSEventType.RISK_HALT:
                    await self._on_risk_halt((event.payload or {}).get("reason", ""))
                    continue

                oms_order_id = getattr(event, "oms_order_id", None)
                setup_id = self._order_to_setup.get(oms_order_id, "") if oms_order_id else ""
                setup = self.active_setups.get(setup_id)
                if not setup:
                    self._forget_order(oms_order_id)
                    continue
                order_kind = self._order_kind.get(oms_order_id, "")
                tracked_qty = self._order_requested_qty.get(oms_order_id, 0)

                if event.event_type in (OMSEventType.FILL, OMSEventType.ORDER_FILLED):
                    if order_kind == "add_entry":
                        fill_price, fill_qty = self._extract_fill_details(
                            event,
                            fallback_price=setup.entry_price or setup.fill_price,
                            fallback_qty=tracked_qty,
                        )
                        self._forget_order(oms_order_id)
                        filled_qty = self._apply_add_fill(setup, fill_price, fill_qty)
                        if filled_qty > 0:
                            logger.info(
                                "%s: Add filled %d @ %.2f, qty_open=%d",
                                setup.symbol,
                                filled_qty,
                                fill_price,
                                setup.qty_open,
                            )
                        continue

                    if order_kind in {"tp1", "tp2", "stop", "partial_close"}:
                        fill_price, fill_qty = self._extract_fill_details(
                            event,
                            fallback_price=setup.current_stop if order_kind == "stop" else setup.fill_price,
                            fallback_qty=tracked_qty or setup.qty_open,
                        )
                        self._forget_order(oms_order_id)
                        filled_qty = self._apply_exit_fill(setup, fill_price, fill_qty)
                        if filled_qty <= 0:
                            continue

                        pos = self.positions.get(setup.symbol)
                        if order_kind == "tp1":
                            setup.tp1_done = True
                            if pos:
                                pos.tp1_done = True
                        elif order_kind == "tp2":
                            setup.tp2_done = True
                            if pos:
                                pos.tp2_done = True

                        if order_kind == "stop" or setup.qty_open <= 0 or (pos and pos.qty <= 0):
                            # Cancel remaining bracket orders before finalizing
                            await self._cancel_remaining_exit_orders(setup, except_kind=order_kind)
                            await self._finalize_closed_setup(
                                setup,
                                fill_price,
                                f"{order_kind}_fill",
                                datetime.now(timezone.utc),
                                order_id=oms_order_id or "",
                            )
                        else:
                            logger.info(
                                "%s: %s filled %d @ %.2f, qty_open=%d",
                                setup.symbol,
                                order_kind.upper(),
                                filled_qty,
                                fill_price,
                                setup.qty_open,
                            )
                            # Amend stop qty to match reduced position after partial TP
                            if order_kind in ("tp1", "tp2") and setup.stop_order_id and setup.qty_open > 0:
                                await self._amend_stop_qty(setup)
                        continue

                    if order_kind != "primary_entry":
                        continue
                    if setup.state in (SetupState.FILLED, SetupState.ACTIVE):
                        continue

                    fill_price, fill_qty = self._extract_fill_details(
                        event,
                        fallback_price=setup.entry_price,
                        fallback_qty=tracked_qty or setup.shares_planned,
                    )
                    self._forget_order(oms_order_id)
                    setup.state = SetupState.FILLED
                    setup.fill_price = fill_price
                    setup.fill_qty = fill_qty
                    setup.fill_ts = datetime.now(timezone.utc)
                    setup.qty_open = setup.fill_qty
                    setup.avg_entry = setup.fill_price

                    # Update position state
                    pos = self.positions.setdefault(
                        setup.symbol,
                        PositionState(
                            symbol=setup.symbol,
                            direction=setup.direction,
                            campaign_id=setup.campaign_id,
                            box_version=setup.box_version,
                        ),
                    )
                    pos.qty += setup.fill_qty
                    pos.avg_cost = setup.fill_price
                    pos.current_stop = setup.current_stop
                    pos.total_risk_dollars += setup.final_risk_dollars
                    if pos.regime_at_entry is None:
                        pos.regime_at_entry = setup.regime_at_entry

                    # Update campaign state
                    campaign = self.campaigns.get(setup.symbol)
                    if campaign:
                        if campaign.state == CampaignState.BREAKOUT:
                            campaign.state = CampaignState.POSITION_OPEN
                        # Track re-entry count (spec §21)
                        if campaign.last_exit_date is not None:
                            dir_key = "LONG" if setup.direction == Direction.LONG else "SHORT"
                            bv = campaign.box_version
                            campaign.reentry_count.setdefault(dir_key, {})
                            campaign.reentry_count[dir_key][bv] = campaign.reentry_count[dir_key].get(bv, 0) + 1
                            campaign.last_exit_date = None  # consumed

                    logger.info("%s: Filled %d @ %.2f", setup.symbol, setup.fill_qty, setup.fill_price)

                    # Hook 4: Instrumentation trade entry
                    if self._kit:
                        campaign = self.campaigns.get(setup.symbol)
                        ctx = campaign.daily_context if campaign else {}
                        active_pos = {s: p for s, p in self.positions.items() if p.qty > 0}
                        total_risk = sum(p.total_risk_dollars for p in active_pos.values())
                        # Correlated pairs
                        correlated = []
                        for sym_b, pos_b in self.positions.items():
                            if pos_b.qty <= 0 or sym_b == setup.symbol:
                                continue
                            pair_key = tuple(sorted([setup.symbol, sym_b]))
                            corr = self.correlation_map.get(pair_key, 0.0)
                            if abs(corr) > 0.5:
                                correlated.append({
                                    "symbol": sym_b,
                                    "direction": "LONG" if pos_b.direction == Direction.LONG else "SHORT",
                                    "correlation": round(corr, 3),
                                    "same_direction": (pos_b.direction == setup.direction),
                                })
                        _s3_cfg = self._config.get(setup.symbol)
                        _cfg_dict = dataclasses.asdict(_s3_cfg) if _s3_cfg else {}
                        _param_set_id = hashlib.md5(
                            json.dumps(_cfg_dict, sort_keys=True, default=str).encode()
                        ).hexdigest()[:8]
                        self._kit.log_entry(
                            trade_id=setup.setup_id,
                            pair=setup.symbol,
                            side="LONG" if setup.direction == Direction.LONG else "SHORT",
                            entry_price=setup.fill_price,
                            position_size=float(setup.fill_qty),
                            position_size_quote=setup.fill_price * setup.fill_qty,
                            entry_signal=setup.entry_type.value if hasattr(setup.entry_type, "value") else str(setup.entry_type),
                            entry_signal_id=setup.setup_id,
                            entry_signal_strength=setup.quality_mult if hasattr(setup, "quality_mult") else 0.5,
                            active_filters=["score_threshold", "displacement", "eligibility"],
                            passed_filters=["score_threshold", "displacement", "eligibility"],
                            filter_decisions=setup.filter_decisions if hasattr(setup, "filter_decisions") else [],
                            strategy_params={
                                "param_set_id": _param_set_id,
                                "config": _cfg_dict,
                                "final_risk_dollars": setup.final_risk_dollars,
                                "entry_type": setup.entry_type.value if hasattr(setup.entry_type, "value") else str(setup.entry_type),
                                "quality_mult": setup.quality_mult,
                                "score_total": ctx.get("score_total"),
                                "score_threshold": ctx.get("score_threshold"),
                                "regime_4h": ctx.get("regime_4h").value if ctx.get("regime_4h") and hasattr(ctx["regime_4h"], "value") else str(ctx.get("regime_4h")),
                                "trade_regime": str(ctx.get("trade_regime")),
                                "chop_mode": ctx.get("chop_mode"),
                                "atr14_d": ctx.get("atr14_d"),
                                "risk_regime": ctx.get("risk_regime"),
                                "cb_mult": ctx.get("cb_mult"),
                                "expiry_mult": ctx.get("expiry_mult"),
                                "base_risk_pct": self._base_risk_pct if hasattr(self, "_base_risk_pct") else 0.01,
                            },
                            expected_entry_price=setup.fill_price,
                            signal_factors=[
                                {"factor_name": "score_total", "factor_value": ctx.get("score_total", 0),
                                 "threshold": ctx.get("score_threshold", 2), "contribution": "evidence_quality"},
                                {"factor_name": "score_vol", "factor_value": ctx.get("score_vol", 0),
                                 "threshold": 0, "contribution": "volume_confirmation"},
                                {"factor_name": "score_squeeze", "factor_value": ctx.get("score_squeeze", 0),
                                 "threshold": 0, "contribution": "compression_quality"},
                                {"factor_name": "score_regime", "factor_value": ctx.get("score_regime", 0),
                                 "threshold": 0, "contribution": "regime_alignment"},
                                {"factor_name": "score_consec", "factor_value": ctx.get("score_consec", 0),
                                 "threshold": 0, "contribution": "consecutive_closes"},
                                {"factor_name": "displacement", "factor_value": ctx.get("disp", 0),
                                 "threshold": ctx.get("disp_th", 0), "contribution": "breakout_magnitude"},
                                {"factor_name": "quality_mult", "factor_value": ctx.get("quality_mult", 0),
                                 "threshold": 0.25, "contribution": "composite_sizing_quality"},
                            ],
                            sizing_inputs={
                                "target_risk_pct": self._base_risk_pct if hasattr(self, "_base_risk_pct") else 0.01,
                                "account_equity": self._equity if hasattr(self, "_equity") else 0.0,
                                "volatility_basis": setup.final_risk_dollars,
                                "sizing_model": "breakout_r_risk",
                            },
                            portfolio_state_at_entry={
                                "num_positions": len(active_pos),
                                "total_risk_dollars": round(total_risk, 2),
                                "total_exposure_pct": round(total_risk / self._equity, 4) if self._equity > 0 else 0,
                                "long_positions": sum(1 for p in active_pos.values() if p.direction == Direction.LONG),
                                "short_positions": sum(1 for p in active_pos.values() if p.direction == Direction.SHORT),
                            },
                            correlated_pairs_detail=correlated if correlated else None,
                            concurrent_positions_strategy=len(self.active_setups),
                        )

                        self._kit.on_order_event(
                            order_id=oms_order_id,
                            pair=setup.symbol,
                            side="LONG" if setup.direction == Direction.LONG else "SHORT",
                            order_type=setup.entry_type.value if hasattr(setup.entry_type, "value") else str(setup.entry_type),
                            status="FILLED",
                            requested_qty=float(setup.shares_planned),
                            filled_qty=float(setup.fill_qty),
                            requested_price=setup.entry_price,
                            fill_price=setup.fill_price,
                            related_trade_id=setup.setup_id,
                            strategy_id=STRATEGY_ID,
                        )

                    # Submit bracket orders (stop + TP) for position protection
                    await self._submit_bracket_orders(setup)

                elif event.event_type in (
                    OMSEventType.ORDER_CANCELLED,
                    OMSEventType.ORDER_REJECTED,
                    OMSEventType.ORDER_EXPIRED,
                ):
                    self._forget_order(oms_order_id)
                    if order_kind == "stop" and setup.qty_open > 0:
                        logger.warning(
                            "%s: Protective stop order terminal (%s) while %d shares remain open",
                            setup.symbol,
                            event.event_type.value,
                            setup.qty_open,
                        )
                        continue
                    if order_kind != "primary_entry":
                        continue
                    if setup.state in (SetupState.ARMED, SetupState.TRIGGERED):
                        if event.event_type == OMSEventType.ORDER_EXPIRED:
                            setup.state = SetupState.EXPIRED
                        else:
                            setup.state = SetupState.CANCELLED
                        self.active_setups.pop(setup.setup_id, None)
                        logger.info("%s: Order terminal (%s)", setup.symbol, event.event_type.value)

                        if self._kit:
                            status_map = {
                                OMSEventType.ORDER_CANCELLED: "CANCELLED",
                                OMSEventType.ORDER_REJECTED: "REJECTED",
                                OMSEventType.ORDER_EXPIRED: "EXPIRED",
                            }
                            self._kit.on_order_event(
                                order_id=oms_order_id,
                                pair=setup.symbol,
                                side="LONG" if setup.direction == Direction.LONG else "SHORT",
                                order_type=setup.entry_type.value if hasattr(setup.entry_type, "value") else str(setup.entry_type),
                                status=status_map.get(event.event_type, "CANCELLED"),
                                requested_qty=float(setup.shares_planned),
                                requested_price=setup.entry_price,
                                strategy_id=STRATEGY_ID,
                            )

            except Exception:
                logger.exception("Error processing OMS event")

    async def _on_risk_halt(self, reason: str) -> None:
        """Pause new entries and cancel live entry intents."""
        if self._risk_halted:
            return

        self._risk_halted = True
        self._risk_halt_reason = reason or "OMS risk halt"
        logger.error("Breakout risk halt engaged: %s", self._risk_halt_reason)

        for oms_order_id, order_kind in list(self._order_kind.items()):
            if order_kind not in {"primary_entry", "add_entry"}:
                continue
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=oms_order_id,
                    )
                )
            except Exception:
                logger.warning(
                    "Failed to cancel Breakout order %s during risk halt",
                    oms_order_id,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _refresh_equity(self) -> None:
        """Fetch current account equity."""
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                for item in self._ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == accounts[0]:
                        raw = float(item.value)
                        self._equity = raw * self._equity_alloc_pct + self._equity_offset
                        if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
                            self._kit.ctx.drawdown_tracker.update_equity(self._equity)
                        return
        except Exception:
            logger.warning("Could not refresh equity, using $%.2f", self._equity)

    async def _fetch_hourly_bars(self, symbol: str, duration: str = "5 D") -> list[Any] | None:
        """Fetch recent hourly bars from IB."""
        contract = self.contracts.get(symbol)
        if not contract:
            return None
        try:
            return await self._ib.ib.reqHistoricalDataAsync(
                contract, endDateTime="", durationStr=duration,
                barSizeSetting="1 hour", whatToShow="TRADES",
                useRTH=True, formatDate=1,
            )
        except Exception:
            logger.warning("Could not fetch hourly bars for %s", symbol)
            return None

    def _get_tp_r_multiples(self, trade_regime: TradeRegime, tp_scale: float = 1.0) -> tuple[float, float]:
        """Return (TP1_R, TP2_R) for the given trade regime, scaled per-symbol."""
        if trade_regime == TradeRegime.ALIGNED:
            return TP1_R_ALIGNED * tp_scale, TP2_R_ALIGNED * tp_scale
        if trade_regime == TradeRegime.CAUTION:
            return TP1_R_CAUTION * tp_scale, TP2_R_CAUTION * tp_scale
        return TP1_R_NEUTRAL * tp_scale, TP2_R_NEUTRAL * tp_scale

    def _update_correlations(self) -> None:
        """Update rolling correlations between all symbol pairs."""
        symbols = list(self._config.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a, sym_b = symbols[i], symbols[j]
                hs_a = self.hourly_states.get(sym_a)
                hs_b = self.hourly_states.get(sym_b)
                if not hs_a or not hs_b or len(hs_a.closes) < 20 or len(hs_b.closes) < 20:
                    continue
                # Compute 4H returns (approximate from hourly closes, every 4th)
                c_a = np.array(hs_a.closes[-CORR_LOOKBACK_BARS * 4:])
                c_b = np.array(hs_b.closes[-CORR_LOOKBACK_BARS * 4:])
                if len(c_a) > 4 and len(c_b) > 4:
                    ret_a = np.diff(c_a[::4]) / c_a[::4][:-1]
                    ret_b = np.diff(c_b[::4]) / c_b[::4][:-1]
                    corr = rolling_correlation(ret_a, ret_b, CORR_LOOKBACK_BARS)
                    pair = tuple(sorted([sym_a, sym_b]))
                    self.correlation_map[pair] = corr
