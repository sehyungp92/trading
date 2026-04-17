"""AKC-Helix Swing — async event-driven core engine."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

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
    ADD_4H_R,
    ADD_1H_R,
    ADD_MAX_BARS,
    ADD_MIN_BARS,
    ADD_OVERNIGHT_R,
    ADD_PRICE_GATE_ATR_MULT,
    ADD_RISK_FRAC,
    ADX_UPPER_GATE,
    BE_ATR1H_OFFSET,
    CATCHUP_OVERSHOOT_FRAC,
    CATCHUP_OVERSHOOT_OPEN_FRAC,
    CATCHUP_TTL_MIN,
    CLASS_B_BAIL_BARS,
    CLASS_B_BAIL_R_THRESH,
    CLASS_B_MIN_ADX,
    CLASS_B_MOM_LOOKBACK,
    CONSEC_STOPS_HALVE,
    DAILY_STOP_R,
    DISABLE_CLASS_A,
    EMA_4H_FAST,
    EMA_4H_SLOW,
    HIGH_VOL_PCT,
    PARTIAL_2P5_FRAC,
    PARTIAL_5_FRAC,
    PARTIAL_5_TRAIL_BONUS,
    R_BE,
    R_BE_1H,
    R_PARTIAL_2P5,
    R_PARTIAL_5,
    RESCUE_SLIP_FRAC,
    RESCUE_TTL_MIN,
    STALE_1H_BARS,
    STALE_4H_BARS,
    STALE_FLATTEN_R_FLOOR,
    STALE_R_THRESH,
    STALE_TIGHTEN_ATR_MULT,
    STOP_1H_STD,
    STRATEGY_ID,
    SYMBOL_CONFIGS,
    TRAIL_FADE_FLOOR,
    TRAIL_FADE_MIN_R,
    TRAIL_FADE_ONSET_BARS,
    TRAIL_FADE_PENALTY,
    TRAIL_PROFIT_DELAY_BARS,
    TRAIL_STALL_FLOOR,
    TRAIL_STALL_ONSET,
    TRAIL_STALL_RATE,
    TRAIL_TIMEDECAY_FLOOR,
    TRAIL_TIMEDECAY_ONSET,
    TRAIL_TIMEDECAY_RATE,
    TTL_1H_HOURS,
    TTL_4H_HOURS,
    TTL_ADD_HOURS,
    WEEKLY_STOP_R,
    PORTFOLIO_CAP_R,
    SymbolConfig,
)
from .indicators import (
    atr,
    compute_daily_state,
    compute_regime_4h,
    ema,
    macd,
    scan_pivots,
)
from .models import (
    CircuitBreakerState,
    DailyState,
    Direction,
    LegType,
    PivotStore,
    Regime,
    SetupClass,
    SetupInstance,
    SetupState,
    TFState,
)

logger = logging.getLogger(__name__)

from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _next_session_open_et(now_utc: datetime, start_hhmm: str = "03:00") -> datetime:
    """Compute the next session open time in ET, returned as UTC."""
    try:
        et_now = now_utc.astimezone(ET)
    except Exception:
        et_now = now_utc
    parts = start_hhmm.split(":")
    h, m = int(parts[0]), int(parts[1])
    candidate = et_now.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= et_now:
        candidate += timedelta(days=1)
    # Skip weekends
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def _next_daily_close_et(now_utc: datetime) -> datetime:
    """Compute the next daily close (16:00 ET), returned as UTC."""
    try:
        et_now = now_utc.astimezone(ET)
    except Exception:
        et_now = now_utc
    candidate = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
    if candidate <= et_now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


class HelixEngine:
    """Core AKC-Helix v2.0 event-driven engine."""

    # Cross-strategy coordination: size boost when ATRSS confirms same direction
    ATRSS_SIZE_BOOST: float = 1.25

    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        config: dict[str, SymbolConfig],
        trade_recorder: TradeRecorder | None = None,
        equity: float = 100_000.0,
        news_calendar: list[tuple[str, datetime]] | None = None,
        coordinator: Any = None,
        market_calendar: Any | None = None,
        instrumentation_kit: Any | None = None,
        equity_offset: float = 0.0,
        equity_alloc_pct: float = 1.0,
    ) -> None:
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = instruments
        self._config = config
        self._recorder = trade_recorder
        self._equity = equity
        self._equity_offset = equity_offset
        self._equity_alloc_pct = equity_alloc_pct
        self._news_calendar: list[tuple[str, datetime]] = news_calendar or []
        self._coordinator = coordinator
        self._market_cal = market_calendar
        self._kit = instrumentation_kit

        # Wire drawdown tracker with initial equity
        if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
            self._kit.ctx.drawdown_tracker.update_equity(self._equity)

        # Per-symbol state
        self.daily_states: dict[str, DailyState] = {}
        self.tf_states: dict[str, dict[str, TFState]] = {}    # sym → {"1H": ..., "4H": ...}
        self.pivots: dict[str, dict[str, PivotStore]] = {}     # sym → {"1H": ..., "4H": ...}
        self.regime_4h: dict[str, Regime] = {}                  # sym → Regime
        self.div_mag_history: dict[str, list[float]] = {}       # sym → list of div_mag_norm values
        self.active_setups: dict[str, SetupInstance] = {}       # setup_id → active
        self.pending_setups: dict[str, SetupInstance] = {}      # setup_id → armed
        self.queued_setups: dict[str, SetupInstance] = {}       # setup_id → queued (outside window)
        self.circuit_breakers: dict[str, CircuitBreakerState] = {}
        self.contracts: dict[str, Any] = {}                     # symbol → (Contract, spec)

        # Order tracking
        self._order_to_setup: dict[str, str] = {}   # oms_order_id → setup_id
        self._oca_counter: int = 0

        # Class B pivot dedup (per-symbol last pivot timestamps)
        self._last_b_long_l2_ts: dict[str, datetime | None] = {}
        self._last_b_short_h2_ts: dict[str, datetime | None] = {}

        # Class D pivot dedup (per-symbol last pivot timestamps)
        self._last_d_long_l2_ts: dict[str, datetime | None] = {}
        self._last_d_short_h2_ts: dict[str, datetime | None] = {}

        # USO regime tracking (2G)
        self._regime_streaks: dict[str, int] = {}
        self._prev_regimes: dict[str, Regime | None] = {}

        # Live market data tickers (populated via reqMktData)
        self._tickers: dict[str, Any] = {}
        self._risk_halted = False
        self._risk_halt_reason = ""
        # Signal evolution ring buffer for TA alpha decay detector
        self._signal_ring: dict[str, deque] = {}  # sym → deque of snapshots

        # Async tasks
        self._event_task: asyncio.Task | None = None
        self._cycle_task: asyncio.Task | None = None
        self._trigger_task: asyncio.Task | None = None
        self._timer_tasks: dict[str, asyncio.Task] = {}
        self._event_queue: asyncio.Queue | None = None
        self._running = False
        self._resubscribing = False

    # ------------------------------------------------------------------
    # Signal evolution tracking (for TA alpha decay detector)
    # ------------------------------------------------------------------

    def _snapshot_signal_state(self, sym: str, daily: Any,
                               regime_4h: str, setup_class: str = "",
                               div_mag_norm: float = 0.0, vol_factor: float = 1.0) -> None:
        """Capture signal components for evolution tracking."""
        ring = self._signal_ring.setdefault(sym, deque(maxlen=10))
        ring.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "adx": getattr(daily, "adx", None),
            "regime_4h": regime_4h,
            "setup_class": setup_class,
            "div_mag_norm": div_mag_norm,
            "vol_factor": vol_factor,
        })

    def _build_signal_evolution(self, sym: str, n: int = 5) -> list[dict]:
        """Return last N signal snapshots with bars_ago index."""
        ring = self._signal_ring.get(sym)
        if not ring:
            return []
        items = list(ring)[-n:]
        return [{"bars_ago": n - 1 - i, **s} for i, s in enumerate(items)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to events, load initial bar history, start hourly scheduler."""
        logger.info("Helix engine starting …")
        self._running = True

        # Subscribe to OMS events
        self._event_queue = self._oms.stream_events(STRATEGY_ID)
        self._event_task = asyncio.create_task(self._process_events())

        # Resolve contracts for each symbol
        for sym, cfg in self._config.items():
            if cfg.contract_expiry:
                try:
                    from libs.broker_ibkr.mapping.contract_factory import ContractFactory
                    cf: ContractFactory = getattr(self._ib, "_contract_factory", None)
                    if cf:
                        contract, spec = await cf.resolve(sym, cfg.contract_expiry)
                        self.contracts[sym] = (contract, spec)
                except Exception as e:
                    logger.warning("Could not resolve contract for %s: %s", sym, e)

        # Initialize per-symbol state containers
        for sym in self._config:
            self.tf_states.setdefault(sym, {"1H": TFState(tf_label="1H"), "4H": TFState(tf_label="4H")})
            self.pivots.setdefault(sym, {"1H": PivotStore(), "4H": PivotStore()})
            self.regime_4h.setdefault(sym, Regime.CHOP)
            self.circuit_breakers.setdefault(sym, CircuitBreakerState())

        # Subscribe to live market data for spread gate + trigger detection
        for sym in self._config:
            contract = self._get_contract(sym)
            if contract:
                try:
                    qualified = await self._ib.ib.qualifyContractsAsync(contract)
                    if not qualified:
                        logger.warning("Could not qualify contract for %s", sym)
                        continue
                    contract = qualified[0]
                    self._tickers[sym] = self._ib.ib.reqMktData(contract, '', False, False)
                except Exception as e:
                    logger.warning("Could not subscribe mkt data for %s: %s", sym, e)

        # Load initial bar history and compute initial states
        await self._load_initial_bars()

        # Register ticker-update callback as primary trigger detector
        self._ib.ib.pendingTickersEvent += self._on_ticker_update

        # Register farm-recovery handler for automatic market data resubscription
        self._ib.register_farm_recovery_callback("default", self._on_farm_recovery)

        # Start hourly cycle scheduler
        self._cycle_task = asyncio.create_task(self._hourly_scheduler())
        # Start fallback trigger monitor (15s safety net if ticker events lag)
        self._trigger_task = asyncio.create_task(self._trigger_monitor())
        # Start window-close scheduler (spec s1.2: cancel at 15:45 ET immediately)
        self._window_close_task: asyncio.Task | None = asyncio.create_task(self._window_close_scheduler())
        logger.info("Helix engine started for %s", list(self._config.keys()))

    async def stop(self) -> None:
        """Cancel all pending, cleanup."""
        logger.info("Helix engine stopping …")
        self._running = False

        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        if self._trigger_task:
            self._trigger_task.cancel()
            try:
                await self._trigger_task
            except asyncio.CancelledError:
                pass

        wc_task = getattr(self, '_window_close_task', None)
        if wc_task:
            wc_task.cancel()
            try:
                await wc_task
            except asyncio.CancelledError:
                pass

        # Cancel all timer tasks
        for task in self._timer_tasks.values():
            task.cancel()
        for task in self._timer_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._timer_tasks.clear()

        # Unregister ticker callback
        try:
            self._ib.ib.pendingTickersEvent -= self._on_ticker_update
        except Exception:
            pass

        # Unsubscribe market data
        for sym in list(self._tickers):
            try:
                contract = self._get_contract(sym)
                if contract:
                    self._ib.ib.cancelMktData(contract)
            except Exception:
                pass
        self._tickers.clear()

        # Cancel pending setups
        for setup_id, setup in list(self.pending_setups.items()):
            for oid in (setup.primary_order_id, setup.catchup_order_id, setup.rescue_order_id):
                if oid:
                    try:
                        await self._oms.submit_intent(
                            Intent(
                                intent_type=IntentType.CANCEL_ORDER,
                                strategy_id=STRATEGY_ID,
                                target_oms_order_id=oid,
                            )
                        )
                    except Exception as e:
                        logger.warning("Error cancelling order %s: %s", oid, e)

        logger.info("Helix engine stopped")

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

            try:
                self._ib.ib.pendingTickersEvent -= self._on_ticker_update
            except Exception:
                pass

            for sym in list(self._tickers):
                contract = self._get_contract(sym)
                if contract:
                    try:
                        self._ib.ib.cancelMktData(contract)
                    except Exception:
                        pass
            self._tickers.clear()

            await asyncio.sleep(1.0)

            for sym in self._config:
                contract = self._get_contract(sym)
                if contract:
                    try:
                        self._tickers[sym] = self._ib.ib.reqMktData(contract, '', False, False)
                    except Exception as e:
                        logger.warning("Resubscribe failed for %s: %s", sym, e)

            self._ib.ib.pendingTickersEvent += self._on_ticker_update
            logger.info("Market data resubscription complete: %d tickers", len(self._tickers))
        finally:
            self._resubscribing = False

    # ------------------------------------------------------------------
    # Hourly scheduler
    # ------------------------------------------------------------------

    async def _hourly_scheduler(self) -> None:
        """Sleep until the next hour boundary, then run the hourly cycle."""
        while self._running:
            now = datetime.now(timezone.utc)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
            wait_secs = (next_hour - now).total_seconds()
            logger.debug("Next cycle in %.0fs at %s", wait_secs, next_hour)
            await asyncio.sleep(wait_secs)
            if not self._running:
                break
            try:
                await self._hourly_cycle()
            except Exception:
                logger.exception("Error in hourly cycle")

    # ------------------------------------------------------------------
    # Window-close scheduler (spec s1.2: cancel at 15:45 ET immediately)
    # ------------------------------------------------------------------

    async def _window_close_scheduler(self) -> None:
        """Sleep until 15:45 ET each trading day, then cancel all unfilled orders."""
        while self._running:
            now = datetime.now(timezone.utc)
            try:
                et_now = now.astimezone(ET)
            except Exception:
                et_now = now
            # Compute next 15:45 ET
            target = et_now.replace(hour=15, minute=45, second=0, microsecond=0)
            if target <= et_now:
                target += timedelta(days=1)
            # Skip weekends
            while target.weekday() >= 5:
                target += timedelta(days=1)
            wait_secs = (target - et_now).total_seconds()
            logger.debug("Window-close scheduler: next cancel at %s (%.0fs)", target, wait_secs)
            await asyncio.sleep(wait_secs)
            if not self._running:
                break
            try:
                await self._cancel_all_unfilled("window_close_1545")
            except Exception:
                logger.exception("Error in window-close cancellation")

    async def _cancel_all_unfilled(self, reason: str) -> None:
        """Cancel all unfilled entry and add orders immediately (spec s1.2)."""
        for setup_id in list(self.pending_setups):
            setup = self.pending_setups[setup_id]
            await self._cancel_setup(setup, reason)
        # Also cancel queued setups
        for setup_id in list(self.queued_setups):
            del self.queued_setups[setup_id]
        logger.info("Window-close: cancelled all unfilled orders (%s)", reason)

    # ------------------------------------------------------------------
    # Core hourly cycle
    # ------------------------------------------------------------------

    async def _refresh_equity(self) -> None:
        """Fetch current account equity from IB."""
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                account = accounts[0]
                for item in self._ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == account:
                        raw = float(item.value)
                        if raw > 0:
                            new_equity = raw * self._equity_alloc_pct + self._equity_offset
                            self._equity = new_equity
                            if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
                                self._kit.ctx.drawdown_tracker.update_equity(new_equity)
                            logger.debug("Equity updated to $%.2f", new_equity)
                        return
        except Exception:
            logger.warning("Could not refresh equity from IB, using $%.2f", self._equity)

    async def _hourly_cycle(self) -> None:
        """Execute the hourly cycle per spec."""
        now = datetime.now(timezone.utc)
        logger.info("=== Helix hourly cycle %s ===", now.isoformat())

        # 1. Refresh equity
        await self._refresh_equity()

        # Determine if this is a 4H boundary
        is_4h_boundary = now.hour % 4 == 0

        # 2. Per-symbol update
        for sym in self._config:
            try:
                await self._cycle_symbol(sym, now, is_4h_boundary)
            except Exception:
                logger.exception("Error in cycle for %s", sym)

        # 3. Manage pending setups (TTL, invalidation, rescue)
        await self._manage_pending_setups(now)

        # 4. Manage active setups
        await self._manage_active_setups(now)

        # 5. Check queued setups (spec s1.2) — arm if window open + structure valid
        await self._process_queued_setups(now)

        # 6. Detect new setups
        all_candidates = self._detect_new_setups(now, is_4h_boundary)

        # 6b. Emit indicator snapshots for detected setups
        if self._kit and all_candidates:
            for _setup in all_candidates:
                _daily = self.daily_states.get(_setup.symbol)
                self._kit.on_indicator_snapshot(
                    pair=_setup.symbol,
                    indicators={
                        "ema_fast_d": _daily.ema_fast if _daily else 0,
                        "ema_slow_d": _daily.ema_slow if _daily else 0,
                        "atr_d": _daily.atr_d if _daily else 0,
                        "trend_strength": _daily.trend_strength if _daily else 0,
                        "adx": _daily.adx if _daily else 0,
                        "vol_factor": _daily.vol_factor if _daily else 1.0,
                        "div_mag_norm": getattr(_setup, "div_mag_norm", 0),
                    },
                    signal_name=f"helix_{_setup.setup_class.value.lower()}",
                    signal_strength=getattr(_setup, "div_mag_norm", 0.5),
                    decision="enter",
                    strategy_id="AKC_HELIX",
                    exchange_timestamp=now,
                )
                _snap = self._kit.capture_snapshot(_setup.symbol)
                if _snap:
                    self._kit.on_orderbook_context(
                        pair=_setup.symbol,
                        best_bid=_snap.get("bid", 0),
                        best_ask=_snap.get("ask", 0),
                        trade_context="signal_eval",
                        exchange_timestamp=now,
                    )

        # 7. Allocate accepted setups (or queue if outside window)
        if all_candidates:
            accepted = allocator.allocate(
                all_candidates,
                self.active_setups,
                self.daily_states,
                self._equity,
                self._instruments,
                self.circuit_breakers,
            )

            # Hook 3: Log allocator rejections as missed opportunities
            if self._kit:
                accepted_ids = {s.setup_id for s in accepted}
                for setup in all_candidates:
                    if setup.setup_id not in accepted_ids:
                        self._kit.log_missed(
                            pair=setup.symbol,
                            side="LONG" if setup.direction == Direction.LONG else "SHORT",
                            signal=setup.setup_class.value,
                            signal_id=setup.setup_id,
                            signal_strength=0.5,
                            blocked_by="allocator",
                            block_reason="rejected by portfolio allocator",
                            strategy_params={
                                "setup_class": setup.setup_class.value,
                                "origin_tf": setup.origin_tf,
                                "div_mag_norm": setup.div_mag_norm,
                                "adx": setup.adx_at_entry,
                                "regime_4h": setup.regime_4h_at_entry,
                                "vol_factor": setup.vol_factor_at_placement,
                                "size_mult": setup.setup_size_mult,
                            },
                        )

            for setup in accepted:
                cfg = self._config.get(setup.symbol)
                if cfg is None:
                    continue
                try:
                    now_et = now.astimezone(ET)
                except Exception:
                    now_et = now
                if gates.is_entry_window_open(now_et, cfg):
                    await self._arm_setup(setup, now)
                else:
                    # Queue for arming at next window open (spec s1.2)
                    setup.state = SetupState.NEW
                    self.queued_setups[setup.setup_id] = setup
                    logger.info("Queued %s %s (outside window)", setup.symbol, setup.setup_id[:8])

        # Hook 1: Market snapshot + regime classification (post-decision)
        if self._kit:
            for sym in self._config:
                self._kit.classify_regime(sym)
                self._kit.capture_snapshot(sym)

    async def _cycle_symbol(
        self, sym: str, now: datetime, is_4h_boundary: bool
    ) -> None:
        """Per-symbol: fetch bars, update indicators, scan pivots."""
        cfg = self._config[sym]

        # Check if daily bar boundary (UTC 00:xx)
        is_daily_boundary = now.hour == 0

        # Fetch and update 1H always
        bars_1h = await self._fetch_bars(sym, cfg, "1 hour", "30 D")
        if bars_1h is not None:
            self._update_tf_state(sym, "1H", bars_1h)

        # Fetch and update 4H on boundary
        if is_4h_boundary:
            bars_4h = await self._fetch_bars(sym, cfg, "4 hours", "60 D")
            if bars_4h is not None:
                self._update_tf_state(sym, "4H", bars_4h)
                # Update 4H regime (v2.0)
                self._update_regime_4h(sym, bars_4h)

        # Fetch and update daily on boundary
        if is_daily_boundary:
            bars_d = await self._fetch_bars(sym, cfg, "1 day", "200 D")
            if bars_d is not None:
                closes = np.array([b.close for b in bars_d], dtype=float)
                highs = np.array([b.high for b in bars_d], dtype=float)
                lows = np.array([b.low for b in bars_d], dtype=float)
                last_date = str(bars_d[-1].date) if bars_d else None
                prev = self.daily_states.get(sym)
                self.daily_states[sym] = compute_daily_state(
                    closes, highs, lows, prev, last_date,
                )
                if self._kit and len(closes) > 0:
                    self._kit.record_close(sym, float(closes[-1]))
                # Track regime streaks for USO gates (2G)
                new_regime = self.daily_states[sym].regime
                prev_regime = self._prev_regimes.get(sym)
                if prev_regime is not None and new_regime == prev_regime:
                    self._regime_streaks[sym] = self._regime_streaks.get(sym, 1) + 1
                else:
                    self._regime_streaks[sym] = 1
                self._prev_regimes[sym] = new_regime

    def _update_tf_state(
        self, sym: str, tf: str, bars: list[Any]
    ) -> None:
        """Update TFState and scan for new pivots on this timeframe."""
        if not bars:
            return

        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        bar_times = [
            datetime.fromisoformat(str(b.date)) if not isinstance(b.date, datetime)
            else b.date
            for b in bars
        ]

        # Compute indicators
        from .config import ATR_DAILY_PERIOD
        atr_arr = atr(highs, lows, closes, ATR_DAILY_PERIOD)
        line, sig, hist = macd(closes)

        # Update TFState
        tf_state = self.tf_states[sym][tf]
        tf_state.atr = float(atr_arr[-1])
        tf_state.macd_line = float(line[-1])
        tf_state.macd_signal = float(sig[-1])
        tf_state.macd_hist = float(hist[-1])
        tf_state.close = float(closes[-1])
        tf_state.bar_time = bar_times[-1] if bar_times else None

        # Rolling MACD history (last 50)
        tf_state.macd_line_history = [float(v) for v in line[-50:]]
        tf_state.macd_hist_history = [float(v) for v in hist[-50:]]

        # Rolling highs/lows for chandelier
        cfg = self._config[sym]
        lookback = max(cfg.chandelier_lookback, 30)
        tf_state.highs = [float(v) for v in highs[-lookback:]]
        tf_state.lows = [float(v) for v in lows[-lookback:]]

        # Scan for new pivots
        store = self.pivots[sym][tf]
        existing_count = len(store.highs) + len(store.lows)
        new_pivots = scan_pivots(highs, lows, line, hist, atr_arr, bar_times)

        # Only add pivots we haven't seen (after last known timestamp)
        last_h_ts = store.highs[-1].ts if store.highs else datetime.min.replace(tzinfo=timezone.utc)
        last_l_ts = store.lows[-1].ts if store.lows else datetime.min.replace(tzinfo=timezone.utc)
        for p in new_pivots:
            if p.kind.value == "H" and p.ts > last_h_ts:
                store.add(p)
            elif p.kind.value == "L" and p.ts > last_l_ts:
                store.add(p)

    def _update_regime_4h(self, sym: str, bars_4h: list[Any]) -> None:
        """Compute 4H regime from EMA(20)/EMA(50) on 4H close prices (v2.0)."""
        if not bars_4h or len(bars_4h) < EMA_4H_SLOW:
            return
        closes_4h = np.array([b.close for b in bars_4h], dtype=float)
        ema_f = ema(closes_4h, EMA_4H_FAST)
        ema_s = ema(closes_4h, EMA_4H_SLOW)
        self.regime_4h[sym] = compute_regime_4h(
            float(closes_4h[-1]), float(ema_f[-1]), float(ema_s[-1]),
        )

    # ------------------------------------------------------------------
    # Queued setup processing (spec s1.2)
    # ------------------------------------------------------------------

    async def _process_queued_setups(self, now: datetime) -> None:
        """Arm queued setups when entry window opens, if structure still valid."""
        for setup_id in list(self.queued_setups):
            setup = self.queued_setups[setup_id]
            cfg = self._config.get(setup.symbol)
            if cfg is None:
                continue

            try:
                now_et = now.astimezone(ET)
            except Exception:
                now_et = now

            # Check window open
            if not gates.is_entry_window_open(now_et, cfg):
                continue

            # Check structure still valid
            tf_key = "4H" if setup.origin_tf == "4H" else "1H"
            pivot_store = self.pivots.get(setup.symbol, {}).get(tf_key)
            if pivot_store and signals.is_structure_invalidated(setup, pivot_store):
                del self.queued_setups[setup_id]
                logger.info("Queued setup %s invalidated", setup.setup_id[:8])
                continue

            # Check no pause active
            cb = self.circuit_breakers.get(setup.symbol, CircuitBreakerState())
            if not gates.circuit_breaker_ok(cb, now):
                continue

            # Gap rule (spec s1.4): if price already beyond trigger by
            # more than 0.20 × ATR1H, skip the setup instance.
            tf1h = self.tf_states.get(setup.symbol, {}).get("1H")
            if tf1h:
                price_now = self._get_current_price(setup.symbol) or tf1h.close
                gap_overshoot_cap = 0.20 * tf1h.atr
                if setup.direction == Direction.LONG and price_now > setup.bos_level:
                    overshoot = price_now - setup.bos_level
                    if overshoot > gap_overshoot_cap:
                        del self.queued_setups[setup_id]
                        logger.info("Queued %s %s skipped: gap overshoot %.4f > cap %.4f",
                                    setup.symbol, setup.setup_id[:8], overshoot, gap_overshoot_cap)
                        continue
                elif setup.direction == Direction.SHORT and price_now < setup.bos_level:
                    overshoot = setup.bos_level - price_now
                    if overshoot > gap_overshoot_cap:
                        del self.queued_setups[setup_id]
                        logger.info("Queued %s %s skipped: gap overshoot %.4f > cap %.4f",
                                    setup.symbol, setup.setup_id[:8], overshoot, gap_overshoot_cap)
                        continue

            # Arm it (is_open_arm=True for gap-aware catch-up cap per spec s11.3)
            del self.queued_setups[setup_id]
            await self._arm_setup(setup, now, is_open_arm=True)

    # ------------------------------------------------------------------
    # Setup detection
    # ------------------------------------------------------------------

    def _detect_new_setups(
        self, now: datetime, is_4h_boundary: bool
    ) -> list[SetupInstance]:
        """Detect new setups across all symbols (Class A, B, C, D)."""
        candidates: list[SetupInstance] = []

        for sym, cfg in self._config.items():
            daily = self.daily_states.get(sym)
            if daily is None:
                continue

            tf1h = self.tf_states.get(sym, {}).get("1H")
            tf4h = self.tf_states.get(sym, {}).get("4H")
            pivots_1h = self.pivots.get(sym, {}).get("1H")
            pivots_4h = self.pivots.get(sym, {}).get("4H")

            if tf1h is None or pivots_1h is None:
                continue

            # Corridor inversion check (spec s10.2): if MinStop > corridor_cap,
            # disable 4H setups for this instrument for the day.
            _4h_disabled_by_corridor = False
            if daily.atr_d > 0:
                from .config import STOP_4H_MULT
                cap_dollars = 1.4 * daily.atr_d * (self._instruments.get(sym).point_value
                                                    if self._instruments.get(sym) else 1.0)
                if cfg.min_stop_floor_dollars > cap_dollars:
                    _4h_disabled_by_corridor = True

            # Skip if symbol already has pending or active setup
            sym_busy = any(
                s.symbol == sym and s.state in (
                    SetupState.ARMED, SetupState.TRIGGERED, SetupState.FILLED, SetupState.ACTIVE,
                )
                for s in list(self.pending_setups.values()) + list(self.active_setups.values())
            )
            if sym_busy:
                continue

            # ADX upper gate: skip all setups when ADX overextended
            # Snapshot signal state for evolution tracking
            self._snapshot_signal_state(
                sym, daily,
                regime_4h=str(self.regime_4h.get(sym, "unknown")),
            )

            if ADX_UPPER_GATE < 999 and daily.adx > ADX_UPPER_GATE:
                continue

            div_hist = self.div_mag_history.setdefault(sym, [])

            # Class A: 4H hidden divergence continuation (only on 4H boundary)
            if is_4h_boundary and tf4h is not None and pivots_4h is not None and not _4h_disabled_by_corridor and not DISABLE_CLASS_A:
                setup_4h = signals.detect_class_a(
                    sym, pivots_4h, daily, tf4h, cfg, div_hist, now,
                )
                if setup_4h is not None:
                    # USO gates (2G)
                    if sym == "USO":
                        # (a) Block counter-regime entries on USO
                        if (setup_4h.direction == Direction.LONG and daily.regime == Regime.BEAR) or \
                           (setup_4h.direction == Direction.SHORT and daily.regime == Regime.BULL):
                            continue
                        # (c) Regime stability gate: require 3+ consecutive regime days for counter-regime Class A
                        if (setup_4h.direction == Direction.LONG and daily.regime != Regime.BULL) or \
                           (setup_4h.direction == Direction.SHORT and daily.regime != Regime.BEAR):
                            if self._regime_streaks.get(sym, 0) < 3:
                                continue
                    div_hist.append(setup_4h.div_mag_norm)
                    candidates.append(setup_4h)
                    continue  # one setup per symbol per cycle

            # Class C: 4H classic divergence reversal (only on 4H boundary, gated)
            if is_4h_boundary and tf4h is not None and pivots_4h is not None and not _4h_disabled_by_corridor:
                setup_c = signals.detect_class_c(
                    sym, pivots_4h, daily, tf4h, cfg, div_hist, now,
                )
                if setup_c is not None:
                    # USO gate (2G-b): Disable Class C on USO
                    if sym == "USO":
                        continue
                    div_hist.append(setup_c.div_mag_norm)
                    candidates.append(setup_c)
                    continue  # one setup per symbol per cycle

            # Class B: 1H hidden divergence continuation (every hour)
            if not (daily.extreme_vol):
                setup_b = signals.detect_class_b(
                    sym, pivots_1h, daily, tf1h, cfg, div_hist, now,
                )
                if setup_b is not None:
                    # Quality filter: reject Class B in CHOP, counter-trend, or low ADX
                    _b_ok = True
                    if daily.regime == Regime.CHOP:
                        _b_ok = False
                    elif setup_b.direction == Direction.LONG and daily.regime == Regime.BEAR:
                        _b_ok = False
                    elif setup_b.direction == Direction.SHORT and daily.regime == Regime.BULL:
                        _b_ok = False
                    elif daily.adx < CLASS_B_MIN_ADX:
                        _b_ok = False
                    # Momentum gate: MACD line must trend in trade direction
                    if _b_ok and len(tf1h.macd_line_history) >= CLASS_B_MOM_LOOKBACK:
                        recent = tf1h.macd_line_history[-CLASS_B_MOM_LOOKBACK:]
                        if setup_b.direction == Direction.LONG:
                            if not all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                                _b_ok = False
                        else:
                            if not all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                                _b_ok = False
                    # Pivot dedup: skip if same L2/H2 pivot timestamp as last B detection
                    if _b_ok and setup_b.pivot_2 is not None:
                        if setup_b.direction == Direction.LONG:
                            if self._last_b_long_l2_ts.get(sym) == setup_b.pivot_2.ts:
                                _b_ok = False
                        else:
                            if self._last_b_short_h2_ts.get(sym) == setup_b.pivot_2.ts:
                                _b_ok = False
                    if _b_ok:
                        # Record pivot timestamp for dedup
                        if setup_b.pivot_2 is not None:
                            if setup_b.direction == Direction.LONG:
                                self._last_b_long_l2_ts[sym] = setup_b.pivot_2.ts
                            else:
                                self._last_b_short_h2_ts[sym] = setup_b.pivot_2.ts
                        div_hist.append(setup_b.div_mag_norm)
                        candidates.append(setup_b)
                        continue  # one setup per symbol per cycle

            # Class D: 1H momentum continuation (every hour)
            # Disabled in extreme vol (spec s6)
            if not (daily.extreme_vol):
                setup_1h = signals.detect_class_d(
                    sym, pivots_1h, daily, tf1h, cfg, now,
                )
                if setup_1h is not None:
                    # Pivot dedup for Class D (2M)
                    _d_ok = True
                    if setup_1h.pivot_2 is not None:
                        if setup_1h.direction == Direction.LONG:
                            if self._last_d_long_l2_ts.get(sym) == setup_1h.pivot_2.ts:
                                _d_ok = False
                        else:
                            if self._last_d_short_h2_ts.get(sym) == setup_1h.pivot_2.ts:
                                _d_ok = False
                    if _d_ok:
                        if setup_1h.pivot_2 is not None:
                            if setup_1h.direction == Direction.LONG:
                                self._last_d_long_l2_ts[sym] = setup_1h.pivot_2.ts
                            else:
                                self._last_d_short_h2_ts[sym] = setup_1h.pivot_2.ts
                        candidates.append(setup_1h)

        return candidates

    # ------------------------------------------------------------------
    # Arming setups
    # ------------------------------------------------------------------

    async def _arm_setup(self, setup: SetupInstance, now: datetime, is_open_arm: bool = False) -> None:
        """Run full eligibility gates, then place primary entry + conditional catch-up."""
        if self._risk_halted:
            logger.warning(
                "Helix arming suppressed while OMS risk halt is active: %s",
                self._risk_halt_reason or "unspecified",
            )
            return
        inst = self._instruments.get(setup.symbol)
        if inst is None:
            return
        cfg = self._config.get(setup.symbol)
        if cfg is None:
            return
        daily = self.daily_states.get(setup.symbol)
        if daily is None:
            return

        # Compute unit1 risk
        vf = daily.vol_factor
        setup.unit1_risk_dollars = allocator.compute_unit1_risk(
            self._equity, cfg.base_risk_pct, vf,
        )
        setup.vol_factor_at_placement = vf

        # Rule 2: size boost when ATRSS has a concurrent position on same symbol+direction
        effective_size_mult = setup.setup_size_mult
        if self._coordinator:
            direction_str = "LONG" if setup.direction == Direction.LONG else "SHORT"
            if self._coordinator.has_atrss_position(setup.symbol, direction_str):
                original_size_mult = effective_size_mult
                effective_size_mult *= self.ATRSS_SIZE_BOOST
                logger.info(
                    "COORD Rule 2: Boosting %s size by %.0f%% (ATRSS active same direction)",
                    setup.symbol, (self.ATRSS_SIZE_BOOST - 1) * 100,
                )
                self._coordinator.log_action(
                    action="size_boost",
                    trigger_strategy="ATRSS",
                    target_strategy="AKC_HELIX",
                    symbol=setup.symbol,
                    rule="rule_2",
                    details={"boost_factor": self.ATRSS_SIZE_BOOST,
                             "original_size_mult": original_size_mult,
                             "effective_size_mult": effective_size_mult,
                             "direction": direction_str},
                    outcome="applied",
                )

        # Position size
        if setup.qty_planned <= 0:
            setup.qty_planned = allocator.compute_position_size(
                setup.bos_level, setup.stop0,
                setup.unit1_risk_dollars, effective_size_mult,
                inst.point_value, cfg.max_contracts,
            )
        if setup.qty_planned <= 0:
            return

        # --- Run full eligibility gates before arming (spec s11.1) ---
        try:
            now_et = now.astimezone(ET)
        except Exception:
            now_et = now

        # Compute real-time spread from IBKR market data (spec s7.1)
        spread_ticks, spread_dollars, spread_bps = self._get_spread_info(setup.symbol, cfg)

        # Compute current risk tallies
        portfolio_r = 0.0
        instrument_r = 0.0
        pending_r = 0.0
        for s in list(self.active_setups.values()):
            if s.state not in (SetupState.FILLED, SetupState.ACTIVE):
                continue
            s_inst = self._instruments.get(s.symbol)
            if s_inst is None:
                continue
            s_daily = self.daily_states.get(s.symbol)
            s_vf = s_daily.vol_factor if s_daily else 1.0
            s_cfg = self._config.get(s.symbol)
            s_brp = s_cfg.base_risk_pct if s_cfg else cfg.base_risk_pct
            s_u1 = allocator.compute_unit1_risk(self._equity, s_brp, s_vf)
            if s_u1 > 0:
                r = allocator.compute_risk_r(
                    s.fill_price or s.bos_level,
                    s.current_stop or s.stop0,
                    s.qty_open or s.qty_planned,
                    s_inst.point_value, s_u1,
                )
                portfolio_r += r
                if s.symbol == setup.symbol:
                    instrument_r += r
        for s in list(self.pending_setups.values()):
            if s.state not in (SetupState.ARMED, SetupState.TRIGGERED):
                continue
            s_inst = self._instruments.get(s.symbol)
            if s_inst is None:
                continue
            s_daily = self.daily_states.get(s.symbol)
            s_vf = s_daily.vol_factor if s_daily else 1.0
            s_cfg = self._config.get(s.symbol)
            s_brp = s_cfg.base_risk_pct if s_cfg else cfg.base_risk_pct
            s_u1 = allocator.compute_unit1_risk(self._equity, s_brp, s_vf)
            if s_u1 > 0:
                r = allocator.compute_risk_r(
                    s.bos_level, s.stop0, s.qty_planned,
                    s_inst.point_value, s_u1,
                )
                pending_r += r

        cb = self.circuit_breakers.get(setup.symbol, CircuitBreakerState())
        ok, reason = gates.full_eligibility_check(
            setup, now_et, daily, cfg, spread_ticks,
            portfolio_r, pending_r, instrument_r, cb,
            self._news_calendar,
            {**self.active_setups, **self.pending_setups},
            inst.point_value,
            self.tf_states.get(setup.symbol, {}).get("1H", TFState()).atr,
            spread_dollars=spread_dollars,
            spread_bps=spread_bps,
        )

        # Emit filter decision for full eligibility gate
        if self._kit:
            self._kit.on_filter_decision(
                pair=setup.symbol, filter_name="full_eligibility_check",
                passed=ok, threshold=0.0, actual_value=0.0,
                signal_name=f"helix_{setup.setup_class.value.lower()}",
                strategy_id="AKC_HELIX",
            )

        if not ok:
            logger.info("Setup %s %s blocked by gate: %s", setup.symbol, setup.setup_id[:8], reason)
            return

        # Record gate decisions for telemetry
        setup.gate_decisions = [
            {"filter_name": "spread_gate", "threshold": getattr(cfg, 'max_spread_ticks', 0),
             "actual_value": spread_ticks, "passed": True,
             "margin_pct": round((getattr(cfg, 'max_spread_ticks', spread_ticks) - spread_ticks) / max(getattr(cfg, 'max_spread_ticks', 1), 0.01) * 100, 1)},
            {"filter_name": "heat_cap", "threshold": PORTFOLIO_CAP_R,
             "actual_value": round(portfolio_r + pending_r, 2), "passed": True,
             "margin_pct": round((PORTFOLIO_CAP_R - (portfolio_r + pending_r)) / max(PORTFOLIO_CAP_R, 0.01) * 100, 1)},
        ]

        # Assign OCA group
        self._oca_counter += 1
        setup.oca_group = f"HELIX_{setup.symbol}_{self._oca_counter}"

        # R price (dollars per unit of risk)
        setup.r_price = abs(setup.bos_level - setup.stop0)

        # Min R-price gate (2K): reject setups with tiny risk range
        min_r_price = 0.003 * setup.bos_level
        if setup.r_price < min_r_price:
            return

        # TTL
        if setup.origin_tf == "4H":
            setup.expiry_ts = now + timedelta(hours=TTL_4H_HOURS)
        else:
            setup.expiry_ts = now + timedelta(hours=TTL_1H_HOURS)

        # Side
        side = OrderSide.BUY if setup.direction == Direction.LONG else OrderSide.SELL

        tick = cfg.tick_size

        if setup.direction == Direction.LONG:
            trigger = round_to_tick(setup.bos_level, tick, "up")
        else:
            trigger = round_to_tick(setup.bos_level, tick, "down")

        risk_ctx = RiskContext(
            stop_for_risk=setup.stop0,
            planned_entry_price=trigger,
            risk_dollars=RiskCalculator.compute_order_risk_dollars(
                trigger, setup.stop0, setup.qty_planned, inst.point_value,
            ),
        )

        # ETF: Stop-Market (spec s11.1); Futures: Stop-Limit
        if cfg.is_etf:
            primary_order = OMSOrder(
                strategy_id=STRATEGY_ID,
                instrument=inst,
                side=side,
                qty=setup.qty_planned,
                order_type=OrderType.STOP,
                stop_price=trigger,
                tif="GTC",
                role=OrderRole.ENTRY,
                entry_policy=EntryPolicy(
                    ttl_seconds=int((setup.expiry_ts - now).total_seconds()),
                ),
                risk_context=risk_ctx,
                oca_group=setup.oca_group,
                oca_type=1,
            )
        else:
            # Futures: adaptive offset selection
            if daily.vol_pct > HIGH_VOL_PCT:
                offset_ticks = cfg.offset_wide_ticks
            else:
                offset_ticks = cfg.offset_tight_ticks
            setup.offset_ticks_at_placement = offset_ticks
            limit_offset = offset_ticks * tick
            if setup.direction == Direction.LONG:
                limit_price = trigger + limit_offset
            else:
                limit_price = trigger - limit_offset
            primary_order = OMSOrder(
                strategy_id=STRATEGY_ID,
                instrument=inst,
                side=side,
                qty=setup.qty_planned,
                order_type=OrderType.STOP_LIMIT,
                stop_price=trigger,
                limit_price=limit_price,
                tif="GTC",
                role=OrderRole.ENTRY,
                entry_policy=EntryPolicy(
                    ttl_seconds=int((setup.expiry_ts - now).total_seconds()),
                ),
                risk_context=risk_ctx,
                oca_group=setup.oca_group,
                oca_type=1,
            )

        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID,
                order=primary_order,
            )
        )
        if receipt.oms_order_id:
            setup.primary_order_id = receipt.oms_order_id
            self._order_to_setup[receipt.oms_order_id] = setup.setup_id

            if self._kit:
                self._kit.on_order_event(
                    order_id=receipt.oms_order_id,
                    pair=setup.symbol,
                    side="LONG" if setup.direction == Direction.LONG else "SHORT",
                    order_type="STOP_LIMIT",
                    status="SUBMITTED",
                    requested_qty=float(setup.qty_planned),
                    requested_price=setup.bos_level,
                    strategy_id=STRATEGY_ID,
                )

        # Conditional catch-up LIMIT (spec s11.4): only if price already broke BoS
        tf1h = self.tf_states.get(setup.symbol, {}).get("1H")
        price_now = tf1h.close if tf1h else 0.0
        already_broke = False
        if setup.direction == Direction.LONG and price_now > trigger:
            already_broke = True
        elif setup.direction == Direction.SHORT and price_now < trigger:
            already_broke = True

        if already_broke:
            overshoot = abs(price_now - trigger)
            # Spec s11.3: 0.20 at open re-arming, 0.15 intraday
            frac = CATCHUP_OVERSHOOT_OPEN_FRAC if is_open_arm else CATCHUP_OVERSHOOT_FRAC
            overshoot_cap = frac * (tf1h.atr if tf1h else setup.r_price)
            if 0 < overshoot <= overshoot_cap:
                if setup.direction == Direction.LONG:
                    catchup_limit = round_to_tick(price_now + 2 * tick, tick, "up")
                else:
                    catchup_limit = round_to_tick(price_now - 2 * tick, tick, "down")

                setup.catchup_expiry_ts = now + timedelta(minutes=CATCHUP_TTL_MIN)

                catchup_order = OMSOrder(
                    strategy_id=STRATEGY_ID,
                    instrument=inst,
                    side=side,
                    qty=setup.qty_planned,
                    order_type=OrderType.LIMIT,
                    limit_price=catchup_limit,
                    tif="GTC",
                    role=OrderRole.ENTRY,
                    entry_policy=EntryPolicy(
                        ttl_seconds=CATCHUP_TTL_MIN * 60,
                    ),
                    risk_context=RiskContext(
                        stop_for_risk=setup.stop0,
                        planned_entry_price=catchup_limit,
                        risk_dollars=RiskCalculator.compute_order_risk_dollars(
                            catchup_limit, setup.stop0, setup.qty_planned, inst.point_value,
                        ),
                    ),
                    oca_group=setup.oca_group,
                    oca_type=1,
                )

                cu_receipt = await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.NEW_ORDER,
                        strategy_id=STRATEGY_ID,
                        order=catchup_order,
                    )
                )
                if cu_receipt.oms_order_id:
                    setup.catchup_order_id = cu_receipt.oms_order_id
                    self._order_to_setup[cu_receipt.oms_order_id] = setup.setup_id
                    self._schedule_order_ttl(setup, "catchup", CATCHUP_TTL_MIN * 60)

        setup.state = SetupState.ARMED
        setup.armed_ts = now
        self.pending_setups[setup.setup_id] = setup

        logger.info(
            "ARMED %s %s %s %s qty=%d trigger=%.4f stop=%.4f",
            setup.symbol, setup.setup_class.value,
            "LONG" if setup.direction == Direction.LONG else "SHORT",
            setup.setup_id[:8], setup.qty_planned, trigger, setup.stop0,
        )

    # ------------------------------------------------------------------
    # Pending setup management
    # ------------------------------------------------------------------

    async def _manage_pending_setups(self, now: datetime) -> None:
        """Manage TTL, invalidation, rescue, catch-up TTL, window-close."""
        for setup_id in list(self.pending_setups):
            setup = self.pending_setups[setup_id]
            cfg = self._config.get(setup.symbol)
            if cfg is None:
                continue

            # TTL expiry
            if setup.expiry_ts and now >= setup.expiry_ts:
                await self._cancel_setup(setup, "ttl_expired")
                continue

            # Catch-up TTL
            if setup.catchup_order_id and setup.catchup_expiry_ts and now >= setup.catchup_expiry_ts:
                try:
                    await self._oms.submit_intent(
                        Intent(
                            intent_type=IntentType.CANCEL_ORDER,
                            strategy_id=STRATEGY_ID,
                            target_oms_order_id=setup.catchup_order_id,
                        )
                    )
                    self._order_to_setup.pop(setup.catchup_order_id, None)
                    setup.catchup_order_id = ""
                except Exception as e:
                    logger.warning("Error cancelling catch-up %s: %s", setup.catchup_order_id, e)

            # Structure invalidation
            tf_key = "4H" if setup.origin_tf == "4H" else "1H"
            pivot_store = self.pivots.get(setup.symbol, {}).get(tf_key)
            if pivot_store and signals.is_structure_invalidated(setup, pivot_store):
                await self._cancel_setup(setup, "structure_invalidated")
                continue

            # Window close cancellation
            try:
                et = now.astimezone(ET)
            except Exception:
                et = now
            if not gates.is_entry_window_open(et, cfg):
                await self._cancel_setup(setup, "window_closed")
                continue

            # Rescue check (spec s11.5): 5 min after trigger
            if (
                setup.state == SetupState.TRIGGERED
                and setup.triggered_ts
                and now >= setup.triggered_ts + timedelta(minutes=5)
                and not setup.rescue_order_id
            ):
                await self._maybe_rescue(setup, now)

            # End-of-bar backstop (spec s11.6): cancel if triggered but not
            # filled by close of next 1H bar (i.e., >1 hour after trigger)
            if (
                setup.state == SetupState.TRIGGERED
                and setup.triggered_ts
                and now >= setup.triggered_ts + timedelta(hours=1)
            ):
                await self._cancel_setup(setup, "end_of_bar_backstop")

    async def _cancel_setup(self, setup: SetupInstance, reason: str) -> None:
        """Cancel all orders and timers for a pending setup."""
        self._cancel_setup_timers(setup.setup_id)
        for oid in (setup.primary_order_id, setup.catchup_order_id, setup.rescue_order_id):
            if oid:
                try:
                    await self._oms.submit_intent(
                        Intent(
                            intent_type=IntentType.CANCEL_ORDER,
                            strategy_id=STRATEGY_ID,
                            target_oms_order_id=oid,
                        )
                    )
                    self._order_to_setup.pop(oid, None)
                except Exception as e:
                    logger.warning("Error cancelling order %s: %s", oid, e)

        setup.state = SetupState.CANCELLED
        self.pending_setups.pop(setup.setup_id, None)
        logger.info("Cancelled setup %s %s: %s", setup.symbol, setup.setup_id[:8], reason)

    async def _maybe_rescue(self, setup: SetupInstance, now: datetime) -> None:
        """Place rescue LIMIT order after primary trigger + 5 min."""
        if self._risk_halted:
            logger.warning(
                "Helix rescue suppressed for %s while OMS risk halt is active: %s",
                setup.symbol,
                self._risk_halt_reason or "unspecified",
            )
            return
        inst = self._instruments.get(setup.symbol)
        if inst is None:
            return
        cfg = self._config.get(setup.symbol)
        if cfg is None:
            return

        # Cancel primary
        if setup.primary_order_id:
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=setup.primary_order_id,
                    )
                )
                self._order_to_setup.pop(setup.primary_order_id, None)
                setup.primary_order_id = ""
            except Exception as e:
                logger.warning("Error cancelling primary for rescue: %s", e)

        # Teleport check (spec s11.5): price must not be too far from trigger
        # Distance limit = teleport_offset_mult × entry offset (not corridor)
        tf1h = self.tf_states.get(setup.symbol, {}).get("1H")
        if tf1h is None:
            return
        price_now = self._get_current_price(setup.symbol) or tf1h.close
        ot = setup.offset_ticks_at_placement or cfg.offset_tight_ticks
        entry_offset = ot * cfg.tick_size
        teleport_dist = cfg.teleport_offset_mult * entry_offset
        if setup.direction == Direction.LONG:
            if price_now > setup.bos_level + teleport_dist:
                await self._cancel_setup(setup, "teleport_too_far")
                return
        else:
            if price_now < setup.bos_level - teleport_dist:
                await self._cancel_setup(setup, "teleport_too_far")
                return

        # Slippage check (spec s11.5): move from trigger ≤ 0.5 × offset
        slip = abs(price_now - setup.bos_level)
        if entry_offset > 0 and slip > RESCUE_SLIP_FRAC * entry_offset:
            await self._cancel_setup(setup, "rescue_slip_exceeded")
            return

        # Place rescue LIMIT (±2 ticks per spec s11.5)
        side = OrderSide.BUY if setup.direction == Direction.LONG else OrderSide.SELL
        tick = cfg.tick_size
        if setup.direction == Direction.LONG:
            rescue_limit = round_to_tick(price_now + 2 * tick, tick, "up")
        else:
            rescue_limit = round_to_tick(price_now - 2 * tick, tick, "down")

        setup.rescue_expiry_ts = now + timedelta(minutes=RESCUE_TTL_MIN)

        rescue_order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=side,
            qty=setup.qty_planned,
            order_type=OrderType.LIMIT,
            limit_price=rescue_limit,
            tif="GTC",
            role=OrderRole.ENTRY,
            entry_policy=EntryPolicy(ttl_seconds=RESCUE_TTL_MIN * 60),
            risk_context=RiskContext(
                stop_for_risk=setup.stop0,
                planned_entry_price=rescue_limit,
                risk_dollars=RiskCalculator.compute_order_risk_dollars(
                    rescue_limit, setup.stop0, setup.qty_planned, inst.point_value,
                ),
            ),
            oca_group=setup.oca_group,
            oca_type=1,
        )

        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID,
                order=rescue_order,
            )
        )
        if receipt.oms_order_id:
            setup.rescue_order_id = receipt.oms_order_id
            self._order_to_setup[receipt.oms_order_id] = setup.setup_id
            self._schedule_order_ttl(setup, "rescue", RESCUE_TTL_MIN * 60)
            logger.info("Rescue placed for %s %s at %.4f", setup.symbol, setup.setup_id[:8], rescue_limit)

    # ------------------------------------------------------------------
    # Active setup management (spec s13, s14, s15)
    # ------------------------------------------------------------------

    async def _manage_active_setups(self, now: datetime) -> None:
        """Manage active positions: stops, partials, trailing, stale, adds."""
        for setup_id in list(self.active_setups):
            setup = self.active_setups[setup_id]
            try:
                # Timeout safety: force-close CLOSING setups after 120s
                if setup.state == SetupState.CLOSING:
                    flatten_reason = getattr(setup, '_flatten_reason', 'FLATTEN')
                    closing_since = getattr(setup, '_closing_since', None)
                    if closing_since and (now - closing_since).total_seconds() > 120:
                        logger.warning(
                            "CLOSING timeout %s %s -- forcing CLOSED after 120s",
                            setup.symbol, setup.setup_id[:8])
                        setup.state = SetupState.CLOSED
                        self.active_setups.pop(setup_id, None)
                    continue
                await self._manage_active(setup, now)
            except Exception:
                logger.exception("Error managing active setup %s", setup_id)

    async def _manage_active(self, setup: SetupInstance, now: datetime) -> None:
        """Per-setup management: R calc, BE, partials, trailing, stale."""
        # Skip management for setups awaiting flatten fill
        if setup.state == SetupState.CLOSING:
            return
        cfg = self._config.get(setup.symbol)
        if cfg is None:
            return
        inst = self._instruments.get(setup.symbol)
        if inst is None:
            return

        tf1h = self.tf_states.get(setup.symbol, {}).get("1H")
        daily = self.daily_states.get(setup.symbol)
        if tf1h is None or daily is None:
            return

        # Increment bars held
        setup.bars_held_1h += 1
        if now.hour % 4 == 0:
            setup.bars_held_4h += 1

        # Compute current R (spec s13.1)
        if setup.r_price <= 0:
            return
        if setup.direction == Direction.LONG:
            r_now = (tf1h.close - setup.fill_price) / setup.r_price
        else:
            r_now = (setup.fill_price - tf1h.close) / setup.r_price

        # R_state includes realized PnL from partials (spec s13.1)
        pv = inst.point_value
        unrealized = (tf1h.close - setup.fill_price) * pv * setup.qty_open if setup.direction == Direction.LONG \
            else (setup.fill_price - tf1h.close) * pv * setup.qty_open
        r_state = (setup.realized_pnl + unrealized) / setup.unit1_risk_dollars if setup.unit1_risk_dollars > 0 else r_now

        # Catastrophic loss protection: flatten if loss exceeds -2R
        if r_now < -2.0:
            logger.warning("%s catastrophic cap: R=%.2f < -2.0, flattening", setup.symbol, r_now)
            await self._flatten_setup(setup, reason="CATASTROPHIC")
            return

        # Class B bail trigger (2H): exit early if underwater after CLASS_B_BAIL_BARS
        if (setup.setup_class == SetupClass.CLASS_B
                and setup.bars_held_1h >= CLASS_B_BAIL_BARS
                and r_now < CLASS_B_BAIL_R_THRESH
                and not setup.trail_active):
            logger.info("%s Class B bail: %d bars, R=%.2f < %.1f",
                        setup.symbol, setup.bars_held_1h, r_now, CLASS_B_BAIL_R_THRESH)
            await self._flatten_setup(setup, reason="CLASS_B_BAIL")
            return

        new_stop = setup.current_stop

        # Track peak MFE in R for stalled-winner detection
        if r_now > setup.mfe_r_peak:
            setup.mfe_r_peak = r_now
            setup.bar_of_max_mfe = setup.bars_held_1h

        # Track trough MAE in R for adverse-excursion analysis
        if r_now < setup.mae_r_trough:
            setup.mae_r_trough = r_now

        # Per-class BE threshold: R_BE for 4H origin, R_BE_1H for 1H origin
        be_threshold = R_BE_1H if setup.origin_tf == "1H" else R_BE

        # +BE → move stop to breakeven (spec s13.2)
        if r_now >= be_threshold and not setup.trail_active:
            be_stop = stops.compute_be_stop(
                setup.direction, setup.fill_price, tf1h.atr, cfg.tick_size,
            )
            if setup.direction == Direction.LONG and be_stop > new_stop:
                new_stop = be_stop
                setup.stop_source = "BE"
                logger.info("%s BE triggered at %.2fR → stop %.4f", setup.symbol, r_now, be_stop)
            elif setup.direction == Direction.SHORT and be_stop < new_stop:
                new_stop = be_stop
                setup.stop_source = "BE"
                logger.info("%s BE triggered at %.2fR → stop %.4f", setup.symbol, r_now, be_stop)

        # +2.5R → partial 50% (spec s13.3)
        if r_now >= R_PARTIAL_2P5 and not setup.partial_2p5_done:
            partial_qty = max(1, int(setup.qty_open * PARTIAL_2P5_FRAC))
            await self._partial_exit(setup, partial_qty)
            setup.partial_2p5_done = True
            # Ratchet stop
            ratchet = stops.compute_ratchet_stop(
                setup.direction, setup.fill_price, setup.r_price, cfg.tick_size,
            )
            if setup.direction == Direction.LONG and ratchet > new_stop:
                new_stop = ratchet
            elif setup.direction == Direction.SHORT and ratchet < new_stop:
                new_stop = ratchet

        # +5R → partial 25% + trail bonus (spec s13.4)
        if r_now >= R_PARTIAL_5 and not setup.partial_5_done:
            partial_qty = max(1, int(setup.qty_open * PARTIAL_5_FRAC))
            await self._partial_exit(setup, partial_qty)
            setup.partial_5_done = True
            setup.trailing_mult_bonus += PARTIAL_5_TRAIL_BONUS

        # Track bars at or above BE threshold for trailing profit delay (cumulative)
        if r_now >= be_threshold:
            setup.bars_at_r1 += 1

        # Track bars with negative AND declining histogram (matches backtest)
        hist_list = tf1h.macd_hist_history
        if len(hist_list) >= 2:
            if setup.direction == Direction.LONG:
                fading = tf1h.macd_hist < 0 and hist_list[-1] < hist_list[-2]
            else:
                fading = tf1h.macd_hist > 0 and hist_list[-1] > hist_list[-2]
            if fading:
                setup.bars_neg_fading_hist += 1
            else:
                setup.bars_neg_fading_hist = 0

        # Trailing chandelier (activate after profit delay at BE threshold)
        if r_now >= be_threshold and setup.bars_at_r1 >= TRAIL_PROFIT_DELAY_BARS:
            setup.trail_active = True
            # Momentum hold: only activate when R_state > 2 (spec s14.2)
            momentum_strong = False
            if r_state > 2.0 and len(tf1h.macd_line_history) >= 6:
                momentum_strong = stops.is_momentum_strong(
                    tf1h.macd_line, tf1h.macd_line_history[-6], tf1h.macd_hist,
                    direction=setup.direction,
                )

            # Regime deterioration (spec s14.4): transition-based, not static
            regime_deteriorated = False
            if setup.regime_at_entry is not None:
                was_aligned = (
                    (setup.direction == Direction.LONG and setup.regime_at_entry == "BULL")
                    or (setup.direction == Direction.SHORT and setup.regime_at_entry == "BEAR")
                )
                if was_aligned and daily.regime == Regime.CHOP:
                    regime_deteriorated = True

            # Regime flip: daily regime opposes position direction
            regime_flipped = (
                (setup.direction == Direction.LONG and daily.regime == Regime.BEAR)
                or (setup.direction == Direction.SHORT and daily.regime == Regime.BULL)
            )

            trail_mult = stops.compute_trailing_mult(
                r_state, momentum_strong, regime_deteriorated, regime_flipped,
                setup.trailing_mult_bonus,
            )

            # Momentum fade tightening
            if setup.bars_neg_fading_hist >= TRAIL_FADE_ONSET_BARS and r_state > TRAIL_FADE_MIN_R:
                trail_mult = max(TRAIL_FADE_FLOOR, trail_mult - TRAIL_FADE_PENALTY)

            # Time-decay trailing: after onset bars at +1R, tighten per bar
            if setup.bars_at_r1 > TRAIL_TIMEDECAY_ONSET:
                decay = (setup.bars_at_r1 - TRAIL_TIMEDECAY_ONSET) * TRAIL_TIMEDECAY_RATE
                trail_mult = max(TRAIL_TIMEDECAY_FLOOR, trail_mult - decay)

            # Stalled winner decay: profitable but no new MFE for onset+ bars
            bars_since_peak = setup.bars_held_1h - setup.bar_of_max_mfe
            if r_state > 0.5 and bars_since_peak >= TRAIL_STALL_ONSET:
                stall_decay = min(1.0, bars_since_peak * TRAIL_STALL_RATE)
                trail_mult = max(TRAIL_STALL_FLOOR, trail_mult - stall_decay)

            chandelier = stops.compute_chandelier_stop(
                setup.direction, tf1h.highs, tf1h.lows,
                cfg.chandelier_lookback, tf1h.atr, trail_mult, cfg.tick_size,
            )
            # Determine trail source for exit reason granularity
            _trail_source = "TRAIL"
            if bars_since_peak >= TRAIL_STALL_ONSET and r_state > 0.5:
                _trail_source = "TRAIL_STALL"
            if setup.direction == Direction.LONG and chandelier > new_stop:
                new_stop = chandelier
                setup.stop_source = _trail_source
            elif setup.direction == Direction.SHORT and chandelier < new_stop:
                new_stop = chandelier
                setup.stop_source = _trail_source

        # Class C min hold: prevent premature exits before reversal develops
        class_c_min_hold = setup.setup_class == SetupClass.CLASS_C and setup.bars_held_1h < 12

        # Regime flip exit (2I): flatten when daily regime opposes position direction
        if daily and not class_c_min_hold:
            if ((setup.direction == Direction.LONG and daily.regime == Regime.BEAR)
                    or (setup.direction == Direction.SHORT and daily.regime == Regime.BULL)):
                await self._flatten_setup(setup, reason="BIAS_FLIP")
                return

        # Early stale: if 20+ bars and trail never activated, flatten losers
        if setup.bars_held_1h >= 20 and not setup.trail_active and r_state < 0 and not class_c_min_hold:
            logger.info("%s early stale flatten: %d bars, R_state=%.2f, trail never activated",
                        setup.symbol, setup.bars_held_1h, r_state)
            await self._flatten_setup(setup, reason="EARLY_STALE")
            return

        # Stale management (simplified to match backtest)
        stale_bars = STALE_1H_BARS if setup.origin_tf == "1H" else STALE_4H_BARS
        bars_held = setup.bars_held_1h if setup.origin_tf == "1H" else setup.bars_held_4h
        if bars_held >= stale_bars and r_state < STALE_R_THRESH and not class_c_min_hold:
            if r_state >= STALE_FLATTEN_R_FLOOR:
                if not setup.trail_active:
                    await self._flatten_setup(setup, reason="STALE")
                    return
                # trail active: let it tighten naturally
            else:
                await self._flatten_setup(setup, reason="STALE")
                return

        # Update stop if changed (never loosen)
        if new_stop != setup.current_stop:
            safe = False
            if setup.direction == Direction.LONG and new_stop > setup.current_stop:
                safe = True
            elif setup.direction == Direction.SHORT and new_stop < setup.current_stop:
                safe = True
            if safe:
                await self._update_stop(setup, new_stop)

        # Add-on check (spec s15, change #5: time+R trigger)
        if setup.add_allowed and not setup.add_done and setup.qty_open > 0:
            min_r = ADD_4H_R if setup.origin_tf == "4H" else ADD_1H_R
            # Teleport penalty (spec s11.2): add delayed to +2R
            if setup.add_min_r_override > 0:
                min_r = max(min_r, setup.add_min_r_override)
            # Change #5: bars window — not too early, not too late
            bars_ok = ADD_MIN_BARS <= setup.bars_held_1h <= ADD_MAX_BARS
            if r_now >= min_r and bars_ok:
                await self._try_add(setup, now)

        # Overnight add-risk rule (spec s15.4)
        # ETF: at 15:40 ET, flatten add if R < 2.0
        # Futures: at 16:25 ET, flatten add if R < 2.0
        if setup.add_done and setup.qty_open > setup.fill_qty:
            try:
                now_et = now.astimezone(ET)
            except Exception:
                now_et = now
            if cfg.is_etf:
                add_close_check = (now_et.hour == 15 and now_et.minute >= 40)
            else:
                add_close_check = (now_et.hour == 16 and now_et.minute >= 25)
            if add_close_check and r_now < ADD_OVERNIGHT_R:
                # Flatten add unit only
                add_qty_open = setup.qty_open - setup.fill_qty
                if add_qty_open > 0:
                    await self._partial_exit(setup, add_qty_open)
                    logger.info("%s overnight add flatten: R=%.2f < %.1f",
                                setup.symbol, r_now, ADD_OVERNIGHT_R)

    async def _try_add(self, setup: SetupInstance, now: datetime) -> None:
        """Try to place an add-on entry (spec s15.2)."""
        if self._risk_halted:
            logger.warning(
                "Helix add suppressed for %s while OMS risk halt is active: %s",
                setup.symbol,
                self._risk_halt_reason or "unspecified",
            )
            return
        cfg = self._config.get(setup.symbol)
        if cfg is None:
            return
        daily = self.daily_states.get(setup.symbol)
        if daily is None:
            return

        # Time gates (spec s15.1): entry window open + before 15:30/15:00 ET
        try:
            now_et = now.astimezone(ET)
        except Exception:
            now_et = now
        if not gates.is_entry_window_open(now_et, cfg):
            return
        add_cutoff_h, add_cutoff_m = (15, 30) if not cfg.is_etf else (15, 0)
        if now_et.hour > add_cutoff_h or (now_et.hour == add_cutoff_h and now_et.minute > add_cutoff_m):
            return
        if gates.is_news_blocked(now_et, setup.symbol, self._news_calendar):
            return

        pivots_1h = self.pivots.get(setup.symbol, {}).get("1H")
        tf1h = self.tf_states.get(setup.symbol, {}).get("1H")
        if pivots_1h is None or tf1h is None:
            return

        inst = self._instruments.get(setup.symbol)
        if inst is None:
            return

        # Try structural confirmation first (spec s15.2)
        add = signals.detect_add_setup(
            setup.symbol, setup.direction, pivots_1h, tf1h,
            setup.bos_level, cfg, daily, now,
        )

        add_risk = ADD_RISK_FRAC * setup.unit1_risk_dollars
        side = OrderSide.BUY if setup.direction == Direction.LONG else OrderSide.SELL
        tick = cfg.tick_size

        if add is not None:
            # Structural confirmation found — use pivot-based entry
            risk_per_contract = abs(add.bos_level - add.stop0) * inst.point_value
            if risk_per_contract <= 0:
                return
            add_qty = max(1, int(add_risk / risk_per_contract))

            if setup.direction == Direction.LONG:
                trigger = round_to_tick(add.bos_level, tick, "up")
                limit_price = trigger + cfg.offset_tight_ticks * tick
            else:
                trigger = round_to_tick(add.bos_level, tick, "down")
                limit_price = trigger - cfg.offset_tight_ticks * tick
            add_stop = add.stop0

            add_order = OMSOrder(
                strategy_id=STRATEGY_ID,
                instrument=inst,
                side=side,
                qty=add_qty,
                order_type=OrderType.STOP_LIMIT,
                stop_price=trigger,
                limit_price=limit_price,
                tif="GTC",
                role=OrderRole.ENTRY,
                entry_policy=EntryPolicy(ttl_seconds=TTL_ADD_HOURS * 3600),
                risk_context=RiskContext(
                    stop_for_risk=add_stop,
                    planned_entry_price=trigger,
                    risk_dollars=RiskCalculator.compute_order_risk_dollars(
                        trigger, add_stop, add_qty, inst.point_value,
                    ),
                ),
            )
        else:
            # Simplified time+R fallback: MARKET order with simpler sizing (matches backtest)
            price_offset = ADD_PRICE_GATE_ATR_MULT * tf1h.atr
            if setup.direction == Direction.LONG:
                if tf1h.close < setup.bos_level + price_offset:
                    return
            else:
                if tf1h.close > setup.bos_level - price_offset:
                    return

            risk_per_contract = setup.r_price * inst.point_value
            if risk_per_contract <= 0:
                return
            add_qty = max(1, int(add_risk / risk_per_contract))

            add_order = OMSOrder(
                strategy_id=STRATEGY_ID,
                instrument=inst,
                side=side,
                qty=add_qty,
                order_type=OrderType.MARKET,
                tif="GTC",
                role=OrderRole.ENTRY,
                entry_policy=EntryPolicy(ttl_seconds=TTL_ADD_HOURS * 3600),
                risk_context=RiskContext(
                    stop_for_risk=setup.current_stop,
                    planned_entry_price=tf1h.close,
                    risk_dollars=RiskCalculator.compute_order_risk_dollars(
                        tf1h.close, setup.current_stop, add_qty, inst.point_value,
                    ),
                ),
            )

        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID,
                order=add_order,
            )
        )
        if receipt.oms_order_id:
            self._order_to_setup[receipt.oms_order_id] = setup.setup_id
            setup.add_done = True
            logger.info(
                "ADD placed for %s %s qty=%d%s",
                setup.symbol, setup.setup_id[:8], add_qty,
                " (price-based)" if add is None else "",
            )

    # ------------------------------------------------------------------
    # Position actions
    # ------------------------------------------------------------------

    async def _partial_exit(self, setup: SetupInstance, qty: int) -> None:
        """Market order to exit partial qty."""
        inst = self._instruments.get(setup.symbol)
        if inst is None:
            return
        if qty <= 0 or qty > setup.qty_open:
            return

        exit_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY

        order = OMSOrder(
            strategy_id=STRATEGY_ID,
            instrument=inst,
            side=exit_side,
            qty=qty,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.EXIT,
        )

        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID,
                order=order,
            )
        )
        if receipt.oms_order_id:
            # Track order for fill reconciliation
            self._order_to_setup[receipt.oms_order_id] = setup.setup_id

            # Estimate realized PnL from partial (use current close as proxy;
            # will be corrected on actual fill in _on_fill)
            tf1h = self.tf_states.get(setup.symbol, {}).get("1H")
            if tf1h and inst:
                if setup.direction == Direction.LONG:
                    partial_pnl = (tf1h.close - setup.fill_price) * inst.point_value * qty
                else:
                    partial_pnl = (setup.fill_price - tf1h.close) * inst.point_value * qty
                setup.realized_pnl += partial_pnl
                setup._partial_pnl_estimate = partial_pnl  # store for correction on fill
            setup.qty_open -= qty
            setup._pending_partial_qty = qty  # track for fill reconciliation
            logger.info("Partial exit %s qty=%d, remaining=%d", setup.symbol, qty, setup.qty_open)

            # Update stop qty
            if setup.stop_order_id and setup.qty_open > 0:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.REPLACE_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=setup.stop_order_id,
                        new_qty=setup.qty_open,
                        new_stop_price=setup.current_stop,
                    )
                )

    async def _update_stop(self, setup: SetupInstance, new_stop: float,
                           adjustment_type: str = "trailing", trigger: str = "helix_trail") -> None:
        """Replace stop via OMS, never loosen."""
        if not setup.stop_order_id:
            return

        old_stop = setup.current_stop
        setup.current_stop = new_stop
        if self._kit and old_stop != new_stop:
            self._kit.log_stop_adjustment(
                trade_id=setup.trade_id or f"HELIX-{setup.symbol}",
                symbol=setup.symbol, old_stop=old_stop, new_stop=new_stop,
                adjustment_type=adjustment_type, trigger=trigger,
            )
        try:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.REPLACE_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=setup.stop_order_id,
                    new_stop_price=new_stop,
                )
            )
            logger.debug("Updated stop for %s to %.4f", setup.symbol, new_stop)
        except Exception as e:
            logger.warning("Error updating stop for %s: %s", setup.symbol, e)

    async def _flatten_setup(self, setup: SetupInstance, reason: str = "FLATTEN") -> None:
        """Flatten entire position + cancel pending orders."""
        inst = self._instruments.get(setup.symbol)
        if inst is None:
            return

        # Cancel stop order
        if setup.stop_order_id:
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=STRATEGY_ID,
                        target_oms_order_id=setup.stop_order_id,
                    )
                )
            except Exception as e:
                logger.warning("Error cancelling stop for %s: %s", setup.symbol, e)

        # Flatten
        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.FLATTEN,
                strategy_id=STRATEGY_ID,
                instrument_symbol=inst.symbol,
            )
        )

        # Hook 5: Instrumentation trade exit (flatten)
        if self._kit and setup.fill_price > 0:
            tid = setup.trade_id or setup.setup_id
            # Use last known price as exit price estimate
            current_price = self._get_current_price(setup.symbol)
            exit_price = current_price if current_price > 0 else setup.fill_price
            # Compute MFE/MAE prices from R values
            if setup.direction == Direction.LONG:
                _mfe_price = setup.fill_price + setup.mfe_r_peak * setup.r_price if setup.r_price > 0 else setup.fill_price
                _mae_price = setup.fill_price + setup.mae_r_trough * setup.r_price if setup.r_price > 0 else setup.fill_price
                _pnl_pct = (exit_price - setup.fill_price) / setup.fill_price if setup.fill_price > 0 else None
            else:
                _mfe_price = setup.fill_price - setup.mfe_r_peak * setup.r_price if setup.r_price > 0 else setup.fill_price
                _mae_price = setup.fill_price - setup.mae_r_trough * setup.r_price if setup.r_price > 0 else setup.fill_price
                _pnl_pct = (setup.fill_price - exit_price) / setup.fill_price if setup.fill_price > 0 else None
            _mfe_pct = abs(setup.mfe_r_peak * setup.r_price / setup.fill_price) if setup.fill_price > 0 and setup.r_price > 0 else None
            _mae_pct = abs(setup.mae_r_trough * setup.r_price / setup.fill_price) if setup.fill_price > 0 and setup.r_price > 0 else None
            self._kit.log_exit(
                trade_id=tid,
                exit_price=exit_price,
                exit_reason=reason,
                expected_exit_price=exit_price,
                mfe_price=_mfe_price,
                mae_price=_mae_price,
                mfe_r=setup.mfe_r_peak,
                mae_r=setup.mae_r_trough,
                mfe_pct=_mfe_pct,
                mae_pct=_mae_pct,
                pnl_pct=_pnl_pct,
                fill_qty=float(setup.qty_open),
            )

        # Track flatten order for fill reconciliation
        if receipt.oms_order_id:
            self._order_to_setup[receipt.oms_order_id] = setup.setup_id
            setup.state = SetupState.CLOSING
            setup._flatten_reason = reason
            setup._closing_since = datetime.now(timezone.utc)
            logger.info("Flatten %s %s → %s (awaiting fill)",
                        setup.symbol, setup.setup_id[:8], receipt.result)
        else:
            # No order ID means immediate (e.g. position already flat at broker)
            setup.state = SetupState.CLOSED
            self.active_setups.pop(setup.setup_id, None)
            logger.info("Flatten %s %s → %s (immediate)",
                        setup.symbol, setup.setup_id[:8], receipt.result)

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def _process_events(self) -> None:
        """Listen on OMS event queue and route to handlers."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                await self._handle_event(event)
            except Exception:
                logger.exception("Error handling event %s", event)

    async def _handle_event(self, event: Any) -> None:
        """Route an OMS event to the appropriate handler."""
        etype = event.event_type
        oms_id = event.oms_order_id

        # Cross-strategy coordination events
        if etype == OMSEventType.COORDINATION:
            await self._handle_coordination(event.payload or {})
            return

        if etype == OMSEventType.FILL or etype == OMSEventType.ORDER_FILLED:
            await self._on_fill(oms_id, event.payload or {})
        elif etype == OMSEventType.RISK_HALT:
            await self._on_risk_halt((event.payload or {}).get("reason", ""))
        elif etype == OMSEventType.ORDER_REJECTED:
            await self._on_terminal(oms_id, etype)
        elif etype in (OMSEventType.ORDER_CANCELLED, OMSEventType.ORDER_EXPIRED):
            await self._on_terminal(oms_id, etype)
        elif etype == OMSEventType.ORDER_WORKING:
            # ORDER_WORKING = order accepted; only mark TRIGGERED if price
            # has actually crossed the stop trigger (bos_level).
            setup_id = self._order_to_setup.get(oms_id)
            if setup_id and setup_id in self.pending_setups:
                setup = self.pending_setups[setup_id]
                if setup.state == SetupState.ARMED and self._is_stop_triggered(setup):
                    setup.state = SetupState.TRIGGERED
                    setup.triggered_ts = datetime.now(timezone.utc)
                    self._schedule_rescue_timer(setup)
                    self._schedule_backstop_timer(setup)

    async def _on_risk_halt(self, reason: str) -> None:
        """Pause new entries and cancel live entry intents."""
        if self._risk_halted:
            return

        self._risk_halted = True
        self._risk_halt_reason = reason or "OMS risk halt"
        logger.error("Helix risk halt engaged: %s", self._risk_halt_reason)

        await self._cancel_all_unfilled("risk_halt")

        for oms_order_id in list(self._order_to_setup):
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
                    "Failed to cancel Helix order %s during risk halt",
                    oms_order_id,
                )

    async def _handle_coordination(self, payload: dict) -> None:
        """Handle cross-strategy coordination events from StrategyCoordinator."""
        coord_type = payload.get("coordination_type", "")
        symbol = payload.get("symbol", "")

        if coord_type == "TIGHTEN_STOP_BE":
            # Rule 1: ATRSS entered on this symbol → tighten Helix stop to breakeven
            await self._tighten_stop_to_breakeven(symbol)

    async def _tighten_stop_to_breakeven(self, symbol: str) -> None:
        """Move stop to breakeven for all active setups on the given symbol.

        Rule 1: When ATRSS enters while Helix has an open position on the same
        symbol, tighten the Helix stop to breakeven to protect against the 64%
        loser rate observed in backtesting.
        """
        for setup in list(self.active_setups.values()):
            if setup.symbol != symbol:
                continue
            if not setup.fill_price or setup.fill_price <= 0:
                continue

            # Compute breakeven with small ATR offset
            tf1h = self.tf_states.get(symbol, {}).get("1H")
            atr_offset = BE_ATR1H_OFFSET * tf1h.atr if (tf1h and tf1h.atr > 0) else 0
            direction_str = "LONG" if setup.direction == Direction.LONG else "SHORT"
            if setup.direction == Direction.LONG:
                be_level = setup.fill_price + atr_offset
                if setup.current_stop < be_level:
                    old_stop = setup.current_stop
                    logger.info(
                        "COORD Rule 1: Tightening Helix %s stop to BE %.4f (was %.4f)",
                        symbol, be_level, old_stop,
                    )
                    await self._update_stop(setup, be_level,
                                            adjustment_type="coordination_tighten", trigger="coord_rule_1_be")
                    if self._coordinator:
                        self._coordinator.log_action(
                            action="tighten_stop_be",
                            trigger_strategy="ATRSS",
                            target_strategy="AKC_HELIX",
                            symbol=symbol,
                            rule="rule_1",
                            details={"old_stop": old_stop, "new_stop": be_level,
                                     "direction": direction_str, "fill_price": setup.fill_price},
                            outcome="applied",
                        )
                else:
                    if self._coordinator:
                        self._coordinator.log_action(
                            action="tighten_stop_be",
                            trigger_strategy="ATRSS",
                            target_strategy="AKC_HELIX",
                            symbol=symbol,
                            rule="rule_1",
                            details={"current_stop": setup.current_stop, "be_level": be_level,
                                     "direction": direction_str},
                            outcome="skipped_already_tighter",
                        )
            else:
                be_level = setup.fill_price - atr_offset
                if setup.current_stop > be_level:
                    old_stop = setup.current_stop
                    logger.info(
                        "COORD Rule 1: Tightening Helix %s stop to BE %.4f (was %.4f)",
                        symbol, be_level, old_stop,
                    )
                    await self._update_stop(setup, be_level,
                                            adjustment_type="coordination_tighten", trigger="coord_rule_1_be")
                    if self._coordinator:
                        self._coordinator.log_action(
                            action="tighten_stop_be",
                            trigger_strategy="ATRSS",
                            target_strategy="AKC_HELIX",
                            symbol=symbol,
                            rule="rule_1",
                            details={"old_stop": old_stop, "new_stop": be_level,
                                     "direction": direction_str, "fill_price": setup.fill_price},
                            outcome="applied",
                        )
                else:
                    if self._coordinator:
                        self._coordinator.log_action(
                            action="tighten_stop_be",
                            trigger_strategy="ATRSS",
                            target_strategy="AKC_HELIX",
                            symbol=symbol,
                            rule="rule_1",
                            details={"current_stop": setup.current_stop, "be_level": be_level,
                                     "direction": direction_str},
                            outcome="skipped_already_tighter",
                        )

    async def _on_fill(self, oms_order_id: str | None, payload: dict) -> None:
        """Handle a fill event — create position state, place protective stop."""
        if not oms_order_id:
            return

        setup_id = self._order_to_setup.pop(oms_order_id, None)
        if setup_id is None:
            # Could be a stop fill or partial exit fill
            await self._on_stop_fill(oms_order_id, payload)
            return

        # Find the setup
        setup = self.pending_setups.pop(setup_id, None)
        if setup is None:
            # Might be an add fill, partial fill, or flatten fill on an active setup
            setup = self.active_setups.get(setup_id)
            if setup is None:
                return

            # --- Handle flatten fill (CLOSING state) ---
            if setup.state == SetupState.CLOSING:
                fill_price = payload.get("price", 0.0)
                reason = getattr(setup, '_flatten_reason', 'FLATTEN')
                logger.info("FLATTEN FILL %s @ %.4f (%s)",
                            setup.symbol, fill_price, reason)
                setup.state = SetupState.CLOSED
                self.active_setups.pop(setup.setup_id, None)
                return

            # --- Handle partial exit fill (correct PnL estimate) ---
            pending_qty = getattr(setup, '_pending_partial_qty', 0)
            if pending_qty > 0:
                fill_price = payload.get("price", 0.0)
                inst = self._instruments.get(setup.symbol)
                pv = inst.point_value if inst else 1.0
                # Correct the estimated PnL with actual fill price
                estimate = getattr(setup, '_partial_pnl_estimate', 0.0)
                if setup.direction == Direction.LONG:
                    actual_pnl = (fill_price - setup.fill_price) * pv * pending_qty
                else:
                    actual_pnl = (setup.fill_price - fill_price) * pv * pending_qty
                correction = actual_pnl - estimate
                setup.realized_pnl += correction
                setup._pending_partial_qty = 0
                setup._partial_pnl_estimate = 0.0
                logger.info("PARTIAL FILL %s qty=%d @ %.4f (correction=%.2f)",
                            setup.symbol, pending_qty, fill_price, correction)
                return

            # Handle add fill
            fill_price = payload.get("price", 0.0)
            fill_qty = int(payload.get("qty", 0))
            setup.qty_open += fill_qty
            logger.info("ADD FILL %s qty=%d @ %.4f, total open=%d",
                        setup.symbol, fill_qty, fill_price, setup.qty_open)
            # Move stop to at least breakeven on add fill
            tf1h = self.tf_states.get(setup.symbol, {}).get("1H")
            if tf1h and setup.fill_price > 0:
                atr_offset = BE_ATR1H_OFFSET * tf1h.atr
                be_level = (setup.fill_price + atr_offset
                            if setup.direction == Direction.LONG
                            else setup.fill_price - atr_offset)
                if setup.direction == Direction.LONG:
                    new_stop = max(setup.current_stop, be_level)
                else:
                    new_stop = min(setup.current_stop, be_level)
                if new_stop != setup.current_stop:
                    logger.info("ADD BE tighten %s: stop %.4f → %.4f",
                                setup.symbol, setup.current_stop, new_stop)
                    await self._update_stop(setup, new_stop,
                                            adjustment_type="breakeven", trigger="add_on_be")
            return

        fill_price = payload.get("price", setup.bos_level)
        fill_qty = int(payload.get("qty", setup.qty_planned))
        fill_time = datetime.now(timezone.utc)

        # Cancel associated timer tasks
        self._cancel_setup_timers(setup.setup_id)

        # Cancel OCA siblings
        for sibling_oid in (setup.primary_order_id, setup.catchup_order_id, setup.rescue_order_id):
            if sibling_oid and sibling_oid != oms_order_id:
                try:
                    await self._oms.submit_intent(
                        Intent(
                            intent_type=IntentType.CANCEL_ORDER,
                            strategy_id=STRATEGY_ID,
                            target_oms_order_id=sibling_oid,
                        )
                    )
                    self._order_to_setup.pop(sibling_oid, None)
                except Exception:
                    pass

        # Update setup state
        setup.state = SetupState.FILLED
        setup.fill_price = fill_price
        setup.fill_qty = fill_qty
        setup.fill_ts = fill_time
        setup.qty_open = fill_qty
        setup.current_stop = setup.stop0

        # ETF slippage guard (spec s11.2)
        cfg = self._config.get(setup.symbol)
        if cfg and cfg.is_etf:
            slip_dollars = abs(fill_price - setup.bos_level)
            slip_bps = (slip_dollars / setup.bos_level * 10_000) if setup.bos_level > 0 else 0.0
            slip_exceeded = False
            if cfg.slip_max_dollars > 0 or cfg.slip_max_bps > 0:
                slip_exceeded = (slip_dollars > cfg.slip_max_dollars
                                 or slip_bps > cfg.slip_max_bps)
            if slip_exceeded:
                setup.teleport_fill = True
                setup.add_min_r_override = 2.0  # penalty: no add until +2R
                logger.warning(
                    "%s TELEPORT FILL: slip=$%.4f (%.1f bps) exceeds limits ($%.2f / %d bps) — add delayed to +2R",
                    setup.symbol, slip_dollars, slip_bps,
                    cfg.slip_max_dollars, int(cfg.slip_max_bps),
                )

        # Record trade entry
        if self._recorder:
            try:
                trade_id = await self._recorder.record_entry(
                    strategy_id=STRATEGY_ID,
                    instrument=setup.symbol,
                    direction="LONG" if setup.direction == Direction.LONG else "SHORT",
                    quantity=fill_qty,
                    entry_price=Decimal(str(fill_price)),
                    entry_ts=fill_time,
                    setup_tag=setup.setup_class.value,
                    entry_type=setup.setup_class.value,
                    meta={
                        "setup_id": setup.setup_id,
                        "adx_at_entry": setup.adx_at_entry,
                        "regime_4h_at_entry": setup.regime_4h_at_entry,
                        "size_mult": setup.setup_size_mult,
                        "vol_factor": setup.vol_factor_at_placement,
                    },
                )
                setup.trade_id = trade_id
            except Exception:
                logger.exception("Error recording entry for %s", setup.symbol)

        # Place protective stop
        inst = self._instruments.get(setup.symbol)
        if inst:
            stop_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
            stop_order = OMSOrder(
                strategy_id=STRATEGY_ID,
                instrument=inst,
                side=stop_side,
                qty=fill_qty,
                order_type=OrderType.STOP,
                stop_price=setup.stop0,
                tif="GTC",
                role=OrderRole.STOP,
            )
            stop_receipt = await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.NEW_ORDER,
                    strategy_id=STRATEGY_ID,
                    order=stop_order,
                )
            )
            if stop_receipt.oms_order_id:
                setup.stop_order_id = stop_receipt.oms_order_id

        # Move to active — record regime at entry for transition tracking
        setup.state = SetupState.ACTIVE
        daily = self.daily_states.get(setup.symbol)
        if daily:
            setup.regime_at_entry = daily.regime.value
        self.active_setups[setup.setup_id] = setup

        logger.info(
            "FILL %s %s %s %d @ %.4f (stop=%.4f)",
            setup.symbol, setup.setup_class.value,
            "LONG" if setup.direction == Direction.LONG else "SHORT",
            fill_qty, fill_price, setup.stop0,
        )

        # Hook 4: Instrumentation trade entry
        if self._kit:
            side_str = "LONG" if setup.direction == Direction.LONG else "SHORT"
            active = {sid: s for sid, s in self.active_setups.items() if s.state in (SetupState.FILLED, SetupState.ACTIVE)}
            # Correlated pairs (basket peers)
            from .config import BASKET_SYMBOLS
            correlated = []
            if setup.symbol in BASKET_SYMBOLS:
                for sid, peer in active.items():
                    if peer.symbol in BASKET_SYMBOLS and peer.symbol != setup.symbol:
                        correlated.append({
                            "symbol": peer.symbol,
                            "direction": "LONG" if peer.direction == Direction.LONG else "SHORT",
                            "relationship": "basket_peer",
                            "same_direction": (peer.direction == setup.direction),
                        })
            _cfg_dict = dataclasses.asdict(cfg) if cfg else {}
            _param_set_id = hashlib.md5(
                json.dumps(_cfg_dict, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            self._kit.log_entry(
                trade_id=setup.trade_id or setup.setup_id,
                pair=setup.symbol,
                side=side_str,
                entry_price=fill_price,
                position_size=float(fill_qty),
                position_size_quote=fill_price * fill_qty,
                entry_signal=setup.setup_class.value,
                entry_signal_id=setup.setup_id,
                entry_signal_strength=0.5,
                active_filters=["spread_gate", "heat_cap"],
                passed_filters=["spread_gate", "heat_cap"],
                filter_decisions=setup.gate_decisions,
                strategy_params={
                    "param_set_id": _param_set_id,
                    "config": _cfg_dict,
                    "adx_at_entry": setup.adx_at_entry,
                    "regime_4h": setup.regime_4h_at_entry,
                    "size_mult": setup.setup_size_mult,
                    "setup_class": setup.setup_class.value,
                    "origin_tf": setup.origin_tf,
                    "div_mag_norm": setup.div_mag_norm,
                    "vol_factor": setup.vol_factor_at_placement,
                    "bos_level": setup.bos_level,
                    "stop0": setup.stop0,
                    "base_risk_pct": self._base_risk_pct if hasattr(self, '_base_risk_pct') else 0.01,
                    "r_price": setup.r_price,
                    "unit1_risk_dollars": setup.unit1_risk_dollars,
                },
                expected_entry_price=setup.bos_level,
                signal_factors=[
                    {"factor_name": "adx", "factor_value": setup.adx_at_entry, "threshold": 20.0, "contribution": "trend_strength"},
                    {"factor_name": "setup_class", "factor_value": setup.setup_class.value, "threshold": "CLASS_A", "contribution": "setup_quality"},
                    {"factor_name": "size_mult", "factor_value": setup.setup_size_mult, "threshold": 0.5, "contribution": "conviction"},
                    {"factor_name": "div_mag_norm", "factor_value": setup.div_mag_norm, "threshold": 0.5, "contribution": "divergence_magnitude"},
                    {"factor_name": "vol_factor", "factor_value": setup.vol_factor_at_placement, "threshold": 1.0, "contribution": "volatility_regime"},
                    {"factor_name": "regime_4h", "factor_value": setup.regime_4h_at_entry or "unknown", "threshold": "BULL", "contribution": "higher_tf_regime"},
                    {"factor_name": "origin_tf", "factor_value": setup.origin_tf, "threshold": "4H", "contribution": "timeframe_origin"},
                ],
                sizing_inputs={
                    "target_risk_pct": self._base_risk_pct if hasattr(self, '_base_risk_pct') else 0.01,
                    "account_equity": self._equity if hasattr(self, '_equity') else 0.0,
                    "volatility_basis": setup.adx_at_entry,
                    "sizing_model": "helix_class_mult",
                },
                portfolio_state_at_entry={
                    "num_positions": len(active),
                    "symbols_held": [s.symbol for s in active.values()],
                },
                signal_evolution=self._build_signal_evolution(setup.symbol),
                correlated_pairs_detail=correlated if correlated else None,
                concurrent_positions_strategy=len(self.active_setups),
                fill_order_id=oms_order_id,
                fill_qty=float(fill_qty),
                fill_time_ms=int(fill_time.timestamp() * 1000),
            )

            self._kit.on_order_event(
                order_id=oms_order_id,
                pair=setup.symbol,
                side="LONG" if setup.direction == Direction.LONG else "SHORT",
                order_type="STOP_LIMIT",
                status="FILLED",
                requested_qty=float(setup.qty_planned),
                filled_qty=float(fill_qty),
                requested_price=setup.bos_level,
                fill_price=fill_price,
                related_trade_id=setup.trade_id or setup.setup_id,
                strategy_id=STRATEGY_ID,
            )

    async def _on_stop_fill(self, oms_order_id: str, payload: dict) -> None:
        """Handle a stop-loss fill — record exit, update circuit breaker."""
        for setup_id, setup in list(self.active_setups.items()):
            if setup.stop_order_id == oms_order_id:
                fill_price = payload.get("price", setup.current_stop)
                fill_time = datetime.now(timezone.utc)

                # Compute R
                if setup.r_price > 0:
                    if setup.direction == Direction.LONG:
                        realized_r = (fill_price - setup.fill_price) / setup.r_price
                    else:
                        realized_r = (setup.fill_price - fill_price) / setup.r_price
                else:
                    realized_r = 0.0

                inst = self._instruments.get(setup.symbol)
                pv = inst.point_value if inst else 1.0
                if setup.direction == Direction.LONG:
                    pnl_usd = (fill_price - setup.fill_price) * pv * setup.qty_open
                else:
                    pnl_usd = (setup.fill_price - fill_price) * pv * setup.qty_open

                # Record exit
                if self._recorder and setup.trade_id:
                    try:
                        await self._recorder.record_exit(
                            trade_id=setup.trade_id,
                            exit_price=Decimal(str(fill_price)),
                            exit_ts=fill_time,
                            exit_reason=f"STOP_{setup.stop_source}",
                            realized_r=Decimal(str(round(realized_r, 4))),
                            realized_usd=Decimal(str(round(pnl_usd, 2))),
                            duration_bars=setup.bars_held_1h,
                        )
                    except Exception:
                        logger.exception("Error recording exit for %s", setup.symbol)

                # Update circuit breaker
                cb = self.circuit_breakers.get(setup.symbol, CircuitBreakerState())
                cb.daily_realized_r += realized_r
                cb.weekly_realized_r += realized_r
                if realized_r < 0:
                    cb.consecutive_stops += 1
                    if cb.consecutive_stops >= CONSEC_STOPS_HALVE:
                        cb.halved_until = fill_time + timedelta(hours=24)
                        logger.warning("%s %d consecutive stops — halving size",
                                       setup.symbol, cb.consecutive_stops)
                else:
                    cb.consecutive_stops = 0

                if cb.daily_realized_r <= DAILY_STOP_R:
                    # Spec s8.4: pause until next session open
                    cfg = self._config.get(setup.symbol)
                    if cfg and cfg.is_etf:
                        next_open = _next_session_open_et(fill_time, cfg.entry_window_start_et)
                    else:
                        next_open = _next_session_open_et(fill_time, "03:00")
                    cb.paused_until = next_open
                    logger.warning("%s daily stop hit (%.2fR) — pausing until %s",
                                   setup.symbol, cb.daily_realized_r, next_open.isoformat())
                if cb.weekly_realized_r <= WEEKLY_STOP_R:
                    # Spec s8.4: pause until next Daily close
                    next_daily = _next_daily_close_et(fill_time)
                    cb.paused_until = next_daily
                    logger.warning("%s weekly stop hit (%.2fR) — pausing until %s",
                                   setup.symbol, cb.weekly_realized_r, next_daily.isoformat())
                self.circuit_breakers[setup.symbol] = cb

                # Hook 5: Instrumentation trade exit + process scoring
                if self._kit:
                    tid = setup.trade_id or setup.setup_id
                    if setup.direction == Direction.LONG:
                        _mfe_price = setup.fill_price + setup.mfe_r_peak * setup.r_price if setup.r_price > 0 else setup.fill_price
                        _mae_price = setup.fill_price + setup.mae_r_trough * setup.r_price if setup.r_price > 0 else setup.fill_price
                        _pnl_pct = (fill_price - setup.fill_price) / setup.fill_price if setup.fill_price > 0 else None
                    else:
                        _mfe_price = setup.fill_price - setup.mfe_r_peak * setup.r_price if setup.r_price > 0 else setup.fill_price
                        _mae_price = setup.fill_price - setup.mae_r_trough * setup.r_price if setup.r_price > 0 else setup.fill_price
                        _pnl_pct = (setup.fill_price - fill_price) / setup.fill_price if setup.fill_price > 0 else None
                    _mfe_pct = abs(setup.mfe_r_peak * setup.r_price / setup.fill_price) if setup.fill_price > 0 and setup.r_price > 0 else None
                    _mae_pct = abs(setup.mae_r_trough * setup.r_price / setup.fill_price) if setup.fill_price > 0 and setup.r_price > 0 else None
                    stop_reason = f"STOP_{setup.stop_source}"
                    self._kit.log_exit(
                        trade_id=tid,
                        exit_price=fill_price,
                        exit_reason=stop_reason,
                        expected_exit_price=setup.current_stop,
                        mfe_price=_mfe_price,
                        mae_price=_mae_price,
                        mfe_r=setup.mfe_r_peak,
                        mae_r=setup.mae_r_trough,
                        mfe_pct=_mfe_pct,
                        mae_pct=_mae_pct,
                        pnl_pct=_pnl_pct,
                        fill_order_id=oms_order_id,
                        fill_qty=float(setup.qty_open),
                    )

                    self._kit.on_order_event(
                        order_id=oms_order_id,
                        pair=setup.symbol,
                        side="SELL" if setup.direction == Direction.LONG else "BUY",
                        order_type="STOP",
                        status="FILLED",
                        requested_qty=float(setup.qty_open),
                        filled_qty=float(setup.qty_open),
                        requested_price=setup.current_stop,
                        fill_price=fill_price,
                        related_trade_id=setup.trade_id or setup.setup_id,
                        strategy_id=STRATEGY_ID,
                    )

                # Clean up
                setup.state = SetupState.CLOSED
                self.active_setups.pop(setup_id, None)
                logger.info("STOPPED OUT %s @ %.4f (%.2fR) after %d bars",
                            setup.symbol, fill_price, realized_r, setup.bars_held_1h)
                break

    async def _on_terminal(self, oms_order_id: str | None, etype: Any) -> None:
        """Clean up on cancel/reject/expire."""
        if not oms_order_id:
            return
        setup_id = self._order_to_setup.pop(oms_order_id, None)
        if setup_id:
            logger.info("Order %s terminal (%s) for setup %s", oms_order_id, etype, setup_id[:8])

            setup = self.pending_setups.get(setup_id) or self.active_setups.get(setup_id)
            if self._kit and setup:
                status_map = {
                    OMSEventType.ORDER_REJECTED: "REJECTED",
                    OMSEventType.ORDER_CANCELLED: "CANCELLED",
                    OMSEventType.ORDER_EXPIRED: "EXPIRED",
                }
                self._kit.on_order_event(
                    order_id=oms_order_id,
                    pair=setup.symbol,
                    side="LONG" if setup.direction == Direction.LONG else "SHORT",
                    order_type="STOP_LIMIT",
                    status=status_map.get(etype, "CANCELLED"),
                    requested_qty=float(setup.qty_planned),
                    requested_price=setup.bos_level,
                    strategy_id=STRATEGY_ID,
                )

    # ------------------------------------------------------------------
    # Live market data helpers
    # ------------------------------------------------------------------

    def _get_current_price(self, sym: str) -> float:
        """Best available current price: live ticker → 1H close fallback."""
        ticker = self._tickers.get(sym)
        if ticker is not None:
            last = getattr(ticker, 'last', None)
            if last is not None and last != -1 and last == last and last > 0:
                return float(last)
            bid = getattr(ticker, 'bid', None)
            ask = getattr(ticker, 'ask', None)
            if (bid and ask and bid > 0 and ask > 0
                    and bid == bid and ask == ask):
                return float((bid + ask) / 2)
        tf1h = self.tf_states.get(sym, {}).get("1H")
        return tf1h.close if tf1h else 0.0

    def _get_spread_info(self, sym: str, cfg: SymbolConfig) -> tuple[float, float, float]:
        """Compute real-time spread from IBKR market data.

        Returns (spread_ticks, spread_dollars, spread_bps).
        All ``float('inf')`` when data is unavailable so the spread
        gate fails closed (spec s7.1 fail-closed rule).
        """
        ticker = self._tickers.get(sym)
        if ticker is None or cfg.tick_size <= 0:
            return float('inf'), float('inf'), float('inf')
        bid = getattr(ticker, 'bid', None)
        ask = getattr(ticker, 'ask', None)
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return float('inf'), float('inf'), float('inf')
        if bid != bid or ask != ask:  # NaN check
            return float('inf'), float('inf'), float('inf')
        spread = ask - bid
        mid = (ask + bid) / 2.0
        spread_ticks = spread / cfg.tick_size
        spread_dollars = spread
        spread_bps = (spread / mid * 10_000) if mid > 0 else float('inf')
        return spread_ticks, spread_dollars, spread_bps

    def _is_stop_triggered(self, setup: SetupInstance) -> bool:
        """Check if price has crossed the stop trigger (bos_level)."""
        price = self._get_current_price(setup.symbol)
        if price <= 0:
            return False
        if setup.direction == Direction.LONG:
            return price >= setup.bos_level
        return price <= setup.bos_level

    # ------------------------------------------------------------------
    # Ticker-event trigger detection (primary) + polling fallback
    # ------------------------------------------------------------------

    def _on_ticker_update(self, tickers: set) -> None:
        """pendingTickersEvent callback — primary trigger detector."""
        now = datetime.now(timezone.utc)
        try:
            now_et = now.astimezone(ET)
        except Exception:
            now_et = now
        updated_syms: set[str] = set()
        for t in tickers:
            c = getattr(t, 'contract', None)
            if c:
                updated_syms.add(getattr(c, 'symbol', ''))
        for setup_id, setup in list(self.pending_setups.items()):
            if setup.state != SetupState.ARMED or setup.symbol not in updated_syms:
                continue
            if self._is_stop_triggered(setup):
                setup.state = SetupState.TRIGGERED
                setup.triggered_ts = now
                self._schedule_rescue_timer(setup)
                self._schedule_backstop_timer(setup)
                logger.info("TRIGGERED %s %s via ticker event",
                            setup.symbol, setup.setup_id[:8])

    async def _trigger_monitor(self) -> None:
        """Fallback 15s loop in case ticker events miss a crossing."""
        while self._running:
            await asyncio.sleep(15)
            if not self._running:
                break
            now = datetime.now(timezone.utc)
            try:
                now_et = now.astimezone(ET)
            except Exception:
                now_et = now
            for setup_id, setup in list(self.pending_setups.items()):
                if setup.state != SetupState.ARMED:
                    continue
                if not setup.primary_order_id:
                    continue
                if self._is_stop_triggered(setup):
                    setup.state = SetupState.TRIGGERED
                    setup.triggered_ts = now
                    self._schedule_rescue_timer(setup)
                    self._schedule_backstop_timer(setup)
                    logger.info(
                        "TRIGGERED %s %s via price monitor",
                        setup.symbol, setup.setup_id[:8],
                    )

    # ------------------------------------------------------------------
    # Event-driven timer tasks (rescue, TTL, backstop)
    # ------------------------------------------------------------------

    def _schedule_rescue_timer(self, setup: SetupInstance) -> None:
        """Schedule rescue evaluation 5 minutes after trigger."""
        key = f"rescue_{setup.setup_id}"
        if key in self._timer_tasks:
            return
        self._timer_tasks[key] = asyncio.create_task(
            self._rescue_timer(setup.setup_id)
        )

    async def _rescue_timer(self, setup_id: str) -> None:
        key = f"rescue_{setup_id}"
        try:
            await asyncio.sleep(5 * 60)
            setup = self.pending_setups.get(setup_id)
            if (setup and setup.state == SetupState.TRIGGERED
                    and not setup.rescue_order_id):
                await self._maybe_rescue(setup, datetime.now(timezone.utc))
        except asyncio.CancelledError:
            pass
        finally:
            self._timer_tasks.pop(key, None)

    def _schedule_backstop_timer(self, setup: SetupInstance) -> None:
        """Schedule end-of-bar backstop: cancel 1 hour after trigger."""
        key = f"backstop_{setup.setup_id}"
        if key in self._timer_tasks:
            return
        self._timer_tasks[key] = asyncio.create_task(
            self._backstop_timer(setup.setup_id)
        )

    async def _backstop_timer(self, setup_id: str) -> None:
        key = f"backstop_{setup_id}"
        try:
            await asyncio.sleep(3600)
            setup = self.pending_setups.get(setup_id)
            if setup and setup.state == SetupState.TRIGGERED:
                await self._cancel_setup(setup, "end_of_bar_backstop")
        except asyncio.CancelledError:
            pass
        finally:
            self._timer_tasks.pop(key, None)

    def _schedule_order_ttl(
        self, setup: SetupInstance, order_type: str, ttl_seconds: int,
    ) -> None:
        """Schedule TTL cancellation for a catch-up or rescue order."""
        key = f"{order_type}_ttl_{setup.setup_id}"
        old = self._timer_tasks.pop(key, None)
        if old:
            old.cancel()
        self._timer_tasks[key] = asyncio.create_task(
            self._order_ttl_timer(setup.setup_id, order_type, ttl_seconds)
        )

    async def _order_ttl_timer(
        self, setup_id: str, order_type: str, ttl_seconds: int,
    ) -> None:
        key = f"{order_type}_ttl_{setup_id}"
        try:
            await asyncio.sleep(ttl_seconds)
            setup = self.pending_setups.get(setup_id)
            if setup is None:
                return
            if order_type == "catchup" and setup.catchup_order_id:
                try:
                    await self._oms.submit_intent(
                        Intent(
                            intent_type=IntentType.CANCEL_ORDER,
                            strategy_id=STRATEGY_ID,
                            target_oms_order_id=setup.catchup_order_id,
                        )
                    )
                    self._order_to_setup.pop(setup.catchup_order_id, None)
                    setup.catchup_order_id = ""
                except Exception as e:
                    logger.warning("Timer: error cancelling catch-up %s: %s",
                                   setup.catchup_order_id, e)
            elif order_type == "rescue" and setup.rescue_order_id:
                await self._cancel_setup(setup, "rescue_ttl_expired")
        except asyncio.CancelledError:
            pass
        finally:
            self._timer_tasks.pop(key, None)

    def _cancel_setup_timers(self, setup_id: str) -> None:
        """Cancel all timer tasks associated with a setup."""
        for prefix in ("rescue_", "backstop_", "catchup_ttl_", "rescue_ttl_"):
            key = f"{prefix}{setup_id}"
            task = self._timer_tasks.pop(key, None)
            if task:
                task.cancel()

    # ------------------------------------------------------------------
    # Historical data loading
    # ------------------------------------------------------------------

    async def _load_initial_bars(self) -> None:
        """Load enough historical bars to seed all indicators and pivots."""
        for sym in self._config:
            try:
                cfg = self._config[sym]

                # Daily bars
                bars_d = await self._fetch_bars(sym, cfg, "1 day", "200 D")
                if bars_d is not None:
                    closes = np.array([b.close for b in bars_d], dtype=float)
                    highs = np.array([b.high for b in bars_d], dtype=float)
                    lows = np.array([b.low for b in bars_d], dtype=float)
                    last_date = str(bars_d[-1].date) if bars_d else None
                    self.daily_states[sym] = compute_daily_state(
                        closes, highs, lows, None, last_date,
                    )
                    logger.info(
                        "%s daily: regime=%s atr=%.2f vf=%.2f",
                        sym, self.daily_states[sym].regime.value,
                        self.daily_states[sym].atr_d,
                        self.daily_states[sym].vol_factor,
                    )

                # 1H bars (30 days for chandelier + pivot history)
                bars_1h = await self._fetch_bars(sym, cfg, "1 hour", "30 D")
                if bars_1h is not None:
                    self._update_tf_state(sym, "1H", bars_1h)
                    logger.info(
                        "%s 1H: pivots H=%d L=%d",
                        sym,
                        len(self.pivots[sym]["1H"].highs),
                        len(self.pivots[sym]["1H"].lows),
                    )

                # 4H bars (60 days for sufficient pivot history)
                bars_4h = await self._fetch_bars(sym, cfg, "4 hours", "60 D")
                if bars_4h is not None:
                    self._update_tf_state(sym, "4H", bars_4h)
                    logger.info(
                        "%s 4H: pivots H=%d L=%d",
                        sym,
                        len(self.pivots[sym]["4H"].highs),
                        len(self.pivots[sym]["4H"].lows),
                    )

            except Exception:
                logger.exception("Error loading initial bars for %s", sym)

    async def _fetch_bars(
        self, sym: str, cfg: SymbolConfig, bar_size: str, duration: str,
    ) -> list[Any] | None:
        """Fetch historical bars from IB."""
        try:
            contract = self._get_contract(sym)
            if contract is None:
                return None

            bars = await self._ib.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
            return bars if bars else None
        except Exception:
            logger.exception("Error fetching %s bars for %s", bar_size, sym)
            return None

    def _get_contract(self, sym: str) -> Any | None:
        """Get the IB contract for a symbol from cache or build a generic one."""
        if sym in self.contracts:
            return self.contracts[sym][0]

        cfg = self._config[sym]
        try:
            if cfg.is_etf:
                from ib_async import Stock
                return Stock(symbol=sym, exchange=cfg.exchange, currency="USD")
            else:
                from ib_async import Future
                c = Future(
                    symbol=sym,
                    exchange=cfg.exchange,
                    currency="USD",
                )
                if cfg.trading_class:
                    c.tradingClass = cfg.trading_class
                return c
        except Exception:
            logger.warning("Cannot build contract for %s", sym)
            return None
