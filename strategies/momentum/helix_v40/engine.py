"""AKC-Helix NQ TrendWrap v4.0 Apex Trail — main engine orchestrator (event-driven).

v4.0 changes:
- Only Class M + F detection (removed A, R, DivergenceHistory, halt queue)
- DOW block: Monday + Wednesday
- Hour-of-day sizing
- Simplified priority (M > F > T)
- Class T: 4H trend continuation (longs only, half-size)
- ETH_EUROPE regime: surgical reopening (longs, score=2, strong trend, low vol)
- RTH_DEAD enhanced sizing (0.70→0.85 when conditions met)
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import date as date_type, datetime, time, timedelta, timezone
from typing import Optional

import pytz

from .config import (
    STRATEGY_ID, TF, SetupClass, SetupState, Setup, PositionState, Bar,
    SessionBlock, SUNDAY_GAP_ATR_FRAC, SPREAD_RECHECK_BARS,
    MAX_CONCURRENT_POSITIONS, DUPLICATE_OVERRIDE_MIN_R,
    VOL_50_80_SIZING_MULT, DOW_BLOCKED, HOUR_SIZE_MULT,
    CLASS_T_MIN_BARS_SINCE_M, DRAWDOWN_THROTTLE_ENABLED,
)
from strategies.momentum.instrumentation.src.config_snapshot import snapshot_config_module
from strategies.momentum.helix_v40 import config as strategy_config
from .indicators import BarSeries, VolEngine
from .pivots import PivotDetector
from .signals import SignalEngine, alignment_score, trend_strength
from .gates import check_gates, GateResult
from .risk import RiskEngine
from libs.risk.drawdown_throttle import DrawdownThrottle
from .execution import ExecutionEngine

from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.events import OMSEventType
from .positions import PositionManager, unrealized_r
from .diagnostics import DiagnosticTracker
from .session import (
    get_session_block, session_size_mult, entries_allowed,
    is_halt, is_reopen_dead, is_pre_halt_cancel,
    is_sunday_reopen, NewsCalendar,
)

logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


class Helix4Engine:
    """Event-driven engine for AKC-Helix NQ TrendWrap v4.0 Apex Trail."""

    def __init__(self, ib_session, oms_service, instruments, trade_recorder=None,
                 equity: float = 100_000.0, instrumentation=None,
                 equity_alloc_pct: float = 1.0):
        self.ib = ib_session
        self.oms = oms_service
        self.instruments = instruments
        self.trade_recorder = trade_recorder
        self.equity = equity
        self._equity_alloc_pct = equity_alloc_pct
        self._instr = instrumentation

        from strategies.momentum.instrumentation.src.facade import InstrumentationKit
        self._kit = InstrumentationKit(self._instr, strategy_type="helix")

        self.nq_inst = instruments[0]

        # Bar series
        self.h1 = BarSeries(TF.H1, maxlen=500)
        self.h4 = BarSeries(TF.H4, maxlen=300)
        self.daily = BarSeries(TF.D1, maxlen=200)

        # Pivots
        self.pivots_1h = PivotDetector(TF.H1)
        self.pivots_4h = PivotDetector(TF.H4)

        # Volatility
        self.vol = VolEngine()

        # Sub-engines
        self.signals = SignalEngine()
        self.risk = RiskEngine(equity, self.vol, point_value=self.nq_inst.point_value)
        self.exec = ExecutionEngine(oms_service, self.nq_inst)
        self.diagnostics = DiagnosticTracker(point_value=self.nq_inst.point_value)
        self.positions = PositionManager(
            self.exec, self.risk, self.signals,
            point_value=self.nq_inst.point_value,
            diagnostics=self.diagnostics,
        )
        self.news = NewsCalendar()

        # Drawdown throttle
        self._throttle = DrawdownThrottle(equity)

        # State
        self._running = False
        self._ts_history: deque[float] = deque(maxlen=10)

        # Dedup
        self._placed_signatures: set[tuple] = set()
        self._sig_expiry: dict[tuple, datetime] = {}

        # Class T: track last M placement bar per direction
        self._last_m_bar: dict[int, int] = {}
        self._bar_idx_1h: int = 0

        # Spread re-check
        self._spread_recheck: list[tuple[Setup, int]] = []

        # Execution cascade timestamps (#16)
        self._cascade_ts: dict[str, datetime] = {}

        # Bid/ask cache
        self._bid: Optional[float] = None
        self._ask: Optional[float] = None

        # Signal evolution ring buffer (M2)
        self._signal_ring: deque = deque(maxlen=10)

        # Event-driven infrastructure
        self._bar_streams: dict[TF, object] = {}
        self._ticker = None
        self._contract = None
        self._eval_lock = asyncio.Lock()
        self._oms_task: Optional[asyncio.Task] = None
        self._timer_task: Optional[asyncio.Task] = None
        self._last_session_actions: dict[str, datetime] = {}

    def _dd_tier_name(self) -> str:
        mult = self._throttle.dd_size_mult
        if mult >= 1.0:
            return "full"
        elif mult >= 0.5:
            return "half"
        elif mult >= 0.25:
            return "quarter"
        return "halt"

    def _build_gate_filter_decisions(self, setup, now_et) -> list[dict]:
        """Build structured filter decisions from current gate state."""
        from .config import HIGH_VOL_M_THRESHOLD
        from .session import get_session_block, max_spread_for_session
        block = get_session_block(now_et)
        decisions = []

        # Heat total
        total_risk = self.risk.open_risk_r + self.risk.pending_risk_r
        cap = self.risk.heat_cap_r()
        decisions.append({
            "filter_name": "heat_total",
            "threshold": round(cap, 3),
            "actual_value": round(total_risk, 3),
            "passed": total_risk <= cap,
            "margin_pct": round((total_risk - cap) / abs(cap) * 100, 1) if cap > 0 else None,
        })

        # Spread
        if self._bid is not None and self._ask is not None:
            spread = self._ask - self._bid
            max_sp = max_spread_for_session(block)
            decisions.append({
                "filter_name": "spread",
                "threshold": round(max_sp, 4),
                "actual_value": round(spread, 4),
                "passed": spread <= max_sp,
                "margin_pct": round((spread - max_sp) / abs(max_sp) * 100, 1) if max_sp > 0 else None,
            })

        # High vol
        decisions.append({
            "filter_name": "high_vol",
            "threshold": HIGH_VOL_M_THRESHOLD,
            "actual_value": round(self.vol.vol_pct, 1),
            "passed": self.vol.vol_pct <= HIGH_VOL_M_THRESHOLD,
            "margin_pct": round((self.vol.vol_pct - HIGH_VOL_M_THRESHOLD) / abs(HIGH_VOL_M_THRESHOLD) * 100, 1)
                if HIGH_VOL_M_THRESHOLD > 0 else None,
        })

        return decisions

    def _snapshot_signal_state(self) -> dict:
        """Capture current 1H signal state for evolution tracking (M2)."""
        return {
            "close": self.h1.last_close,
            "ema_fast": self.h1.ema_fast(),
            "ema_slow": self.h1.ema_slow(),
            "atr": self.h1.current_atr(),
            "macd_line": self.h1.macd_line_now(),
            "macd_hist": self.h1.macd_hist_now(),
        }

    def _build_signal_evolution(self, n: int = 5) -> list[dict]:
        """Return last n signal snapshots with bars_ago labels."""
        items = list(self._signal_ring)[-n:]
        return [{"bars_ago": n - 1 - i, **s} for i, s in enumerate(items)]

    def _log_missed(self, setup, blocked_by: str, block_reason: str, **extra):
        if not self._kit.active:
            return
        try:
            from .session import get_session_block
            block = get_session_block(datetime.now(ET))
            self._kit.log_missed(
                pair=self.nq_inst.symbol,
                side="LONG" if setup.direction == 1 else "SHORT",
                signal=f"Class_{setup.cls.value}",
                signal_id=setup.setup_id if hasattr(setup, 'setup_id') else "",
                signal_strength=setup.alignment_score / 3.0,
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_params={
                    "cls": setup.cls.value,
                    "score": setup.alignment_score,
                    "entry_stop": setup.entry_stop,
                    "stop0": setup.stop0,
                    **extra,
                },
                session_type=block.value,
                concurrent_positions=len(self.positions.positions),
                drawdown_pct=self._throttle.dd_pct if hasattr(self._throttle, 'dd_pct') else None,
                drawdown_tier=self._dd_tier_name(),
                signal_evolution=self._build_signal_evolution(),
            )
        except Exception:
            pass

    async def start(self) -> None:
        self._running = True
        self._contract = await self._qualify_contract()
        await self._subscribe_and_backfill()
        self._oms_task = asyncio.create_task(self._listen_oms_events())
        self._timer_task = asyncio.create_task(self._session_timer())
        logger.info("Helix4Engine started (event-driven)")

    def get_position_snapshot(self) -> list[dict]:
        """Return current position state for heartbeat emission."""
        result = []
        for pos in self.positions.positions:
            result.append({
                "strategy_type": "helix",
                "direction": "LONG" if pos.direction == 1 else "SHORT",
                "entry_price": pos.avg_entry,
                "qty": pos.contracts,
                "unrealized_pnl_r": round(
                    (self.h1.close[-1] - pos.avg_entry) * pos.direction
                    * self.nq_inst.point_value * pos.contracts
                    / max(pos.unit1_risk_usd, 1e-9), 3
                ) if len(self.h1.close) > 0 else 0.0,
            })
        return result

    async def stop(self) -> None:
        self._running = False
        for bars in self._bar_streams.values():
            try:
                self.ib.ib.cancelHistoricalData(bars)
            except Exception:
                pass
        if self._ticker:
            try:
                self.ib.ib.cancelMktData(self._ticker.contract)
            except Exception:
                pass
        for task in [self._oms_task, self._timer_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("Helix4Engine stopped")

    async def _qualify_contract(self):
        from ib_async import ContFuture
        contract = ContFuture(symbol="NQ", exchange="CME", currency="USD")
        contracts = await self.ib.ib.qualifyContractsAsync(contract)
        return contracts[0] if contracts else contract

    @staticmethod
    def _normalize_ts(raw_ts) -> datetime:
        if raw_ts is None:
            return datetime.now(timezone.utc)
        if isinstance(raw_ts, datetime):
            return raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
        if isinstance(raw_ts, date_type):
            return datetime(raw_ts.year, raw_ts.month, raw_ts.day, tzinfo=timezone.utc)
        if isinstance(raw_ts, str):
            try:
                dt = datetime.fromisoformat(raw_ts)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    async def _subscribe_and_backfill(self) -> None:
        for bar_size, duration, series, tf in [
            ("1 day", "200 D", self.daily, TF.D1),
            ("4 hours", "60 D", self.h4, TF.H4),
            ("1 hour", "30 D", self.h1, TF.H1),
            ("5 mins", "5 D", None, TF.M5),
        ]:
            try:
                bars = await self.ib.ib.reqHistoricalDataAsync(
                    self._contract, endDateTime="", durationStr=duration,
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=False, formatDate=1, keepUpToDate=True,
                )
                self._bar_streams[tf] = bars
                if series is not None:
                    for b in bars:
                        ts = self._normalize_ts(b.date if hasattr(b, 'date') else None)
                        series.add_bar(Bar(
                            ts=ts, open=b.open, high=b.high,
                            low=b.low, close=b.close, volume=b.volume,
                        ))
                logger.info("Subscribed %s: %d initial bars", tf.value, len(bars))
            except Exception as e:
                logger.error("Subscribe %s failed: %s", tf.value, e)

        self.vol.update(self.daily)
        for bar_obj in list(self.h1.bars):
            self.pivots_1h.on_bar(bar_obj, self.h1)
        for bar_obj in list(self.h4.bars):
            self.pivots_4h.on_bar(bar_obj, self.h4)

        self.daily._recompute()
        if self.daily._atr is not None:
            n = len(self.daily._atr)
            for i in range(max(0, n - 10), n):
                self._ts_history.append(self.daily.trend_strength_at(i))
        ts_3d = self._ts_history[-4] if len(self._ts_history) >= 4 else None
        self.signals.update_trend_strength_3d(ts_3d)

        for tf, bars in self._bar_streams.items():
            bars.updateEvent += lambda b, hasNew, _tf=tf: self._on_bar_update(b, hasNew, _tf)

        try:
            self._ticker = self.ib.ib.reqMktData(self._contract, '', False, False)
            self._ticker.updateEvent += self._on_tick
            logger.info("Streaming market data subscribed")
        except Exception as e:
            logger.warning("Market data subscription failed: %s", e)

    def _on_bar_update(self, bars, hasNewBar: bool, tf: TF) -> None:
        if not hasNewBar or not self._running:
            return
        asyncio.create_task(self._process_bar_event(bars, tf))

    async def _process_bar_event(self, bars, tf: TF) -> None:
        async with self._eval_lock:
            try:
                b = bars[-1]
                ts = self._normalize_ts(b.date if hasattr(b, 'date') else None)
                bar = Bar(
                    ts=ts, open=b.open, high=b.high,
                    low=b.low, close=b.close, volume=b.volume,
                )

                if tf == TF.M5:
                    now_et = datetime.now(ET)
                    await self.positions.check_catastrophic_5m(bar.high, bar.low, now_et)
                    await self.positions.check_profit_target_5m(bar.high, bar.low, now_et)
                    return

                series = {TF.H1: self.h1, TF.H4: self.h4, TF.D1: self.daily}[tf]
                if series.bars and series.bars[-1].ts == bar.ts:
                    return
                series.add_bar(bar)

                if tf == TF.H1:
                    self.pivots_1h.on_bar(bar, self.h1)
                    self.signals.tick_bars()
                    self._bar_idx_1h += 1
                    self._signal_ring.append(self._snapshot_signal_state())
                    await self._on_eval_tick()
                elif tf == TF.H4:
                    self.pivots_4h.on_bar(bar, self.h4)
                elif tf == TF.D1:
                    self.vol.update(self.daily)
                    self._throttle.update_equity(self.equity)
                    self._throttle.daily_reset()
                    self._ts_history.append(trend_strength(self.daily))
                    ts_3d = self._ts_history[-4] if len(self._ts_history) >= 4 else None
                    self.signals.update_trend_strength_3d(ts_3d)

            except Exception as e:
                logger.error("Bar event error (%s): %s", tf.value, e, exc_info=True)

    def _on_tick(self, ticker) -> None:
        if hasattr(ticker, 'bid') and ticker.bid and ticker.bid > 0:
            self._bid = ticker.bid
        if hasattr(ticker, 'ask') and ticker.ask and ticker.ask > 0:
            self._ask = ticker.ask

    def _should_run_once(self, action: str, now_et: datetime) -> bool:
        last = self._last_session_actions.get(action)
        if last and last.date() == now_et.date():
            return False
        self._last_session_actions[action] = now_et
        return True

    async def _session_timer(self) -> None:
        while self._running:
            try:
                now_et = datetime.now(ET)
                if is_pre_halt_cancel(now_et) and self._should_run_once("pre_halt_cancel", now_et):
                    async with self._eval_lock:
                        for s in list(self.exec.pending_setups):
                            self.risk.release_pending_risk(s.direction, s.armed_risk_r)
                        await self.exec.cancel_all_pending_entries("PRE_HALT")

                if now_et.minute % 5 == 0 and self._should_run_once(f"equity_{now_et.hour}_{now_et.minute}", now_et):
                    await self._refresh_equity()

                await asyncio.sleep(15)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Session timer error: %s", e)

    async def _on_eval_tick(self) -> None:
        now_et = datetime.now(ET)
        logger.info("Eval tick at %s", now_et.strftime("%H:%M"))

        last_price = self.h1.last_close

        await self.positions.manage_all(
            now_et, last_price, self.h1, self.h4, self.daily, self.news,
            pivots_1h=self.pivots_1h, bid=self._bid, ask=self._ask,
        )

        await self._ensure_stops()

        for setup in self.exec.expire_setups(now_et):
            self.risk.release_pending_risk(setup.direction, setup.armed_risk_r)
            await self.exec.cancel_setup(setup, "TTL_EXPIRED")

        await self._invalidate_broken_setups()
        await self._process_spread_rechecks(now_et)
        self._expire_signatures(now_et)

        block = get_session_block(now_et)

        # Session transition tracking (#17)
        for pos in self.positions.positions:
            if pos._last_session and pos._last_session != block.value:
                ur = unrealized_r(pos, last_price)
                pos.session_transitions.append({
                    "from_session": pos._last_session,
                    "to_session": block.value,
                    "transition_time": now_et.isoformat(),
                    "unrealized_pnl_r": round(ur, 4) if ur is not None else None,
                    "bars_held": pos.bars_held_1h,
                    "price_at_transition": last_price,
                })
            pos._last_session = block.value

        if entries_allowed(block) and not is_halt(now_et) and not is_reopen_dead(now_et):
            await self._detect_and_arm(now_et, last_price)

    async def _detect_and_arm(self, now_et: datetime, last_price: float) -> None:
        # Drawdown throttle: daily loss cap halt (matches backtest helix_engine.py:667)
        if DRAWDOWN_THROTTLE_ENABLED and self._throttle.daily_halted:
            return

        # DOW block: Monday + Wednesday
        if now_et.weekday() in DOW_BLOCKED:
            return

        candidates: list[Setup] = []

        # Class M (on 1H bar close)
        m_setups = self.signals.detect_class_M(
            self.pivots_1h, self.h1, self.h4, self.daily, now_et)
        candidates.extend(m_setups)

        # Class F — DISABLED (backtest use_class_F=False, PF=0.50, -$6.3k)
        # Ablation confirmed fast re-entry trades are net losers.

        # Class T: 4H trend continuation (longs only)
        t_setups = self.signals.detect_class_T(
            self.pivots_1h, self.h1, self.h4, self.daily, now_et)
        # Suppress if Class M fired same bar same direction
        m_dirs = {s.direction for s in m_setups}
        t_setups = [s for s in t_setups if s.direction not in m_dirs]
        # Suppress if M placed recently same direction
        t_setups = [s for s in t_setups
                    if (self._bar_idx_1h - self._last_m_bar.get(s.direction, -999))
                       >= CLASS_T_MIN_BARS_SINCE_M]
        candidates.extend(t_setups)

        # Phase 2B: emit indicator snapshot at signal evaluation
        if self._kit.active:
            try:
                best = candidates[0] if candidates else None
                self._kit.on_indicator_snapshot(
                    pair=self.nq_inst.symbol,
                    indicators={
                        "ema_fast": self.h1.ema_fast(),
                        "ema_slow": self.h1.ema_slow(),
                        "atr_14": self.h1.current_atr(),
                        "macd_line": self.h1.macd_line_now(),
                        "macd_hist": self.h1.macd_hist_now(),
                        "vol_pct": self.vol.vol_pct,
                        "trend_strength": trend_strength(self.daily),
                    },
                    signal_name=f"helix_class_{best.cls.value}" if best else "helix_eval",
                    signal_strength=best.alignment_score / 3.0 if best else 0.0,
                    decision="enter" if candidates else "skip",
                    strategy_type="helix",
                    exchange_timestamp=now_et,
                    context={
                        "session": get_session_block(now_et).value,
                        "contract_month": getattr(self._contract, 'lastTradeDateOrContractMonth', ''),
                        "signal_class": best.cls.value if best else "",
                        "concurrent_positions": len(self.positions.positions),
                        "drawdown_tier": self._dd_tier_name(),
                    },
                )
            except Exception:
                pass

        # Record signal detection timestamps (#16)
        for setup in candidates:
            self._cascade_ts[setup.setup_id] = now_et

        if not candidates:
            return

        # Priority: M > F > T
        candidates.sort(key=lambda s: _priority(s))

        active_count = len(self.positions.positions)

        for setup in candidates:
            if active_count >= MAX_CONCURRENT_POSITIONS:
                self._log_missed(setup, "max_concurrent", f"active={active_count}")
                continue

            # Dedup: Class T uses breakout_pivot.ts instead of P1.ts (P1=None)
            sig = (
                setup.P1.ts if setup.P1 else (
                    setup.breakout_pivot.ts if setup.breakout_pivot else None),
                setup.P2.ts if setup.P2 else None,
                setup.direction,
                setup.cls.value,
            )
            if sig in self._placed_signatures:
                continue

            # Short min score >= 1
            if setup.direction == -1 and setup.alignment_score < 1:
                self._log_missed(setup, "short_min_score", f"score={setup.alignment_score}")
                continue

            # Short ETH_QUALITY_AM block (unless alignment >= 2)
            block = get_session_block(now_et)
            if setup.direction == -1 and block == SessionBlock.ETH_QUALITY_AM and setup.alignment_score < 2:
                self._log_missed(setup, "ETH_QUALITY_AM_short", f"score={setup.alignment_score}")
                continue

            # Duplicate blocking
            dup_blocking = False
            for pos in self.positions.positions:
                if pos.direction == setup.direction and pos.origin_class == setup.cls:
                    cur_r = unrealized_r(pos, last_price, self.nq_inst.point_value)
                    if cur_r < DUPLICATE_OVERRIDE_MIN_R:
                        dup_blocking = True
                        break
            if dup_blocking:
                self._log_missed(setup, "duplicate_position", f"cur_r={cur_r:.2f}")
                continue

            # Pending dedup
            if any(s.cls == setup.cls and s.direction == setup.direction
                   for s in self.exec.pending_setups):
                continue

            # Gate check
            sess_mult = session_size_mult(block)
            gate = check_gates(
                setup=setup, now_et=now_et, h1=self.h1, daily=self.daily,
                vol=self.vol, news=self.news, bid=self._bid, ask=self._ask,
                open_risk_r=self.risk.open_risk_r,
                pending_risk_r=self.risk.pending_risk_r,
                dir_risk_r=self.risk.dir_risk_r.get(setup.direction, 0.0),
                heat_cap_r=self.risk.heat_cap_r(),
                heat_cap_dir_r=self.risk.heat_cap_dir_r(),
                collect_all=True,
            )

            if not gate:
                if "spread_" in gate.reason and gate.reason != "spread_missing":
                    if not any(s is setup for s, _ in self._spread_recheck):
                        self._spread_recheck.append((setup, SPREAD_RECHECK_BARS))
                self._log_missed(setup, f"gate_{gate.reason}", gate.reason)
                continue

            # Stash gate results for instrumentation
            setup._filter_decisions = self._build_gate_filter_decisions(setup, now_et)

            # RTH_DEAD enhanced sizing
            ts_daily = trend_strength(self.daily)
            rtd_mult = self.risk.rtd_dead_enhanced_mult(
                block, setup.alignment_score, ts_daily)
            sess_mult *= rtd_mult

            # ETH_EUROPE regime sizing
            if block == SessionBlock.ETH_EUROPE:
                sess_mult *= self.risk.eth_europe_regime_mult(block)

            # Size
            unit_risk = self.risk.compute_unit1_risk_usd(setup)
            setup.unit1_risk_usd = unit_risk
            setup.setup_size_mult = self.risk.setup_size_mult(setup)
            setup.session_size_mult = sess_mult

            hour_mult = self.risk.hour_size_mult(now_et.hour)
            dow_mult = self.risk.dow_size_mult(now_et.weekday())
            contracts = self.risk.size_contracts(
                setup, sess_mult, hour_mult=hour_mult, dow_mult=dow_mult)

            # Drawdown throttle (v7: disabled by default)
            if DRAWDOWN_THROTTLE_ENABLED:
                dd_mult = self._throttle.dd_size_mult
                if dd_mult <= 0.0:
                    self._log_missed(setup, "drawdown_throttle", f"dd_mult={dd_mult:.2f}")
                    continue
                if dd_mult < 1.0:
                    contracts = max(1, int(contracts * dd_mult))

            # Vol 50-80 sizing penalty
            if VOL_50_80_SIZING_MULT < 1.0 and 50 < self.vol.vol_pct < 80 and contracts > 1:
                contracts = max(1, int(contracts * VOL_50_80_SIZING_MULT))

            if contracts < 1:
                self._log_missed(setup, "sizing_zero", f"contracts={contracts}")
                continue

            risk_r = self.risk.compute_risk_r(
                setup.entry_stop, setup.stop0, contracts, unit_risk)
            setup.armed_risk_r = risk_r

            armed = await self.exec.arm_setup(setup, now_et, contracts)
            if armed:
                self.risk.add_pending_risk(setup.direction, risk_r)
                self._placed_signatures.add(sig)
                self._sig_expiry[sig] = now_et + timedelta(seconds=setup.ttl_seconds())

                # Track M placements for Class T suppression
                if setup.cls == SetupClass.M:
                    self._last_m_bar[setup.direction] = self._bar_idx_1h

                if setup.direction == 1 and last_price > setup.entry_stop:
                    await self.exec.place_catch_up(setup, last_price, self.h1.current_atr(), now_et)
                elif setup.direction == -1 and last_price < setup.entry_stop:
                    await self.exec.place_catch_up(setup, last_price, self.h1.current_atr(), now_et)

    async def _invalidate_broken_setups(self) -> None:
        last_price = self.h1.last_close
        for setup in list(self.exec.pending_setups):
            if setup.placed_ts is not None:
                placed_age_h = (datetime.now(ET) - setup.placed_ts).total_seconds() / 3600
                if placed_age_h >= 1.0:
                    if setup.direction == 1 and last_price > setup.entry_stop:
                        self.risk.release_pending_risk(setup.direction, setup.armed_risk_r)
                        await self.exec.cancel_setup(setup, "EOB_BACKSTOP_LONG")
                        continue
                    elif setup.direction == -1 and last_price < setup.entry_stop:
                        self.risk.release_pending_risk(setup.direction, setup.armed_risk_r)
                        await self.exec.cancel_setup(setup, "EOB_BACKSTOP_SHORT")
                        continue

            if setup.P2 is None:
                continue
            if setup.direction == 1:
                detector = self.pivots_1h
                recent_pl = detector.most_recent_pivot_low()
                if recent_pl and recent_pl.ts > setup.P2.ts and recent_pl.price <= setup.P2.price:
                    self.risk.release_pending_risk(setup.direction, setup.armed_risk_r)
                    await self.exec.cancel_setup(setup, "STRUCTURE_INVALID_PL")
            else:
                detector = self.pivots_1h
                recent_ph = detector.most_recent_pivot_high()
                if recent_ph and recent_ph.ts > setup.P2.ts and recent_ph.price >= setup.P2.price:
                    self.risk.release_pending_risk(setup.direction, setup.armed_risk_r)
                    await self.exec.cancel_setup(setup, "STRUCTURE_INVALID_PH")

    async def _process_spread_rechecks(self, now_et: datetime) -> None:
        remaining = []
        for setup, bars_left in self._spread_recheck:
            bars_left -= 1
            gate = check_gates(
                setup=setup, now_et=now_et, h1=self.h1, daily=self.daily,
                vol=self.vol, news=self.news, bid=self._bid, ask=self._ask,
                open_risk_r=self.risk.open_risk_r, pending_risk_r=self.risk.pending_risk_r,
                dir_risk_r=self.risk.dir_risk_r.get(setup.direction, 0.0),
                heat_cap_r=self.risk.heat_cap_r(), heat_cap_dir_r=self.risk.heat_cap_dir_r(),
            )
            if gate:
                block = get_session_block(now_et)
                sess_mult = session_size_mult(block)
                # RTH_DEAD enhanced sizing
                ts_daily = trend_strength(self.daily)
                rtd_mult = self.risk.rtd_dead_enhanced_mult(
                    block, setup.alignment_score, ts_daily)
                sess_mult *= rtd_mult
                # ETH_EUROPE regime sizing
                if block == SessionBlock.ETH_EUROPE:
                    sess_mult *= self.risk.eth_europe_regime_mult(block)
                setup.unit1_risk_usd = self.risk.compute_unit1_risk_usd(setup)
                setup.setup_size_mult = self.risk.setup_size_mult(setup)
                setup.session_size_mult = sess_mult
                hour_mult = self.risk.hour_size_mult(now_et.hour)
                dow_mult = self.risk.dow_size_mult(now_et.weekday())
                contracts = self.risk.size_contracts(
                    setup, sess_mult, hour_mult=hour_mult, dow_mult=dow_mult)
                if DRAWDOWN_THROTTLE_ENABLED:
                    dd_m = self._throttle.dd_size_mult
                    if dd_m <= 0.0:
                        contracts = 0
                    elif dd_m < 1.0:
                        contracts = max(1, int(contracts * dd_m))
                if contracts >= 1:
                    risk_r = self.risk.compute_risk_r(
                        setup.entry_stop, setup.stop0, contracts, setup.unit1_risk_usd)
                    setup.armed_risk_r = risk_r
                    armed = await self.exec.arm_setup(setup, now_et, contracts)
                    if armed:
                        self.risk.add_pending_risk(setup.direction, risk_r)
            elif bars_left > 0:
                remaining.append((setup, bars_left))
        self._spread_recheck = remaining

    # ── OMS fill handling ───────────────────────────────────────────

    async def on_fill(self, oms_order_id: str, fill_price: float, qty: int, now_et: datetime) -> None:
        for setup in list(self.exec.pending_setups):
            if setup.entry_oms_id == oms_order_id or setup.catchup_oms_id == oms_order_id:
                await self._handle_entry_fill(setup, fill_price, qty, now_et)
                return
        for pos in list(self.positions.positions):
            if pos.stop_oms_id == oms_order_id:
                await self._handle_stop_fill(pos, fill_price, qty)
                return
        for pos in list(self.positions.positions):
            if oms_order_id in pos.exit_oms_ids:
                self._reconcile_exit_fill(pos, oms_order_id, fill_price, qty)
                return

    async def _handle_entry_fill(
        self, setup: Setup, fill_price: float, qty: int, now_et: datetime,
    ) -> None:
        atr1 = self.h1.current_atr()
        is_teleport, is_catastrophic = self.exec.check_teleport(
            fill_price, setup.entry_stop, now_et, atr1)

        if is_catastrophic:
            logger.warning("CATASTROPHIC fill %s: %.2f vs %.2f", setup.setup_id, fill_price, setup.entry_stop)
            await self.oms.submit_intent(Intent(
                intent_type=IntentType.FLATTEN,
                strategy_id=STRATEGY_ID,
                instrument_symbol=self.nq_inst.symbol,
            ))
            self.risk.release_pending_risk(setup.direction, setup.armed_risk_r)
            setup.state = SetupState.EXITED
            if setup in self.exec.pending_setups:
                self.exec.pending_setups.remove(setup)
            return

        pos = self.positions.open_position(setup, fill_price, qty, now_et)
        pos.teleport_penalty = is_teleport
        setup.state = SetupState.FILLED

        # Session transition tracking (#17)
        block = get_session_block(now_et)
        pos.entry_session = block.value
        pos._last_session = block.value

        block = get_session_block(now_et)
        self.diagnostics.start_tracking(
            pos=pos, setup=setup, vol_pct=self.vol.vol_pct,
            session_block=block.value, atr_1h=atr1,
            atr_daily=self.daily.current_atr(),
        )

        risk_r = self.risk.compute_risk_r(fill_price, setup.stop0, qty, setup.unit1_risk_usd)
        pos.current_risk_r = risk_r
        self.risk.promote_to_open(setup.direction, setup.armed_risk_r)
        delta = risk_r - setup.armed_risk_r
        if abs(delta) > 1e-9:
            self.risk.adjust_open_risk(setup.direction, delta)

        await self.exec.place_stop(pos, setup.stop0)

        if setup in self.exec.pending_setups:
            self.exec.pending_setups.remove(setup)

        logger.info("Entry fill %s: %d @ %.2f, stop=%.2f, teleport=%s",
                     setup.setup_id, qty, fill_price, setup.stop0, is_teleport)

        if self._kit.active:
            try:
                from .session import get_session_block
                block = get_session_block(datetime.now(ET))
                config_snapshot = snapshot_config_module(strategy_config)

                # Execution cascade timestamps (#16)
                signal_detected_at = self._cascade_ts.pop(setup.setup_id, now_et)
                fill_received_at = now_et
                exec_ts = {
                    "signal_detected_at": signal_detected_at.isoformat(),
                    "fill_received_at": fill_received_at.isoformat(),
                    "cascade_duration_ms": round(
                        (fill_received_at - signal_detected_at).total_seconds() * 1000
                    ),
                }

                # Capture portfolio state at entry (G4)
                portfolio_state = None
                try:
                    risk_state = await self.oms.get_portfolio_risk()
                    portfolio_state = {
                        "total_exposure_r": risk_state.open_risk_R,
                        "daily_realized_pnl": risk_state.daily_realized_pnl,
                        "daily_realized_r": risk_state.daily_realized_R,
                        "weekly_realized_pnl": risk_state.weekly_realized_pnl,
                        "weekly_realized_r": risk_state.weekly_realized_R,
                        "open_risk_r": risk_state.open_risk_R,
                        "pending_entry_risk_r": risk_state.pending_entry_risk_R,
                        "halted": risk_state.halted,
                    }
                except Exception:
                    portfolio_state = None

                self._kit.log_entry(
                    trade_id=setup.setup_id,
                    pair=self.nq_inst.symbol,
                    side="LONG" if setup.direction == 1 else "SHORT",
                    entry_price=fill_price,
                    position_size=qty,
                    position_size_quote=qty * fill_price * self.nq_inst.point_value,
                    entry_signal=f"Class_{setup.cls.value}",
                    entry_signal_id=setup.setup_id,
                    entry_signal_strength=setup.alignment_score / 3.0,
                    expected_entry_price=setup.entry_stop,
                    strategy_params={
                        "stop0": setup.stop0,
                        "class": setup.cls.value,
                        "alignment_score": setup.alignment_score,
                        **config_snapshot,
                    },
                    signal_factors=[
                        {"factor_name": "alignment_score", "factor_value": setup.alignment_score,
                         "threshold": 1, "contribution": setup.alignment_score / 3.0},
                    ],
                    filter_decisions=getattr(setup, '_filter_decisions', []),
                    sizing_inputs={
                        "unit_risk_usd": setup.unit1_risk_usd,
                        "setup_size_mult": setup.setup_size_mult,
                        "session_size_mult": setup.session_size_mult,
                        "hour_mult": self.risk.hour_size_mult(datetime.now(ET).hour),
                        "dow_mult": self.risk.dow_size_mult(datetime.now(ET).weekday()),
                        "dd_mult": self._throttle.dd_size_mult,
                        "contracts": qty,
                    },
                    session_type=block.value,
                    contract_month=getattr(self._contract, 'lastTradeDateOrContractMonth', ''),
                    concurrent_positions=len(self.positions.positions),
                    drawdown_pct=self._throttle.dd_pct if hasattr(self._throttle, 'dd_pct') else None,
                    drawdown_tier=self._dd_tier_name(),
                    drawdown_size_mult=self._throttle.dd_size_mult,
                    portfolio_state=portfolio_state,
                    signal_evolution=self._build_signal_evolution(),
                    execution_timestamps=exec_ts,
                )

                # Phase 2B: emit orderbook context at entry
                if self._bid is not None and self._ask is not None:
                    self._kit.on_orderbook_context(
                        pair=self.nq_inst.symbol,
                        best_bid=self._bid,
                        best_ask=self._ask,
                        trade_context="entry",
                        related_trade_id=setup.setup_id,
                        exchange_timestamp=now_et,
                    )
            except Exception:
                pass

    async def _handle_stop_fill(self, pos: PositionState, fill_price: float, qty: int) -> None:
        pv = self.nq_inst.point_value
        actual_pnl = (fill_price - pos.avg_entry) * pos.direction * pv * qty
        pos.realized_partial_usd += actual_pnl
        pos.contracts = max(0, pos.contracts - qty)
        if pos.contracts <= 0:
            now_et = datetime.now(ET)
            exit_reason = "TRAILING_STOP" if pos.trailing_active else "INITIAL_STOP"
            r_mult = (pos.realized_partial_usd / (pos.unit1_risk_usd * pos.entry_contracts)
                      if pos.unit1_risk_usd > 0 and pos.entry_contracts > 0 else 0.0)
            self._throttle.update_equity(self.equity)
            self._throttle.record_trade_close(r_mult)
            self.positions._close_position(pos, exit_reason, fill_price, now_et)
            if self._kit.active:
                try:
                    self._kit.log_exit(
                        trade_id=pos.origin_setup_id,
                        exit_price=fill_price,
                        exit_reason=exit_reason,
                        mfe_r=pos.peak_mfe_r,
                        mae_r=pos.peak_mae_r,
                        mfe_price=pos.highest_since_entry if pos.direction == 1 else pos.lowest_since_entry,
                        mae_price=pos.lowest_since_entry if pos.direction == 1 else pos.highest_since_entry,
                        session_transitions=pos.session_transitions or None,
                    )
                    if self._bid is not None and self._ask is not None:
                        self._kit.on_orderbook_context(
                            pair=self.nq_inst.symbol,
                            best_bid=self._bid,
                            best_ask=self._ask,
                            trade_context="exit",
                            related_trade_id=pos.origin_setup_id,
                            exchange_timestamp=now_et,
                        )
                except Exception:
                    pass

    def _reconcile_exit_fill(
        self, pos: PositionState, oms_order_id: str, fill_price: float, qty: int,
    ) -> None:
        pv = self.nq_inst.point_value
        actual_pnl = (fill_price - pos.avg_entry) * pos.direction * pv * qty
        if oms_order_id in pos.pending_exit_estimates:
            estimated = pos.pending_exit_estimates.pop(oms_order_id)
            correction = actual_pnl - estimated
            pos.realized_partial_usd += correction
        else:
            pos.realized_partial_usd += actual_pnl
        if oms_order_id in pos.exit_oms_ids:
            pos.exit_oms_ids.remove(oms_order_id)
        if pos.contracts <= 0:
            now_et = datetime.now(ET)
            r_mult = (pos.realized_partial_usd / (pos.unit1_risk_usd * pos.entry_contracts)
                      if pos.unit1_risk_usd > 0 and pos.entry_contracts > 0 else 0.0)
            self._throttle.update_equity(self.equity)
            self._throttle.record_trade_close(r_mult)
            self.positions._close_position(pos, "EXIT_FILL", fill_price, now_et)
            if self._kit.active:
                try:
                    self._kit.log_exit(
                        trade_id=pos.origin_setup_id,
                        exit_price=fill_price,
                        exit_reason="EXIT_FILL",
                        mfe_r=pos.peak_mfe_r,
                        mae_r=pos.peak_mae_r,
                        mfe_price=pos.highest_since_entry if pos.direction == 1 else pos.lowest_since_entry,
                        mae_price=pos.lowest_since_entry if pos.direction == 1 else pos.highest_since_entry,
                        session_transitions=pos.session_transitions or None,
                    )
                    if self._bid is not None and self._ask is not None:
                        self._kit.on_orderbook_context(
                            pair=self.nq_inst.symbol,
                            best_bid=self._bid,
                            best_ask=self._ask,
                            trade_context="exit",
                            related_trade_id=pos.origin_setup_id,
                        )
                except Exception:
                    pass

    def _handle_order_cancelled(self, oms_order_id: str) -> None:
        for setup in list(self.exec.pending_setups):
            if setup.entry_oms_id == oms_order_id or setup.catchup_oms_id == oms_order_id:
                self.risk.release_pending_risk(setup.direction, setup.armed_risk_r)
                setup.state = SetupState.CANCELED
                if setup in self.exec.pending_setups:
                    self.exec.pending_setups.remove(setup)
                return
        for pos in self.positions.positions:
            if pos.stop_oms_id == oms_order_id:
                pos.stop_oms_id = None
                logger.warning("Stop cancelled for %s — will re-place on next eval", pos.pos_id)
                return

    def _handle_order_rejected(self, oms_order_id: str) -> None:
        self._handle_order_cancelled(oms_order_id)

    async def _ensure_stops(self) -> None:
        for pos in self.positions.positions:
            if pos.contracts <= 0:
                continue
            if pos.stop_oms_id is None:
                await self.exec.place_stop(pos, pos.stop_price)

    async def _refresh_equity(self) -> None:
        try:
            accounts = self.ib.ib.managedAccounts()
            if accounts:
                summary = await self.ib.ib.accountSummaryAsync(accounts[0])
                for item in summary:
                    if item.tag == "NetLiquidation" and item.currency == "USD":
                        self.equity = float(item.value) * self._equity_alloc_pct
                        self.risk.update_equity(self.equity)
                        self._throttle.update_equity(self.equity)
                        break
        except Exception:
            pass

    def _expire_signatures(self, now_et: datetime) -> None:
        expired = [sig for sig, exp in self._sig_expiry.items() if now_et >= exp]
        for sig in expired:
            self._placed_signatures.discard(sig)
            del self._sig_expiry[sig]

    async def _listen_oms_events(self) -> None:
        queue = self.oms.stream_events(STRATEGY_ID)
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=5.0)
                if not hasattr(event, 'event_type'):
                    continue
                et = event.event_type
                oms_id = getattr(event, 'oms_order_id', None)
                payload = getattr(event, 'payload', None) or {}

                if et == OMSEventType.FILL:
                    if oms_id and payload:
                        async with self._eval_lock:
                            await self.on_fill(
                                oms_order_id=oms_id,
                                fill_price=payload.get('price', 0),
                                qty=int(payload.get('qty', 0)),
                                now_et=getattr(event, 'timestamp', datetime.now(ET)),
                            )
                elif et in (OMSEventType.ORDER_CANCELLED, OMSEventType.ORDER_EXPIRED):
                    if oms_id:
                        async with self._eval_lock:
                            self._handle_order_cancelled(oms_id)
                elif et == OMSEventType.ORDER_REJECTED:
                    if oms_id:
                        async with self._eval_lock:
                            self._handle_order_rejected(oms_id)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("OMS event error: %s", e, exc_info=True)


def _priority(setup: Setup) -> int:
    """Lower = higher priority. M > F > T."""
    if setup.cls == SetupClass.M and not setup.is_reentry:
        return 0
    if setup.cls == SetupClass.M and setup.is_reentry:
        return 1
    if setup.cls == SetupClass.F:
        return 2
    if setup.cls == SetupClass.T:
        return 3
    return 4
