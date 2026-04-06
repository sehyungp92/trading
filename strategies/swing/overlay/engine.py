"""EMA crossover overlay engine — deploys idle capital into ETFs.

Daily-rebalancing capital allocator that places market orders directly via the
IB API, bypassing OMS entirely.  Runs on the same 16:15 ET daily schedule as
KeltnerEngine.  State persisted to JSON for crash recovery.

Ported from backtest/engine/unified_portfolio_engine.py (legacy "ema" mode).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import OverlayConfig

logger = logging.getLogger(__name__)

try:
    import zoneinfo
    _ET = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    _ET = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# EMA computation (ported from backtest)
# ---------------------------------------------------------------------------

def _compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed — matches backtest/_overlay_ema exactly."""
    out = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return out
    out[period - 1] = np.mean(series[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OverlayEngine:
    """Async live engine for idle-capital EMA crossover overlay."""

    def __init__(
        self,
        ib_session: Any,
        equity: float,
        config: OverlayConfig,
        market_calendar: Any | None = None,
        instrumentation: Any | None = None,
        equity_offset: float = 0.0,
        db_pool: Any | None = None,
    ) -> None:
        self._ib = ib_session
        self._equity = equity
        self._config = config
        self._market_cal = market_calendar
        self._instr = instrumentation
        self._equity_offset = equity_offset  # paper capital offset applied on refresh
        self._db_pool = db_pool

        # Resolved IB contracts: symbol -> Contract
        self._contracts: dict[str, Any] = {}

        # Current overlay shares (loaded from / saved to state file)
        self._shares: dict[str, int] = {sym: 0 for sym in config.symbols}
        self._last_rebalance_date: str = ""
        # Trade IDs for open overlay positions (persisted for exit instrumentation)
        self._entry_trade_ids: dict[str, str] = {}
        # Last EMA crossover signals (persisted for transition detection)
        self._last_signals: dict[str, bool] = {}

        # Async state
        self._daily_task: asyncio.Task | None = None
        self._running = False

        # BRS bear regime check (wired by coordinator)
        self._bear_regime_check: Any = None

    def set_bear_regime_check(self, fn) -> None:
        """Register callable returning True when BRS detects BEAR_STRONG."""
        self._bear_regime_check = fn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Qualify ETF contracts, load state, launch daily scheduler."""
        logger.info("Overlay engine starting …")
        self._running = True

        # Resolve ETF contracts (same pattern as KeltnerEngine)
        for sym in self._config.symbols:
            try:
                from ib_async import Stock
                contract = Stock(sym, "SMART", "USD")
                qualified = await self._ib.ib.qualifyContractsAsync(contract)
                if qualified:
                    self._contracts[sym] = qualified[0]
            except Exception as e:
                logger.warning("Overlay: could not resolve contract for %s: %s", sym, e)

        # Load persisted state
        self._load_state()

        # Launch daily scheduler
        self._daily_task = asyncio.create_task(self._daily_scheduler())

        logger.info(
            "Overlay engine started (symbols: %s, shares: %s)",
            list(self._contracts.keys()), self._shares,
        )

    async def stop(self) -> None:
        """Cancel scheduler, save state."""
        logger.info("Overlay engine stopping …")
        self._running = False

        if self._daily_task:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info("Overlay engine stopped")

    # ------------------------------------------------------------------
    # Daily scheduler (same pattern as KeltnerEngine)
    # ------------------------------------------------------------------

    async def _daily_scheduler(self) -> None:
        """Sleep until 16:15 ET each trading day, then rebalance."""
        while self._running:
            now = datetime.now(timezone.utc)
            try:
                now_et = now.astimezone(_ET)
            except Exception:
                now_et = now
            target = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
            if target <= now_et:
                target += timedelta(days=1)
            # Skip weekends and holidays
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
                await self._daily_rebalance()
            except Exception:
                logger.exception("Overlay: error in daily rebalance")

    # ------------------------------------------------------------------
    # Rebalance logic (ported from backtest legacy "ema" mode)
    # ------------------------------------------------------------------

    async def _daily_rebalance(self) -> None:
        """Fetch bars, compute EMAs, rebalance overlay positions."""
        logger.info("Overlay: === Daily rebalance ===")

        # 1. Refresh equity from IB
        await self._refresh_equity()

        available = max(self._equity * self._config.max_equity_pct, 0.0)

        # 2-3. Fetch bars and compute EMAs per symbol
        signals: dict[str, bool] = {}
        prices: dict[str, float] = {}

        for sym in self._config.symbols:
            contract = self._contracts.get(sym)
            if not contract:
                signals[sym] = False
                continue

            try:
                bars = await self._ib.ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr="200 D",
                    barSizeSetting="1 day", whatToShow="TRADES",
                    useRTH=True, formatDate=1,
                )
            except Exception:
                logger.warning("Overlay: failed to fetch bars for %s", sym)
                signals[sym] = False
                continue

            if not bars or len(bars) < 50:
                logger.warning(
                    "Overlay: insufficient bars for %s (%d)",
                    sym, len(bars) if bars else 0,
                )
                signals[sym] = False
                continue

            closes = np.array([b.close for b in bars], dtype=float)
            prices[sym] = float(closes[-1])

            # Get EMA periods (per-symbol override or default)
            fast, slow = self._config.ema_overrides.get(
                sym, (self._config.ema_fast, self._config.ema_slow),
            )

            ema_fast = _compute_ema(closes, fast)
            ema_slow = _compute_ema(closes, slow)

            # 4. Determine bullish: fast EMA > slow EMA at latest bar
            if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
                signals[sym] = False
            else:
                signals[sym] = bool(ema_fast[-1] > ema_slow[-1])

            logger.info(
                "Overlay: %s EMA(%d)=%.2f EMA(%d)=%.2f → %s",
                sym, fast, ema_fast[-1], slow, ema_slow[-1],
                "BULLISH" if signals[sym] else "BEARISH",
            )

        # 4b. Log signal transitions via coordination logger
        if self._last_signals:
            for sym in self._config.symbols:
                old = self._last_signals.get(sym)
                new = signals.get(sym, False)
                if old is not None and old != new:
                    try:
                        if self._instr and getattr(self._instr, 'coordination_logger', None):
                            self._instr.coordination_logger.log_action(
                                action="overlay_signal_change",
                                trigger_strategy="OVERLAY",
                                target_strategy="ALL",
                                symbol=sym,
                                rule="ema_crossover",
                                details={
                                    "old_bullish": old,
                                    "new_bullish": new,
                                    "direction": "BULLISH" if new else "BEARISH",
                                },
                                outcome="emitted",
                            )
                    except Exception:
                        pass
        self._last_signals = dict(signals)

        # 5. Compute target shares
        if self._config.weights is None:
            bullish_w = {s: 1.0 for s in self._config.symbols if signals.get(s)}
        else:
            bullish_w = {
                s: self._config.weights.get(s, 1.0)
                for s in self._config.symbols if signals.get(s)
            }
        total_w = sum(bullish_w.values())

        target_shares: dict[str, int] = {}
        for sym in self._config.symbols:
            price = prices.get(sym, 0.0)
            if signals.get(sym) and price > 0 and total_w > 0:
                alloc = available * bullish_w[sym] / total_w
                target_shares[sym] = int(alloc / price)  # floor to whole shares
            else:
                target_shares[sym] = 0

        # BRS bear regime: block new overlay entries, allow exits of existing
        if self._bear_regime_check:
            try:
                bear_active = self._bear_regime_check()
            except Exception:
                logger.warning("Overlay: bear regime check failed, proceeding without guard")
                bear_active = False
            if bear_active:
                for sym in self._config.symbols:
                    if self._shares.get(sym, 0) == 0 and target_shares.get(sym, 0) > 0:
                        target_shares[sym] = 0
                        logger.info(
                            "Overlay: BRS BEAR_STRONG -- blocking new entry for %s", sym,
                        )

        # 6-8. Compute deltas and place orders
        for sym in self._config.symbols:
            target = target_shares.get(sym, 0)
            current = self._shares.get(sym, 0)
            delta = target - current

            if delta == 0:
                logger.info("Overlay: %s no change (target=%d, current=%d)", sym, target, current)
                continue

            contract = self._contracts.get(sym)
            if not contract:
                continue

            # Detect entry (0 → >0) and exit (>0 → 0) transitions
            entering = current == 0 and target > 0
            exiting = current > 0 and target == 0

            action = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)

            try:
                from ib_async import MarketOrder
                order = MarketOrder(action, qty)
                trade = self._ib.ib.placeOrder(contract, order)
                logger.info(
                    "Overlay: %s %s %d shares (current=%d → target=%d)",
                    sym, action, qty, current, target,
                )

                # Wait for fill (with timeout)
                fill_price = prices.get(sym, 0.0)
                try:
                    await asyncio.wait_for(trade.filledEvent, timeout=60.0)
                    fill_price = trade.orderStatus.avgFillPrice or fill_price
                    logger.info(
                        "Overlay: %s fill confirmed — %d @ %.2f",
                        sym, qty, fill_price,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Overlay: %s fill timeout — order may fill at next RTH open",
                        sym,
                    )

                # Update tracked shares
                self._shares[sym] = target

                # Hook 4: entry instrumentation (0 → >0)
                if self._instr and entering:
                    try:
                        from strategies.swing.instrumentation.src.hooks import safe_instrument
                        regime = self._instr.regime_classifier.current_regime(sym)
                        tid = f"overlay_{sym}_{datetime.now(timezone.utc).isoformat()}"
                        self._entry_trade_ids[sym] = tid
                        safe_instrument(
                            self._instr.trade_logger.log_entry,
                            trade_id=tid,
                            pair=sym,
                            side="LONG",
                            entry_price=fill_price,
                            position_size=float(target),
                            position_size_quote=float(target) * fill_price,
                            entry_signal="ema_crossover_overlay",
                            entry_signal_id=f"overlay_{sym}",
                            entry_signal_strength=0.5,
                            active_filters=[],
                            passed_filters=[],
                            strategy_params={"ema_overrides": str(self._config.ema_overrides.get(sym))},
                            strategy_id="OVERLAY",
                            expected_entry_price=prices.get(sym, 0),
                            market_regime=regime,
                        )
                    except Exception:
                        pass

                # Hook 5: exit instrumentation (>0 → 0) + process scoring
                if self._instr and exiting:
                    try:
                        from strategies.swing.instrumentation.src.hooks import safe_instrument
                        tid = self._entry_trade_ids.pop(sym, f"overlay_{sym}")
                        trade_event = safe_instrument(
                            self._instr.trade_logger.log_exit,
                            trade_id=tid,
                            exit_price=fill_price,
                            exit_reason="EMA_BEARISH",
                        )
                        if trade_event:
                            safe_instrument(
                                self._instr.process_scorer.score_and_write,
                                trade_event.to_dict(),
                                "OVERLAY",
                                self._instr.data_dir,
                            )
                    except Exception:
                        pass

            except Exception:
                logger.exception("Overlay: failed to place order for %s", sym)
                continue

        # Hook 1: regime classification + market snapshot (post-rebalance)
        if self._instr:
            try:
                from strategies.swing.instrumentation.src.hooks import safe_instrument, async_safe_instrument
                for sym in self._config.symbols:
                    await async_safe_instrument(self._instr.regime_classifier.classify, sym)
                    safe_instrument(self._instr.snapshot_service.capture_now, sym)
            except Exception:
                pass

        self._last_rebalance_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._save_state()

        # Persist overlay positions to DB for dashboard visibility
        await self._persist_positions_to_db(prices)

        logger.info("Overlay: rebalance complete — shares: %s", self._shares)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load overlay state from JSON file."""
        path = Path(self._config.state_file)
        if not path.exists():
            logger.info("Overlay: no state file found, starting fresh")
            return
        try:
            data = json.loads(path.read_text())
            self._shares = {sym: data.get("shares", {}).get(sym, 0) for sym in self._config.symbols}
            self._last_rebalance_date = data.get("last_rebalance_date", "")
            self._entry_trade_ids = data.get("entry_trade_ids", {})
            self._last_signals = data.get("last_signals", {})
            logger.info(
                "Overlay: loaded state — shares=%s, last_rebalance=%s",
                self._shares, self._last_rebalance_date,
            )
        except Exception:
            logger.warning("Overlay: failed to load state file, starting fresh")

    def _save_state(self) -> None:
        """Save overlay state to JSON file."""
        path = Path(self._config.state_file)
        data = {
            "shares": self._shares,
            "last_rebalance_date": self._last_rebalance_date,
            "entry_trade_ids": self._entry_trade_ids,
            "last_signals": self._last_signals,
        }
        try:
            path.write_text(json.dumps(data, indent=2))
            logger.debug("Overlay: state saved to %s", path)
        except Exception:
            logger.warning("Overlay: failed to save state file")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_signals(self) -> dict[str, bool]:
        """Return last computed EMA crossover signals per symbol."""
        return dict(self._last_signals)

    def get_positions(self) -> dict[str, int]:
        """Return current overlay share counts."""
        return dict(self._shares)

    def get_state_summary(self) -> str:
        """Return human-readable state summary for logging."""
        parts = [f"{sym}={qty}" for sym, qty in self._shares.items() if qty > 0]
        if not parts:
            return "Overlay: no positions"
        return f"Overlay: {', '.join(parts)} (last rebalance: {self._last_rebalance_date})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _persist_positions_to_db(self, prices: dict[str, float]) -> None:
        """Write current overlay positions to PostgreSQL for dashboard visibility."""
        if not self._db_pool:
            return
        try:
            from libs.oms.persistence.postgres import PgStore
            store = PgStore(self._db_pool)
            now_utc = datetime.now(timezone.utc)
            rows = []
            for sym in self._config.symbols:
                shares = self._shares.get(sym, 0)
                price = prices.get(sym, 0.0)
                notional = shares * price
                pct = notional / self._equity if self._equity > 0 else 0.0
                rows.append({
                    "symbol": sym,
                    "shares": shares,
                    "notional": notional,
                    "pct_of_nav": pct,
                    "rebalance_ts": now_utc,
                })
            await store.upsert_overlay_positions(rows)
            logger.info("Overlay: positions persisted to DB (%d symbols)", len(rows))
        except Exception:
            logger.warning("Overlay: failed to persist positions to DB", exc_info=True)

    async def _refresh_equity(self) -> None:
        """Fetch current account equity from IB (applies paper capital offset)."""
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                summary = await self._ib.ib.accountSummaryAsync(accounts[0])
                for item in summary:
                    if item.tag == "NetLiquidation" and item.currency == "USD":
                        raw = float(item.value)
                        self._equity = raw + self._equity_offset
                        logger.info("Overlay: equity refreshed — $%.2f", self._equity)
                        return
        except Exception:
            logger.warning("Overlay: could not refresh equity, using $%.2f", self._equity)
