"""Thin IB data-source bridges."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)

from ib_async import (
    IB,
    ScannerSubscription,
    TickByTickAllLast,
    TickByTickBidAsk,
)

from libs.broker_ibkr.mapping.contract_factory import ContractFactory

from .config import PROXY_SYMBOLS, ScannerSettings
from .models import CachedSymbol, MinuteBar, QuoteSnapshot


@dataclass
class UniversePoolManager:
    cache: dict[str, CachedSymbol]
    cap: int = 20

    def shortlist(self, scanner_symbols: Iterable[str]) -> list[str]:
        unique = []
        seen = set()
        for symbol in scanner_symbols:
            normalized = symbol.upper()
            if normalized in self.cache and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
            if len(unique) >= self.cap:
                break
        return unique


class MinuteBarBuilder:
    """Build minute bars from streaming last-trade updates."""

    def __init__(self) -> None:
        self._current_minute: datetime | None = None
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0.0
        self._dollar_value = 0.0
        self._last_cumulative_volume = 0.0

    def update(self, ts: datetime, price: float, cumulative_volume: float) -> MinuteBar | None:
        minute = ts.replace(second=0, microsecond=0)
        volume_delta = max(0.0, cumulative_volume - self._last_cumulative_volume)
        self._last_cumulative_volume = max(self._last_cumulative_volume, cumulative_volume)

        if self._current_minute is None:
            self._reset(minute, price, volume_delta)
            return None

        if minute == self._current_minute:
            self._high = max(self._high, price)
            self._low = min(self._low, price)
            self._close = price
            self._volume += volume_delta
            self._dollar_value += price * volume_delta
            return None

        closed = MinuteBar(
            ts=self._current_minute,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            dollar_value=self._dollar_value,
        )
        self._reset(minute, price, volume_delta)
        return closed

    def _reset(self, minute: datetime, price: float, volume_delta: float) -> None:
        self._current_minute = minute
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._volume = volume_delta
        self._dollar_value = price * volume_delta


class TradeFlowWindow:
    """Signed 90-second dollar imbalance from tick-by-tick data."""

    def __init__(self, window_seconds: int = 90) -> None:
        self._window = timedelta(seconds=window_seconds)
        self._events: deque[tuple[datetime, float]] = deque()

    def update(self, ts: datetime, price: float, size: float, bid: float, ask: float) -> None:
        midpoint = 0.0
        if bid > 0 and ask > 0:
            midpoint = (bid + ask) / 2.0
        signed_value = price * size
        if midpoint > 0:
            if price < midpoint:
                signed_value *= -1
        self._events.append((ts, signed_value))
        self._trim(ts)

    def imbalance(self, now: datetime) -> float:
        self._trim(now)
        total = sum(abs(value) for _, value in self._events)
        if total <= 0:
            return 0.0
        return sum(value for _, value in self._events) / total

    def _trim(self, now: datetime) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


class IBScannerSource:
    """Thin wrapper around one or more IB scanner subscriptions."""

    def __init__(self, ib: IB, settings: ScannerSettings) -> None:
        self._ib = ib
        self._settings = settings
        self._handles = []
        self._queue: asyncio.Queue[list[str]] = asyncio.Queue()
        self._latest: set[str] = set()
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for scan_code in self._settings.scan_codes:
            handle = self._ib.reqScannerSubscription(
                ScannerSubscription(
                    numberOfRows=self._settings.rows_per_scan,
                    instrument=self._settings.instrument,
                    locationCode=self._settings.location_code,
                    scanCode=scan_code,
                    abovePrice=self._settings.above_price,
                    aboveVolume=self._settings.above_volume,
                    stockTypeFilter=self._settings.stock_type_filter,
                )
            )
            handle.updateEvent += self._on_update
            self._handles.append(handle)

    async def stop(self) -> None:
        if not self._running:
            return
        for handle in self._handles:
            try:
                handle.updateEvent -= self._on_update
            except Exception:
                pass
            self._ib.cancelScannerSubscription(handle)
        self._handles.clear()
        self._running = False

    async def restart(self) -> None:
        """Stop and re-start scanner subscriptions (e.g. after IBKR reconnect)."""
        await self.stop()
        await self.start()

    async def next_update(self) -> list[str]:
        return await self._queue.get()

    def latest_symbols(self) -> list[str]:
        return sorted(self._latest)

    def _on_update(self, rows) -> None:
        symbols = {
            row.contractDetails.contract.symbol.upper()
            for row in rows
            if getattr(row, "contractDetails", None) and getattr(row.contractDetails, "contract", None)
        }
        if symbols:
            self._latest = symbols
            self._queue.put_nowait(sorted(symbols))


class IBMarketDataSource:
    """Light market-data bridge that feeds quotes, bars, and imbalance to the engine."""

    def __init__(
        self,
        ib: IB,
        contract_factory: ContractFactory,
        on_quote: Callable[[str, QuoteSnapshot, float], Any] | Callable[[str, QuoteSnapshot, float], Awaitable[Any]],
        on_bar: Callable[[str, MinuteBar], Any] | Callable[[str, MinuteBar], Awaitable[Any]],
    ) -> None:
        self._ib = ib
        self._factory = contract_factory
        self._on_quote = on_quote
        self._on_bar = on_bar
        self._builders: dict[str, MinuteBarBuilder] = {}
        self._flows: dict[str, TradeFlowWindow] = {}
        self._processed_ticks: dict[str, int] = {}
        self._cumulative_value: dict[str, float] = {}
        self._contracts: dict[str, Any] = {}
        self._instruments: dict[str, Any] = {}
        self._last_quote_ts: dict[str, datetime] = {}
        self._last_midpoints: dict[str, float] = {}
        self._last_spreads: dict[str, float] = {}
        self._quote_expansion_streaks: dict[str, int] = {}
        self._halted_state: dict[str, bool] = {}
        self._blacklisted: dict[str, datetime] = {}  # symbol -> expiry (UTC)
        self._last_farm_ok_ts: float = 0.0  # monotonic timestamp of last 2104 warning
        self._last_farm_blip_ts: float = 0.0  # monotonic timestamp of last 2103/2119 warning
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._ib.pendingTickersEvent += self._handle_pending_tickers
        self._ib.errorEvent += self._on_ib_error
        self._running = True

    async def stop(self) -> None:
        if not self._running:
            return
        self._ib.pendingTickersEvent -= self._handle_pending_tickers
        self._ib.errorEvent -= self._on_ib_error
        for symbol in list(self._contracts):
            self._remove_symbol(symbol)
        self._running = False

    def invalidate_subscriptions(self) -> None:
        """Clear tracked subscriptions after IBKR reconnect.

        Old subscriptions are already dead on the broker side, so we just
        clear local state.  The next ``ensure_symbols`` call will
        re-resolve and re-subscribe everything.
        """
        self._contracts.clear()
        self._builders.clear()
        self._flows.clear()
        self._processed_ticks.clear()
        self._cumulative_value.clear()
        self._instruments.clear()
        self._last_quote_ts.clear()
        self._last_midpoints.clear()
        self._last_spreads.clear()
        self._quote_expansion_streaks.clear()
        self._halted_state.clear()
        self._blacklisted.clear()

    def _remove_symbol(self, symbol: str) -> None:
        contract = self._contracts.pop(symbol, None)
        if contract is not None:
            self._ib.cancelTickByTickData(contract, "Last")
            self._ib.cancelTickByTickData(contract, "BidAsk")
            self._ib.cancelMktData(contract)
        self._builders.pop(symbol, None)
        self._flows.pop(symbol, None)
        self._processed_ticks.pop(symbol, None)
        self._cumulative_value.pop(symbol, None)
        self._instruments.pop(symbol, None)
        self._last_quote_ts.pop(symbol, None)
        self._last_midpoints.pop(symbol, None)
        self._last_spreads.pop(symbol, None)
        self._quote_expansion_streaks.pop(symbol, None)
        self._halted_state.pop(symbol, None)

    # 10089 = market data subscription required (fatal for symbol)
    # 10189 = tick-by-tick denied (non-fatal, degrade to reqMktData only)
    _BLACKLIST_ERRORS = frozenset({10089})
    _TICK_BY_TICK_ERRORS = frozenset({10189})
    _FARM_BLIP_CODES = frozenset({2103, 2119})  # farm broken / farm connecting
    _FARM_OK_CODES = frozenset({2104})
    _FARM_RECONNECT_GRACE_S = 30.0
    _BLACKLIST_DURATION = timedelta(hours=1)

    def _on_ib_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        # Track farm blip start (2103 = broken, 2119 = connecting)
        if errorCode in self._FARM_BLIP_CODES:
            self._last_farm_blip_ts = time.monotonic()
            return

        # Track farm reconnection events (warning 2104 = "farm connection is OK")
        if errorCode in self._FARM_OK_CODES:
            self._last_farm_ok_ts = time.monotonic()
            # Clear blacklist — symbols rejected during farm blip are now reachable
            if self._blacklisted:
                cleared = list(self._blacklisted.keys())
                self._blacklisted.clear()
                logger.info(
                    "Farm reconnected — cleared blacklist for %s", ", ".join(cleared),
                )
            return

        symbol = getattr(contract, "symbol", "").upper() if contract else ""
        if errorCode in self._TICK_BY_TICK_ERRORS:
            if symbol:
                logger.warning(
                    "Tick-by-tick denied for %s (code %d), continuing with reqMktData only: %s",
                    symbol, errorCode, errorString,
                )
            return
        if errorCode not in self._BLACKLIST_ERRORS:
            return
        if not symbol or symbol not in self._contracts:
            return

        # If this error arrived within the grace window of a farm reconnect,
        # treat it as transient: drop the subscription so the main loop
        # re-subscribes on the next cycle, but do NOT blacklist.
        now_mono = time.monotonic()
        since_farm = min(
            now_mono - self._last_farm_ok_ts,
            now_mono - self._last_farm_blip_ts,
        )
        if since_farm < self._FARM_RECONNECT_GRACE_S:
            logger.info(
                "Transient market-data error for %s (code %d) during farm reconnect "
                "(%.1fs ago) — will retry next cycle: %s",
                symbol, errorCode, since_farm, errorString,
            )
            self._remove_symbol(symbol)
            return

        logger.warning(
            "Market data permission denied for %s (code %d), blacklisting for 1h: %s",
            symbol, errorCode, errorString,
        )
        self._remove_symbol(symbol)
        self._blacklisted[symbol] = datetime.now(timezone.utc) + self._BLACKLIST_DURATION

    async def ensure_symbols(self, instruments: Iterable[Any]) -> None:
        wanted = {instrument.symbol: instrument for instrument in instruments}
        for symbol in list(self._contracts):
            if symbol not in wanted and symbol not in PROXY_SYMBOLS:
                self._remove_symbol(symbol)

        now = datetime.now(timezone.utc)
        for symbol, instrument in wanted.items():
            if symbol in self._contracts:
                continue
            # Skip symbols blacklisted due to permission errors
            if symbol in self._blacklisted:
                if now < self._blacklisted[symbol]:
                    continue
                del self._blacklisted[symbol]
            contract, _ = await self._factory.resolve(symbol=instrument.root or instrument.symbol, instrument=instrument)
            self._contracts[symbol] = contract
            self._instruments[symbol] = instrument
            self._builders[symbol] = MinuteBarBuilder()
            self._flows[symbol] = TradeFlowWindow()
            self._processed_ticks[symbol] = 0
            self._cumulative_value[symbol] = 0.0
            self._quote_expansion_streaks[symbol] = 0
            self._halted_state[symbol] = False
            self._ib.reqMktData(contract)
            self._ib.reqTickByTickData(contract, "Last")
            self._ib.reqTickByTickData(contract, "BidAsk")

    def _handle_pending_tickers(self, tickers) -> None:
        now = datetime.now(timezone.utc)
        for ticker in tickers:
            contract = getattr(ticker, "contract", None)
            symbol = getattr(contract, "symbol", "").upper()
            if symbol not in self._contracts:
                continue

            last = float(getattr(ticker, "last", 0.0) or 0.0)
            bid = float(getattr(ticker, "bid", 0.0) or 0.0)
            ask = float(getattr(ticker, "ask", 0.0) or 0.0)
            volume = float(getattr(ticker, "volume", 0.0) or 0.0)
            trades = getattr(ticker, "tickByTicks", []) or []
            processed = self._processed_ticks.get(symbol, 0)

            bid_past_low = False
            ask_past_high = False
            past_limit = False
            saw_trade = False

            for trade in trades[processed:]:
                if isinstance(trade, TickByTickAllLast):
                    trade_ts = trade.time if isinstance(trade.time, datetime) else now
                    price = float(trade.price)
                    size = float(trade.size)
                    saw_trade = saw_trade or (price > 0 and size > 0)
                    self._flows[symbol].update(trade_ts, price, size, bid, ask)
                    self._cumulative_value[symbol] += price * size
                    past_limit = past_limit or bool(getattr(getattr(trade, "tickAttribLast", None), "pastLimit", False))
                elif isinstance(trade, TickByTickBidAsk):
                    attrs = getattr(trade, "tickAttribBidAsk", None)
                    bid_past_low = bid_past_low or bool(getattr(attrs, "bidPastLow", False))
                    ask_past_high = ask_past_high or bool(getattr(attrs, "askPastHigh", False))

            self._processed_ticks[symbol] = len(trades)
            imbalance = self._flows[symbol].imbalance(now)
            midpoint = ((bid + ask) / 2.0) if bid > 0 and ask > 0 else 0.0
            previous_midpoint = self._last_midpoints.get(symbol, 0.0)
            previous_quote_ts = self._last_quote_ts.get(symbol)
            elapsed = max(0.0, (now - previous_quote_ts).total_seconds()) if previous_quote_ts else 0.0
            quote_gap_pct = (abs(midpoint - previous_midpoint) / previous_midpoint) if midpoint > 0 and previous_midpoint > 0 else 0.0
            midpoint_velocity = (quote_gap_pct / elapsed) if elapsed > 0 else 0.0
            spread_pct = ((ask - bid) / midpoint) if midpoint > 0 and ask > 0 and bid > 0 else 0.0
            previous_spread = self._last_spreads.get(symbol, 0.0)
            expansion_streak = self._quote_expansion_streaks.get(symbol, 0)
            if spread_pct > previous_spread and not saw_trade:
                expansion_streak += 1
            elif saw_trade or spread_pct <= previous_spread:
                expansion_streak = 0

            halted = bool(getattr(ticker, "halted", 0) or getattr(ticker, "delayedHalted", 0))
            resumed_from_halt = self._halted_state.get(symbol, False) and not halted
            self._halted_state[symbol] = halted
            self._last_midpoints[symbol] = midpoint if midpoint > 0 else previous_midpoint
            self._last_spreads[symbol] = spread_pct
            self._last_quote_ts[symbol] = now
            self._quote_expansion_streaks[symbol] = expansion_streak

            quote = QuoteSnapshot(
                ts=now,
                bid=bid,
                ask=ask,
                last=last or ((bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0),
                bid_size=float(getattr(ticker, "bidSize", 0.0) or 0.0),
                ask_size=float(getattr(ticker, "askSize", 0.0) or 0.0),
                cumulative_volume=volume,
                cumulative_value=self._cumulative_value.get(symbol, 0.0),
                vwap=float(getattr(ticker, "vwap", 0.0) or 0.0) or None,
                is_halted=halted,
                past_limit=past_limit,
                bid_past_low=bid_past_low,
                ask_past_high=ask_past_high,
                spread_pct=spread_pct,
                quote_gap_pct=quote_gap_pct,
                midpoint_velocity=midpoint_velocity,
                quote_expansion_streak=expansion_streak,
                resumed_from_halt=resumed_from_halt,
            )
            self._dispatch(self._on_quote(symbol, quote, imbalance))

            price = quote.last if quote.last > 0 else (quote.bid + quote.ask) / 2.0
            if price <= 0:
                continue
            bar = self._builders[symbol].update(now, price, quote.cumulative_volume)
            if bar is not None:
                self._dispatch(self._on_bar(symbol, bar))

    @staticmethod
    def _dispatch(result: Any) -> None:
        if inspect.isawaitable(result):
            asyncio.create_task(result)
