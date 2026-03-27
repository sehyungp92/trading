"""
Reference extraction from strategies/stock/alcb/engine.py
Methods needed for T2 engine port (task C4).
"""

# =============================================================================
# 1. _reconcile_after_reconnect  (lines 163-171)
# =============================================================================

async def _reconcile_after_reconnect(self) -> None:
    if self._oms is None:
        return
    logger.warning("IB reconnected; requesting OMS reconciliation")
    try:
        await self._oms.request_reconciliation()
        logger.info("OMS reconciliation complete")
    except Exception as exc:
        logger.error("OMS reconciliation failed: %s", exc, exc_info=exc)


# =============================================================================
# 2. _save_state  (lines 2533-2536)
# =============================================================================

async def _save_state(self, reason: str) -> None:
    persist_intraday_state(self.snapshot_state(), settings=self._settings)
    self._last_save_ts = datetime.now(timezone.utc)
    self._diagnostics.log_decision("STATE_SAVE", {"reason": reason})


# =============================================================================
# 3. hydrate_state  (lines 192-228)
# =============================================================================

def hydrate_state(self, snapshot: IntradayStateSnapshot) -> None:
    for market in snapshot.markets:
        self._markets[market.symbol] = market

    self._portfolio.open_positions.clear()
    self._portfolio.pending_entry_risk.clear()
    self._order_index.clear()
    self._pending_plans.clear()
    self._order_submission_times.clear()
    for stored in snapshot.symbols:
        current = self._symbols.get(stored.symbol)
        if current is None:
            self._symbols[stored.symbol] = stored
        else:
            current.campaign = (
                current.campaign
                if self._campaign_rebuilt(current.campaign, stored.campaign)
                else stored.campaign
            )
            current.intraday_score = stored.intraday_score
            current.intraday_detail = dict(stored.intraday_detail)
            current.mode = stored.mode
            current.last_transition_reason = stored.last_transition_reason
            current.last_30m_bar_time = stored.last_30m_bar_time
            current.entry_order = stored.entry_order
            current.stop_order = stored.stop_order
            current.exit_order = stored.exit_order
            current.tp1_order = stored.tp1_order
            current.tp2_order = stored.tp2_order
            current.position = stored.position
            current.pending_hard_exit = stored.pending_hard_exit
            current.pending_add = stored.pending_add
            current.last_signal_factors = dict(stored.last_signal_factors)
            stored = current
        if stored.position is not None:
            self._portfolio.open_positions[stored.symbol] = stored.position
        self._restore_order_state(stored.symbol, stored)


# =============================================================================
# 4. subscription_instruments  (lines 265-274)
# =============================================================================

def subscription_instruments(self) -> list:
    instruments = build_proxy_instruments()
    seen = {instrument.symbol for instrument in instruments}
    for symbol in sorted(self._hot_symbols | set(self._portfolio.open_positions)):
        item = self._items.get(symbol)
        if item is None or symbol in seen:
            continue
        instruments.append(build_stock_instrument(item))
        seen.add(symbol)
    return instruments


# =============================================================================
# 5. polling_instruments  (lines 276-282)
# =============================================================================

def polling_instruments(self) -> list[tuple[Any, int]]:
    requests: list[tuple[Any, int]] = []
    for item in self._artifact.overflow:
        if item.symbol in self._hot_symbols:
            continue
        requests.append((build_stock_instrument(item), self._settings.cold_poll_interval_s))
    return requests


# =============================================================================
# 6. _expected_stop_cancels usage patterns
# =============================================================================

# --- Initialization (line 121) ---
# self._expected_stop_cancels: set[str] = set()

# --- Pattern A: _drop_stop_order_state (lines 513-522) ---
def _drop_stop_order_state(self, symbol: str) -> None:
    state = self._symbols[symbol]
    stop_order_id = self._current_stop_order_id(state)
    if stop_order_id:
        self._clear_order_tracking(stop_order_id)
        self._expected_stop_cancels.add(stop_order_id)
    state.stop_order = None
    if state.position is not None:
        state.position.stop_order_id = ""
        state.position.stop_submitted_at = None

# --- Pattern B: _cancel_stop (lines 1732-1741) ---
async def _cancel_stop(self, symbol: str) -> None:
    state = self._symbols[symbol]
    stop_order_id = self._current_stop_order_id(state)
    if not stop_order_id:
        return
    self._expected_stop_cancels.add(stop_order_id)
    if state.stop_order is not None:
        state.stop_order.cancel_requested = True
    await self._cancel_order(stop_order_id)

# --- Pattern C: Cancel handler (lines 2512-2526) ---
# Inside _handle_cancel event handler:
#   if role == "STOP" and state.position is not None:
#       current_stop_order_id = self._current_stop_order_id(state)
#       if current_stop_order_id and event.oms_order_id and current_stop_order_id != event.oms_order_id:
#           self._expected_stop_cancels.discard(event.oms_order_id)
#           return
#       if event.oms_order_id in self._expected_stop_cancels:
#           self._expected_stop_cancels.discard(event.oms_order_id)
#           state.stop_order = None
#           state.position.stop_order_id = ""
#           state.position.stop_submitted_at = None
#           return
#       # Unexpected cancel -> emergency market exit
#       state.stop_order = None
#       state.position.stop_order_id = ""
#       state.position.stop_submitted_at = None
#       await self._submit_market_exit(symbol, state.position.qty_open, OrderRole.EXIT)


# =============================================================================
# 7. stop()  (lines 181-190)
# =============================================================================

async def stop(self) -> None:
    self._running = False
    await self._cancel_open_entries("shutdown")
    await self._save_state("stop")
    for task in (self._pulse_task, self._event_task):
        if task is None:
            continue
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


# =============================================================================
# 8. _restore_order_state  (lines 2538-2564)
# =============================================================================

def _restore_order_state(self, symbol: str, state: SymbolRuntimeState) -> None:
    if state.entry_order is not None:
        self._order_index[state.entry_order.oms_order_id] = (symbol, "ENTRY")
        self._track_order_submission(state.entry_order.oms_order_id, state.entry_order.submitted_at)
        if state.entry_order.risk_dollars is not None:
            self._portfolio.pending_entry_risk[symbol] = state.entry_order.risk_dollars
        restored_plan = self._plan_from_order_state(symbol, state.entry_order)
        if restored_plan is not None:
            self._pending_plans[state.entry_order.oms_order_id] = restored_plan
    if state.stop_order is not None:
        self._order_index[state.stop_order.oms_order_id] = (symbol, "STOP")
        self._track_order_submission(state.stop_order.oms_order_id, state.stop_order.submitted_at)
        if state.position is not None:
            state.position.stop_order_id = state.stop_order.oms_order_id
            state.position.stop_submitted_at = state.stop_order.submitted_at
    if state.exit_order is not None:
        self._order_index[state.exit_order.oms_order_id] = (symbol, OrderRole.EXIT.value)
        self._track_order_submission(state.exit_order.oms_order_id, state.exit_order.submitted_at)
    if state.tp1_order is not None:
        self._order_index[state.tp1_order.oms_order_id] = (symbol, "TP1")
        self._track_order_submission(state.tp1_order.oms_order_id, state.tp1_order.submitted_at)
    if state.tp2_order is not None:
        self._order_index[state.tp2_order.oms_order_id] = (symbol, "TP2")
        self._track_order_submission(state.tp2_order.oms_order_id, state.tp2_order.submitted_at)
    # Fallback: position has a stop_order_id but stop_order was lost
    if state.stop_order is None and state.position is not None and state.position.stop_order_id:
        self._order_index[state.position.stop_order_id] = (symbol, "STOP")
        self._track_order_submission(state.position.stop_order_id, state.position.stop_submitted_at)


# =============================================================================
# 9. snapshot_state  (lines 255-263) -- bonus, needed by _save_state
# =============================================================================

def snapshot_state(self) -> IntradayStateSnapshot:
    return IntradayStateSnapshot(
        trade_date=self._artifact.trade_date,
        saved_at=datetime.now(timezone.utc),
        symbols=list(self._symbols.values()),
        markets=list(self._markets.values()),
        last_decision_code="snapshot",
        meta={"hot_symbols": sorted(self._hot_symbols)},
    )
