"""OMS factory for proper initialization with all dependencies."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

import asyncpg

from ..config.risk_config import RiskConfig, StrategyRiskConfig
from ..coordination.coordinator import StrategyCoordinator
from ..engine.fill_processor import FillProcessor
from ..events.bus import EventBus
from ..execution.router import ExecutionRouter
from ..intent.handler import IntentHandler
from ..models.order import OrderRole, OrderSide
from ..models.position import Position
from ..models.risk_state import StrategyRiskState, PortfolioRiskState
from ..persistence.in_memory import InMemoryRepository
from ..persistence.repository import OMSRepository
from ..reconciliation.orchestrator import ReconciliationOrchestrator
from ..risk.calendar import EventCalendar
from ..risk.gateway import RiskGateway
from .oms_service import OMSService

if TYPE_CHECKING:
    from shared.market_calendar import MarketCalendar

logger = logging.getLogger(__name__)


_ET = ZoneInfo("America/New_York")


def _trade_date_for(timestamp: datetime | None = None):
    """Return the current US trading date in America/New_York."""
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET).date()


async def build_oms_service(
    adapter,  # IBKRExecutionAdapter
    strategy_id: str,
    unit_risk_dollars: float,
    daily_stop_R: float = 2.0,
    heat_cap_R: float = 1.25,
    portfolio_daily_stop_R: float = 3.0,
    calendar: Optional[EventCalendar] = None,
    recon_interval_s: float = 120.0,
    db_pool: Optional[asyncpg.Pool] = None,
    market_calendar: Optional["MarketCalendar"] = None,
) -> OMSService:
    """Build a fully wired OMS service.

    Args:
        adapter: IBKRExecutionAdapter instance
        strategy_id: Strategy identifier
        unit_risk_dollars: Dollar risk per 1R unit for position sizing
        daily_stop_R: Strategy daily stop in R units
        heat_cap_R: Portfolio heat cap in R units
        portfolio_daily_stop_R: Portfolio daily stop in R units
        calendar: Optional event calendar for blackouts
        recon_interval_s: Reconciliation interval in seconds
        db_pool: Optional asyncpg pool for PostgreSQL persistence.
                 If provided, uses OMSRepository; otherwise InMemoryRepository.

    Returns:
        Fully initialized OMSService ready for start()
    """
    # Event bus
    bus = EventBus()

    # Repository: use PostgreSQL if pool provided, otherwise in-memory
    if db_pool is not None:
        repo = OMSRepository(db_pool)
        logger.info(f"Using PostgreSQL repository for strategy {strategy_id}")
    else:
        repo = InMemoryRepository()
        logger.info(f"Using in-memory repository for strategy {strategy_id}")

    # Risk configuration
    strat_cfg = StrategyRiskConfig(
        strategy_id=strategy_id,
        daily_stop_R=daily_stop_R,
        unit_risk_dollars=unit_risk_dollars,
    )
    risk_config = RiskConfig(
        heat_cap_R=heat_cap_R,
        portfolio_daily_stop_R=portfolio_daily_stop_R,
        strategy_configs={strategy_id: strat_cfg},
    )

    # Event calendar (empty if not provided)
    if calendar is None:
        calendar = EventCalendar()

    # Risk state providers
    strategy_risk_states: dict[str, StrategyRiskState] = {}
    portfolio_risk_state = PortfolioRiskState(trade_date=_trade_date_for())
    # C5: Track open positions by (strategy_id, symbol) to prevent collision
    # when a single strategy trades multiple symbols concurrently
    open_positions: dict[tuple[str, str], dict] = {}

    async def get_strategy_risk(sid: str) -> StrategyRiskState:
        # L1 fix: reset risk state at date boundary
        today = _trade_date_for()
        if sid in strategy_risk_states:
            existing = strategy_risk_states[sid]
            if existing.trade_date != today:
                logger.info(f"Date boundary detected for {sid}: resetting daily risk state")
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=today,
                    open_risk_dollars=existing.open_risk_dollars,
                    open_risk_R=existing.open_risk_R,
                )
        if sid not in strategy_risk_states:
            strategy_risk_states[sid] = StrategyRiskState(strategy_id=sid, trade_date=today)
        return strategy_risk_states[sid]

    async def get_portfolio_risk() -> PortfolioRiskState:
        # L1 fix: reset portfolio risk state at date boundary
        today = _trade_date_for()
        if portfolio_risk_state.trade_date != today:
            logger.info("Date boundary detected: resetting portfolio daily risk state")
            portfolio_risk_state.trade_date = today
            portfolio_risk_state.daily_realized_pnl = 0.0
            portfolio_risk_state.daily_realized_R = 0.0
            portfolio_risk_state.halted = False
            portfolio_risk_state.halt_reason = ""
        portfolio_risk_state.pending_entry_risk_R = await repo.get_pending_entry_risk_R(
            unit_risk_dollars
        )
        return portfolio_risk_state

    async def get_working_order_count(sid: str) -> int:
        return await repo.count_working_orders(sid)

    # Fill processor for OMS order state updates
    fill_proc = FillProcessor(repo)

    # Risk gateway
    risk_gateway = RiskGateway(
        config=risk_config,
        calendar=calendar,
        get_strategy_risk=get_strategy_risk,
        get_portfolio_risk=get_portfolio_risk,
        get_working_order_count=get_working_order_count,
        market_calendar=market_calendar,
    )

    # Execution router
    router = ExecutionRouter(adapter, repo, bus=bus)

    # Intent handler
    handler = IntentHandler(risk_gateway, router, repo, bus)

    # Reconciler
    reconciler = ReconciliationOrchestrator(adapter, repo, bus)

    # C4 fix: Order timeout monitor for stuck ROUTED / CANCEL_REQUESTED states
    from ..engine.timeout_monitor import OrderTimeoutMonitor
    timeout_monitor = OrderTimeoutMonitor(repo, bus)

    # Wire adapter callbacks to bus (with risk state updates)
    _wire_adapter_callbacks(
        adapter, bus, repo, fill_proc, router,
        strategy_risk_states, portfolio_risk_state, unit_risk_dollars, open_positions,
    )

    # Build OMS service
    oms = OMSService(
        intent_handler=handler,
        bus=bus,
        reconciler=reconciler,
        router=router,
        recon_interval_s=recon_interval_s,
        timeout_monitor=timeout_monitor,
    )

    logger.info(f"OMS factory built for strategy {strategy_id}")
    return oms


async def build_multi_strategy_oms(
    adapter,
    strategies: list[dict],
    heat_cap_R: float = 1.5,
    portfolio_daily_stop_R: float = 3.0,
    calendar: Optional[EventCalendar] = None,
    recon_interval_s: float = 120.0,
    db_pool: Optional[asyncpg.Pool] = None,
    market_calendar: Optional["MarketCalendar"] = None,
) -> tuple["OMSService", "StrategyCoordinator"]:
    """Build a shared OMS service for multiple strategies.

    Args:
        adapter: IBKRExecutionAdapter instance (shared)
        strategies: List of strategy config dicts, each with:
            - id: Strategy identifier (e.g. "ATRSS")
            - unit_risk_dollars: Dollar risk per 1R
            - daily_stop_R: Strategy daily stop in R units
            - priority: int (lower = higher priority)
            - max_working_orders: int (optional, default 4)
        heat_cap_R: Portfolio heat cap in R units (shared)
        portfolio_daily_stop_R: Portfolio daily stop in R units
        calendar: Optional event calendar for blackouts
        recon_interval_s: Reconciliation interval in seconds
        db_pool: Optional asyncpg pool for PostgreSQL persistence

    Returns:
        Tuple of (OMSService, StrategyCoordinator) ready for start()
    """
    # Event bus (shared)
    bus = EventBus()

    # Repository (shared)
    if db_pool is not None:
        repo = OMSRepository(db_pool)
        logger.info("Using PostgreSQL repository for multi-strategy OMS")
    else:
        repo = InMemoryRepository()
        logger.info("Using in-memory repository for multi-strategy OMS")

    # Build per-strategy risk configs
    strategy_configs: dict[str, StrategyRiskConfig] = {}
    unit_risk_map: dict[str, float] = {}  # strategy_id → unit_risk_dollars
    for s in strategies:
        sid = s["id"]
        urd = s["unit_risk_dollars"]
        cfg = StrategyRiskConfig(
            strategy_id=sid,
            daily_stop_R=s.get("daily_stop_R", 2.0),
            unit_risk_dollars=urd,
            priority=s.get("priority", 99),
            max_heat_R=s.get("max_heat_R", 0.0),
            max_working_orders=s.get("max_working_orders", 4),
        )
        strategy_configs[sid] = cfg
        unit_risk_map[sid] = urd

    risk_config = RiskConfig(
        heat_cap_R=heat_cap_R,
        portfolio_daily_stop_R=portfolio_daily_stop_R,
        strategy_configs=strategy_configs,
    )

    if calendar is None:
        calendar = EventCalendar()

    # Risk state providers (shared across strategies)
    strategy_risk_states: dict[str, StrategyRiskState] = {}
    portfolio_risk_state = PortfolioRiskState(trade_date=_trade_date_for())
    # Track positions by (strategy_id, symbol) for correct multi-strategy P&L
    open_positions: dict[tuple[str, str], dict] = {}

    # Use highest-priority strategy's URD as portfolio normalization base.
    # This ensures 1R means the same dollar amount regardless of which
    # strategy is requesting entry — matching backtest behavior.
    _sorted_cfgs = sorted(strategy_configs.values(), key=lambda c: c.priority)
    portfolio_urd = _sorted_cfgs[0].unit_risk_dollars if _sorted_cfgs else 1.0

    async def get_strategy_risk(sid: str) -> StrategyRiskState:
        today = _trade_date_for()
        if sid in strategy_risk_states:
            existing = strategy_risk_states[sid]
            if existing.trade_date != today:
                logger.info(f"Date boundary detected for {sid}: resetting daily risk state")
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=today,
                    open_risk_dollars=existing.open_risk_dollars,
                    open_risk_R=existing.open_risk_R,
                )
        if sid not in strategy_risk_states:
            strategy_risk_states[sid] = StrategyRiskState(strategy_id=sid, trade_date=today)
        return strategy_risk_states[sid]

    async def get_portfolio_risk() -> PortfolioRiskState:
        today = _trade_date_for()
        if portfolio_risk_state.trade_date != today:
            logger.info("Date boundary detected: resetting portfolio daily risk state")
            portfolio_risk_state.trade_date = today
            portfolio_risk_state.daily_realized_pnl = 0.0
            portfolio_risk_state.daily_realized_R = 0.0
            portfolio_risk_state.halted = False
            portfolio_risk_state.halt_reason = ""
        portfolio_risk_state.pending_entry_risk_R = await repo.get_pending_entry_risk_R(
            portfolio_urd
        )
        return portfolio_risk_state

    async def get_working_order_count(sid: str) -> int:
        return await repo.count_working_orders(sid)

    # Fill processor (shared)
    fill_proc = FillProcessor(repo)

    # Risk gateway (shared, now with multiple strategy configs + priorities)
    risk_gateway = RiskGateway(
        config=risk_config,
        calendar=calendar,
        get_strategy_risk=get_strategy_risk,
        get_portfolio_risk=get_portfolio_risk,
        get_working_order_count=get_working_order_count,
        market_calendar=market_calendar,
    )

    # Execution router (shared)
    router = ExecutionRouter(adapter, repo, bus=bus)

    # Intent handler (shared)
    handler = IntentHandler(risk_gateway, router, repo, bus)

    # Reconciler (shared)
    reconciler = ReconciliationOrchestrator(adapter, repo, bus)

    # Timeout monitor
    from ..engine.timeout_monitor import OrderTimeoutMonitor
    timeout_monitor = OrderTimeoutMonitor(repo, bus)

    # Coordinator (cross-strategy signals)
    coordinator = StrategyCoordinator(bus, repo)

    # Wire adapter callbacks (multi-strategy aware)
    _wire_adapter_callbacks_multi(
        adapter, bus, repo, fill_proc, router,
        strategy_risk_states, portfolio_risk_state,
        unit_risk_map, open_positions, coordinator,
    )

    # Build OMS service
    oms = OMSService(
        intent_handler=handler,
        bus=bus,
        reconciler=reconciler,
        router=router,
        recon_interval_s=recon_interval_s,
        timeout_monitor=timeout_monitor,
    )

    strategy_ids = [s["id"] for s in strategies]
    logger.info(f"Multi-strategy OMS factory built for: {strategy_ids}")
    return oms, coordinator


def _task_exception_handler(task: "asyncio.Task", bus: EventBus = None) -> None:
    """Log exceptions from fire-and-forget callback tasks and escalate.

    H4: On callback exception, emit RISK_HALT to pause strategy engines
    so that inconsistent state is not silently ignored.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            f"CRITICAL: Unhandled exception in OMS callback task: {exc}",
            exc_info=exc,
        )
        # H4: Escalate to RISK_HALT so strategies pause
        if bus is not None:
            try:
                bus.emit_risk_halt(
                    strategy_id="",
                    reason=f"OMS callback exception: {exc}",
                )
            except Exception:
                logger.error("Failed to emit RISK_HALT after callback exception")


def _wire_adapter_callbacks(
    adapter, bus: EventBus, repo, fill_proc: FillProcessor, router,
    strategy_risk_states, portfolio_risk_state, unit_risk_dollars, open_positions,
) -> None:
    """Wire IBKRExecutionAdapter callbacks to OMS event bus.

    Callbacks look up the order from the repository to get the strategy_id,
    then emit appropriate events to the bus for strategy routing.
    Also wires FillProcessor for OMS order state and updates risk state.
    """
    import asyncio
    from ..models.events import OMSEvent, OMSEventType
    from ..models.order import OrderStatus
    from ..engine.state_machine import transition
    from datetime import datetime, timezone

    # C5 fix: store task references to prevent GC and enable exception handling
    _background_tasks: set[asyncio.Task] = set()

    def _create_tracked_task(coro) -> asyncio.Task:
        """Create an asyncio task with exception handler and prevent GC."""
        task = asyncio.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        # H4: pass bus so exception handler can emit RISK_HALT
        task.add_done_callback(lambda t: _task_exception_handler(t, bus=bus))
        return task

    def _get_running_loop():
        """Get the running event loop, or None if not in async context."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    # H7 fix: retry config for pacing errors
    MAX_PACING_RETRIES = 3
    PACING_RETRY_BASE_DELAY_S = 2.0

    def on_ack(oms_order_id: str, broker_ref) -> None:
        """Handle order acknowledgment from broker."""
        loop = _get_running_loop()
        if loop is None:
            logger.debug(f"Adapter ack (no loop): {oms_order_id}")
            return

        async def _emit_ack():
            order = await repo.get_order(oms_order_id)
            if order:
                # C6 fix: use state machine transition instead of direct assignment
                if transition(order, OrderStatus.ACKED):
                    order.broker_order_ref = broker_ref
                    order.acked_at = datetime.now(timezone.utc)
                    order.last_update_at = order.acked_at
                    await repo.save_order(order)
                    bus.emit_order_event(order)
                    logger.debug(f"Adapter ack emitted: {oms_order_id} for {order.strategy_id}")
                else:
                    logger.warning(
                        f"Adapter ack: invalid transition for {oms_order_id} "
                        f"(current status={order.status.value})"
                    )
            else:
                logger.warning(f"Adapter ack for unknown order: {oms_order_id}")

        _create_tracked_task(_emit_ack())

    def on_reject(oms_order_id: str, reason: str, error_code: int, retryable: bool) -> None:
        """Handle order rejection from broker."""
        loop = _get_running_loop()
        if loop is None:
            logger.warning(f"Adapter reject (no loop): {oms_order_id} - {reason}")
            return

        async def _emit_reject():
            order = await repo.get_order(oms_order_id)
            if not order:
                logger.warning(f"Adapter reject for unknown order: {oms_order_id} - {reason}")
                return

            # H5/H7 fix: retry for retryable pacing errors using persisted retry_count
            if retryable and error_code > 0:
                if order.retry_count < MAX_PACING_RETRIES:
                    order.retry_count += 1
                    delay = PACING_RETRY_BASE_DELAY_S * (2 ** (order.retry_count - 1))
                    logger.warning(
                        f"Retryable reject for {oms_order_id} (code={error_code}): "
                        f"retry {order.retry_count}/{MAX_PACING_RETRIES} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    # Re-route through the execution router
                    try:
                        # Reset status to RISK_APPROVED for re-routing
                        order.status = OrderStatus.RISK_APPROVED
                        order.last_update_at = datetime.now(timezone.utc)
                        await repo.save_order(order)
                        await router.route(order)
                        return
                    except Exception as e:
                        logger.error(f"Retry failed for {oms_order_id}: {e}")

            # Terminal rejection
            if transition(order, OrderStatus.REJECTED):
                order.reject_reason = reason
                order.last_update_at = datetime.now(timezone.utc)
                await repo.save_order(order)
                bus.emit_order_event(order)
                logger.warning(f"Adapter reject emitted: {oms_order_id} for {order.strategy_id} - {reason}")
            else:
                logger.warning(
                    f"Adapter reject: invalid transition for {oms_order_id} "
                    f"(current status={order.status.value})"
                )

        _create_tracked_task(_emit_reject())

    def on_fill(
        oms_order_id: str,
        exec_id: str,
        price: float,
        qty: float,
        timestamp,
        commission: float,
    ) -> None:
        """Handle fill from broker - update OMS order state, risk state, emit event."""
        loop = _get_running_loop()
        if loop is None:
            logger.info(f"Adapter fill (no loop): {oms_order_id} {qty}@{price}")
            return

        async def _emit_fill():
            order = await repo.get_order(oms_order_id)
            if not order:
                logger.warning(f"Adapter fill for unknown order: {oms_order_id}")
                return

            # 1. Update OMS order state (filled_qty, status, avg_fill_price)
            fill_ts = timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc)
            await fill_proc.process_fill(oms_order_id, exec_id, price, qty, fill_ts, commission)

            # 2. Emit fill event to strategy
            fill_data = {
                "exec_id": exec_id,
                "price": price,
                "qty": qty,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "commission": commission,
                "client_order_id": getattr(order, 'client_order_id', ""),
            }
            bus.emit_fill_event(order.strategy_id, oms_order_id, fill_data)

            # 3. Update risk state
            sid = order.strategy_id
            # C5: key by (strategy_id, symbol) to prevent multi-symbol collision
            instr_sym = order.instrument.symbol if order.instrument else ""
            pos_key = (sid, instr_sym)

            if sid not in strategy_risk_states:
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=_trade_date_for(fill_ts)
                )
            strat_risk = strategy_risk_states[sid]

            if order.role == OrderRole.ENTRY and order.risk_context:
                risk_per_contract = (
                    order.risk_context.risk_dollars / order.qty if order.qty > 0 else 0
                )
                fill_risk = risk_per_contract * qty
                fill_risk_R = fill_risk / unit_risk_dollars if unit_risk_dollars > 0 else 0

                strat_risk.open_risk_dollars += fill_risk
                strat_risk.open_risk_R += fill_risk_R
                portfolio_risk_state.open_risk_dollars += fill_risk
                portfolio_risk_state.open_risk_R += fill_risk_R

                # C5: Track entry for exit P&L computation keyed by (sid, symbol)
                pos = open_positions.get(pos_key)
                pv = order.instrument.point_value if order.instrument else 1.0
                if pos is None:
                    open_positions[pos_key] = {
                        "entry_price": price,
                        "risk_per_contract_R": fill_risk_R / qty if qty > 0 else 0,
                        "point_value": pv,
                        "side": order.side,
                        "open_qty": qty,
                    }
                else:
                    old_total = pos["entry_price"] * pos["open_qty"]
                    pos["open_qty"] += qty
                    pos["entry_price"] = (old_total + price * qty) / pos["open_qty"]

            elif order.role in (OrderRole.EXIT, OrderRole.STOP, OrderRole.TP):
                pos = open_positions.get(pos_key)
                if pos:
                    # Reduce open risk
                    released_R = pos["risk_per_contract_R"] * qty
                    released_dollars = released_R * unit_risk_dollars

                    strat_risk.open_risk_R = max(0, strat_risk.open_risk_R - released_R)
                    strat_risk.open_risk_dollars = max(0, strat_risk.open_risk_dollars - released_dollars)
                    portfolio_risk_state.open_risk_R = max(0, portfolio_risk_state.open_risk_R - released_R)
                    portfolio_risk_state.open_risk_dollars = max(0, portfolio_risk_state.open_risk_dollars - released_dollars)

                    # Compute realized P&L
                    pv = pos["point_value"]
                    if pos["side"] == OrderSide.BUY:
                        pnl = (price - pos["entry_price"]) * pv * qty
                    else:
                        pnl = (pos["entry_price"] - price) * pv * qty

                    pnl_R = pnl / unit_risk_dollars if unit_risk_dollars > 0 else 0
                    strat_risk.daily_realized_pnl += pnl
                    strat_risk.daily_realized_R += pnl_R
                    portfolio_risk_state.daily_realized_pnl += pnl
                    portfolio_risk_state.daily_realized_R += pnl_R

                    pos["open_qty"] = max(0, pos["open_qty"] - qty)
                    if pos["open_qty"] <= 0:
                        del open_positions[pos_key]

            # 4. Update OMS position for FLATTEN handler and reconciliation
            pos_data = open_positions.get(pos_key)
            if pos_data:
                net = pos_data["open_qty"] if pos_data["side"] == OrderSide.BUY else -pos_data["open_qty"]
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=net,
                    avg_price=pos_data["entry_price"],
                    realized_pnl=strat_risk.daily_realized_pnl,
                    open_risk_dollars=strat_risk.open_risk_dollars,
                    open_risk_R=strat_risk.open_risk_R,
                    last_update_at=fill_ts,
                )
            else:
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=0,
                    realized_pnl=strat_risk.daily_realized_pnl,
                    last_update_at=fill_ts,
                )
            await repo.save_position(oms_pos)

            logger.info(f"Adapter fill processed: {oms_order_id} {qty}@{price} for {sid}/{instr_sym}")

        _create_tracked_task(_emit_fill())

    def on_status(oms_order_id: str, status: str, remaining: float) -> None:
        """Handle status update from broker."""
        loop = _get_running_loop()
        if loop is None:
            logger.debug(f"Adapter status (no loop): {oms_order_id} {status}")
            return

        async def _emit_status():
            order = await repo.get_order(oms_order_id)
            if order:
                # Map broker status string to OrderStatus
                status_map = {
                    "Submitted": OrderStatus.WORKING,
                    "PreSubmitted": OrderStatus.ROUTED,
                    "Filled": OrderStatus.FILLED,
                    "Cancelled": OrderStatus.CANCELLED,
                    "ApiCancelled": OrderStatus.CANCELLED,
                    "PendingCancel": OrderStatus.CANCEL_REQUESTED,
                }
                new_status = status_map.get(status)
                if new_status and new_status != order.status:
                    # C6 fix: use state machine transition instead of direct assignment
                    if transition(order, new_status):
                        order.remaining_qty = int(remaining)
                        order.last_update_at = datetime.now(timezone.utc)
                        await repo.save_order(order)
                        bus.emit_order_event(order)
                        logger.debug(f"Adapter status emitted: {oms_order_id} {status} for {order.strategy_id}")
                    else:
                        logger.warning(
                            f"Adapter status: invalid transition for {oms_order_id} "
                            f"{order.status.value} -> {new_status.value}"
                        )
            else:
                logger.debug(f"Adapter status for unknown order: {oms_order_id} {status}")

        _create_tracked_task(_emit_status())

    adapter.on_ack = on_ack
    adapter.on_reject = on_reject
    adapter.on_fill = on_fill
    adapter.on_status = on_status


def _wire_adapter_callbacks_multi(
    adapter, bus: EventBus, repo, fill_proc: FillProcessor, router,
    strategy_risk_states, portfolio_risk_state,
    unit_risk_map: dict[str, float],
    open_positions: dict[tuple[str, str], dict],
    coordinator: "StrategyCoordinator",
) -> None:
    """Wire IBKRExecutionAdapter callbacks for multi-strategy OMS.

    Key differences from single-strategy:
    - unit_risk_dollars looked up per order's strategy_id
    - open_positions keyed by (strategy_id, symbol) for concurrent positions
    - coordinator notified on fills for cross-strategy signals
    """
    import asyncio
    from ..models.events import OMSEvent, OMSEventType
    from ..models.order import OrderStatus
    from ..engine.state_machine import transition
    from datetime import datetime, timezone

    _background_tasks: set[asyncio.Task] = set()

    def _create_tracked_task(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        # H4: pass bus so exception handler can emit RISK_HALT
        task.add_done_callback(lambda t: _task_exception_handler(t, bus=bus))
        return task

    def _get_running_loop():
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    MAX_PACING_RETRIES = 3
    PACING_RETRY_BASE_DELAY_S = 2.0

    def on_ack(oms_order_id: str, broker_ref) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_ack():
            order = await repo.get_order(oms_order_id)
            if order:
                if transition(order, OrderStatus.ACKED):
                    order.broker_order_ref = broker_ref
                    order.acked_at = datetime.now(timezone.utc)
                    order.last_update_at = order.acked_at
                    await repo.save_order(order)
                    bus.emit_order_event(order)

        _create_tracked_task(_emit_ack())

    def on_reject(oms_order_id: str, reason: str, error_code: int, retryable: bool) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_reject():
            order = await repo.get_order(oms_order_id)
            if not order:
                return

            # H5: use persisted retry_count instead of transient dynamic attr
            if retryable and error_code > 0:
                if order.retry_count < MAX_PACING_RETRIES:
                    order.retry_count += 1
                    delay = PACING_RETRY_BASE_DELAY_S * (2 ** (order.retry_count - 1))
                    logger.warning(
                        f"Retryable reject for {oms_order_id} (code={error_code}): "
                        f"retry {order.retry_count}/{MAX_PACING_RETRIES} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    try:
                        order.status = OrderStatus.RISK_APPROVED
                        order.last_update_at = datetime.now(timezone.utc)
                        await repo.save_order(order)
                        await router.route(order)
                        return
                    except Exception as e:
                        logger.error(f"Retry failed for {oms_order_id}: {e}")

            if transition(order, OrderStatus.REJECTED):
                order.reject_reason = reason
                order.last_update_at = datetime.now(timezone.utc)
                await repo.save_order(order)
                bus.emit_order_event(order)

        _create_tracked_task(_emit_reject())

    def on_fill(
        oms_order_id: str,
        exec_id: str,
        price: float,
        qty: float,
        timestamp,
        commission: float,
    ) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_fill():
            order = await repo.get_order(oms_order_id)
            if not order:
                logger.warning(f"Adapter fill for unknown order: {oms_order_id}")
                return

            fill_ts = timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc)
            await fill_proc.process_fill(oms_order_id, exec_id, price, qty, fill_ts, commission)

            fill_data = {
                "exec_id": exec_id,
                "price": price,
                "qty": qty,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "commission": commission,
                "client_order_id": getattr(order, 'client_order_id', ""),
            }
            bus.emit_fill_event(order.strategy_id, oms_order_id, fill_data)

            # Per-strategy unit_risk_dollars lookup
            sid = order.strategy_id
            urd = unit_risk_map.get(sid, 1.0)
            instr_sym = order.instrument.symbol if order.instrument else ""
            pos_key = (sid, instr_sym)

            if sid not in strategy_risk_states:
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=_trade_date_for(fill_ts)
                )
            strat_risk = strategy_risk_states[sid]

            if order.role == OrderRole.ENTRY and order.risk_context:
                risk_per_contract = (
                    order.risk_context.risk_dollars / order.qty if order.qty > 0 else 0
                )
                fill_risk = risk_per_contract * qty
                fill_risk_R = fill_risk / urd if urd > 0 else 0
                # Portfolio R uses consistent base so 1R means the same dollars
                fill_risk_portfolio_R = fill_risk / portfolio_urd if portfolio_urd > 0 else 0

                strat_risk.open_risk_dollars += fill_risk
                strat_risk.open_risk_R += fill_risk_R
                portfolio_risk_state.open_risk_dollars += fill_risk
                portfolio_risk_state.open_risk_R += fill_risk_portfolio_R

                pv = order.instrument.point_value if order.instrument else 1.0
                pos = open_positions.get(pos_key)
                if pos is None:
                    open_positions[pos_key] = {
                        "entry_price": price,
                        "risk_per_contract_R": fill_risk_R / qty if qty > 0 else 0,
                        "risk_per_contract_portfolio_R": fill_risk_portfolio_R / qty if qty > 0 else 0,
                        "point_value": pv,
                        "side": order.side,
                        "open_qty": qty,
                    }
                else:
                    old_total = pos["entry_price"] * pos["open_qty"]
                    pos["open_qty"] += qty
                    pos["entry_price"] = (old_total + price * qty) / pos["open_qty"]

                # Notify coordinator of entry fill
                direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
                coordinator.on_fill(
                    strategy_id=sid, symbol=instr_sym,
                    side=order.side.value, role="ENTRY", price=price,
                )
                coordinator.on_position_update(
                    strategy_id=sid, symbol=instr_sym,
                    qty=int(open_positions[pos_key]["open_qty"]),
                    direction=direction, entry_price=open_positions[pos_key]["entry_price"],
                )

            elif order.role in (OrderRole.EXIT, OrderRole.STOP, OrderRole.TP):
                pos = open_positions.get(pos_key)
                if pos:
                    released_R = pos["risk_per_contract_R"] * qty
                    released_dollars = released_R * urd
                    released_portfolio_R = pos.get("risk_per_contract_portfolio_R", released_R) * qty

                    strat_risk.open_risk_R = max(0, strat_risk.open_risk_R - released_R)
                    strat_risk.open_risk_dollars = max(0, strat_risk.open_risk_dollars - released_dollars)
                    portfolio_risk_state.open_risk_R = max(0, portfolio_risk_state.open_risk_R - released_portfolio_R)
                    portfolio_risk_state.open_risk_dollars = max(0, portfolio_risk_state.open_risk_dollars - released_dollars)

                    pv = pos["point_value"]
                    if pos["side"] == OrderSide.BUY:
                        pnl = (price - pos["entry_price"]) * pv * qty
                    else:
                        pnl = (pos["entry_price"] - price) * pv * qty

                    pnl_R = pnl / urd if urd > 0 else 0
                    pnl_portfolio_R = pnl / portfolio_urd if portfolio_urd > 0 else 0
                    strat_risk.daily_realized_pnl += pnl
                    strat_risk.daily_realized_R += pnl_R
                    portfolio_risk_state.daily_realized_pnl += pnl
                    portfolio_risk_state.daily_realized_R += pnl_portfolio_R

                    pos["open_qty"] = max(0, pos["open_qty"] - qty)
                    remaining = int(pos["open_qty"])
                    if remaining <= 0:
                        del open_positions[pos_key]

                    # Notify coordinator of position change
                    direction = "LONG" if pos["side"] == OrderSide.BUY else "SHORT"
                    coordinator.on_fill(
                        strategy_id=sid, symbol=instr_sym,
                        side=order.side.value, role=order.role.value, price=price,
                    )
                    coordinator.on_position_update(
                        strategy_id=sid, symbol=instr_sym,
                        qty=remaining, direction=direction,
                    )

            # Save OMS position
            pos_data = open_positions.get(pos_key)
            if pos_data:
                net = pos_data["open_qty"] if pos_data["side"] == OrderSide.BUY else -pos_data["open_qty"]
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=net,
                    avg_price=pos_data["entry_price"],
                    realized_pnl=strat_risk.daily_realized_pnl,
                    open_risk_dollars=strat_risk.open_risk_dollars,
                    open_risk_R=strat_risk.open_risk_R,
                    last_update_at=fill_ts,
                )
            else:
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=0,
                    realized_pnl=strat_risk.daily_realized_pnl,
                    last_update_at=fill_ts,
                )
            await repo.save_position(oms_pos)

            logger.info(f"Multi-OMS fill: {oms_order_id} {qty}@{price} for {sid}/{instr_sym}")

        _create_tracked_task(_emit_fill())

    def on_status(oms_order_id: str, status: str, remaining: float) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_status():
            order = await repo.get_order(oms_order_id)
            if order:
                status_map = {
                    "Submitted": OrderStatus.WORKING,
                    "PreSubmitted": OrderStatus.ROUTED,
                    "Filled": OrderStatus.FILLED,
                    "Cancelled": OrderStatus.CANCELLED,
                    "ApiCancelled": OrderStatus.CANCELLED,
                    "PendingCancel": OrderStatus.CANCEL_REQUESTED,
                }
                new_status = status_map.get(status)
                if new_status and new_status != order.status:
                    if transition(order, new_status):
                        order.remaining_qty = int(remaining)
                        order.last_update_at = datetime.now(timezone.utc)
                        await repo.save_order(order)
                        bus.emit_order_event(order)

        _create_tracked_task(_emit_status())

    adapter.on_ack = on_ack
    adapter.on_reject = on_reject
    adapter.on_fill = on_fill
    adapter.on_status = on_status
