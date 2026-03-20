"""IBKRExecutionAdapter - sole public interface for OMS."""
import logging
from datetime import datetime
from typing import Callable

from ib_async import Trade

from ..client.error_map import classify_error
from ..client.session import IBSession
from ..client.throttler import PacingChannel
from ..logging.audit import log_broker_command, log_broker_response
from ..logging.trace_ids import generate_trace_id
from ..mapping.contract_factory import ContractFactory
from ..mapping.order_mapper import OrderMapper
from ..models.types import (
    BrokerOrderRef,
    ExecutionReport,
    OrderStatusEvent,
    PositionSnapshot,
)
from ..reconciler.snapshots import SnapshotFetcher
from ..risk_support.tick_rules import round_to_tick
from ..state.cache import IBCache

logger = logging.getLogger(__name__)


class OrderNotFoundError(Exception):
    pass


class IBKRExecutionAdapter:
    """Narrow interface between OMS and IBKR.

    OMS should only import this class + typed models.
    """

    def __init__(
        self, session: IBSession, contract_factory: ContractFactory, account: str
    ):
        self._session = session
        self._factory = contract_factory
        self._account = account
        self._cache = IBCache()
        self._snapshots = SnapshotFetcher(session.ib)

        # Callbacks — OMS registers these
        self.on_ack: Callable[[str, BrokerOrderRef], None] = lambda *a: None
        self.on_reject: Callable[[str, str, int, bool], None] = lambda *a: None
        self.on_fill: Callable[
            [str, str, float, float, datetime, float], None
        ] = lambda *a: None
        self.on_status: Callable[[str, str, float], None] = lambda *a: None
        self.on_positions_snapshot: Callable[
            [list[PositionSnapshot]], None
        ] = lambda *a: None

        # Wire IB events
        self._session.ib.orderStatusEvent += self._handle_order_status
        self._session.ib.execDetailsEvent += self._handle_exec_details
        self._session.ib.errorEvent += self._handle_error

    async def submit_order(
        self,
        oms_order_id: str,
        contract_symbol: str,
        contract_expiry: str,
        action: str,
        order_type: str,
        qty: int,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: str = "DAY",
        oca_group: str = "",
        oca_type: int = 0,
        client_order_id: str | None = None,
    ) -> BrokerOrderRef:
        """Submit an order to IBKR. Returns BrokerOrderRef on successful transmission."""
        trace_id = generate_trace_id(oms_order_id)
        await self._session.throttled(PacingChannel.ORDERS)

        contract, spec = await self._factory.resolve(contract_symbol, contract_expiry)

        # Round prices to tick
        if limit_price is not None:
            limit_price = round_to_tick(limit_price, spec.tick_size)
        if stop_price is not None:
            stop_price = round_to_tick(stop_price, spec.tick_size)

        ib_order = OrderMapper.to_ib_order(
            action=action,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            account=self._account,
            oca_group=oca_group,
            oca_type=oca_type,
            order_ref=client_order_id or oms_order_id,
        )

        log_broker_command(
            trace_id,
            oms_order_id,
            "submit",
            {
                "symbol": contract_symbol,
                "action": action,
                "type": order_type,
                "qty": qty,
                "limit": limit_price,
                "stop": stop_price,
            },
        )

        trade = self._session.ib.placeOrder(contract, ib_order)
        ref = BrokerOrderRef(
            broker_order_id=trade.order.orderId,
            perm_id=trade.order.permId,
            con_id=contract.conId,
        )
        self._cache.register_order(oms_order_id, ref.broker_order_id, ref.perm_id)
        return ref

    async def cancel_order(self, broker_order_id: int, perm_id: int = 0) -> None:
        """Cancel an order by broker order ID."""
        await self._session.throttled(PacingChannel.ORDERS)
        oms_id = self._cache.lookup_oms_id(broker_order_id)
        trace_id = generate_trace_id(oms_id or "")
        log_broker_command(
            trace_id,
            oms_id or "",
            "cancel",
            {"broker_order_id": broker_order_id},
        )
        for trade in self._session.ib.openTrades():
            if trade.order.orderId == broker_order_id:
                self._session.ib.cancelOrder(trade.order)
                return
        logger.warning(
            f"Order {broker_order_id} not found in open trades for cancel"
        )

    async def replace_order(
        self,
        broker_order_id: int,
        new_qty: int | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> BrokerOrderRef:
        """Modify an order (IB cancel/replace)."""
        await self._session.throttled(PacingChannel.ORDERS)
        for trade in self._session.ib.openTrades():
            if trade.order.orderId == broker_order_id:
                if new_qty is not None:
                    trade.order.totalQuantity = new_qty
                if new_limit_price is not None:
                    trade.order.lmtPrice = new_limit_price
                if new_stop_price is not None:
                    trade.order.auxPrice = new_stop_price
                self._session.ib.placeOrder(trade.contract, trade.order)
                return BrokerOrderRef(broker_order_id, trade.order.permId, 0)
        raise OrderNotFoundError(f"Order {broker_order_id} not found")

    async def request_open_orders(self) -> list[OrderStatusEvent]:
        # M6: Route snapshot requests through throttler to respect IB pacing
        await self._session.throttled(PacingChannel.ORDERS)
        return await self._snapshots.fetch_open_orders()

    async def request_positions(self) -> list[PositionSnapshot]:
        # M6: Route snapshot requests through throttler to respect IB pacing
        await self._session.throttled(PacingChannel.ORDERS)
        return await self._snapshots.fetch_positions()

    async def request_executions(
        self, since_ts: datetime | None = None
    ) -> list[ExecutionReport]:
        # M6: Route snapshot requests through throttler to respect IB pacing
        await self._session.throttled(PacingChannel.ORDERS)
        return await self._snapshots.fetch_executions(since_ts)

    @property
    def is_ready(self) -> bool:
        return self._session.is_ready

    @property
    def is_congested(self) -> bool:
        return self._session.is_congested

    @property
    def cache(self) -> IBCache:
        return self._cache

    def _handle_order_status(self, trade: Trade) -> None:
        """Route IB orderStatus events to OMS callbacks."""
        oms_id = self._cache.lookup_oms_id(trade.order.orderId)
        if not oms_id:
            return

        status = trade.orderStatus.status
        trace_id = generate_trace_id(oms_id)
        log_broker_response(
            trace_id,
            oms_id,
            "status",
            {
                "status": status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
            },
        )

        if status in ("PreSubmitted", "Submitted"):
            ref = BrokerOrderRef(trade.order.orderId, trade.order.permId, 0)
            self.on_ack(oms_id, ref)
        elif status == "Cancelled":
            self.on_status(oms_id, "CANCELLED", trade.orderStatus.remaining)
        elif status == "Inactive":
            self.on_reject(oms_id, "Order inactive", 0, False)

        self.on_status(oms_id, status, trade.orderStatus.remaining)

    def _handle_exec_details(self, trade: Trade, fill) -> None:
        """Route IB execution/fill events to OMS callbacks."""
        exec_id = fill.execution.execId
        if self._cache.is_fill_seen(exec_id):
            return

        self._cache.mark_fill_seen(exec_id)
        oms_id = self._cache.lookup_oms_id(trade.order.orderId)
        if not oms_id:
            return

        trace_id = generate_trace_id(oms_id)
        commission = (
            fill.commissionReport.commission if fill.commissionReport else 0.0
        )
        log_broker_response(
            trace_id,
            oms_id,
            "fill",
            {
                "exec_id": exec_id,
                "price": fill.execution.price,
                "qty": fill.execution.shares,
                "commission": commission,
            },
        )

        self.on_fill(
            oms_id,
            exec_id,
            fill.execution.price,
            fill.execution.shares,
            fill.execution.time,
            commission,
        )

    def _handle_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Route IB errors to OMS callbacks."""
        oms_id = self._cache.lookup_oms_id(reqId)
        if oms_id:
            category, retryable = classify_error(errorCode, errorString)
            self.on_reject(oms_id, errorString, errorCode, retryable)
