"""OMS persistence repository."""
import json
import logging
from datetime import datetime
from typing import Optional, Any

try:
    import asyncpg.exceptions
except ImportError:
    asyncpg = None  # type: ignore

from ..models.fill import Fill
from ..models.instrument import Instrument
from ..models.instrument_registry import InstrumentRegistry
from ..models.order import (
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskContext,
)
from ..models.position import Position

logger = logging.getLogger(__name__)


class OMSRepository:
    """Persistence layer for OMS state. Event-sourcing pattern:
    1. Insert into order_events (append-only)
    2. Update orders (current state)
    """

    def __init__(self, pool):  # asyncpg pool
        self._pool = pool

    async def save_order(self, order: OMSOrder) -> None:
        """Upsert current order state."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orders (
                    oms_order_id, client_order_id, strategy_id, account_id,
                    instrument_symbol, side, qty, order_type, limit_price, stop_price,
                    tif, role, status, broker, broker_order_id, perm_id, oca_group,
                    filled_qty, remaining_qty, avg_fill_price, reprice_count,
                    entry_policy, risk_context,
                    created_at, submitted_at, acked_at, last_update_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22::jsonb,$23::jsonb,$24,$25,$26,$27)
                ON CONFLICT (oms_order_id) DO UPDATE SET
                    status=$13, broker_order_id=$15, perm_id=$16,
                    filled_qty=$18, remaining_qty=$19, avg_fill_price=$20,
                    reprice_count=$21, submitted_at=$25, acked_at=$26, last_update_at=$27
                """,
                order.oms_order_id,
                order.client_order_id,
                order.strategy_id,
                order.account_id,
                order.instrument.symbol if order.instrument else "",
                order.side.value,
                order.qty,
                order.order_type.value,
                order.limit_price,
                order.stop_price,
                order.tif,
                order.role.value,
                order.status.value,
                order.broker,
                order.broker_order_id,
                order.perm_id,
                order.oca_group,
                order.filled_qty,
                order.remaining_qty,
                order.avg_fill_price,
                order.reprice_count,
                json.dumps(order.entry_policy.__dict__) if order.entry_policy else None,
                json.dumps(order.risk_context.__dict__) if order.risk_context else None,
                order.created_at,
                order.submitted_at,
                order.acked_at,
                order.last_update_at,
            )

    async def save_event(self, oms_order_id: str, event_type: str, payload: dict) -> None:
        """Append to order_events (immutable audit log)."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO order_events (oms_order_id, event_type, payload) VALUES ($1, $2, $3::jsonb)",
                    oms_order_id,
                    event_type,
                    json.dumps(payload),
                )
        except Exception as exc:
            if asyncpg and isinstance(exc, asyncpg.exceptions.ForeignKeyViolationError):
                logger.warning(
                    "save_event skipped: order %s not found in orders table (event_type=%s)",
                    oms_order_id, event_type,
                )
            else:
                raise

    async def save_fill(self, fill: Fill) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO fills (fill_id, oms_order_id, broker_fill_id, price, qty, fill_ts, fees)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (broker_fill_id) DO NOTHING
                    """,
                    fill.fill_id,
                    fill.oms_order_id,
                    fill.broker_fill_id,
                    fill.price,
                    fill.qty,
                    fill.timestamp,
                    fill.fees,
                )
        except Exception as exc:
            if asyncpg and isinstance(exc, asyncpg.exceptions.ForeignKeyViolationError):
                logger.warning(
                    "save_fill skipped: order %s not found in orders table (fill_id=%s)",
                    fill.oms_order_id, fill.fill_id,
                )
            else:
                raise

    async def fill_exists(self, broker_fill_id: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM fills WHERE broker_fill_id = $1", broker_fill_id
            )
        return row is not None

    async def get_order(self, oms_order_id: str) -> Optional[OMSOrder]:
        """Load order by ID. Returns None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM orders WHERE oms_order_id = $1", oms_order_id
            )
        if not row:
            return None
        return self._row_to_order(dict(row))

    async def get_order_id_by_client_order_id(
        self, strategy_id: str, client_order_id: str
    ) -> Optional[str]:
        """Look up oms_order_id by client_order_id for idempotency."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT oms_order_id FROM orders
                   WHERE strategy_id = $1 AND client_order_id = $2""",
                strategy_id,
                client_order_id,
            )
        return row["oms_order_id"] if row else None

    async def get_order_id_by_broker_order_id(
        self, broker_order_id: int
    ) -> Optional[str]:
        """Resolve an OMS order ID from a persisted broker order ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT oms_order_id FROM orders
                   WHERE broker_order_id = $1::text""",
                str(broker_order_id),
            )
        return row["oms_order_id"] if row else None

    async def get_pending_entry_risk_R(self, unit_risk_dollars: float) -> float:
        """Sum risk_R of working ENTRY orders. Includes PARTIALLY_FILLED scaled by remaining qty."""
        working_statuses = (
            OrderStatus.RISK_APPROVED.value,
            OrderStatus.ROUTED.value,
            OrderStatus.ACKED.value,
            OrderStatus.WORKING.value,
            OrderStatus.PARTIALLY_FILLED.value,
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT risk_context, qty, remaining_qty, status FROM orders
                   WHERE role = $1 AND status IN ($2, $3, $4, $5, $6)
                   AND risk_context IS NOT NULL""",
                OrderRole.ENTRY.value,
                *working_statuses,
            )
        return self._sum_pending_risk(rows) / unit_risk_dollars if unit_risk_dollars > 0 else 0.0

    async def get_working_orders(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[OMSOrder]:
        """Get all non-terminal orders for a strategy."""
        terminal = (
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.DONE.value,
        )
        async with self._pool.acquire() as conn:
            if instrument_symbol:
                rows = await conn.fetch(
                    """SELECT * FROM orders
                       WHERE strategy_id=$1 AND instrument_symbol=$2
                       AND status NOT IN ($3, $4, $5, $6, $7)""",
                    strategy_id,
                    instrument_symbol,
                    *terminal,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM orders
                       WHERE strategy_id=$1 AND status NOT IN ($2, $3, $4, $5, $6)""",
                    strategy_id,
                    *terminal,
                )
        return [self._row_to_order(dict(r)) for r in rows]

    async def count_working_orders(self, strategy_id: str) -> int:
        """Count non-terminal orders for a strategy."""
        terminal = (
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.DONE.value,
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) as cnt FROM orders
                   WHERE strategy_id=$1 AND status NOT IN ($2,$3,$4,$5,$6)""",
                strategy_id,
                *terminal,
            )
        return row["cnt"] if row else 0

    async def get_positions(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[Position]:
        async with self._pool.acquire() as conn:
            if instrument_symbol:
                rows = await conn.fetch(
                    "SELECT * FROM positions WHERE strategy_id=$1 AND instrument_symbol=$2",
                    strategy_id,
                    instrument_symbol,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM positions WHERE strategy_id=$1", strategy_id
                )
        return [self._row_to_position(dict(r)) for r in rows]

    async def save_position(self, position: Position) -> None:
        """Upsert position."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO positions
                    (account_id, instrument_symbol, strategy_id, net_qty, avg_price,
                     realized_pnl, unrealized_pnl, open_risk_dollars, open_risk_R, last_update_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
                ON CONFLICT (account_id, instrument_symbol, strategy_id) DO UPDATE SET
                    net_qty = EXCLUDED.net_qty,
                    avg_price = EXCLUDED.avg_price,
                    realized_pnl = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    open_risk_dollars = EXCLUDED.open_risk_dollars,
                    open_risk_R = EXCLUDED.open_risk_R,
                    last_update_at = now()
                """,
                position.account_id,
                position.instrument_symbol,
                position.strategy_id,
                position.net_qty,
                position.avg_price,
                position.realized_pnl,
                position.unrealized_pnl,
                position.open_risk_dollars,
                position.open_risk_R,
            )

    async def get_all_working_orders(self) -> list[OMSOrder]:
        """Get all non-terminal orders across all strategies."""
        terminal = (
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.DONE.value,
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM orders
                   WHERE status NOT IN ($1, $2, $3, $4, $5)""",
                *terminal,
            )
        return [self._row_to_order(dict(r)) for r in rows]

    async def get_all_positions(self) -> list[Position]:
        """Get all positions across all strategies."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM positions")
        return [self._row_to_position(dict(r)) for r in rows]

    async def get_positions_for_strategies(
        self, strategy_ids: list[str],
    ) -> list[Position]:
        """Get positions for specific strategies only (family-scoped)."""
        if not strategy_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM positions WHERE strategy_id = ANY($1::text[])",
                strategy_ids,
            )
        return [self._row_to_position(dict(r)) for r in rows]

    async def get_pending_entry_risk_R_for_strategies(
        self, strategy_ids: list[str], unit_risk_dollars: float,
    ) -> float:
        """Sum risk_R of working ENTRY orders for specific strategies (family-scoped)."""
        if not strategy_ids or unit_risk_dollars <= 0:
            return 0.0
        working_statuses = (
            OrderStatus.RISK_APPROVED.value,
            OrderStatus.ROUTED.value,
            OrderStatus.ACKED.value,
            OrderStatus.WORKING.value,
            OrderStatus.PARTIALLY_FILLED.value,
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT risk_context, qty, remaining_qty, status FROM orders
                   WHERE role = $1 AND status IN ($2, $3, $4, $5, $6)
                   AND risk_context IS NOT NULL
                   AND strategy_id = ANY($7::text[])""",
                OrderRole.ENTRY.value,
                *working_statuses,
                strategy_ids,
            )
        return self._sum_pending_risk(rows) / unit_risk_dollars

    @staticmethod
    def _sum_pending_risk(rows) -> float:
        """Sum risk_dollars from pending entry order rows."""
        total = 0.0
        for row in rows:
            rc = row.get("risk_context")
            if not rc:
                continue
            data = json.loads(rc) if isinstance(rc, str) else rc
            risk = data.get("risk_dollars", 0.0)
            if row["status"] == OrderStatus.PARTIALLY_FILLED.value:
                qty = row.get("qty") or 1
                remaining = row.get("remaining_qty") or 0
                risk = risk * (remaining / qty) if qty > 0 else 0.0
            total += risk
        return total

    def _row_to_order(self, row: dict) -> OMSOrder:
        """Convert DB row to OMSOrder."""
        entry_policy = None
        if row.get("entry_policy"):
            ep = row["entry_policy"]
            ep_data = ep if isinstance(ep, dict) else json.loads(ep)
            entry_policy = EntryPolicy(**ep_data)

        risk_context = None
        if row.get("risk_context"):
            rc = row["risk_context"]
            rc_data = rc if isinstance(rc, dict) else json.loads(rc)
            risk_context = RiskContext(**rc_data)

        # Look up instrument from registry, fall back to minimal stub
        instrument = None
        if row.get("instrument_symbol"):
            instrument = InstrumentRegistry.get(row["instrument_symbol"])
            if not instrument:
                logger.warning(f"Unknown instrument {row['instrument_symbol']}, using stub")
                instrument = Instrument(
                    symbol=row["instrument_symbol"],
                    root=row["instrument_symbol"],
                    venue="",
                    tick_size=0.01,
                    tick_value=0.01,
                    multiplier=1.0,
                )

        return OMSOrder(
            oms_order_id=row["oms_order_id"],
            client_order_id=row.get("client_order_id") or "",
            strategy_id=row["strategy_id"],
            account_id=row.get("account_id") or "",
            instrument=instrument,
            side=OrderSide(row["side"]),
            qty=row["qty"],
            order_type=OrderType(row["order_type"]),
            limit_price=row.get("limit_price"),
            stop_price=row.get("stop_price"),
            tif=row.get("tif") or "DAY",
            role=OrderRole(row["role"]),
            entry_policy=entry_policy,
            risk_context=risk_context,
            broker=row.get("broker") or "IBKR",
            broker_order_id=row.get("broker_order_id"),
            perm_id=row.get("perm_id"),
            oca_group=row.get("oca_group") or "",
            status=OrderStatus(row["status"]),
            created_at=row.get("created_at"),
            submitted_at=row.get("submitted_at"),
            acked_at=row.get("acked_at"),
            last_update_at=row.get("last_update_at"),
            filled_qty=row.get("filled_qty") or 0.0,
            remaining_qty=row.get("remaining_qty") or 0.0,
            avg_fill_price=row.get("avg_fill_price") or 0.0,
            reprice_count=row.get("reprice_count") or 0,
        )

    def _row_to_position(self, row: dict) -> Position:
        return Position(
            account_id=row["account_id"],
            instrument_symbol=row["instrument_symbol"],
            strategy_id=row["strategy_id"],
            net_qty=row.get("net_qty") or 0.0,
            avg_price=row.get("avg_price") or 0.0,
            realized_pnl=row.get("realized_pnl") or 0.0,
            unrealized_pnl=row.get("unrealized_pnl") or 0.0,
            open_risk_dollars=row.get("open_risk_dollars") or 0.0,
            open_risk_R=row.get("open_risk_R") or 0.0,
            last_update_at=row.get("last_update_at"),
        )
