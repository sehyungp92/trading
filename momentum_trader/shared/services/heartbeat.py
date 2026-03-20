"""Heartbeat service for strategy and adapter health monitoring.

Strategies and adapters should call these functions periodically
to update their health state in the database.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from ..oms.persistence.postgres import PgStore
from ..oms.persistence.schema import StrategyStateRow, AdapterStateRow


class HeartbeatService:
    """Manages health state updates for strategies and adapters."""

    def __init__(self, store: PgStore):
        self._store = store

    async def strategy_heartbeat(
        self,
        strategy_id: str,
        mode: str = "RUNNING",
        heat_r: Decimal = Decimal("0"),
        daily_pnl_r: Decimal = Decimal("0"),
        last_decision_code: Optional[str] = None,
        last_decision_details: Optional[dict] = None,
        last_seen_bar_ts: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update strategy health state. Call every 10-30 seconds."""
        row = StrategyStateRow(
            strategy_id=strategy_id,
            mode=mode,
            last_heartbeat_ts=datetime.now(timezone.utc),
            heat_r=heat_r,
            daily_pnl_r=daily_pnl_r,
            last_decision_code=last_decision_code,
            last_decision_details_json=json.dumps(last_decision_details)
            if last_decision_details
            else "{}",
            last_seen_bar_ts=last_seen_bar_ts,
            last_error=error,
            last_error_ts=datetime.now(timezone.utc) if error else None,
        )
        await self._store.upsert_strategy_state(row)

    async def adapter_heartbeat(
        self,
        adapter_id: str,
        connected: bool,
        broker: str = "IBKR",
    ) -> None:
        """Update adapter connection state. Call every 10-30 seconds."""
        row = AdapterStateRow(
            adapter_id=adapter_id,
            broker=broker,
            connected=connected,
            last_heartbeat_ts=datetime.now(timezone.utc),
        )
        await self._store.upsert_adapter_state(row)

    async def record_disconnect(
        self,
        adapter_id: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record adapter disconnect event."""
        await self._store.record_adapter_disconnect(adapter_id, error_code, error_message)

    async def record_reconnect(self, adapter_id: str) -> None:
        """Record adapter reconnection."""
        await self._store.record_adapter_connect(adapter_id)


async def emit_heartbeat(
    store: PgStore,
    strategy_id: str,
    heat_r: float = 0.0,
    daily_pnl_r: float = 0.0,
    mode: str = "RUNNING",
    decision_code: str = None,
    sidecar_diagnostics: Optional[dict] = None,
) -> None:
    """Simple heartbeat for strategy engines."""
    details = {"sidecar": sidecar_diagnostics} if sidecar_diagnostics else None
    svc = HeartbeatService(store)
    await svc.strategy_heartbeat(
        strategy_id=strategy_id,
        mode=mode,
        heat_r=Decimal(str(heat_r)),
        daily_pnl_r=Decimal(str(daily_pnl_r)),
        last_decision_code=decision_code,
        last_decision_details=details,
    )
