"""Plugin adapter for Helix v4.0 momentum strategy."""
from __future__ import annotations

from typing import Any

from strategies.contracts import RuntimeContext
from .engine import Helix4Engine


class HelixV40Plugin:
    strategy_id = "AKC_Helix_v40"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engine = Helix4Engine(
            ib_session=ctx.session,
            oms_service=ctx.oms,
            instruments=list(ctx.contracts.values()),
            trade_recorder=getattr(ctx.instrumentation, "trade_recorder", None),
            equity=getattr(ctx.portfolio, "allocation", 100_000.0),
            instrumentation=ctx.instrumentation,
        )

    async def start(self) -> None:
        await self._engine.start()

    async def stop(self) -> None:
        await self._engine.stop()

    def health_status(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "running": getattr(self._engine, "_running", False),
        }

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        pass

    def snapshot_state(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id}

    async def on_market_data(self, event: Any) -> None:
        pass  # Engine subscribes directly via IB session

    async def on_order_event(self, event: Any) -> None:
        pass  # Engine listens via OMS event bus

    async def on_fill_event(self, event: Any) -> None:
        pass  # Engine listens via OMS event bus
