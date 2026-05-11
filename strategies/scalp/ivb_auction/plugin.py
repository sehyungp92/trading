from __future__ import annotations

from typing import Any

from strategies.contracts import RuntimeContext
from strategies.core.capital import resolve_plugin_nav

from .engine import IvbAuctionEngine


class IvbAuctionPlugin:
    strategy_id = "SCALP_IVB_AUCTION"

    def __init__(self, ctx: RuntimeContext) -> None:
        engine_config = dict(getattr(ctx.manifest, "engine_config", {}) or {})
        self._engine = IvbAuctionEngine(
            ib_session=ctx.session,
            oms_service=ctx.oms,
            instruments=dict(ctx.contracts),
            trade_recorder=getattr(ctx.instrumentation, "trade_recorder", None),
            equity=resolve_plugin_nav(ctx, self.strategy_id),
            instrumentation=ctx.instrumentation,
            **engine_config,
        )

    async def start(self) -> None:
        await self._engine.start()

    async def stop(self) -> None:
        await self._engine.stop()

    def health_status(self) -> dict[str, Any]:
        return self._engine.health_status()

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        await self._engine.hydrate(snapshot)

    def snapshot_state(self) -> dict[str, Any]:
        return self._engine.snapshot_state()

    async def on_market_data(self, event: Any) -> None:
        pass

    async def on_order_event(self, event: Any) -> None:
        pass

    async def on_fill_event(self, event: Any) -> None:
        pass
