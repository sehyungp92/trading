"""Plugin adapter for US Opening Range Breakout stock strategy."""
from __future__ import annotations

import logging
from typing import Any

from strategies.contracts import RuntimeContext
from .config import StrategySettings
from .diagnostics import JsonlDiagnostics
from .engine import USORBEngine

logger = logging.getLogger(__name__)


class USORBPlugin:
    strategy_id = "US_ORB_v1"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        manifest = ctx.manifest
        settings = StrategySettings()

        # account_id from the connection group tied to this strategy
        conn_group = ctx.registry.connection_groups[manifest.connection_group]
        account_id = conn_group.account_id or ""

        # NAV: use portfolio capital allocation; fallback to initial equity
        nav = ctx.portfolio.capital.strategy_allocations.get(
            self.strategy_id,
            ctx.portfolio.capital.initial_equity,
        )

        trade_recorder = getattr(ctx.instrumentation, "trade_recorder", None)
        diagnostics = JsonlDiagnostics(settings.diagnostics_dir, enabled=True)

        # US_ORB uses a plain dict cache for market data
        cache: dict[str, Any] = {}

        self._engine = USORBEngine(
            oms_service=ctx.oms,
            cache=cache,
            account_id=account_id,
            nav=nav,
            settings=settings,
            trade_recorder=trade_recorder,
            diagnostics=diagnostics,
            instrumentation=ctx.instrumentation,
        )

    # -- lifecycle --------------------------------------------------------

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
        pass  # Stock engines load artifacts on start

    def snapshot_state(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id}

    async def on_market_data(self, event: Any) -> None:
        pass

    async def on_order_event(self, event: Any) -> None:
        pass

    async def on_fill_event(self, event: Any) -> None:
        pass
