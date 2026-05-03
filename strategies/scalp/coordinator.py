from __future__ import annotations

from typing import Any

from strategies.contracts import RuntimeContext


class ScalpFamilyCoordinator:
    family_id = "scalp"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._plugins: list[Any] = []

    async def start(self) -> None:
        from strategies.scalp.ivb_auction.plugin import IvbAuctionPlugin
        from strategies.scalp.po3_reversal.plugin import Po3ReversalPlugin

        self._plugins = [IvbAuctionPlugin(self._ctx), Po3ReversalPlugin(self._ctx)]
        for plugin in self._plugins:
            await plugin.start()

    async def stop(self) -> None:
        for plugin in reversed(self._plugins):
            await plugin.stop()
        self._plugins.clear()

    def health_status(self) -> dict[str, Any]:
        return {
            "family": self.family_id,
            "strategies": {
                plugin.strategy_id: plugin.health_status()
                for plugin in self._plugins
            },
        }

