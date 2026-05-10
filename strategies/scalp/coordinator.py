"""RESEARCH-PHASE STUB — DO NOT ENABLE.

The scalp family is intentionally in research phase as of 2026-05-10. This
coordinator is **not registered** in ``apps/runtime/runtime.py``'s
``_FAMILY_COORDINATORS`` map, and both scalp YAML entries
(``SCALP_IVB_AUCTION`` / ``SCALP_PO3_REVERSAL``) are ``enabled: false``.

That deliberate state should not change without first rebuilding this
coordinator to mirror the momentum pattern:

    * per-strategy ``build_oms_service`` (currently absent)
    * ``ctx.account_gate`` wiring (currently absent)
    * ``PortfolioRulesConfig`` (currently absent)
    * ``InstrumentationManager`` + sidecar (currently absent)
    * heartbeat task (currently absent)

If you flip ``enabled: true`` AND register this coordinator in
``_FAMILY_COORDINATORS`` without those, the stub will dispatch the IVB
auction and PO3 reversal plugins with no risk gate, no instrumentation,
and no heartbeat. That is unsafe in any environment beyond a unit-test
harness.

The audit reports that surfaced this state are:

    * ``docs/repo-audit-2026-05-09.md`` (F-S2)
    * ``docs/live-paper-trading-strategy-oms-audit-2026-05-09.md`` (P2 scalp)
    * ``docs/live-paper-trading-audit-2026-05-10.md`` (P1-5)

User direction (2026-05-10): keep the file in place; do not delete or
rebuild. This module is preserved as a research-phase artifact.
"""
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

