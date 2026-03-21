"""Strategy contracts for the compatibility-first kernel scaffold."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from libs.config.models import PortfolioConfig, StrategyManifest, StrategyRegistryConfig


@dataclass(slots=True)
class RuntimeContext:
    """Dependency surface for strategy plugin factories."""

    manifest: StrategyManifest
    registry: StrategyRegistryConfig
    portfolio: PortfolioConfig
    session: Any = None
    market_data: Any = None
    oms: Any = None
    state_store: Any = None
    instrumentation: Any = None
    contracts: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    clock: Any = None
    db_pool: Any = None
    account_gate: Any = None
    family_coordinator: Any = None


@runtime_checkable
class StrategyPlugin(Protocol):
    """Full lifecycle protocol for strategy engines managed by the runtime."""

    strategy_id: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health_status(self) -> dict[str, Any]: ...

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a persisted snapshot (optional)."""
        ...

    def snapshot_state(self) -> dict[str, Any]:
        """Return serializable state for persistence (optional)."""
        ...

    async def on_market_data(self, event: Any) -> None:
        """Handle incoming market data tick/bar (optional)."""
        ...

    async def on_order_event(self, event: Any) -> None:
        """Handle order status change (optional)."""
        ...

    async def on_fill_event(self, event: Any) -> None:
        """Handle fill notification (optional)."""
        ...

