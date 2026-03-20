"""Main OMS service."""
import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from ..intent.handler import IntentHandler
    from ..events.bus import EventBus
    from ..execution.router import ExecutionRouter
    from ..reconciliation.orchestrator import ReconciliationOrchestrator
    from ..engine.timeout_monitor import OrderTimeoutMonitor
    from ..models.intent import Intent, IntentReceipt
    from ..models.risk_state import PortfolioRiskState, StrategyRiskState

logger = logging.getLogger(__name__)


class OMSService:
    """Main OMS runtime. Exposes intent API, manages lifecycle."""

    def __init__(
        self,
        intent_handler: "IntentHandler",
        bus: "EventBus",
        reconciler: "ReconciliationOrchestrator",
        router: "ExecutionRouter" = None,
        recon_interval_s: float = 120.0,
        timeout_monitor: "OrderTimeoutMonitor | None" = None,
        get_portfolio_risk: "Callable[[], Awaitable[PortfolioRiskState]] | None" = None,
        get_strategy_risk: "Callable[[str], Awaitable[StrategyRiskState]] | None" = None,
    ):
        self._handler = intent_handler
        self._bus = bus
        self._reconciler = reconciler
        self._router = router
        self._recon_interval = recon_interval_s
        self._timeout_monitor = timeout_monitor
        self._get_portfolio_risk = get_portfolio_risk
        self._get_strategy_risk = get_strategy_risk
        self._ready = asyncio.Event()
        self._recon_task: asyncio.Task | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def event_bus(self) -> "EventBus":
        return self._bus

    async def start(self) -> None:
        """Start OMS: reconcile, start router, start timeout monitor, then accept intents."""
        await self._reconciler.startup_reconciliation()
        if self._router:
            await self._router.start()
        # C4 fix: start order timeout monitor
        if self._timeout_monitor:
            await self._timeout_monitor.start()
        self._recon_task = asyncio.create_task(self._periodic_recon_loop())
        self._ready.set()
        logger.info("OMS service ready")

    async def stop(self) -> None:
        self._ready.clear()
        if self._timeout_monitor:
            await self._timeout_monitor.stop()
        if self._router:
            await self._router.stop()
        if self._recon_task:
            self._recon_task.cancel()
            try:
                await self._recon_task
            except asyncio.CancelledError:
                pass

    async def submit_intent(self, intent: "Intent") -> "IntentReceipt":
        await self._ready.wait()
        return await self._handler.submit(intent)

    def stream_events(self, strategy_id: str) -> "asyncio.Queue":
        """Returns an asyncio.Queue that receives OMSEvent objects for this strategy."""
        return self._bus.subscribe(strategy_id)

    def stream_all_events(self) -> "asyncio.Queue":
        """For dashboard/logging — receives all events."""
        return self._bus.subscribe_all()

    async def get_portfolio_risk(self) -> "PortfolioRiskState":
        if self._get_portfolio_risk is None:
            raise RuntimeError("Portfolio risk provider is not configured")
        return await self._get_portfolio_risk()

    async def get_strategy_risk(self, strategy_id: str) -> "StrategyRiskState":
        if self._get_strategy_risk is None:
            raise RuntimeError("Strategy risk provider is not configured")
        return await self._get_strategy_risk(strategy_id)

    async def _periodic_recon_loop(self) -> None:
        while True:
            await asyncio.sleep(self._recon_interval)
            try:
                await self._reconciler.periodic_reconciliation()
            except Exception as e:
                logger.warning("Periodic recon failed: %s", e)
