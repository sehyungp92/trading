"""InstrumentationContext — single injectable object bundling all services.

Passed as ``instrumentation=ctx`` to every strategy engine.  When ``None``,
engines silently skip all instrumentation calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("instrumentation.context")


@dataclass
class InstrumentationContext:
    """Bundles all instrumentation services into one injectable object."""

    snapshot_service: object = None   # MarketSnapshotService
    trade_logger: object = None       # TradeLogger
    missed_logger: object = None      # MissedOpportunityLogger
    process_scorer: object = None     # ProcessScorer
    daily_builder: object = None      # DailySnapshotBuilder
    regime_classifier: object = None  # RegimeClassifier
    sidecar: object = None            # Sidecar
    drawdown_tracker: object = None   # DrawdownTracker
    overnight_gap_tracker: object = None  # OvernightGapTracker
    coordination_logger: object = None   # CoordinationLogger
    order_logger: object = None           # OrderLogger
    indicator_logger: object = None       # IndicatorLogger
    filter_logger: object = None          # FilterLogger
    orderbook_logger: object = None       # OrderBookLogger
    experiment_registry: object = None    # ExperimentRegistry
    overlay_state_provider: object = None  # Callable[[], dict[str, bool]]
    bot_id: str = ""
    data_dir: str = "instrumentation/data"

    _started: bool = field(default=False, repr=False)

    def start(self) -> None:
        """Start background services (sidecar thread)."""
        if self._started:
            return
        try:
            if self.sidecar is not None:
                self.sidecar.start()
        except Exception as e:
            logger.warning("Sidecar start failed: %s", e)
        self._started = True
        logger.info("InstrumentationContext started")

    def stop(self) -> None:
        """Stop background services."""
        if not self._started:
            return
        try:
            if self.sidecar is not None:
                self.sidecar.stop()
        except Exception as e:
            logger.warning("Sidecar stop failed: %s", e)
        self._started = False
        logger.info("InstrumentationContext stopped")
