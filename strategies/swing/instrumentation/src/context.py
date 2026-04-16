"""InstrumentationContext — single injectable object bundling all services.

Passed as ``instrumentation=ctx`` to every strategy engine.  When ``None``,
engines silently skip all instrumentation calls.
"""
from __future__ import annotations

import logging
import os
import threading
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
    post_exit_tracker: object = None      # PostExitTracker
    bot_id: str = ""
    data_dir: str = "instrumentation/data"
    get_regime_ctx: object = None       # Callable[[], RegimeContext | None]
    get_applied_config: object = None   # Callable[[], PortfolioRulesConfig | None]

    _started: bool = field(default=False, repr=False)
    _backfill_stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _backfill_thread: Optional[threading.Thread] = field(default=None, repr=False)

    def start(self) -> None:
        """Start background services (sidecar thread, post-exit backfill)."""
        if self._started:
            return

        # Enforce HMAC auth in paper/live — match stock bootstrap behavior
        env = os.environ.get("TRADING_MODE", os.environ.get("TRADING_ENV", "dev"))
        if env in ("paper", "live") and self.sidecar is not None:
            hmac_secret = getattr(self.sidecar, "hmac_secret", b"")
            if not hmac_secret:
                raise RuntimeError(
                    f"HMAC secret is required in {env} mode. "
                    f"Set INSTRUMENTATION_HMAC_SECRET environment variable."
                )

        try:
            if self.sidecar is not None:
                self.sidecar.start()
        except Exception as e:
            logger.warning("Sidecar start failed: %s", e)
        try:
            if self.post_exit_tracker is not None:
                self._backfill_stop.clear()
                self._backfill_thread = threading.Thread(
                    target=self._backfill_loop, daemon=True,
                    name="post-exit-backfill",
                )
                self._backfill_thread.start()
                logger.info("Post-exit backfill thread started")
        except Exception as e:
            logger.warning("Post-exit backfill thread start failed: %s", e)
        self._started = True
        logger.info("InstrumentationContext started")

    def stop(self) -> None:
        """Stop background services."""
        if not self._started:
            return
        try:
            if self._backfill_thread is not None:
                self._backfill_stop.set()
                self._backfill_thread.join(timeout=10)
                self._backfill_thread = None
                logger.info("Post-exit backfill thread stopped")
        except Exception as e:
            logger.warning("Post-exit backfill thread stop failed: %s", e)
        try:
            if self.sidecar is not None:
                self.sidecar.stop()
        except Exception as e:
            logger.warning("Sidecar stop failed: %s", e)
        self._started = False
        logger.info("InstrumentationContext stopped")

    def _backfill_loop(self) -> None:
        """Periodically run post-exit backfill (every 30 min)."""
        while not self._backfill_stop.wait(timeout=1800):
            try:
                self.post_exit_tracker.run_backfill()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("Post-exit backfill error: %s", e)
