"""Instrumentation bootstrap — wires all components and integrates with OMS EventBus."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    from libs.oms.services.oms_service import OMSService

from .market_snapshot import MarketSnapshotService
from .trade_logger import TradeLogger
from .missed_opportunity import MissedOpportunityLogger
from .process_scorer import ProcessScorer
from .daily_snapshot import DailySnapshotBuilder
from .regime_classifier import RegimeClassifier
from .sidecar import Sidecar
from .experiment import ExperimentRegistry
from .order_logger import OrderLogger
from .config_watcher import ConfigWatcher

logger = logging.getLogger("instrumentation.bootstrap")


def _load_config(strategy_id: str, strategy_type: str) -> dict:
    """Load instrumentation_config.yaml and override bot_id with strategy_id."""
    config_path = Path("instrumentation/config/instrumentation_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    config["bot_id"] = strategy_id
    config["strategy_type"] = strategy_type
    config.setdefault("data_dir", "instrumentation/data")
    config.setdefault("data_source_id", "ibkr_cme_nq")

    # Experiment tracking from config or environment
    config["experiment_id"] = config.get("experiment_id") or os.environ.get("EXPERIMENT_ID")
    config["experiment_variant"] = config.get("experiment_variant") or os.environ.get("EXPERIMENT_VARIANT")

    return config


class InstrumentationManager:
    """
    Central bootstrap that creates all instrumentation services and connects
    them to the OMS event bus.

    Usage in strategy main.py:
        instr = InstrumentationManager(oms, strategy_id, strategy_type)
        await instr.start()
        # ... run strategy ...
        await instr.stop()
    """

    def __init__(
        self,
        oms: "OMSService",
        strategy_id: str,
        strategy_type: str,
        data_provider=None,
        pg_store=None,
        family_strategy_ids: list[str] | None = None,
    ):
        self._oms = oms
        self._strategy_id = strategy_id
        self._config = _load_config(strategy_id, strategy_type)

        self.snapshot_service = MarketSnapshotService(self._config, data_provider)
        self.process_scorer = ProcessScorer()
        self.trade_logger = TradeLogger(
            self._config, self.snapshot_service,
            process_scorer=self.process_scorer,
            strategy_type=strategy_type,
            pg_store=pg_store,
            family_strategy_ids=family_strategy_ids,
        )
        self.missed_logger = MissedOpportunityLogger(self._config, self.snapshot_service)
        self.order_logger = OrderLogger(self._config, strategy_type=strategy_type)
        self.experiment_registry = ExperimentRegistry()
        self.daily_builder = DailySnapshotBuilder(self._config, experiment_registry=self.experiment_registry)
        self.regime_classifier = RegimeClassifier(data_provider=data_provider)
        self.sidecar = Sidecar(self._config)

        # Phase 2B: config change detection — only monitor this strategy's config
        _strategy_config_map = {
            "helix": ["strategy.config"],
            "nqdtc": ["strategy_2.config"],
            "vdubus": ["strategy_3.config"],
        }
        config_modules = _strategy_config_map.get(strategy_type, [])
        try:
            self.config_watcher = ConfigWatcher(
                bot_id=strategy_id,
                config_modules=config_modules,
                data_dir=self._config["data_dir"],
            )
        except Exception:
            self.config_watcher = None

        self._event_queue: Optional[asyncio.Queue] = None
        self._event_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        self._running = False

    def get_sidecar_diagnostics(self) -> Optional[dict]:
        """Return sidecar health diagnostics for heartbeat (#24)."""
        try:
            return self.sidecar.get_diagnostics()
        except Exception:
            return None

    def emit_heartbeat(
        self,
        active_positions: int,
        open_orders: int,
        uptime_s: float,
        error_count_1h: int,
        positions: Optional[list] = None,
        portfolio_exposure: Optional[dict] = None,
    ) -> None:
        """Proxy to facade kit for heartbeat emission with position state."""
        try:
            from .facade import InstrumentationKit
            kit = InstrumentationKit(self, strategy_type=self._config.get("strategy_type", ""))
            kit.emit_heartbeat(
                active_positions, open_orders, uptime_s, error_count_1h,
                positions=positions, portfolio_exposure=portfolio_exposure,
            )
        except Exception:
            pass

    async def start(self) -> None:
        """Subscribe to OMS events and start background tasks."""
        if self._running:
            return
        self._running = True

        try:
            self._event_queue = self._oms.stream_all_events()
            self._event_task = asyncio.create_task(self._event_loop())
        except Exception as e:
            logger.warning("Failed to subscribe to OMS events: %s", e)

        interval = self._config.get("market_snapshots", {}).get("interval_seconds", 60)
        self._snapshot_task = asyncio.create_task(self._periodic_snapshot_loop(interval))

        try:
            self.sidecar.start()
        except Exception as e:
            logger.warning("Failed to start sidecar: %s", e)

        logger.info("Instrumentation started for %s", self._strategy_id)

    async def stop(self) -> None:
        """Shutdown: build daily snapshot, stop background tasks, stop sidecar."""
        self._running = False

        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass

        if self._event_queue:
            try:
                self._oms.event_bus.unsubscribe_all(self._event_queue)
            except Exception:
                pass

        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            snapshot = self.daily_builder.build(today)
            self.daily_builder.save(snapshot)
            logger.info("Daily snapshot saved for %s", today)
        except Exception as e:
            logger.warning("Failed to build daily snapshot: %s", e)

        try:
            self.sidecar.run_once()
            self.sidecar.stop()
        except Exception as e:
            logger.warning("Sidecar stop error: %s", e)

        logger.info("Instrumentation stopped for %s", self._strategy_id)

    _ORDER_STATUS_MAP = None  # lazily initialised

    async def _event_loop(self) -> None:
        """Process OMS events — logs RISK_DENIAL, order status, and fills."""
        from libs.oms.models.events import OMSEventType

        if InstrumentationManager._ORDER_STATUS_MAP is None:
            InstrumentationManager._ORDER_STATUS_MAP = {
                OMSEventType.ORDER_FILLED: "FILLED",
                OMSEventType.ORDER_PARTIALLY_FILLED: "PARTIAL_FILL",
                OMSEventType.ORDER_REJECTED: "REJECTED",
                OMSEventType.ORDER_CANCELLED: "CANCELLED",
                OMSEventType.ORDER_EXPIRED: "EXPIRED",
            }

        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                if event.event_type == OMSEventType.RISK_DENIAL:
                    self._handle_risk_denial(event)
                elif event.event_type in self._ORDER_STATUS_MAP:
                    self._handle_order_status(event, self._ORDER_STATUS_MAP[event.event_type])
                elif event.event_type == OMSEventType.FILL:
                    self._handle_fill_event(event)
            except Exception as e:
                logger.warning("Error processing OMS event: %s", e)

    def _handle_risk_denial(self, event) -> None:
        """Log risk denials as missed opportunities with available context."""
        try:
            payload = event.payload or {}
            reason = payload.get("reason", "unknown")

            # Use enriched payload from IntentHandler, fall back to defaults
            symbols = self._config.get("market_snapshots", {}).get("symbols", [])
            pair = payload.get("symbol") or (symbols[0] if symbols else "NQ")
            side = payload.get("side", "UNKNOWN")
            signal = payload.get("signal_name", f"risk_denial_{self._strategy_id}")
            signal_strength = payload.get("signal_strength", 0.0)

            self.missed_logger.log_missed(
                pair=pair,
                side=side,
                signal=signal,
                signal_id=event.oms_order_id or "",
                signal_strength=signal_strength,
                blocked_by="risk_gateway",
                block_reason=reason,
                strategy_type=self._config.get("strategy_type"),
                exchange_timestamp=event.timestamp,
            )
        except Exception as e:
            logger.warning("Failed to log risk denial as missed: %s", e)

    def _handle_order_status(self, event, status_label: str) -> None:
        """Log order status transitions (FILLED, REJECTED, CANCELLED, etc.)."""
        try:
            payload = event.payload or {}
            self.order_logger.log_order(
                order_id=event.oms_order_id or "",
                pair=payload.get("symbol", "NQ"),
                side=payload.get("side", ""),
                order_type=payload.get("order_type", ""),
                status=status_label,
                requested_qty=payload.get("qty", 0),
                reject_reason=payload.get("rejection_reason", ""),
                strategy_type=self._config.get("strategy_type", ""),
                exchange_timestamp=event.timestamp,
            )
        except Exception as e:
            logger.warning("Failed to log order status %s: %s", status_label, e)

    def _handle_fill_event(self, event) -> None:
        """Log fill events with execution details."""
        try:
            payload = event.payload or {}
            self.order_logger.log_order(
                order_id=event.oms_order_id or "",
                pair=payload.get("symbol", "NQ"),
                side=payload.get("side", ""),
                order_type=payload.get("order_type", ""),
                status="FILL",
                requested_qty=payload.get("requested_qty", 0),
                filled_qty=payload.get("qty", 0),
                fill_price=payload.get("price"),
                strategy_type=self._config.get("strategy_type", ""),
                exchange_timestamp=event.timestamp,
            )
        except Exception as e:
            logger.warning("Failed to log fill event: %s", e)

    async def _periodic_snapshot_loop(self, interval: int) -> None:
        """Capture market snapshots at regular intervals."""
        while self._running:
            try:
                self.snapshot_service.run_periodic()
            except Exception as e:
                logger.warning("Periodic snapshot failed: %s", e)
            # Config change detection (Phase 2B)
            try:
                if self.config_watcher:
                    self.config_watcher.check()
            except Exception as e:
                logger.warning("Config watcher check failed: %s", e)
            # Post-exit price backfill
            try:
                if hasattr(self.snapshot_service, '_data_provider') and self.snapshot_service._data_provider:
                    self.trade_logger.run_post_exit_backfill(self.snapshot_service._data_provider)
            except Exception as e:
                logger.warning("Post-exit backfill failed: %s", e)
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
