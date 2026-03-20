"""Main runtime entrypoints for IARIC."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from datetime import date, datetime, time, timezone
from pathlib import Path

from instrumentation.src.strategy_data_providers import IARICInstrumentationDataProvider
from shared.services.heartbeat import emit_heartbeat

from .artifact_store import load_watchlist_artifact
from .config import ET, STRATEGY_ID, StrategySettings
from .data import IBMarketDataSource
from .diagnostics import JsonlDiagnostics
from .engine import IARICEngine
from .research import run_daily_selection
from .research_generator import generate_research_snapshot
from .session import bootstrap_runtime, shutdown_runtime

logger = logging.getLogger(__name__)


async def run_intraday_session(
    trading_date: date | None = None,
    artifact=None,
    settings: StrategySettings | None = None,
) -> None:
    cfg = settings or StrategySettings()
    config_dir = Path(__file__).resolve().parents[1] / "config"
    services = await bootstrap_runtime(config_dir=config_dir, settings=cfg)
    diagnostics = JsonlDiagnostics(cfg.diagnostics_dir)
    session_date = trading_date or datetime.now(ET).date()
    if artifact is not None:
        active_artifact = artifact
    else:
        try:
            active_artifact = load_watchlist_artifact(session_date, settings=cfg)
        except FileNotFoundError:
            active_artifact = run_daily_selection(session_date, settings=cfg, diagnostics=diagnostics)
    engine = IARICEngine(
        oms_service=services.oms,
        artifact=active_artifact,
        account_id=services.account_id,
        nav=services.allocated_nav,
        settings=cfg,
        trade_recorder=services.trade_recorder,
        diagnostics=diagnostics,
        instrumentation=services.instrumentation_kit,
    )
    snapshot = IARICEngine.try_load_state(active_artifact.trade_date, settings=cfg)
    if snapshot is not None:
        engine.hydrate_state(snapshot)

    # Reconcile OMS vs broker positions on startup to catch orphans from prior crashes
    try:
        await engine._reconcile_after_reconnect()
    except Exception:
        logger.exception("Startup position reconciliation failed — continuing")

    market_data = IBMarketDataSource(
        ib=services.session.ib,
        contract_factory=services.contract_factory,
        on_quote=engine.on_quote,
        on_bar=engine.on_bar,
    )

    async def _on_reconnect() -> None:
        await engine._reconcile_after_reconnect()
        market_data.invalidate_subscriptions()
        logger.info("Market data subscriptions invalidated after reconnect — main loop will re-subscribe")

    services.session.set_reconnect_callback(_on_reconnect)
    stop_event = asyncio.Event()
    started_at = datetime.now(timezone.utc)

    def _record_runtime_error(error_type: str, exc: Exception, *, severity: str = "medium") -> None:
        if services.instrumentation_kit is None:
            return
        services.instrumentation_kit.log_error(
            error_type=error_type,
            message=str(exc),
            severity=severity,
            category="dependency",
            context={"component": "strategy_iaric.main"},
            exc=exc,
        )

    async def heartbeat_loop() -> None:
        while True:
            try:
                risk_state = await services.oms.get_strategy_risk(STRATEGY_ID)
                heat_r = float(getattr(risk_state, "open_risk_R", 0.0))
                daily_pnl_r = float(getattr(risk_state, "daily_realized_R", 0.0))
                sidecar_diagnostics = (
                    services.instrumentation.get_sidecar_diagnostics()
                    if services.instrumentation is not None
                    else None
                )
                if services.bootstrap.has_db:
                    await emit_heartbeat(
                        services.bootstrap.pg_store,
                        STRATEGY_ID,
                        heat_r=heat_r,
                        daily_pnl_r=daily_pnl_r,
                        mode="RUNNING",
                        sidecar_diagnostics=sidecar_diagnostics,
                    )
                if services.instrumentation_kit is not None:
                    positions = engine.get_position_snapshot()
                    gross_notional = sum(
                        abs(float(pos.get("entry_price", 0.0) or 0.0) * float(pos.get("qty", 0.0) or 0.0))
                        for pos in positions
                    )
                    exposure_pct = (
                        (gross_notional / services.allocated_nav) * 100.0
                        if services.allocated_nav > 0
                        else 0.0
                    )
                    services.instrumentation_kit.emit_heartbeat(
                        active_positions=len(positions),
                        open_orders=engine.open_order_count(),
                        uptime_s=(datetime.now(timezone.utc) - started_at).total_seconds(),
                        error_count_1h=(
                            services.instrumentation.recent_error_count_1h()
                            if services.instrumentation is not None
                            else 0
                        ),
                        positions=positions,
                        portfolio_exposure={
                            "total_positions": len(positions),
                            "open_orders": engine.open_order_count(),
                            "heat_r": heat_r,
                            "daily_pnl_r": daily_pnl_r,
                            "gross_notional": round(gross_notional, 2),
                            "allocated_nav": round(services.allocated_nav, 2),
                            "exposure_pct": round(exposure_pct, 4),
                            "by_strategy": {
                                STRATEGY_ID: {
                                    "positions": len(positions),
                                    "open_orders": engine.open_order_count(),
                                }
                            },
                        },
                    )
            except Exception as exc:
                logger.exception("Heartbeat failed")
                _record_runtime_error("heartbeat_loop_error", exc)
            await asyncio.sleep(30)

    heartbeat_task: asyncio.Task | None = None

    def _handle_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_stop)

    try:
        if services.instrumentation is not None:
            services.instrumentation.attach_data_provider(IARICInstrumentationDataProvider(engine))
            await services.instrumentation.start()

        await engine.start()
        await market_data.start()
        await market_data.ensure_hot_symbols(engine.subscription_instruments())
        await market_data.poll_due_bars(engine.polling_instruments())
        heartbeat_task = asyncio.create_task(heartbeat_loop())

        while not stop_event.is_set():
            now = datetime.now(ET)
            if now.date() != active_artifact.trade_date or now.time() >= time(16, 1):
                stop_event.set()
                continue
            try:
                await asyncio.wait_for(services.session.wait_until_ready(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            try:
                await market_data.ensure_hot_symbols(engine.subscription_instruments())
                await market_data.poll_due_bars(engine.polling_instruments(), now=now)
            except Exception as exc:
                logger.exception("Intraday market-data loop failed")
                _record_runtime_error("intraday_market_data_error", exc, severity="high")
                await asyncio.sleep(5.0)
                continue
            await asyncio.sleep(5.0)
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        await market_data.stop()
        await engine.stop()
        await shutdown_runtime(services)


async def main() -> None:
    cfg = StrategySettings()
    logger.info("IARIC strategy started — premarket %s–%s, market %s–%s ET",
                cfg.premarket_start, cfg.premarket_end, cfg.market_open, cfg.close_block_start)
    ran_selection_for: date | None = None
    while True:
        now = datetime.now(ET)
        today = now.date()
        try:
            if cfg.premarket_start <= now.time() <= cfg.premarket_end and ran_selection_for != today:
                config_dir = Path(__file__).resolve().parents[1] / "config"
                services = await bootstrap_runtime(config_dir=config_dir, settings=cfg)
                try:
                    await generate_research_snapshot(today, ib=services.session.ib, settings=cfg)
                finally:
                    await shutdown_runtime(services)
                run_daily_selection(today, settings=cfg, diagnostics=JsonlDiagnostics(cfg.diagnostics_dir))
                ran_selection_for = today
            if cfg.market_open <= now.time() < cfg.close_block_start:
                await run_intraday_session(trading_date=today, settings=cfg)
        except Exception:
            logger.exception("IARIC main loop failed")
        await asyncio.sleep(5)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
