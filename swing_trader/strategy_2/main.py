"""AKC-Helix Swing v1.3 — entry point that wires everything together."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

logger = logging.getLogger("akchelix")


async def main() -> None:
    """Wire up IB session, OMS, instruments, and start the Helix engine."""
    from shared.ibkr_core.config.loader import IBKRConfig
    from shared.ibkr_core.client.session import IBSession
    from shared.ibkr_core.mapping.contract_factory import ContractFactory
    from shared.ibkr_core.adapters.execution_adapter import IBKRExecutionAdapter
    from shared.oms.services.factory import build_oms_service
    from shared.oms.risk.calculator import RiskCalculator
    from shared.services.bootstrap import bootstrap_database
    from instrumentation.src.bootstrap import bootstrap_kit

    from .config import (
        PORTFOLIO_CAP_R,
        STRATEGY_ID,
        SYMBOL_CONFIGS,
        build_instruments,
    )
    from .engine import HelixEngine

    # -------------------------------------------------------------------
    # 1. Load IBKR configuration
    # -------------------------------------------------------------------
    config_dir = Path(__file__).resolve().parent.parent / "config"
    ibkr_config = IBKRConfig(config_dir)
    logger.info("Loaded IBKR config: host=%s port=%d", ibkr_config.profile.host, ibkr_config.profile.port)

    # -------------------------------------------------------------------
    # 2. Create and connect IB session
    # -------------------------------------------------------------------
    session = IBSession(ibkr_config)
    await session.start()
    await session.wait_ready()
    logger.info("IB session connected")

    # -------------------------------------------------------------------
    # 3. Create execution adapter
    # -------------------------------------------------------------------
    contract_factory = ContractFactory(
        ib=session.ib,
        templates=ibkr_config.contracts,
        routes=ibkr_config.routes,
    )
    adapter = IBKRExecutionAdapter(
        session=session,
        contract_factory=contract_factory,
        account=ibkr_config.profile.account_id,
    )

    # -------------------------------------------------------------------
    # 4. Bootstrap database (graceful degradation)
    # -------------------------------------------------------------------
    bootstrap_ctx = await bootstrap_database()
    trade_recorder = bootstrap_ctx.trade_recorder

    # -------------------------------------------------------------------
    # 5. Register instruments
    # -------------------------------------------------------------------
    instruments = build_instruments()
    logger.info("Registered %d instruments", len(instruments))

    # -------------------------------------------------------------------
    # 6. Build OMS service
    # -------------------------------------------------------------------
    equity = 100_000.0  # fallback default
    try:
        accounts = session.ib.managedAccounts()
        if accounts:
            summary = await session.ib.accountSummaryAsync(accounts[0])
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    equity = float(item.value)
                    logger.info("Account equity from IB: $%.2f", equity)
                    break
    except Exception:
        logger.warning("Could not fetch equity from IB, using default $%.2f", equity)

    unit_risk_dollars = RiskCalculator.compute_unit_risk_dollars(
        nav=equity, unit_risk_pct=0.005,  # 0.50% base risk per Helix spec s8.1
    )

    oms = await build_oms_service(
        adapter=adapter,
        strategy_id=STRATEGY_ID,
        unit_risk_dollars=unit_risk_dollars,
        daily_stop_R=2.5,
        heat_cap_R=PORTFOLIO_CAP_R,
        portfolio_daily_stop_R=5.0,
        db_pool=bootstrap_ctx.pool,
    )

    # -------------------------------------------------------------------
    # 7. Start OMS
    # -------------------------------------------------------------------
    await oms.start()
    logger.info("OMS service started")

    # -------------------------------------------------------------------
    # 8. Optionally load news calendar
    # -------------------------------------------------------------------
    news_calendar: list[tuple[str, object]] = []
    # Could load from JSON/YAML file here if needed:
    #   news_path = config_dir / "news_calendar.json"
    #   if news_path.exists():
    #       import json
    #       from datetime import datetime
    #       with open(news_path) as f:
    #           raw = json.load(f)
    #       for entry in raw:
    #           news_calendar.append((entry["type"], datetime.fromisoformat(entry["datetime"])))

    # -------------------------------------------------------------------
    # 9. Bootstrap instrumentation kit
    # -------------------------------------------------------------------
    kit = bootstrap_kit(
        strategy_id=STRATEGY_ID,
        symbols=list(SYMBOL_CONFIGS.keys()),
    )
    kit._ctx.start()

    # -------------------------------------------------------------------
    # 10. Create and start Helix engine
    # -------------------------------------------------------------------
    engine = HelixEngine(
        ib_session=session,
        oms_service=oms,
        instruments=instruments,
        config=SYMBOL_CONFIGS,
        trade_recorder=trade_recorder,
        equity=equity,
        news_calendar=news_calendar,
        instrumentation_kit=kit,
    )
    await engine.start()

    # -------------------------------------------------------------------
    # 11. Run until interrupted
    # -------------------------------------------------------------------
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows does not support add_signal_handler
            pass

    logger.info("AKC-Helix v1.3 running — press Ctrl+C to stop")

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # -------------------------------------------------------------------
    # 12. Graceful shutdown
    # -------------------------------------------------------------------
    logger.info("Shutting down …")
    await engine.stop()
    await oms.stop()
    await session.stop()
    kit._ctx.stop()

    if bootstrap_ctx.has_db:
        from shared.services.bootstrap import shutdown_database
        await shutdown_database(bootstrap_ctx)

    logger.info("AKC-Helix v1.3 shutdown complete")


def _setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    _setup_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
