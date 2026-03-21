"""AKC-Helix NQ TrendWrap v4.0 Apex Trail — entry point wiring IB session, OMS, and engine."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger("helix_v40")


async def main() -> None:
    from libs.broker_ibkr.config.loader import IBKRConfig
    from libs.broker_ibkr.session import IBSession
    from libs.broker_ibkr.mapping.contract_factory import ContractFactory
    from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter
    from libs.oms.services.factory import build_oms_service
    from libs.oms.risk.calculator import RiskCalculator
    from libs.services.bootstrap import bootstrap_database

    # Capital allocation from unified config
    _repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from libs.config.capital_bootstrap import bootstrap_capital
    from libs.risk import AccountRiskGate

    from .config import STRATEGY_ID, BASE_RISK_PCT, HEAT_CAP_R, DAILY_STOP_R, PORTFOLIO_DAILY_STOP_R, build_instruments
    from .engine import Helix4Engine

    config_dir = Path(__file__).resolve().parent.parent / "config"
    ibkr_config = IBKRConfig(config_dir)
    logger.info("IBKR config: %s:%d", ibkr_config.profile.host, ibkr_config.profile.port)

    session = IBSession(ibkr_config)
    await session.start()
    await session.wait_ready()
    logger.info("IB session connected")

    contract_factory = ContractFactory(
        ib=session.ib, templates=ibkr_config.contracts, routes=ibkr_config.routes,
    )
    adapter = IBKRExecutionAdapter(
        session=session, contract_factory=contract_factory,
        account=ibkr_config.profile.account_id,
    )

    bootstrap_ctx = await bootstrap_database()
    trade_recorder = bootstrap_ctx.trade_recorder

    instruments = build_instruments()
    logger.info("Registered %d instruments", len(instruments))

    from libs.oms.persistence.db_config import get_environment
    paper_mode = get_environment() == "paper"
    paper_equity_pool = None

    if paper_mode:
        from libs.oms.paper_equity import load_paper_equity
        equity = await load_paper_equity(bootstrap_ctx.pool)
        paper_equity_pool = bootstrap_ctx.pool
        logger.info("Paper mode equity: $%.2f", equity)
    else:
        equity = 100_000.0
        try:
            accounts = session.ib.managedAccounts()
            if accounts:
                summary = await session.ib.accountSummaryAsync(accounts[0])
                for item in summary:
                    if item.tag == "NetLiquidation" and item.currency == "USD":
                        equity = float(item.value)
                        logger.info("Equity: $%.2f", equity)
                        break
        except Exception:
            logger.warning("Using default equity $%.2f", equity)

    import os as _os
    _unified_config_dir = Path(_os.environ.get("CONFIG_DIR", str(Path(__file__).resolve().parent.parent.parent / "config")))
    try:
        _alloc = bootstrap_capital(equity, _unified_config_dir).get(STRATEGY_ID)
        if _alloc:
            _allocated_nav = _alloc.allocated_nav
            logger.info("Capital allocation: %s → $%.2f (%.1f%% of $%.2f)", STRATEGY_ID, _allocated_nav, _alloc.capital_pct, equity)
        else:
            _allocated_nav = equity
            logger.warning("Strategy %s not found in unified config, using full equity", STRATEGY_ID)
    except Exception as e:
        _allocated_nav = equity
        logger.warning("bootstrap_capital failed (%s), using full equity", e)

    unit_risk = RiskCalculator.compute_unit_risk_dollars(nav=_allocated_nav, unit_risk_pct=BASE_RISK_PCT)
    _live_equity = [_allocated_nav]

    from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
    portfolio_rules = PortfolioRulesConfig(initial_equity=_allocated_nav)

    account_gate = AccountRiskGate(bootstrap_ctx.pool) if bootstrap_ctx.pool else None

    oms = await build_oms_service(
        adapter=adapter,
        strategy_id=STRATEGY_ID,
        unit_risk_dollars=unit_risk,
        daily_stop_R=DAILY_STOP_R,
        heat_cap_R=HEAT_CAP_R,
        portfolio_daily_stop_R=PORTFOLIO_DAILY_STOP_R,
        db_pool=bootstrap_ctx.pool,
        portfolio_rules_config=portfolio_rules,
        get_current_equity=lambda: _live_equity[0],
        paper_equity_pool=paper_equity_pool,
        family_id="momentum",
        account_gate=account_gate,
    )
    await oms.start()
    logger.info("OMS started")

    # Instrumentation
    instr = None
    try:
        from strategies.momentum.instrumentation.src.bootstrap import InstrumentationManager
        instr = InstrumentationManager(oms, STRATEGY_ID, strategy_type="helix")
        await instr.start()
    except Exception as e:
        logger.warning("Instrumentation init failed (non-fatal): %s", e)

    engine = Helix4Engine(
        ib_session=session,
        oms_service=oms,
        instruments=instruments,
        trade_recorder=trade_recorder,
        equity=equity,
        instrumentation=instr,
    )
    await engine.start()

    heartbeat_task = None
    if bootstrap_ctx.has_db:
        from libs.services.heartbeat import emit_heartbeat

        _hb_start_time = time.time()

        async def _heartbeat_loop():
            while True:
                try:
                    sidecar_diag = instr.get_sidecar_diagnostics() if instr else None
                    is_conn = session.is_connected
                    mode = "RUNNING" if is_conn else "DEGRADED"
                    await emit_heartbeat(
                        bootstrap_ctx.pg_store, STRATEGY_ID, mode=mode,
                        sidecar_diagnostics=sidecar_diag,
                    )
                except Exception as e:
                    logger.warning("Heartbeat failed: %s", e)
                # Sync live equity from engine refresh
                try:
                    _live_equity[0] = engine.equity
                except Exception:
                    pass
                if instr:
                    try:
                        positions_data = engine.get_position_snapshot()
                        instr.emit_heartbeat(
                            active_positions=len(positions_data),
                            open_orders=0,
                            uptime_s=(time.time() - _hb_start_time),
                            error_count_1h=0,
                            positions=positions_data,
                        )
                    except Exception:
                        pass
                await asyncio.sleep(30)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    logger.info("AKC-Helix v4.0 Apex Trail running — Ctrl+C to stop")
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    if heartbeat_task:
        heartbeat_task.cancel()

    logger.info("Shutting down")
    if instr:
        try:
            await instr.stop()
        except Exception as e:
            logger.warning("Instrumentation shutdown error: %s", e)
    await engine.stop()
    await oms.stop()
    await session.stop()
    if bootstrap_ctx.has_db:
        from libs.services.bootstrap import shutdown_database
        await shutdown_database(bootstrap_ctx)
    logger.info("Shutdown complete")


def _setup_logging() -> None:
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
