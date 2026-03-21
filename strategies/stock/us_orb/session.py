"""Thin runtime bootstrap for the ORB strategy."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from strategies.stock.instrumentation.src.bootstrap import InstrumentationManager
from strategies.stock.instrumentation.src.facade import InstrumentationKit
from strategies.stock.instrumentation.src.pg_bridge import InstrumentedTradeRecorder
from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter
from libs.broker_ibkr.session import IBSession
from libs.broker_ibkr.config.loader import IBKRConfig
from libs.broker_ibkr.mapping.contract_factory import ContractFactory
from libs.oms.persistence.db_config import get_environment
from libs.oms.risk.calculator import RiskCalculator
from libs.oms.services.factory import build_oms_service
from libs.services.deployment import resolve_paper_nav, resolve_strategy_capital_allocation
from libs.services.bootstrap import BootstrapContext, bootstrap_database, shutdown_database

from .config import STRATEGY_ID, StrategySettings

logger = logging.getLogger(__name__)


@dataclass
class RuntimeServices:
    config: IBKRConfig
    session: IBSession
    contract_factory: ContractFactory
    adapter: IBKRExecutionAdapter
    oms: object
    bootstrap: BootstrapContext
    trade_recorder: object | None
    instrumentation: InstrumentationManager | None
    instrumentation_kit: InstrumentationKit | None
    account_id: str
    deploy_mode: str
    capital_fraction: float
    raw_nav: float
    allocated_nav: float

    @property
    def nav(self) -> float:
        return self.allocated_nav


async def fetch_nav(session: IBSession, account_id: str = "", default_nav: float = 100_000.0) -> float:
    environment = get_environment()

    def _fallback_or_raise(reason: str, exc: Exception | None = None) -> float:
        if environment in ("dev", "backtest"):
            logger.warning(
                "Falling back to default NAV %.2f in %s because %s",
                default_nav,
                environment,
                reason,
            )
            return float(default_nav)
        message = (
            f"Unable to fetch IB NetLiquidation for {STRATEGY_ID} in {environment}: {reason}. "
            "Refusing to size positions off a guessed NAV."
        )
        if exc is not None:
            raise RuntimeError(message) from exc
        raise RuntimeError(message)

    try:
        accounts = session.ib.managedAccounts()
    except Exception as exc:
        return _fallback_or_raise("managedAccounts() failed", exc)
    if not accounts:
        return _fallback_or_raise("no managed accounts were returned")

    target = account_id if account_id and account_id in accounts else accounts[0]
    if account_id and account_id not in accounts:
        logger.warning("configured account_id %s not in managedAccounts %s — falling back to %s",
                        account_id, accounts, target)

    try:
        summary = await session.ib.accountSummaryAsync(target)
    except Exception as exc:
        return _fallback_or_raise("accountSummaryAsync() failed", exc)

    for item in summary:
        if item.tag == "NetLiquidation" and item.currency == "USD":
            return float(item.value)

    return _fallback_or_raise("NetLiquidation USD was missing from the account summary")


async def bootstrap_runtime(
    config_dir: str | Path,
    settings: StrategySettings | None = None,
) -> RuntimeServices:
    settings = settings or StrategySettings()
    capital_allocation = resolve_strategy_capital_allocation(STRATEGY_ID, raw_nav=0.0)
    capital_allocation.assert_enabled()
    config = IBKRConfig(Path(config_dir))

    session = IBSession(config)
    await session.start()
    await session.wait_ready()

    contract_factory = ContractFactory(
        ib=session.ib,
        templates=config.contracts,
        routes=config.routes,
    )
    adapter = IBKRExecutionAdapter(
        session=session,
        contract_factory=contract_factory,
        account=config.profile.account_id,
    )

    bootstrap = await bootstrap_database()
    raw_nav = await fetch_nav(session, account_id=config.profile.account_id)
    effective_nav = resolve_paper_nav(raw_nav, Path("data/strategy_orb/paper_capital_state.json"))
    # Unified capital allocation (family + within-family split)
    _unified_config_dir = Path(os.environ.get("CONFIG_DIR", str(Path(__file__).resolve().parent.parent.parent / "config")))
    try:
        from libs.config.capital_bootstrap import bootstrap_capital
        from libs.risk import AccountRiskGate
        _alloc = bootstrap_capital(effective_nav, _unified_config_dir).get(STRATEGY_ID)
        if _alloc:
            _allocated_nav = _alloc.allocated_nav
            logger.info("Unified capital allocation: %s → $%.2f (%.1f%% of $%.2f)", STRATEGY_ID, _allocated_nav, _alloc.capital_pct, effective_nav)
        else:
            raise ValueError(f"Strategy {STRATEGY_ID} not in unified config")
    except Exception as e:
        logger.warning("Unified allocation failed (%s), falling back to deployment.py", e)
        capital_allocation = resolve_strategy_capital_allocation(STRATEGY_ID, raw_nav=effective_nav)
        capital_allocation.assert_enabled()
        capital_allocation.assert_positive_allocated_nav()
        _allocated_nav = capital_allocation.allocated_nav
        logger.info(
            "Fallback allocation: %s deploy_mode=%s allocated_nav=%.2f capital_pct=%.2f",
            STRATEGY_ID, capital_allocation.deploy_mode, _allocated_nav, capital_allocation.capital_pct,
        )

    unit_risk = RiskCalculator.compute_unit_risk_dollars(_allocated_nav, settings.base_risk_pct)
    try:
        account_gate = AccountRiskGate(bootstrap.pool) if bootstrap.pool else None
    except NameError:
        account_gate = None  # libs not available outside unified deployment

    oms = await build_oms_service(
        adapter=adapter,
        strategy_id=STRATEGY_ID,
        unit_risk_dollars=unit_risk,
        daily_stop_R=settings.daily_stop_r,
        heat_cap_R=settings.heat_cap_r,
        portfolio_daily_stop_R=settings.portfolio_daily_stop_r,
        db_pool=bootstrap.pool,
        portfolio_rules_config=None,
        get_current_equity=lambda: _allocated_nav,
        family_id="stock",
        account_gate=account_gate,
    )
    await oms.start()

    instrumentation = InstrumentationManager(
        oms=oms,
        strategy_id=STRATEGY_ID,
        strategy_type="strategy_orb",
    )
    instrumentation_kit = InstrumentationKit(instrumentation, strategy_type="strategy_orb")

    trade_recorder = bootstrap.trade_recorder
    if bootstrap.trade_recorder is not None or instrumentation_kit is not None:
        trade_recorder = InstrumentedTradeRecorder(
            bootstrap.trade_recorder,
            instrumentation_kit,
            strategy_id=STRATEGY_ID,
            strategy_type="strategy_orb",
        )

    return RuntimeServices(
        config=config,
        session=session,
        contract_factory=contract_factory,
        adapter=adapter,
        oms=oms,
        bootstrap=bootstrap,
        trade_recorder=trade_recorder,
        instrumentation=instrumentation,
        instrumentation_kit=instrumentation_kit,
        account_id=config.profile.account_id,
        deploy_mode=capital_allocation.deploy_mode,
        capital_fraction=capital_allocation.capital_fraction,
        raw_nav=capital_allocation.raw_nav,
        allocated_nav=capital_allocation.allocated_nav,
    )


async def shutdown_runtime(services: RuntimeServices) -> None:
    if services.instrumentation is not None:
        await services.instrumentation.stop()
    await services.oms.stop()
    await services.session.stop()
    if services.bootstrap.has_db:
        await shutdown_database(services.bootstrap)
