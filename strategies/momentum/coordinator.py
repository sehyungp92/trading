"""Momentum family coordinator — each strategy gets its own OMS instance.

Unlike swing (shared OMS), momentum strategies are independent:
  - AKC_Helix_v40        (Helix4Engine)     — NQ/MNQ futures trend-following
  - NQDTC_v2.1           (NQDTCEngine)      — MNQ day-trade continuation
  - VdubusNQ_v4          (VdubNQv4Engine)   — MNQ Vdubus signals
  - DownturnDominator_v1 (DownturnEngine)   — MNQ short-only regime scalping

Cross-strategy coordination is config-driven via PortfolioRulesConfig
(cooldown pairs, direction filter), NOT via in-process signaling.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from typing import Any

from libs.oms.persistence.db_config import get_environment

from strategies.contracts import RuntimeContext

logger = logging.getLogger(__name__)


class MomentumFamilyCoordinator:
    """Lifecycle manager for four momentum strategies.

    Each strategy receives its own OMS built via ``build_oms_service``.
    Shared across all four: *db_pool*, *AccountRiskGate*, *family_id*.
    """

    family_id = "momentum"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engines: list = []
        self._oms_services: list = []
        self._instrumentations: list = []
        self._strategy_ids: list[str] = []
        self._portfolio_checkers: list = []
        self._base_portfolio_rules: Any = None
        self._base_max_family_contracts: int = 0
        self._regime_ctx: Any = None
        self._heartbeat_task: asyncio.Task | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:  # noqa: C901 — wiring is inherently sequential
        """Import, build OMS, and start all four momentum engines."""
        from libs.broker_ibkr.config.loader import IBKRConfig
        from libs.broker_ibkr.mapping.contract_factory import ContractFactory
        from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter
        from libs.oms.services.factory import build_oms_service
        from libs.oms.risk.calculator import RiskCalculator
        from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
        from libs.config.capital_bootstrap import bootstrap_capital

        ctx = self._ctx
        session = ctx.session
        db_pool = ctx.db_pool
        account_gate = ctx.account_gate

        if session is None:
            raise RuntimeError(
                "Momentum family requires an IB session (connect_ib=True)"
            )

        # ── Execution adapter (shared IB session, one adapter instance) ──
        config_dir = Path(os.environ.get("CONFIG_DIR", str(Path(__file__).resolve().parent.parent.parent / "config")))
        ibkr_config = IBKRConfig(config_dir)
        contract_factory = ContractFactory(
            ib=session.ib,
            templates=ibkr_config.contracts,
            routes=ibkr_config.routes,
        )

        # ── Resolve equity ───────────────────────────────────────────
        paper_mode = get_environment() == "paper"
        paper_equity_pool: Any = None
        equity: float

        if paper_mode:
            # Paper equity is stored in DB; query directly.
            row = await db_pool.fetchrow(
                "SELECT equity FROM paper_equity WHERE account_scope = $1",
                self.family_id,
            )
            if row is not None:
                equity = float(row["equity"])
            else:
                equity = float(os.getenv("PAPER_INITIAL_EQUITY", "10000"))
            paper_equity_pool = db_pool
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
                            logger.info("Account equity: $%.2f", equity)
                            break
            except Exception:
                logger.warning("Using default equity $%.2f", equity)

        # ── Capital allocation ───────────────────────────────────────
        try:
            allocs = bootstrap_capital(equity, config_dir)
        except Exception as exc:
            logger.warning("bootstrap_capital failed (%s), using full equity", exc)
            allocs = {}

        # ── Dynamic MNQ contract cap ──────────────────────────────────
        target_leverage = 12.0
        try:
            from ib_async import ContFuture
            mnq_cont = ContFuture("MNQ", "CME")
            await session.ib.qualifyContractsAsync(mnq_cont)
            bars = await session.ib.reqHistoricalDataAsync(
                mnq_cont, "", "1 D", "1 day", "TRADES", False,
            )
            mnq_price = bars[-1].close if bars else 21000.0
        except Exception:
            mnq_price = 21000.0
            logger.warning("MNQ price fetch failed, using default %.0f", mnq_price)

        mnq_notional = mnq_price * 2.0  # MNQ point_value
        max_family_contracts = max(2, int(equity * target_leverage / mnq_notional))
        self._base_max_family_contracts = max_family_contracts
        logger.info(
            "Dynamic MNQ cap: %d contracts (equity=$%.0f, %.0fx leverage, MNQ=%.0f)",
            max_family_contracts, equity, target_leverage, mnq_price,
        )

        # ── Strategy descriptors ─────────────────────────────────────
        descriptors = self._build_strategy_descriptors()

        # Pre-compute all strategy IDs for family-scoped portfolio rules
        all_strategy_ids = tuple(d["strategy_id"] for d in descriptors)

        for desc in descriptors:
            sid = desc["strategy_id"]
            self._strategy_ids.append(sid)

            # Per-strategy adapter (same session, own adapter instance)
            adapter = IBKRExecutionAdapter(
                session=session,
                contract_factory=contract_factory,
                account=ibkr_config.profile.account_id,
            )

            # Resolve allocated NAV
            alloc = allocs.get(sid)
            allocated_nav = alloc.allocated_nav if alloc else equity
            if alloc:
                logger.info(
                    "Capital allocation: %s -> $%.2f (%.1f%% of $%.2f)",
                    sid, allocated_nav, alloc.capital_pct, equity,
                )
            else:
                logger.warning(
                    "Strategy %s not in unified config, using full equity", sid,
                )

            # Risk
            unit_risk = RiskCalculator.compute_unit_risk_dollars(
                nav=allocated_nav, unit_risk_pct=desc["base_risk_pct"],
            )
            _live_equity = [allocated_nav]

            portfolio_rules = PortfolioRulesConfig(
                initial_equity=allocated_nav,
                directional_cap_R=3.5,
                directional_cap_long_R=3.5,
                directional_cap_short_R=5.0,
                max_family_contracts_mnq_eq=max_family_contracts,
                family_strategy_ids=all_strategy_ids,
                symbol_collision_action="none",
                nqdtc_direction_filter_enabled=False,
                strategy_priorities=(
                    ("VdubusNQ_v4", 0),
                    ("NQDTC_v2.1", 1),
                    ("DownturnDominator_v1", 1),
                    ("AKC_Helix_v40", 2),
                ),
                priority_headroom_R=1.0,
                priority_reserve_threshold=1,
            )

            if self._base_portfolio_rules is None:
                self._base_portfolio_rules = portfolio_rules

            # Build per-strategy OMS
            oms = await build_oms_service(
                adapter=adapter,
                strategy_id=sid,
                unit_risk_dollars=unit_risk,
                daily_stop_R=desc["daily_stop_R"],
                heat_cap_R=desc["heat_cap_R"],
                portfolio_daily_stop_R=desc["portfolio_daily_stop_R"],
                db_pool=db_pool,
                portfolio_rules_config=portfolio_rules,
                get_current_equity=lambda eq=_live_equity: eq[0],
                paper_equity_pool=paper_equity_pool,
                paper_equity_scope=sid,
                paper_initial_equity=allocated_nav,
                live_equity=_live_equity if not paper_equity_pool else None,
                family_id=self.family_id,
                account_gate=account_gate,
            )
            await oms.start()
            logger.info("OMS started for %s", sid)
            self._oms_services.append(oms)
            self._portfolio_checkers.append(getattr(oms, '_portfolio_checker', None))

            # Instrumentation (non-fatal)
            instr = None
            try:
                from .instrumentation.src.bootstrap import InstrumentationManager
                instr = InstrumentationManager(
                    oms, sid, strategy_type=desc["instr_type"],
                    family_strategy_ids=list(all_strategy_ids),
                    get_regime_ctx=lambda: self._regime_ctx,
                    get_applied_config=lambda: self._portfolio_checkers[0]._cfg if self._portfolio_checkers and self._portfolio_checkers[0] else None,
                )
                await instr.start()
            except Exception as exc:
                logger.warning(
                    "Instrumentation init failed for %s (non-fatal): %s", sid, exc,
                )
            self._instrumentations.append(instr)

            # Build engine
            engine_cls = desc["engine_cls"]
            engine_kwargs: dict[str, Any] = dict(
                ib_session=session,
                oms_service=oms,
                instruments=desc["build_instruments"](),
                trade_recorder=desc.get("trade_recorder"),
                equity=equity,
                instrumentation=instr,
            )
            engine_kwargs.update(desc.get("engine_extra_kwargs", {}))

            engine = engine_cls(**engine_kwargs)
            await engine.start()
            logger.info("Engine started for %s", sid)
            self._engines.append(engine)

        # ── Heartbeat background task ──────────────────────────────────
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            "MomentumFamilyCoordinator started %d strategies", len(self._engines),
        )

    async def stop(self) -> None:
        """Stop engines, instrumentation, and OMS instances in reverse order."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        for i in reversed(range(len(self._engines))):
            sid = self._strategy_ids[i]

            # Stop engine
            try:
                await self._engines[i].stop()
                logger.info("Engine stopped for %s", sid)
            except Exception as exc:
                logger.error("Error stopping engine %s: %s", sid, exc)

            # Stop instrumentation
            try:
                instr = self._instrumentations[i]
                if instr is not None:
                    await instr.stop()
            except Exception as exc:
                logger.warning("Error stopping instrumentation %s: %s", sid, exc)

            # Stop OMS
            try:
                await self._oms_services[i].stop()
                logger.info("OMS stopped for %s", sid)
            except Exception as exc:
                logger.error("Error stopping OMS %s: %s", sid, exc)

        self._engines.clear()
        self._oms_services.clear()
        self._instrumentations.clear()
        self._strategy_ids.clear()
        logger.info("MomentumFamilyCoordinator stopped")

    def health_status(self) -> dict[str, Any]:
        """Return health of all four momentum engines."""
        result: dict[str, Any] = {"family": self.family_id, "strategies": {}}
        for i, engine in enumerate(self._engines):
            sid = self._strategy_ids[i]
            try:
                result["strategies"][sid] = engine.health_status()
            except Exception as exc:
                result["strategies"][sid] = {"error": str(exc)}
        return result

    def apply_regime(self, ctx: "RegimeContext") -> None:
        """Apply regime context to all momentum portfolio rules."""
        import dataclasses
        from regime.integration import build_momentum_rules

        if self._base_portfolio_rules is None:
            logger.warning("apply_regime called before start() — skipping")
            return

        prev_regime = getattr(self._regime_ctx, "regime", None)
        self._regime_ctx = ctx
        new_rules = build_momentum_rules(ctx, self._base_portfolio_rules, self._base_max_family_contracts)

        for checker in self._portfolio_checkers:
            if checker is not None:
                checker.update_config(dataclasses.replace(
                    new_rules, initial_equity=checker._cfg.initial_equity,
                ))

        r = new_rules
        changed = f" (was {prev_regime})" if prev_regime and prev_regime != ctx.regime else ""
        logger.info("Momentum regime applied: %s%s (cap=%.1fR, long=%.1fR, short=%.1fR, "
                    "risk=%.2fx, contracts=%d, oppose=%.1f, disabled=%s)",
                    ctx.regime, changed, r.directional_cap_R, r.directional_cap_long_R,
                    r.directional_cap_short_R, r.regime_unit_risk_mult,
                    r.max_family_contracts_mnq_eq, r.nqdtc_oppose_size_mult,
                    r.disabled_strategies or "none")

    # ── heartbeat ──────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        heartbeat = getattr(self._ctx, "heartbeat", None)
        if heartbeat is None:
            return
        session = self._ctx.session
        while True:
            try:
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                return
            for i, sid in enumerate(self._strategy_ids):
                try:
                    rs = getattr(self._oms_services[i], "_portfolio_risk_state", None)
                    await heartbeat.strategy_heartbeat(
                        strategy_id=sid,
                        heat_r=Decimal(str(rs.open_risk_R)) if rs else Decimal("0"),
                        daily_pnl_r=Decimal(str(rs.daily_realized_R)) if rs else Decimal("0"),
                        mode="HALTED" if (rs and rs.halted) else "RUNNING",
                    )
                except Exception:
                    logger.debug("Heartbeat failed for %s", sid, exc_info=True)
            try:
                connected = session.ib.isConnected() if session else False
                await heartbeat.adapter_heartbeat(self.family_id, connected=connected)
            except Exception:
                logger.debug("Adapter heartbeat failed", exc_info=True)

    # ── internal ─────────────────────────────────────────────────────

    def _build_strategy_descriptors(self) -> list[dict[str, Any]]:
        """Return per-strategy wiring descriptors.

        Risk params for Helix come from its config module.
        NQDTC and VdubusNQ hardcode their OMS risk params (matching main.py).
        """
        ctx = self._ctx

        # Trade recorder from context
        trade_recorder = getattr(ctx, "trade_recorder", None)
        if trade_recorder is None:
            trade_recorder = getattr(ctx.instrumentation, "trade_recorder", None)

        # ── Helix v4.0 ──────────────────────────────────────────────
        from strategies.momentum.helix_v40.config import (
            STRATEGY_ID as HELIX_ID,
            BASE_RISK_PCT as HELIX_RISK_PCT,
            DAILY_STOP_R as HELIX_DAILY_STOP_R,
            HEAT_CAP_R as HELIX_HEAT_CAP_R,
            PORTFOLIO_DAILY_STOP_R as HELIX_PORTFOLIO_DAILY_STOP_R,
            build_instruments as helix_build_instruments,
        )
        from strategies.momentum.helix_v40.engine import Helix4Engine

        # ── NQDTC v2.1 ──────────────────────────────────────────────
        from strategies.momentum.nqdtc.config import (
            STRATEGY_ID as NQDTC_ID,
            RISK_PCT as NQDTC_RISK_PCT,
            build_instruments as nqdtc_build_instruments,
        )
        from strategies.momentum.nqdtc.engine import NQDTCEngine

        # ── VdubusNQ v4 ─────────────────────────────────────────────
        from strategies.momentum.vdub.config import (
            STRATEGY_ID as VDUB_ID,
            BASE_RISK_PCT as VDUB_RISK_PCT,
            build_instruments as vdub_build_instruments,
        )
        from strategies.momentum.vdub.engine import VdubNQv4Engine

        # ── Downturn Dominator v1 ──────────────────────────────────
        from strategies.momentum.downturn.config import (
            STRATEGY_ID as DOWNTURN_ID,
            BASE_RISK_PCT as DOWNTURN_RISK_PCT,
            DAILY_STOP_R as DOWNTURN_DAILY_STOP_R,
            HEAT_CAP_R as DOWNTURN_HEAT_CAP_R,
            PORTFOLIO_DAILY_STOP_R as DOWNTURN_PORTFOLIO_DAILY_STOP_R,
            build_instruments as downturn_build_instruments,
        )
        from strategies.momentum.downturn.engine import DownturnEngine

        return [
            # ── AKC_Helix_v40 ───────────────────────────────────────
            {
                "strategy_id": HELIX_ID,
                "base_risk_pct": HELIX_RISK_PCT,
                "daily_stop_R": HELIX_DAILY_STOP_R,
                "heat_cap_R": HELIX_HEAT_CAP_R,
                "portfolio_daily_stop_R": HELIX_PORTFOLIO_DAILY_STOP_R,
                "build_instruments": helix_build_instruments,
                "engine_cls": Helix4Engine,
                "instr_type": "helix",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {},
            },
            # ── NQDTC_v2.1 ─────────────────────────────────────────
            {
                "strategy_id": NQDTC_ID,
                "base_risk_pct": NQDTC_RISK_PCT,
                "daily_stop_R": 2.5,
                "heat_cap_R": 3.5,
                "portfolio_daily_stop_R": 1.5,
                "build_instruments": nqdtc_build_instruments,
                "engine_cls": NQDTCEngine,
                "instr_type": "nqdtc",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {
                    "state_dir": Path("/app/state"),
                },
            },
            # ── VdubusNQ_v4 ────────────────────────────────────────
            {
                "strategy_id": VDUB_ID,
                "base_risk_pct": VDUB_RISK_PCT,
                "daily_stop_R": 2.5,
                "heat_cap_R": 3.5,
                "portfolio_daily_stop_R": 1.5,
                "build_instruments": vdub_build_instruments,
                "engine_cls": VdubNQv4Engine,
                "instr_type": "vdubus",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {},
            },
            # ── DownturnDominator_v1 ──────────────────────────────
            {
                "strategy_id": DOWNTURN_ID,
                "base_risk_pct": DOWNTURN_RISK_PCT,
                "daily_stop_R": DOWNTURN_DAILY_STOP_R,
                "heat_cap_R": DOWNTURN_HEAT_CAP_R,
                "portfolio_daily_stop_R": DOWNTURN_PORTFOLIO_DAILY_STOP_R,
                "build_instruments": downturn_build_instruments,
                "engine_cls": DownturnEngine,
                "instr_type": "downturn",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {
                    "state_dir": Path("/app/state"),
                },
            },
        ]
