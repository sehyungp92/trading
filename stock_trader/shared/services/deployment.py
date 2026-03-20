"""Deployment-mode and capital-allocation helpers for live strategy runtimes."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEPLOY_MODE_ENV = "STOCK_TRADER_DEPLOY_MODE"
IARIC_ALLOCATION_ENV = "STOCK_TRADER_CAPITAL_ALLOCATION_IARIC_PCT"
US_ORB_ALLOCATION_ENV = "STOCK_TRADER_CAPITAL_ALLOCATION_US_ORB_PCT"
ALCB_ALLOCATION_ENV = "STOCK_TRADER_CAPITAL_ALLOCATION_ALCB_PCT"
PAPER_CAPITAL_ENV = "STOCK_TRADER_PAPER_CAPITAL"

_VALID_DEPLOY_MODES = ("both", "iaric", "us_orb", "alcb")
_STRATEGY_TO_MODE = {
    "IARIC_v1": "iaric",
    "US_ORB_v1": "us_orb",
    "ALCB_v1": "alcb",
}
_MODE_TO_STRATEGY = {mode: strategy for strategy, mode in _STRATEGY_TO_MODE.items()}


class DeploymentConfigError(ValueError):
    """Raised when deployment-mode or allocation environment is invalid."""


@dataclass(frozen=True)
class StrategyCapitalAllocation:
    """Resolved deployment-mode and NAV allocation for a single strategy."""

    strategy_id: str
    deploy_mode: str
    enabled_for_strategy: bool
    capital_fraction: float
    raw_nav: float
    allocated_nav: float

    @property
    def capital_pct(self) -> float:
        return self.capital_fraction * 100.0

    def assert_enabled(self) -> None:
        if self.enabled_for_strategy:
            return
        allowed_strategy = _MODE_TO_STRATEGY[self.deploy_mode]
        raise RuntimeError(
            f"{self.strategy_id} is disabled by {DEPLOY_MODE_ENV}={self.deploy_mode}. "
            f"Start only {allowed_strategy} or set {DEPLOY_MODE_ENV}=both."
        )

    def assert_positive_allocated_nav(self) -> None:
        if not math.isfinite(self.raw_nav) or self.raw_nav <= 0:
            raise RuntimeError(
                f"{self.strategy_id} requires a positive raw NAV; got {self.raw_nav!r}."
            )
        if not math.isfinite(self.allocated_nav) or self.allocated_nav <= 0:
            raise RuntimeError(
                f"{self.strategy_id} requires a positive allocated NAV; got {self.allocated_nav!r}."
            )


def _strategy_mode(strategy_id: str) -> str:
    try:
        return _STRATEGY_TO_MODE[strategy_id]
    except KeyError as exc:
        raise DeploymentConfigError(f"Unsupported strategy_id for deployment allocation: {strategy_id}") from exc


def _parse_deploy_mode() -> str:
    raw_mode = (os.environ.get(DEPLOY_MODE_ENV) or "both").strip().lower()
    if raw_mode not in _VALID_DEPLOY_MODES:
        valid = ", ".join(_VALID_DEPLOY_MODES)
        raise DeploymentConfigError(f"{DEPLOY_MODE_ENV} must be one of {valid}; got {raw_mode!r}")
    return raw_mode


def _parse_positive_pct(env_name: str, default: float) -> float:
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise DeploymentConfigError(f"{env_name} must be numeric; got {raw_value!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise DeploymentConfigError(f"{env_name} must be > 0; got {raw_value!r}")
    return value


def resolve_paper_nav(raw_nav: float, state_file: Path) -> float:
    """Apply paper-capital virtual NAV if configured, otherwise pass through.

    Formula: virtual_nav = paper_capital + (raw_nav - baseline)

    The baseline is persisted to *state_file* on first call so subsequent
    sessions track only the strategy's own P&L.
    """
    paper_capital_str = os.environ.get(PAPER_CAPITAL_ENV)
    if paper_capital_str is None or not paper_capital_str.strip():
        return raw_nav

    try:
        paper_capital = float(paper_capital_str)
    except ValueError:
        logger.warning("%s is not numeric (%r) — using raw NAV", PAPER_CAPITAL_ENV, paper_capital_str)
        return raw_nav

    if not math.isfinite(paper_capital) or paper_capital <= 0:
        logger.warning("%s must be > 0 (got %r) — using raw NAV", PAPER_CAPITAL_ENV, paper_capital_str)
        return raw_nav

    # Only apply in paper/dev mode
    from shared.oms.persistence.db_config import get_environment

    environment = get_environment()
    if environment == "live":
        logger.warning(
            "%s=%s is set but ignored in live mode — using full account NAV.",
            PAPER_CAPITAL_ENV,
            paper_capital_str,
        )
        return raw_nav

    # Load or create baseline
    baseline = _load_or_create_baseline(state_file, raw_nav, paper_capital)
    offset = raw_nav - baseline
    virtual_nav = paper_capital + offset

    logger.info(
        "Paper capital mode: %s=%.2f baseline=%.2f actual=%.2f offset=%+.2f virtual_nav=%.2f",
        PAPER_CAPITAL_ENV,
        paper_capital,
        baseline,
        raw_nav,
        offset,
        virtual_nav,
    )
    return virtual_nav


def _load_or_create_baseline(state_file: Path, raw_nav: float, paper_capital: float) -> float:
    """Load persisted baseline or create it from the current raw_nav."""
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text())
            baseline = float(data["baseline_net_liq"])
            stored_capital = float(data["paper_capital"])
            if stored_capital != paper_capital:
                logger.warning(
                    "Paper capital changed (%.2f → %.2f). Resetting baseline to current NAV %.2f",
                    stored_capital,
                    paper_capital,
                    raw_nav,
                )
            elif not math.isfinite(baseline):
                logger.warning("Invalid baseline %r in %s. Resetting.", baseline, state_file)
            else:
                return baseline
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Corrupt paper capital state file %s (%s). Resetting baseline.", state_file, exc)

    # Create or reset baseline
    state_file.parent.mkdir(parents=True, exist_ok=True)
    data = {"baseline_net_liq": raw_nav, "paper_capital": paper_capital}
    state_file.write_text(json.dumps(data))
    logger.info("Paper capital baseline recorded: %.2f (state file: %s)", raw_nav, state_file)
    return raw_nav


def resolve_strategy_capital_allocation(strategy_id: str, raw_nav: float) -> StrategyCapitalAllocation:
    """Resolve deploy mode and effective capital allocation for a strategy."""
    strategy_mode = _strategy_mode(strategy_id)
    deploy_mode = _parse_deploy_mode()
    nav_value = float(raw_nav)

    # --- Strategy-level split (IARIC / US_ORB / ALCB) ---
    if deploy_mode == "both":
        iaric_pct = _parse_positive_pct(IARIC_ALLOCATION_ENV, 100.0 / 3.0)
        us_orb_pct = _parse_positive_pct(US_ORB_ALLOCATION_ENV, 100.0 / 3.0)
        alcb_pct = _parse_positive_pct(ALCB_ALLOCATION_ENV, 100.0 / 3.0)
        total_pct = iaric_pct + us_orb_pct + alcb_pct
        if not math.isclose(total_pct, 100.0, rel_tol=0.0, abs_tol=1e-6):
            raise DeploymentConfigError(
                f"{IARIC_ALLOCATION_ENV} + {US_ORB_ALLOCATION_ENV} + {ALCB_ALLOCATION_ENV} must equal 100; got {total_pct:.6f}"
            )
        allocations = {
            "iaric": iaric_pct / 100.0,
            "us_orb": us_orb_pct / 100.0,
            "alcb": alcb_pct / 100.0,
        }
        capital_fraction = allocations[strategy_mode]
        enabled = True
    else:
        enabled = strategy_mode == deploy_mode
        capital_fraction = 1.0 if enabled else 0.0

    allocated_nav = nav_value * capital_fraction if enabled else 0.0
    return StrategyCapitalAllocation(
        strategy_id=strategy_id,
        deploy_mode=deploy_mode,
        enabled_for_strategy=enabled,
        capital_fraction=capital_fraction,
        raw_nav=nav_value,
        allocated_nav=allocated_nav,
    )
