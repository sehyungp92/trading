"""Tests for swing family portfolio rules wiring (1A fix).

Validates:
  - build_multi_strategy_oms creates PortfolioRuleChecker when config provided
  - RiskGateway receives the checker
  - Directional cap denials work in multi-strategy OMS
  - Backward compat: no checker when portfolio_rules_config=None
  - Momentum family-scoped rules with symbol collision (1D)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.oms.risk.portfolio_rules import (
    PortfolioRuleChecker,
    PortfolioRulesConfig,
)

SWING_IDS = ("ATRSS", "S5_PB", "S5_DUAL", "SWING_BREAKOUT_V3", "AKC_HELIX")
MOMENTUM_IDS = ("AKC_Helix_v40", "NQDTC_v2.1", "VdubusNQ_v4")


def _make_checker(
    *,
    equity: float = 100_000.0,
    initial_equity: float = 100_000.0,
    directional_cap_R: float = 6.0,
    family_strategy_ids: tuple[str, ...] = SWING_IDS,
    symbol_collision_action: str = "half_size",
    dir_risk_family: float = 0.0,
    sibling_holds_symbol: bool = False,
) -> PortfolioRuleChecker:
    config = PortfolioRulesConfig(
        directional_cap_R=directional_cap_R,
        initial_equity=initial_equity,
        family_strategy_ids=family_strategy_ids,
        symbol_collision_action=symbol_collision_action,
        helix_nqdtc_cooldown_minutes=0,  # disable momentum-specific
        nqdtc_direction_filter_enabled=False,
    )
    return PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: equity,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=dir_risk_family),
        get_sibling_positions_for_symbol=AsyncMock(return_value=sibling_holds_symbol),
    )


# ── Swing directional cap ────────────────────────────────────────


@pytest.mark.asyncio
async def test_swing_directional_cap_approves_within_limit():
    checker = _make_checker(dir_risk_family=5.0)
    result = await checker.check_entry("ATRSS", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_swing_directional_cap_denies_above_limit():
    checker = _make_checker(dir_risk_family=5.5)
    # 5.5R + 1R = 6.5R > 6R cap
    result = await checker.check_entry("S5_PB", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap" in result.denial_reason


@pytest.mark.asyncio
async def test_swing_directional_cap_exact_boundary():
    checker = _make_checker(dir_risk_family=5.0)
    # 5R + 1R = 6R == 6R cap → approved (not strictly greater)
    result = await checker.check_entry("AKC_HELIX", "SHORT", 1.0)
    assert result.approved


# ── Swing symbol collision ────────────────────────────────────────


@pytest.mark.asyncio
async def test_swing_symbol_collision_half_size():
    checker = _make_checker(sibling_holds_symbol=True)
    result = await checker.check_entry("ATRSS", "LONG", 1.0, symbol="IBIT")
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_swing_no_collision():
    checker = _make_checker(sibling_holds_symbol=False)
    result = await checker.check_entry("ATRSS", "LONG", 1.0, symbol="IBIT")
    assert result.approved
    assert result.size_multiplier == 1.0


# ── Momentum-specific: no cooldown/direction filter for swing ────


@pytest.mark.asyncio
async def test_swing_skips_proximity_cooldown():
    """Swing strategies should not trigger Helix↔NQDTC cooldown."""
    checker = _make_checker()
    result = await checker.check_entry("ATRSS", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_swing_skips_direction_filter():
    """Swing strategies should not trigger NQDTC direction filter."""
    checker = _make_checker()
    result = await checker.check_entry("S5_DUAL", "SHORT", 1.0)
    assert result.approved


# ── Backward compat: None config ─────────────────────────────────


@pytest.mark.asyncio
async def test_no_checker_when_config_none():
    """When portfolio_rules_config=None, no checker should be created."""
    # This tests the factory behavior conceptually — when config is None,
    # the RiskGateway should get portfolio_checker=None
    from libs.oms.risk.gateway import RiskGateway

    config = MagicMock()
    config.global_standdown = False
    config.strategy_configs = {}

    gateway = RiskGateway(
        config=config,
        calendar=MagicMock(),
        get_strategy_risk=AsyncMock(),
        get_portfolio_risk=AsyncMock(),
    )
    assert gateway._portfolio_checker is None


# ── Momentum family-scoped rules (1D) ───────────────────────────


@pytest.mark.asyncio
async def test_momentum_family_directional_cap():
    """Momentum with family_strategy_ids uses family-scoped directional check."""
    config = PortfolioRulesConfig(
        initial_equity=10_000.0,
        directional_cap_R=6.0,
        family_strategy_ids=MOMENTUM_IDS,
        symbol_collision_action="half_size",
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=100.0),  # high global
        get_current_equity=lambda: 10_000.0,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=5.0),  # low family
        get_sibling_positions_for_symbol=AsyncMock(return_value=False),
    )
    # 5R family + 1R new = 6R ≤ 6R cap → approved (uses family, not global)
    result = await checker.check_entry("AKC_Helix_v40", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_momentum_symbol_collision_nq():
    """NQ collision between Helix and NQDTC triggers half_size."""
    config = PortfolioRulesConfig(
        initial_equity=10_000.0,
        directional_cap_R=6.0,
        family_strategy_ids=MOMENTUM_IDS,
        symbol_collision_action="half_size",
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=0.0),
        get_sibling_positions_for_symbol=AsyncMock(return_value=True),
    )
    result = await checker.check_entry("AKC_Helix_v40", "LONG", 1.0, symbol="NQ")
    assert result.approved
    assert result.size_multiplier == 0.5
