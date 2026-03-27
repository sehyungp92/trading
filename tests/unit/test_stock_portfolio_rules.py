"""Tests for stock family portfolio rules integration.

Validates:
  - Drawdown tiers reduce stock sizing
  - Directional cap with family-scoped query
  - Symbol collision blocks/reduces when sibling holds same symbol
  - Momentum-specific checks are no-ops for stock strategy IDs
  - check_entry symbol param is backward-compatible (Optional, defaults to None)
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from libs.oms.risk.portfolio_rules import (
    PortfolioRuleChecker,
    PortfolioRulesConfig,
)

STOCK_IDS = ("IARIC_v1", "US_ORB_v1", "ALCB_v1")


def _make_checker(
    *,
    equity: float = 10_000.0,
    initial_equity: float = 10_000.0,
    directional_cap_R: float = 8.0,
    family_strategy_ids: tuple[str, ...] = STOCK_IDS,
    symbol_collision_action: str = "none",
    dir_risk_global: float = 0.0,
    dir_risk_family: float = 0.0,
    sibling_holds_symbol: bool = False,
) -> PortfolioRuleChecker:
    config = PortfolioRulesConfig(
        directional_cap_R=directional_cap_R,
        initial_equity=initial_equity,
        family_strategy_ids=family_strategy_ids,
        symbol_collision_action=symbol_collision_action,
    )
    return PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=dir_risk_global),
        get_current_equity=lambda: equity,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=dir_risk_family),
        get_sibling_positions_for_symbol=AsyncMock(return_value=sibling_holds_symbol),
    )


# ── Drawdown tiers ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drawdown_tier_full_size_when_no_drawdown():
    checker = _make_checker(equity=10_000.0, initial_equity=10_000.0)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved
    assert result.size_multiplier == 1.0


@pytest.mark.asyncio
async def test_drawdown_tier_half_size_at_10pct_dd():
    # 10% drawdown → falls in [0.08, 0.12) tier → 0.50 multiplier
    checker = _make_checker(equity=9_000.0, initial_equity=10_000.0)
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_drawdown_tier_quarter_size_at_13pct_dd():
    # 13% drawdown → falls in [0.12, 0.15) tier → 0.25 multiplier
    checker = _make_checker(equity=8_700.0, initial_equity=10_000.0)
    result = await checker.check_entry("US_ORB_v1", "LONG", 1.0)
    assert result.approved
    assert result.size_multiplier == 0.25


@pytest.mark.asyncio
async def test_drawdown_tier_halt_at_16pct_dd():
    # 16% drawdown → beyond 0.15 tier → halt (0.0)
    checker = _make_checker(equity=8_400.0, initial_equity=10_000.0)
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert not result.approved
    assert "drawdown_halt" in result.denial_reason


# ── Family-scoped directional cap ───────────────────────────────────


@pytest.mark.asyncio
async def test_directional_cap_uses_family_scoped_query():
    """When family_strategy_ids is set, directional cap uses the family-scoped callback."""
    checker = _make_checker(dir_risk_family=7.0, dir_risk_global=20.0)
    # 7R family + 1R new = 8R ≤ 8R cap → approved
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_directional_cap_blocks_when_family_exceeds():
    checker = _make_checker(dir_risk_family=7.5)
    # 7.5R family + 1R new = 8.5R > 8R cap → blocked
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap" in result.denial_reason


@pytest.mark.asyncio
async def test_directional_cap_disabled_when_zero():
    checker = _make_checker(directional_cap_R=0, dir_risk_family=100.0)
    result = await checker.check_entry("US_ORB_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_directional_cap_falls_back_to_global_without_family_ids():
    """Without family_strategy_ids, falls back to global directional risk query."""
    checker = _make_checker(
        family_strategy_ids=(),
        dir_risk_global=9.0,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap" in result.denial_reason


# ── Symbol collision ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_symbol_collision_block_mode():
    checker = _make_checker(
        symbol_collision_action="block",
        sibling_holds_symbol=True,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert not result.approved
    assert "symbol_collision" in result.denial_reason


@pytest.mark.asyncio
async def test_symbol_collision_half_size_mode():
    checker = _make_checker(
        symbol_collision_action="half_size",
        sibling_holds_symbol=True,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_symbol_collision_no_collision():
    checker = _make_checker(
        symbol_collision_action="block",
        sibling_holds_symbol=False,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == 1.0


@pytest.mark.asyncio
async def test_symbol_collision_ignored_when_action_none():
    checker = _make_checker(
        symbol_collision_action="none",
        sibling_holds_symbol=True,
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved


@pytest.mark.asyncio
async def test_symbol_collision_ignored_without_symbol():
    checker = _make_checker(
        symbol_collision_action="block",
        sibling_holds_symbol=True,
    )
    # No symbol passed → skip collision check
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


# ── Momentum checks are no-ops for stock IDs ───────────────────────


@pytest.mark.asyncio
async def test_proximity_cooldown_noop_for_stock():
    """Proximity cooldown only fires for Helix/NQDTC IDs — stock IDs pass through."""
    checker = _make_checker()
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_direction_filter_noop_for_stock():
    """Direction filter only fires for VdubusNQ — stock IDs pass through."""
    checker = _make_checker()
    result = await checker.check_entry("ALCB_v1", "SHORT", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_chop_throttle_noop_for_stock():
    """Chop throttle only fires for Helix — stock IDs pass through."""
    checker = _make_checker()
    result = await checker.check_entry("US_ORB_v1", "LONG", 1.0)
    assert result.approved


# ── Backward compatibility ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_entry_without_symbol_param():
    """check_entry works without symbol param (backward compat with momentum gateway)."""
    checker = _make_checker()
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_combined_drawdown_and_collision_multipliers():
    """Drawdown tier and symbol collision multipliers compound."""
    checker = _make_checker(
        equity=9_000.0,
        initial_equity=10_000.0,  # 10% DD → 0.5x
        symbol_collision_action="half_size",
        sibling_holds_symbol=True,  # → 0.5x
    )
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0, symbol="AAPL")
    assert result.approved
    assert result.size_multiplier == pytest.approx(0.25)  # 0.5 * 0.5


# ── Robustness / edge cases ───────────────────────────────────────


@pytest.mark.asyncio
async def test_symbol_collision_excludes_requesting_strategy():
    """Sibling query must exclude the requesting strategy from the lookup."""
    sibling_mock = AsyncMock(return_value=True)
    config = PortfolioRulesConfig(
        directional_cap_R=0,
        family_strategy_ids=STOCK_IDS,
        symbol_collision_action="block",
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_sibling_positions_for_symbol=sibling_mock,
    )
    await checker.check_entry("US_ORB_v1", "LONG", 1.0, symbol="TSLA")
    # Must be called with siblings only — US_ORB_v1 excluded
    sibling_mock.assert_awaited_once()
    called_ids = sibling_mock.call_args[0][0]
    assert "US_ORB_v1" not in called_ids
    assert set(called_ids) == {"IARIC_v1", "ALCB_v1"}


def test_invalid_collision_action_raises():
    """Invalid symbol_collision_action raises ValueError at config init."""
    with pytest.raises(ValueError, match="Invalid symbol_collision_action"):
        PortfolioRulesConfig(symbol_collision_action="quarter_size")


# ── Priority-based directional cap reservation ────────────────────


PRIORITY_KWARGS = dict(
    strategy_priorities=(("IARIC_v1", 0), ("ALCB_v1", 1), ("US_ORB_v1", 2)),
    priority_headroom_R=3.0,
    priority_reserve_threshold=1,
)


def _make_priority_checker(
    *,
    dir_risk_family: float = 0.0,
    **extra,
) -> PortfolioRuleChecker:
    kw = {**PRIORITY_KWARGS, **extra}
    config = PortfolioRulesConfig(
        directional_cap_R=8.0,
        initial_equity=10_000.0,
        family_strategy_ids=STOCK_IDS,
        symbol_collision_action="none",
        **kw,
    )
    return PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=dir_risk_family),
        get_sibling_positions_for_symbol=AsyncMock(return_value=False),
    )


@pytest.mark.asyncio
async def test_priority_disabled_when_headroom_zero():
    """headroom_R=0 → backward compatible, no priority enforcement."""
    checker = _make_priority_checker(
        dir_risk_family=6.0,
        priority_headroom_R=0.0,
    )
    # 6R + 1R = 7R < 8R cap, remaining=2R < 3R headroom, but headroom disabled
    result = await checker.check_entry("US_ORB_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_priority_iaric_passes_in_reserved_headroom():
    """Priority 0 (IARIC) passes even when remaining <= headroom_R."""
    checker = _make_priority_checker(dir_risk_family=6.0)
    # remaining = 8-6 = 2R <= 3R headroom, but IARIC priority 0 <= threshold 1
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_priority_alcb_passes_in_reserved_headroom():
    """Priority 1 (ALCB) passes even when remaining <= headroom_R."""
    checker = _make_priority_checker(dir_risk_family=6.0)
    # remaining = 2R <= 3R headroom, ALCB priority 1 <= threshold 1
    result = await checker.check_entry("ALCB_v1", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_priority_orb_blocked_in_reserved_headroom():
    """Priority 2 (ORB) blocked when remaining <= headroom_R but total < hard cap."""
    checker = _make_priority_checker(dir_risk_family=6.0)
    # remaining = 2R <= 3R headroom, ORB priority 2 > threshold 1 → blocked
    result = await checker.check_entry("US_ORB_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap_reserved" in result.denial_reason
    assert "priority 2" in result.denial_reason


@pytest.mark.asyncio
async def test_priority_hard_cap_blocks_all_regardless_of_priority():
    """All strategies blocked when total > hard cap, priority irrelevant."""
    checker = _make_priority_checker(dir_risk_family=7.5)
    # 7.5R + 1R = 8.5R > 8R → hard cap blocks even IARIC
    result = await checker.check_entry("IARIC_v1", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap:" in result.denial_reason
    assert "directional_cap_reserved" not in result.denial_reason


@pytest.mark.asyncio
async def test_priority_unknown_strategy_gets_default_99():
    """Strategy not in strategy_priorities gets default priority 99 (blocked when headroom active)."""
    checker = _make_priority_checker(dir_risk_family=6.0)
    result = await checker.check_entry("UNKNOWN_v1", "LONG", 1.0)
    assert not result.approved
    assert "priority 99" in result.denial_reason


@pytest.mark.asyncio
async def test_priority_orb_passes_when_headroom_sufficient():
    """ORB passes when remaining > headroom_R (plenty of capacity)."""
    checker = _make_priority_checker(dir_risk_family=3.0)
    # remaining = 8-3 = 5R > 3R headroom → reservation not triggered
    result = await checker.check_entry("US_ORB_v1", "LONG", 1.0)
    assert result.approved
