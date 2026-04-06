"""Regime-to-config mapping tables and builder functions.

All policy for how macro regimes translate into portfolio rules, strategy
profiles, and overlay weights is centralized here -- not scattered across
coordinators.
"""
from __future__ import annotations

import dataclasses
import logging

from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from regime.context import RegimeContext

logger = logging.getLogger(__name__)

_VALID_REGIMES = frozenset({"G", "R", "S", "D"})


def _validated_regime(regime: str) -> str:
    """Return regime if valid, else fall back to 'G' with a warning."""
    if regime in _VALID_REGIMES:
        return regime
    logger.warning("Unknown regime %r, falling back to Recovery (G)", regime)
    return "G"

# ── Tier 1: Portfolio rules per regime per family ──────────────────────

STOCK_RULES: dict[str, dict] = {
    "G": {"directional_cap_R": 8.0,  "regime_unit_risk_mult": 1.0, "priority_headroom_R": 3.0, "symbol_collision_action": "half_size"},
    "R": {"directional_cap_R": 6.0,  "regime_unit_risk_mult": 0.9, "priority_headroom_R": 3.0, "symbol_collision_action": "half_size"},
    "S": {"directional_cap_R": 5.0,  "regime_unit_risk_mult": 0.7, "priority_headroom_R": 2.0, "symbol_collision_action": "block"},
    "D": {"directional_cap_R": 4.0,  "regime_unit_risk_mult": 0.5, "priority_headroom_R": 1.5, "symbol_collision_action": "block"},
}

MOMENTUM_RULES: dict[str, dict] = {
    "G": {"directional_cap_R": 3.5, "directional_cap_long_R": 3.5, "directional_cap_short_R": 0.0,
           "regime_unit_risk_mult": 1.0, "nqdtc_oppose_size_mult": 0.0, "nqdtc_direction_filter_enabled": True,
           "max_contracts_scale": 1.0},
    "R": {"directional_cap_R": 3.0, "directional_cap_long_R": 3.0, "directional_cap_short_R": 1.0,
           "regime_unit_risk_mult": 0.9, "nqdtc_oppose_size_mult": 0.0, "nqdtc_direction_filter_enabled": True,
           "max_contracts_scale": 0.9},
    "S": {"directional_cap_R": 2.5, "directional_cap_long_R": 2.0, "directional_cap_short_R": 2.0,
           "regime_unit_risk_mult": 0.7, "nqdtc_oppose_size_mult": 0.5, "nqdtc_direction_filter_enabled": True,
           "max_contracts_scale": 0.7},
    "D": {"directional_cap_R": 2.0, "directional_cap_long_R": 1.5, "directional_cap_short_R": 2.5,
           "regime_unit_risk_mult": 0.5, "nqdtc_oppose_size_mult": 0.5, "nqdtc_direction_filter_enabled": True,
           "max_contracts_scale": 0.5},
}

SWING_RULES: dict[str, dict] = {
    "G": {"directional_cap_R": 6.0, "regime_unit_risk_mult": 1.0},
    "R": {"directional_cap_R": 5.0, "regime_unit_risk_mult": 0.9},
    "S": {"directional_cap_R": 4.0, "regime_unit_risk_mult": 0.8},
    "D": {"directional_cap_R": 3.0, "regime_unit_risk_mult": 0.6},
}

# ── DD tiers per regime (all families) ─────────────────────────────────

DD_TIERS: dict[str, tuple[tuple[float, float], ...]] = {
    "G": ((0.08, 1.0), (0.12, 0.50), (0.15, 0.25), (1.00, 0.00)),
    "R": ((0.08, 1.0), (0.12, 0.50), (0.15, 0.25), (1.00, 0.00)),
    "S": ((0.07, 1.0), (0.11, 0.50), (0.14, 0.25), (1.00, 0.00)),
    "D": ((0.06, 1.0), (0.10, 0.50), (0.13, 0.25), (1.00, 0.00)),
}

# ── Overlay weights (swing only) ──────────────────────────────────────

OVERLAY_WEIGHTS: dict[str, dict[str, float]] = {
    "G": {"QQQ": 0.60, "GLD": 0.40},
    "R": {"QQQ": 0.50, "GLD": 0.50},
    "S": {"QQQ": 0.20, "GLD": 0.80},
    "D": {"QQQ": 0.00, "GLD": 1.00},
}

# ── Tier 2: Per-regime strategy profiles ──────────────────────────────

STOCK_PROFILES: dict[str, dict] = {
    "G": {"alcb_max_positions": 8, "iaric_pb_max_positions": 8, "disabled": frozenset()},
    "R": {"alcb_max_positions": 6, "iaric_pb_max_positions": 5, "disabled": frozenset()},
    "S": {"alcb_max_positions": 4, "iaric_pb_max_positions": 3, "disabled": frozenset({"US_ORB_v1"})},
    "D": {"alcb_max_positions": 3, "iaric_pb_max_positions": 2, "disabled": frozenset({"US_ORB_v1"})},
}

MOMENTUM_PROFILES: dict[str, dict] = {
    "G": {"disabled": frozenset({"DownturnDominator_v1"})},
    "R": {"disabled": frozenset({"DownturnDominator_v1"})},
    "S": {"disabled": frozenset()},
    "D": {"disabled": frozenset()},
}


# ── Builder functions ──────────────────────────────────────────────────

def build_stock_rules(ctx: RegimeContext, base_cfg: PortfolioRulesConfig) -> PortfolioRulesConfig:
    """Return new PortfolioRulesConfig with regime-adjusted values for stock family."""
    regime = _validated_regime(ctx.regime)
    r = STOCK_RULES[regime]
    profile = STOCK_PROFILES[regime]
    return dataclasses.replace(
        base_cfg,
        directional_cap_R=r["directional_cap_R"],
        regime_unit_risk_mult=r["regime_unit_risk_mult"],
        priority_headroom_R=r["priority_headroom_R"],
        symbol_collision_action=r["symbol_collision_action"],
        dd_tiers=DD_TIERS[regime],
        disabled_strategies=profile["disabled"],
    )


def build_momentum_rules(
    ctx: RegimeContext,
    base_cfg: PortfolioRulesConfig,
    base_max_contracts: int,
) -> PortfolioRulesConfig:
    """Return new PortfolioRulesConfig with regime-adjusted values for momentum family."""
    regime = _validated_regime(ctx.regime)
    r = MOMENTUM_RULES[regime]
    profile = MOMENTUM_PROFILES[regime]
    return dataclasses.replace(
        base_cfg,
        directional_cap_R=r["directional_cap_R"],
        directional_cap_long_R=r["directional_cap_long_R"],
        directional_cap_short_R=r["directional_cap_short_R"],
        nqdtc_direction_filter_enabled=r["nqdtc_direction_filter_enabled"],
        nqdtc_oppose_size_mult=r["nqdtc_oppose_size_mult"],
        max_family_contracts_mnq_eq=int(base_max_contracts * r["max_contracts_scale"]),
        regime_unit_risk_mult=r["regime_unit_risk_mult"],
        dd_tiers=DD_TIERS[regime],
        disabled_strategies=profile["disabled"],
    )


def build_swing_rules(ctx: RegimeContext, base_cfg: PortfolioRulesConfig) -> PortfolioRulesConfig:
    """Return new PortfolioRulesConfig with regime-adjusted values for swing family."""
    regime = _validated_regime(ctx.regime)
    r = SWING_RULES[regime]
    return dataclasses.replace(
        base_cfg,
        directional_cap_R=r["directional_cap_R"],
        regime_unit_risk_mult=r["regime_unit_risk_mult"],
        dd_tiers=DD_TIERS[regime],
    )
