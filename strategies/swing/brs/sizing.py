"""BRS position sizing -- per-symbol risk with quality/vol adjustments.

Spec 16.1: base_risk * risk_regime_adj * quality_mult * VolFactor
"""
from __future__ import annotations

import math

from .models import (
    BRSRegime,
    DailyContext,
    Direction,
    EntrySignal,
)
from .config import BRSConfig, BRSSymbolConfig


def compute_position_size(
    signal: EntrySignal,
    equity: float,
    sym_cfg: BRSSymbolConfig,
    daily_ctx: DailyContext,
    cfg: BRSConfig,
    current_heat_r: float = 0.0,
    concurrent_positions: int = 0,
) -> int:
    """Compute position size in shares.

    Returns:
        Number of shares (0 if blocked by portfolio limits)
    """
    if signal.risk_per_unit <= 0 or equity <= 0:
        return 0

    if concurrent_positions >= cfg.max_concurrent:
        return 0

    heat_cap = cfg.heat_cap_r
    if daily_ctx.vol.extreme_vol:
        heat_cap = cfg.crisis_heat_cap_r

    if current_heat_r >= heat_cap:
        return 0

    base_risk_pct = sym_cfg.base_risk_pct
    risk_dollars = equity * base_risk_pct

    risk_dollars *= daily_ctx.vol.base_risk_adj

    quality_mult = signal.quality_score
    risk_dollars *= quality_mult

    regime_mult_map = {
        BRSRegime.BEAR_TREND: cfg.size_mult_bear_trend,
        BRSRegime.BEAR_STRONG: cfg.size_mult_bear_strong,
        BRSRegime.RANGE_CHOP: cfg.size_mult_range_chop,
        BRSRegime.BEAR_FORMING: cfg.size_mult_bear_forming,
    }
    risk_dollars *= regime_mult_map.get(daily_ctx.regime, 1.0)

    if signal.direction == Direction.LONG:
        if daily_ctx.regime in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND, BRSRegime.BEAR_FORMING):
            risk_dollars *= 0.50

    if signal.direction == Direction.SHORT and daily_ctx.regime == BRSRegime.BEAR_FORMING:
        risk_dollars *= 0.65

    remaining_heat = heat_cap - current_heat_r
    if remaining_heat <= 0:
        return 0

    max_risk_for_heat = remaining_heat * equity * base_risk_pct
    risk_dollars = min(risk_dollars, max_risk_for_heat)

    cost_buffer = cfg.commission_per_share * 2  # round-trip
    effective_risk_per_share = signal.risk_per_unit + cost_buffer

    if effective_risk_per_share <= 0:
        return 0

    shares = int(math.floor(risk_dollars / effective_risk_per_share))
    return max(0, shares)
