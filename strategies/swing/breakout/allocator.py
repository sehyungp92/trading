"""Multi-Asset Swing Breakout v3.3-ETF — sizing, quality, risk allocation.

Quality multiplier, expiry, risk regime, correlation, share sizing.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import (
    ADD_RISK_LOWVOLH_MULT,
    ADD_RISK_MULT,
    CAMPAIGN_RISK_BUDGET_MULT,
    CORR_MULT_ALIGNED,
    CORR_MULT_OTHER,
    CORR_THRESHOLD,
    CORR_LOOKBACK_BARS,
    EXPIRY_MULT_FLOOR,
    MAX_ADDS_PER_CAMPAIGN,
    QUALITY_MULT_MAX,
    QUALITY_MULT_MIN,
    REGIME_MULT_ALIGNED,
    REGIME_MULT_CAUTION,
    REGIME_MULT_NEUTRAL,
    RISK_REGIME_MAX,
    RISK_REGIME_MIN,
    SQUEEZE_MULT_GOOD,
    SQUEEZE_MULT_LOOSE,
    SQUEEZE_MULT_NEUTRAL,
    SYMBOL_CONFIGS,
)
from .models import (
    DailyContext,
    Direction,
    PositionState,
    Regime4H,
    SymbolCampaign,
    TradeRegime,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk regime adjustment (spec §14.2)
# ---------------------------------------------------------------------------

def compute_risk_regime_adj(
    base_risk_pct: float,
    risk_regime: float,
) -> float:
    """base_risk_adj = base_risk * clamp(1.0/risk_regime, 0.75, 1.05)."""
    if risk_regime <= 0:
        return base_risk_pct
    adj = max(RISK_REGIME_MIN, min(RISK_REGIME_MAX, 1.0 / risk_regime))
    return base_risk_pct * adj


# ---------------------------------------------------------------------------
# Quality multiplier components (spec §14.3)
# ---------------------------------------------------------------------------

def compute_regime_mult(direction: Direction, regime_4h: Regime4H) -> float:
    """Regime multiplier: Aligned 1.00 / Neutral 0.65 / Caution 0.40."""
    aligned = (
        (direction == Direction.LONG and regime_4h == Regime4H.BULL_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BEAR_TREND)
    )
    opposes = (
        (direction == Direction.LONG and regime_4h == Regime4H.BEAR_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BULL_TREND)
    )
    if aligned:
        return REGIME_MULT_ALIGNED
    if opposes:
        return REGIME_MULT_CAUTION
    return REGIME_MULT_NEUTRAL


def compute_disp_mult(
    disp: float,
    disp_hist: list[float],
) -> float:
    """disp_mult = 0.70 + 0.30 * disp_norm (T70/T90 from past-only history).

    disp_norm = (disp - T70) / (T90 - T70), clamped [0, 1].
    """
    if len(disp_hist) < 20:
        return 0.85  # neutral default
    arr = np.array(disp_hist)
    t70 = float(np.quantile(arr, 0.70))
    t90 = float(np.quantile(arr, 0.90))
    if t90 <= t70:
        return 0.85
    disp_norm = max(0.0, min(1.0, (disp - t70) / (t90 - t70)))
    return 0.70 + 0.30 * disp_norm


def compute_squeeze_mult(sq_good: bool, sq_loose: bool) -> float:
    """Squeeze multiplier: good 1.05, neutral 1.00, loose 0.85."""
    if sq_good:
        return SQUEEZE_MULT_GOOD
    if sq_loose:
        return SQUEEZE_MULT_LOOSE
    return SQUEEZE_MULT_NEUTRAL


def compute_corr_mult(
    direction: Direction,
    symbol: str,
    positions: dict[str, PositionState],
    correlation_map: dict[tuple[str, str], float],
    regime_at_entry: str | None = None,
) -> float:
    """Correlation multiplier (spec §18.1).

    If any same-direction open peer corr > 0.70:
        0.85 if both aligned-at-entry, else 0.70.
    """
    for sym, pos in positions.items():
        if pos.qty <= 0 or pos.direction != direction:
            continue
        pair = tuple(sorted([symbol, sym]))
        corr = correlation_map.get(pair, 0.0)
        if corr > CORR_THRESHOLD:
            # Check if both positions were aligned at entry
            new_aligned = _is_aligned_at_entry(direction, regime_at_entry)
            existing_aligned = _is_aligned_at_entry(pos.direction, pos.regime_at_entry)
            if new_aligned and existing_aligned:
                return CORR_MULT_ALIGNED
            return CORR_MULT_OTHER
    return 1.0


def _is_aligned_at_entry(direction: Direction, regime_at_entry: str | None) -> bool:
    """Check if direction was aligned with the 4H regime stored at entry."""
    if regime_at_entry is None:
        return False
    if direction == Direction.LONG and regime_at_entry == Regime4H.BULL_TREND.value:
        return True
    if direction == Direction.SHORT and regime_at_entry == Regime4H.BEAR_TREND.value:
        return True
    return False


# ---------------------------------------------------------------------------
# Composite quality multiplier (spec §14.3)
# ---------------------------------------------------------------------------

def compute_quality_mult(
    direction: Direction,
    regime_4h: Regime4H,
    disp: float,
    disp_hist: list[float],
    sq_good: bool,
    sq_loose: bool,
    symbol: str,
    positions: dict[str, PositionState],
    correlation_map: dict[tuple[str, str], float],
    score_total: int,
    regime_at_entry: str | None = None,
) -> float:
    """quality_mult = clamp(regime * disp * squeeze * corr, 0.25, 1.0) * score_adj."""
    regime_m = compute_regime_mult(direction, regime_4h)
    disp_m = compute_disp_mult(disp, disp_hist)
    squeeze_m = compute_squeeze_mult(sq_good, sq_loose)
    corr_m = compute_corr_mult(direction, symbol, positions, correlation_map,
                                regime_at_entry=regime_at_entry or regime_4h.value)

    raw = regime_m * disp_m * squeeze_m * corr_m
    clamped = max(QUALITY_MULT_MIN, min(QUALITY_MULT_MAX, raw))

    # Score adjustment (spec §9.3)
    score_adj = max(0.85, min(1.15, 1.0 + 0.05 * score_total))
    return clamped * score_adj


# ---------------------------------------------------------------------------
# Final risk dollars (spec §14.5)
# ---------------------------------------------------------------------------

def compute_final_risk(
    equity: float,
    base_risk_pct: float,
    risk_regime: float,
    quality_mult: float,
    expiry_mult: float,
    cb_mult: float = 1.0,
) -> float:
    """Compute final_risk_dollars = equity * final_risk_pct.

    final_risk_pct = base_risk_adj * quality_mult * expiry_mult
    Clamped between 0.20*base_risk_adj and base_risk_adj.
    """
    base_adj = compute_risk_regime_adj(base_risk_pct, risk_regime)
    raw_pct = base_adj * quality_mult * expiry_mult * cb_mult
    floor_pct = 0.20 * base_adj
    capped_pct = max(floor_pct, min(base_adj, raw_pct))
    return equity * capped_pct


# ---------------------------------------------------------------------------
# Share sizing (spec §14.6)
# ---------------------------------------------------------------------------

def compute_shares(
    final_risk_dollars: float,
    entry_price: float,
    stop_price: float,
    fee_bps_est: float = 0.0,
) -> int:
    """Risk-based share sizing.

    shares = floor(final_risk_dollars / (risk_per_share + cost_buffer))
    """
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0
    cost_buffer = fee_bps_est * entry_price  # per-share friction estimate
    denominator = risk_per_share + cost_buffer
    if denominator <= 0:
        return 0
    return max(1, int(final_risk_dollars // denominator))


# ---------------------------------------------------------------------------
# Add sizing (spec §13.3)
# ---------------------------------------------------------------------------

def compute_add_risk_dollars(
    initial_risk_dollars: float,
    rvol_h: float,
) -> float:
    """Add risk = 0.5× initial risk, reduced to 0.7× that if RVOL_H < 0.8."""
    base = ADD_RISK_MULT * initial_risk_dollars
    if rvol_h < 0.8:
        base *= ADD_RISK_LOWVOLH_MULT
    return base


def add_budget_ok(
    campaign: SymbolCampaign,
    initial_risk_dollars: float,
    add_risk_dollars: float,
) -> bool:
    """Check add count and campaign risk budget."""
    if campaign.add_count >= MAX_ADDS_PER_CAMPAIGN:
        return False
    if campaign.campaign_risk_used + add_risk_dollars > CAMPAIGN_RISK_BUDGET_MULT * initial_risk_dollars:
        return False
    return True


# ---------------------------------------------------------------------------
# Micro guard (spec §14.4)
# ---------------------------------------------------------------------------

def micro_guard_ok(
    expiry_mult: float,
    disp_mult: float,
    trade_regime: TradeRegime,
    final_risk_pct: float,
    base_risk_adj: float,
) -> bool:
    """Skip if floored risk + expiry_mult < 0.60 + caution tier (unless disp_mult >= 0.85)."""
    if final_risk_pct <= 0.20 * base_risk_adj:
        # Risk was floored
        if expiry_mult < 0.60 and trade_regime == TradeRegime.CAUTION:
            return disp_mult >= 0.85
    return True
