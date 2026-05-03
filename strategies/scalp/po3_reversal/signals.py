from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import (
    PREFERRED_SMT_STRENGTH,
    SCORE_THRESHOLD_A,
    SCORE_THRESHOLD_A_PLUS,
    SCORE_THRESHOLD_B,
    SetupTier,
)


WEIGHTS: dict[str, float] = {
    "h4_location": 1.0,
    "daily_h4_alignment": 1.0,
    "liquidity_sweep": 1.0,
    "smt_present": 1.25,
    "smt_strength": 0.5,
    "ifvg_close_through": 1.0,
    "ifvg_retest_rejection": 0.5,
    "displacement_quality": 0.75,
    "asian_range_swept": 0.5,
    "spread_clean": 0.5,
    "atr_normal": 0.5,
    "h1_target_clean": 0.75,
}


@dataclass(frozen=True, slots=True)
class Po3SignalScore:
    total: float
    tier: SetupTier
    components: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def score_components(
    *,
    h4_location: bool,
    daily_h4_alignment: bool,
    liquidity_sweep: bool,
    smt_present: bool,
    smt_strength: float,
    ifvg_close_through: bool,
    ifvg_retest_rejection: bool,
    displacement_body_pct: float,
    asian_range_swept: bool = False,
    spread_clean: bool = True,
    atr_normal: bool = True,
    h1_target_clean: bool = True,
    tier_hint: SetupTier = SetupTier.A,
) -> Po3SignalScore:
    components = {
        "h4_location": 1.0 if h4_location else 0.0,
        "daily_h4_alignment": 1.0 if daily_h4_alignment else 0.0,
        "liquidity_sweep": 1.0 if liquidity_sweep else 0.0,
        "smt_present": 1.0 if smt_present else 0.0,
        "smt_strength": min(max(smt_strength / PREFERRED_SMT_STRENGTH, 0.0), 1.0),
        "ifvg_close_through": 1.0 if ifvg_close_through else 0.0,
        "ifvg_retest_rejection": 1.0 if ifvg_retest_rejection else 0.0,
        "displacement_quality": min(max((displacement_body_pct - 0.35) / 0.25, 0.0), 1.0),
        "asian_range_swept": 1.0 if asian_range_swept else 0.0,
        "spread_clean": 1.0 if spread_clean else 0.0,
        "atr_normal": 1.0 if atr_normal else 0.0,
        "h1_target_clean": 1.0 if h1_target_clean else 0.0,
    }
    total = sum(WEIGHTS[key] * components[key] for key in WEIGHTS)
    tier = _resolve_tier(total, tier_hint)
    threshold = threshold_for_tier(tier_hint)
    return Po3SignalScore(
        total=float(total),
        tier=tier,
        components=components,
        threshold=threshold,
        passed=total >= threshold,
        details={"weights": dict(WEIGHTS)},
    )


def threshold_for_tier(tier: SetupTier) -> float:
    if tier is SetupTier.A_PLUS:
        return SCORE_THRESHOLD_A_PLUS
    if tier is SetupTier.B:
        return SCORE_THRESHOLD_B
    if tier is SetupTier.A:
        return SCORE_THRESHOLD_A
    return float("inf")


def _resolve_tier(score: float, tier_hint: SetupTier) -> SetupTier:
    if score >= SCORE_THRESHOLD_A_PLUS and tier_hint in {SetupTier.A, SetupTier.A_PLUS}:
        return SetupTier.A_PLUS
    if tier_hint is SetupTier.B and score >= SCORE_THRESHOLD_B:
        return SetupTier.B
    if score >= SCORE_THRESHOLD_A:
        return SetupTier.A
    return SetupTier.NONE

