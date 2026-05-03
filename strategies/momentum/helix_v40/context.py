"""Shared contextual quality helpers for Helix v4.0."""
from __future__ import annotations

from .config import SessionBlock, Setup, SetupClass


def context_quality_score_for_features(
    setup_class: SetupClass,
    direction: int,
    block: SessionBlock,
    vol_pct: float | None,
    strong_trend: bool,
) -> int:
    score = 0

    if strong_trend:
        score += 1

    if block == SessionBlock.ETH_QUALITY_PM:
        score += 2

    if setup_class == SetupClass.T:
        score += 2

    if vol_pct is None:
        return score

    if setup_class == SetupClass.M and direction == 1 and vol_pct >= 80:
        score += 2
    if setup_class == SetupClass.M and direction == -1 and 50 <= vol_pct <= 80:
        score += 2
        if block == SessionBlock.RTH_PRIME2:
            score += 1
    if setup_class == SetupClass.M and direction == 1 and vol_pct < 20:
        score += 1
    if setup_class == SetupClass.T and direction == 1 and vol_pct < 20:
        score += 1
    if setup_class == SetupClass.F and direction == 1 and vol_pct < 20:
        score -= 2
    if setup_class == SetupClass.M and direction == -1 and block == SessionBlock.ETH_QUALITY_AM:
        score -= 1

    return score


def context_quality_score(
    setup: Setup,
    block: SessionBlock,
    vol_pct: float | None,
) -> int:
    return context_quality_score_for_features(
        setup.cls,
        setup.direction,
        block,
        vol_pct,
        setup.strong_trend,
    )


def meets_context_quality(
    min_score: int,
    actual_score: int,
) -> bool:
    return actual_score >= min_score
