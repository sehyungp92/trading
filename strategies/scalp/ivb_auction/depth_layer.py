from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DepthAssessment:
    available: bool = False
    confirmed: bool | None = None
    depth_imbalance_score: float = 0.0
    queue_depletion_score: float = 0.0
    veto_reason: str = ""


@dataclass(frozen=True, slots=True)
class DepthAdjustedSignal:
    allowed: bool
    size_multiplier: float
    reason: str


def apply_depth_confirmation(
    *,
    core_score: float,
    min_core_score: float,
    base_size_multiplier: float,
    assessment: DepthAssessment,
) -> DepthAdjustedSignal:
    if not assessment.available:
        return DepthAdjustedSignal(True, base_size_multiplier, "depth_unavailable")
    if core_score < min_core_score:
        return DepthAdjustedSignal(False, 0.0, "core_score_below_hard_threshold")
    if assessment.confirmed is False:
        return DepthAdjustedSignal(False, 0.0, assessment.veto_reason or "depth_veto")
    if assessment.confirmed is True:
        upgrade = min(
            0.50,
            0.30 * max(0.0, assessment.depth_imbalance_score)
            + 0.20 * max(0.0, assessment.queue_depletion_score),
        )
        return DepthAdjustedSignal(True, min(1.0, base_size_multiplier + upgrade), "depth_confirmed")
    return DepthAdjustedSignal(True, base_size_multiplier, "depth_neutral")

