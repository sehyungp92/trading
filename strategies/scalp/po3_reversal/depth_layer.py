from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DepthAssessment:
    available: bool = False
    confirmed: bool | None = None
    imbalance_score: float = 0.0
    queue_score: float = 0.0
    veto_reason: str = ""


@dataclass(frozen=True, slots=True)
class DepthAdjustedSignal:
    allowed: bool
    size_multiplier: float
    reason: str = ""


def apply_depth_layer(
    *,
    core_score: float,
    core_threshold: float,
    base_size_multiplier: float,
    depth: DepthAssessment,
) -> DepthAdjustedSignal:
    """Live-only refinement: never creates a signal below core threshold."""
    if not depth.available:
        return DepthAdjustedSignal(True, base_size_multiplier, "depth_unavailable")
    if core_score < core_threshold:
        return DepthAdjustedSignal(False, 0.0, "core_score_below_threshold")
    if depth.confirmed is False:
        return DepthAdjustedSignal(False, 0.0, depth.veto_reason or "depth_veto")
    improvement = 0.0
    if depth.confirmed is True:
        improvement = min(0.25, max(0.0, 0.15 * depth.imbalance_score + 0.10 * depth.queue_score))
    return DepthAdjustedSignal(True, min(1.0, base_size_multiplier + improvement), "depth_confirmed")

