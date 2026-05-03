from __future__ import annotations

from dataclasses import dataclass, field

from .config import FULL_SIZE_SCORE, HALF_SIZE_SCORE, THREE_QUARTER_SIZE_SCORE


BASE_WEIGHTS: dict[str, float] = {
    "regime": 20.0,
    "retest": 20.0,
    "absorption": 15.0,
    "delta": 15.0,
    "target": 15.0,
    "volatility": 10.0,
    "time": 5.0,
}


@dataclass(frozen=True, slots=True)
class IvbSignalScore:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    available_components: tuple[str, ...] = ()
    size_multiplier: float = 0.0
    footprint_available: bool = False


def score_signal(
    *,
    regime_quality: float,
    retest_quality: float,
    target_quality: float,
    volatility_quality: float,
    time_quality: float,
    absorption_quality: float | None = None,
    delta_confirmation: float | None = None,
) -> IvbSignalScore:
    raw_components = {
        "regime": _clip(regime_quality),
        "retest": _clip(retest_quality),
        "absorption": None if absorption_quality is None else _clip(absorption_quality),
        "delta": None if delta_confirmation is None else _clip(delta_confirmation),
        "target": _clip(target_quality),
        "volatility": _clip(volatility_quality),
        "time": _clip(time_quality),
    }
    available = tuple(key for key, value in raw_components.items() if value is not None)
    available_weight = sum(BASE_WEIGHTS[key] for key in available)
    total = 0.0
    components: dict[str, float] = {}
    for key in available:
        normalized_weight = BASE_WEIGHTS[key] / available_weight * 100.0 if available_weight else 0.0
        contribution = normalized_weight * float(raw_components[key])
        components[key] = contribution
        total += contribution
    return IvbSignalScore(
        total=float(total),
        components=components,
        available_components=available,
        size_multiplier=size_multiplier_for_score(total),
        footprint_available=raw_components["absorption"] is not None and raw_components["delta"] is not None,
    )


def size_multiplier_for_score(score: float) -> float:
    if score >= FULL_SIZE_SCORE:
        return 1.0
    if score >= THREE_QUARTER_SIZE_SCORE:
        return 0.75
    if score >= HALF_SIZE_SCORE:
        return 0.50
    return 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

