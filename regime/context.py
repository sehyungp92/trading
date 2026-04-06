"""RegimeContext: downstream-facing dataclass for strategy coordinators."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeContext:
    regime: str                         # dominant macro regime (G/R/S/D)
    regime_confidence: float            # 0-1, posterior peakedness
    stress_level: float                 # 0-1, P(stress). 0.0 when stress model disabled
    stress_onset: bool                  # True if stress crossed above threshold this week
    shift_velocity: float               # rate of change in stress_level
    suggested_leverage_mult: float      # 0-1, recommended sizing scalar
    regime_allocations: dict[str, float]  # {SPY: 0.25, TLT: 0.55, ...}
    computed_at: str = ""                   # ISO timestamp for staleness detection
