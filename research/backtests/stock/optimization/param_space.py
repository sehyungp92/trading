"""Parameter space definitions for stock strategy optimization."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class ParamRange:
    """A single parameter with its optimization range."""

    name: str
    low: float
    high: float
    step: float
    default: float

    def values(self) -> list[float]:
        vals = []
        v = self.low
        while v <= self.high + 1e-9:
            vals.append(round(v, 6))
            v += self.step
        return vals

    @property
    def n_values(self) -> int:
        return len(self.values())


# ---------------------------------------------------------------------------
# ALCB parameter space
# ---------------------------------------------------------------------------

ALCB_PARAM_SPACE: list[ParamRange] = [
    ParamRange("base_risk_fraction", 0.003, 0.007, 0.001, 0.005),
    ParamRange("heat_cap_r", 4.0, 8.0, 1.0, 6.0),
    ParamRange("tp1_aligned_r", 1.0, 1.5, 0.25, 1.25),
    ParamRange("tp2_aligned_r", 2.0, 3.0, 0.5, 2.5),
    ParamRange("stale_exit_days", 8, 14, 2, 10),
    ParamRange("atr_stop_mult_std", 0.75, 1.5, 0.25, 1.0),
    ParamRange("min_containment", 0.70, 0.90, 0.05, 0.80),
]

# ---------------------------------------------------------------------------
# IARIC parameter space
# ---------------------------------------------------------------------------

IARIC_PARAM_SPACE: list[ParamRange] = [
    ParamRange("base_risk_fraction", 0.0015, 0.0035, 0.0005, 0.0025),
    ParamRange("partial_r_multiple", 1.0, 2.0, 0.25, 1.5),
    ParamRange("partial_exit_fraction", 0.33, 0.60, 0.09, 0.50),
    ParamRange("time_stop_minutes", 30, 60, 15, 45),
    ParamRange("avwap_band_pct", 0.003, 0.008, 0.001, 0.005),
]


def build_grid(params: list[ParamRange]) -> list[dict[str, float]]:
    """Generate all parameter combinations as a list of dicts."""
    names = [p.name for p in params]
    value_lists = [p.values() for p in params]
    grid = []
    for combo in product(*value_lists):
        grid.append(dict(zip(names, combo)))
    return grid


def grid_size(params: list[ParamRange]) -> int:
    """Total number of parameter combinations."""
    total = 1
    for p in params:
        total *= p.n_values
    return total
