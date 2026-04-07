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

    def to_snapshot_dict(self) -> dict:
        """Serialize for DailySnapshot instrumentation."""
        return {
            "macro_regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "stress_level": self.stress_level,
            "stress_onset": self.stress_onset,
            "shift_velocity": self.shift_velocity,
            "suggested_leverage_mult": self.suggested_leverage_mult,
            "computed_at": self.computed_at,
        }


_APPLIED_CONFIG_SNAPSHOT_KEYS = frozenset({
    "directional_cap_R", "regime_unit_risk_mult",
    "disabled_strategies", "directional_cap_long_R",
    "directional_cap_short_R", "nqdtc_oppose_size_mult",
    "max_family_contracts_mnq_eq", "dd_tiers",
})


def serialize_applied_config(cfg) -> dict | None:
    """Serialize PortfolioRulesConfig for DailySnapshot instrumentation."""
    if cfg is None:
        return None
    from dataclasses import asdict
    return {
        k: v if not isinstance(v, frozenset) else sorted(v)
        for k, v in asdict(cfg).items()
        if k in _APPLIED_CONFIG_SNAPSHOT_KEYS
    }
