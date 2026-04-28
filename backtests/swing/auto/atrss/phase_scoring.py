"""ATRSS phase-specific scoring -- weights are immutable, gates are progressive.

Two scoring regimes:
  - R1 (r1_independent): Original high thresholds for independent-account mode.
  - R9 (r9_synchronized): Rescaled for honest synchronized/fee-net conditions.

Active regime is selected via SCORING_REGIME module variable.
"""
from __future__ import annotations

from .scoring import ATRSSCompositeScore, ATRSSMetrics, composite_score

# ---------------------------------------------------------------------------
# Active scoring regime -- set to "r9" for honest conditions
# ---------------------------------------------------------------------------
SCORING_REGIME: str = "r9"

# Weights are immutable across all phases (None = use defaults)
PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,
    2: None,
    3: None,
    4: None,
}

# Progressive hard rejects per phase -- keyed by regime
_PHASE_HARD_REJECTS_R1: dict[int, dict[str, float]] = {
    1: {"min_trades": 100, "max_dd_pct": 0.07, "min_pf": 2.0, "min_wr": 0.55},
    2: {"min_trades": 120, "max_dd_pct": 0.06, "min_pf": 2.5, "min_wr": 0.58},
    3: {"min_trades": 140, "max_dd_pct": 0.055, "min_pf": 3.0, "min_wr": 0.60},
    4: {"min_trades": 150, "max_dd_pct": 0.05, "min_pf": 3.5, "min_wr": 0.65},
}

_PHASE_HARD_REJECTS_R9: dict[int, dict[str, float]] = {
    1: {"min_trades": 20, "max_dd_pct": 0.12, "min_pf": 1.0, "min_wr": 0.50},
    2: {"min_trades": 25, "max_dd_pct": 0.10, "min_pf": 1.2, "min_wr": 0.55},
    3: {"min_trades": 30, "max_dd_pct": 0.08, "min_pf": 1.5, "min_wr": 0.58},
    4: {"min_trades": 35, "max_dd_pct": 0.07, "min_pf": 1.8, "min_wr": 0.60},
}


def _get_phase_hard_rejects() -> dict[int, dict[str, float]]:
    return _PHASE_HARD_REJECTS_R9 if SCORING_REGIME == "r9" else _PHASE_HARD_REJECTS_R1


# Module-level export -- set once from SCORING_REGIME at import time.
# Internal callers should use _get_phase_hard_rejects() for runtime dispatch.
PHASE_HARD_REJECTS: dict[int, dict[str, float]] = (
    _PHASE_HARD_REJECTS_R9 if SCORING_REGIME == "r9" else _PHASE_HARD_REJECTS_R1
)

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Structural Fixes", ["profit_factor", "total_r", "total_trades"]),
    2: ("Exit Cleanup", ["profit_factor", "mfe_capture", "total_r"]),
    3: ("Signal & Filtering", ["total_trades", "win_rate", "trades_per_month"]),
    4: ("Fine-tune", ["calmar_r", "total_r", "sharpe"]),
}

_ULTIMATE_TARGETS_R1 = {
    "total_r": 300.0,
    "profit_factor": 8.0,
    "max_dd_pct": 0.015,
    "calmar_r": 70.0,
    "total_trades": 300,
    "mfe_capture": 0.80,
    "win_rate": 0.80,
    "trades_per_month": 5.0,
}

_ULTIMATE_TARGETS_R9 = {
    # Aspirational targets above Phase 0 vanilla baseline (2026-04-27):
    # Vanilla actuals: R=190.8, PF=4.47, DD=2.07%, cal_r=40.9, n=290,
    #                  MFE=0.654, WR=74.1%, TPM=5.0
    "total_r": 250.0,
    "profit_factor": 6.0,
    "max_dd_pct": 0.012,
    "calmar_r": 60.0,
    "total_trades": 350,
    "mfe_capture": 0.80,
    "win_rate": 0.85,
    "trades_per_month": 6.0,
}

ULTIMATE_TARGETS = _ULTIMATE_TARGETS_R9 if SCORING_REGIME == "r9" else _ULTIMATE_TARGETS_R1


def _scoring_profile() -> str:
    """Return the composite_score profile name for the active regime."""
    return "r9_synchronized" if SCORING_REGIME == "r9" else "r1_independent"


def score_phase_metrics(
    phase: int,
    metrics: ATRSSMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
    profile: str | None = None,
) -> ATRSSCompositeScore:
    """Score metrics for a specific phase with phase-appropriate hard rejects."""
    rejects = hard_rejects or _get_phase_hard_rejects().get(phase, {})
    scoring_profile = profile or _scoring_profile()
    # ATRSS uses fixed weights -- weight_overrides ignored
    return composite_score(metrics, weights=None, hard_rejects=rejects, profile=scoring_profile)
