"""ATRSS phase-specific scoring -- weights are immutable, gates are progressive."""
from __future__ import annotations

from .scoring import ATRSSCompositeScore, ATRSSMetrics, composite_score

# Weights are immutable across all phases (None = use defaults)
PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,
    2: None,
    3: None,
    4: None,
}

# Progressive hard rejects per phase
PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 100, "max_dd_pct": 0.07, "min_pf": 2.0, "min_wr": 0.55},
    2: {"min_trades": 120, "max_dd_pct": 0.06, "min_pf": 2.5, "min_wr": 0.58},
    3: {"min_trades": 140, "max_dd_pct": 0.055, "min_pf": 3.0, "min_wr": 0.60},
    4: {"min_trades": 150, "max_dd_pct": 0.05, "min_pf": 3.5, "min_wr": 0.65},
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Exit Cleanup", ["profit_factor", "mfe_capture", "total_r"]),
    2: ("Signal & Filtering", ["total_trades", "win_rate", "trades_per_month"]),
    3: ("Entry & Fill Optimization", ["trades_per_month", "total_trades", "win_rate"]),
    4: ("Sizing & Fine-tune", ["calmar_r", "total_r", "sharpe"]),
}

ULTIMATE_TARGETS = {
    "total_r": 300.0,
    "profit_factor": 8.0,
    "max_dd_pct": 0.015,
    "calmar_r": 70.0,
    "total_trades": 300,
    "mfe_capture": 0.80,
    "win_rate": 0.80,
    "trades_per_month": 5.0,
}


def score_phase_metrics(
    phase: int,
    metrics: ATRSSMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> ATRSSCompositeScore:
    """Score metrics for a specific phase with phase-appropriate hard rejects."""
    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    # ATRSS uses fixed weights -- weight_overrides ignored
    return composite_score(metrics, weights=None, hard_rejects=rejects)
