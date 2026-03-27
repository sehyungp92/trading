"""Phase-specific scoring functions for multi-phase regime optimization.

Each phase has its own scoring function with different weight distributions:
  Phase 1: 60% regime health / 40% financial — break HMM collapse
  Phase 2: 40% regime health / 60% financial — fix features + quadrant separation
  Phase 3: 25% regime health / 75% financial+crisis — crisis integration
  Phase 4: 15% regime health / 85% financial+validation — production readiness
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.backtests.regime.analysis.metrics import PortfolioMetrics
from research.backtests.regime.auto.scoring import CompositeScore


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Regime stats computation (runs in worker, where signals DataFrame is available)
# ---------------------------------------------------------------------------

# Known crisis periods for crisis response scoring (Phase 3+)
CRISIS_PERIODS = {
    "GFC":       ("2008-09-01", "2009-03-01", "D"),
    "COVID":     ("2020-02-15", "2020-04-15", "D"),
    "Inflation": ("2022-01-01", "2022-10-01", "S"),
}

REGIME_COLS = ["P_G", "P_R", "P_S", "P_D"]
REGIME_LABELS = ["G", "R", "S", "D"]


def compute_regime_stats(signals: pd.DataFrame) -> dict:
    """Compute regime health statistics from a signals DataFrame.

    Args:
        signals: DataFrame with columns P_G, P_R, P_S, P_D (posterior probabilities).

    Returns:
        dict with: n_active_regimes, regime_entropy, transition_rate,
        avg_posterior_entropy, conf_std, dominant_dist, crisis_response,
        historical_alignment.
    """
    if signals.empty or not all(c in signals.columns for c in REGIME_COLS):
        return _empty_stats()

    probs = signals[REGIME_COLS].values  # (T, 4)
    dominant = np.argmax(probs, axis=1)

    # Regime distribution (fraction of time each regime is dominant)
    regime_counts = np.bincount(dominant, minlength=4) / len(dominant)

    # Number of active regimes (>2% of weeks)
    n_active = int(np.sum(regime_counts > 0.02))

    # Shannon entropy of regime distribution (normalized to [0, 1])
    regime_entropy = _shannon_entropy(regime_counts)

    # Transition rate (regime changes per week)
    transitions = np.sum(dominant[1:] != dominant[:-1])
    transition_rate = transitions / len(dominant) if len(dominant) > 1 else 0.0

    # Average posterior entropy (measures uncertainty/spread of posteriors)
    posterior_entropies = np.array([_shannon_entropy(row) for row in probs])
    avg_posterior_entropy = float(np.mean(posterior_entropies))

    # Confidence std (from Conf column if available)
    conf_std = float(signals["Conf"].std()) if "Conf" in signals.columns else 0.0

    # Dominant regime distribution dict
    dominant_dist = {REGIME_LABELS[i]: float(regime_counts[i]) for i in range(4)}

    # Crisis response
    crisis_response = _compute_crisis_response(signals)

    return {
        "n_active_regimes": n_active,
        "regime_entropy": regime_entropy,
        "transition_rate": transition_rate,
        "avg_posterior_entropy": avg_posterior_entropy,
        "conf_std": conf_std,
        "dominant_dist": dominant_dist,
        "crisis_response": crisis_response,
    }


def _empty_stats() -> dict:
    return {
        "n_active_regimes": 0,
        "regime_entropy": 0.0,
        "transition_rate": 0.0,
        "avg_posterior_entropy": 0.0,
        "conf_std": 0.0,
        "dominant_dist": {"G": 0, "R": 0, "S": 0, "D": 0},
        "crisis_response": 0.0,
    }


def _shannon_entropy(p: np.ndarray) -> float:
    """Normalized Shannon entropy (0 = one state, 1 = uniform)."""
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / np.log(4))  # normalize by log(4) for 4 states


def _compute_crisis_response(signals: pd.DataFrame) -> float:
    """Score how well regime assignments match known crisis periods."""
    regime_col_map = {"G": "P_G", "R": "P_R", "S": "P_S", "D": "P_D"}
    scores = []

    for name, (start, end, expected) in CRISIS_PERIODS.items():
        crisis_slice = signals.loc[start:end]
        if crisis_slice.empty:
            continue
        expected_col = regime_col_map[expected]
        dominant = crisis_slice[REGIME_COLS].idxmax(axis=1)
        match_frac = float((dominant == expected_col).mean())
        scores.append(match_frac)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Phase scoring functions
# ---------------------------------------------------------------------------

def phase_1_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    """Phase 1: Fix HMM Dynamics — 60% regime health / 40% financial."""
    # Hard rejects
    if metrics.max_drawdown_pct > 0.40:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Max DD {metrics.max_drawdown_pct:.1%} > 40%")
    if regime_stats["n_active_regimes"] < 3:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Only {regime_stats['n_active_regimes']} active regimes < 3")
    if regime_stats["transition_rate"] < 0.003:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Transition rate {regime_stats['transition_rate']:.4f} < 0.003")

    # Components
    entropy_c = _clip01(regime_stats["regime_entropy"])                           # 30%
    transition_c = _clip01(regime_stats["transition_rate"] / 0.02)               # 15%
    posterior_c = _clip01(regime_stats["avg_posterior_entropy"])                   # 15%
    sharpe_c = _clip01(metrics.sharpe / 1.5)                                     # 15%
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.40)                   # 15%
    cagr_c = _clip01(math.log(1 + metrics.cagr) / math.log(1.15)) if metrics.cagr > 0 else 0.0  # 10%

    total = (0.30 * entropy_c + 0.15 * transition_c + 0.15 * posterior_c
             + 0.15 * sharpe_c + 0.15 * inv_dd_c + 0.10 * cagr_c)

    # NOTE: Phase 1 repurposes CompositeScore slots since only .total is used
    # by the optimizer. calmar→transition, sortino→entropy. Don't read these
    # fields as literal financial metrics for Phase 1 scores.
    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=transition_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=entropy_c,
        total=total,
    )


def phase_2_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    """Phase 2: Fix Features — 40% regime health / 60% financial."""
    # Hard rejects
    if metrics.max_drawdown_pct > 0.35:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Max DD {metrics.max_drawdown_pct:.1%} > 35%")
    if regime_stats["n_active_regimes"] < 3:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Only {regime_stats['n_active_regimes']} active regimes < 3")
    if metrics.sharpe < 0.3:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Sharpe {metrics.sharpe:.3f} < 0.3")

    # Components
    sharpe_c = _clip01(metrics.sharpe / 1.5)                                     # 20%
    calmar_c = _clip01(metrics.calmar / 5.0)                                     # 15%
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.35)                   # 15%
    cagr_c = _clip01(math.log(1 + metrics.cagr) / math.log(1.15)) if metrics.cagr > 0 else 0.0  # 10%
    entropy_c = _clip01(regime_stats["regime_entropy"])                           # 20%
    posterior_c = _clip01(regime_stats["avg_posterior_entropy"])                   # 10%
    transition_c = _clip01(regime_stats["transition_rate"] / 0.02)               # 10%

    total = (0.20 * sharpe_c + 0.15 * calmar_c + 0.15 * inv_dd_c + 0.10 * cagr_c
             + 0.20 * entropy_c + 0.10 * posterior_c + 0.10 * transition_c)

    # NOTE: Phase 2 repurposes sortino slot for entropy. Only .total matters.
    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=entropy_c,
        total=total,
    )


def phase_3_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    """Phase 3: Crisis Integration — 25% regime health / 75% financial+crisis."""
    # Hard rejects
    if metrics.max_drawdown_pct > 0.30:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Max DD {metrics.max_drawdown_pct:.1%} > 30%")
    if metrics.sharpe < 0.5:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Sharpe {metrics.sharpe:.3f} < 0.5")
    if regime_stats["crisis_response"] < 0.2:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Crisis response {regime_stats['crisis_response']:.2f} < 0.2")
    if regime_stats["n_active_regimes"] < 3:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Only {regime_stats['n_active_regimes']} active regimes < 3")

    # Components
    sharpe_c = _clip01(metrics.sharpe / 1.5)                                     # 20%
    calmar_c = _clip01(metrics.calmar / 5.0)                                     # 15%
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.30)                   # 15%
    cagr_c = _clip01(math.log(1 + metrics.cagr) / math.log(1.15)) if metrics.cagr > 0 else 0.0  # 10%
    sortino_c = _clip01(metrics.sortino / 2.5)                                   # 10%
    crisis_c = _clip01(regime_stats["crisis_response"])                           # 15%
    entropy_c = _clip01(regime_stats["regime_entropy"])                           # 10%
    conf_cal_c = _clip01(regime_stats["conf_std"] / 0.2)                         # 5%

    total = (0.20 * sharpe_c + 0.15 * calmar_c + 0.15 * inv_dd_c + 0.10 * cagr_c
             + 0.10 * sortino_c + 0.15 * crisis_c + 0.10 * entropy_c + 0.05 * conf_cal_c)

    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=sortino_c,
        total=total,
    )


def phase_4_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    """Phase 4: Historical Validation — 15% regime health / 85% financial+validation."""
    # Hard rejects
    if metrics.max_drawdown_pct > 0.25:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Max DD {metrics.max_drawdown_pct:.1%} > 25%")
    if metrics.sharpe < 0.7:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Sharpe {metrics.sharpe:.3f} < 0.7")

    # historical_alignment is added to regime_stats by the phase-4 runner
    hist_align = regime_stats.get("historical_alignment", 0.0)
    if hist_align < 0.3:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Historical alignment {hist_align:.2f} < 0.3")

    # Components
    sharpe_c = _clip01(metrics.sharpe / 1.5)                                     # 20%
    calmar_c = _clip01(metrics.calmar / 5.0)                                     # 15%
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.25)                   # 10%
    cagr_c = _clip01(math.log(1 + metrics.cagr) / math.log(1.15)) if metrics.cagr > 0 else 0.0  # 10%
    sortino_c = _clip01(metrics.sortino / 2.5)                                   # 10%
    crisis_c = _clip01(regime_stats["crisis_response"])                           # 10%
    entropy_c = _clip01(regime_stats["regime_entropy"])                           # 10%
    hist_c = _clip01(hist_align)                                                  # 15%

    total = (0.20 * sharpe_c + 0.15 * calmar_c + 0.10 * inv_dd_c + 0.10 * cagr_c
             + 0.10 * sortino_c + 0.10 * crisis_c + 0.10 * entropy_c + 0.15 * hist_c)

    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=sortino_c,
        total=total,
    )


# ---------------------------------------------------------------------------
# Phase scorer registry
# ---------------------------------------------------------------------------

_PHASE_SCORERS = {
    1: phase_1_score,
    2: phase_2_score,
    3: phase_3_score,
    4: phase_4_score,
}


def get_phase_scorer(phase: int):
    """Return the scoring function for a given phase."""
    if phase not in _PHASE_SCORERS:
        raise ValueError(f"Unknown phase: {phase}. Valid phases: 1-4")
    return _PHASE_SCORERS[phase]
