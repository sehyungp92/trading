"""Downturn R2 per-phase candidate selection from experiment categories.

3-phase correction-capture specialist optimization:

Phase 1: Regime Purification -- counter-blocking, correction detection, regime params
Phase 2: Capture Optimization -- exits, trailing, breakeven, scale-out, hold period
Phase 3: Risk Tuning & Polish -- finetune accepted params, sizing, engine ablation
"""
from __future__ import annotations

from backtests.momentum.auto.downturn.experiment_categories import (
    get_category_experiments,
)


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Get experiment candidates for a specific phase.

    Args:
        phase: Optimization phase (1-3)
        prior_mutations: Cumulative mutations from prior phases (for phase 3 finetune)
        suggested_experiments: Analyzer-suggested experiments (prepended)
    """
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates(prior_mutations or {})
    else:
        candidates = []

    # Prepend suggested experiments from analyzer
    if suggested_experiments:
        # Deduplicate by name
        existing_names = {name for name, _ in candidates}
        new = [(n, m) for n, m in suggested_experiments if n not in existing_names]
        candidates = new + candidates

    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """R2 Phase 1: Regime purification -- eliminate counter-regime, improve correction detection."""
    return get_category_experiments([
        "R2_COUNTER_BLOCKING", "R2_REGIME_COMBOS", "R2_REGIME_PARAMS",
        "FREQUENCY", "FAST_CRASH", "CONVICTION", "STRUCTURAL",
        "BEAR_STRUCTURE", "R8_INTRADAY_REGIME", "R8_ENTRY_FILTERS",
        "PROGRESSIVE_SMA",
    ])


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """R2 Phase 2: Capture optimization -- exits, trailing, breakeven, scale-out."""
    candidates = get_category_experiments([
        "EXIT", "EXIT_V2", "EXIT_ADAPTIVE", "TRAIL_REDESIGN",
        "R8_EXIT_IMPROVEMENTS", "HOLD_PERIOD",
    ])
    # R2 exit combos
    candidates.append(("r2_combo_pf_adapt", {
        "flags.profit_floor_trail": True,
        "flags.adaptive_profit_floor": True,
        "param_overrides.profit_floor_r_threshold": 1.0,
    }))
    return candidates


def _phase_3_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """R2 Phase 3: Risk tuning -- finetune accepted params, sizing, engine ablation."""
    candidates: list[tuple[str, dict]] = []

    # A. Auto-generated finetune from accepted numeric params (+/-10/20%)
    candidates.extend(_finetune_candidates(prior_mutations))

    # B. Sizing experiments
    candidates.extend(get_category_experiments(["SIZING"]))

    # C. Engine ablation (verify each engine still adds value post-regime-filtering)
    candidates.extend([
        ("r2_disable_reversal", {"flags.reversal_engine": False}),
        ("r2_disable_breakdown", {"flags.breakdown_engine": False}),
        ("r2_disable_fade", {"flags.fade_engine": False}),
        ("r2_enable_momentum", {"flags.momentum_signal": True}),
    ])

    return candidates


def _finetune_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Generate +/-10/20% variants around accepted numeric mutations."""
    candidates: list[tuple[str, dict]] = []

    if not prior_mutations:
        return candidates

    for key, value in prior_mutations.items():
        if not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue

        for pct_label, pct in [("m20", 0.80), ("m10", 0.90), ("p10", 1.10), ("p20", 1.20)]:
            new_val = value * pct
            if isinstance(value, int):
                new_val = int(round(new_val))
                if new_val == value:
                    continue
            else:
                new_val = round(new_val, 6)

            name = f"finetune_{key.replace('.', '_')}_{pct_label}"
            candidates.append((name, {key: new_val}))

    return candidates
