"""BRS per-phase candidate selection from experiment categories.

Phase 1: REGIME + ENTRY params (NO ablations) + STRUCTURAL (~60 candidates)
Phase 2: SIGNAL_SELECT — signal combination experiments (~6 candidates)
Phase 3: EXIT + VOLATILITY (~30 candidates)
Phase 4: SIZING + FINETUNE (~30 candidates)
"""
from __future__ import annotations

from backtests.swing.auto.brs.experiment_categories import (
    get_category_experiments,
)


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Get experiment candidates for a specific phase.

    Args:
        phase: Optimization phase (1-4)
        prior_mutations: Cumulative mutations from prior phases
        suggested_experiments: Experiments suggested by phase analyzer

    Returns:
        List of (name, mutations_dict) candidates
    """
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates()
    elif phase == 4:
        candidates = _phase_4_candidates(prior_mutations or {})
    else:
        candidates = []

    # Merge suggested experiments (from phase analyzer)
    if suggested_experiments:
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """REGIME + ENTRY params (NO ablations) + STRUCTURAL categories.

    Signal ablations are excluded — they run in Phase 2 after params are tuned.
    """
    # Get ENTRY experiments but EXCLUDE signal ablations
    entry_exps = [
        (n, m) for n, m in get_category_experiments(["ENTRY"])
        if not n.startswith("entry_abl_") and n != "entry_reenable_s1_only"
    ]
    regime_exps = get_category_experiments(["REGIME"])
    structural_exps = get_category_experiments(["STRUCTURAL"])
    return regime_exps + entry_exps + structural_exps


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """SIGNAL_SELECT — signal combination experiments tested after Phase 1 tuning."""
    return get_category_experiments(["SIGNAL_SELECT"])


def _phase_3_candidates() -> list[tuple[str, dict]]:
    """EXIT + VOLATILITY categories."""
    return get_category_experiments(["EXIT", "VOLATILITY"])


def _phase_4_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """SIZING + adaptive fine-tuning around accepted values from Phase 1-3."""
    sizing = get_category_experiments(["SIZING"])

    if not prior_mutations:
        return sizing

    finetune = []
    for key, value in prior_mutations.items():
        if not isinstance(value, (int, float)):
            continue

        # Generate ±10% and ±20% variants
        for pct_label, pct in [("m20", 0.80), ("m10", 0.90), ("p10", 1.10), ("p20", 1.20)]:
            new_val = value * pct
            if isinstance(value, int):
                new_val = int(round(new_val))
                if new_val == value:
                    continue
            else:
                new_val = round(new_val, 6)

            name = f"finetune_{key}_{pct_label}"
            finetune.append((name, {key: new_val}))

    return sizing + finetune
