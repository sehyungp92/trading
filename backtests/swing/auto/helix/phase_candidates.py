"""Helix per-phase candidate selection from experiment categories.

Phase 1: CLASS_PRUNING + ENTRY_QUALITY + P1 structural (~21 candidates)
Phase 2: STOP_PLACEMENT + TRAILING + INLINE_TRAILING + STALE_BAIL + PARTIALS_BE (~60 candidates)
Phase 3: VOLATILITY + ADDON + P3 structural (~25 candidates)
Phase 4: CIRCUIT_BREAKER + FINETUNE (~30+ candidates)
"""
from __future__ import annotations

from .experiment_categories import (
    _P1_STRUCTURAL,
    _P3_STRUCTURAL,
    get_category_experiments,
)


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
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

    if suggested_experiments:
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """Signal pruning + entry gates + P1 structural ablation."""
    pruning = get_category_experiments(["CLASS_PRUNING"])
    entry = get_category_experiments(["ENTRY_QUALITY"])
    structural = [
        (n, m) for n, m in get_category_experiments(["STRUCTURAL_ABLATION"])
        if n in _P1_STRUCTURAL
    ]
    return pruning + entry + structural


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """Exit management: stops, trailing, inline trailing, stale/bail, partials/BE."""
    return get_category_experiments([
        "STOP_PLACEMENT", "TRAILING", "INLINE_TRAILING",
        "STALE_BAIL", "PARTIALS_BE",
    ])


def _phase_3_candidates() -> list[tuple[str, dict]]:
    """Volatility + addon + P3 structural ablation."""
    vol_addon = get_category_experiments(["VOLATILITY", "ADDON"])
    structural = [
        (n, m) for n, m in get_category_experiments(["STRUCTURAL_ABLATION"])
        if n in _P3_STRUCTURAL
    ]
    return vol_addon + structural


def _phase_4_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Circuit breaker + auto-generated finetune of prior accepted mutations."""
    circuit = get_category_experiments(["CIRCUIT_BREAKER"])

    if not prior_mutations:
        return circuit

    finetune = []
    for key, value in prior_mutations.items():
        # bool must be checked first: isinstance(True, int) is True in Python
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue

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

    return circuit + finetune
