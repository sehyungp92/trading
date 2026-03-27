"""Per-phase candidate lists for multi-phase regime optimization.

Each phase targets specific root causes with tailored parameter ranges.
Phase 1: HMM dynamics (sticky prior, rolling window, warm start)
Phase 2: Features (new/drop features, z-score, covariance)
Phase 3: Crisis + portfolio (crisis overlay, confidence, ventilator, leverage, risk budget)
Phase 4: Fine-tuning (narrow ranges around prior phase optima)
"""
from __future__ import annotations

from typing import Any


def get_phase_candidates(
    phase: int,
    prior_diagnostics: dict | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return candidate list for a given phase.

    Args:
        phase: 1-4
        prior_diagnostics: Optional diagnostics from previous phase (for adaptive candidates).

    Returns:
        List of (name, mutations_dict) tuples.
    """
    if phase == 1:
        return _phase_1_candidates()
    elif phase == 2:
        return _phase_2_candidates()
    elif phase == 3:
        return _phase_3_candidates()
    elif phase == 4:
        return _phase_4_candidates(prior_diagnostics)
    else:
        raise ValueError(f"Unknown phase: {phase}. Valid phases: 1-4")


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """Phase 1: Fix HMM Dynamics — all HMM-affecting candidates."""
    candidates = []

    # Sticky prior (reduce from default 50)
    for diag in [5, 10, 15, 20]:
        candidates.append((f"sticky_diag_{diag}", {"sticky_diag": float(diag)}))

    for offdiag in [3, 5, 8]:
        candidates.append((f"sticky_offdiag_{offdiag}", {"sticky_offdiag": float(offdiag)}))

    # Rolling window (switch from expanding)
    for years in [5, 7, 10]:
        candidates.append((
            f"rolling_{years}y",
            {"use_expanding_window": False, "rolling_window_years": years},
        ))

    # Warm start
    candidates.append(("no_warm_start", {"use_warm_start": False}))
    for std in [0.1, 0.3]:
        candidates.append((
            f"warm_start_perturb_{std}",
            {"warm_start_perturb_std": std},
        ))

    # Refit frequency
    for freq in ["6ME", "QE"]:
        candidates.append((f"refit_{freq}", {"refit_freq": freq}))

    # OOS guard relaxation (allow structural change)
    for tol in [1.0, 2.0, 5.0]:
        candidates.append((f"ll_tol_{tol}", {"refit_ll_tolerance": tol}))

    return candidates


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """Phase 2: Fix Features — feature engineering candidates."""
    candidates = []

    # New features (HMM-affecting)
    candidates.append(("add_commodity", {"use_commodity_feature": True}))
    candidates.append(("add_real_rates", {"use_real_rates_feature": True}))

    # Feature drops (HMM-affecting)
    candidates.append(("drop_momentum_breadth", {"drop_momentum_breadth": True}))
    candidates.append(("drop_eq_bond_corr", {"drop_eq_bond_corr": True}))

    # Z-score window (HMM-affecting)
    for window in [126, 504]:
        candidates.append((f"z_window_{window}", {"z_window": window}))
    for minp in [30, 90]:
        candidates.append((f"z_minp_{minp}", {"z_minp": minp}))

    # Covariance window (non-HMM)
    for cov_win in [42, 126]:
        candidates.append((f"cov_window_{cov_win}", {"cov_window": cov_win}))

    # HMM covariance type
    candidates.append(("cov_type_diag", {"covariance_type": "diag"}))

    # Feature swap combinations
    candidates.append((
        "swap_eqbond_for_commodity",
        {"use_commodity_feature": True, "drop_eq_bond_corr": True},
    ))
    candidates.append((
        "swap_breadth_for_realrates",
        {"use_real_rates_feature": True, "drop_momentum_breadth": True},
    ))
    candidates.append((
        "add_both_new_features",
        {"use_commodity_feature": True, "use_real_rates_feature": True},
    ))
    candidates.append((
        "add_both_drop_both",
        {
            "use_commodity_feature": True,
            "use_real_rates_feature": True,
            "drop_momentum_breadth": True,
            "drop_eq_bond_corr": True,
        },
    ))

    return candidates


def _phase_3_candidates() -> list[tuple[str, dict]]:
    """Phase 3: Crisis Integration & Portfolio Tuning — mostly non-HMM."""
    candidates = []

    # Crisis overlay
    candidates.append(("crisis_weights_vix_heavy", {"crisis_weights": (0.6, 0.3, 0.1)}))
    candidates.append(("crisis_weights_spread_heavy", {"crisis_weights": (0.3, 0.5, 0.2)}))
    for a in [0.5, 2.0, 3.0]:
        candidates.append((f"crisis_logit_a_{a}", {"crisis_logit_a": a}))
    for b in [-0.5, 0.5]:
        candidates.append((f"crisis_logit_b_{b}", {"crisis_logit_b": b}))

    # Confidence
    for floor in [0.4, 0.5]:
        candidates.append((f"conf_floor_{floor}", {"conf_floor": floor}))
    for sw in [0.2, 0.8]:
        candidates.append((f"stability_weight_{sw}", {"stability_weight": sw}))

    # Ventilator
    for lam in [0.5, 1.0]:
        candidates.append((f"vent_lambda_{lam}", {"ventilator_lambda": lam}))
    for thresh in [0.15, 0.35]:
        candidates.append((f"delta_rho_thresh_{thresh}", {"delta_rho_threshold": thresh}))
    for exempt in [-0.20, 0.0]:
        candidates.append((f"delta_rho_exempt_{exempt}", {"delta_rho_exempt": exempt}))

    # Leverage
    for lmax in [1.0, 1.3, 1.5]:
        candidates.append((f"L_max_{lmax}", {"L_max": lmax}))
    for kappa in [1.0, 1.5]:
        candidates.append((f"kappa_cap_{kappa}", {"kappa_totalvol_cap": kappa}))
    for vol in [0.08, 0.10]:
        candidates.append((f"target_vol_{vol}", {"base_target_vol_annual": vol}))

    # Risk budget
    for floor in [0.03, 0.08]:
        candidates.append((f"sigma_floor_{floor}", {"sigma_floor_annual": floor}))
    for psm in [0.35, 0.50]:
        candidates.append((f"per_strat_max_{psm}", {"per_strat_max": psm}))

    return candidates


def _phase_4_candidates(prior_diagnostics: dict | None = None) -> list[tuple[str, dict]]:
    """Phase 4: Fine-tuning — narrow ranges around prior optima.

    If prior_diagnostics has 'cumulative_mutations', generates candidates
    around accepted values (±20% or ±1 step). Otherwise, returns defaults.
    """
    candidates = []
    accepted = (prior_diagnostics or {}).get("cumulative_mutations", {})

    if accepted:
        # Generate narrow-range candidates around each accepted numeric param
        _FINE_TUNE = {
            "sticky_diag":           lambda v: [v * 0.8, v * 1.2],
            "sticky_offdiag":        lambda v: [max(1, v - 1), v + 1],
            "rolling_window_years":  lambda v: [max(3, v - 1), v + 1],
            "warm_start_perturb_std": lambda v: [max(0.01, v * 0.5), v * 1.5],
            "refit_ll_tolerance":    lambda v: [max(0.1, v * 0.7), v * 1.3],
            "conf_floor":            lambda v: [max(0.1, v - 0.05), min(0.9, v + 0.05)],
            "L_max":                 lambda v: [max(0.5, v - 0.1), v + 0.1],
            "base_target_vol_annual": lambda v: [max(0.03, v - 0.01), v + 0.01],
            "sigma_floor_annual":    lambda v: [max(0.01, v - 0.01), v + 0.01],
            "crisis_logit_a":        lambda v: [max(0.1, v - 0.5), v + 0.5],
        }
        for param, gen_fn in _FINE_TUNE.items():
            if param in accepted:
                val = accepted[param]
                if not isinstance(val, (int, float)):
                    continue
                for new_val in gen_fn(val):
                    new_val = round(new_val, 4)
                    if new_val != val:
                        candidates.append((
                            f"fine_{param}_{new_val}",
                            {param: float(new_val)},
                        ))
        # If rolling window was accepted, also try adjacent years
        if "rolling_window_years" in accepted:
            for adj in [accepted["rolling_window_years"] - 1,
                        accepted["rolling_window_years"] + 1]:
                if 3 <= adj <= 15:
                    candidates.append((
                        f"fine_rolling_{adj}y",
                        {"use_expanding_window": False, "rolling_window_years": adj},
                    ))

    # Always include some defaults (may overlap with adaptive — greedy handles dupes)
    for diag in [8, 12, 18]:
        candidates.append((f"fine_sticky_{diag}", {"sticky_diag": float(diag)}))

    candidates.append(("fine_refit_4ME", {"refit_freq": "4ME"}))

    for std in [0.05, 0.15, 0.2]:
        candidates.append((f"fine_perturb_{std}", {"warm_start_perturb_std": std}))

    for floor in [0.35, 0.45]:
        candidates.append((f"fine_conf_{floor}", {"conf_floor": floor}))

    for lmax in [1.1, 1.2]:
        candidates.append((f"fine_L_max_{lmax}", {"L_max": lmax}))

    # Deduplicate by name
    seen = set()
    deduped = []
    for name, muts in candidates:
        if name not in seen:
            seen.add(name)
            deduped.append((name, muts))
    return deduped
