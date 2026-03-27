"""Configuration for the MR-AWQ Meta Allocator v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

REGIMES = ["G", "R", "S", "D"]  # Goldilocks, Reflation, Stagflation, Deflation

REGIME_TARGETS = {
    "G": np.array([+1.0, -1.0]),  # growth up, inflation down
    "R": np.array([+1.0, +1.0]),  # growth up, inflation up
    "S": np.array([-1.0, +1.0]),  # growth down, inflation up
    "D": np.array([-1.0, -1.0]),  # growth down, inflation down
}


@dataclass
class MetaConfig:
    # -- Rebalance --
    rebalance_freq: str = "W-FRI"
    cash_col: str = "CASH"
    ann_factor: float = 252.0

    # -- Macro standardisation --
    z_window: int = 252
    z_minp: int = 60

    # -- HMM --
    n_states: int = 4
    covariance_type: str = "full"
    n_iter_first_fit: int = 400
    n_iter_refit: int = 200
    tol: float = 1e-3
    min_covar: float = 1e-6
    random_state: int = 7
    sticky_diag: float = 50.0
    sticky_offdiag: float = 2.0

    # -- Refit --
    refit_freq: str = "YE"
    use_expanding_window: bool = True
    use_warm_start: bool = True
    refit_validation_window: int = 63
    refit_ll_tolerance: float = 0.5

    # -- Crisis overlay --
    crisis_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    crisis_logit_a: float = 1.0
    crisis_logit_b: float = 0.0

    # -- Confidence (entropy x stability) --
    conf_floor: float = 0.3
    stability_weight: float = 0.5

    # -- Risk budgeting (correlation-adjusted) --
    sigma_floor_annual: float = 0.05
    per_strat_max: float = 0.40
    strat_vol_span: int = 63
    cov_window: int = 63
    shrinkage_target: str = "ledoit_wolf"

    # -- Ventilator (delta-rho anticipatory) --
    ventilator_lambda: float = 0.8
    ventilator_vmin: float = 0.2
    rho_short_window: int = 20
    rho_long_window: int = 60
    delta_rho_threshold: float = 0.25
    delta_rho_exempt: float = -0.10
    pnl_confirm_days: int = 10
    risk_on_set: Tuple[str, ...] = ("SPY", "EFA", "IBIT")

    # -- Phase B leverage (fixed sigma-star, no confidence modulation) --
    L_max: float = 1.0
    kappa_totalvol_cap: float = 1.25
    base_target_vol_annual: float = 0.12
    ewma_downside_span: int = 20
    ewma_total_span: int = 60
    s_floor: float = 0.3
    gamma: float = 0.1
    dd_ladder: Tuple[Tuple[float, float], ...] = (
        (-0.08, 1.0),
        (-0.12, 0.7),
        (-0.16, 0.5),
        (-0.20, 0.3),
    )

    # -- Autoresearch phase additions --
    rolling_window_years: int = 7
    warm_start_perturb_std: float = 0.0
    use_commodity_feature: bool = False
    use_real_rates_feature: bool = False
    drop_momentum_breadth: bool = False
    drop_eq_bond_corr: bool = False
