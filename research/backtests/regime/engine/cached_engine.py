"""Cached HMM engine for fast evaluation of non-HMM parameter candidates.

Caches fitted HMM models from a baseline run, then replays the allocation
pipeline (crisis, confidence, covariance, risk budget, ventilator, leverage)
with different MetaConfig params — skipping the expensive HMM fitting step.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from regime.config import MetaConfig
from regime.features import build_observation_matrix
from regime.hmm import fit_or_refit_hmm
from regime.inference import compute_confidence, compute_crisis_prob
from regime.leverage import compute_leverage
from regime.portfolio import (
    apply_ventilator,
    blend_policy_portfolios,
    confidence_fallback,
    default_regime_budgets,
    estimate_shrunk_covariance,
    weights_from_risk_budget,
)

# Parameters that affect HMM fitting or feature construction (Xz).
# Candidates touching ONLY other params can reuse cached HMM models.
HMM_AFFECTING_PARAMS = frozenset({
    # Feature construction (changes Xz input to HMM)
    "z_window", "z_minp",
    # HMM architecture
    "n_states", "covariance_type", "sticky_diag", "sticky_offdiag",
    "n_iter_first_fit", "n_iter_refit", "tol", "min_covar", "random_state",
    # Refit strategy
    "refit_freq", "use_expanding_window", "use_warm_start",
    "refit_validation_window", "refit_ll_tolerance",
    # Phase additions
    "rolling_window_years", "warm_start_perturb_std",
    "use_commodity_feature", "use_real_rates_feature",
    "drop_momentum_breadth", "drop_eq_bond_corr",
})


@dataclass
class HMMCache:
    """Cached HMM state from a baseline engine run."""
    Xz: pd.DataFrame
    g_idx: int
    i_idx: int
    models: List[Tuple[pd.Timestamp, GaussianHMM]]
    weekly_dates: pd.DatetimeIndex


def mutations_affect_hmm(mutations: dict) -> bool:
    """Check if any mutation key affects HMM fitting."""
    return bool(set(mutations) & HMM_AFFECTING_PARAMS)


def hmm_cache_key(mutations: dict) -> frozenset:
    """Compute cache key from HMM-affecting mutations only."""
    return frozenset(
        (k, repr(v)) for k, v in sorted(mutations.items())
        if k in HMM_AFFECTING_PARAMS
    )


def build_hmm_cache(
    macro_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    market_df: pd.DataFrame,
    growth_feature: str,
    inflation_feature: str,
    cfg: MetaConfig,
) -> HMMCache:
    """Run HMM fitting pipeline and cache models at each refit date."""
    Xz, g_idx, i_idx = build_observation_matrix(
        macro_df, market_df, strat_ret_df, cfg,
        growth_feature=growth_feature,
        inflation_feature=inflation_feature,
    )

    refit_dates_raw = pd.date_range(
        Xz.index.min(), Xz.index.max(), freq=cfg.refit_freq
    )
    snap_idx = Xz.index.searchsorted(refit_dates_raw, side="right") - 1
    snap_idx = snap_idx[(snap_idx >= 0) & (snap_idx < len(Xz.index))]
    refit_dates = Xz.index[np.unique(snap_idx)]

    weekly_dates = pd.date_range(
        Xz.index.min(), Xz.index.max(), freq=cfg.rebalance_freq
    )
    weekly_dates = weekly_dates.intersection(Xz.index)

    # Fit HMM at each refit date, cache deep copies
    models: List[Tuple[pd.Timestamp, GaussianHMM]] = []
    model: Optional[GaussianHMM] = None
    refit_pointer = 0

    for dt in weekly_dates:
        while refit_pointer < len(refit_dates) and refit_dates[refit_pointer] <= dt:
            rd = refit_dates[refit_pointer]
            if cfg.use_expanding_window:
                X_train = Xz.loc[:rd].values
            else:
                window_start = rd - pd.DateOffset(years=cfg.rolling_window_years)
                X_train = Xz.loc[window_start:rd].values
                if len(X_train) < 252:
                    X_train = Xz.loc[:rd].values  # fallback for early dates
            first_fit = model is None
            model, _ = fit_or_refit_hmm(
                X_train=X_train, cfg=cfg,
                growth_idx=g_idx, infl_idx=i_idx,
                prev_model=model, first_fit=first_fit,
            )
            models.append((rd, copy.deepcopy(model)))
            refit_pointer += 1

    return HMMCache(
        Xz=Xz, g_idx=g_idx, i_idx=i_idx,
        models=models, weekly_dates=weekly_dates,
    )


def run_from_cache(
    cache: HMMCache,
    strat_ret_df: pd.DataFrame,
    market_df: pd.DataFrame,
    cfg: MetaConfig,
) -> pd.DataFrame:
    """Run allocation pipeline using cached HMM models.

    Skips HMM fitting (the ~90% bottleneck) and re-runs all downstream steps
    (crisis, confidence, covariance, risk budget, blend, ventilator, leverage)
    with the provided cfg.
    """
    Xz = cache.Xz
    sleeves = strat_ret_df.columns.tolist()
    regime_budgets, w_neutral = default_regime_budgets(sleeves)

    p_crisis_daily = compute_crisis_prob(
        market_df.reindex(Xz.index).ffill(),
        strat_ret_df.reindex(Xz.index).ffill(),
        cfg,
    ).fillna(0.0)

    # Recompute rebalance dates from cfg (may differ from cache baseline)
    rebal_dates = pd.date_range(
        Xz.index.min(), Xz.index.max(), freq=cfg.rebalance_freq
    )
    rebal_dates = rebal_dates.intersection(Xz.index)

    rows = []
    model: Optional[GaussianHMM] = None
    model_pointer = 0
    exempt_state: Dict[str, int] = {}
    P_prev: Optional[np.ndarray] = None

    for dt in rebal_dates:
        # Advance to the latest cached model at or before dt
        while model_pointer < len(cache.models) and cache.models[model_pointer][0] <= dt:
            model = cache.models[model_pointer][1]
            model_pointer += 1

        if model is None:
            continue

        # Posteriors from cached model (fast — no fitting)
        x_dt = Xz.loc[[dt]].values
        P_grsd = model.predict_proba(x_dt)[0]

        conf = compute_confidence(P_grsd, P_prev, cfg)
        P_prev = P_grsd.copy()

        p_crisis = float(p_crisis_daily.loc[dt])

        hist = strat_ret_df.loc[:dt]
        cov_annual = estimate_shrunk_covariance(hist, cfg)

        wG = weights_from_risk_budget(regime_budgets["G"], cov_annual, cfg)
        wR = weights_from_risk_budget(regime_budgets["R"], cov_annual, cfg)
        wS = weights_from_risk_budget(regime_budgets["S"], cov_annual, cfg)
        wD = weights_from_risk_budget(regime_budgets["D"], cov_annual, cfg)

        w_active = blend_policy_portfolios(P_grsd, wG, wR, wS, wD)
        w_pre = confidence_fallback(w_active, w_neutral, conf)
        w_post, exempt_state = apply_ventilator(
            w=w_pre.copy(), p_crisis=p_crisis, hist_ret=hist,
            cfg=cfg, exempt_state=exempt_state,
        )
        L = compute_leverage(w_post, hist, cfg)

        row = {
            "date": dt,
            "P_G": P_grsd[0], "P_R": P_grsd[1],
            "P_S": P_grsd[2], "P_D": P_grsd[3],
            "Conf": conf, "p_crisis": p_crisis, "L": L,
        }
        for c in sleeves:
            row[f"w_{c}"] = float(w_post.get(c, 0.0))
            row[f"pi_{c}"] = float(L * w_post.get(c, 0.0))
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")
