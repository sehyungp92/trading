"""Section 8/10: End-to-end weekly signal engine orchestrator."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .config import MetaConfig
from .features import build_observation_matrix
from .hmm import fit_or_refit_hmm
from .inference import compute_confidence, compute_crisis_prob
from .leverage import compute_leverage
from .portfolio import (
    apply_ventilator,
    blend_policy_portfolios,
    confidence_fallback,
    default_regime_budgets,
    estimate_shrunk_covariance,
    weights_from_risk_budget,
)


def run_signal_engine(
    macro_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    market_df: pd.DataFrame,
    growth_feature: str,
    inflation_feature: str,
    cfg: MetaConfig,
) -> pd.DataFrame:

    required_cols = ["SPY", "EFA", "TLT", "GLD", "IBIT", cfg.cash_col]
    for c in required_cols:
        assert c in strat_ret_df.columns, f"Missing: {c}"

    # -- Build expanded observation matrix --
    Xz, g_idx, i_idx = build_observation_matrix(
        macro_df, market_df, strat_ret_df, cfg,
        growth_feature=growth_feature,
        inflation_feature=inflation_feature,
    )

    sleeves = strat_ret_df.columns.tolist()
    regime_budgets, w_neutral = default_regime_budgets(sleeves)

    p_crisis_daily = compute_crisis_prob(
        market_df.reindex(Xz.index).ffill(),
        strat_ret_df.reindex(Xz.index).ffill(),
        cfg,
    ).fillna(0.0)  # warm-up period: assume no crisis when data is insufficient

    # Snap refit dates to nearest prior trading day (handles weekends/holidays)
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

    rows = []
    model: Optional[GaussianHMM] = None
    refit_pointer = 0
    exempt_state: Dict[str, int] = {}
    P_prev: Optional[np.ndarray] = None

    for dt in weekly_dates:
        # -- Refit HMM if scheduled (with OOS guard) --
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

            model, diag = fit_or_refit_hmm(
                X_train=X_train,
                cfg=cfg,
                growth_idx=g_idx,
                infl_idx=i_idx,
                prev_model=model,
                first_fit=first_fit,
            )
            refit_pointer += 1

        if model is None:
            continue  # no model fitted yet

        # -- Posterior (aligned to [G,R,S,D]) --
        x_dt = Xz.loc[[dt]].values
        P_grsd = model.predict_proba(x_dt)[0]

        # -- Confidence with stability --
        conf = compute_confidence(P_grsd, P_prev, cfg)
        P_prev = P_grsd.copy()

        p_crisis = float(p_crisis_daily.loc[dt])

        # -- Shrunk covariance for risk budgeting --
        hist = strat_ret_df.loc[:dt]
        cov_annual = estimate_shrunk_covariance(hist, cfg)

        # -- Regime portfolio weights via correlation-adjusted budgets --
        wG = weights_from_risk_budget(regime_budgets["G"], cov_annual, cfg)
        wR = weights_from_risk_budget(regime_budgets["R"], cov_annual, cfg)
        wS = weights_from_risk_budget(regime_budgets["S"], cov_annual, cfg)
        wD = weights_from_risk_budget(regime_budgets["D"], cov_annual, cfg)

        # -- Probability-weighted blend --
        w_active = blend_policy_portfolios(P_grsd, wG, wR, wS, wD)

        # -- Confidence fallback --
        w_pre = confidence_fallback(w_active, w_neutral, conf)

        # -- Anticipatory ventilator --
        w_post, exempt_state = apply_ventilator(
            w=w_pre.copy(),
            p_crisis=p_crisis,
            hist_ret=hist,
            cfg=cfg,
            exempt_state=exempt_state,
        )

        # -- Phase B leverage (decoupled from confidence) --
        L = compute_leverage(w_post, hist, cfg)

        # -- Record --
        row = {
            "date": dt,
            "P_G": P_grsd[0],
            "P_R": P_grsd[1],
            "P_S": P_grsd[2],
            "P_D": P_grsd[3],
            "Conf": conf,
            "p_crisis": p_crisis,
            "L": L,
        }
        for c in sleeves:
            row[f"w_{c}"] = float(w_post.get(c, 0.0))
            row[f"pi_{c}"] = float(L * w_post.get(c, 0.0))
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    cfg = MetaConfig(sigma_floor_annual=0.05, L_max=1.0)

    macro_df = pd.read_parquet("macro_features.parquet")
    strat_ret_df = pd.read_parquet("strategy_returns.parquet")
    market_df = pd.read_parquet("market_stress.parquet")

    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="growth_feature",
        inflation_feature="inflation_feature",
        cfg=cfg,
    )

    signals.to_parquet("meta_signals_weekly_v2.parquet")
    print(signals.tail())
