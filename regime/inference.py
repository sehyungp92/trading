"""Section 3: Phase A outputs — confidence and crisis probability."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import MetaConfig
from .utils import compute_avg_pairwise_corr, rolling_zscore, sigmoid


def compute_confidence(
    P_grsd: np.ndarray,
    P_prev: Optional[np.ndarray],
    cfg: MetaConfig,
) -> float:
    """Entropy x posterior stability confidence score."""
    eps = 1e-12
    p = np.clip(P_grsd, eps, 1.0)

    # Entropy component (normalised)
    H = -np.sum(p * np.log(p))
    Hmax = np.log(4.0)
    conf_entropy = 1.0 - H / Hmax

    # Stability component (L1 distance between consecutive posteriors)
    if P_prev is not None:
        posterior_shift = np.sum(np.abs(P_grsd - P_prev))  # range [0, 2]
        stability = 1.0 - 0.5 * posterior_shift  # range [0, 1]
    else:
        stability = 1.0  # first week — assume stable

    # Blend
    raw = conf_entropy * (
        (1.0 - cfg.stability_weight) + cfg.stability_weight * stability
    )
    conf = cfg.conf_floor + (1.0 - cfg.conf_floor) * raw
    return float(np.clip(conf, cfg.conf_floor, 1.0))


def compute_crisis_prob(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cfg: MetaConfig,
) -> pd.Series:
    """VIX + spread + correlation composite crisis probability."""
    vix_z = rolling_zscore(market_df[["VIX"]], cfg.z_window, cfg.z_minp)["VIX"]
    spr_z = rolling_zscore(market_df[["SPREAD"]], cfg.z_window, cfg.z_minp)["SPREAD"]

    non_cash = strat_ret_df.drop(columns=[cfg.cash_col], errors="ignore")
    avg_corr = compute_avg_pairwise_corr(non_cash, window=cfg.rho_short_window)
    corr_z = rolling_zscore(avg_corr.to_frame(), cfg.z_window, cfg.z_minp).iloc[:, 0]

    w1, w2, w3 = cfg.crisis_weights
    composite = w1 * vix_z + w2 * spr_z + w3 * corr_z
    return sigmoid(cfg.crisis_logit_a * (composite - cfg.crisis_logit_b)).rename(
        "p_crisis"
    )
