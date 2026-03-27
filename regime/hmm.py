"""Section 2: Sticky HMM with cosine-based semantic alignment and OOS refit guard."""

from __future__ import annotations

import itertools
import logging
from typing import Optional, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

from .config import REGIMES, REGIME_TARGETS, MetaConfig


# ---------------------------------------------------------------------------
# Transition prior
# ---------------------------------------------------------------------------

def build_transmat_prior(n_states: int, diag: float, offdiag: float) -> np.ndarray:
    prior = np.full((n_states, n_states), offdiag)
    np.fill_diagonal(prior, diag)
    return prior


# ---------------------------------------------------------------------------
# Cosine alignment cost
# ---------------------------------------------------------------------------

def state_regime_cost(
    mean_vec: np.ndarray, regime: str, growth_idx: int, infl_idx: int
) -> float:
    """Negative cosine similarity between state's (growth, inflation) projection
    and the canonical regime direction. Continuous, no margin cliffs."""
    target = REGIME_TARGETS[regime]
    observed = np.array([mean_vec[growth_idx], mean_vec[infl_idx]])
    obs_norm = np.linalg.norm(observed) + 1e-8
    tgt_norm = np.linalg.norm(target)
    cosine_sim = np.dot(observed, target) / (obs_norm * tgt_norm)

    magnitude_bonus = obs_norm * max(cosine_sim, 0.0)
    return float(-cosine_sim - 0.3 * magnitude_bonus)


def build_cost_matrix(
    means: np.ndarray, growth_idx: int, infl_idx: int
) -> np.ndarray:
    n_states = means.shape[0]
    C = np.zeros((n_states, 4))
    for k in range(n_states):
        for j, r in enumerate(REGIMES):
            C[k, j] = state_regime_cost(means[k], r, growth_idx, infl_idx)
    return C


# ---------------------------------------------------------------------------
# Exact 4x4 assignment (brute-force permutations)
# ---------------------------------------------------------------------------

def solve_assignment(C: np.ndarray) -> np.ndarray:
    best_cost, best_perm = np.inf, None
    for perm in itertools.permutations(range(4)):
        cost = sum(C[perm[j], j] for j in range(4))
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return np.array(best_perm, dtype=int)  # new_idx -> old_idx


def align_hmm_inplace(
    hmm: GaussianHMM, growth_idx: int, infl_idx: int
) -> dict:
    C = build_cost_matrix(hmm.means_, growth_idx, infl_idx)
    order = solve_assignment(C)
    hmm.means_ = hmm.means_[order]
    hmm.covars_ = hmm.covars_[order]
    hmm.startprob_ = hmm.startprob_[order]
    hmm.transmat_ = hmm.transmat_[np.ix_(order, order)]
    return {"cost_matrix": C, "new_order": order}


# ---------------------------------------------------------------------------
# HMM initialization and warm-start helpers
# ---------------------------------------------------------------------------

def init_hmm(cfg: MetaConfig, n_iter: int) -> GaussianHMM:
    """Create a fresh GaussianHMM with sticky transition prior."""
    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        n_iter=n_iter,
        tol=cfg.tol,
        min_covar=cfg.min_covar,
        random_state=cfg.random_state,
        params="stmc",
        init_params="stmc",
    )
    model.transmat_prior = build_transmat_prior(
        cfg.n_states, cfg.sticky_diag, cfg.sticky_offdiag
    )
    return model


def warm_start_from_previous(
    model: GaussianHMM, prev_model: GaussianHMM
) -> None:
    """Copy learned parameters from a previous model as initialization."""
    model.means_prior = prev_model.means_.copy()
    model.means_ = prev_model.means_.copy()
    model.covars_ = prev_model.covars_.copy()
    model.startprob_ = prev_model.startprob_.copy()
    model.transmat_ = prev_model.transmat_.copy()
    model.init_params = ""  # skip random init, use copied params


# ---------------------------------------------------------------------------
# Fit / refit with OOS guard
# ---------------------------------------------------------------------------

def fit_or_refit_hmm(
    X_train: np.ndarray,
    cfg: MetaConfig,
    growth_idx: int,
    infl_idx: int,
    prev_model: Optional[GaussianHMM],
    first_fit: bool,
) -> Tuple[GaussianHMM, dict]:
    n_iter = cfg.n_iter_first_fit if first_fit else cfg.n_iter_refit
    model = init_hmm(cfg, n_iter)

    if cfg.use_warm_start and prev_model and not first_fit:
        warm_start_from_previous(model, prev_model)
        if cfg.warm_start_perturb_std > 0:
            # Vary seed by training size so each refit gets different noise
            seed = (cfg.random_state + X_train.shape[0]) % (2**31)
            model.means_ += np.random.RandomState(seed).randn(
                *model.means_.shape
            ) * cfg.warm_start_perturb_std

    try:
        model.fit(X_train)
    except Exception as exc:
        if prev_model is not None:
            logger.warning("HMM fit failed (%s), keeping previous model", exc)
            return prev_model, {"refit_rejected": True, "fit_error": str(exc)}
        raise  # first fit — no fallback available

    align_hmm_inplace(model, growth_idx, infl_idx)

    # -- OOS refit guard --
    if prev_model is not None and not first_fit:
        val_start = max(0, len(X_train) - cfg.refit_validation_window)
        X_val = X_train[val_start:]

        ll_new = model.score(X_val)
        ll_old = prev_model.score(X_val)

        if ll_new < ll_old - cfg.refit_ll_tolerance:
            return prev_model, {
                "refit_rejected": True,
                "ll_new": ll_new,
                "ll_old": ll_old,
            }

    return model, {"refit_rejected": False}
