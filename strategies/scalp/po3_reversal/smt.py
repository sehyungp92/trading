from __future__ import annotations

from collections.abc import Sequence

from .config import MIN_SMT_STRENGTH, SMT_MAX_LAG_BARS, TradeDirection
from .indicators import atr
from .models import PriceBar, SmtDivergence


def detect_smt_divergence(
    nq_bars: Sequence[PriceBar],
    es_bars: Sequence[PriceBar],
    direction: TradeDirection | int,
    *,
    max_lag_bars: int = SMT_MAX_LAG_BARS,
    min_strength: float = MIN_SMT_STRENGTH,
) -> SmtDivergence:
    direction = TradeDirection(int(direction))
    if direction is TradeDirection.FLAT or len(nq_bars) < 4 or len(es_bars) < 4:
        return SmtDivergence(False, direction=direction, reason="insufficient_context")

    max_lag = min(max_lag_bars, len(nq_bars) - 1, len(es_bars) - 1)
    best: SmtDivergence | None = None
    for lag in range(max_lag + 1):
        nq_idx = len(nq_bars) - 1
        es_idx = len(es_bars) - 1 - lag
        if es_idx <= 0:
            continue
        candidate = _detect_at_indices(nq_bars, es_bars, nq_idx, es_idx, direction, lag)
        if candidate.present and candidate.strength >= min_strength:
            if best is None or candidate.strength > best.strength:
                best = candidate
    return best or SmtDivergence(False, direction=direction, reason="no_material_divergence")


def _detect_at_indices(
    nq_bars: Sequence[PriceBar],
    es_bars: Sequence[PriceBar],
    nq_idx: int,
    es_idx: int,
    direction: TradeDirection,
    lag: int,
) -> SmtDivergence:
    nq_current = nq_bars[nq_idx]
    es_current = es_bars[es_idx]
    nq_prior = nq_bars[max(0, nq_idx - 8):nq_idx]
    es_prior = es_bars[max(0, es_idx - 8):es_idx]
    nq_atr = max(atr(nq_bars[max(0, nq_idx - 20):nq_idx + 1], 14), 0.25)
    es_atr = max(atr(es_bars[max(0, es_idx - 20):es_idx + 1], 14), 0.25)

    if direction is TradeDirection.LONG:
        nq_ref = min(bar.low for bar in nq_prior)
        es_ref = min(bar.low for bar in es_prior)
        es_swept = es_current.low < es_ref
        nq_held = nq_current.low >= nq_ref - 0.25
        nq_move_z = abs(nq_current.low - nq_ref) / nq_atr
        es_move_z = abs(es_current.low - es_ref) / es_atr
        strength = max(0.0, es_move_z - nq_move_z)
        return SmtDivergence(
            present=es_swept and nq_held,
            direction=direction,
            strength=float(strength),
            nq_extreme=nq_current.low,
            es_extreme=es_current.low,
            lag_bars=lag,
            reason="bullish_es_sweeps_nq_holds",
        )

    nq_ref = max(bar.high for bar in nq_prior)
    es_ref = max(bar.high for bar in es_prior)
    es_swept = es_current.high > es_ref
    nq_held = nq_current.high <= nq_ref + 0.25
    nq_move_z = abs(nq_current.high - nq_ref) / nq_atr
    es_move_z = abs(es_current.high - es_ref) / es_atr
    strength = max(0.0, es_move_z - nq_move_z)
    return SmtDivergence(
        present=es_swept and nq_held,
        direction=direction,
        strength=float(strength),
        nq_extreme=nq_current.high,
        es_extreme=es_current.high,
        lag_bars=lag,
        reason="bearish_es_sweeps_nq_holds",
    )

