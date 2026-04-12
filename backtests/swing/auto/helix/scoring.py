"""Helix composite scoring -- 7-component structure targeting exit efficiency and waste.

Components (default weights):
  - net_profit  (22%): log(1+return)/log(3), equity curve total
  - pf          (14%): (PF-1)/2, win quality
  - exit_efficiency (18%): sum_R / sum_MFE_pos, aggregate MFE capture
  - waste_ratio (14%): 1 - (|stale_R| + |short_hold_R|) / gross_win_R
  - tail_preservation (10%): big_winner_pct / 0.60
  - inv_dd      (8%): 1 - max_r_dd / 25.0
  - frequency   (14%): trades / 500

Hard rejects: <200 trades, PF<1.2, max_r_dd>25, tail_pct<0.30, min_regime_pf<0.80
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Base weights -- sum to 1.0
W_NET_PROFIT = 0.22
W_PF = 0.14
W_EXIT_EFF = 0.18
W_WASTE = 0.14
W_TAIL = 0.10
W_INV_DD = 0.08
W_FREQUENCY = 0.14


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


@dataclass(frozen=True)
class HelixCompositeScore:
    """Frozen composite score."""
    net_profit_component: float = 0.0
    pf_component: float = 0.0
    exit_efficiency_component: float = 0.0
    waste_ratio_component: float = 0.0
    tail_preservation_component: float = 0.0
    inv_dd_component: float = 0.0
    frequency_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class HelixMetrics:
    """Metrics extracted from a Helix backtest run."""
    total_trades: int = 0
    profit_factor: float = 0.0
    net_return_pct: float = 0.0
    max_r_dd: float = 0.0
    exit_efficiency: float = 0.0
    waste_ratio: float = 0.0
    tail_pct: float = 0.0
    bull_pf: float = 0.0
    bear_pf: float = 0.0
    min_regime_pf: float = 0.0
    total_r: float = 0.0
    gross_win_r: float = 0.0
    gross_loss_r: float = 0.0
    stale_r: float = 0.0
    short_hold_r: float = 0.0
    big_winner_r: float = 0.0
    sharpe: float = 0.0
    calmar_r: float = 0.0
    win_rate: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0


def composite_score(
    metrics: HelixMetrics,
    weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> HelixCompositeScore:
    """Compute the Helix composite score."""
    hr = hard_rejects or {}
    min_trades = hr.get("min_trades", 200)
    min_pf = hr.get("min_pf", 1.2)
    max_dd = hr.get("max_r_dd", 25.0)
    min_tail = hr.get("min_tail_pct", 0.30)
    min_rpf = hr.get("min_regime_pf", 0.80)

    # Hard rejects
    if metrics.total_trades < min_trades:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Too few trades: {metrics.total_trades} < {min_trades:.0f}",
        )
    if metrics.profit_factor < min_pf:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"PF too low: {metrics.profit_factor:.2f} < {min_pf}",
        )
    if metrics.max_r_dd > max_dd:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Max R DD too high: {metrics.max_r_dd:.1f} > {max_dd}",
        )
    if metrics.tail_pct < min_tail:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Tail preservation too low: {metrics.tail_pct:.2f} < {min_tail}",
        )
    if metrics.min_regime_pf < min_rpf:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Min regime PF too low: {metrics.min_regime_pf:.2f} < {min_rpf}",
        )

    w = weights or {}
    w_np = w.get("net_profit", W_NET_PROFIT)
    w_pf = w.get("pf", W_PF)
    w_ee = w.get("exit_efficiency", W_EXIT_EFF)
    w_wa = w.get("waste_ratio", W_WASTE)
    w_ta = w.get("tail_preservation", W_TAIL)
    w_dd = w.get("inv_dd", W_INV_DD)
    w_fr = w.get("frequency", W_FREQUENCY)

    # Components
    np_return = max(metrics.net_return_pct / 100.0, 0.0)
    np_c = _clip01(math.log(1.0 + np_return) / math.log(3.0))
    pf_c = _clip01((metrics.profit_factor - 1.0) / 2.0)
    ee_c = _clip01(metrics.exit_efficiency / 0.40)
    wa_c = _clip01(metrics.waste_ratio)
    ta_c = _clip01(metrics.tail_pct / 0.60)
    dd_c = _clip01(1.0 - metrics.max_r_dd / 25.0)
    fr_c = _clip01(metrics.total_trades / 500.0)

    total = (
        w_np * np_c
        + w_pf * pf_c
        + w_ee * ee_c
        + w_wa * wa_c
        + w_ta * ta_c
        + w_dd * dd_c
        + w_fr * fr_c
    )

    return HelixCompositeScore(
        net_profit_component=np_c,
        pf_component=pf_c,
        exit_efficiency_component=ee_c,
        waste_ratio_component=wa_c,
        tail_preservation_component=ta_c,
        inv_dd_component=dd_c,
        frequency_component=fr_c,
        total=total,
    )


def extract_helix_metrics(
    result,
    initial_equity: float,
) -> HelixMetrics:
    """Extract HelixMetrics from a Helix portfolio result."""
    all_trades = []
    for sr in result.symbol_results.values():
        all_trades.extend(sr.trades)

    if not all_trades:
        return HelixMetrics()

    wins = [t for t in all_trades if t.r_multiple > 0]
    losses = [t for t in all_trades if t.r_multiple <= 0]

    gross_win = sum(t.r_multiple for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.r_multiple for t in losses)) if losses else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    total_r = sum(t.r_multiple for t in all_trades)

    # Regime-specific PF
    bull_trades = [t for t in all_trades if getattr(t, "regime_at_entry", "") == "BULL"]
    bear_trades = [t for t in all_trades if getattr(t, "regime_at_entry", "") == "BEAR"]

    def _regime_pf(trades):
        w = sum(t.r_multiple for t in trades if t.r_multiple > 0)
        l = abs(sum(t.r_multiple for t in trades if t.r_multiple <= 0))
        return w / l if l > 0 else 999.0

    bull_pf = _regime_pf(bull_trades) if bull_trades else 999.0
    bear_pf = _regime_pf(bear_trades) if bear_trades else 999.0
    min_regime_pf = min(bull_pf, bear_pf)

    # Exit efficiency: aggregate sum_R / sum_MFE_pos
    sum_r = total_r
    sum_mfe_pos = sum(t.mfe_r for t in all_trades if t.mfe_r > 0)
    exit_eff = sum_r / sum_mfe_pos if sum_mfe_pos > 0 else 0.0

    # Waste ratio: 1 - (|stale_R| + |short_hold_R|) / gross_win_R
    stale_trades = [t for t in all_trades if getattr(t, "exit_reason", "") == "STALE"]
    short_hold_trades = [t for t in all_trades if getattr(t, "bars_held", 0) <= 10 and t.r_multiple < 0]
    stale_r = abs(sum(t.r_multiple for t in stale_trades)) if stale_trades else 0.0
    short_hold_r = abs(sum(t.r_multiple for t in short_hold_trades)) if short_hold_trades else 0.0
    waste = (stale_r + short_hold_r) / gross_win if gross_win > 0 else 0.0
    waste_ratio = max(0.0, 1.0 - waste)

    # Tail preservation: big winners (>=3R) as pct of gross win R
    big_winners = [t for t in wins if t.r_multiple >= 3.0]
    big_winner_r = sum(t.r_multiple for t in big_winners) if big_winners else 0.0
    tail_pct = big_winner_r / gross_win if gross_win > 0 else 0.0

    # Max R drawdown
    cum_r = np.cumsum([t.r_multiple for t in all_trades])
    peak_r = np.maximum.accumulate(cum_r)
    r_dd = peak_r - cum_r
    max_r_dd = float(np.max(r_dd)) if len(r_dd) > 0 else 0.0

    # Net return
    eq = result.combined_equity
    if len(eq) > 0:
        net_ret = (eq[-1] - initial_equity) / initial_equity * 100
    else:
        net_ret = 0.0

    # Sharpe
    sharpe = 0.0
    if len(eq) > 2:
        hourly_returns = np.diff(eq) / eq[:-1]
        hourly_returns = hourly_returns[~np.isnan(hourly_returns)]
        if len(hourly_returns) > 1:
            mu = float(np.mean(hourly_returns))
            sigma = float(np.std(hourly_returns))
            if sigma > 0:
                sharpe = mu / sigma * math.sqrt(252.0 * 7.0)

    # Calmar (R-based)
    calmar_r = total_r / max_r_dd if max_r_dd > 0 else 0.0

    # Win rate
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    avg_win = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0

    return HelixMetrics(
        total_trades=len(all_trades),
        profit_factor=pf,
        net_return_pct=net_ret,
        max_r_dd=max_r_dd,
        exit_efficiency=max(0.0, min(1.0, exit_eff)),
        waste_ratio=waste_ratio,
        tail_pct=tail_pct,
        bull_pf=bull_pf,
        bear_pf=bear_pf,
        min_regime_pf=min_regime_pf,
        total_r=total_r,
        gross_win_r=gross_win,
        gross_loss_r=gross_loss,
        stale_r=stale_r,
        short_hold_r=short_hold_r,
        big_winner_r=big_winner_r,
        sharpe=sharpe,
        calmar_r=calmar_r,
        win_rate=wr,
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
    )
