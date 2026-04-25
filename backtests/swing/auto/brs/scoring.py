"""BRS composite scoring on synchronized, fee-net, campaign-level results."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from backtests.swing.analysis.brs_trade_metrics import (
    BEAR_REGIMES,
    NON_DOWNTURN_REGIMES,
    summarize_brs_campaigns,
)


@dataclass(frozen=True)
class BRSCompositeScore:
    calmar_component: float = 0.0
    pf_component: float = 0.0
    inv_dd_component: float = 0.0
    net_profit_component: float = 0.0
    bear_alpha_component: float = 0.0
    frequency_component: float = 0.0
    detection_quality_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


W_NET_PROFIT = 0.25
W_PF = 0.13
W_CALMAR = 0.09
W_INV_DD = 0.08
W_BEAR_ALPHA = 0.12
W_FREQUENCY = 0.15
W_DETECTION = 0.18


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


@dataclass
class BRSMetrics:
    total_trades: int = 0
    campaign_count: int = 0
    realized_leg_count: int = 0
    bear_trades: int = 0
    bear_trade_wr: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    bear_alpha_pct: float = 0.0
    regime_f1: float = 0.0
    exit_efficiency: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    net_return_pct: float = 0.0
    bear_pf: float = 0.0
    detection_latency_days: float = 999.0
    bias_latency_days: float = 999.0
    bear_capture_ratio: float = 0.0
    crisis_coverage: float = 0.0
    total_net_pnl_dollars: float = 0.0
    bear_net_pnl_dollars: float = 0.0
    non_downturn_net_pnl_dollars: float = 0.0
    downturn_net_pnl_share: float = 0.0
    non_downturn_net_pnl_share: float = 0.0


def composite_score(
    metrics: BRSMetrics,
    weights: dict[str, float] | None = None,
) -> BRSCompositeScore:
    if metrics.total_trades < 10:
        return BRSCompositeScore(rejected=True, reject_reason=f"Too few campaigns: {metrics.total_trades} < 10")
    if metrics.max_dd_pct > 0.30:
        return BRSCompositeScore(rejected=True, reject_reason=f"Max DD too high: {metrics.max_dd_pct:.1%} > 30%")
    if metrics.total_net_pnl_dollars > 0 and metrics.non_downturn_net_pnl_share > 0.50:
        return BRSCompositeScore(
            rejected=True,
            reject_reason=(
                "Net PnL is too dependent on RANGE_CHOP/BULL_TREND "
                f"({metrics.non_downturn_net_pnl_share:.0%})."
            ),
        )

    w = weights or {}
    w_np = w.get("net_profit", W_NET_PROFIT)
    w_pf = w.get("pf", W_PF)
    w_ca = w.get("calmar", W_CALMAR)
    w_dd = w.get("inv_dd", W_INV_DD)
    w_ba = w.get("bear_alpha", W_BEAR_ALPHA)
    w_fr = w.get("frequency", W_FREQUENCY)
    w_dq = w.get("detection_quality", W_DETECTION)

    np_return = max(metrics.net_return_pct / 100.0, 0.0)
    np_c = _clip01(math.log(1.0 + np_return) / math.log(4.0))
    pf_c = _clip01((metrics.profit_factor - 1.0) / 2.0)
    calmar_c = _clip01(metrics.calmar / 10.0)
    inv_dd_c = _clip01(1.0 - metrics.max_dd_pct / 0.30)
    bear_alpha_c = _clip01(metrics.bear_alpha_pct / 20.0)
    freq_c = _clip01(metrics.total_trades / 80.0)
    bias_lat_score = _clip01(1.0 - metrics.bias_latency_days / 20.0)
    trade_lat_score = _clip01(1.0 - metrics.detection_latency_days / 30.0)
    coverage_score = _clip01(metrics.crisis_coverage)
    dq_c = 0.4 * bias_lat_score + 0.3 * trade_lat_score + 0.3 * coverage_score

    total = (
        w_np * np_c
        + w_pf * pf_c
        + w_ca * calmar_c
        + w_dd * inv_dd_c
        + w_ba * bear_alpha_c
        + w_fr * freq_c
        + w_dq * dq_c
    )

    return BRSCompositeScore(
        net_profit_component=np_c,
        pf_component=pf_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        bear_alpha_component=bear_alpha_c,
        frequency_component=freq_c,
        detection_quality_component=dq_c,
        total=total,
    )


def extract_brs_metrics(result, initial_equity: float) -> BRSMetrics:
    all_trades = []
    all_crisis_logs = []
    for symbol_result in result.symbol_results.values():
        all_trades.extend(symbol_result.trades)
        all_crisis_logs.extend(getattr(symbol_result, "crisis_state_log", []))

    campaigns = summarize_brs_campaigns(all_trades)
    if not campaigns:
        return BRSMetrics()

    bear_campaigns = [campaign for campaign in campaigns if campaign.regime_entry in BEAR_REGIMES]
    win_campaigns = [campaign for campaign in campaigns if campaign.fee_net_pnl_dollars > 0]
    loss_campaigns = [campaign for campaign in campaigns if campaign.fee_net_pnl_dollars <= 0]
    bear_wins = [campaign for campaign in bear_campaigns if campaign.fee_net_pnl_dollars > 0]

    total_wins = sum(campaign.fee_net_pnl_dollars for campaign in win_campaigns)
    total_losses = abs(sum(campaign.fee_net_pnl_dollars for campaign in loss_campaigns))
    profit_factor = total_wins / total_losses if total_losses > 0 else 999.0

    bear_wins_total = sum(campaign.fee_net_pnl_dollars for campaign in bear_wins)
    bear_losses_total = abs(sum(campaign.fee_net_pnl_dollars for campaign in bear_campaigns if campaign.fee_net_pnl_dollars <= 0))
    bear_pf = bear_wins_total / bear_losses_total if bear_losses_total > 0 else 999.0

    eq = result.combined_equity
    if len(eq) > 0:
        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / peak
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        net_ret = (float(eq[-1]) - initial_equity) / initial_equity * 100.0
    else:
        max_dd = 0.0
        net_ret = 0.0

    total_net_pnl = sum(campaign.fee_net_pnl_dollars for campaign in campaigns)
    bear_net_pnl = sum(campaign.fee_net_pnl_dollars for campaign in bear_campaigns)
    non_downturn_net_pnl = sum(
        campaign.fee_net_pnl_dollars
        for campaign in campaigns
        if campaign.regime_entry in NON_DOWNTURN_REGIMES
    )
    positive_total_net_pnl = sum(max(0.0, campaign.fee_net_pnl_dollars) for campaign in campaigns)
    positive_non_downturn = sum(
        max(0.0, campaign.fee_net_pnl_dollars)
        for campaign in campaigns
        if campaign.regime_entry in NON_DOWNTURN_REGIMES
    )
    non_downturn_share = positive_non_downturn / positive_total_net_pnl if positive_total_net_pnl > 0 else 0.0
    downturn_share = 1.0 - non_downturn_share if positive_total_net_pnl > 0 else 0.0

    captured = sum(campaign.fee_net_r_multiple for campaign in campaigns)
    available = sum(campaign.mfe_r for campaign in campaigns if campaign.mfe_r > 0)
    exit_eff = captured / available if available > 0 else 0.0

    sharpe = 0.0
    if len(eq) > 2:
        hourly_returns = np.diff(eq) / eq[:-1]
        hourly_returns = hourly_returns[~np.isnan(hourly_returns)]
        if len(hourly_returns) > 1:
            mu = float(np.mean(hourly_returns))
            sigma = float(np.std(hourly_returns))
            if sigma > 0:
                sharpe = mu / sigma * math.sqrt(252.0 * 7.0)

    bear_capture = bear_net_pnl / abs(total_net_pnl) if total_net_pnl != 0 else 0.0

    regime_f1 = 0.0
    if bear_campaigns:
        precision = len(bear_wins) / len(bear_campaigns) if bear_campaigns else 0.0
        recall = len(bear_wins) / len(win_campaigns) if win_campaigns else 0.0
        if precision + recall > 0:
            regime_f1 = 2 * precision * recall / (precision + recall)

    detection_latency = 999.0
    bias_latency = 999.0
    crisis_coverage = 0.0
    try:
        from backtests.swing.analysis.brs_diagnostics import CRISIS_WINDOWS

        latency_days = []
        windows_with_trades = 0
        for name, crisis_start, crisis_end in CRISIS_WINDOWS:
            crisis_campaigns = [
                campaign
                for campaign in campaigns
                if campaign.entry_time is not None
                and crisis_start <= _naive_dt(campaign.entry_time) <= crisis_end
            ]
            if crisis_campaigns:
                windows_with_trades += 1
                first_entry = min(_naive_dt(campaign.entry_time) for campaign in crisis_campaigns if campaign.entry_time is not None)
                latency_days.append((first_entry - crisis_start).total_seconds() / 86400.0)
        if latency_days:
            detection_latency = sum(latency_days) / len(latency_days)
        if CRISIS_WINDOWS:
            crisis_coverage = windows_with_trades / len(CRISIS_WINDOWS)

        bias_latency_days = []
        for name, crisis_start, _crisis_end in CRISIS_WINDOWS:
            crisis_entries = [entry for entry in all_crisis_logs if entry["crisis"] == name and entry["bias_confirmed"] == "SHORT"]
            if crisis_entries:
                first_date = min(datetime.strptime(entry["date"], "%Y-%m-%d") for entry in crisis_entries)
                bias_latency_days.append((first_date - crisis_start).total_seconds() / 86400.0)
        if bias_latency_days:
            bias_latency = sum(bias_latency_days) / len(bias_latency_days)
    except Exception:
        pass

    calmar = abs(net_ret / 100.0) / max_dd if max_dd > 0 else 0.0
    return BRSMetrics(
        total_trades=len(campaigns),
        campaign_count=len(campaigns),
        realized_leg_count=len(all_trades),
        bear_trades=len(bear_campaigns),
        bear_trade_wr=len(bear_wins) / len(bear_campaigns) * 100.0 if bear_campaigns else 0.0,
        profit_factor=profit_factor,
        max_dd_pct=max_dd,
        bear_alpha_pct=bear_net_pnl / initial_equity * 100.0,
        regime_f1=regime_f1,
        exit_efficiency=max(0.0, min(1.0, exit_eff)),
        calmar=calmar,
        sharpe=sharpe,
        net_return_pct=net_ret,
        bear_pf=bear_pf,
        detection_latency_days=detection_latency,
        bias_latency_days=bias_latency,
        bear_capture_ratio=max(0.0, min(1.0, bear_capture)),
        crisis_coverage=crisis_coverage,
        total_net_pnl_dollars=total_net_pnl,
        bear_net_pnl_dollars=bear_net_pnl,
        non_downturn_net_pnl_dollars=non_downturn_net_pnl,
        downturn_net_pnl_share=downturn_share,
        non_downturn_net_pnl_share=non_downturn_share,
    )


def _naive_dt(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value
