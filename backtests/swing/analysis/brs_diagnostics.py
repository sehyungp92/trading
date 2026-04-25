"""BRS diagnostics centered on fee-net, campaign-level downturn alpha."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd

from backtests.swing.engine.backtest_engine import SymbolResult

from .brs_trade_metrics import BEAR_REGIMES, NON_DOWNTURN_REGIMES, summarize_brs_campaigns


CRISIS_WINDOWS = [
    ("2022 Bear", datetime(2022, 1, 3), datetime(2022, 10, 13)),
    ("SVB", datetime(2023, 3, 8), datetime(2023, 3, 15)),
    ("Aug 2024 Unwind", datetime(2024, 8, 1), datetime(2024, 8, 5)),
    ("Tariff Shock", datetime(2025, 2, 21), datetime(2025, 4, 7)),
    ("Mar 2026 Slow Burn", datetime(2026, 3, 5), datetime(2026, 3, 27)),
]


@dataclass
class RegimeMetrics:
    regime: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0
    avg_bars_held: float = 0.0


@dataclass
class EntryTypeMetrics:
    entry_type: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0


@dataclass
class ConvictionBucket:
    bucket: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    expectancy: float = 0.0


@dataclass
class ExitReasonMetrics:
    exit_reason: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0
    fee_net_pnl_dollars: float = 0.0
    avg_bars_held: float = 0.0


@dataclass
class ContributionMetrics:
    downturn_net_pnl_dollars: float = 0.0
    non_downturn_net_pnl_dollars: float = 0.0
    downturn_positive_share: float = 0.0
    non_downturn_positive_share: float = 0.0
    range_chop_net_pnl_dollars: float = 0.0
    bull_trend_net_pnl_dollars: float = 0.0


@dataclass
class ConcentrationMetrics:
    top_1_positive_share: float = 0.0
    top_5_positive_share: float = 0.0
    campaigns_for_half_positive_pnl: int = 0
    median_fee_net_r: float = 0.0
    p25_fee_net_r: float = 0.0
    p75_fee_net_r: float = 0.0


@dataclass
class CrisisResult:
    name: str
    start: datetime
    end: datetime
    brs_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0
    brs_trades: int = 0


@dataclass
class BRSDiagnostics:
    regime_metrics: list[RegimeMetrics] = field(default_factory=list)
    entry_type_metrics: list[EntryTypeMetrics] = field(default_factory=list)
    conviction_buckets: list[ConvictionBucket] = field(default_factory=list)
    exit_reason_metrics: list[ExitReasonMetrics] = field(default_factory=list)
    crisis_results: list[CrisisResult] = field(default_factory=list)
    contribution_metrics: ContributionMetrics = field(default_factory=ContributionMetrics)
    concentration_metrics: ConcentrationMetrics = field(default_factory=ConcentrationMetrics)
    bear_alpha_pct: float = 0.0
    total_trades: int = 0
    realized_leg_count: int = 0
    bear_trades: int = 0
    report: str = ""


def compute_brs_diagnostics(
    results: dict[str, SymbolResult],
    initial_equity: float,
    combined_equity: np.ndarray | None = None,
    combined_timestamps: np.ndarray | None = None,
) -> BRSDiagnostics:
    all_trades = []
    for symbol_result in results.values():
        all_trades.extend(symbol_result.trades)

    campaigns = summarize_brs_campaigns(all_trades)
    diagnostics = BRSDiagnostics(
        total_trades=len(campaigns),
        realized_leg_count=len(all_trades),
    )
    diagnostics.regime_metrics = _compute_regime_metrics(campaigns)
    diagnostics.entry_type_metrics = _compute_entry_type_metrics(campaigns)
    diagnostics.conviction_buckets = _compute_conviction_buckets(campaigns)
    diagnostics.exit_reason_metrics = _compute_exit_reason_metrics(campaigns)
    diagnostics.contribution_metrics = _compute_contribution_metrics(campaigns)
    diagnostics.concentration_metrics = _compute_concentration_metrics(campaigns)

    if combined_equity is not None and combined_timestamps is not None:
        diagnostics.crisis_results = _compute_crisis_returns(combined_equity, combined_timestamps)
        for crisis_result in diagnostics.crisis_results:
            crisis_result.brs_trades = sum(
                1
                for campaign in campaigns
                if campaign.entry_time is not None
                and crisis_result.start <= _naive_dt(campaign.entry_time) <= crisis_result.end
            )

    bear_campaigns = [campaign for campaign in campaigns if campaign.regime_entry in BEAR_REGIMES]
    diagnostics.bear_trades = len(bear_campaigns)
    diagnostics.bear_alpha_pct = (
        sum(campaign.fee_net_pnl_dollars for campaign in bear_campaigns) / initial_equity * 100.0
        if bear_campaigns
        else 0.0
    )
    diagnostics.report = _build_report(diagnostics)
    return diagnostics


def _compute_regime_metrics(campaigns) -> list[RegimeMetrics]:
    buckets: dict[str, list] = {}
    for campaign in campaigns:
        buckets.setdefault(campaign.regime_entry or "UNKNOWN", []).append(campaign)

    results = []
    for regime, bucket in sorted(buckets.items()):
        wins = [campaign for campaign in bucket if campaign.fee_net_pnl_dollars > 0]
        losses = [campaign for campaign in bucket if campaign.fee_net_pnl_dollars <= 0]
        gross_win = sum(campaign.fee_net_pnl_dollars for campaign in wins)
        gross_loss = abs(sum(campaign.fee_net_pnl_dollars for campaign in losses))
        count = len(bucket)
        results.append(
            RegimeMetrics(
                regime=regime,
                trade_count=count,
                win_rate=len(wins) / count * 100.0 if count > 0 else 0.0,
                avg_r=sum(campaign.fee_net_r_multiple for campaign in bucket) / count if count > 0 else 0.0,
                total_r=sum(campaign.fee_net_r_multiple for campaign in bucket),
                profit_factor=gross_win / gross_loss if gross_loss > 0 else 999.0,
                avg_bars_held=sum(campaign.bars_held for campaign in bucket) / count if count > 0 else 0.0,
            )
        )
    return results


def _compute_entry_type_metrics(campaigns) -> list[EntryTypeMetrics]:
    buckets: dict[str, list] = {}
    for campaign in campaigns:
        buckets.setdefault(campaign.entry_type or "UNKNOWN", []).append(campaign)

    results = []
    for entry_type, bucket in sorted(buckets.items()):
        wins = [campaign for campaign in bucket if campaign.fee_net_pnl_dollars > 0]
        losses = [campaign for campaign in bucket if campaign.fee_net_pnl_dollars <= 0]
        gross_win = sum(campaign.fee_net_pnl_dollars for campaign in wins)
        gross_loss = abs(sum(campaign.fee_net_pnl_dollars for campaign in losses))
        count = len(bucket)
        results.append(
            EntryTypeMetrics(
                entry_type=entry_type,
                trade_count=count,
                win_rate=len(wins) / count * 100.0 if count > 0 else 0.0,
                avg_r=sum(campaign.fee_net_r_multiple for campaign in bucket) / count if count > 0 else 0.0,
                total_r=sum(campaign.fee_net_r_multiple for campaign in bucket),
                profit_factor=gross_win / gross_loss if gross_loss > 0 else 999.0,
            )
        )
    return results


def _compute_conviction_buckets(campaigns) -> list[ConvictionBucket]:
    bucket_defs = [("0-25", 0, 25), ("25-50", 25, 50), ("50-75", 50, 75), ("75-100", 75, 100)]
    results = []
    for label, lo, hi in bucket_defs:
        bucket = [
            campaign
            for campaign in campaigns
            if lo <= campaign.score_entry < hi or (hi == 100 and campaign.score_entry == 100)
        ]
        count = len(bucket)
        wins = [campaign for campaign in bucket if campaign.fee_net_pnl_dollars > 0]
        win_rate = len(wins) / count if count > 0 else 0.0
        avg_r = sum(campaign.fee_net_r_multiple for campaign in bucket) / count if count > 0 else 0.0
        avg_win = sum(campaign.fee_net_r_multiple for campaign in wins) / len(wins) if wins else 0.0
        losses = [campaign.fee_net_r_multiple for campaign in bucket if campaign.fee_net_pnl_dollars <= 0]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        results.append(
            ConvictionBucket(
                bucket=label,
                trade_count=count,
                win_rate=win_rate * 100.0,
                avg_r=avg_r,
                expectancy=win_rate * avg_win + (1 - win_rate) * avg_loss,
            )
        )
    return results


def _compute_exit_reason_metrics(campaigns) -> list[ExitReasonMetrics]:
    buckets: dict[str, list] = {}
    for campaign in campaigns:
        buckets.setdefault(campaign.terminal_exit_reason or "UNKNOWN", []).append(campaign)

    results = []
    for exit_reason, bucket in sorted(buckets.items()):
        wins = [campaign for campaign in bucket if campaign.fee_net_pnl_dollars > 0]
        losses = [campaign for campaign in bucket if campaign.fee_net_pnl_dollars <= 0]
        gross_win = sum(campaign.fee_net_pnl_dollars for campaign in wins)
        gross_loss = abs(sum(campaign.fee_net_pnl_dollars for campaign in losses))
        count = len(bucket)
        results.append(
            ExitReasonMetrics(
                exit_reason=exit_reason,
                trade_count=count,
                win_rate=len(wins) / count * 100.0 if count > 0 else 0.0,
                avg_r=sum(campaign.fee_net_r_multiple for campaign in bucket) / count if count > 0 else 0.0,
                total_r=sum(campaign.fee_net_r_multiple for campaign in bucket),
                profit_factor=gross_win / gross_loss if gross_loss > 0 else 999.0,
                fee_net_pnl_dollars=sum(campaign.fee_net_pnl_dollars for campaign in bucket),
                avg_bars_held=sum(campaign.bars_held for campaign in bucket) / count if count > 0 else 0.0,
            )
        )
    return results


def _compute_contribution_metrics(campaigns) -> ContributionMetrics:
    downturn_net = sum(
        campaign.fee_net_pnl_dollars for campaign in campaigns if campaign.regime_entry in BEAR_REGIMES
    )
    non_downturn_net = sum(
        campaign.fee_net_pnl_dollars for campaign in campaigns if campaign.regime_entry in NON_DOWNTURN_REGIMES
    )
    positive_total = sum(max(0.0, campaign.fee_net_pnl_dollars) for campaign in campaigns)
    positive_downturn = sum(
        max(0.0, campaign.fee_net_pnl_dollars) for campaign in campaigns if campaign.regime_entry in BEAR_REGIMES
    )
    positive_non_downturn = sum(
        max(0.0, campaign.fee_net_pnl_dollars)
        for campaign in campaigns
        if campaign.regime_entry in NON_DOWNTURN_REGIMES
    )
    return ContributionMetrics(
        downturn_net_pnl_dollars=downturn_net,
        non_downturn_net_pnl_dollars=non_downturn_net,
        downturn_positive_share=positive_downturn / positive_total if positive_total > 0 else 0.0,
        non_downturn_positive_share=positive_non_downturn / positive_total if positive_total > 0 else 0.0,
        range_chop_net_pnl_dollars=sum(
            campaign.fee_net_pnl_dollars for campaign in campaigns if campaign.regime_entry == "RANGE_CHOP"
        ),
        bull_trend_net_pnl_dollars=sum(
            campaign.fee_net_pnl_dollars for campaign in campaigns if campaign.regime_entry == "BULL_TREND"
        ),
    )


def _compute_concentration_metrics(campaigns) -> ConcentrationMetrics:
    fee_net_rs = np.array([campaign.fee_net_r_multiple for campaign in campaigns], dtype=np.float64)
    positive_pnls = sorted(
        (campaign.fee_net_pnl_dollars for campaign in campaigns if campaign.fee_net_pnl_dollars > 0),
        reverse=True,
    )
    positive_total = sum(positive_pnls)

    campaigns_for_half = 0
    running_positive = 0.0
    if positive_total > 0:
        for value in positive_pnls:
            running_positive += value
            campaigns_for_half += 1
            if running_positive >= positive_total * 0.5:
                break

    return ConcentrationMetrics(
        top_1_positive_share=(positive_pnls[0] / positive_total) if positive_total > 0 and positive_pnls else 0.0,
        top_5_positive_share=(sum(positive_pnls[:5]) / positive_total) if positive_total > 0 else 0.0,
        campaigns_for_half_positive_pnl=campaigns_for_half,
        median_fee_net_r=float(np.median(fee_net_rs)) if len(fee_net_rs) > 0 else 0.0,
        p25_fee_net_r=float(np.percentile(fee_net_rs, 25)) if len(fee_net_rs) > 0 else 0.0,
        p75_fee_net_r=float(np.percentile(fee_net_rs, 75)) if len(fee_net_rs) > 0 else 0.0,
    )


def _compute_crisis_returns(equity: np.ndarray, timestamps: np.ndarray) -> list[CrisisResult]:
    results = []
    for name, start, end in CRISIS_WINDOWS:
        indices = []
        for idx, timestamp in enumerate(timestamps):
            dt = _to_datetime(timestamp)
            if dt is not None and start <= dt <= end:
                indices.append(idx)

        if not indices:
            results.append(CrisisResult(name=name, start=start, end=end))
            continue

        start_equity = float(equity[indices[0]])
        end_equity = float(equity[indices[-1]])
        brs_return = (end_equity - start_equity) / start_equity * 100.0 if start_equity > 0 else 0.0
        results.append(
            CrisisResult(
                name=name,
                start=start,
                end=end,
                brs_return_pct=brs_return,
            )
        )
    return results


def _build_report(diag: BRSDiagnostics) -> str:
    buf = StringIO()
    buf.write("=" * 70 + "\n")
    buf.write("BRS DIAGNOSTICS REPORT\n")
    buf.write("=" * 70 + "\n\n")
    buf.write(f"Campaigns:      {diag.total_trades}\n")
    buf.write(f"Realized legs:  {diag.realized_leg_count}\n")
    buf.write(f"Bear campaigns: {diag.bear_trades}\n")
    buf.write(f"Bear alpha:     {diag.bear_alpha_pct:.1f}% (fee-net)\n\n")

    buf.write("--- Regime Metrics (campaign / fee-net) ---\n")
    buf.write(f"{'Regime':<15} {'Count':>6} {'WR%':>6} {'AvgR':>7} {'TotalR':>8} {'PF':>6} {'AvgBars':>8}\n")
    for entry in diag.regime_metrics:
        buf.write(
            f"{entry.regime:<15} {entry.trade_count:>6} {entry.win_rate:>5.1f}% "
            f"{entry.avg_r:>7.2f} {entry.total_r:>8.1f} {entry.profit_factor:>6.2f} {entry.avg_bars_held:>8.0f}\n"
        )
    buf.write("\n")

    buf.write("--- Entry Type Metrics (campaign / fee-net) ---\n")
    buf.write(f"{'Type':<20} {'Count':>6} {'WR%':>6} {'AvgR':>7} {'TotalR':>8} {'PF':>6}\n")
    for entry in diag.entry_type_metrics:
        buf.write(
            f"{entry.entry_type:<20} {entry.trade_count:>6} {entry.win_rate:>5.1f}% "
            f"{entry.avg_r:>7.2f} {entry.total_r:>8.1f} {entry.profit_factor:>6.2f}\n"
        )
    buf.write("\n")

    buf.write("--- Conviction Buckets (campaign / fee-net) ---\n")
    buf.write(f"{'Bucket':<10} {'Count':>6} {'WR%':>6} {'AvgR':>7} {'Expect':>8}\n")
    for entry in diag.conviction_buckets:
        buf.write(
            f"{entry.bucket:<10} {entry.trade_count:>6} {entry.win_rate:>5.1f}% "
            f"{entry.avg_r:>7.2f} {entry.expectancy:>8.3f}\n"
        )
    buf.write("\n")

    buf.write("--- Terminal Exit Reason Metrics (campaign / fee-net) ---\n")
    buf.write(f"{'Exit':<18} {'Count':>6} {'WR%':>6} {'AvgR':>7} {'TotalR':>8} {'PF':>6} {'FeeNet$':>10}\n")
    for entry in diag.exit_reason_metrics:
        buf.write(
            f"{entry.exit_reason:<18} {entry.trade_count:>6} {entry.win_rate:>5.1f}% "
            f"{entry.avg_r:>7.2f} {entry.total_r:>8.1f} {entry.profit_factor:>6.2f} "
            f"{entry.fee_net_pnl_dollars:>+10.0f}\n"
        )
    buf.write("\n")

    buf.write("--- Downturn Alignment Audit ---\n")
    buf.write(
        f"Downturn fee-net PnL:      ${diag.contribution_metrics.downturn_net_pnl_dollars:+,.0f}\n"
    )
    buf.write(
        f"Non-downturn fee-net PnL:  ${diag.contribution_metrics.non_downturn_net_pnl_dollars:+,.0f}\n"
    )
    buf.write(
        f"Downturn positive share:   {diag.contribution_metrics.downturn_positive_share:>7.1%}\n"
    )
    buf.write(
        f"Non-downturn positive share:{diag.contribution_metrics.non_downturn_positive_share:>6.1%}\n"
    )
    buf.write(
        f"RANGE_CHOP fee-net PnL:    ${diag.contribution_metrics.range_chop_net_pnl_dollars:+,.0f}\n"
    )
    buf.write(
        f"BULL_TREND fee-net PnL:    ${diag.contribution_metrics.bull_trend_net_pnl_dollars:+,.0f}\n"
    )
    buf.write("\n")

    buf.write("--- Robustness / Concentration ---\n")
    buf.write(
        f"Top 1 winner share of positive PnL: {diag.concentration_metrics.top_1_positive_share:>6.1%}\n"
    )
    buf.write(
        f"Top 5 winners share of positive PnL: {diag.concentration_metrics.top_5_positive_share:>5.1%}\n"
    )
    buf.write(
        f"Campaigns for 50% of positive PnL:   {diag.concentration_metrics.campaigns_for_half_positive_pnl}\n"
    )
    buf.write(
        f"Fee-net R quartiles:                "
        f"P25={diag.concentration_metrics.p25_fee_net_r:+.2f} "
        f"Median={diag.concentration_metrics.median_fee_net_r:+.2f} "
        f"P75={diag.concentration_metrics.p75_fee_net_r:+.2f}\n"
    )
    buf.write("\n")

    if diag.crisis_results:
        buf.write("--- Crisis Window Returns ---\n")
        buf.write(f"{'Crisis':<20} {'BRS%':>8} {'Campaigns':>10}\n")
        for entry in diag.crisis_results:
            buf.write(f"{entry.name:<20} {entry.brs_return_pct:>7.1f}% {entry.brs_trades:>10d}\n")

    return buf.getvalue()


def _to_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _naive_dt(value)
    try:
        return _naive_dt(pd.Timestamp(value).to_pydatetime())
    except Exception:
        pass
    return None


def _naive_dt(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value
