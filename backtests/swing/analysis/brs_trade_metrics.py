"""Shared fee-net trade and campaign helpers for BRS analysis."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from backtests.swing.engine.backtest_engine import TradeRecord

BEAR_REGIMES = {"BEAR_STRONG", "BEAR_TREND", "BEAR_FORMING"}
NON_DOWNTURN_REGIMES = {"RANGE_CHOP", "BULL_TREND"}


@dataclass(frozen=True)
class BRSCampaignSummary:
    campaign_id: str
    symbol: str
    direction: int
    entry_type: str
    regime_entry: str
    terminal_exit_reason: str
    signal_time: datetime | None
    fill_time: datetime | None
    entry_time: datetime | None
    exit_time: datetime | None
    qty: int
    realized_leg_count: int
    gross_pnl_dollars: float
    fee_net_pnl_dollars: float
    gross_r_multiple: float
    fee_net_r_multiple: float
    commission: float
    bars_held: int
    mfe_r: float
    mae_r: float
    score_entry: float


def trade_risk_dollars(trade: TradeRecord) -> float:
    if trade.qty <= 0:
        return 0.0
    return abs(trade.entry_price - trade.initial_stop) * trade.qty


def trade_fee_net_pnl(trade: TradeRecord) -> float:
    return trade.pnl_dollars - trade.commission


def trade_fee_net_r_multiple(trade: TradeRecord) -> float:
    risk_dollars = trade_risk_dollars(trade)
    if risk_dollars <= 0:
        return 0.0
    return trade_fee_net_pnl(trade) / risk_dollars


def summarize_brs_campaigns(trades: list[TradeRecord]) -> list[BRSCampaignSummary]:
    buckets: dict[object, list[TradeRecord]] = {}
    for idx, trade in enumerate(trades):
        key = _campaign_key(trade, idx)
        buckets.setdefault(key, []).append(trade)

    summaries: list[BRSCampaignSummary] = []
    for key, bucket in buckets.items():
        ordered = sorted(bucket, key=lambda trade: (_sort_dt(trade.entry_time), _sort_dt(trade.exit_time)))
        first = ordered[0]
        last = ordered[-1]
        total_risk = sum(trade_risk_dollars(trade) for trade in ordered)
        gross_pnl = sum(trade.pnl_dollars for trade in ordered)
        fee_net_pnl = sum(trade_fee_net_pnl(trade) for trade in ordered)

        summaries.append(
            BRSCampaignSummary(
                campaign_id=str(key),
                symbol=first.symbol,
                direction=first.direction,
                entry_type=first.entry_type,
                regime_entry=first.regime_entry,
                terminal_exit_reason=last.exit_reason,
                signal_time=_min_dt(getattr(trade, "signal_time", None) for trade in ordered),
                fill_time=_min_dt(getattr(trade, "fill_time", None) for trade in ordered),
                entry_time=_min_dt(trade.entry_time for trade in ordered),
                exit_time=_max_dt(trade.exit_time for trade in ordered),
                qty=sum(max(0, trade.qty) for trade in ordered),
                realized_leg_count=len(ordered),
                gross_pnl_dollars=gross_pnl,
                fee_net_pnl_dollars=fee_net_pnl,
                gross_r_multiple=(gross_pnl / total_risk) if total_risk > 0 else 0.0,
                fee_net_r_multiple=(fee_net_pnl / total_risk) if total_risk > 0 else 0.0,
                commission=sum(trade.commission for trade in ordered),
                bars_held=max((trade.bars_held for trade in ordered), default=0),
                mfe_r=max((trade.mfe_r for trade in ordered), default=0.0),
                mae_r=max((trade.mae_r for trade in ordered), default=0.0),
                score_entry=first.score_entry,
            )
        )

    return sorted(summaries, key=lambda summary: (_sort_dt(summary.entry_time), summary.campaign_id))


def _campaign_key(trade: TradeRecord, idx: int) -> object:
    if getattr(trade, "campaign_id", ""):
        return trade.campaign_id
    signal_time = getattr(trade, "signal_time", None) or trade.entry_time
    fill_time = getattr(trade, "fill_time", None) or trade.entry_time
    return (
        trade.symbol,
        _dt_key(signal_time),
        _dt_key(fill_time),
        round(trade.entry_price, 6),
        round(trade.initial_stop, 6),
        trade.entry_type,
        trade.regime_entry,
    )


def _dt_key(value: datetime | None) -> str:
    if value is None:
        return ""
    return _normalize_dt(value).isoformat()


def _sort_dt(value: datetime | None) -> datetime:
    return _normalize_dt(value)


def _min_dt(values) -> datetime | None:
    valid = [_normalize_dt(value) for value in values if value is not None]
    return min(valid) if valid else None


def _max_dt(values) -> datetime | None:
    valid = [_normalize_dt(value) for value in values if value is not None]
    return max(valid) if valid else None


def _normalize_dt(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min
    return value.replace(tzinfo=None) if value.tzinfo else value
