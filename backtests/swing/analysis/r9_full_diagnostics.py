"""R9 BRS diagnostics on the synchronized, fee-net campaign engine."""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing._aliases import install

install()

from backtest.config_brs import BRSConfig
from backtest.engine.brs_portfolio_engine import load_brs_data, run_brs_synchronized
from backtests.swing.analysis.brs_diagnostics import CRISIS_WINDOWS, compute_brs_diagnostics
from backtests.swing.analysis.brs_trade_metrics import summarize_brs_campaigns
from backtests.swing.auto.brs.config_mutator import mutate_brs_config
from backtests.swing.auto.brs.scoring import extract_brs_metrics

R9_MUTATIONS = {}
DATA_DIR = Path("backtests/swing/data/raw")
INITIAL_EQUITY = 10_000.0


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "auto" / "brs" / "output" / "r9_full_diagnostics.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = mutate_brs_config(BRSConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR), R9_MUTATIONS)
    data = load_brs_data(config)
    result = run_brs_synchronized(data, config)

    all_trades = []
    for symbol_result in result.symbol_results.values():
        all_trades.extend(symbol_result.trades)
    campaigns = summarize_brs_campaigns(all_trades)
    metrics = extract_brs_metrics(result, INITIAL_EQUITY)
    diagnostics = compute_brs_diagnostics(
        result.symbol_results,
        INITIAL_EQUITY,
        combined_equity=result.combined_equity,
        combined_timestamps=result.combined_timestamps,
    )

    lines = []
    _out(lines, "=" * 80)
    _out(lines, "BRS R9 FULL DIAGNOSTICS (SYNCHRONIZED / FEE-NET)")
    _out(lines, "=" * 80)
    _out(lines, "")
    _out(lines, f"Campaigns: {metrics.campaign_count}")
    _out(lines, f"Realized legs: {metrics.realized_leg_count}")
    _out(lines, f"Net return: {metrics.net_return_pct:.1f}%")
    _out(lines, f"Fee-net PnL: ${metrics.total_net_pnl_dollars:,.2f}")
    _out(lines, f"Profit factor: {metrics.profit_factor:.2f}")
    _out(lines, f"Max drawdown: {metrics.max_dd_pct:.1%}")
    _out(lines, f"Downturn share: {metrics.downturn_net_pnl_share:.0%}")
    _out(lines, f"Non-downturn share: {metrics.non_downturn_net_pnl_share:.0%}")
    _out(lines, "")

    _out(lines, "A) Per-Symbol Campaign Summary")
    _out(lines, "-" * 80)
    by_symbol = defaultdict(list)
    for campaign in campaigns:
        by_symbol[campaign.symbol].append(campaign)
    for symbol in sorted(by_symbol):
        _out(lines, _campaign_summary_line(symbol, by_symbol[symbol]))
    _out(lines, "")

    _out(lines, "B) Per-Regime Campaign Summary")
    _out(lines, "-" * 80)
    by_regime = defaultdict(list)
    for campaign in campaigns:
        by_regime[campaign.regime_entry or "UNKNOWN"].append(campaign)
    for regime in sorted(by_regime):
        _out(lines, _campaign_summary_line(regime, by_regime[regime]))
    _out(lines, "")

    _out(lines, "C) Per-Entry-Type Campaign Summary")
    _out(lines, "-" * 80)
    by_entry = defaultdict(list)
    for campaign in campaigns:
        by_entry[campaign.entry_type or "UNKNOWN"].append(campaign)
    for entry_type in sorted(by_entry):
        _out(lines, _campaign_summary_line(entry_type, by_entry[entry_type]))
    _out(lines, "")

    _out(lines, "D) Crisis Window Campaign Coverage")
    _out(lines, "-" * 80)
    for name, crisis_start, crisis_end in CRISIS_WINDOWS:
        crisis_campaigns = [
            campaign
            for campaign in campaigns
            if campaign.entry_time is not None and crisis_start <= _naive_dt(campaign.entry_time) <= crisis_end
        ]
        first_entry = min((_naive_dt(campaign.entry_time) for campaign in crisis_campaigns if campaign.entry_time), default=None)
        first_str = first_entry.strftime("%Y-%m-%d %H:%M") if first_entry else "N/A"
        pnl = sum(campaign.fee_net_pnl_dollars for campaign in crisis_campaigns)
        _out(
            lines,
            f"{name:25s} campaigns={len(crisis_campaigns):3d} fee_net=${pnl:+,.0f} first_entry={first_str}",
        )
    _out(lines, "")

    _out(lines, "E) Diagnostics Snapshot")
    _out(lines, "-" * 80)
    for entry in diagnostics.regime_metrics:
        _out(
            lines,
            f"Regime {entry.regime:15s} count={entry.trade_count:3d} WR={entry.win_rate:5.1f}% "
            f"avgR={entry.avg_r:+.2f} PF={entry.profit_factor:.2f}",
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved diagnostics to {output_path}")


def _campaign_summary_line(label: str, campaigns) -> str:
    count = len(campaigns)
    if count == 0:
        return f"{label:25s} campaigns=  0"
    wins = [campaign for campaign in campaigns if campaign.fee_net_pnl_dollars > 0]
    losses = [campaign for campaign in campaigns if campaign.fee_net_pnl_dollars <= 0]
    gross_win = sum(campaign.fee_net_pnl_dollars for campaign in wins)
    gross_loss = abs(sum(campaign.fee_net_pnl_dollars for campaign in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    avg_r = sum(campaign.fee_net_r_multiple for campaign in campaigns) / count
    total_r = sum(campaign.fee_net_r_multiple for campaign in campaigns)
    pnl = sum(campaign.fee_net_pnl_dollars for campaign in campaigns)
    wr = len(wins) / count * 100.0
    return (
        f"{label:25s} campaigns={count:3d} WR={wr:5.1f}% "
        f"avgR={avg_r:+.2f} totR={total_r:+.2f} PF={pf:.2f} fee_net=${pnl:+,.0f}"
    )


def _out(lines: list[str], text: str) -> None:
    print(text)
    lines.append(text)


def _naive_dt(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value


if __name__ == "__main__":
    main()
