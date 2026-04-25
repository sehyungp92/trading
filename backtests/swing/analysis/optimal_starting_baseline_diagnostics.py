"""Full diagnostics for the truthful BRS starting baseline."""
from __future__ import annotations

import json
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
from backtests.swing.auto.brs.scoring import composite_score, extract_brs_metrics

DATA_DIR = Path("backtests/swing/data/raw")
INITIAL_EQUITY = 100_000.0
OUTPUT_DIR = Path("backtests/swing/auto/brs/output")
OUTPUT_PATH = OUTPUT_DIR / "optimal_starting_baseline_full_diagnostics.txt"
MUTATIONS_PATH = OUTPUT_DIR / "optimal_starting_baseline_mutations.json"

OPTIMAL_STARTING_BASELINE_MUTATIONS = {
    "disable_s1": False,
    "symbol_configs.GLD.adx_on": 14,
    "symbol_configs.GLD.adx_off": 12,
    "scale_out_enabled": True,
    "scale_out_pct": 0.33,
    "bd_donchian_period": 10,
    "lh_swing_lookback": 5,
    "scale_out_target_r": 3.6,
    "symbol_configs.QQQ.stop_floor_atr": 0.6,
    "symbol_configs.GLD.stop_buffer_atr": 0.18,
    "profit_floor_scale": 4.1472,
    "peak_drop_enabled": True,
    "peak_drop_pct": -0.02,
    "peak_drop_lookback": 16,
    "bd_arm_bars": 24,
    "bias_4h_accel_enabled": True,
    "chop_short_entry_enabled": True,
    "persistence_override_bars": 7,
    "adx_strong": 25,
    "lh_arm_bars": 26,
    "bt_volume_mult": 2.0,
    "symbol_configs.QQQ.stop_buffer_atr": 0.3,
    "persist_quality_mult_bd": 0.96,
    "min_hold_bars": 5,
    "chop_quality_mult": 0.84,
    "stop_floor_bear_strong_mult": 1.5,
    "bd_max_stop_atr": 4.2,
    "be_trigger_r": 0.75,
    "param_overrides.extreme_vol_pct": 90,
    "pyramid_enabled": False,
    "size_mult_bear_trend": 1.3,
    "symbol_configs.QQQ.base_risk_pct": 0.0035,
    "symbol_configs.GLD.base_risk_pct": 0.005,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MUTATIONS_PATH.write_text(
        json.dumps(OPTIMAL_STARTING_BASELINE_MUTATIONS, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = mutate_brs_config(
        BRSConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR),
        OPTIMAL_STARTING_BASELINE_MUTATIONS,
    )
    data = load_brs_data(config)
    result = run_brs_synchronized(data, config)

    all_trades = []
    for symbol_result in result.symbol_results.values():
        all_trades.extend(symbol_result.trades)
    campaigns = summarize_brs_campaigns(all_trades)
    by_symbol = defaultdict(list)
    by_regime = defaultdict(list)
    by_entry = defaultdict(list)
    for campaign in campaigns:
        by_symbol[campaign.symbol].append(campaign)
        by_regime[campaign.regime_entry or "UNKNOWN"].append(campaign)
        by_entry[campaign.entry_type or "UNKNOWN"].append(campaign)

    metrics = extract_brs_metrics(result, INITIAL_EQUITY)
    score = composite_score(metrics)
    diagnostics = compute_brs_diagnostics(
        result.symbol_results,
        INITIAL_EQUITY,
        combined_equity=result.combined_equity,
        combined_timestamps=result.combined_timestamps,
    )

    lines: list[str] = []
    _out(lines, "=" * 88)
    _out(lines, "BRS OPTIMAL STARTING BASELINE FULL DIAGNOSTICS (TRUTHFUL / SYNCHRONIZED / FEE-NET)")
    _out(lines, "=" * 88)
    _out(lines, "")
    _out(lines, "Baseline intent: maximize expected return and campaign frequency with a downturn/correction bias,")
    _out(lines, "while keeping the truthful broker-style synchronized engine and avoiding unnecessary complexity.")
    _out(lines, "")

    _out(lines, "A) Topline")
    _out(lines, "-" * 88)
    _out(lines, f"Campaigns: {metrics.campaign_count}")
    _out(lines, f"Realized legs: {metrics.realized_leg_count}")
    _out(lines, f"Net return: {metrics.net_return_pct:.1f}%")
    _out(lines, f"Fee-net PnL: ${metrics.total_net_pnl_dollars:,.2f}")
    _out(lines, f"Profit factor: {metrics.profit_factor:.2f}")
    _out(lines, f"Max drawdown: {metrics.max_dd_pct:.1%}")
    _out(lines, f"Downturn share: {metrics.downturn_net_pnl_share:.0%}")
    _out(lines, f"Non-downturn share: {metrics.non_downturn_net_pnl_share:.0%}")
    _out(lines, f"Bear alpha: {metrics.bear_alpha_pct:.1f}%")
    _out(lines, f"Crisis coverage: {metrics.crisis_coverage:.0%}")
    _out(lines, f"Composite score: {score.total:.4f}")
    if score.rejected:
        _out(lines, f"Score gate: REJECTED ({score.reject_reason})")
    _out(lines, "")

    _out(lines, "B) Strength / Weakness Snapshot")
    _out(lines, "-" * 88)
    strongest_regime, strongest_regime_pnl = _best_bucket(by_regime)
    strongest_entry, strongest_entry_pnl = _best_bucket(by_entry)
    weakest_conviction = min(
        (bucket for bucket in diagnostics.conviction_buckets if bucket.trade_count > 0),
        key=lambda bucket: bucket.expectancy,
        default=None,
    )
    weakest_exit = min(
        diagnostics.exit_reason_metrics,
        key=lambda bucket: bucket.fee_net_pnl_dollars,
        default=None,
    )
    if strongest_regime is not None:
        _out(lines, f"Strongest regime driver: {strongest_regime} fee_net=${strongest_regime_pnl:+,.0f}")
    if strongest_entry is not None:
        _out(lines, f"Strongest entry driver: {strongest_entry} fee_net=${strongest_entry_pnl:+,.0f}")
    if weakest_conviction is not None:
        _out(
            lines,
            f"Weakest conviction bucket: {weakest_conviction.bucket} expectancy={weakest_conviction.expectancy:+.3f}",
        )
    if weakest_exit is not None:
        _out(
            lines,
            f"Biggest exit drag: {weakest_exit.exit_reason} fee_net=${weakest_exit.fee_net_pnl_dollars:+,.0f} "
            f"across {weakest_exit.trade_count} campaigns",
        )
    _out(
        lines,
        f"Positive-PnL concentration: top1={diagnostics.concentration_metrics.top_1_positive_share:.1%} "
        f"top5={diagnostics.concentration_metrics.top_5_positive_share:.1%}",
    )
    _out(
        lines,
        f"Bias alignment: downturn_positive_share={diagnostics.contribution_metrics.downturn_positive_share:.1%} "
        f"non_downturn_positive_share={diagnostics.contribution_metrics.non_downturn_positive_share:.1%}",
    )
    _out(lines, "")

    _out(lines, "C) Locked Baseline Mutations")
    _out(lines, "-" * 88)
    for key, value in sorted(OPTIMAL_STARTING_BASELINE_MUTATIONS.items()):
        _out(lines, f"{key} = {value}")
    _out(lines, "")

    _out(lines, "D) Per-Symbol Campaign Summary")
    _out(lines, "-" * 88)
    for symbol in sorted(by_symbol):
        _out(lines, _campaign_summary_line(symbol, by_symbol[symbol]))
    _out(lines, "")

    _out(lines, "E) Per-Regime Campaign Summary")
    _out(lines, "-" * 88)
    for regime in sorted(by_regime):
        _out(lines, _campaign_summary_line(regime, by_regime[regime]))
    _out(lines, "")

    _out(lines, "F) Per-Entry-Type Campaign Summary")
    _out(lines, "-" * 88)
    for entry_type in sorted(by_entry):
        _out(lines, _campaign_summary_line(entry_type, by_entry[entry_type]))
    _out(lines, "")

    _out(lines, "G) Crisis Window Campaign Coverage")
    _out(lines, "-" * 88)
    for name, crisis_start, crisis_end in CRISIS_WINDOWS:
        crisis_campaigns = [
            campaign
            for campaign in campaigns
            if campaign.entry_time is not None and crisis_start <= _naive_dt(campaign.entry_time) <= crisis_end
        ]
        first_entry = min(
            (_naive_dt(campaign.entry_time) for campaign in crisis_campaigns if campaign.entry_time is not None),
            default=None,
        )
        first_str = first_entry.strftime("%Y-%m-%d %H:%M") if first_entry else "N/A"
        pnl = sum(campaign.fee_net_pnl_dollars for campaign in crisis_campaigns)
        _out(
            lines,
            f"{name:25s} campaigns={len(crisis_campaigns):3d} fee_net=${pnl:+,.0f} first_entry={first_str}",
        )
    _out(lines, "")

    _out(lines, "H) Detailed BRS Diagnostics")
    _out(lines, "-" * 88)
    lines.extend(diagnostics.report.rstrip().splitlines())

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved diagnostics to {OUTPUT_PATH}")
    print(f"Saved mutations to {MUTATIONS_PATH}")


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


def _best_bucket(buckets: dict[str, list]) -> tuple[str | None, float]:
    best_label = None
    best_pnl = float("-inf")
    for label, campaigns in buckets.items():
        pnl = sum(campaign.fee_net_pnl_dollars for campaign in campaigns)
        if pnl > best_pnl:
            best_label = label
            best_pnl = pnl
    if best_label is None:
        return None, 0.0
    return best_label, best_pnl


def _out(lines: list[str], text: str) -> None:
    print(text)
    lines.append(text)


def _naive_dt(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value


if __name__ == "__main__":
    main()
