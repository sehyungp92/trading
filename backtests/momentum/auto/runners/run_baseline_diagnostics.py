"""Run v6 baseline portfolio with full diagnostics.

Usage:
    cd trading
    python -u backtests/momentum/auto/run_baseline_diagnostics.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.momentum.cli import (
    _load_helix_data,
    _load_nqdtc_data,
    _load_vdubus_data,
)
from backtests.momentum.config_helix import Helix4BacktestConfig
from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.config_portfolio import PortfolioBacktestConfig
from backtests.momentum.config_vdubus import VdubusBacktestConfig
from backtests.momentum.engine.helix_engine import Helix4Engine
from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
from backtests.momentum.engine.portfolio_engine import PortfolioBacktester
from backtests.momentum.engine.vdubus_engine import VdubusEngine
from backtests.momentum.auto.scoring import composite_score, extract_metrics

EQUITY = 10_000.0
DATA_DIR = ROOT / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "momentum" / "auto" / "output"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 72)
    print("  MOMENTUM V6 BASELINE - FULL DIAGNOSTICS")
    print("=" * 72)

    # ── Load data ──
    print("\n  Loading bar data...")
    helix_data = _load_helix_data("NQ", DATA_DIR)
    nqdtc_data = _load_nqdtc_data("NQ", DATA_DIR)
    vdubus_data = _load_vdubus_data("NQ", DATA_DIR)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ── Run engines ──
    print("\n  Running Helix4 engine...")
    t1 = time.time()
    helix_cfg = Helix4BacktestConfig(initial_equity=EQUITY, fixed_qty=10)
    helix_engine = Helix4Engine(symbol="NQ", bt_config=helix_cfg)
    helix_result = helix_engine.run(
        helix_data["minute_bars"], helix_data["hourly"], helix_data["four_hour"],
        helix_data["daily"], helix_data["hourly_idx_map"],
        helix_data["four_hour_idx_map"], helix_data["daily_idx_map"],
    )
    print(f"  Helix4: {len(helix_result.trades)} trades in {time.time()-t1:.1f}s")

    print("  Running NQDTC engine...")
    t1 = time.time()
    nqdtc_cfg = NQDTCBacktestConfig(symbols=["MNQ"], initial_equity=EQUITY, fixed_qty=10)
    nqdtc_engine = NQDTCEngine(symbol="MNQ", bt_config=nqdtc_cfg)
    nqdtc_result = nqdtc_engine.run(
        nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"], nqdtc_data["hourly"],
        nqdtc_data["four_hour"], nqdtc_data["daily"],
        nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
        nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
        daily_es=nqdtc_data.get("daily_es"),
        daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
    )
    print(f"  NQDTC:  {len(nqdtc_result.trades)} trades in {time.time()-t1:.1f}s")

    print("  Running Vdubus engine...")
    t1 = time.time()
    from backtests.momentum.config_vdubus import VdubusAblationFlags
    vdubus_cfg = VdubusBacktestConfig(
        initial_equity=EQUITY, fixed_qty=10,
        flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
    )
    vdubus_engine = VdubusEngine(symbol="NQ", bt_config=vdubus_cfg)
    vdubus_result = vdubus_engine.run(
        vdubus_data["bars_15m"], vdubus_data.get("bars_5m"), vdubus_data["hourly"],
        vdubus_data["daily_es"], vdubus_data["hourly_idx_map"],
        vdubus_data["daily_es_idx_map"], vdubus_data.get("five_to_15_idx_map"),
    )
    print(f"  Vdubus: {len(vdubus_result.trades)} trades in {time.time()-t1:.1f}s")

    # ── Portfolio backtester ──
    print("\n  Running portfolio backtester (v6 baseline)...")
    t1 = time.time()
    portfolio_cfg = PortfolioBacktestConfig()
    backtester = PortfolioBacktester(portfolio_cfg)
    portfolio_result = backtester.run(
        helix_result.trades, nqdtc_result.trades, vdubus_result.trades,
    )
    print(f"  Portfolio: {len(portfolio_result.trades)} approved, "
          f"{len(portfolio_result.blocked_trades)} blocked in {time.time()-t1:.1f}s")

    # ── Compute metrics ──
    init_eq = portfolio_cfg.portfolio.initial_equity
    eq = portfolio_result.equity_curve
    ts = np.array(portfolio_result.equity_timestamps, dtype='datetime64[ns]') if portfolio_result.equity_timestamps else np.array([])
    portfolio_metrics = extract_metrics(portfolio_result.trades, eq, ts, init_eq)
    portfolio_score = composite_score(portfolio_metrics, init_eq, strategy="portfolio", equity_curve=eq)

    # Per-strategy metrics
    strat_results = {
        "helix": (helix_result.trades, helix_result.equity_curve, helix_result.timestamps),
        "nqdtc": (nqdtc_result.trades, nqdtc_result.equity_curve, nqdtc_result.timestamps),
        "vdubus": (vdubus_result.trades, vdubus_result.equity_curve, vdubus_result.time_series),
    }
    strat_metrics = {}
    strat_scores = {}
    for name, (trades, ec, ts_arr) in strat_results.items():
        if ts_arr is not None and len(ts_arr) > 0:
            if hasattr(ts_arr[0], 'timestamp'):
                ts_numeric = np.array([dt.timestamp() for dt in ts_arr])
            else:
                ts_numeric = ts_arr
        else:
            ts_numeric = np.array([])
        m = extract_metrics(trades, ec, ts_numeric, EQUITY)
        s = composite_score(m, EQUITY, strategy=name, equity_curve=ec)
        strat_metrics[name] = m
        strat_scores[name] = s

    # ── Print comprehensive report ──
    lines = []
    lines.append("=" * 72)
    lines.append("  V6 BASELINE PORTFOLIO DIAGNOSTICS")
    lines.append("=" * 72)

    # A. Portfolio Overview
    lines.append("\n  A. PORTFOLIO OVERVIEW")
    lines.append("  " + "-" * 55)
    m = portfolio_metrics
    lines.append(f"    Composite Score:     {portfolio_score.total:.4f}")
    lines.append(f"      Calmar (0.20):     {portfolio_score.calmar_component:.4f}")
    lines.append(f"      PF (0.30):         {portfolio_score.pf_component:.4f}")
    lines.append(f"      Inv DD (0.15):     {portfolio_score.inv_dd_component:.4f}")
    lines.append(f"      Net Profit (0.35): {portfolio_score.net_profit_component:.4f}")
    if portfolio_score.rejected:
        lines.append(f"    REJECTED: {portfolio_score.reject_reason}")
    lines.append(f"    Total trades:        {m.total_trades}")
    lines.append(f"    Win rate:            {m.win_rate:.1%}")
    lines.append(f"    Profit factor:       {m.profit_factor:.2f}")
    lines.append(f"    Net profit:          ${m.net_profit:,.0f}")
    lines.append(f"    CAGR:                {m.cagr:.1%}")
    lines.append(f"    Sharpe:              {m.sharpe:.2f}")
    lines.append(f"    Sortino:             {m.sortino:.2f}")
    lines.append(f"    Calmar:              {m.calmar:.2f}")
    lines.append(f"    Max drawdown:        {m.max_drawdown_pct:.2%} (${m.max_drawdown_dollar:,.0f})")
    lines.append(f"    Expectancy:          {m.expectancy:.2f}R (${m.expectancy_dollar:.0f})")
    lines.append(f"    Trades/month:        {m.trades_per_month:.1f}")
    lines.append(f"    Avg hold hours:      {m.avg_hold_hours:.1f}")
    lines.append(f"    Total commissions:   ${m.total_commissions:,.0f}")
    lines.append(f"    Tail loss (5%):      ${m.tail_loss_pct:,.0f}")
    lines.append(f"    Tail loss R:         {m.tail_loss_r:.2f}R")

    # B. Per-strategy breakdown
    lines.append("\n  B. PER-STRATEGY BREAKDOWN")
    lines.append("  " + "-" * 55)
    lines.append(f"    {'Strategy':<10s} {'Trades':>6s} {'WR%':>6s} {'PF':>6s} {'Net$':>10s} {'DD%':>7s} {'Score':>7s}")
    lines.append("    " + "-" * 55)
    for name in ["helix", "nqdtc", "vdubus"]:
        sm = strat_metrics[name]
        ss = strat_scores[name]
        lines.append(
            f"    {name:<10s} {sm.total_trades:>6d} {sm.win_rate:>5.1%} "
            f"{sm.profit_factor:>6.2f} ${sm.net_profit:>+9,.0f} "
            f"{sm.max_drawdown_pct:>6.2%} {ss.total:>7.4f}"
        )
    lines.append(
        f"    {'PORTFOLIO':<10s} {m.total_trades:>6d} {m.win_rate:>5.1%} "
        f"{m.profit_factor:>6.2f} ${m.net_profit:>+9,.0f} "
        f"{m.max_drawdown_pct:>6.2%} {portfolio_score.total:>7.4f}"
    )

    # C. Strategy contribution
    lines.append("\n  C. STRATEGY CONTRIBUTION")
    lines.append("  " + "-" * 55)
    total_pnl = sum(sm.net_profit for sm in strat_metrics.values())
    for name in ["helix", "nqdtc", "vdubus"]:
        sm = strat_metrics[name]
        pct = sm.net_profit / total_pnl * 100 if total_pnl else 0
        lines.append(f"    {name:<10s}: ${sm.net_profit:>+9,.0f} ({pct:>5.1f}% of raw PnL)")
    lines.append(f"    {'Raw total':<10s}: ${total_pnl:>+9,.0f}")
    lines.append(f"    {'Portfolio':<10s}: ${m.net_profit:>+9,.0f} (after blocking rules)")
    blocked_pnl = total_pnl - m.net_profit
    lines.append(f"    {'Blocked':<10s}: ${blocked_pnl:>+9,.0f}")

    # D. Rule blocks
    lines.append("\n  D. PORTFOLIO RULE BLOCKS")
    lines.append("  " + "-" * 55)
    if portfolio_result.rule_blocks:
        lines.append(f"    {'Rule':<35s} {'Count':>6s} {'Blocked PnL':>12s}")
        lines.append("    " + "-" * 55)
        for rule, count in sorted(portfolio_result.rule_blocks.items(), key=lambda x: -x[1]):
            blocked_pnl_rule = portfolio_result.rule_blocked_pnl.get(rule, 0)
            lines.append(f"    {rule:<35s} {count:>6d} ${blocked_pnl_rule:>+11,.0f}")
    else:
        lines.append("    No rule blocks recorded.")

    # E. Win/Loss distribution
    lines.append("\n  E. WIN/LOSS DISTRIBUTION")
    lines.append("  " + "-" * 55)
    pnls = np.array([getattr(t, 'adjusted_pnl', getattr(t, 'pnl_dollars', 0.0)) for t in portfolio_result.trades])
    if len(pnls) > 0:
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        lines.append(f"    Winners:  {len(wins):>5d}  avg=${np.mean(wins):>+,.0f}  max=${np.max(wins):>+,.0f}")
        if len(losses) > 0:
            lines.append(f"    Losers:   {len(losses):>5d}  avg=${np.mean(losses):>+,.0f}  max=${np.min(losses):>+,.0f}")
        lines.append(f"    Avg win/avg loss ratio: {abs(np.mean(wins)/np.mean(losses)):.2f}" if len(losses) > 0 else "")

        # R-multiple distribution
        r_mults = []
        for t in portfolio_result.trades:
            risk = abs(getattr(t, 'entry_price', 0) - getattr(t, 'initial_stop', 0))
            pv = getattr(t, 'point_value', 2.0)
            qty = abs(getattr(t, 'portfolio_qty', getattr(t, 'qty', 1)))
            risk_d = risk * pv * qty if risk > 0 else 1.0
            r_mults.append(getattr(t, 'adjusted_pnl', getattr(t, 'pnl_dollars', 0.0)) / risk_d)
        r_arr = np.array(r_mults)
        lines.append(f"    R-distribution: mean={np.mean(r_arr):.2f}R, "
                     f"median={np.median(r_arr):.2f}R, "
                     f"std={np.std(r_arr):.2f}R")
        lines.append(f"    Big winners (>2R):  {np.sum(r_arr > 2)}")
        lines.append(f"    Big losers (<-1R):  {np.sum(r_arr < -1)}")

    # F. Direction analysis
    lines.append("\n  F. DIRECTION ANALYSIS")
    lines.append("  " + "-" * 55)
    for direction, dir_val in [("LONG", 1), ("SHORT", -1)]:
        dir_trades = [t for t in portfolio_result.trades
                      if getattr(t, 'direction', 0) == dir_val]
        if dir_trades:
            dir_pnl = np.array([getattr(t, 'adjusted_pnl', getattr(t, 'pnl_dollars', 0.0)) for t in dir_trades])
            wr = float(np.mean(dir_pnl > 0)) * 100
            lines.append(f"    {direction:<6s}: {len(dir_trades):>4d} trades, "
                        f"WR={wr:.0f}%, net=${np.sum(dir_pnl):>+,.0f}, "
                        f"avg=${np.mean(dir_pnl):>+,.0f}")

    # G. Entry class breakdown
    lines.append("\n  G. ENTRY CLASS BREAKDOWN")
    lines.append("  " + "-" * 55)
    from collections import defaultdict
    ec_stats = defaultdict(lambda: {"count": 0, "pnls": []})
    for t in portfolio_result.trades:
        ec = getattr(t, 'strategy_id', getattr(t, 'entry_class', 'unknown'))
        ec_stats[ec]["count"] += 1
        ec_stats[ec]["pnls"].append(getattr(t, 'adjusted_pnl', getattr(t, 'pnl_dollars', 0.0)))
    lines.append(f"    {'Class':<25s} {'Count':>5s} {'WR%':>6s} {'Net$':>10s} {'Avg$':>10s}")
    lines.append("    " + "-" * 55)
    for ec, data in sorted(ec_stats.items(), key=lambda x: -sum(x[1]["pnls"])):
        arr = np.array(data["pnls"])
        wr = float(np.mean(arr > 0)) * 100
        lines.append(f"    {ec:<25s} {data['count']:>5d} {wr:>5.1f}% ${np.sum(arr):>+9,.0f} ${np.mean(arr):>+9,.0f}")

    report = "\n".join(lines)
    print(report)

    # ── Save report ──
    report_path = OUTPUT_DIR / "v6_baseline_diagnostics.txt"
    report_path.write_text(report)
    print(f"\n  Report saved to: {report_path}")

    # ── Save structured data ──
    baseline_data = {
        "score": portfolio_score.total,
        "score_components": {
            "calmar": portfolio_score.calmar_component,
            "pf": portfolio_score.pf_component,
            "inv_dd": portfolio_score.inv_dd_component,
            "net_profit": portfolio_score.net_profit_component,
        },
        "portfolio_metrics": {
            "total_trades": m.total_trades,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "net_profit": m.net_profit,
            "cagr": m.cagr,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "calmar": m.calmar,
            "max_drawdown_pct": m.max_drawdown_pct,
            "max_drawdown_dollar": m.max_drawdown_dollar,
            "expectancy": m.expectancy,
            "expectancy_dollar": m.expectancy_dollar,
            "trades_per_month": m.trades_per_month,
            "avg_hold_hours": m.avg_hold_hours,
            "total_commissions": m.total_commissions,
        },
        "per_strategy": {
            name: {
                "trades": strat_metrics[name].total_trades,
                "win_rate": strat_metrics[name].win_rate,
                "profit_factor": strat_metrics[name].profit_factor,
                "net_profit": strat_metrics[name].net_profit,
                "max_drawdown_pct": strat_metrics[name].max_drawdown_pct,
                "score": strat_scores[name].total,
            }
            for name in ["helix", "nqdtc", "vdubus"]
        },
        "rule_blocks": dict(portfolio_result.rule_blocks) if portfolio_result.rule_blocks else {},
        "rule_blocked_pnl": {k: float(v) for k, v in portfolio_result.rule_blocked_pnl.items()} if portfolio_result.rule_blocked_pnl else {},
    }
    json_path = OUTPUT_DIR / "v6_baseline.json"
    json_path.write_text(json.dumps(baseline_data, indent=2, default=str))
    print(f"  JSON saved to: {json_path}")

    # ── Run analysis modules ──
    print("\n" + "=" * 72)
    print("  RUNNING ANALYSIS MODULES")
    print("=" * 72)

    all_trades = portfolio_result.trades
    strategies_dict = {
        "helix": helix_result.trades,
        "nqdtc": nqdtc_result.trades,
        "vdubus": vdubus_result.trades,
        "PORTFOLIO": portfolio_result.trades,
    }

    analysis_reports = {}

    # 1. Entry timing
    try:
        from backtests.momentum.analysis.entry_timing_optimization import generate_entry_timing_report
        r = generate_entry_timing_report(all_trades)
        analysis_reports["entry_timing"] = r
        print("  [OK] Entry timing optimization")
    except Exception as e:
        print(f"  [FAIL] Entry timing: {e}")

    # 2. Loss streak
    try:
        from backtests.momentum.analysis.loss_streak_analysis import generate_loss_streak_report
        r = generate_loss_streak_report(all_trades, strategies=strategies_dict)
        analysis_reports["loss_streak"] = r
        print("  [OK] Loss streak analysis")
    except Exception as e:
        print(f"  [FAIL] Loss streak: {e}")

    # 3. Session profitability
    try:
        from backtests.momentum.analysis.session_profitability import generate_session_profitability_report
        r = generate_session_profitability_report(all_trades)
        analysis_reports["session_profitability"] = r
        print("  [OK] Session profitability")
    except Exception as e:
        print(f"  [FAIL] Session profitability: {e}")

    # 4. Drawdown attribution
    try:
        from backtests.momentum.analysis.drawdown_attribution import generate_drawdown_report
        r = generate_drawdown_report(all_trades, portfolio_result.equity_curve)
        analysis_reports["drawdown_attribution"] = r
        print("  [OK] Drawdown attribution")
    except Exception as e:
        print(f"  [FAIL] Drawdown attribution: {e}")

    # 5. Cross-strategy correlation
    try:
        from backtests.momentum.analysis.cross_strategy_correlation import generate_correlation_report
        r = generate_correlation_report(strategies_dict)
        analysis_reports["cross_strategy_correlation"] = r
        print("  [OK] Cross-strategy correlation")
    except Exception as e:
        print(f"  [FAIL] Cross-strategy correlation: {e}")

    # 6. Capital efficiency
    try:
        from backtests.momentum.analysis.capital_efficiency import generate_capital_efficiency_report
        r = generate_capital_efficiency_report(all_trades, EQUITY)
        analysis_reports["capital_efficiency"] = r
        print("  [OK] Capital efficiency")
    except Exception as e:
        print(f"  [FAIL] Capital efficiency: {e}")

    # 7. Seasonality
    try:
        from backtests.momentum.analysis.seasonality_calendar import generate_seasonality_report
        r = generate_seasonality_report(all_trades)
        analysis_reports["seasonality"] = r
        print("  [OK] Seasonality calendar")
    except Exception as e:
        print(f"  [FAIL] Seasonality: {e}")

    # 8. Weakness report
    try:
        from backtests.momentum.analysis.weakness_report import generate_weakness_report
        r = generate_weakness_report(all_trades, strategies=strategies_dict)
        analysis_reports["weakness_report"] = r
        print("  [OK] Weakness report")
    except Exception as e:
        print(f"  [FAIL] Weakness report: {e}")

    # 9. Portfolio diagnostics
    try:
        from backtests.momentum.analysis.portfolio_diagnostics import generate_portfolio_diagnostics
        r = generate_portfolio_diagnostics(
            portfolio_result, helix_result, nqdtc_result, vdubus_result,
            initial_equity=EQUITY,
        )
        analysis_reports["portfolio_diagnostics"] = r
        print("  [OK] Portfolio diagnostics")
    except Exception as e:
        print(f"  [FAIL] Portfolio diagnostics: {e}")

    # 10. Signal conflict
    try:
        from backtests.momentum.analysis.signal_conflict import generate_signal_conflict_report
        r = generate_signal_conflict_report(strategies_dict)
        analysis_reports["signal_conflict"] = r
        print("  [OK] Signal conflict")
    except Exception as e:
        print(f"  [FAIL] Signal conflict: {e}")

    # 11. Overnight gap
    try:
        from backtests.momentum.analysis.overnight_gap import generate_overnight_gap_report
        r = generate_overnight_gap_report(all_trades)
        analysis_reports["overnight_gap"] = r
        print("  [OK] Overnight gap")
    except Exception as e:
        print(f"  [FAIL] Overnight gap: {e}")

    # Save all analysis reports
    full_report = "\n\n".join(analysis_reports.values())
    analysis_path = OUTPUT_DIR / "v6_baseline_analysis.txt"
    analysis_path.write_text(full_report, encoding="utf-8")
    print(f"\n  Full analysis saved to: {analysis_path}")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
