"""R9 BRS Full Diagnostics -- comprehensive analysis of Round 9 mutations."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Setup path and aliases
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing._aliases import install
install()

from backtest.config_brs import BRSConfig
from backtest.engine.brs_portfolio_engine import load_brs_data, run_brs_independent
from backtests.swing.auto.brs.config_mutator import mutate_brs_config

# ---------------------------------------------------------------------------
# R9 cumulative mutations
# ---------------------------------------------------------------------------
# R9 leverage-capped: all optimized values are now BRSConfig defaults.
# No mutations needed — BRSConfig() produces the recommended profile.
R9_MUTATIONS = {}

DATA_DIR = Path("backtests/swing/data/raw")
INITIAL_EQUITY = 10_000.0

CRISIS_WINDOWS = [
    ("2022 Bear", datetime(2022, 1, 3), datetime(2022, 10, 13)),
    ("SVB", datetime(2023, 3, 8), datetime(2023, 3, 15)),
    ("Aug 2024 Unwind", datetime(2024, 8, 1), datetime(2024, 8, 5)),
    ("Tariff Shock", datetime(2025, 2, 21), datetime(2025, 4, 7)),
    ("Mar 2026 Slow Burn", datetime(2026, 3, 5), datetime(2026, 3, 27)),
]

BEAR_REGIMES = {"BEAR_STRONG", "BEAR_TREND", "BEAR_FORMING"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pf(wins_total: float, losses_total: float) -> float:
    if losses_total == 0:
        return float("inf") if wins_total > 0 else 0.0
    return wins_total / abs(losses_total)


def _wr(trades: list) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.r_multiple > 0) / len(trades) * 100


def _avg_r(trades: list) -> float:
    if not trades:
        return 0.0
    return sum(t.r_multiple for t in trades) / len(trades)


def _total_r(trades: list) -> float:
    return sum(t.r_multiple for t in trades)


def _compute_pf(trades: list) -> float:
    wins = sum(t.r_multiple for t in trades if t.r_multiple > 0)
    losses = sum(t.r_multiple for t in trades if t.r_multiple < 0)
    return _pf(wins, losses)


def _trade_time(t, attr: str) -> datetime | None:
    val = getattr(t, attr, None)
    if val is None:
        return None
    if hasattr(val, "astype"):
        try:
            return val.astype("datetime64[s]").astype(datetime)
        except Exception:
            pass
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo else val
    return None


def _print_trade_table(trades: list, label: str, f=None) -> None:
    n = len(trades)
    if n == 0:
        line = f"  {label:30s}  n={n:3d}  (no trades)"
    else:
        wr = _wr(trades)
        ar = _avg_r(trades)
        tr = _total_r(trades)
        pf = _compute_pf(trades)
        line = f"  {label:30s}  n={n:3d}  WR={wr:5.1f}%  avgR={ar:+.2f}  totR={tr:+.2f}  PF={pf:.2f}"
    print(line)
    if f:
        f.write(line + "\n")


def _out(text: str, f=None) -> None:
    print(text)
    if f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path = Path(__file__).resolve().parent.parent / "auto" / "brs" / "output" / "r9_full_diagnostics.txt"

    config = BRSConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR)
    config = mutate_brs_config(config, R9_MUTATIONS)
    data = load_brs_data(config)
    result = run_brs_independent(data, config)

    all_trades = []
    crisis_logs: dict[str, list[dict]] = {}
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            t.symbol = sym
            all_trades.append(t)
        if hasattr(sr, "crisis_state_log"):
            crisis_logs[sym] = sr.crisis_state_log

    all_trades.sort(key=lambda t: _trade_time(t, "entry_time") or datetime.min)

    with open(output_path, "w", encoding="utf-8") as f:

        _out("=" * 80, f)
        _out("BRS R9 FULL DIAGNOSTICS", f)
        _out("=" * 80, f)
        _out("", f)

        _out(f"Total trades: {len(all_trades)}", f)
        total_pnl = sum(t.pnl_dollars for t in all_trades)
        total_r = _total_r(all_trades)
        _out(f"Total PnL: ${total_pnl:,.2f}", f)
        _out(f"Total R: {total_r:+.2f}", f)
        _out(f"Win Rate: {_wr(all_trades):.1f}%", f)
        _out(f"Profit Factor: {_compute_pf(all_trades):.2f}", f)
        _out("", f)

        # A) Per-symbol
        _out("=" * 80, f)
        _out("A) PER-SYMBOL TRADE SUMMARY", f)
        _out("=" * 80, f)
        by_symbol = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)
        for sym in sorted(by_symbol):
            _print_trade_table(by_symbol[sym], sym, f)
        _out("", f)

        # B) Per-regime
        _out("=" * 80, f)
        _out("B) PER-REGIME BREAKDOWN", f)
        _out("=" * 80, f)
        by_regime = defaultdict(list)
        for t in all_trades:
            by_regime[t.regime_entry or "UNKNOWN"].append(t)
        for regime in sorted(by_regime):
            _print_trade_table(by_regime[regime], regime, f)
        _out("", f)

        # C) Per-entry-type
        _out("=" * 80, f)
        _out("C) PER-ENTRY-TYPE BREAKDOWN", f)
        _out("=" * 80, f)
        by_entry = defaultdict(list)
        for t in all_trades:
            by_entry[t.entry_type or "UNKNOWN"].append(t)
        for etype in sorted(by_entry):
            _print_trade_table(by_entry[etype], etype, f)
        _out("", f)

        # D) Crisis window analysis
        _out("=" * 80, f)
        _out("D) CRISIS WINDOW ANALYSIS", f)
        _out("=" * 80, f)
        for cname, cstart, cend in CRISIS_WINDOWS:
            crisis_trades = []
            for t in all_trades:
                et = _trade_time(t, "entry_time")
                if et and cstart <= et <= cend:
                    crisis_trades.append(t)
            n = len(crisis_trades)
            tr = _total_r(crisis_trades)
            first_entry = None
            if crisis_trades:
                first_entry = min(_trade_time(t, "entry_time") for t in crisis_trades)
            first_str = first_entry.strftime("%Y-%m-%d %H:%M") if first_entry else "N/A"
            _out(f"  {cname:25s}  {cstart.date()} -> {cend.date()}  "
                 f"trades={n:3d}  totR={tr:+.2f}  first_entry={first_str}", f)
            if crisis_trades:
                for t in crisis_trades:
                    et = _trade_time(t, "entry_time")
                    xt = _trade_time(t, "exit_time")
                    _out(f"    {t.symbol:5s} {t.entry_type:20s} {t.regime_entry:15s} "
                         f"entry={et.strftime('%Y-%m-%d %H:%M') if et else 'N/A':16s} "
                         f"exit={xt.strftime('%Y-%m-%d %H:%M') if xt else 'N/A':16s} "
                         f"R={t.r_multiple:+.2f}  PnL=${t.pnl_dollars:+,.0f}  bars={t.bars_held}", f)
        _out("", f)

        # E) Position utilization during crisis
        _out("=" * 80, f)
        _out("E) POSITION UTILIZATION (crisis windows)", f)
        _out("=" * 80, f)
        for sym in sorted(crisis_logs):
            logs = crisis_logs[sym]
            if not logs:
                _out(f"  {sym}: no crisis state log entries", f)
                continue
            by_crisis = defaultdict(list)
            for entry in logs:
                by_crisis[entry["crisis"]].append(entry)
            for cname, cstart, cend in CRISIS_WINDOWS:
                entries = by_crisis.get(cname, [])
                if not entries:
                    continue
                short_bars = [e for e in entries if e.get("bias_confirmed") == "SHORT"]
                n_short = len(short_bars)
                n_in_pos = sum(1 for e in short_bars if e.get("in_position", False))
                util = (n_in_pos / n_short * 100) if n_short > 0 else 0.0
                _out(f"  {sym:5s} | {cname:25s} | total_bars={len(entries):3d} "
                     f"SHORT_bars={n_short:3d}  in_position={n_in_pos:3d}  util={util:5.1f}%", f)
        _out("", f)

        # F) Trade gap analysis
        _out("=" * 80, f)
        _out("F) TRADE GAP ANALYSIS (top 10 longest gaps)", f)
        _out("=" * 80, f)
        sorted_trades = sorted(all_trades, key=lambda t: _trade_time(t, "entry_time") or datetime.min)
        gaps = []
        for i in range(len(sorted_trades) - 1):
            xt = _trade_time(sorted_trades[i], "exit_time")
            et_next = _trade_time(sorted_trades[i + 1], "entry_time")
            if xt and et_next:
                gap_days = (et_next - xt).total_seconds() / 86400
                gaps.append((gap_days, xt, et_next, sorted_trades[i], sorted_trades[i + 1]))

        all_short_dates = set()
        for sym, logs in crisis_logs.items():
            for entry in logs:
                if entry.get("bias_confirmed") == "SHORT":
                    all_short_dates.add(entry["date"])

        gaps.sort(key=lambda x: -x[0])
        for rank, (gap_days, xt, et_next, t_prev, t_next) in enumerate(gaps[:10], 1):
            in_short = False
            d = xt
            while d <= et_next:
                if d.strftime("%Y-%m-%d") in all_short_dates:
                    in_short = True
                    break
                d += __import__("datetime").timedelta(days=1)
            short_flag = " ** SHORT bias overlap" if in_short else ""
            _out(f"  #{rank:2d}  gap={gap_days:6.1f}d  "
                 f"exit={xt.strftime('%Y-%m-%d'):10s} ({t_prev.symbol:5s}) -> "
                 f"entry={et_next.strftime('%Y-%m-%d'):10s} ({t_next.symbol:5s})"
                 f"{short_flag}", f)
        _out("", f)

        # G) Non-bear trades
        _out("=" * 80, f)
        _out("G) NON-BEAR TRADES (entered outside BEAR_STRONG/BEAR_TREND/BEAR_FORMING)", f)
        _out("=" * 80, f)
        non_bear = [t for t in all_trades if t.regime_entry not in BEAR_REGIMES]
        if non_bear:
            for t in non_bear:
                et = _trade_time(t, "entry_time")
                _out(f"  {t.symbol:5s} {t.entry_type:20s} regime={t.regime_entry:15s} "
                     f"entry={et.strftime('%Y-%m-%d %H:%M') if et else 'N/A':16s} "
                     f"R={t.r_multiple:+.2f}  PnL=${t.pnl_dollars:+,.0f}", f)
        else:
            _out("  (none)", f)
        _out("", f)

        # H) Quarterly trade distribution
        _out("=" * 80, f)
        _out("H) QUARTERLY TRADE DISTRIBUTION", f)
        _out("=" * 80, f)
        by_quarter = defaultdict(int)
        by_quarter_r = defaultdict(float)
        for t in all_trades:
            et = _trade_time(t, "entry_time")
            if et:
                q = (et.month - 1) // 3 + 1
                key = f"{et.year}-Q{q}"
                by_quarter[key] += 1
                by_quarter_r[key] += t.r_multiple
        for qk in sorted(by_quarter):
            n = by_quarter[qk]
            tr = by_quarter_r[qk]
            bar = "#" * n
            _out(f"  {qk:8s}  n={n:3d}  totR={tr:+6.2f}  {bar}", f)
        _out("", f)

        # I) Quality score distribution
        _out("=" * 80, f)
        _out("I) QUALITY / SCORE_ENTRY DISTRIBUTION", f)
        _out("=" * 80, f)
        scores = [t.score_entry for t in all_trades if t.score_entry != 0.0]
        if scores:
            import statistics
            _out(f"  Trades with score_entry != 0: {len(scores)}/{len(all_trades)}", f)
            _out(f"  Mean:   {statistics.mean(scores):.2f}", f)
            _out(f"  Median: {statistics.median(scores):.2f}", f)
            _out(f"  Min:    {min(scores):.2f}", f)
            _out(f"  Max:    {max(scores):.2f}", f)
            if len(scores) >= 2:
                _out(f"  Stdev:  {statistics.stdev(scores):.2f}", f)

            buckets = defaultdict(list)
            for t in all_trades:
                if t.score_entry == 0.0:
                    buckets["0.0 (unscored)"].append(t)
                elif t.score_entry < 2.0:
                    buckets["< 2.0"].append(t)
                elif t.score_entry < 3.0:
                    buckets["2.0-3.0"].append(t)
                elif t.score_entry < 4.0:
                    buckets["3.0-4.0"].append(t)
                elif t.score_entry < 5.0:
                    buckets["4.0-5.0"].append(t)
                else:
                    buckets[">= 5.0"].append(t)

            _out("", f)
            _out("  Score bucket breakdown:", f)
            for bk in ["0.0 (unscored)", "< 2.0", "2.0-3.0", "3.0-4.0", "4.0-5.0", ">= 5.0"]:
                if bk in buckets:
                    _print_trade_table(buckets[bk], f"score {bk}", f)
        else:
            qscores = [t.quality_score for t in all_trades if t.quality_score != 0.0]
            if qscores:
                import statistics
                _out(f"  Trades with quality_score != 0: {len(qscores)}/{len(all_trades)}", f)
                _out(f"  Mean:   {statistics.mean(qscores):.2f}", f)
                _out(f"  Median: {statistics.median(qscores):.2f}", f)
            else:
                _out("  No score_entry or quality_score data available.", f)
        _out("", f)

        # J) Pyramid analysis (R9 new)
        _out("=" * 80, f)
        _out("J) PYRAMID ANALYSIS (R9)", f)
        _out("=" * 80, f)
        # Detect pyramid add-ons: consecutive same-symbol trades with overlapping time
        pyramid_candidates = []
        by_sym_trades = defaultdict(list)
        for t in all_trades:
            by_sym_trades[t.symbol].append(t)
        # Pyramid trades show up as higher qty relative to base position
        # With pyramid_scale=1.0, add-on doubles the position
        for sym, sym_trades in by_sym_trades.items():
            for t in sym_trades:
                _out(f"  {sym:5s} {t.entry_type:20s} {t.regime_entry:15s} "
                     f"qty={t.qty:5d}  R={t.r_multiple:+.2f}  "
                     f"PnL=${t.pnl_dollars:+,.0f}  bars={t.bars_held}", f)
        _out("", f)

        # K) R8 vs R9 comparison
        _out("=" * 80, f)
        _out("K) R8 vs R9 COMPARISON", f)
        _out("=" * 80, f)
        from backtests.swing.auto.brs.scoring import extract_brs_metrics
        from dataclasses import asdict

        # R8 baseline: override current R9 defaults back to R8 values
        R8_MUTATIONS = {
            "disable_s1": False, "symbol_configs.GLD.adx_on": 16, "symbol_configs.GLD.adx_off": 14,
            "scale_out_enabled": True, "scale_out_pct": 0.33, "bd_donchian_period": 10,
            "lh_swing_lookback": 5, "scale_out_target_r": 3.6, "symbol_configs.QQQ.stop_floor_atr": 0.6,
            "symbol_configs.GLD.stop_buffer_atr": 0.18, "profit_floor_scale": 3.456,
            "peak_drop_enabled": True, "peak_drop_pct": -0.02, "peak_drop_lookback": 15,
            "bd_arm_bars": 24, "bias_4h_accel_enabled": True, "chop_short_entry_enabled": True,
            "persistence_override_bars": 5, "adx_strong": 25, "lh_arm_bars": 22,
            "bt_volume_mult": 2.0, "symbol_configs.QQQ.stop_buffer_atr": 0.3,
            # Revert R9-specific defaults back to R8 values
            "pyramid_enabled": False, "size_mult_bear_trend": 1.0,
            "chop_quality_mult": 0.60, "persist_quality_mult_bd": 0.60,
            "bd_max_stop_atr": 3.0, "be_trigger_r": 0.5, "min_hold_bars": 3,
            "stop_floor_bear_strong_mult": 1.3, "lh_arm_bars": 22,
            "symbol_configs.QQQ.base_risk_pct": 0.0072,
            "symbol_configs.GLD.base_risk_pct": 0.0072,
        }
        r8_cfg = BRSConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR)
        r8_cfg = mutate_brs_config(r8_cfg, R8_MUTATIONS)
        r8_data = load_brs_data(r8_cfg)
        r8_result = run_brs_independent(r8_data, r8_cfg)
        r8m = asdict(extract_brs_metrics(r8_result, INITIAL_EQUITY))
        r9m = asdict(extract_brs_metrics(result, INITIAL_EQUITY))

        comp_keys = [
            ("Total Trades", "total_trades", "d"),
            ("Profit Factor", "profit_factor", "f"),
            ("Max DD %", "max_dd_pct", "f"),
            ("Net Return %", "net_return_pct", "f"),
            ("Sharpe", "sharpe", "f"),
            ("Calmar", "calmar", "f"),
            ("Bear Alpha %", "bear_alpha_pct", "f"),
            ("Detection Latency", "detection_latency_days", "f"),
            ("Bias Latency", "bias_latency_days", "f"),
            ("Crisis Coverage", "crisis_coverage", "f"),
        ]
        header = f"  {'Metric':<30} {'R8':>12} {'R9':>12} {'Delta':>12}"
        _out(header, f)
        _out("  " + "-" * 68, f)
        for label, key, fmt in comp_keys:
            r8v = r8m.get(key, 0)
            r9v = r9m.get(key, 0)
            if fmt == "d":
                _out(f"  {label:<30} {r8v:>12d} {r9v:>12d} {r9v-r8v:>+12d}", f)
            else:
                _out(f"  {label:<30} {r8v:>12.3f} {r9v:>12.3f} {r9v-r8v:>+12.3f}", f)
        _out("", f)

        _out("=" * 80, f)
        _out("R9 DIAGNOSTICS COMPLETE", f)
        _out("=" * 80, f)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
