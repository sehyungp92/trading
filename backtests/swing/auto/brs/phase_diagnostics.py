"""BRS 6 diagnostic modules (D1-D6).

D1: Regime accuracy during crisis windows (always)
D2: Signal quality by entry type x regime (always)
D3: Exit efficiency per trade (Phase 2+)
D4: Drawdown attribution (Phase 2+)
D5: VIX correlation (Phase 3+)
D6: Phase delta (always)
"""
from __future__ import annotations

import logging
from io import StringIO

from backtests.swing.analysis.brs_diagnostics import CRISIS_WINDOWS
from backtests.swing.auto.brs.phase_gates import check_phase_gate
from backtests.swing.auto.brs.scoring import BRSMetrics

logger = logging.getLogger(__name__)


def generate_phase_diagnostics(
    phase: int,
    metrics: BRSMetrics,
    greedy_result: dict | None,
    state_dict: dict | None,
    all_trades: list | None = None,
    force_all_modules: bool = False,
    crisis_state_logs: list[dict] | None = None,
) -> str:
    """Generate full diagnostic report for a BRS phase.

    Args:
        phase: Current optimization phase
        metrics: BRS metrics from backtest
        greedy_result: Greedy optimization result dict
        state_dict: Phase state dict (for phase delta)
        all_trades: List of TradeRecord for detailed analysis
        force_all_modules: Enable all diagnostic modules regardless of phase
        crisis_state_logs: Daily state snapshots during crisis windows (from engine)

    Returns:
        Multi-section diagnostic report string
    """
    buf = StringIO()
    buf.write(f"{'='*70}\n")
    buf.write(f"BRS PHASE {phase} DIAGNOSTICS\n")
    buf.write(f"{'='*70}\n\n")

    # D1: Regime accuracy (always enabled)
    _write_d1_regime_accuracy(buf, metrics, all_trades)

    # D2: Signal quality by entry type (always enabled)
    _write_d2_signal_quality(buf, all_trades)

    # D3: Exit efficiency (Phase 2+ or forced)
    if phase >= 2 or force_all_modules:
        _write_d3_exit_efficiency(buf, all_trades)

    # D4: Drawdown attribution (Phase 2+ or forced)
    if phase >= 2 or force_all_modules:
        _write_d4_drawdown_attribution(buf, all_trades, metrics)

    # D5: Hold time & conviction analysis (Phase 3+ or forced)
    if phase >= 3 or force_all_modules:
        _write_d5_hold_conviction(buf, all_trades, metrics)

    # D6: Phase delta (always enabled)
    _write_d6_phase_delta(buf, phase, metrics, state_dict)

    # D7: LH/BD signal-specific analysis (always enabled for new signals)
    _write_d7_signal_analysis(buf, all_trades, metrics)

    # D8: Crisis root-cause analysis (always enabled)
    _write_d8_crisis_root_cause(buf, crisis_state_logs)

    # Gate assessment
    _write_gate_assessment(buf, phase, metrics, greedy_result)

    return buf.getvalue()


def get_enabled_modules(phase: int) -> list[str]:
    """Return list of enabled diagnostic module IDs for this phase."""
    modules = ["D1", "D2", "D6"]  # Always enabled
    if phase >= 2:
        modules.extend(["D3", "D4"])
    if phase >= 3:
        modules.append("D5")
    return modules


def get_diagnostic_gaps(phase: int, metrics: BRSMetrics) -> list[str]:
    """Detect weaknesses without corresponding diagnostic coverage.

    Returns list of gap descriptions that trigger 'improve_diagnostics' decision.
    """
    gaps = []
    enabled = get_enabled_modules(phase)

    # If exit efficiency is weak but D3 not enabled
    if metrics.exit_efficiency < 0.35 and "D3" not in enabled:
        gaps.append("Exit efficiency below target but D3 (exit analysis) not enabled")

    # If drawdown is high but D4 not enabled
    if metrics.max_dd_pct > 0.15 and "D4" not in enabled:
        gaps.append("High drawdown but D4 (drawdown attribution) not enabled")

    # If bear trade quality is low but D5 not enabled
    if metrics.bear_trade_wr < 55 and "D5" not in enabled:
        gaps.append("Low bear WR but D5 (hold/conviction analysis) not enabled")

    return gaps


# ---------------------------------------------------------------------------
# D1: Regime accuracy during crisis windows
# ---------------------------------------------------------------------------

def _write_d1_regime_accuracy(buf: StringIO, metrics: BRSMetrics, trades: list | None = None) -> None:
    buf.write("--- D1: Regime Accuracy ---\n")
    buf.write(f"  Regime F1:           {metrics.regime_f1:.3f}\n")
    buf.write(f"  Detection latency:   {metrics.detection_latency_days:.1f} days\n")

    buf.write(f"\n  {'Crisis Window':<25} {'Trades':>6} {'WR%':>6} {'AvgR':>7} {'PnL':>10} {'Latency':>8}\n")
    for name, start, end in CRISIS_WINDOWS:
        if trades:
            crisis_trades = [
                t for t in trades
                if getattr(t, "entry_time", None) is not None
                and start <= t.entry_time.replace(tzinfo=None) <= end
            ]
            n = len(crisis_trades)
            if n > 0:
                wins = sum(1 for t in crisis_trades if t.r_multiple > 0)
                wr = wins / n * 100
                avg_r = sum(t.r_multiple for t in crisis_trades) / n
                pnl = sum(t.pnl_dollars for t in crisis_trades)
                first_entry = min(t.entry_time.replace(tzinfo=None) for t in crisis_trades)
                latency = (first_entry - start).total_seconds() / 86400
                buf.write(f"  {name:<25} {n:>6} {wr:>5.1f}% {avg_r:>7.2f} {pnl:>9.0f}$ {latency:>6.1f}d\n")
            else:
                buf.write(f"  {name:<25} {'--':>6} {'--':>6} {'--':>7} {'--':>10} {'NO COVERAGE':>8}\n")
        else:
            buf.write(f"  {name:<25} {start.date()} to {end.date()}\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D2: Signal quality by entry type x regime
# ---------------------------------------------------------------------------

def _write_d2_signal_quality(buf: StringIO, trades: list | None) -> None:
    buf.write("--- D2: Signal Quality by Entry Type ---\n")
    if not trades:
        buf.write("  No trade data available\n\n")
        return

    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for t in trades:
        key = f"{t.entry_type}/{t.regime_entry}"
        buckets[key].append(t)

    buf.write(f"  {'Type/Regime':<30} {'N':>4} {'WR%':>6} {'AvgR':>7} {'PF':>6}\n")
    for key in sorted(buckets):
        tl = buckets[key]
        n = len(tl)
        wins = sum(1 for t in tl if t.r_multiple > 0)
        wr = wins / n * 100 if n > 0 else 0
        avg_r = sum(t.r_multiple for t in tl) / n if n > 0 else 0
        gw = sum(t.r_multiple for t in tl if t.r_multiple > 0)
        gl = abs(sum(t.r_multiple for t in tl if t.r_multiple <= 0))
        pf = gw / gl if gl > 0 else 999
        buf.write(f"  {key:<30} {n:>4} {wr:>5.1f}% {avg_r:>7.2f} {pf:>6.2f}\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D3: Exit efficiency per trade
# ---------------------------------------------------------------------------

def _write_d3_exit_efficiency(buf: StringIO, trades: list | None) -> None:
    buf.write("--- D3: Exit Efficiency ---\n")
    if not trades:
        buf.write("  No trade data available\n\n")
        return

    from collections import defaultdict
    by_reason: dict[str, list] = defaultdict(list)
    for t in trades:
        by_reason[t.exit_reason].append(t)

    buf.write(f"  {'Exit Reason':<20} {'N':>4} {'AvgR':>7} {'Avg MFE':>8}\n")
    for reason in sorted(by_reason):
        tl = by_reason[reason]
        n = len(tl)
        avg_r = sum(t.r_multiple for t in tl) / n if n > 0 else 0
        avg_mfe = sum(t.mfe_r for t in tl) / n if n > 0 else 0
        buf.write(f"  {reason:<20} {n:>4} {avg_r:>7.2f} {avg_mfe:>8.2f}\n")

    # Overall captured/available
    total_captured = sum(t.r_multiple for t in trades)
    total_available = sum(t.mfe_r for t in trades)
    eff = total_captured / total_available if total_available > 0 else 0
    buf.write(f"\n  Overall efficiency: {eff:.1%} (captured {total_captured:.1f}R of {total_available:.1f}R available)\n\n")


# ---------------------------------------------------------------------------
# D4: Drawdown attribution
# ---------------------------------------------------------------------------

def _write_d4_drawdown_attribution(buf: StringIO, trades: list | None, metrics: BRSMetrics) -> None:
    buf.write("--- D4: Drawdown Attribution ---\n")
    buf.write(f"  Max drawdown: {metrics.max_dd_pct:.1%}\n")

    if not trades:
        buf.write("  No trade data available\n\n")
        return

    # Find worst trades contributing to drawdowns
    worst = sorted(trades, key=lambda t: t.r_multiple)[:5]
    buf.write(f"\n  Top 5 worst trades:\n")
    for t in worst:
        buf.write(f"    {t.symbol} {t.entry_type} {t.regime_entry}: "
                 f"R={t.r_multiple:+.2f} MFE={t.mfe_r:.2f} exit={t.exit_reason}\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D5: Hold time & conviction analysis
# ---------------------------------------------------------------------------

def _write_d5_hold_conviction(buf: StringIO, trades: list | None, metrics: BRSMetrics) -> None:
    buf.write("--- D5: Hold Time & Conviction Analysis ---\n")
    if not trades:
        buf.write("  No trade data available\n\n")
        return

    # Hold duration buckets (bars_held; 7 bars ≈ 1 day RTH)
    hold_buckets = [
        ("<8h", lambda t: t.bars_held < 8),
        ("8-24h", lambda t: 8 <= t.bars_held < 24),
        ("1-3d", lambda t: 24 <= t.bars_held < 72),
        ("3-7d", lambda t: 72 <= t.bars_held < 168),
        ("7d+", lambda t: t.bars_held >= 168),
    ]

    buf.write(f"\n  Hold Duration:\n")
    buf.write(f"  {'Bucket':<12} {'N':>4} {'WR%':>6} {'AvgR':>7} {'PF':>6}\n")
    for label, pred in hold_buckets:
        bucket = [t for t in trades if pred(t)]
        if not bucket:
            continue
        n = len(bucket)
        wins = sum(1 for t in bucket if t.r_multiple > 0)
        wr = wins / n * 100
        avg_r = sum(t.r_multiple for t in bucket) / n
        gw = sum(t.r_multiple for t in bucket if t.r_multiple > 0)
        gl = abs(sum(t.r_multiple for t in bucket if t.r_multiple <= 0))
        pf = gw / gl if gl > 0 else 999
        buf.write(f"  {label:<12} {n:>4} {wr:>5.1f}% {avg_r:>7.2f} {pf:>6.2f}\n")

    # Quality score buckets (quality_score maps to conviction at entry)
    qual_buckets = [
        ("Q 0-0.25", lambda t: t.quality_score < 0.25),
        ("Q 0.25-0.5", lambda t: 0.25 <= t.quality_score < 0.50),
        ("Q 0.5-0.75", lambda t: 0.50 <= t.quality_score < 0.75),
        ("Q 0.75-1.0", lambda t: t.quality_score >= 0.75),
    ]

    buf.write(f"\n  Entry Quality Score:\n")
    buf.write(f"  {'Bucket':<12} {'N':>4} {'WR%':>6} {'AvgR':>7} {'PF':>6}\n")
    for label, pred in qual_buckets:
        bucket = [t for t in trades if pred(t)]
        if not bucket:
            continue
        n = len(bucket)
        wins = sum(1 for t in bucket if t.r_multiple > 0)
        wr = wins / n * 100
        avg_r = sum(t.r_multiple for t in bucket) / n
        gw = sum(t.r_multiple for t in bucket if t.r_multiple > 0)
        gl = abs(sum(t.r_multiple for t in bucket if t.r_multiple <= 0))
        pf = gw / gl if gl > 0 else 999
        buf.write(f"  {label:<12} {n:>4} {wr:>5.1f}% {avg_r:>7.2f} {pf:>6.2f}\n")

    buf.write("\n")


# ---------------------------------------------------------------------------
# D6: Phase delta (before/after comparison)
# ---------------------------------------------------------------------------

def _write_d6_phase_delta(buf: StringIO, phase: int, metrics: BRSMetrics, state_dict: dict | None) -> None:
    buf.write("--- D6: Phase Delta ---\n")

    _LOWER_IS_BETTER = {"max_dd_pct", "detection_latency_days"}
    _COMPARE_METRICS = [
        ("sharpe", "Sharpe"),
        ("max_dd_pct", "Max DD"),
        ("bear_pf", "Bear PF"),
        ("calmar", "Calmar"),
        ("exit_efficiency", "Exit Eff"),
        ("regime_f1", "Regime F1"),
        ("bear_alpha_pct", "Bear Alpha%"),
    ]

    prev_metrics = None
    prev_score = None
    if state_dict and "phase_results" in state_dict:
        prev_phase = phase - 1
        prev = state_dict["phase_results"].get(str(prev_phase)) or state_dict["phase_results"].get(prev_phase)
        if prev:
            prev_score = prev.get("final_score", 0)
            prev_metrics = prev.get("final_metrics", {})

    if prev_metrics and prev_score is not None:
        buf.write(f"  Comparing Phase {phase} vs Phase {phase - 1}:\n")
        buf.write(f"    Score: {prev_score:.4f}\n\n")
        buf.write(f"  {'Metric':<15} {'Prior':>8} {'Current':>8} {'Delta':>8} {'Direction':>12}\n")
        for metric_key, label in _COMPARE_METRICS:
            prior_val = prev_metrics.get(metric_key, 0)
            current_val = getattr(metrics, metric_key, 0)
            delta = current_val - prior_val
            if abs(delta) < 0.001:
                direction = "FLAT"
            elif metric_key in _LOWER_IS_BETTER:
                direction = "IMPROVED" if delta < 0 else "!!! REGRESSED"
            else:
                direction = "IMPROVED" if delta > 0 else "!!! REGRESSED"
            buf.write(f"  {label:<15} {prior_val:>8.3f} {current_val:>8.3f} {delta:>+8.3f} {direction:>12}\n")
    else:
        buf.write(f"  No prior phase data for comparison\n")

    buf.write(f"\n  Current metrics:\n")
    buf.write(f"    Trades:     {metrics.total_trades}\n")
    buf.write(f"    Bear PF:    {metrics.bear_pf:.2f}\n")
    buf.write(f"    Max DD:     {metrics.max_dd_pct:.1%}\n")
    buf.write(f"    Calmar:     {metrics.calmar:.2f}\n")
    buf.write(f"    Sharpe:     {metrics.sharpe:.2f}\n")
    buf.write(f"    Bear alpha: {metrics.bear_alpha_pct:.1f}%\n\n")


# ---------------------------------------------------------------------------
# D7: LH/BD signal-specific analysis
# ---------------------------------------------------------------------------

def _write_d7_signal_analysis(buf: StringIO, trades: list | None, metrics: BRSMetrics) -> None:
    buf.write("--- D7: LH/BD Signal Analysis ---\n")
    if not trades:
        buf.write("  No trade data available\n\n")
        return

    from collections import defaultdict

    # Separate by new signal types
    lh_trades = [t for t in trades if getattr(t, "entry_type", "") == "LH_REJECTION"]
    bd_trades = [t for t in trades if getattr(t, "entry_type", "") == "BD_CONTINUATION"]
    legacy_trades = [t for t in trades if getattr(t, "entry_type", "") not in ("LH_REJECTION", "BD_CONTINUATION")]

    def _signal_stats(label: str, tl: list) -> None:
        n = len(tl)
        if n == 0:
            buf.write(f"\n  {label}: 0 trades\n")
            return
        wins = [t for t in tl if t.r_multiple > 0]
        losses = [t for t in tl if t.r_multiple <= 0]
        wr = len(wins) / n * 100
        avg_r = sum(t.r_multiple for t in tl) / n
        gw = sum(t.r_multiple for t in wins) if wins else 0
        gl = abs(sum(t.r_multiple for t in losses)) if losses else 0
        pf = gw / gl if gl > 0 else 999
        total_r = sum(t.r_multiple for t in tl)
        avg_hold = sum(t.bars_held for t in tl) / n if n > 0 else 0

        buf.write(f"\n  {label}: {n} trades\n")
        buf.write(f"    Win rate:    {wr:.1f}%\n")
        buf.write(f"    Avg R:       {avg_r:+.2f}\n")
        buf.write(f"    Total R:     {total_r:+.1f}\n")
        buf.write(f"    PF:          {pf:.2f}\n")
        buf.write(f"    Avg hold:    {avg_hold:.0f} bars ({avg_hold/7:.1f} days)\n")

        # Per-regime breakdown
        by_regime: dict[str, list] = defaultdict(list)
        for t in tl:
            by_regime[t.regime_entry].append(t)
        if len(by_regime) > 1:
            buf.write(f"    By regime:\n")
            for regime in sorted(by_regime):
                rtl = by_regime[regime]
                rn = len(rtl)
                rwr = sum(1 for t in rtl if t.r_multiple > 0) / rn * 100
                ravg = sum(t.r_multiple for t in rtl) / rn
                buf.write(f"      {regime:<15} N={rn:<3} WR={rwr:>5.1f}% AvgR={ravg:>+.2f}\n")

        # Per-symbol breakdown
        by_sym: dict[str, list] = defaultdict(list)
        for t in tl:
            by_sym[t.symbol].append(t)
        if len(by_sym) > 1:
            buf.write(f"    By symbol:\n")
            for sym in sorted(by_sym):
                stl = by_sym[sym]
                sn = len(stl)
                swr = sum(1 for t in stl if t.r_multiple > 0) / sn * 100
                savg = sum(t.r_multiple for t in stl) / sn
                buf.write(f"      {sym:<6} N={sn:<3} WR={swr:>5.1f}% AvgR={savg:>+.2f}\n")

        # Stop distance analysis
        stop_dists = [abs(t.initial_stop - t.entry_price) / t.entry_price * 100
                      for t in tl if hasattr(t, "initial_stop") and hasattr(t, "entry_price")
                      and t.entry_price > 0]
        if stop_dists:
            buf.write(f"    Stop dist:   min={min(stop_dists):.2f}% avg={sum(stop_dists)/len(stop_dists):.2f}% max={max(stop_dists):.2f}%\n")

        # Exit reason breakdown
        by_exit: dict[str, list] = defaultdict(list)
        for t in tl:
            by_exit[t.exit_reason].append(t)
        buf.write(f"    Exit reasons:\n")
        for reason in sorted(by_exit):
            etl = by_exit[reason]
            en = len(etl)
            eavg = sum(t.r_multiple for t in etl) / en
            buf.write(f"      {reason:<20} N={en:<3} AvgR={eavg:>+.2f}\n")

    _signal_stats("LH_REJECTION (primary)", lh_trades)
    _signal_stats("BD_CONTINUATION (secondary)", bd_trades)
    if legacy_trades:
        _signal_stats("LEGACY signals", legacy_trades)

    # Signal concentration analysis
    total = len(trades)
    if total > 0:
        buf.write(f"\n  Signal Mix:\n")
        buf.write(f"    LH_REJECTION:     {len(lh_trades):>3} ({len(lh_trades)/total*100:.0f}%)\n")
        buf.write(f"    BD_CONTINUATION:  {len(bd_trades):>3} ({len(bd_trades)/total*100:.0f}%)\n")
        if legacy_trades:
            buf.write(f"    Legacy:           {len(legacy_trades):>3} ({len(legacy_trades)/total*100:.0f}%)\n")
        buf.write(f"    Total:            {total:>3}\n")

    # Alpha contribution
    lh_pnl = sum(t.pnl_dollars for t in lh_trades) if lh_trades else 0
    bd_pnl = sum(t.pnl_dollars for t in bd_trades) if bd_trades else 0
    total_pnl = sum(t.pnl_dollars for t in trades) if trades else 0
    buf.write(f"\n  PnL Attribution:\n")
    buf.write(f"    LH_REJECTION:     ${lh_pnl:>+10,.0f}\n")
    buf.write(f"    BD_CONTINUATION:  ${bd_pnl:>+10,.0f}\n")
    if legacy_trades:
        leg_pnl = sum(t.pnl_dollars for t in legacy_trades)
        buf.write(f"    Legacy:           ${leg_pnl:>+10,.0f}\n")
    buf.write(f"    Total:            ${total_pnl:>+10,.0f}\n")

    buf.write("\n")


# ---------------------------------------------------------------------------
# D8: Crisis root-cause analysis
# ---------------------------------------------------------------------------

def _write_d8_crisis_root_cause(buf: StringIO, crisis_logs: list[dict] | None) -> None:
    """Analyze WHY trades didn't fire during crisis windows.

    Shows daily regime state progression through each crisis to identify:
    - Was regime_on False? (ADX too low)
    - Was bias not confirmed? (hold_count insufficient)
    - Did crash override fire?
    - What was the detection latency?
    """
    buf.write("--- D8: Crisis Root-Cause Analysis ---\n")
    if not crisis_logs:
        buf.write("  No crisis state data available (engine did not log crisis windows)\n\n")
        return

    from collections import defaultdict
    by_crisis: dict[str, list[dict]] = defaultdict(list)
    for entry in crisis_logs:
        by_crisis[entry["crisis"]].append(entry)

    for crisis_name in [c[0] for c in CRISIS_WINDOWS]:
        logs = by_crisis.get(crisis_name, [])
        buf.write(f"\n  [{crisis_name}] — {len(logs)} daily bars logged\n")
        if not logs:
            buf.write("    NO DATA: crisis window outside backtest date range\n")
            continue

        # Summary stats
        regime_on_days = sum(1 for d in logs if d["regime_on"])
        short_confirmed_days = sum(1 for d in logs if d["bias_confirmed"] == "SHORT")
        crash_override_days = sum(1 for d in logs if d.get("crash_override"))
        in_position_days = sum(1 for d in logs if d["in_position"])
        min_return = min(d["daily_return"] for d in logs)
        max_atr_ratio = max(d["atr_ratio"] for d in logs)

        buf.write(f"    Regime ON days:      {regime_on_days}/{len(logs)}\n")
        buf.write(f"    SHORT confirmed:     {short_confirmed_days}/{len(logs)}\n")
        buf.write(f"    Crash override days:  {crash_override_days}/{len(logs)}\n")
        buf.write(f"    In position days:    {in_position_days}/{len(logs)}\n")
        buf.write(f"    Worst daily return:  {min_return:+.2f}%\n")
        buf.write(f"    Max ATR ratio:       {max_atr_ratio:.2f}\n")

        # Detection: first day bias confirmed SHORT
        first_short = next((d for d in logs if d["bias_confirmed"] == "SHORT"), None)
        if first_short:
            buf.write(f"    First SHORT confirm: {first_short['date']} "
                     f"(ADX={first_short['adx']}, regime={first_short['regime']}, "
                     f"crash_override={first_short.get('crash_override', False)})\n")
        else:
            buf.write(f"    First SHORT confirm: NEVER — bias never confirmed during crisis\n")

        # Root cause breakdown: why wasn't SHORT confirmed on each day?
        no_confirm_reasons: dict[str, int] = defaultdict(int)
        for d in logs:
            if d["bias_confirmed"] == "SHORT" or d["in_position"]:
                continue
            if d["bias_raw"] != "SHORT":
                no_confirm_reasons["raw_bias_not_SHORT"] += 1
            elif not d["regime_on"] and not d.get("crash_override"):
                no_confirm_reasons["regime_OFF_no_crash_override"] += 1
            elif d["hold_count"] < 2:
                no_confirm_reasons["hold_count_insufficient"] += 1
            else:
                no_confirm_reasons["other"] += 1

        if no_confirm_reasons:
            buf.write(f"    Blocking reasons (non-confirmed, non-positioned days):\n")
            for reason, count in sorted(no_confirm_reasons.items(), key=lambda x: -x[1]):
                buf.write(f"      {reason}: {count} days\n")

        # Day-by-day table (compact — first 10 days only for space)
        show_days = logs[:10]
        buf.write(f"\n    {'Date':<12} {'Reg':>5} {'ADX':>5} {'RawB':>6} {'Conf':>6} "
                 f"{'Hold':>4} {'Ret%':>6} {'ATRr':>5} {'CrOvr':>5} {'InPos':>5}\n")
        for d in show_days:
            r_on = "ON" if d["regime_on"] else "OFF"
            buf.write(f"    {d['date']:<12} {r_on:>5} {d['adx']:>5.1f} "
                     f"{d['bias_raw']:>6} {d['bias_confirmed']:>6} "
                     f"{d['hold_count']:>4} {d['daily_return']:>+5.1f}% "
                     f"{d['atr_ratio']:>5.2f} {'Y' if d.get('crash_override') else 'N':>5} "
                     f"{'Y' if d['in_position'] else 'N':>5}\n")
        if len(logs) > 10:
            buf.write(f"    ... ({len(logs) - 10} more days)\n")

    buf.write("\n")


# ---------------------------------------------------------------------------
# Gate assessment
# ---------------------------------------------------------------------------

def _write_gate_assessment(buf: StringIO, phase: int, metrics: BRSMetrics, gr: dict | None) -> None:
    buf.write("--- Gate Assessment ---\n")
    gate = check_phase_gate(phase, metrics, gr)
    buf.write(f"  Result: {'PASS' if gate.passed else 'FAIL'}\n")
    if gate.failure_category:
        buf.write(f"  Category: {gate.failure_category}\n")
    for c in gate.criteria:
        status = "PASS" if c.passed else "FAIL"
        buf.write(f"  [{status}] {c.name}: {c.actual:.3f} vs {c.target:.3f}\n")
    for rec in gate.recommendations:
        buf.write(f"  -> {rec}\n")
    buf.write("\n")
