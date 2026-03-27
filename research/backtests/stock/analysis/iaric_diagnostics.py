"""IARIC comprehensive diagnostics -- 28-section deep analysis.

Provides strategy-specific diagnostic insight into IARIC backtest
results, covering setup types, confidence/location grades, acceptance
counting, 6-factor sizing decomposition, timing windows, exit analysis,
micropressure, sponsorship, MFE/MAE, and more.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from research.backtests.stock.analysis.iaric_filter_attribution import (
    iaric_filter_attribution_report,
)
from research.backtests.stock.analysis.iaric_shadow_tracker import IARICShadowTracker
from research.backtests.stock.models import TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(t: TradeRecord, key: str, default=None):
    """Safe metadata access."""
    return t.metadata.get(key, default)


def _hdr(title: str) -> str:
    """Section header."""
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def _group_stats(trades: list[TradeRecord]) -> str:
    """Standard stats block: n, win%, mean R, median R, PF, total R."""
    if not trades:
        return "    (no trades)"
    n = len(trades)
    wins = sum(1 for t in trades if t.is_winner)
    wr = wins / n
    rs = [t.r_multiple for t in trades]
    mean_r = float(np.mean(rs))
    median_r = float(np.median(rs))
    total_r = sum(rs)
    gross_p = sum(r for r in rs if r > 0)
    gross_l = abs(sum(r for r in rs if r < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    return (
        f"    n={n}, WR={wr:.1%}, Mean R={mean_r:+.3f}, "
        f"Median R={median_r:+.3f}, PF={pf:.2f}, Total R={total_r:+.2f}"
    )


def _hold_hours(t: TradeRecord) -> float:
    return (t.exit_time - t.entry_time).total_seconds() / 3600


def _has_metadata(trades: list[TradeRecord]) -> bool:
    if not trades:
        return False
    return bool(trades[0].metadata and "confidence" in trades[0].metadata)


# ---------------------------------------------------------------------------
# Diagnostic sections
# ---------------------------------------------------------------------------


def _s01_overview(trades: list[TradeRecord]) -> str:
    """S01: Overview statistics."""
    lines = [_hdr("1. Overview")]
    n = len(trades)
    wins = sum(1 for t in trades if t.is_winner)
    rs = [t.r_multiple for t in trades]
    pnls = [t.pnl_net for t in trades]
    total_pnl = sum(pnls)
    mean_r = float(np.mean(rs)) if rs else 0
    median_r = float(np.median(rs)) if rs else 0
    total_r = sum(rs)
    gross_p = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    avg_hold = float(np.mean([_hold_hours(t) for t in trades])) if trades else 0

    # Sharpe approximation from R-multiples
    sharpe = float(np.mean(rs)) / float(np.std(rs)) * (252 ** 0.5) if len(rs) > 1 and np.std(rs) > 0 else 0

    # Max drawdown in R
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak)) if len(cum) > 0 else 0

    lines.append(f"  Trades: {n}")
    lines.append(f"  Win Rate: {wins/n:.1%}" if n else "  Win Rate: N/A")
    lines.append(f"  Mean R: {mean_r:+.3f}  |  Median R: {median_r:+.3f}")
    lines.append(f"  Total R: {total_r:+.2f}  |  Total PnL: ${total_pnl:,.2f}")
    lines.append(f"  Profit Factor: {pf:.2f}  |  Sharpe (approx): {sharpe:.2f}")
    lines.append(f"  Max Drawdown: {max_dd:+.2f}R")
    lines.append(f"  Avg Hold: {avg_hold:.1f}h")
    return "\n".join(lines)


def _s02_signal_funnel(
    trades: list[TradeRecord],
    daily_selections: dict | None,
    shadow_tracker: IARICShadowTracker | None,
) -> str:
    """S02: Signal funnel (evaluated -> entered)."""
    lines = [_hdr("2. Signal Funnel")]
    if shadow_tracker:
        lines.append(shadow_tracker.funnel_report())
    elif daily_selections:
        total_candidates = sum(len(a.tradable) for a in daily_selections.values())
        lines.append(f"  Total tradable candidates evaluated: {total_candidates}")
        lines.append(f"  Trades taken: {len(trades)}")
        if total_candidates > 0:
            lines.append(f"  Conversion: {len(trades)/total_candidates:.1%}")
    else:
        lines.append("  No funnel data available (pass --shadow to enable)")
    return "\n".join(lines)


def _s03_setup_type(trades: list[TradeRecord]) -> str:
    """S03: PANIC_FLUSH vs DRIFT_EXHAUSTION vs HOT vs WARM."""
    lines = [_hdr("3. Setup Type Breakdown")]
    if not _has_metadata(trades):
        lines.append("  (no metadata -- run with upgraded engine)")
        return "\n".join(lines)

    by_type: dict[str, list[TradeRecord]] = {}
    for t in trades:
        st = _meta(t, "setup_type", "UNKNOWN") or "UNKNOWN"
        by_type.setdefault(st, []).append(t)

    for st in ["PANIC_FLUSH", "DRIFT_EXHAUSTION", "HOT", "WARM", "UNKNOWN"]:
        if st in by_type:
            lines.append(f"  {st}:")
            lines.append(_group_stats(by_type[st]))
    return "\n".join(lines)


def _s04_confidence_level(trades: list[TradeRecord]) -> str:
    """S04: GREEN vs YELLOW outcomes -- is YELLOW(0.65x) worth trading?"""
    lines = [_hdr("4. Confidence Level")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    by_conf: dict[str, list[TradeRecord]] = {}
    for t in trades:
        conf = _meta(t, "confidence", "UNKNOWN") or "UNKNOWN"
        by_conf.setdefault(conf, []).append(t)

    for conf in ["GREEN", "YELLOW", "UNKNOWN"]:
        if conf in by_conf:
            lines.append(f"  {conf}:")
            lines.append(_group_stats(by_conf[conf]))

    # YELLOW value analysis
    yellow = by_conf.get("YELLOW", [])
    green = by_conf.get("GREEN", [])
    if yellow and green:
        y_mean = float(np.mean([t.r_multiple for t in yellow]))
        g_mean = float(np.mean([t.r_multiple for t in green]))
        lines.append(f"\n  YELLOW edge-positive? Mean R={y_mean:+.3f} (vs GREEN {g_mean:+.3f})")
        lines.append(f"  YELLOW trades as % of total: {len(yellow)/len(trades):.0%}")
        lines.append(f"  Impact of removing YELLOW: {sum(t.r_multiple for t in yellow):+.2f}R lost")

    return "\n".join(lines)


def _s05_location_grade(trades: list[TradeRecord]) -> str:
    """S05: A/B/C location grade -- does grade predict R?"""
    lines = [_hdr("5. Location Grade")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    by_grade: dict[str, list[TradeRecord]] = {}
    for t in trades:
        grade = _meta(t, "location_grade", "B") or "B"
        by_grade.setdefault(grade, []).append(t)

    for grade in ["A", "B", "C"]:
        if grade in by_grade:
            mult = {"A": "1.00", "B": "0.90", "C": "0.70"}[grade]
            lines.append(f"  Grade {grade} (×{mult}):")
            lines.append(_group_stats(by_grade[grade]))

    return "\n".join(lines)


def _s06_sponsorship_state(trades: list[TradeRecord]) -> str:
    """S06: STRONG -> STALE -> WEAK decomposition."""
    lines = [_hdr("6. Sponsorship State")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    by_spons: dict[str, list[TradeRecord]] = {}
    for t in trades:
        sp = _meta(t, "sponsorship_state", "UNKNOWN") or "UNKNOWN"
        by_spons.setdefault(sp, []).append(t)

    for sp in ["STRONG", "ACCUMULATE", "NEUTRAL", "STALE", "WEAK", "UNKNOWN"]:
        if sp in by_spons:
            lines.append(f"  {sp}:")
            lines.append(_group_stats(by_spons[sp]))
    return "\n".join(lines)


def _s07_micropressure(trades: list[TradeRecord]) -> str:
    """S07: ACCUMULATE/NEUTRAL by proxy mode."""
    lines = [_hdr("7. Micropressure Signal")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    by_mp: dict[str, list[TradeRecord]] = {}
    for t in trades:
        mp = _meta(t, "micropressure_signal", "NEUTRAL") or "NEUTRAL"
        by_mp.setdefault(mp, []).append(t)

    for mp in ["ACCUMULATE", "NEUTRAL", "DISTRIBUTE", "UNKNOWN"]:
        if mp in by_mp:
            lines.append(f"  {mp}:")
            lines.append(_group_stats(by_mp[mp]))

    lines.append("\n  Note: backtest uses PROXY mode (no tick data)")
    return "\n".join(lines)


def _s08_acceptance_count(trades: list[TradeRecord]) -> str:
    """S08: Performance by actual acceptance count -- higher = better?"""
    lines = [_hdr("8. Acceptance Count")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    by_count: dict[int, list[TradeRecord]] = {}
    for t in trades:
        ac = _meta(t, "acceptance_count", 0) or 0
        by_count.setdefault(ac, []).append(t)

    for count in sorted(by_count):
        req = ""
        group = by_count[count]
        # Show required count if available
        reqs = [_meta(t, "required_acceptance_count", 0) or 0 for t in group]
        avg_req = float(np.mean(reqs)) if reqs else 0
        lines.append(f"  Count={count} (avg required={avg_req:.1f}):")
        lines.append(_group_stats(group))

    return "\n".join(lines)


def _s09_conviction_adders(trades: list[TradeRecord]) -> str:
    """S09: Per-adder frequency + performance impact."""
    lines = [_hdr("9. Conviction Adders")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    adder_trades: dict[str, list[TradeRecord]] = {}
    adder_freq: dict[str, int] = {}

    for t in trades:
        adders = _meta(t, "conviction_adders", []) or []
        for adder in adders:
            adder_trades.setdefault(adder, []).append(t)
            adder_freq[adder] = adder_freq.get(adder, 0) + 1

    n = len(trades)
    for adder in sorted(adder_freq, key=lambda x: -adder_freq[x]):
        freq = adder_freq[adder]
        group = adder_trades[adder]
        lines.append(f"  {adder} (present in {freq}/{n} = {freq/n:.0%}):")
        lines.append(_group_stats(group))

    if not adder_freq:
        lines.append("  No adders recorded")
    return "\n".join(lines)


def _s10_conviction_multiplier(trades: list[TradeRecord]) -> str:
    """S10: Bucketed conviction multiplier vs R."""
    lines = [_hdr("10. Conviction Multiplier")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    buckets = [(0, 0.6, "<0.6"), (0.6, 0.8, "0.6-0.8"), (0.8, 1.0, "0.8-1.0"),
               (1.0, 1.2, "1.0-1.2"), (1.2, 999, ">1.2")]
    for lo, hi, label in buckets:
        group = [t for t in trades if lo <= (_meta(t, "conviction_multiplier", 1.0) or 1.0) < hi]
        if group:
            lines.append(f"  {label}:")
            lines.append(_group_stats(group))

    return "\n".join(lines)


def _s11_regime_tier(trades: list[TradeRecord]) -> str:
    """S11: A vs B regime tier -- regime gate value."""
    lines = [_hdr("11. Regime Tier")]
    by_tier: dict[str, list[TradeRecord]] = {}
    for t in trades:
        tier = t.regime_tier or "?"
        by_tier.setdefault(tier, []).append(t)

    for tier in sorted(by_tier):
        lines.append(f"  Tier {tier}:")
        lines.append(_group_stats(by_tier[tier]))
    return "\n".join(lines)


def _s12_regime_x_confidence(trades: list[TradeRecord]) -> str:
    """S12: Cross-tab regime × confidence."""
    lines = [_hdr("12. Regime × Confidence")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    regimes = sorted(set(t.regime_tier or "?" for t in trades))
    confs = sorted(set(_meta(t, "confidence", "?") or "?" for t in trades))

    hdr = f"  {'':10s}"
    for c in confs:
        hdr += f" {c:>12s}"
    lines.append(hdr)
    lines.append("  " + "-" * (10 + 13 * len(confs)))

    for regime in regimes:
        row = f"  {regime:<10s}"
        for c in confs:
            group = [t for t in trades if (t.regime_tier or "?") == regime
                     and (_meta(t, "confidence", "?") or "?") == c]
            if group:
                mean_r = float(np.mean([t.r_multiple for t in group]))
                row += f" {mean_r:>+8.3f}({len(group):>2})"
            else:
                row += f" {'--':>12s}"
        lines.append(row)
    return "\n".join(lines)


def _s13_timing_window(trades: list[TradeRecord]) -> str:
    """S13: 5 timing windows with multipliers."""
    lines = [_hdr("13. Timing Window")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    by_window: dict[str, list[TradeRecord]] = {}
    for t in trades:
        tw = _meta(t, "timing_window", "UNKNOWN") or "UNKNOWN"
        by_window.setdefault(tw, []).append(t)

    window_mults = {
        "09:35-10:30": 1.00, "10:30-12:00": 0.85, "12:00-13:30": 0.70,
        "13:30-14:30": 0.90, "14:30-15:00": 0.75,
    }

    for tw in ["09:35-10:30", "10:30-12:00", "12:00-13:30", "13:30-14:30", "14:30-15:00", "OUTSIDE", "UNKNOWN"]:
        if tw in by_window:
            mult = window_mults.get(tw, 0.0)
            lines.append(f"  {tw} (×{mult:.2f}):" if mult else f"  {tw}:")
            lines.append(_group_stats(by_window[tw]))

    return "\n".join(lines)


def _s14_exit_reason(trades: list[TradeRecord]) -> str:
    """S14: Exit reason deep dive."""
    lines = [_hdr("14. Exit Reason")]
    by_reason: dict[str, list[TradeRecord]] = {}
    for t in trades:
        reason = t.exit_reason or "UNKNOWN"
        by_reason.setdefault(reason, []).append(t)

    total = len(trades)
    for reason in sorted(by_reason, key=lambda r: -len(by_reason[r])):
        group = by_reason[reason]
        n = len(group)
        pct = n / total if total > 0 else 0
        avg_hold_bars = float(np.mean([t.hold_bars for t in group]))
        lines.append(f"  {reason} ({n}, {pct:.0%}):")
        lines.append(_group_stats(group))
        lines.append(f"      Avg hold: {avg_hold_bars:.0f} bars")

    return "\n".join(lines)


def _s15_time_stop(trades: list[TradeRecord]) -> str:
    """S15: Time stop analysis -- % triggered, avg R, MFE before stop."""
    lines = [_hdr("15. Time Stop Analysis")]
    time_stops = [t for t in trades if t.exit_reason == "TIME_STOP"]
    n = len(trades)

    if not time_stops:
        lines.append("  No time stop exits")
        return "\n".join(lines)

    lines.append(f"  Time stops: {len(time_stops)}/{n} ({len(time_stops)/n:.1%})")
    lines.append(_group_stats(time_stops))

    # MFE analysis: how much upside was reached before time stop
    if _has_metadata(trades):
        mfe_rs = [_meta(t, "mfe_r", 0) or 0 for t in time_stops]
        if mfe_rs:
            lines.append(f"  Mean MFE before time stop: {float(np.mean(mfe_rs)):.3f}R")
            lines.append(f"  % that reached 0.5R+ MFE: {sum(1 for m in mfe_rs if m >= 0.5)/len(mfe_rs):.0%}")

    return "\n".join(lines)


def _s16_partial_take(trades: list[TradeRecord]) -> str:
    """S16: Partial take -- capture rate, runner post-partial R."""
    lines = [_hdr("16. Partial Take Analysis")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    partial = [t for t in trades if _meta(t, "partial_taken", False)]
    no_partial = [t for t in trades if not _meta(t, "partial_taken", False)]
    n = len(trades)

    lines.append(f"  Partial hit rate: {len(partial)}/{n} ({len(partial)/n:.1%})" if n else "  N/A")
    lines.append(f"\n  With partial:")
    lines.append(_group_stats(partial))
    lines.append(f"  Without partial:")
    lines.append(_group_stats(no_partial))

    # Partial fraction analysis
    fracs = [_meta(t, "partial_qty_fraction", 0) or 0 for t in partial]
    if fracs:
        lines.append(f"\n  Avg partial fraction: {float(np.mean(fracs)):.0%}")

    return "\n".join(lines)


def _s17_carry_analysis(trades: list[TradeRecord]) -> str:
    """S17: Carry attempted %, held %, next-day outcome."""
    lines = [_hdr("17. Carry Analysis")]
    # In Tier 2 daily reset backtest, carry is simplified
    eod = [t for t in trades if t.exit_reason == "EOD_FLATTEN"]
    non_eod = [t for t in trades if t.exit_reason != "EOD_FLATTEN"]

    lines.append(f"  EOD flatten exits: {len(eod)} ({len(eod)/len(trades):.0%})" if trades else "  N/A")
    lines.append(_group_stats(eod))
    lines.append(f"\n  Non-EOD exits:")
    lines.append(_group_stats(non_eod))

    if _has_metadata(trades):
        setup_tags = [_meta(t, "setup_tag", "UNCLASSIFIED") for t in eod]
        tag_counts: dict[str, int] = {}
        for tag in setup_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if tag_counts:
            lines.append(f"\n  EOD flatten setup tags: {tag_counts}")

    return "\n".join(lines)


def _s18_avwap_breakdown(trades: list[TradeRecord]) -> str:
    """S18: AVWAP breakdown exits -- true breakdowns vs false positives."""
    lines = [_hdr("18. AVWAP Breakdown")]
    avwap_exits = [t for t in trades if t.exit_reason == "AVWAP_BREAKDOWN"]

    if not avwap_exits:
        lines.append("  No AVWAP breakdown exits")
        return "\n".join(lines)

    lines.append(f"  AVWAP breakdown exits: {len(avwap_exits)}")
    lines.append(_group_stats(avwap_exits))

    winners = [t for t in avwap_exits if t.is_winner]
    lines.append(f"  Winners (false positive exits): {len(winners)}/{len(avwap_exits)}")
    if winners:
        lines.append(f"    Avg R of false positives: {float(np.mean([t.r_multiple for t in winners])):+.3f}")

    return "\n".join(lines)


def _s19_mfe_mae(trades: list[TradeRecord]) -> str:
    """S19: MFE/MAE distributions, capture ratio, edge ratio."""
    lines = [_hdr("19. MFE / MAE Analysis")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    if winners:
        mfe_rs = [_meta(t, "mfe_r", 0) or 0 for t in winners]
        actual_rs = [t.r_multiple for t in winners]
        efficiencies = [a / m if m > 0 else 0 for a, m in zip(actual_rs, mfe_rs)]
        lines.append(f"  Winners ({len(winners)}):")
        lines.append(f"    Mean MFE: {float(np.mean(mfe_rs)):.3f}R")
        lines.append(f"    Capture efficiency (R/MFE): {float(np.mean(efficiencies)):.1%}")
        lines.append(f"    Mean giveback (MFE-R): {float(np.mean([m-a for m, a in zip(mfe_rs, actual_rs)])):.3f}R")

    if losers:
        mae_rs = [_meta(t, "mae_r", 0) or 0 for t in losers]
        lines.append(f"  Losers ({len(losers)}):")
        lines.append(f"    Mean MAE: {float(np.mean(mae_rs)):.3f}R")
        if mae_rs:
            p25, p50, p75 = np.percentile(mae_rs, [25, 50, 75])
            lines.append(f"    MAE distribution: P25={p25:.2f}R, P50={p50:.2f}R, P75={p75:.2f}R")

    # Edge ratio: avg winner MFE / avg loser MAE
    if winners and losers:
        avg_w_mfe = float(np.mean([_meta(t, "mfe_r", 0) or 0 for t in winners]))
        avg_l_mae = float(np.mean([_meta(t, "mae_r", 0) or 0 for t in losers]))
        edge_ratio = avg_w_mfe / avg_l_mae if avg_l_mae > 0 else float("inf")
        lines.append(f"\n  Edge ratio (W_MFE / L_MAE): {edge_ratio:.2f}")

    return "\n".join(lines)


def _s20_risk_unit_decomposition(trades: list[TradeRecord]) -> str:
    """S20: 6-factor risk unit breakdown."""
    lines = [_hdr("20. Risk Unit Decomposition")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    risk_units = [_meta(t, "risk_unit_final", 1.0) or 1.0 for t in trades]
    lines.append(f"  Risk unit distribution:")
    lines.append(f"    Mean: {float(np.mean(risk_units)):.3f}")
    lines.append(f"    Median: {float(np.median(risk_units)):.3f}")
    lines.append(f"    Range: [{min(risk_units):.3f}, {max(risk_units):.3f}]")

    # Factor-by-factor breakdown
    factors = [
        ("conviction_multiplier", "Conviction"),
        ("confidence", "Confidence (mult)"),
        ("location_grade", "Location (mult)"),
        ("timing_multiplier", "Timing"),
        ("regime_risk_multiplier", "Regime"),
    ]

    lines.append(f"\n  Factor distributions:")
    for key, label in factors:
        vals = [_meta(t, key, None) for t in trades]
        if key == "confidence":
            # Count by level
            counts: dict[str, int] = {}
            for v in vals:
                counts[v or "?"] = counts.get(v or "?", 0) + 1
            lines.append(f"    {label}: {counts}")
        elif key == "location_grade":
            counts = {}
            for v in vals:
                counts[v or "?"] = counts.get(v or "?", 0) + 1
            lines.append(f"    {label}: {counts}")
        else:
            numeric = [v for v in vals if v is not None and isinstance(v, (int, float))]
            if numeric:
                lines.append(f"    {label}: mean={float(np.mean(numeric)):.3f}, "
                             f"range=[{min(numeric):.3f}, {max(numeric):.3f}]")

    return "\n".join(lines)


def _s21_sector(trades: list[TradeRecord]) -> str:
    """S21: Sector breakdown."""
    lines = [_hdr("21. Sector Performance")]
    by_sector: dict[str, list[TradeRecord]] = {}
    for t in trades:
        sec = t.sector or "UNKNOWN"
        by_sector.setdefault(sec, []).append(t)

    sorted_sectors = sorted(by_sector.items(), key=lambda x: sum(t.r_multiple for t in x[1]), reverse=True)
    for sec, group in sorted_sectors:
        lines.append(f"  {sec}:")
        lines.append(_group_stats(group))
    return "\n".join(lines)


def _s22_monthly_pnl(trades: list[TradeRecord]) -> str:
    """S22: Monthly P&L table."""
    lines = [_hdr("22. Monthly P&L")]
    by_month: dict[str, list[TradeRecord]] = {}
    for t in trades:
        month = t.exit_time.strftime("%Y-%m")
        by_month.setdefault(month, []).append(t)

    lines.append(f"  {'Month':>8s} {'Trades':>6s} {'WR%':>5s} {'Net R':>8s} {'Cum R':>8s}")
    lines.append("  " + "-" * 40)

    cum_r = 0.0
    for month in sorted(by_month):
        group = by_month[month]
        n = len(group)
        wr = sum(1 for t in group if t.is_winner) / n if n else 0
        net_r = sum(t.r_multiple for t in group)
        cum_r += net_r
        lines.append(f"  {month:>8s} {n:>6} {wr:>4.0%} {net_r:>+8.2f} {cum_r:>+8.2f}")
    return "\n".join(lines)


def _s23_day_of_week(trades: list[TradeRecord]) -> str:
    """S23: Day-of-week WR/R."""
    lines = [_hdr("23. Day of Week")]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    by_day: dict[int, list[TradeRecord]] = {}
    for t in trades:
        dow = t.entry_time.weekday()
        by_day.setdefault(dow, []).append(t)

    for i, name in enumerate(days):
        group = by_day.get(i, [])
        if group:
            lines.append(f"  {name}:")
            lines.append(_group_stats(group))
    return "\n".join(lines)


def _s24_hold_duration(trades: list[TradeRecord]) -> str:
    """S24: R by hold-time bucket (in 5m bars)."""
    lines = [_hdr("24. Hold Duration")]
    if not trades:
        return "\n".join(lines)

    # Bucket by bar count
    buckets = [
        (0, 6, "0-30min"),
        (6, 12, "30-60min"),
        (12, 24, "1-2h"),
        (24, 48, "2-4h"),
        (48, 999, "4h+"),
    ]

    for lo, hi, label in buckets:
        group = [t for t in trades if lo <= t.hold_bars < hi]
        if group:
            lines.append(f"  {label}:")
            lines.append(_group_stats(group))
    return "\n".join(lines)


def _s25_rolling_expectancy(trades: list[TradeRecord]) -> str:
    """S25: Rolling 20-trade expectancy trend."""
    lines = [_hdr("25. Rolling Expectancy (20-trade window)")]
    if len(trades) < 20:
        lines.append("  Insufficient trades for rolling analysis")
        return "\n".join(lines)

    rs = [t.r_multiple for t in trades]
    window = 20
    rolling = [float(np.mean(rs[i:i+window])) for i in range(len(rs) - window + 1)]

    lines.append(f"  Start: {rolling[0]:+.3f}R  |  End: {rolling[-1]:+.3f}R")
    lines.append(f"  Min: {min(rolling):+.3f}R  |  Max: {max(rolling):+.3f}R")

    first_half = float(np.mean(rolling[:len(rolling)//2]))
    second_half = float(np.mean(rolling[len(rolling)//2:]))
    if second_half > first_half + 0.05:
        trend = "IMPROVING"
    elif second_half < first_half - 0.05:
        trend = "DEGRADING"
    else:
        trend = "STABLE"
    lines.append(f"  Trend: {trend} (1st half: {first_half:+.3f}R, 2nd half: {second_half:+.3f}R)")
    return "\n".join(lines)


def _s26_winner_loser_profiles(trades: list[TradeRecord]) -> str:
    """S26: Avg MFE/MAE/hold/R for winners vs losers."""
    lines = [_hdr("26. Winner vs Loser Profiles")]
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    if not winners or not losers:
        lines.append("  Need both winners and losers for comparison")
        return "\n".join(lines)

    def _avg_meta(ts: list[TradeRecord], key: str, default=0):
        vals = [_meta(t, key, default) or default for t in ts]
        return float(np.mean(vals)) if vals else 0

    lines.append(f"  {'Metric':<25s} {'Winners':>10s} {'Losers':>10s} {'Delta':>10s}")
    lines.append("  " + "-" * 58)

    metrics = [
        ("Avg R", lambda ts: float(np.mean([t.r_multiple for t in ts]))),
        ("Avg Hold (bars)", lambda ts: float(np.mean([t.hold_bars for t in ts]))),
    ]

    if _has_metadata(trades):
        metrics.extend([
            ("Risk Unit", lambda ts: _avg_meta(ts, "risk_unit_final", 1.0)),
            ("Timing Mult", lambda ts: _avg_meta(ts, "timing_multiplier", 1.0)),
            ("MFE (R)", lambda ts: _avg_meta(ts, "mfe_r", 0)),
            ("MAE (R)", lambda ts: _avg_meta(ts, "mae_r", 0)),
            ("Drop from HOD %", lambda ts: _avg_meta(ts, "drop_from_hod_pct", 0) * 100),
        ])

    for name, fn in metrics:
        w_val = fn(winners)
        l_val = fn(losers)
        delta = w_val - l_val
        lines.append(f"  {name:<25s} {w_val:>10.3f} {l_val:>10.3f} {delta:>+10.3f}")

    # Setup type distribution comparison
    if _has_metadata(trades):
        lines.append("")
        lines.append("  Setup Type Distribution:")
        for label, subset in [("Winners", winners), ("Losers", losers)]:
            types = defaultdict(int)
            for t in subset:
                types[_meta(t, "setup_type", "?")] += 1
            dist = ", ".join(f"{k}:{v}" for k, v in sorted(types.items()))
            lines.append(f"    {label}: {dist}")

    return "\n".join(lines)


def _s27_drawdown_episodes(trades: list[TradeRecord]) -> str:
    """S27: Top 5 drawdown episodes, clustering."""
    lines = [_hdr("27. Drawdown Episodes")]
    if not trades:
        return "\n".join(lines)

    rs = [t.r_multiple for t in trades]
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak

    max_dd = float(np.min(dd))
    max_dd_idx = int(np.argmin(dd))

    lines.append(f"  Max drawdown: {max_dd:+.2f}R (at trade #{max_dd_idx + 1})")

    # Find DD episodes
    episodes: list[tuple[int, int, float]] = []  # (start, end, depth)
    in_dd = False
    ep_start = 0
    ep_depth = 0.0
    for i, d in enumerate(dd):
        if d < 0:
            if not in_dd:
                in_dd = True
                ep_start = i
                ep_depth = d
            else:
                ep_depth = min(ep_depth, d)
        else:
            if in_dd:
                episodes.append((ep_start, i, ep_depth))
                in_dd = False
    if in_dd:
        episodes.append((ep_start, len(dd) - 1, ep_depth))

    episodes.sort(key=lambda x: x[2])
    lines.append(f"  DD episodes: {len(episodes)}")
    lines.append(f"\n  Top 5 deepest episodes:")
    for ep_start, ep_end, ep_depth in episodes[:5]:
        length = ep_end - ep_start
        # Cluster by sector/regime
        ep_trades = trades[ep_start:ep_end + 1]
        sectors = defaultdict(int)
        for t in ep_trades:
            sectors[t.sector or "?"] += 1
        top_sector = max(sectors, key=sectors.get) if sectors else "?"
        lines.append(f"    Trades #{ep_start+1}-#{ep_end+1}: {ep_depth:+.2f}R over {length} trades, "
                      f"dominant sector: {top_sector}")

    # Recovery
    if max_dd < 0:
        recovery_trades = 0
        for i in range(max_dd_idx, len(cum)):
            if cum[i] >= peak[max_dd_idx]:
                recovery_trades = i - max_dd_idx
                break
        if recovery_trades > 0:
            lines.append(f"\n  Recovery from worst DD: {recovery_trades} trades")
        else:
            lines.append(f"\n  Recovery from worst DD: not recovered")

    return "\n".join(lines)


def _s28_stale_penalty(trades: list[TradeRecord]) -> str:
    """S28: Stale 0.85x trades vs non-stale."""
    lines = [_hdr("28. Stale Penalty Analysis")]
    if not _has_metadata(trades):
        lines.append("  (no metadata)")
        return "\n".join(lines)

    stale = [t for t in trades if (_meta(t, "sponsorship_state", "") or "") in ("STALE",)]
    non_stale = [t for t in trades if (_meta(t, "sponsorship_state", "") or "") not in ("STALE",)]

    lines.append(f"  Stale (0.85x penalty):")
    lines.append(_group_stats(stale))
    lines.append(f"  Non-stale:")
    lines.append(_group_stats(non_stale))

    if stale and non_stale:
        stale_mean = float(np.mean([t.r_multiple for t in stale]))
        non_stale_mean = float(np.mean([t.r_multiple for t in non_stale]))
        lines.append(f"\n  Stale penalty impact: {stale_mean - non_stale_mean:+.3f}R per trade")
        lines.append(f"  Stale trades as % of total: {len(stale)/len(trades):.0%}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def iaric_full_diagnostic(
    trades: list[TradeRecord],
    fsm_log: list[dict] | None = None,
    rejection_log: list[dict] | None = None,
    shadow_tracker: IARICShadowTracker | None = None,
    daily_selections: dict | None = None,
) -> str:
    """Generate the full 28-section IARIC diagnostic report.

    Parameters
    ----------
    trades : list[TradeRecord]
        Completed trades from IARIC Tier 2 backtest (with enriched metadata).
    fsm_log : list[dict], optional
        FSM state transition log from engine.
    rejection_log : list[dict], optional
        Setup rejection log from engine.
    shadow_tracker : IARICShadowTracker, optional
        Shadow tracker with rejected setup simulations.
    daily_selections : dict, optional
        {date: WatchlistArtifact} from engine result for funnel analysis.
    """
    if not trades:
        return "No trades to diagnose."

    sections = [
        _s01_overview(trades),
        _s02_signal_funnel(trades, daily_selections, shadow_tracker),
        _s03_setup_type(trades),
        _s04_confidence_level(trades),
        _s05_location_grade(trades),
        _s06_sponsorship_state(trades),
        _s07_micropressure(trades),
        _s08_acceptance_count(trades),
        _s09_conviction_adders(trades),
        _s10_conviction_multiplier(trades),
        _s11_regime_tier(trades),
        _s12_regime_x_confidence(trades),
        _s13_timing_window(trades),
        _s14_exit_reason(trades),
        _s15_time_stop(trades),
        _s16_partial_take(trades),
        _s17_carry_analysis(trades),
        _s18_avwap_breakdown(trades),
        _s19_mfe_mae(trades),
        _s20_risk_unit_decomposition(trades),
        _s21_sector(trades),
        _s22_monthly_pnl(trades),
        _s23_day_of_week(trades),
        _s24_hold_duration(trades),
        _s25_rolling_expectancy(trades),
        _s26_winner_loser_profiles(trades),
        _s27_drawdown_episodes(trades),
        _s28_stale_penalty(trades),
    ]

    # Append filter attribution if shadow tracker available
    if shadow_tracker and shadow_tracker.completed:
        actual_wr = sum(1 for t in trades if t.is_winner) / len(trades) if trades else 0
        sections.append(iaric_filter_attribution_report(shadow_tracker, actual_wr))

    # Deep dive appendices (only when data is available)
    if fsm_log:
        from research.backtests.stock.analysis.iaric_fsm_analysis import iaric_fsm_analysis
        sections.append(iaric_fsm_analysis(fsm_log, rejection_log or [], trades))

    if fsm_log or rejection_log:
        from research.backtests.stock.analysis.iaric_timing_analysis import iaric_timing_analysis
        sections.append(iaric_timing_analysis(trades, fsm_log, rejection_log))

    if len(trades) >= 10:
        from research.backtests.stock.analysis.iaric_micropressure_diagnostics import (
            iaric_micropressure_diagnostics,
        )
        sections.append(iaric_micropressure_diagnostics(trades, rejection_log))

        from research.backtests.stock.analysis.iaric_conviction_analysis import (
            iaric_conviction_analysis,
        )
        sections.append(iaric_conviction_analysis(trades))

    # --- New diagnostic modules (Priorities 2-7) ---
    if len(trades) >= 10:
        from research.backtests.stock.analysis.iaric_stop_calibration import (
            iaric_stop_calibration,
        )
        sections.append(iaric_stop_calibration(trades))

        from research.backtests.stock.analysis.iaric_flow_reversal_attribution import (
            iaric_flow_reversal_attribution,
        )
        sections.append(iaric_flow_reversal_attribution(trades))

        from research.backtests.stock.analysis.iaric_dow_conditional import (
            iaric_dow_conditional,
        )
        sections.append(iaric_dow_conditional(trades))

        from research.backtests.stock.analysis.iaric_carry_optimization import (
            iaric_carry_optimization,
        )
        sections.append(iaric_carry_optimization(trades))

        from research.backtests.stock.analysis.iaric_tier_selectivity import (
            iaric_tier_selectivity,
        )
        sections.append(iaric_tier_selectivity(trades))

    # Rejected trade simulation (requires shadow tracker)
    if shadow_tracker:
        from research.backtests.stock.analysis.iaric_rejected_trade_sim import (
            iaric_rejected_trade_sim,
        )
        sections.append(iaric_rejected_trade_sim(trades, shadow_tracker))

    return "\n".join(sections)
