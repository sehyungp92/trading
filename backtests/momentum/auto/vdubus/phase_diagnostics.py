"""VdubusNQ phase diagnostics -- stale exits, multi-session, capture, session breakdown."""
from __future__ import annotations

from typing import Any

from .scoring import VdubusMetrics


def _net_trade_pnl(trade: Any) -> float:
    return float(getattr(trade, "pnl_dollars", 0.0) or 0.0) - float(getattr(trade, "commission", 0.0) or 0.0)


def generate_phase_diagnostics(
    phase: int,
    metrics: VdubusMetrics,
    greedy_result: dict | None,
    state_dict: dict | None,
    all_trades: list | None = None,
    force_all_modules: bool = False,
) -> str:
    """Generate VdubusNQ-specific phase diagnostics report."""
    lines: list[str] = []
    lines.append(f"{'='*60}")
    lines.append(f"VdubusNQ Phase {phase} Diagnostics")
    lines.append(f"{'='*60}")

    # D1: Core performance (always)
    lines.append("\n--- D1: Core Performance ---")
    lines.append(f"Total trades:    {metrics.total_trades}")
    lines.append(f"Win rate:        {metrics.win_rate:.1%}")
    lines.append(f"Profit factor:   {metrics.profit_factor:.2f}")
    lines.append(f"Net return:      {metrics.net_return_pct:.1f}%")
    lines.append(f"Max drawdown:    {metrics.max_dd_pct:.2%}")
    lines.append(f"Calmar:          {metrics.calmar:.2f}")
    lines.append(f"Sharpe:          {metrics.sharpe:.2f}")
    lines.append(f"Sortino:         {metrics.sortino:.2f}")
    lines.append(f"Avg R:           {metrics.avg_r:.3f}")

    # D2: Exit efficiency (always)
    lines.append("\n--- D2: Exit Efficiency ---")
    lines.append(f"Capture ratio:   {metrics.capture_ratio:.3f} (winners exit_R / MFE)")
    lines.append(f"Stale exit pct:  {metrics.stale_exit_pct:.1%}")
    lines.append(f"Multi-session:   {metrics.multi_session_pct:.1%}")
    lines.append(f"Fast deaths:     {metrics.fast_death_pct:.1%} (<=4 bars)")
    lines.append(f"Avg winner R:    {metrics.avg_winner_r:.3f}")
    lines.append(f"Avg loser R:     {metrics.avg_loser_r:.3f}")
    lines.append(f"Avg MFE R:       {metrics.avg_mfe_r:.3f}")
    lines.append(f"Avg hold hours:  {metrics.avg_hold_hours:.1f}")

    # D3: Session breakdown (phase >= 2 or force)
    if phase >= 2 or force_all_modules:
        lines.append("\n--- D3: Session Breakdown ---")
        lines.append(f"Evening trades:  {metrics.evening_trade_pct:.1%} of total")
        lines.append(f"Evening avg R:   {metrics.evening_avg_r:+.3f}")

        if all_trades:
            _session_breakdown(lines, all_trades)

    # D4: Exit reason distribution (always)
    if all_trades:
        lines.append("\n--- D4: Exit Reasons ---")
        _exit_reason_breakdown(lines, all_trades)

    # D5: Hold duration analysis (phase >= 2 or force)
    if (phase >= 2 or force_all_modules) and all_trades:
        lines.append("\n--- D5: Hold Duration ---")
        _hold_duration_analysis(lines, all_trades)

    # D6: Greedy result summary (always)
    if greedy_result:
        lines.append("\n--- D6: Greedy Result ---")
        lines.append(f"Base score:      {greedy_result.get('base_score', 0):.4f}")
        lines.append(f"Final score:     {greedy_result.get('final_score', 0):.4f}")
        lines.append(f"Accepted:        {greedy_result.get('accepted_count', 0)}")
        lines.append(f"Total candidates:{greedy_result.get('total_candidates', 0)}")
        kept = greedy_result.get("kept_features", [])
        if kept:
            lines.append(f"Kept features:   {', '.join(kept)}")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def get_diagnostic_gaps(phase: int, metrics: VdubusMetrics) -> list[str]:
    """Identify diagnostic gaps for the current phase."""
    gaps: list[str] = []

    if metrics.capture_ratio < 0.45:
        gaps.append("Low MFE capture ratio -- exits leave alpha on the table")
    if metrics.stale_exit_pct > 0.40:
        gaps.append("High stale exit rate -- trail/MFE protection needed")
    if metrics.fast_death_pct > 0.20:
        gaps.append("Many fast deaths (<=4 bars) -- entry precision issue")
    if metrics.total_trades < 120:
        gaps.append("Low trade frequency -- signal gates may be too restrictive")
    if metrics.max_dd_pct > 0.15:
        gaps.append("Elevated drawdown -- risk controls may need tightening")
    if metrics.profit_factor < 2.0:
        gaps.append("Low profit factor -- edge may be degraded")
    if metrics.evening_trade_pct > 0.08 and metrics.evening_avg_r < -0.10:
        gaps.append("Evening trades losing -- consider blocking or tightening")

    return gaps


def _session_breakdown(lines: list[str], trades: list) -> None:
    """Break down performance by entry session."""
    buckets: dict[str, list] = {}
    for t in trades:
        session = getattr(t, 'entry_session', 'UNKNOWN')
        if not session:
            session = 'UNKNOWN'
        buckets.setdefault(session, []).append(t)

    for key in sorted(buckets.keys()):
        group = buckets[key]
        count = len(group)
        wins = sum(1 for t in group if t.r_multiple > 0)
        wr = wins / count if count else 0
        avg_r = sum(t.r_multiple for t in group) / count if count else 0
        total_pnl = sum(_net_trade_pnl(t) for t in group)
        lines.append(f"  {key:12s}: {count:3d} trades, WR={wr:.0%}, avgR={avg_r:+.3f}, PnL=${total_pnl:+,.0f}")


def _exit_reason_breakdown(lines: list[str], trades: list) -> None:
    """Break down performance by exit reason."""
    buckets: dict[str, list] = {}
    for t in trades:
        buckets.setdefault(t.exit_reason, []).append(t)

    for reason in sorted(buckets.keys()):
        group = buckets[reason]
        count = len(group)
        avg_r = sum(t.r_multiple for t in group) / count if count else 0
        total_pnl = sum(_net_trade_pnl(t) for t in group)
        avg_mfe = sum(t.mfe_r for t in group) / count if count else 0
        lines.append(f"  {reason:18s}: {count:3d} trades, avgR={avg_r:+.3f}, avgMFE={avg_mfe:.3f}, PnL=${total_pnl:+,.0f}")


def _hold_duration_analysis(lines: list[str], trades: list) -> None:
    """Analyze performance by hold duration bucket."""
    buckets = {"1-4 bars": [], "5-12 bars": [], "13-32 bars": [], "33+ bars": []}
    for t in trades:
        b = t.bars_held_15m
        if b <= 4:
            buckets["1-4 bars"].append(t)
        elif b <= 12:
            buckets["5-12 bars"].append(t)
        elif b <= 32:
            buckets["13-32 bars"].append(t)
        else:
            buckets["33+ bars"].append(t)

    for label, group in buckets.items():
        if not group:
            continue
        count = len(group)
        wins = sum(1 for t in group if t.r_multiple > 0)
        wr = wins / count if count else 0
        avg_r = sum(t.r_multiple for t in group) / count if count else 0
        total_pnl = sum(_net_trade_pnl(t) for t in group)
        lines.append(f"  {label:12s}: {count:3d} trades, WR={wr:.0%}, avgR={avg_r:+.3f}, PnL=${total_pnl:+,.0f}")
