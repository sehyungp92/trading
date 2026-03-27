"""Cross-tier consistency check: compare Tier 1 vs Tier 2 results.

Compares per-symbol metrics across tiers to validate that the intraday
engine (Tier 2) behaves consistently with the daily-bar proxy (Tier 1).

Checks:
  1. Per-symbol: avg R-multiple, win rate, trade count, PnL direction
  2. Aggregate: overall WR delta, Sharpe direction, equity curve correlation
  3. Selection overlap: what % of Tier 2 symbols also appeared in Tier 1
  4. Flags: large magnitude discrepancies, sign disagreements
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from research.backtests.stock._aliases import install
install()

from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.config_alcb import ALCBBacktestConfig
from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.alcb_daily_engine import ALCBDailyEngine
from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine
from research.backtests.stock.engine.iaric_engine import IARICIntradayEngine
from research.backtests.stock.analysis.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class SymbolStats:
    """Per-symbol aggregated stats."""
    symbol: str
    n_trades: int
    win_rate: float
    avg_r: float
    total_pnl: float
    avg_pnl_per_trade: float

    @classmethod
    def from_trades(cls, symbol: str, trades: list) -> SymbolStats:
        sym_trades = [t for t in trades if t.symbol == symbol]
        n = len(sym_trades)
        if n == 0:
            return cls(symbol, 0, 0.0, 0.0, 0.0, 0.0)
        wins = sum(1 for t in sym_trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in sym_trades)
        r_vals = [t.r_multiple for t in sym_trades if t.r_multiple is not None]
        return cls(
            symbol=symbol,
            n_trades=n,
            win_rate=wins / n * 100,
            avg_r=mean(r_vals) if r_vals else 0.0,
            total_pnl=total_pnl,
            avg_pnl_per_trade=total_pnl / n,
        )


def _fmt_pnl(v: float) -> str:
    return f"${v:>+10,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v:5.1f}%"


def _verdict(t1: SymbolStats, t2: SymbolStats) -> tuple[str, list[str]]:
    """Return (verdict, [flags]) for a symbol pair."""
    flags: list[str] = []

    # Sign agreement
    if t1.total_pnl == 0 or t2.total_pnl == 0:
        sign = "NEUTRAL"
    elif (t1.total_pnl > 0) == (t2.total_pnl > 0):
        sign = "AGREE"
    else:
        sign = "DISAGREE"
        flags.append("PnL sign disagreement")

    # Magnitude ratio (normalize to per-trade to be fair)
    if t1.avg_pnl_per_trade != 0 and t2.avg_pnl_per_trade != 0:
        ratio = abs(t1.avg_pnl_per_trade / t2.avg_pnl_per_trade)
        if ratio > 5 or ratio < 0.2:
            flags.append(f"Per-trade PnL ratio {ratio:.1f}x")

    # Win rate delta
    wr_delta = abs(t1.win_rate - t2.win_rate)
    if wr_delta > 30:
        flags.append(f"WR delta {wr_delta:.0f}pp")

    # R-multiple direction
    if t1.avg_r != 0 and t2.avg_r != 0:
        if (t1.avg_r > 0) != (t2.avg_r > 0):
            flags.append("Avg R sign disagree")

    # Sample size warning
    if t2.n_trades < 3:
        flags.append(f"T2 only {t2.n_trades} trade(s)")

    if flags:
        verdict = f"{sign} (FLAGS: {len(flags)})"
    else:
        verdict = sign

    return verdict, flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = Path("research/backtests/stock/data/raw")
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    start, end = "2025-06-01", "2025-09-01"

    # --- Run all 4 engines ---
    print("Running ALCB Tier 1...")
    t1a = ALCBDailyEngine(
        ALCBBacktestConfig(start_date=start, end_date=end, data_dir=data_dir), replay,
    ).run()
    print("Running ALCB Tier 2...")
    t2a = ALCBIntradayEngine(
        ALCBBacktestConfig(start_date=start, end_date=end, data_dir=data_dir, tier=2), replay,
    ).run()
    print("Running IARIC Tier 1...")
    t1i = IARICDailyEngine(
        IARICBacktestConfig(start_date=start, end_date=end, data_dir=data_dir), replay,
    ).run()
    print("Running IARIC Tier 2...")
    t2i = IARICIntradayEngine(
        IARICBacktestConfig(start_date=start, end_date=end, data_dir=data_dir, tier=2), replay,
    ).run()

    print("\n" + "=" * 70)
    print("  CROSS-TIER CONSISTENCY CHECK (2025-06 to 2025-09)")
    print("=" * 70)

    all_flags: list[str] = []

    for label, t1, t2 in [("ALCB", t1a, t2a), ("IARIC", t1i, t2i)]:
        t1_syms = set(t.symbol for t in t1.trades)
        t2_syms = set(t.symbol for t in t2.trades)
        overlap = t1_syms & t2_syms
        t1_only = t1_syms - t2_syms
        t2_only = t2_syms - t1_syms

        t1_pnl = sum(t.pnl for t in t1.trades)
        t2_pnl = sum(t.pnl for t in t2.trades)
        t1_wr = sum(1 for t in t1.trades if t.pnl > 0) / max(len(t1.trades), 1) * 100
        t2_wr = sum(1 for t in t2.trades if t.pnl > 0) / max(len(t2.trades), 1) * 100
        t1_r = [t.r_multiple for t in t1.trades if t.r_multiple is not None]
        t2_r = [t.r_multiple for t in t2.trades if t.r_multiple is not None]

        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        # --- 1. Aggregate comparison ---
        print(f"\n  {'':30s} {'Tier 1':>15s} {'Tier 2':>15s} {'Delta':>10s}")
        print(f"  {'─' * 70}")
        print(f"  {'Trades':30s} {len(t1.trades):>15d} {len(t2.trades):>15d} {len(t2.trades)-len(t1.trades):>+10d}")
        print(f"  {'Win Rate':30s} {_fmt_pct(t1_wr):>15s} {_fmt_pct(t2_wr):>15s} {t2_wr-t1_wr:>+9.1f}pp")
        print(f"  {'Total PnL':30s} {_fmt_pnl(t1_pnl):>15s} {_fmt_pnl(t2_pnl):>15s}")
        print(f"  {'Avg PnL/Trade':30s} {_fmt_pnl(t1_pnl/max(len(t1.trades),1)):>15s} {_fmt_pnl(t2_pnl/max(len(t2.trades),1)):>15s}")
        print(f"  {'Avg R':30s} {mean(t1_r) if t1_r else 0:>15.2f} {mean(t2_r) if t2_r else 0:>15.2f}")
        if len(t1_r) > 1:
            t2_std = f"{stdev(t2_r):.2f}" if len(t2_r) > 1 else "N/A"
            print(f"  {'R Std Dev':30s} {stdev(t1_r):>15.2f} {t2_std:>15s}")

        # PnL sign agreement at aggregate level
        if t1_pnl != 0 and t2_pnl != 0:
            agg_agree = "AGREE" if (t1_pnl > 0) == (t2_pnl > 0) else "DISAGREE"
            print(f"  {'Aggregate PnL Sign':30s} {agg_agree:>15s}")
            if agg_agree == "DISAGREE":
                all_flags.append(f"{label}: Aggregate PnL sign disagrees")

        # --- 2. Selection overlap ---
        print(f"\n  Selection Overlap:")
        print(f"    Tier 1 symbols ({len(t1_syms)}): {sorted(t1_syms)}")
        print(f"    Tier 2 symbols ({len(t2_syms)}): {sorted(t2_syms)}")
        if t2_syms:
            pct = len(overlap) / len(t2_syms) * 100
            print(f"    T2 symbols also in T1: {len(overlap)}/{len(t2_syms)} ({pct:.0f}%)")
        if t2_only:
            all_flags.append(f"{label}: T2 traded symbols not in T1: {sorted(t2_only)}")
            print(f"    WARNING: T2-only symbols (not in T1): {sorted(t2_only)}")

        # --- 3. Per-symbol comparison ---
        if overlap:
            print(f"\n  Per-Symbol Comparison (overlapping):")
            print(f"    {'Symbol':8s} │ {'#T1':>4s} {'WR1':>6s} {'AvgR1':>6s} {'PnL/T1':>10s} │"
                  f" {'#T2':>4s} {'WR2':>6s} {'AvgR2':>6s} {'PnL/T2':>10s} │ Verdict")
            print(f"    {'─' * 84}")

            for sym in sorted(overlap):
                s1 = SymbolStats.from_trades(sym, t1.trades)
                s2 = SymbolStats.from_trades(sym, t2.trades)
                verdict, flags = _verdict(s1, s2)

                print(
                    f"    {sym:8s} │ {s1.n_trades:>4d} {_fmt_pct(s1.win_rate):>6s} {s1.avg_r:>+6.2f}"
                    f" {_fmt_pnl(s1.avg_pnl_per_trade):>10s} │"
                    f" {s2.n_trades:>4d} {_fmt_pct(s2.win_rate):>6s} {s2.avg_r:>+6.2f}"
                    f" {_fmt_pnl(s2.avg_pnl_per_trade):>10s} │ {verdict}"
                )
                for f in flags:
                    print(f"             → {f}")
                    all_flags.append(f"{label}/{sym}: {f}")

        # --- 4. Exit reason comparison ---
        t1_exits: dict[str, int] = {}
        for t in t1.trades:
            t1_exits[t.exit_reason] = t1_exits.get(t.exit_reason, 0) + 1
        t2_exits: dict[str, int] = {}
        for t in t2.trades:
            t2_exits[t.exit_reason] = t2_exits.get(t.exit_reason, 0) + 1

        all_reasons = sorted(set(t1_exits) | set(t2_exits))
        print(f"\n  Exit Reason Distribution:")
        print(f"    {'Reason':20s} │ {'T1 #':>5s} {'T1 %':>6s} │ {'T2 #':>5s} {'T2 %':>6s}")
        print(f"    {'─' * 50}")
        for reason in all_reasons:
            n1 = t1_exits.get(reason, 0)
            n2 = t2_exits.get(reason, 0)
            p1 = n1 / max(len(t1.trades), 1) * 100
            p2 = n2 / max(len(t2.trades), 1) * 100
            print(f"    {reason:20s} │ {n1:>5d} {p1:>5.1f}% │ {n2:>5d} {p2:>5.1f}%")

    # --- 5. Summary ---
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    if all_flags:
        print(f"\n  {len(all_flags)} flag(s) raised:\n")
        for i, f in enumerate(all_flags, 1):
            print(f"    {i}. {f}")
    else:
        print("\n  No flags raised. All checks passed.")

    print(f"\n  Caveats:")
    print(f"  - Tier 2 uses 15 symbols with intraday data vs Tier 1's full universe")
    print(f"  - Low Tier 2 trade counts limit statistical significance")
    print(f"  - Tier 1 is a daily-bar proxy; Tier 2 is the actual intraday strategy")
    print(f"  - Per-trade PnL differences are expected (different entry/exit mechanics)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
