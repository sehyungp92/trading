"""Deterministic IARIC control/variant runner used to establish a real baseline.

Runs one or more named mutation sets against the same replay bundle and prints a
compact comparison table.  Every economic claim about IARIC must be reproducible
through this entry point.

Usage::

    python -m backtests.stock.auto.iaric.run_baseline --variants control
    python -m backtests.stock.auto.iaric.run_baseline --variants control,event
    python -m backtests.stock.auto.iaric.run_baseline --variants all --folds
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path("backtests/stock/data/raw")
# 30m history for 25 megacap symbols (AAPL, MSFT, NVDA, AMZN, META, GOOGL, ...)
# only begins 2024-03-21/22, so any earlier start trades a universe with the
# megacap core invisible.  Start after the gap so all 98 names have coverage.
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0

# Chronological folds used for robustness.  A candidate must not depend on a
# single regime window to beat the control.
FOLDS = [
    ("2024 H1", "2024-03-25", "2024-06-30"),
    ("2024 H2", "2024-07-01", "2024-12-31"),
    ("2025 H1", "2025-01-01", "2025-06-30"),
    ("2025 H2+", "2025-07-01", "2026-03-01"),
]


def _control_mutations() -> dict:
    """The reference configuration every variant here is expressed as a delta from.

    This deliberately does NOT live in a round directory: rounds get archived,
    and when round_1 was archived this loader broke.  It is also not the *current*
    round's config -- if it were, each promotion would silently re-base every
    variant and the ablations would stop being comparable across rounds.
    """
    path = Path(__file__).resolve().parent / "reference_config.json"
    if not path.exists():  # pre-decoupling layout
        path = Path("backtests/output/stock/iaric/round_1/optimized_config.json")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def variant_definitions() -> dict[str, dict]:
    base = _control_mutations()

    def derive(**over) -> dict:
        out = dict(base)
        out.update({f"param_overrides.{k}": v for k, v in over.items()})
        return out

    variants: dict[str, dict] = {}

    # 0. The honest starting point: score-as-trigger, no event, no confirmation.
    variants["control"] = dict(base)

    # 1. Reversion conjunction only (old recommendation 3).
    variants["oversold"] = derive(pb_v2_open_scored_trigger_policy="oversold")

    # 2. Dislocation event requirement, single and multi.
    variants["dislocation"] = derive(pb_v2_open_scored_trigger_policy="dislocation")
    variants["multi_disloc"] = derive(pb_v2_open_scored_trigger_policy="multi_dislocation")

    # 3. The completed-bar reclaim event alone (no dislocation requirement).
    variants["reclaim_event"] = derive(
        pb_v2_open_scored_confirmation_policy="band_reclaim",
    )

    # 4. Full event stack: dislocation + completed-bar band reclaim + RVOL.
    variants["event"] = derive(
        pb_v2_open_scored_trigger_policy="dislocation",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_v2_open_scored_rvol_min=1.15,
        pb_v2_open_scored_after_bar=0,
        pb_v2_open_scored_entry_window_bars=6,
    )

    # 4b. Event stack with risk anchored on the reclaim instead of the session low.
    variants["event_stop"] = derive(
        pb_v2_open_scored_trigger_policy="dislocation",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_v2_open_scored_rvol_min=1.15,
        pb_v2_open_scored_entry_window_bars=6,
        pb_v2_event_stop_anchor="reclaim_bar",
    )

    # 4c. Reclaim event, no dislocation/RVOL/window narrowing, event-anchored stop.
    variants["reclaim_stop"] = derive(
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_v2_event_stop_anchor="reclaim_bar",
    )

    # 4d. Holding-period hypothesis.  A daily-pullback reversion needs days, but
    # the config forces an intraday EMA-touch exit at +0.03R and flattens at EOD,
    # so a trade that would revert tomorrow is stopped today.  Carry can never
    # engage because the intraday exit always fires first.
    def _hold(**over):
        base_hold = dict(
            pb_carry_enabled=True,
            pb_carry_min_r=0.0,
            pb_max_hold_days=5,
            pb_open_scored_max_hold_days=5,
            pb_open_scored_carry_mfe_gate_r=0.0,
            pb_open_scored_carry_close_pct_min=0.0,
            pb_carry_mfe_gate_r=0.0,
            pb_carry_close_pct_min=0.0,
            pb_v2_ema_reversion_exit=False,
            pb_backtest_intraday_universe_only=True,
        )
        base_hold.update(over)
        return derive(**base_hold)

    variants["control_hold"] = _hold()
    variants["oversold_hold"] = _hold(pb_v2_open_scored_trigger_policy="oversold")
    variants["multi_hold"] = _hold(pb_v2_open_scored_trigger_policy="multi_dislocation")
    variants["event_hold"] = _hold(
        pb_v2_open_scored_trigger_policy="dislocation",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_v2_open_scored_rvol_min=1.15,
        pb_v2_open_scored_entry_window_bars=6,
    )

    # 4e. Entry-quality layers on top of the holding repair.  The remaining
    # bleed is 56% stop-outs at -0.397R against 33% RSI exits at +1.143R, so the
    # only lever left is selecting which dislocations actually revert.
    variants["oversold_score"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_entry_score_family="reversion_event_v1",
    )
    variants["oversold_event"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
    )
    variants["oversold_event_score"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_entry_score_family="reversion_event_v1",
    )
    variants["oversold_rvol"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_v2_open_scored_rvol_min=1.15,
        pb_entry_score_family="reversion_event_v1",
    )

    # 5. Proposed baseline.
    #
    # Four repairs, each independently motivated and each measured:
    #   (a) holding repair  -- a daily-pullback reversion needs days; the forced
    #       intraday EMA-touch exit at +0.03R was the dominant defect.
    #   (b) oversold conjunction -- the reversion condition is now required
    #       rather than being one member of a seven-way OR.
    #   (c) band_reclaim event -- a discrete completed-bar dislocation->reclaim,
    #       so the route has an event instead of using its score as the trigger.
    #   (d) reversion_event_v1 -- the score demoted to a ranker over events, with
    #       the two constant and three wrong-sign components removed.
    #
    # The band depth is set structurally at 0.35 daily ATR.  A sweep peaks far
    # higher at 0.15, but that peak is non-monotonic on a ~50-trade sample and
    # selecting it would repeat the fitted-step-table error this work removed.
    variants["baseline"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_entry_score_family="reversion_event_v1",
        pb_v2_dislocation_band_atr=0.35,
    )

    # 6. Additional entry mechanism: below-open resting bid, additive to the
    #    reclaim event (covers sessions that never reclaim).
    variants["baseline_limit"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_entry_score_family="reversion_event_v1",
        pb_v2_dislocation_band_atr=0.35,
        pb_open_scored_transition="reclaim_or_limit",
        pb_open_scored_limit_anchor="daily_atr",
        pb_open_scored_limit_atr_frac=0.25,
        pb_open_scored_limit_arm_bar=3,
        pb_open_scored_retrace_limit_window_bars=24,
    )

    # Ablations of the baseline, for attribution.
    variants["abl_no_event"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_entry_score_family="reversion_event_v1",
    )
    variants["abl_no_score"] = _hold(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
    )
    variants["abl_no_oversold"] = _hold(
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_entry_score_family="reversion_event_v1",
    )
    variants["abl_no_hold"] = derive(
        pb_v2_open_scored_trigger_policy="oversold",
        pb_v2_open_scored_confirmation_policy="band_reclaim",
        pb_entry_score_family="reversion_event_v1",
        pb_v2_dislocation_band_atr=0.35,
    )

    return variants


def _metrics(result, equity: float) -> dict:
    from backtests.stock.auto.scoring import extract_metrics

    perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, equity)
    trades = result.trades
    rs = [float(t.r_multiple) for t in trades]
    total_r = float(np.sum(rs)) if rs else 0.0
    avg_r = float(np.mean(rs)) if rs else 0.0
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = float(np.sum(wins)) if wins else 0.0
    gross_loss = float(-np.sum(losses)) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    return {
        "n": len(trades),
        "total_r": total_r,
        "avg_r": avg_r,
        "pf": pf,
        "wr": (len(wins) / len(rs) * 100.0) if rs else 0.0,
        "net_profit": float(getattr(perf, "net_profit", 0.0)),
        "ret_pct": float(getattr(perf, "net_profit", 0.0)) / max(equity, 1.0) * 100.0,
        "max_dd": float(getattr(perf, "max_drawdown_pct", 0.0)) * 100.0,
        "sharpe": float(getattr(perf, "sharpe", 0.0)),
        "sortino": float(getattr(perf, "sortino", 0.0)),
        "tpm": float(getattr(perf, "trades_per_month", 0.0)),
    }


def run_variant(replay, muts: dict, start: str, end: str, equity: float) -> dict:
    from backtests.stock.auto.config_mutator import mutate_iaric_config
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

    config = IARICBacktestConfig(
        start_date=start,
        end_date=end,
        initial_equity=equity,
        tier=3,
        data_dir=DATA_DIR,
    )
    config = mutate_iaric_config(config, muts)
    result = IARICPullbackEngine(config, replay, collect_diagnostics=True).run()
    out = _metrics(result, equity)
    out["_result"] = result
    return out


def _fmt(name: str, m: dict) -> str:
    pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return (
        f"  {name:<16} n={m['n']:>4}  totR={m['total_r']:+8.2f}  avgR={m['avg_r']:+.3f}  "
        f"PF={pf:>5}  WR={m['wr']:4.1f}%  ret={m['ret_pct']:+7.2f}%  DD={m['max_dd']:5.2f}%  "
        f"Sh={m['sharpe']:+.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="control")
    ap.add_argument("--folds", action="store_true")
    ap.add_argument("--equity", type=float, default=INITIAL_EQUITY)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--sweep",
        default=None,
        help="param=v1|v2|v3 -- run the chosen variant once per value",
    )
    ap.add_argument(
        "--overrides",
        default=None,
        help="k=v,k=v applied on top of each variant (ints/floats/bools parsed)",
    )
    args = ap.parse_args()

    from backtests.stock.data.replay_cache import load_research_replay_bundle

    defs = variant_definitions()
    names = (
        list(defs.keys())
        if args.variants == "all"
        else [n.strip() for n in args.variants.split(",") if n.strip()]
    )
    unknown = [n for n in names if n not in defs]
    if unknown:
        print(f"unknown variants: {unknown}\navailable: {list(defs)}")
        return 2

    replay = load_research_replay_bundle(DATA_DIR, require_bundle=False).data

    print("=" * 108)
    print(f"  IARIC baseline runner   {START_DATE} -> {END_DATE}   equity=${args.equity:,.0f}")
    print("=" * 108)

    def _coerce(raw: str):
        text = raw.strip()
        low = text.lower()
        if low in {"true", "false"}:
            return low == "true"
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                pass
        return text

    extra: dict = {}
    if args.overrides:
        for pair in args.overrides.split(","):
            if "=" in pair:
                key, raw = pair.split("=", 1)
                extra[f"param_overrides.{key.strip()}"] = _coerce(raw)

    if args.sweep:
        key, _, values = args.sweep.partition("=")
        base_name = names[0]
        for raw in values.split("|"):
            muts = dict(defs[base_name])
            muts.update(extra)
            muts[f"param_overrides.{key.strip()}"] = _coerce(raw)
            m = run_variant(replay, muts, START_DATE, END_DATE, args.equity)
            m.pop("_result")
            print(_fmt(f"{key.strip()}={raw.strip()}", m))
        return 0

    summary: dict[str, dict] = {}
    for name in names:
        muts = dict(defs[name])
        muts.update(extra)
        m = run_variant(replay, muts, START_DATE, END_DATE, args.equity)
        res = m.pop("_result")
        summary[name] = m
        print(_fmt(name, m))
        funnel = getattr(res, "funnel_counters", None) or {}
        if funnel:
            keys = ("triggered", "candidate_pool", "watchlist", "entered", "open_scored_entry")
            parts = [f"{k}={funnel.get(k, 0)}" for k in keys if k in funnel]
            if parts:
                print(f"       funnel: {', '.join(parts)}")

    if args.folds:
        print("\n" + "-" * 108)
        print("  Chronological folds")
        print("-" * 108)
        for label, fs, fe in FOLDS:
            print(f"  [{label}]")
            for name in names:
                muts = dict(defs[name])
                muts.update(extra)
                m = run_variant(replay, muts, fs, fe, args.equity)
                m.pop("_result")
                summary.setdefault(f"{name}@{label}", m)
                print(_fmt(f"  {name}", m))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
