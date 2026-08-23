"""Stage-by-stage audit of the IARIC daily screen.

The engine's funnel counters collapse everything before the trigger into a single
``universe_seen -> triggered`` step, which hides which gate is actually destroying
the aperture.  This replays the same screen and counts every gate separately.
"""
from __future__ import annotations

import io
import sys
from collections import Counter

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtests.stock.auto.config_mutator import mutate_iaric_config
from backtests.stock.auto.iaric.run_baseline import (
    DATA_DIR,
    END_DATE,
    INITIAL_EQUITY,
    START_DATE,
    variant_definitions,
)
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.engine.iaric_pullback_engine import (
    IARICPullbackEngine,
    _evaluate_v2_triggers,
)


def main() -> int:
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    replay = load_research_replay_bundle(DATA_DIR, require_bundle=False).data
    config = IARICBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_equity=INITIAL_EQUITY,
        tier=3,
        data_dir=DATA_DIR,
    )
    config = mutate_iaric_config(config, variant_definitions()["baseline"])
    outer = IARICPullbackEngine(config, replay, collect_diagnostics=False)
    engine = outer._build_engine() if hasattr(outer, "_build_engine") else None
    if engine is None:
        from backtests.stock.engine.iaric_pullback_intraday_hybrid_engine import (
            IARICPullbackIntradayHybridEngine,
        )

        engine = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=False)

    settings = engine._settings
    import datetime as _dt
    _s = _dt.date.fromisoformat(str(config.start_date)[:10])
    _e = _dt.date.fromisoformat(str(config.end_date)[:10])
    dates = [d for d in replay._trading_dates if _s <= d <= _e]
    counts: Counter = Counter()
    trigger_hist: Counter = Counter()
    per_day_triggered: list[int] = []

    for trade_date in dates:
        prev_date = replay.get_prev_trading_date(trade_date)
        if prev_date is None:
            continue
        day_triggered = 0
        for sym, sym_sector_raw, _ in engine._trade_universe:
            counts["00_universe"] += 1
            ind = engine._indicators.get(sym)
            di = engine._date_iloc.get(sym)
            if ind is None or di is None:
                counts["01_no_indicators"] += 1
                continue
            iloc = di.get(prev_date, -1)
            if iloc < 0:
                counts["02_no_prev_bar"] += 1
                continue
            if sym_sector_raw == "benchmark":
                counts["03_benchmark"] += 1
                continue

            closes = replay._daily_arrs[sym]["close"]
            sma_trend_val = ind["sma_trend"][iloc]
            above50 = not np.isnan(sma_trend_val) and closes[iloc] > sma_trend_val
            slope_ok = bool(ind["sma_slope"][iloc])
            sma200 = ind.get("sma200")
            sma200_v = sma200[iloc] if sma200 is not None else np.nan
            above200 = not np.isnan(sma200_v) and closes[iloc] > sma200_v
            sma50_above_200 = (
                not np.isnan(sma200_v)
                and not np.isnan(sma_trend_val)
                and sma_trend_val > sma200_v
            )
            if above50 and slope_ok:
                trend_tier = "STRONG"
            elif settings.pb_v2_allow_secular and above200 and sma50_above_200:
                trend_tier = "SECULAR"
            else:
                counts["04_trend_filter"] += 1
                continue

            prev_close_val = closes[iloc]
            if prev_close_val <= 0:
                counts["05_bad_prev_close"] += 1
                continue
            ohlc = replay.get_daily_ohlc(sym, trade_date)
            if ohlc is None:
                counts["06_no_daily_ohlc"] += 1
                continue
            O = ohlc[0]
            gap_pct = (O - prev_close_val) / prev_close_val * 100
            sma_dist_pct = (
                (prev_close_val - sma_trend_val) / sma_trend_val * 100
                if sma_trend_val > 0 and not np.isnan(sma_trend_val)
                else 0.0
            )
            cdd_val = int(ind["cdd"][iloc])

            if gap_pct < settings.pb_v2_gap_min_pct or gap_pct > settings.pb_v2_gap_max_pct:
                counts["07_gap_range"] += 1
                continue
            if (
                sma_dist_pct < settings.pb_v2_sma_dist_min_pct
                or sma_dist_pct > settings.pb_v2_sma_dist_max_pct
            ):
                counts["08_sma_dist_range"] += 1
                continue
            if cdd_val > settings.pb_cdd_max:
                counts["09_cdd_max"] += 1
                continue

            rs_arr = ind.get("rs_ratio")
            rs_val = 1.0
            if rs_arr is not None and not np.isnan(rs_arr[iloc]):
                rs_val = float(rs_arr[iloc])
            trigs = _evaluate_v2_triggers(
                ind=ind,
                iloc=iloc,
                closes=closes,
                prev_close=prev_close_val,
                gap_pct=gap_pct,
                trend_tier=trend_tier,
                rs_val=rs_val,
                settings=settings,
            )
            if not trigs:
                counts["10_no_trigger"] += 1
                continue
            counts["11_TRIGGERED"] += 1
            day_triggered += 1
            for name, _tier in trigs:
                trigger_hist[name] += 1
        per_day_triggered.append(day_triggered)

    total = counts["00_universe"]
    print("=" * 76)
    print(f"  IARIC daily screen funnel   {START_DATE} -> {END_DATE}")
    print(f"  sessions={len(per_day_triggered)}  universe/day={total / max(len(per_day_triggered), 1):.1f}")
    print("=" * 76)
    running = total
    for key in sorted(counts):
        if key == "00_universe":
            print(f"  {key:<22}{counts[key]:>8}")
            continue
        pct = counts[key] / max(total, 1) * 100
        print(f"  {key:<22}{counts[key]:>8}  ({pct:5.2f}% of universe)")
    print(f"\n  triggered/day: mean={np.mean(per_day_triggered):.2f} "
          f"median={np.median(per_day_triggered):.0f} max={max(per_day_triggered)}")
    print(f"\n  {'Trigger':<20}{'fires':>8}")
    for name, n in trigger_hist.most_common():
        print(f"  {name:<20}{n:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
