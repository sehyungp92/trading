"""Deep-dive on the inverted 15m slope finding for NQDTC.

The synergy analysis found a 53R swing: NQDTC entries where 15m momentum
slope OPPOSES the breakout direction have +25.3R (62% WR), while slope
SUPPORTING has -27.7R (46% WR).

This script investigates:
  1. Does slope explain the Vdubus-day effect?
  2. Interaction with chop mode, session, entry subtype, 1H trend
  3. Slope magnitude (not just direction)
  4. Time stability (monthly)
  5. Mechanical timing: raw MACD hist values and slope at entry
  6. Whether the inverted slope signal could be used as a direct NQDTC filter

Usage:
    python -m scripts.slope_deepdive
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Shared constants ────────────────────────────────────────────────
MNQ_POINT_VALUE = 2.0
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
MNQ_COMMISSION = 0.62
FIXED_QTY = 10
INITIAL_EQUITY = 100_000.0

DATA_DIR = ROOT / "backtest" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtest" / "output"


# ====================================================================
# Enriched trade with slope detail
# ====================================================================

@dataclass
class SlopeEnrichedTrade:
    """NQDTC trade with detailed 15m slope context."""
    # Core trade fields
    direction: int = 0
    entry_subtype: str = ""
    session: str = ""
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    won: bool = False
    exit_reason: str = ""
    # NQDTC context
    composite_regime: str = ""
    chop_mode: str = ""
    exit_tier: str = ""
    tp1_hit: bool = False
    tp2_hit: bool = False
    bars_held_30m: int = 0
    continuation: bool = False
    # 15m slope detail
    macd_hist_val: float = 0.0      # raw MACD histogram value at entry
    slope_3bar: float = 0.0          # 3-bar slope of MACD hist
    slope_prev: float = 0.0          # previous bar's slope
    slope_supports: bool = False     # does slope support trade direction?
    slope_magnitude: float = 0.0     # abs(slope)
    macd_accelerating: bool = False  # is slope increasing in trade direction?
    # 1H trend
    trend_1h: int = 0
    trend_1h_supports: bool = False
    # Vdubus-day
    vdubus_day: bool = False
    date_str: str = ""
    month_str: str = ""
    hour_et: int = 0


# ====================================================================
# Timezone helpers
# ====================================================================

_ET = None

def _get_et():
    global _ET
    if _ET is None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    return _ET

def _to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        from zoneinfo import ZoneInfo
        return dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(_get_et())
    return dt.astimezone(_get_et())

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ====================================================================
# Engine runners
# ====================================================================

def run_nqdtc():
    from backtest.cli import _load_nqdtc_data
    from backtest.config import SlippageConfig
    from backtest.config_nqdtc import NQDTCBacktestConfig
    from backtest.engine.nqdtc_engine import NQDTCEngine

    logger.info("Running NQDTC v2.0...")
    data = _load_nqdtc_data("NQ", DATA_DIR)
    config = NQDTCBacktestConfig(
        symbols=["MNQ"], initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR,
        slippage=SlippageConfig(commission_per_contract=MNQ_COMMISSION),
        fixed_qty=FIXED_QTY, tick_size=MNQ_TICK_SIZE, point_value=MNQ_POINT_VALUE,
        track_signals=False, track_shadows=False,
    )
    engine = NQDTCEngine("MNQ", config)
    result = engine.run(
        data["five_min_bars"], data["thirty_min"], data["hourly"],
        data["four_hour"], data["daily"],
        data["thirty_min_idx_map"], data["hourly_idx_map"],
        data["four_hour_idx_map"], data["daily_idx_map"],
    )
    logger.info("NQDTC: %d trades", len(result.trades))
    return result.trades


def run_vdubus_get_dates() -> set[str]:
    from backtest.cli import _load_vdubus_data
    from backtest.config import SlippageConfig
    from backtest.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtest.engine.vdubus_engine import VdubusEngine
    from strategy_3 import config as C

    logger.info("Running VdubusNQ v4.0...")
    data = _load_vdubus_data("NQ", DATA_DIR)

    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = MNQ_TICK_VALUE
    C.NQ_SPEC["point_value"] = MNQ_POINT_VALUE
    C.RT_COMM_FEES = MNQ_COMMISSION * 2

    try:
        config = VdubusBacktestConfig(
            symbols=["NQ"], initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR,
            slippage=SlippageConfig(commission_per_contract=MNQ_COMMISSION),
            fixed_qty=FIXED_QTY, tick_size=MNQ_TICK_SIZE, point_value=MNQ_POINT_VALUE,
            track_signals=False, track_shadows=False,
            flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
        )
        engine = VdubusEngine("NQ", config)
        result = engine.run(
            data["bars_15m"], data.get("bars_5m"), data["hourly"], data["daily_es"],
            data["hourly_idx_map"], data["daily_es_idx_map"],
            data.get("five_to_15_idx_map"),
        )
    finally:
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    dates = set()
    for t in result.trades:
        if t.entry_time is not None:
            dates.add(_to_et(_ensure_utc(t.entry_time)).strftime("%Y-%m-%d"))
    logger.info("Vdubus: %d trades on %d unique days", len(result.trades), len(dates))
    return dates


# ====================================================================
# 15m slope indicator computation
# ====================================================================

def _build_slope_lookup():
    """Load NQ 15m + NQ 1H data and pre-compute slope indicators.
    Returns lookup(entry_time) -> dict with detailed slope metrics.
    """
    from backtest.cli import _load_vdubus_data
    from strategy_3.indicators import ema, macd_hist
    from strategy_3 import config as C

    data = _load_vdubus_data("NQ", DATA_DIR)

    # NQ 15m
    nq_15m = data["bars_15m"]
    nq_15m_closes = nq_15m.closes
    nq_15m_ts = nq_15m.times
    nq_15m_mom = macd_hist(nq_15m_closes)

    # NQ 1H
    nq_1h = data["hourly"]
    nq_1h_closes = nq_1h.closes
    nq_1h_ts = nq_1h.times
    nq_1h_ema50 = ema(nq_1h_closes, C.HOURLY_EMA_PERIOD)

    def _find_last_idx_before(timestamps, target_dt64):
        n = len(timestamps)
        if n == 0:
            return -1
        lo, hi = 0, n - 1
        result = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if timestamps[mid] <= target_dt64:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def lookup(entry_time: datetime) -> dict:
        dt_utc = _ensure_utc(entry_time)
        target_dt64 = np.datetime64(int(dt_utc.timestamp() * 1e9), "ns")

        result = {
            "macd_hist_val": 0.0,
            "slope_3bar": 0.0,
            "slope_prev": 0.0,
            "slope_long_ok": False,
            "slope_short_ok": False,
            "trend_1h": 0,
        }

        # --- NQ 15m momentum ---
        mi = _find_last_idx_before(nq_15m_ts, target_dt64)
        if mi >= 0 and mi >= C.MOM_N + C.SLOPE_LB + 1:
            mom = nq_15m_mom[:mi + 1]
            if len(mom) >= C.MOM_N + C.SLOPE_LB + 1 and not np.isnan(mom[-1]):
                result["macd_hist_val"] = float(mom[-1])
                slope = float(mom[-1] - mom[-1 - C.SLOPE_LB])
                slope_prev = float(mom[-2] - mom[-2 - C.SLOPE_LB])
                result["slope_3bar"] = slope
                result["slope_prev"] = slope_prev

                lookback = mom[-1 - C.MOM_N:-1]
                valid = lookback[~np.isnan(lookback)]
                if len(valid) > 0:
                    mn, mx = float(np.min(valid)), float(np.max(valid))
                    rng = mx - mn
                    if rng > 0:
                        floor = mn + C.FLOOR_PCT * rng
                        ceiling = mx - C.FLOOR_PCT * rng
                        result["slope_long_ok"] = (
                            slope > 0 or (slope > slope_prev and float(mom[-1]) > floor))
                        result["slope_short_ok"] = (
                            slope < 0 or (slope < slope_prev and float(mom[-1]) < ceiling))

        # --- NQ 1H trend ---
        h1_idx = _find_last_idx_before(nq_1h_ts, target_dt64)
        if h1_idx >= 0 and h1_idx >= C.HOURLY_EMA_PERIOD - 1 and not np.isnan(nq_1h_ema50[h1_idx]):
            result["trend_1h"] = 1 if nq_1h_closes[h1_idx] > nq_1h_ema50[h1_idx] else -1

        return result

    return lookup


# ====================================================================
# Enrich trades
# ====================================================================

def enrich_trades(raw_trades, vdubus_dates, slope_lookup) -> list[SlopeEnrichedTrade]:
    enriched = []
    for t in raw_trades:
        if t.entry_time is None:
            continue
        dt_et = _to_et(_ensure_utc(t.entry_time))
        date_str = dt_et.strftime("%Y-%m-%d")
        month_str = dt_et.strftime("%Y-%m")
        hour_et = dt_et.hour

        sl = slope_lookup(t.entry_time)

        slope_supports = (
            (t.direction > 0 and sl["slope_long_ok"]) or
            (t.direction < 0 and sl["slope_short_ok"])
        )
        # Is MACD hist accelerating in trade direction?
        macd_accel = (
            (t.direction > 0 and sl["slope_3bar"] > 0 and sl["slope_3bar"] > sl["slope_prev"]) or
            (t.direction < 0 and sl["slope_3bar"] < 0 and sl["slope_3bar"] < sl["slope_prev"])
        )
        trend_1h_supports = (sl["trend_1h"] == t.direction)

        enriched.append(SlopeEnrichedTrade(
            direction=t.direction,
            entry_subtype=t.entry_subtype or "",
            session=t.session or "",
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            pnl_dollars=t.pnl_dollars,
            r_multiple=t.r_multiple,
            mfe_r=t.mfe_r,
            mae_r=t.mae_r,
            won=t.r_multiple > 0,
            exit_reason=t.exit_reason or "",
            composite_regime=t.composite_regime or "",
            chop_mode=t.chop_mode or "",
            exit_tier=t.exit_tier or "",
            tp1_hit=t.tp1_hit,
            tp2_hit=t.tp2_hit,
            bars_held_30m=t.bars_held_30m,
            continuation=t.continuation,
            macd_hist_val=sl["macd_hist_val"],
            slope_3bar=sl["slope_3bar"],
            slope_prev=sl["slope_prev"],
            slope_supports=slope_supports,
            slope_magnitude=abs(sl["slope_3bar"]),
            macd_accelerating=macd_accel,
            trend_1h=sl["trend_1h"],
            trend_1h_supports=trend_1h_supports,
            vdubus_day=date_str in vdubus_dates,
            date_str=date_str,
            month_str=month_str,
            hour_et=hour_et,
        ))
    return enriched


# ====================================================================
# Stats helpers
# ====================================================================

def _stats(trades):
    if not trades:
        return {"n": 0, "wr": 0, "avg_r": 0, "med_r": 0, "total_r": 0,
                "avg_mfe": 0, "avg_mae": 0}
    wins = sum(1 for t in trades if t.won)
    rs = [t.r_multiple for t in trades]
    return {
        "n": len(trades),
        "wr": wins / len(trades),
        "avg_r": float(np.mean(rs)),
        "med_r": float(np.median(rs)),
        "total_r": float(np.sum(rs)),
        "avg_mfe": float(np.mean([t.mfe_r for t in trades])),
        "avg_mae": float(np.mean([t.mae_r for t in trades])),
    }


def _fmt(s, label):
    if s["n"] == 0:
        return f"  {label:50s}  n=  0"
    return (
        f"  {label:50s}  n={s['n']:>3d}   "
        f"WR={s['wr']:.0%}   avgR={s['avg_r']:+.2f}   medR={s['med_r']:+.2f}   "
        f"totalR={s['total_r']:+.1f}   MFE={s['avg_mfe']:.2f}   MAE={s['avg_mae']:.2f}"
    )


def _pctiles(values, label):
    if not values:
        return f"  {label}: (no data)"
    a = np.array(values)
    ps = [10, 25, 50, 75, 90]
    pvals = np.percentile(a, ps)
    parts = "  ".join(f"P{p}={v:+.2f}" for p, v in zip(ps, pvals))
    return f"  {label}:  {parts}"


def _dir_str(d):
    return "LONG" if d > 0 else "SHORT"


# ====================================================================
# Build report
# ====================================================================

def build_report(trades: list[SlopeEnrichedTrade]) -> str:
    out: list[str] = []

    sup = [t for t in trades if t.slope_supports]
    opp = [t for t in trades if not t.slope_supports]

    out.append("=" * 90)
    out.append("  15M MOMENTUM SLOPE DEEP-DIVE -- NQDTC BREAKOUT PERFORMANCE")
    out.append("=" * 90)

    out.append(f"\nTotal NQDTC trades: {len(trades)}")
    out.append(f"  Slope SUPPORTS breakout direction: {len(sup)}")
    out.append(f"  Slope OPPOSES breakout direction:  {len(opp)}")

    out.append("\n=== Baseline ===\n")
    out.append(_fmt(_stats(trades), "All trades"))
    out.append(_fmt(_stats(sup), "Slope SUPPORTS (continuation breakout)"))
    out.append(_fmt(_stats(opp), "Slope OPPOSES (reversal breakout)"))

    # ================================================================
    # 1. Does slope explain the Vdubus-day effect?
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  1. SLOPE vs VDUBUS-DAY INTERACTION")
    out.append(f"{'=' * 90}\n")
    out.append("  If the slope fully explains the Vdubus-day effect, then within each")
    out.append("  slope group, Vdubus-day vs non-Vdubus should show no difference.\n")

    for label, group in [("Slope SUPPORTS", sup), ("Slope OPPOSES", opp)]:
        vd = [t for t in group if t.vdubus_day]
        nv = [t for t in group if not t.vdubus_day]
        out.append(f"  --- {label} ---")
        out.append(_fmt(_stats(vd), f"  Vdubus-day"))
        out.append(_fmt(_stats(nv), f"  Non-Vdubus-day"))
        out.append("")

    # Also: slope distribution on Vdubus vs non-Vdubus days
    vd_all = [t for t in trades if t.vdubus_day]
    nv_all = [t for t in trades if not t.vdubus_day]
    vd_opp_pct = sum(1 for t in vd_all if not t.slope_supports) / max(len(vd_all), 1)
    nv_opp_pct = sum(1 for t in nv_all if not t.slope_supports) / max(len(nv_all), 1)
    out.append(f"  Slope-opposing rate: Vdubus-days={vd_opp_pct:.0%} ({sum(1 for t in vd_all if not t.slope_supports)}/{len(vd_all)})   "
               f"Non-Vdubus={nv_opp_pct:.0%} ({sum(1 for t in nv_all if not t.slope_supports)}/{len(nv_all)})")

    # ================================================================
    # 2. Slope x chop mode
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  2. SLOPE x CHOP MODE")
    out.append(f"{'=' * 90}\n")

    for chop in sorted(set(t.chop_mode for t in trades)):
        ch_sup = [t for t in sup if t.chop_mode == chop]
        ch_opp = [t for t in opp if t.chop_mode == chop]
        out.append(f"  --- Chop = {chop} ---")
        out.append(_fmt(_stats(ch_sup), f"  Slope SUPPORTS"))
        out.append(_fmt(_stats(ch_opp), f"  Slope OPPOSES"))
        out.append("")

    # ================================================================
    # 3. Slope x session (ETH vs RTH)
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  3. SLOPE x SESSION")
    out.append(f"{'=' * 90}\n")

    for sess in sorted(set(t.session for t in trades)):
        s_sup = [t for t in sup if t.session == sess]
        s_opp = [t for t in opp if t.session == sess]
        out.append(f"  --- Session = {sess} ---")
        out.append(_fmt(_stats(s_sup), f"  Slope SUPPORTS"))
        out.append(_fmt(_stats(s_opp), f"  Slope OPPOSES"))
        out.append("")

    # ================================================================
    # 4. Slope x entry subtype
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  4. SLOPE x ENTRY SUBTYPE")
    out.append(f"{'=' * 90}\n")

    for sub in sorted(set(t.entry_subtype for t in trades)):
        sub_sup = [t for t in sup if t.entry_subtype == sub]
        sub_opp = [t for t in opp if t.entry_subtype == sub]
        if not sub_sup and not sub_opp:
            continue
        out.append(f"  --- {sub} ---")
        out.append(_fmt(_stats(sub_sup), f"  Slope SUPPORTS"))
        out.append(_fmt(_stats(sub_opp), f"  Slope OPPOSES"))
        out.append("")

    # ================================================================
    # 5. Slope x 1H trend
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  5. SLOPE x 1H TREND")
    out.append(f"{'=' * 90}\n")
    out.append("  Are slope and 1H trend correlated? (both inverted)\n")

    # Cross-tab
    for slope_label, slope_group in [("Slope SUPPORTS", sup), ("Slope OPPOSES", opp)]:
        h1_sup = [t for t in slope_group if t.trend_1h_supports]
        h1_opp = [t for t in slope_group if not t.trend_1h_supports]
        out.append(f"  --- {slope_label} ---")
        out.append(_fmt(_stats(h1_sup), f"  1H trend SUPPORTS"))
        out.append(_fmt(_stats(h1_opp), f"  1H trend OPPOSES"))
        out.append("")

    # Correlation check
    both_support = [t for t in trades if t.slope_supports and t.trend_1h_supports]
    both_oppose = [t for t in trades if not t.slope_supports and not t.trend_1h_supports]
    slope_sup_1h_opp = [t for t in trades if t.slope_supports and not t.trend_1h_supports]
    slope_opp_1h_sup = [t for t in trades if not t.slope_supports and t.trend_1h_supports]

    out.append("  --- 2x2 cross-tab ---")
    out.append(_fmt(_stats(both_support), "Both slope+1H SUPPORT (continuation)"))
    out.append(_fmt(_stats(both_oppose), "Both slope+1H OPPOSE (reversal)"))
    out.append(_fmt(_stats(slope_sup_1h_opp), "Slope supports, 1H opposes"))
    out.append(_fmt(_stats(slope_opp_1h_sup), "Slope opposes, 1H supports"))

    # ================================================================
    # 6. Slope magnitude (continuous, not just direction)
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  6. SLOPE MAGNITUDE ANALYSIS")
    out.append(f"{'=' * 90}\n")

    # Raw slope values
    out.append(_pctiles([t.slope_3bar for t in sup], "Slope value (supports)"))
    out.append(_pctiles([t.slope_3bar for t in opp], "Slope value (opposes)"))
    out.append("")
    out.append(_pctiles([t.macd_hist_val for t in sup], "MACD hist value (supports)"))
    out.append(_pctiles([t.macd_hist_val for t in opp], "MACD hist value (opposes)"))

    # Directional slope: positive = slope in trade direction, negative = against
    dir_slopes = []
    for t in trades:
        ds = t.slope_3bar * t.direction  # positive = with trade, negative = against
        dir_slopes.append((ds, t))

    # Quintile analysis
    out.append(f"\n  --- Quintile by directional slope (slope * direction) ---")
    out.append(f"  Positive = momentum with trade direction, Negative = against\n")
    dir_slopes.sort(key=lambda x: x[0])
    n = len(dir_slopes)
    q_size = n // 5
    for qi in range(5):
        start = qi * q_size
        end = (qi + 1) * q_size if qi < 4 else n
        q_trades = [ds[1] for ds in dir_slopes[start:end]]
        q_vals = [ds[0] for ds in dir_slopes[start:end]]
        q_label = f"Q{qi+1} (slope={min(q_vals):+.1f} to {max(q_vals):+.1f})"
        out.append(_fmt(_stats(q_trades), q_label))

    # Also check: MACD hist sign relative to direction
    out.append(f"\n  --- MACD histogram sign relative to trade direction ---\n")
    macd_with = [t for t in trades if t.macd_hist_val * t.direction > 0]
    macd_against = [t for t in trades if t.macd_hist_val * t.direction < 0]
    macd_zero = [t for t in trades if t.macd_hist_val * t.direction == 0]
    out.append(_fmt(_stats(macd_with), "MACD hist WITH trade direction"))
    out.append(_fmt(_stats(macd_against), "MACD hist AGAINST trade direction"))
    if macd_zero:
        out.append(_fmt(_stats(macd_zero), "MACD hist = 0"))

    # ================================================================
    # 7. Time stability (monthly)
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  7. TIME STABILITY (MONTHLY)")
    out.append(f"{'=' * 90}\n")

    months = sorted(set(t.month_str for t in trades))
    out.append(f"  {'Month':<10s}  {'Supports':^40s}  {'Opposes':^40s}")
    out.append(f"  {'-'*90}")
    for m in months:
        m_sup = [t for t in sup if t.month_str == m]
        m_opp = [t for t in opp if t.month_str == m]
        s_s = _stats(m_sup)
        s_o = _stats(m_opp)
        sup_str = f"n={s_s['n']:>2d} WR={s_s['wr']:.0%} totR={s_s['total_r']:+.1f}" if s_s['n'] else "n= 0"
        opp_str = f"n={s_o['n']:>2d} WR={s_o['wr']:.0%} totR={s_o['total_r']:+.1f}" if s_o['n'] else "n= 0"
        out.append(f"  {m:<10s}  {sup_str:^40s}  {opp_str:^40s}")

    # Consistency check: in how many months does opposing outperform?
    opp_wins_months = 0
    total_months = 0
    for m in months:
        m_sup = [t for t in sup if t.month_str == m]
        m_opp = [t for t in opp if t.month_str == m]
        if m_sup or m_opp:
            total_months += 1
            s_s = _stats(m_sup)
            s_o = _stats(m_opp)
            if s_o["total_r"] > s_s["total_r"]:
                opp_wins_months += 1
    out.append(f"\n  Months where opposing > supporting: {opp_wins_months}/{total_months}")

    # ================================================================
    # 8. Hour of day
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  8. SLOPE x HOUR OF DAY (ET)")
    out.append(f"{'=' * 90}\n")

    hours = sorted(set(t.hour_et for t in trades))
    out.append(f"  {'Hour':<8s}  {'Supports':^40s}  {'Opposes':^40s}")
    out.append(f"  {'-'*90}")
    for h in hours:
        h_sup = [t for t in sup if t.hour_et == h]
        h_opp = [t for t in opp if t.hour_et == h]
        s_s = _stats(h_sup)
        s_o = _stats(h_opp)
        sup_str = f"n={s_s['n']:>2d} WR={s_s['wr']:.0%} totR={s_s['total_r']:+.1f}" if s_s['n'] else "n= 0"
        opp_str = f"n={s_o['n']:>2d} WR={s_o['wr']:.0%} totR={s_o['total_r']:+.1f}" if s_o['n'] else "n= 0"
        out.append(f"  {h:>2d}:00     {sup_str:^40s}  {opp_str:^40s}")

    # ================================================================
    # 9. Mechanical explanation
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  9. MECHANICAL EXPLANATION -- REVERSAL vs CONTINUATION BREAKOUTS")
    out.append(f"{'=' * 90}\n")

    out.append("  NQDTC fires breakouts after 30m box compression.")
    out.append("  The 15m MACD histogram slope at entry tells us:")
    out.append("    - Slope SUPPORTS = momentum was already going the breakout way")
    out.append("      -> Continuation breakout (late entry into existing move)")
    out.append("    - Slope OPPOSES = momentum was going the other way")
    out.append("      -> Reversal breakout (catching a structural shift)\n")

    # MAE analysis: are continuation breakouts getting whipsawed more?
    out.append("  --- Adverse Excursion Profile ---\n")
    out.append(_pctiles([t.mae_r for t in sup], "MAE (continuation/supports)"))
    out.append(_pctiles([t.mae_r for t in opp], "MAE (reversal/opposes)"))
    out.append("")
    out.append(_pctiles([t.mfe_r for t in sup], "MFE (continuation/supports)"))
    out.append(_pctiles([t.mfe_r for t in opp], "MFE (reversal/opposes)"))

    # How quickly do they reach MFE?
    out.append(f"\n  --- Hold Duration ---\n")
    out.append(_pctiles([t.bars_held_30m for t in sup if t.won], "Duration WINNERS (supports)"))
    out.append(_pctiles([t.bars_held_30m for t in opp if t.won], "Duration WINNERS (opposes)"))
    out.append(_pctiles([t.bars_held_30m for t in sup if not t.won], "Duration LOSERS (supports)"))
    out.append(_pctiles([t.bars_held_30m for t in opp if not t.won], "Duration LOSERS (opposes)"))

    # Exit reasons
    out.append(f"\n  --- Exit Reasons ---\n")
    for reason in sorted(set(t.exit_reason for t in trades)):
        cnt_s = sum(1 for t in sup if t.exit_reason == reason)
        cnt_o = sum(1 for t in opp if t.exit_reason == reason)
        avgr_s = np.mean([t.r_multiple for t in sup if t.exit_reason == reason]) if cnt_s else 0
        avgr_o = np.mean([t.r_multiple for t in opp if t.exit_reason == reason]) if cnt_o else 0
        out.append(f"  {reason:<20s}  Supports: {cnt_s:>3d} avgR={avgr_s:+.2f}   "
                   f"Opposes: {cnt_o:>3d} avgR={avgr_o:+.2f}")

    # ================================================================
    # 10. Filter simulation: PREFER reversal breakouts
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  10. FILTER SIMULATIONS -- LEVERAGING THE INVERTED SLOPE")
    out.append(f"{'=' * 90}\n")

    baseline_r = _stats(trades)["total_r"]

    # F1: Block continuation breakouts entirely (keep only reversals)
    out.append("  --- F1: Block slope-supporting (keep only reversal breakouts) ---")
    out.append(_fmt(_stats(opp), "Kept (reversal only)"))
    out.append(_fmt(_stats(sup), "Removed (continuation)"))
    out.append(f"  Delta: {_stats(opp)['total_r'] - baseline_r:+.1f}R\n")

    # F2: Size down continuation, full size reversal
    adj_f2 = []
    for t in trades:
        mult = 0.50 if t.slope_supports else 1.0
        adj_f2.append(t.r_multiple * mult)
    out.append("  --- F2: Size x0.50 for continuation, x1.0 for reversal ---")
    out.append(f"  Adjusted total R: {sum(adj_f2):+.1f}  (baseline {baseline_r:+.1f})")
    out.append(f"  Delta: {sum(adj_f2) - baseline_r:+.1f}R\n")

    # F3: Size up reversal, normal continuation
    adj_f3 = []
    for t in trades:
        mult = 1.25 if not t.slope_supports else 1.0
        adj_f3.append(t.r_multiple * mult)
    out.append("  --- F3: Size x1.25 for reversal, x1.0 for continuation ---")
    out.append(f"  Adjusted total R: {sum(adj_f3):+.1f}  (baseline {baseline_r:+.1f})")
    out.append(f"  Delta: {sum(adj_f3) - baseline_r:+.1f}R\n")

    # F4: MACD hist sign as filter (not slope direction)
    out.append("  --- F4: Block when MACD hist WITH trade direction (hist*dir > 0) ---")
    kept_f4 = [t for t in trades if t.macd_hist_val * t.direction <= 0]
    removed_f4 = [t for t in trades if t.macd_hist_val * t.direction > 0]
    out.append(_fmt(_stats(kept_f4), "Kept (MACD against or zero)"))
    out.append(_fmt(_stats(removed_f4), "Removed (MACD with trade)"))
    out.append(f"  Delta: {_stats(kept_f4)['total_r'] - baseline_r:+.1f}R\n")

    # F5: Combined slope-oppose + 1H-oppose (double reversal confirmation)
    out.append("  --- F5: Block unless BOTH slope AND 1H oppose (pure reversal) ---")
    pure_rev = [t for t in trades if not t.slope_supports and not t.trend_1h_supports]
    rest = [t for t in trades if t.slope_supports or t.trend_1h_supports]
    out.append(_fmt(_stats(pure_rev), "Kept (double reversal)"))
    out.append(_fmt(_stats(rest), "Removed"))
    out.append(f"  Delta: {_stats(pure_rev)['total_r'] - baseline_r:+.1f}R\n")

    # F6: Quintile-based sizing (scale by directional slope, inverted)
    out.append("  --- F6: Quintile-based sizing (inverted: negative slope = more size) ---")
    dir_slopes_sorted = sorted(
        [(t.slope_3bar * t.direction, t) for t in trades],
        key=lambda x: x[0])
    n = len(dir_slopes_sorted)
    q_size = n // 5
    q_mults = {0: 1.25, 1: 1.10, 2: 1.0, 3: 0.75, 4: 0.50}
    adj_f6 = []
    for qi in range(5):
        start = qi * q_size
        end = (qi + 1) * q_size if qi < 4 else n
        for _, t in dir_slopes_sorted[start:end]:
            adj_f6.append(t.r_multiple * q_mults[qi])
    out.append(f"  Quintile mults: Q1(most opposed)=1.25, Q2=1.10, Q3=1.0, Q4=0.75, Q5(most with)=0.50")
    out.append(f"  Adjusted total R: {sum(adj_f6):+.1f}  (baseline {baseline_r:+.1f})")
    out.append(f"  Delta: {sum(adj_f6) - baseline_r:+.1f}R\n")

    # F7: Slope-oppose + Vdubus-day combination
    out.append("  --- F7: Slope-oppose trades on Vdubus-days only ---")
    best_combo = [t for t in trades if not t.slope_supports and t.vdubus_day]
    out.append(_fmt(_stats(best_combo), "Reversal + Vdubus-day"))
    out.append(f"  (This is {len(best_combo)} of {len(trades)} total trades)")
    out.append(f"  Delta if keeping only these: {_stats(best_combo)['total_r'] - baseline_r:+.1f}R\n")

    # ================================================================
    # 11. Individual trade list: top/bottom by directional slope
    # ================================================================
    out.append(f"\n{'=' * 90}")
    out.append("  11. INDIVIDUAL TRADES -- EXTREMES")
    out.append(f"{'=' * 90}\n")

    sorted_by_r = sorted(trades, key=lambda t: t.r_multiple)

    # Top 10 winners
    out.append("  --- Top 10 WINNERS ---")
    out.append(f"  {'Date':12s} {'Hr':>4s} {'Dir':>5s} {'Sub':>16s} {'R':>7s} {'MFE':>6s} {'MAE':>6s} {'Slope':>8s} {'Slope?':>8s} {'1H?':>5s} {'VD?':>4s}")
    for t in sorted_by_r[-10:][::-1]:
        dt_str = _to_et(_ensure_utc(t.entry_time)).strftime("%Y-%m-%d")
        out.append(f"  {dt_str:12s} {t.hour_et:>4d} {_dir_str(t.direction):>5s} {t.entry_subtype:>16s} "
                   f"{t.r_multiple:>+7.2f} {t.mfe_r:>6.2f} {t.mae_r:>6.2f} "
                   f"{t.slope_3bar:>+8.2f} {'SUP' if t.slope_supports else 'OPP':>8s} "
                   f"{'Y' if t.trend_1h_supports else 'N':>5s} {'Y' if t.vdubus_day else 'N':>4s}")

    out.append("")
    out.append("  --- Bottom 10 LOSERS ---")
    out.append(f"  {'Date':12s} {'Hr':>4s} {'Dir':>5s} {'Sub':>16s} {'R':>7s} {'MFE':>6s} {'MAE':>6s} {'Slope':>8s} {'Slope?':>8s} {'1H?':>5s} {'VD?':>4s}")
    for t in sorted_by_r[:10]:
        dt_str = _to_et(_ensure_utc(t.entry_time)).strftime("%Y-%m-%d")
        out.append(f"  {dt_str:12s} {t.hour_et:>4d} {_dir_str(t.direction):>5s} {t.entry_subtype:>16s} "
                   f"{t.r_multiple:>+7.2f} {t.mfe_r:>6.2f} {t.mae_r:>6.2f} "
                   f"{t.slope_3bar:>+8.2f} {'SUP' if t.slope_supports else 'OPP':>8s} "
                   f"{'Y' if t.trend_1h_supports else 'N':>5s} {'Y' if t.vdubus_day else 'N':>4s}")

    out.append(f"\n{'=' * 90}")

    return "\n".join(out)


# ====================================================================
# Main
# ====================================================================

def main():
    vdubus_dates = run_vdubus_get_dates()
    raw_trades = run_nqdtc()

    logger.info("Computing 15m slope indicators...")
    slope_lookup = _build_slope_lookup()

    trades = enrich_trades(raw_trades, vdubus_dates, slope_lookup)
    logger.info("Enriched %d trades", len(trades))

    report = build_report(trades)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "slope_deepdive.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
