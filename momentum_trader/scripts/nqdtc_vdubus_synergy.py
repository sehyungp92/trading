"""NQDTC-Vdubus synergy analysis -- what drives the 37R magnitude split?

Enriches each NQDTC trade with:
  (a) Full NQDTCTradeRecord context (regime, chop, score, displacement, etc.)
  (b) Vdubus regime indicators at NQDTC entry time (ES daily trend, vol state,
      NQ 1H trend, 15m momentum slope)

Then produces 7 analysis sections (A-G) to identify what drives the +17.6R
vs -19.9R split between Vdubus-days and non-Vdubus-days, and tests specific
filter candidates.

Usage:
    python -m scripts.nqdtc_vdubus_synergy
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
# Enriched NQDTC trade record
# ====================================================================

@dataclass
class NQDTCTradeEnriched:
    """NQDTC trade with full engine context + Vdubus regime indicators."""
    # Core trade fields
    direction: int = 0
    entry_subtype: str = ""
    session: str = ""
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_stop: float = 0.0
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    won: bool = False
    exit_reason: str = ""
    # NQDTC engine context
    composite_regime: str = ""
    chop_mode: str = ""
    score_at_entry: float = 0.0
    displacement_at_entry: float = 0.0
    rvol_at_entry: float = 0.0
    quality_mult: float = 0.0
    expiry_mult: float = 0.0
    exit_tier: str = ""
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    continuation: bool = False
    box_width: float = 0.0
    adaptive_L: int = 0
    bars_held_30m: int = 0
    # Vdubus regime indicators at entry time
    vb_daily_trend: int = 0        # ES SMA200 direction
    vb_vol_state: str = ""         # Normal/High/Shock
    vb_trend_1h: int = 0           # NQ EMA50 1H direction
    vb_slope_long_ok: bool = False  # 15m momentum slope supports long
    vb_slope_short_ok: bool = False # 15m momentum slope supports short
    # Classification
    vdubus_day: bool = False       # Whether Vdubus also fired that day
    date_str: str = ""


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
    """Run NQDTC engine, return raw NQDTCTradeRecords."""
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
    """Run Vdubus engine, return set of date strings with fills."""
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
# Vdubus regime indicator computation at NQDTC entry times
# ====================================================================

def _build_vdubus_regime_lookup():
    """Load Vdubus data and pre-compute regime indicators.

    Returns a function: lookup(entry_time) -> dict of indicators.
    """
    from backtest.cli import _load_vdubus_data
    from strategy_3.indicators import atr, sma, ema, macd_hist, percentile_rank, median_val
    from strategy_3 import config as C

    data = _load_vdubus_data("NQ", DATA_DIR)

    # -- ES daily arrays (NumpyBars: .closes, .highs, .lows, .times) --
    es_d = data["daily_es"]
    es_closes = es_d.closes
    es_highs = es_d.highs
    es_lows = es_d.lows
    es_ts = es_d.times  # numpy datetime64 array

    # Pre-compute ES daily indicators over full history
    es_sma200 = sma(es_closes, C.DAILY_SMA_PERIOD)
    es_atr14 = atr(es_highs, es_lows, es_closes, C.VOL_ATR_PERIOD)

    # -- NQ hourly arrays --
    nq_1h = data["hourly"]
    nq_1h_closes = nq_1h.closes
    nq_1h_ts = nq_1h.times

    nq_1h_ema50 = ema(nq_1h_closes, C.HOURLY_EMA_PERIOD)

    # -- NQ 15m arrays --
    nq_15m = data["bars_15m"]
    nq_15m_closes = nq_15m.closes
    nq_15m_ts = nq_15m.times

    nq_15m_mom = macd_hist(nq_15m_closes)

    def _find_last_idx_before(timestamps, target_dt64):
        """Binary search for last index with timestamp <= target.
        timestamps: numpy array of datetime64; target_dt64: numpy.datetime64
        """
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
        """Return Vdubus regime indicators at the given NQDTC entry time."""
        dt_utc = _ensure_utc(entry_time)
        # Convert to numpy datetime64 for comparison with NumpyBars.times
        target_dt64 = np.datetime64(int(dt_utc.timestamp() * 1e9), "ns")

        result = {
            "vb_daily_trend": 0,
            "vb_vol_state": "Unknown",
            "vb_trend_1h": 0,
            "vb_slope_long_ok": False,
            "vb_slope_short_ok": False,
        }

        # --- ES daily trend (SMA200) ---
        di = _find_last_idx_before(es_ts, target_dt64)
        if di >= 0 and di >= C.DAILY_SMA_PERIOD - 1 and not np.isnan(es_sma200[di]):
            result["vb_daily_trend"] = 1 if es_closes[di] > es_sma200[di] else -1

        # --- ES vol state ---
        if di >= 0 and di >= C.VOL_ATR_PERIOD - 1 and not np.isnan(es_atr14[di]):
            pctl = percentile_rank(float(es_atr14[di]), es_atr14[:di + 1], C.VOL_LOOKBACK)
            med = median_val(es_atr14[:di + 1], C.VOL_LOOKBACK)
            if pctl > C.SHOCK_PCTL and float(es_atr14[di]) > C.SHOCK_MED_MULT * med:
                result["vb_vol_state"] = "Shock"
            elif pctl > C.HIGH_PCTL:
                result["vb_vol_state"] = "High"
            else:
                result["vb_vol_state"] = "Normal"

        # --- NQ 1H trend (EMA50) ---
        h1_idx = _find_last_idx_before(nq_1h_ts, target_dt64)
        if h1_idx >= 0 and h1_idx >= C.HOURLY_EMA_PERIOD - 1 and not np.isnan(nq_1h_ema50[h1_idx]):
            result["vb_trend_1h"] = 1 if nq_1h_closes[h1_idx] > nq_1h_ema50[h1_idx] else -1

        # --- NQ 15m momentum slope ---
        mi = _find_last_idx_before(nq_15m_ts, target_dt64)
        if mi >= 0 and mi >= C.MOM_N + C.SLOPE_LB + 1:
            mom_slice = nq_15m_mom[:mi + 1]
            if len(mom_slice) >= C.MOM_N + C.SLOPE_LB + 1 and not np.isnan(mom_slice[-1]):
                slope = float(mom_slice[-1] - mom_slice[-1 - C.SLOPE_LB])
                slope_prev = float(mom_slice[-2] - mom_slice[-2 - C.SLOPE_LB])
                lookback = mom_slice[-1 - C.MOM_N:-1]
                valid = lookback[~np.isnan(lookback)]
                if len(valid) > 0:
                    mn, mx = float(np.min(valid)), float(np.max(valid))
                    rng = mx - mn
                    if rng > 0:
                        floor = mn + C.FLOOR_PCT * rng
                        ceiling = mx - C.FLOOR_PCT * rng
                        result["vb_slope_long_ok"] = (
                            slope > 0 or (slope > slope_prev and float(mom_slice[-1]) > floor))
                        result["vb_slope_short_ok"] = (
                            slope < 0 or (slope < slope_prev and float(mom_slice[-1]) < ceiling))

        return result

    return lookup


# ====================================================================
# Enrich NQDTC trades
# ====================================================================

def enrich_trades(raw_trades, vdubus_dates: set[str], vb_lookup) -> list[NQDTCTradeEnriched]:
    """Convert raw NQDTCTradeRecord list to enriched list."""
    enriched = []
    for t in raw_trades:
        if t.entry_time is None:
            continue
        date_str = _to_et(_ensure_utc(t.entry_time)).strftime("%Y-%m-%d")
        vb = vb_lookup(t.entry_time)

        enriched.append(NQDTCTradeEnriched(
            direction=t.direction,
            entry_subtype=t.entry_subtype or "",
            session=t.session or "",
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            initial_stop=t.initial_stop,
            pnl_dollars=t.pnl_dollars,
            r_multiple=t.r_multiple,
            mfe_r=t.mfe_r,
            mae_r=t.mae_r,
            won=t.r_multiple > 0,
            exit_reason=t.exit_reason or "",
            composite_regime=t.composite_regime or "",
            chop_mode=t.chop_mode or "",
            score_at_entry=t.score_at_entry,
            displacement_at_entry=t.displacement_at_entry,
            rvol_at_entry=t.rvol_at_entry,
            quality_mult=t.quality_mult,
            expiry_mult=t.expiry_mult,
            exit_tier=t.exit_tier or "",
            tp1_hit=t.tp1_hit,
            tp2_hit=t.tp2_hit,
            tp3_hit=t.tp3_hit,
            continuation=t.continuation,
            box_width=t.box_width,
            adaptive_L=t.adaptive_L,
            bars_held_30m=t.bars_held_30m,
            vb_daily_trend=vb["vb_daily_trend"],
            vb_vol_state=vb["vb_vol_state"],
            vb_trend_1h=vb["vb_trend_1h"],
            vb_slope_long_ok=vb["vb_slope_long_ok"],
            vb_slope_short_ok=vb["vb_slope_short_ok"],
            vdubus_day=date_str in vdubus_dates,
            date_str=date_str,
        ))
    return enriched


# ====================================================================
# Stats helpers
# ====================================================================

def _stats(trades: list[NQDTCTradeEnriched]) -> dict:
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


def _fmt(s: dict, label: str) -> str:
    if s["n"] == 0:
        return f"  {label:45s}  n=  0"
    return (
        f"  {label:45s}  n={s['n']:>3d}   "
        f"WR={s['wr']:.0%}   avgR={s['avg_r']:+.2f}   medR={s['med_r']:+.2f}   "
        f"totalR={s['total_r']:+.1f}   MFE={s['avg_mfe']:.2f}   MAE={s['avg_mae']:.2f}"
    )


def _pctiles(values: list[float], label: str) -> str:
    if not values:
        return f"  {label}: (no data)"
    a = np.array(values)
    ps = [10, 25, 50, 75, 90]
    pvals = np.percentile(a, ps)
    parts = "  ".join(f"P{p}={v:+.2f}" for p, v in zip(ps, pvals))
    return f"  {label}:  {parts}"


# ====================================================================
# Analysis sections
# ====================================================================

def build_report(trades: list[NQDTCTradeEnriched]) -> str:
    out: list[str] = []

    vd = [t for t in trades if t.vdubus_day]
    nv = [t for t in trades if not t.vdubus_day]

    out.append("=" * 90)
    out.append("  NQDTC-VDUBUS SYNERGY ANALYSIS")
    out.append("=" * 90)
    out.append(f"\nTotal NQDTC trades: {len(trades)}")
    out.append(f"  On Vdubus-days: {len(vd)}")
    out.append(f"  On non-Vdubus-days: {len(nv)}")

    out.append("\n=== Baseline ===\n")
    out.append(_fmt(_stats(trades), "All NQDTC trades"))
    out.append(_fmt(_stats(vd), "Vdubus-day trades"))
    out.append(_fmt(_stats(nv), "Non-Vdubus-day trades"))

    # ── Section A: R-multiple distribution ──────────────────────────
    out.append(f"\n{'=' * 90}")
    out.append("  SECTION A: R-Multiple Distribution")
    out.append(f"{'=' * 90}\n")

    out.append(_pctiles([t.r_multiple for t in vd], "Vdubus-day R"))
    out.append(_pctiles([t.r_multiple for t in nv], "Non-Vdubus-day R"))
    out.append("")
    out.append(_pctiles([t.mfe_r for t in vd], "Vdubus-day MFE"))
    out.append(_pctiles([t.mfe_r for t in nv], "Non-Vdubus-day MFE"))
    out.append("")
    out.append(_pctiles([t.mae_r for t in vd], "Vdubus-day MAE"))
    out.append(_pctiles([t.mae_r for t in nv], "Non-Vdubus-day MAE"))

    # Tail analysis
    big_win_vd = [t for t in vd if t.r_multiple >= 2.0]
    big_win_nv = [t for t in nv if t.r_multiple >= 2.0]
    big_loss_vd = [t for t in vd if t.r_multiple <= -1.0]
    big_loss_nv = [t for t in nv if t.r_multiple <= -1.0]
    out.append(f"\n  Big wins (>=2R):   Vdubus-day {len(big_win_vd)}/{len(vd)} ({len(big_win_vd)/max(len(vd),1):.0%})   "
               f"Non-Vdubus {len(big_win_nv)}/{len(nv)} ({len(big_win_nv)/max(len(nv),1):.0%})")
    out.append(f"  Big losses (<=-1R): Vdubus-day {len(big_loss_vd)}/{len(vd)} ({len(big_loss_vd)/max(len(vd),1):.0%})   "
               f"Non-Vdubus {len(big_loss_nv)}/{len(nv)} ({len(big_loss_nv)/max(len(nv),1):.0%})")

    # ── Section B: Exit reason cross-tab ───────────────────────────
    out.append(f"\n{'=' * 90}")
    out.append("  SECTION B: Exit Reason Cross-Tab")
    out.append(f"{'=' * 90}\n")

    all_reasons = sorted(set(t.exit_reason for t in trades))
    out.append(f"  {'Exit Reason':<30s}  Vdubus-day (n={len(vd)})   Non-Vdubus (n={len(nv)})")
    out.append(f"  {'-'*80}")
    for reason in all_reasons:
        cnt_vd = sum(1 for t in vd if t.exit_reason == reason)
        cnt_nv = sum(1 for t in nv if t.exit_reason == reason)
        pct_vd = cnt_vd / max(len(vd), 1)
        pct_nv = cnt_nv / max(len(nv), 1)
        avgr_vd = np.mean([t.r_multiple for t in vd if t.exit_reason == reason]) if cnt_vd else 0
        avgr_nv = np.mean([t.r_multiple for t in nv if t.exit_reason == reason]) if cnt_nv else 0
        out.append(f"  {reason:<30s}  {cnt_vd:>3d} ({pct_vd:>5.1%}) avgR={avgr_vd:+.2f}   "
                   f"{cnt_nv:>3d} ({pct_nv:>5.1%}) avgR={avgr_nv:+.2f}")

    # TP hit rates
    tp1_vd = sum(1 for t in vd if t.tp1_hit) / max(len(vd), 1)
    tp2_vd = sum(1 for t in vd if t.tp2_hit) / max(len(vd), 1)
    tp3_vd = sum(1 for t in vd if t.tp3_hit) / max(len(vd), 1)
    tp1_nv = sum(1 for t in nv if t.tp1_hit) / max(len(nv), 1)
    tp2_nv = sum(1 for t in nv if t.tp2_hit) / max(len(nv), 1)
    tp3_nv = sum(1 for t in nv if t.tp3_hit) / max(len(nv), 1)
    out.append(f"\n  TP hit rates:")
    out.append(f"    TP1: Vdubus-day={tp1_vd:.0%}  Non-Vdubus={tp1_nv:.0%}")
    out.append(f"    TP2: Vdubus-day={tp2_vd:.0%}  Non-Vdubus={tp2_nv:.0%}")
    out.append(f"    TP3: Vdubus-day={tp3_vd:.0%}  Non-Vdubus={tp3_nv:.0%}")

    # ── Section C: Entry subtype performance ───────────────────────
    out.append(f"\n{'=' * 90}")
    out.append("  SECTION C: Entry Subtype Performance")
    out.append(f"{'=' * 90}\n")

    subtypes = sorted(set(t.entry_subtype for t in trades))
    for sub in subtypes:
        sub_vd = [t for t in vd if t.entry_subtype == sub]
        sub_nv = [t for t in nv if t.entry_subtype == sub]
        if not sub_vd and not sub_nv:
            continue
        out.append(f"  --- {sub} ---")
        out.append(_fmt(_stats(sub_vd), f"  Vdubus-day"))
        out.append(_fmt(_stats(sub_nv), f"  Non-Vdubus-day"))
        out.append("")

    # ── Section D: NQDTC's own regime/quality fields ───────────────
    out.append(f"\n{'=' * 90}")
    out.append("  SECTION D: NQDTC's Own Regime/Quality as Predictors")
    out.append(f"{'=' * 90}")

    # Composite regime
    out.append("\n--- Composite Regime ---\n")
    regimes = sorted(set(t.composite_regime for t in trades))
    out.append(f"  {'Regime':<15s}  {'Vdubus-day':<35s}  {'Non-Vdubus':<35s}")
    out.append(f"  {'-'*90}")
    for reg in regimes:
        reg_vd = [t for t in vd if t.composite_regime == reg]
        reg_nv = [t for t in nv if t.composite_regime == reg]
        s_vd = _stats(reg_vd)
        s_nv = _stats(reg_nv)
        vd_str = f"n={s_vd['n']:>3d} WR={s_vd['wr']:.0%} avgR={s_vd['avg_r']:+.2f} tot={s_vd['total_r']:+.1f}" if s_vd['n'] else "n=  0"
        nv_str = f"n={s_nv['n']:>3d} WR={s_nv['wr']:.0%} avgR={s_nv['avg_r']:+.2f} tot={s_nv['total_r']:+.1f}" if s_nv['n'] else "n=  0"
        out.append(f"  {reg:<15s}  {vd_str:<35s}  {nv_str:<35s}")

    # Chop mode
    out.append("\n--- Chop Mode ---\n")
    chops = sorted(set(t.chop_mode for t in trades))
    for chop in chops:
        ch_vd = [t for t in vd if t.chop_mode == chop]
        ch_nv = [t for t in nv if t.chop_mode == chop]
        out.append(f"  {chop}:")
        out.append(_fmt(_stats(ch_vd), f"    Vdubus-day"))
        out.append(_fmt(_stats(ch_nv), f"    Non-Vdubus-day"))
        out.append("")

    # Exit tier
    out.append("--- Exit Tier ---\n")
    tiers = sorted(set(t.exit_tier for t in trades))
    for tier in tiers:
        ti_vd = [t for t in vd if t.exit_tier == tier]
        ti_nv = [t for t in nv if t.exit_tier == tier]
        out.append(f"  {tier}:")
        out.append(_fmt(_stats(ti_vd), f"    Vdubus-day"))
        out.append(_fmt(_stats(ti_nv), f"    Non-Vdubus-day"))
        out.append("")

    # Quality mult distribution
    out.append("--- Quality Mult Distribution ---\n")
    out.append(_pctiles([t.quality_mult for t in vd], "Vdubus-day quality_mult"))
    out.append(_pctiles([t.quality_mult for t in nv], "Non-Vdubus quality_mult"))
    out.append("")
    out.append(_pctiles([t.score_at_entry for t in vd], "Vdubus-day score"))
    out.append(_pctiles([t.score_at_entry for t in nv], "Non-Vdubus score"))
    out.append("")
    out.append(_pctiles([t.displacement_at_entry for t in vd], "Vdubus-day displacement"))
    out.append(_pctiles([t.displacement_at_entry for t in nv], "Non-Vdubus displacement"))

    # ── Section E: Vdubus regime indicators as predictors ──────────
    out.append(f"\n{'=' * 90}")
    out.append("  SECTION E: Vdubus Regime Indicators as NQDTC Predictors")
    out.append(f"{'=' * 90}")

    # E1: ES daily trend alignment
    out.append("\n--- E1: ES Daily Trend (SMA200) Alignment ---\n")
    dt_match = [t for t in trades if t.vb_daily_trend == t.direction]
    dt_oppose = [t for t in trades if t.vb_daily_trend == -t.direction]
    dt_flat = [t for t in trades if t.vb_daily_trend == 0]
    out.append(_fmt(_stats(dt_match), "Daily trend MATCHES direction"))
    out.append(_fmt(_stats(dt_oppose), "Daily trend OPPOSES direction"))
    out.append(_fmt(_stats(dt_flat), "Daily trend FLAT"))
    out.append("")
    # Also split by Vdubus-day
    dt_match_vd = [t for t in dt_match if t.vdubus_day]
    dt_match_nv = [t for t in dt_match if not t.vdubus_day]
    dt_oppose_vd = [t for t in dt_oppose if t.vdubus_day]
    dt_oppose_nv = [t for t in dt_oppose if not t.vdubus_day]
    out.append(f"  Cross-reference with Vdubus-day:")
    out.append(_fmt(_stats(dt_match_vd), "  Match + Vdubus-day"))
    out.append(_fmt(_stats(dt_match_nv), "  Match + Non-Vdubus"))
    out.append(_fmt(_stats(dt_oppose_vd), "  Oppose + Vdubus-day"))
    out.append(_fmt(_stats(dt_oppose_nv), "  Oppose + Non-Vdubus"))

    # E2: ES vol state
    out.append("\n--- E2: ES Vol State ---\n")
    for vs in ["Normal", "High", "Shock", "Unknown"]:
        vs_trades = [t for t in trades if t.vb_vol_state == vs]
        if vs_trades:
            out.append(_fmt(_stats(vs_trades), f"Vol State = {vs}"))

    # E3: NQ 1H trend alignment
    out.append("\n--- E3: NQ 1H Trend (EMA50) Alignment ---\n")
    h1_match = [t for t in trades if t.vb_trend_1h == t.direction]
    h1_oppose = [t for t in trades if t.vb_trend_1h == -t.direction]
    h1_flat = [t for t in trades if t.vb_trend_1h == 0]
    out.append(_fmt(_stats(h1_match), "1H trend MATCHES direction"))
    out.append(_fmt(_stats(h1_oppose), "1H trend OPPOSES direction"))
    out.append(_fmt(_stats(h1_flat), "1H trend FLAT"))

    # E4: 15m momentum slope
    out.append("\n--- E4: 15m Momentum Slope Alignment ---\n")
    slope_match = [t for t in trades if
                   (t.direction > 0 and t.vb_slope_long_ok) or
                   (t.direction < 0 and t.vb_slope_short_ok)]
    slope_oppose = [t for t in trades if
                    (t.direction > 0 and not t.vb_slope_long_ok) or
                    (t.direction < 0 and not t.vb_slope_short_ok)]
    out.append(_fmt(_stats(slope_match), "15m slope SUPPORTS direction"))
    out.append(_fmt(_stats(slope_oppose), "15m slope OPPOSES direction"))

    # E5: Combined score
    out.append("\n--- E5: Combined Vdubus Indicator Score ---\n")
    out.append("  Score = (daily_trend matches) + (vol=Normal) + (1H matches) + (slope supports)")
    out.append("")
    for score_val in range(5):
        score_trades = []
        for t in trades:
            s = 0
            if t.vb_daily_trend == t.direction:
                s += 1
            if t.vb_vol_state == "Normal":
                s += 1
            if t.vb_trend_1h == t.direction:
                s += 1
            if (t.direction > 0 and t.vb_slope_long_ok) or (t.direction < 0 and t.vb_slope_short_ok):
                s += 1
            if s == score_val:
                score_trades.append(t)
        if score_trades:
            out.append(_fmt(_stats(score_trades), f"Score = {score_val}/4"))

    # ── Section F: Filter candidate testing ────────────────────────
    out.append(f"\n{'=' * 90}")
    out.append("  SECTION F: Filter Candidate Testing")
    out.append(f"{'=' * 90}")
    out.append(f"\n  Baseline: {_fmt(_stats(trades), 'All trades')}")
    out.append("")

    def _test_filter(label: str, keep_fn, size_fn=None):
        """Test a filter. keep_fn returns True to keep trade.
        size_fn returns multiplier for R-impact (None = block entirely)."""
        kept = [t for t in trades if keep_fn(t)]
        removed = [t for t in trades if not keep_fn(t)]

        if size_fn is not None:
            # Sizing filter: adjust R-multiples
            adj_rs = []
            for t in trades:
                mult = size_fn(t)
                adj_rs.append(t.r_multiple * mult)
            adj_total_r = sum(adj_rs)
            adj_wins = sum(1 for r in adj_rs if r > 0)
            lines = [
                f"\n  --- {label} ---",
                f"  Sizing adjustment (not blocking)",
                f"  Adjusted total R: {adj_total_r:+.1f}  (baseline {_stats(trades)['total_r']:+.1f})",
                f"  Delta: {adj_total_r - _stats(trades)['total_r']:+.1f}R",
            ]
        else:
            s_kept = _stats(kept)
            s_removed = _stats(removed)
            lines = [
                f"\n  --- {label} ---",
                f"  Trades kept: {len(kept)}   Trades removed: {len(removed)}",
                _fmt(s_kept, "Kept"),
                _fmt(s_removed, "Removed"),
                f"  Delta total R: {s_kept['total_r'] - _stats(trades)['total_r']:+.1f}R "
                f"(removed {s_removed['total_r']:+.1f}R of bad trades)",
            ]
        return lines

    # F1: Block when ES daily trend opposes
    out.extend(_test_filter(
        "F1: Block when ES daily trend OPPOSES direction",
        lambda t: t.vb_daily_trend != -t.direction,
    ))

    # F2: Reduce size 35% when vol=HIGH
    out.extend(_test_filter(
        "F2: Size x0.65 when ES vol=HIGH",
        lambda t: True,
        size_fn=lambda t: 0.65 if t.vb_vol_state == "High" else 1.0,
    ))

    # F3: Reduce size 40% when 1H trend misaligned
    out.extend(_test_filter(
        "F3: Size x0.60 when NQ 1H trend OPPOSES direction",
        lambda t: True,
        size_fn=lambda t: 0.60 if t.vb_trend_1h == -t.direction else 1.0,
    ))

    # F4: Block when 15m momentum slope opposes
    out.extend(_test_filter(
        "F4: Block when 15m slope OPPOSES direction",
        lambda t: (t.direction > 0 and t.vb_slope_long_ok) or
                  (t.direction < 0 and t.vb_slope_short_ok),
    ))

    # F5: Raise score threshold when daily trend flat
    out.extend(_test_filter(
        "F5: Block when daily trend FLAT and score < 2.5",
        lambda t: not (t.vb_daily_trend == 0 and t.score_at_entry < 2.5),
    ))

    # F6a: Combo - F1 + F3
    out.extend(_test_filter(
        "F6a: Block daily-oppose + Size x0.60 when 1H-oppose",
        lambda t: t.vb_daily_trend != -t.direction,
        size_fn=lambda t: 0.60 if (t.vb_daily_trend != -t.direction and t.vb_trend_1h == -t.direction) else 1.0,
    ))

    # F6b: Combo - F1 + F3 + F4
    def _f6b_keep(t):
        if t.vb_daily_trend == -t.direction:
            return False
        slope_ok = (t.direction > 0 and t.vb_slope_long_ok) or (t.direction < 0 and t.vb_slope_short_ok)
        if not slope_ok:
            return False
        return True

    out.extend(_test_filter(
        "F6b: Block daily-oppose + Block slope-oppose + Size x0.60 1H-oppose",
        _f6b_keep,
        size_fn=lambda t: 0.60 if t.vb_trend_1h == -t.direction else 1.0,
    ))

    # F7: Combined Vdubus score filter
    def _vb_score(t):
        s = 0
        if t.vb_daily_trend == t.direction: s += 1
        if t.vb_vol_state == "Normal": s += 1
        if t.vb_trend_1h == t.direction: s += 1
        if (t.direction > 0 and t.vb_slope_long_ok) or (t.direction < 0 and t.vb_slope_short_ok): s += 1
        return s

    out.extend(_test_filter(
        "F7: Block when Vdubus score <= 1 (out of 4)",
        lambda t: _vb_score(t) >= 2,
    ))

    # F8: Score-based sizing
    out.append(f"\n  --- F8: Vdubus-score-based sizing ---")
    score_mults = {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.0, 4: 1.0}
    adj_rs_f8 = []
    for t in trades:
        s = _vb_score(t)
        mult = score_mults[s]
        adj_rs_f8.append(t.r_multiple * mult)
    out.append(f"  Score mult map: {score_mults}")
    out.append(f"  Adjusted total R: {sum(adj_rs_f8):+.1f}  (baseline {_stats(trades)['total_r']:+.1f})")
    out.append(f"  Delta: {sum(adj_rs_f8) - _stats(trades)['total_r']:+.1f}R")
    out.append(f"  Trades effectively blocked (score=0): {sum(1 for t in trades if _vb_score(t) == 0)}")
    out.append(f"  Trades at 50% size (score=1): {sum(1 for t in trades if _vb_score(t) == 1)}")
    out.append(f"  Trades at 75% size (score=2): {sum(1 for t in trades if _vb_score(t) == 2)}")
    out.append(f"  Trades at full size (score>=3): {sum(1 for t in trades if _vb_score(t) >= 3)}")

    # ── Section G: Win amplification analysis ──────────────────────
    out.append(f"\n{'=' * 90}")
    out.append("  SECTION G: Win Amplification Analysis")
    out.append(f"{'=' * 90}")

    # Winners on Vdubus-days: exit reason distribution
    vd_wins = [t for t in vd if t.won]
    nv_wins = [t for t in nv if t.won]
    out.append(f"\n--- Winning Trade Exit Reasons ---\n")
    all_reasons_win = sorted(set(t.exit_reason for t in vd_wins + nv_wins))
    out.append(f"  {'Exit Reason':<30s}  Vdubus-day wins   Non-Vdubus wins")
    out.append(f"  {'-'*70}")
    for reason in all_reasons_win:
        cnt_vd = sum(1 for t in vd_wins if t.exit_reason == reason)
        cnt_nv = sum(1 for t in nv_wins if t.exit_reason == reason)
        avgr_vd = np.mean([t.r_multiple for t in vd_wins if t.exit_reason == reason]) if cnt_vd else 0
        avgr_nv = np.mean([t.r_multiple for t in nv_wins if t.exit_reason == reason]) if cnt_nv else 0
        out.append(f"  {reason:<30s}  {cnt_vd:>3d} avgR={avgr_vd:+.2f}         "
                   f"{cnt_nv:>3d} avgR={avgr_nv:+.2f}")

    # Hold duration comparison
    out.append(f"\n--- Hold Duration (30m bars) ---\n")
    out.append(_pctiles([t.bars_held_30m for t in vd], "Vdubus-day duration"))
    out.append(_pctiles([t.bars_held_30m for t in nv], "Non-Vdubus-day duration"))
    out.append("")
    out.append(_pctiles([t.bars_held_30m for t in vd_wins], "Vdubus-day WINNERS duration"))
    out.append(_pctiles([t.bars_held_30m for t in nv_wins], "Non-Vdubus WINNERS duration"))

    # Exit tier distribution for winners
    out.append(f"\n--- Exit Tier Distribution (winners only) ---\n")
    for tier in sorted(set(t.exit_tier for t in vd_wins + nv_wins)):
        cnt_vd = sum(1 for t in vd_wins if t.exit_tier == tier)
        cnt_nv = sum(1 for t in nv_wins if t.exit_tier == tier)
        avgr_vd = np.mean([t.r_multiple for t in vd_wins if t.exit_tier == tier]) if cnt_vd else 0
        avgr_nv = np.mean([t.r_multiple for t in nv_wins if t.exit_tier == tier]) if cnt_nv else 0
        out.append(f"  {tier:<15s}  Vdubus-day: {cnt_vd:>3d} avgR={avgr_vd:+.2f}   "
                   f"Non-Vdubus: {cnt_nv:>3d} avgR={avgr_nv:+.2f}")

    # Aligned-condition analysis: what if we forced ALIGNED tier when conditions good?
    out.append(f"\n--- Aligned-Condition Trades (daily+1H+vol all favorable) ---\n")
    fully_aligned = [t for t in trades if
                     t.vb_daily_trend == t.direction and
                     t.vb_trend_1h == t.direction and
                     t.vb_vol_state == "Normal"]
    not_aligned = [t for t in trades if t not in fully_aligned]  # set difference is slow, but n is small
    not_aligned_set = set(id(t) for t in fully_aligned)
    not_aligned = [t for t in trades if id(t) not in not_aligned_set]
    out.append(_fmt(_stats(fully_aligned), "Fully aligned (daily+1H+vol)"))
    out.append(_fmt(_stats(not_aligned), "Not fully aligned"))
    out.append("")

    # Among fully aligned, what tier are they on?
    for tier in sorted(set(t.exit_tier for t in fully_aligned)):
        tier_trades = [t for t in fully_aligned if t.exit_tier == tier]
        out.append(f"  Fully aligned + exit_tier={tier}: {_fmt(_stats(tier_trades), '')}")

    out.append(f"\n  Implication: If Fully aligned trades that are NOT on Aligned tier "
               f"were upgraded to Aligned,")
    upgradeable = [t for t in fully_aligned if t.exit_tier != "Aligned"]
    out.append(f"  {len(upgradeable)} trades could potentially benefit from wider TP targets.")

    out.append(f"\n{'=' * 90}")

    return "\n".join(out)


# ====================================================================
# Main
# ====================================================================

def main():
    vdubus_dates = run_vdubus_get_dates()
    raw_trades = run_nqdtc()

    logger.info("Computing Vdubus regime indicators...")
    vb_lookup = _build_vdubus_regime_lookup()

    trades = enrich_trades(raw_trades, vdubus_dates, vb_lookup)
    logger.info("Enriched %d NQDTC trades", len(trades))

    report = build_report(trades)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "nqdtc_vdubus_synergy.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
