"""End-to-end validation: incremental indicators vs batch on real NQ data."""
import sys
from pathlib import Path
import numpy as np

# Load data using the same functions as the CLI
from backtest.data.cache import load_bars
from backtest.data.preprocessing import (
    build_numpy_arrays,
    filter_vdubus_session,
    normalize_timezone,
    resample_15m_to_1h,
)
from strategy_3 import indicators as ind
from strategy_3 import config as C

data_dir = Path("backtest/data/raw")
fifteen_min_path = data_dir / "NQ_15m.parquet"

m_df = normalize_timezone(load_bars(fifteen_min_path))
m_df = filter_vdubus_session(m_df)
bars_15m = build_numpy_arrays(m_df)
hourly_df = resample_15m_to_1h(m_df)
hourly = build_numpy_arrays(hourly_df)

n = len(bars_15m)
print(f"Loaded {n} 15m bars, {len(hourly)} 1H bars")

# ---- 1. ATR validation ----
print("\n=== ATR Validation ===")
batch_atr = ind.atr(bars_15m.highs, bars_15m.lows, bars_15m.closes, C.VOL_ATR_PERIOD)
inc_atr = ind.IncrementalATR(n, C.VOL_ATR_PERIOD)
for t in range(n):
    inc_atr.update(t, float(bars_15m.highs[t]), float(bars_15m.lows[t]),
                   float(bars_15m.closes[t]))

# Compare where both are non-NaN
both_valid = ~np.isnan(batch_atr) & ~np.isnan(inc_atr.values)
if both_valid.sum() == 0:
    print("WARNING: no valid ATR values to compare")
else:
    diff = np.abs(batch_atr[both_valid] - inc_atr.values[both_valid])
    print(f"  Valid points:  {both_valid.sum()}")
    print(f"  Max abs diff:  {diff.max():.2e}")
    print(f"  Mean abs diff: {diff.mean():.2e}")
    if diff.max() > 1e-10:
        # Show first few mismatches
        bad = np.where(both_valid & (np.abs(batch_atr - inc_atr.values) > 1e-10))[0]
        print(f"  WARNING: {len(bad)} mismatches > 1e-10")
        for idx in bad[:5]:
            print(f"    t={idx}: batch={batch_atr[idx]:.10f} inc={inc_atr.values[idx]:.10f}")
    else:
        print("  PASS: ATR matches exactly")

# ---- 2. MACD validation ----
print("\n=== MACD Validation ===")
batch_macd = ind.macd_hist(bars_15m.closes)
inc_macd = ind.IncrementalMACD(n)
for t in range(n):
    inc_macd.update(t, float(bars_15m.closes[t]))

both_valid = ~np.isnan(batch_macd) & ~np.isnan(inc_macd.values)
if both_valid.sum() == 0:
    print("WARNING: no valid MACD values to compare")
else:
    diff = np.abs(batch_macd[both_valid] - inc_macd.values[both_valid])
    print(f"  Valid points:  {both_valid.sum()}")
    print(f"  Max abs diff:  {diff.max():.2e}")
    print(f"  Mean abs diff: {diff.mean():.2e}")
    if diff.max() > 1e-10:
        bad = np.where(both_valid & (np.abs(batch_macd - inc_macd.values) > 1e-10))[0]
        print(f"  WARNING: {len(bad)} mismatches > 1e-10")
        for idx in bad[:5]:
            print(f"    t={idx}: batch={batch_macd[idx]:.10f} inc={inc_macd.values[idx]:.10f}")
    else:
        print("  PASS: MACD matches exactly (within 1e-10)")

# ---- 3. 1H ATR pre-computation validation ----
print("\n=== 1H ATR Pre-computation Validation ===")
full_1h_atr = ind.atr(hourly.highs, hourly.lows, hourly.closes)
# Simulate what _on_1h_boundary does: at each h_idx, compare
# full_1h_atr[:h_idx+1][-1] vs recomputing atr(hourly[:h_idx+1])
n_1h = len(hourly)
mismatches_1h = 0
for h_idx in range(C.VOL_ATR_PERIOD, n_1h, 10):  # sample every 10th
    slice_val = ind.atr(hourly.highs[:h_idx+1], hourly.lows[:h_idx+1],
                        hourly.closes[:h_idx+1])[-1]
    precomp_val = full_1h_atr[h_idx]
    if abs(slice_val - precomp_val) > 1e-10:
        mismatches_1h += 1
        if mismatches_1h <= 3:
            print(f"  h_idx={h_idx}: slice={slice_val:.10f} precomp={precomp_val:.10f}")

tested = (n_1h - C.VOL_ATR_PERIOD) // 10 + 1
print(f"  Tested {tested} hourly indices")
if mismatches_1h == 0:
    print("  PASS: 1H ATR pre-computation is identical to on-the-fly slicing")
else:
    print(f"  FAIL: {mismatches_1h} mismatches")

# ---- 4. Pivot pre-computation validation ----
print("\n=== 1H Pivot Pre-computation Validation ===")
all_pivots = ind.confirmed_pivots(hourly.highs, hourly.lows, C.NCONFIRM_1H)
mismatches_piv = 0
# At each h_idx, pivots visible should be those with confirmed_at <= h_idx
for h_idx in range(20, n_1h, 50):  # sample every 50th
    slice_pivots = ind.confirmed_pivots(hourly.highs[:h_idx+1], hourly.lows[:h_idx+1],
                                         C.NCONFIRM_1H)
    filtered_pivots = [p for p in all_pivots if p.confirmed_at <= h_idx]
    if len(slice_pivots) != len(filtered_pivots):
        mismatches_piv += 1
        if mismatches_piv <= 3:
            print(f"  h_idx={h_idx}: slice={len(slice_pivots)} filtered={len(filtered_pivots)}")

tested_piv = (n_1h - 20) // 50 + 1
print(f"  Tested {tested_piv} hourly indices")
if mismatches_piv == 0:
    print("  PASS: Pivot pre-computation with confirmed_at filter matches slicing")
else:
    print(f"  FAIL: {mismatches_piv} mismatches")

# ---- 5. Session VWAP validation ----
print("\n=== Session VWAP Spot-check ===")
# Build session_starts the same way the engine does
from datetime import timezone, timedelta
ET = timezone(timedelta(hours=-5))

session_starts = {}
for i in range(n):
    bt = bars_15m.times[i]
    if hasattr(bt, 'item'):
        bt = bt.item()
    if hasattr(bt, 'astimezone'):
        et = bt.astimezone(ET)
    else:
        et = bt
    date_str = str(et)[:10]
    if date_str not in session_starts:
        session_starts[date_str] = i

# Incremental session VWAP
svwap_inc = np.full(n, np.nan)
cum_tpv = 0.0
cum_vol = 0.0
cur_sess = -1
for t in range(n):
    bt = bars_15m.times[t]
    if hasattr(bt, 'item'):
        bt = bt.item()
    if hasattr(bt, 'astimezone'):
        et = bt.astimezone(ET)
    else:
        et = bt
    date_str = str(et)[:10]
    ss = session_starts.get(date_str, 0)
    if ss != cur_sess:
        cur_sess = ss
        cum_tpv = 0.0
        cum_vol = 0.0
    if t >= ss:
        H = float(bars_15m.highs[t])
        L = float(bars_15m.lows[t])
        C_ = float(bars_15m.closes[t])
        V_ = max(float(bars_15m.volumes[t]), 1.0)
        tp = (H + L + C_) / 3.0
        cum_tpv += tp * V_
        cum_vol += V_
        svwap_inc[t] = cum_tpv / cum_vol

# Batch session VWAP for a few sessions
sample_dates = list(session_starts.keys())[:5]
vwap_diffs = 0
for date_str in sample_dates:
    ss = session_starts[date_str]
    batch_vwap = ind.session_vwap(bars_15m.highs, bars_15m.lows, bars_15m.closes,
                                   bars_15m.volumes, ss)
    # Find end of this session (next session start or end of data)
    all_starts = sorted(session_starts.values())
    si = all_starts.index(ss)
    end = all_starts[si + 1] if si + 1 < len(all_starts) else n
    for t in range(ss, min(end, n)):
        if not np.isnan(batch_vwap[t]) and not np.isnan(svwap_inc[t]):
            if abs(batch_vwap[t] - svwap_inc[t]) > 1e-10:
                vwap_diffs += 1

print(f"  Checked {len(sample_dates)} sessions")
if vwap_diffs == 0:
    print("  PASS: Session VWAP incremental matches batch")
else:
    print(f"  FAIL: {vwap_diffs} mismatches")

print("\n=== All Validation Complete ===")
