"""Monitor optimizer PID and run full diagnostics when complete."""
import json
import os
import sys
import time
import dataclasses

# Wait for optimizer to finish
PID = 8879
print(f"Waiting for PID {PID} to finish...")
while True:
    try:
        os.kill(PID, 0)
        time.sleep(30)
    except OSError:
        print(f"PID {PID} finished.")
        break

PROJECT = "C:/Users/sehyu/Documents/Other/Projects/trading"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# Load final phase state
state = json.load(open("backtests/regime/auto/output/phase_state.json"))
muts = state["cumulative_mutations"]

out = []
def p(s=""):
    print(s)
    out.append(s)

p(f"Final mutations ({len(muts)}):")
p(json.dumps(muts, indent=2))
p()
p(f"Completed phases: {state['completed_phases']}")
for ph, r in state["phase_results"].items():
    m = r.get("final_metrics", {})
    p(f"Phase {ph}: {r['baseline_score']:.4f} -> {r['final_score']:.4f} | rounds: {r['rounds']}")
    p(f"  Sharpe={m.get('sharpe',0):.3f} Sortino={m.get('sortino',0):.3f} MaxDD={m.get('max_drawdown_pct',0):.1%} CAGR={m.get('cagr',0):.2%} Return={m.get('total_return',0):.1%}")
p()

# Run full engine with optimized config
import numpy as np
import pandas as pd
from regime.config import MetaConfig
from regime.engine import run_signal_engine

cfg = dataclasses.replace(MetaConfig(), **muts)

macro_df = pd.read_parquet("backtests/regime/data/macro_features.parquet")
strat_ret_df = pd.read_parquet("backtests/regime/data/strategy_returns.parquet")
market_df = pd.read_parquet("backtests/regime/data/market_stress.parquet")

p("=== Data ===")
p(f"macro: {macro_df.index.min().date()} to {macro_df.index.max().date()} ({len(macro_df)} rows)")
p(f"strat: {strat_ret_df.index.min().date()} to {strat_ret_df.index.max().date()} ({len(strat_ret_df)} rows)")
p(f"market: {market_df.index.min().date()} to {market_df.index.max().date()} ({len(market_df)} rows)")
p()

signals = run_signal_engine(
    macro_df=macro_df, strat_ret_df=strat_ret_df, market_df=market_df,
    growth_feature="OECD_CLI", inflation_feature="OECD_CPI", cfg=cfg,
)
p(f"Signals: {len(signals)} rows, {signals.index.min().date()} to {signals.index.max().date()}")
p(f"Columns: {list(signals.columns)}")
p()

# Posterior analysis
P_cols = ["P_G", "P_R", "P_S", "P_D"]
P = signals[P_cols]
dominant = P.idxmax(axis=1)
dom_prob = P.max(axis=1)

p("=== Posterior Analysis ===")
p(f"Avg P(dominant): {dom_prob.mean():.4f}")
p(f"Min P(dominant): {dom_prob.min():.4f}")
p(f"Max P(dominant): {dom_prob.max():.4f}")
p(f"Std P(dominant): {dom_prob.std():.4f}")
p()

entropy = -(P * np.log(P.clip(lower=1e-12))).sum(axis=1)
H_max = np.log(4)
p(f"Avg posterior entropy: {entropy.mean():.4f} (max={H_max:.4f})")
p(f"Normalized entropy: {(entropy / H_max).mean():.4f}")
p()

# Regime distribution
p("=== Regime Distribution ===")
regime_counts = dominant.value_counts()
for r in P_cols:
    pct = regime_counts.get(r, 0) / len(dominant) * 100
    avg_p = P[r].mean()
    p(f"  {r}: {pct:.1f}% dominant, avg prob={avg_p:.4f}")
p()

p("=== Per-Regime Avg P(dominant) ===")
for r in P_cols:
    mask = dominant == r
    if mask.sum() > 0:
        p(f"  {r}: AvgP(dom)={dom_prob[mask].mean():.4f}, count={mask.sum()}")
p()

# Confidence
p("=== Confidence ===")
p(f"Avg: {signals['Conf'].mean():.4f}, Std: {signals['Conf'].std():.4f}, Min: {signals['Conf'].min():.4f}, Max: {signals['Conf'].max():.4f}")
p()

# Momentum
if "risk_momentum" in signals.columns:
    rm = signals["risk_momentum"]
    p("=== Regime Momentum ===")
    p(f"Avg: {rm.mean():.6f}, Std: {rm.std():.6f}")
    p(f"Min: {rm.min():.6f}, Max: {rm.max():.6f}")
    neg = (rm < -0.05).sum()
    pos = (rm > 0.05).sum()
    p(f"Periods risk_mom < -0.05: {neg} ({neg/len(rm)*100:.1f}%)")
    p(f"Periods risk_mom > +0.05: {pos} ({pos/len(rm)*100:.1f}%)")
    p()

# Crisis
p("=== Crisis Overlay ===")
crisis = signals["p_crisis"]
p(f"Avg: {crisis.mean():.4f}, Max: {crisis.max():.4f}")
active = (crisis > 0.5).sum()
p(f"Weeks p_crisis > 0.5: {active} ({active/len(crisis)*100:.1f}%)")
p()

# Leverage
L = signals["L"]
p("=== Leverage ===")
p(f"Avg: {L.mean():.4f}, Std: {L.std():.4f}, Min: {L.min():.4f}, Max: {L.max():.4f}")
p()

# Portfolio simulation
from backtests.regime.portfolio_sim import simulate_portfolio
perf = simulate_portfolio(signals, strat_ret_df, cfg)
p("=== Portfolio Performance ===")
for k, v in perf.items():
    if isinstance(v, float):
        if any(x in k for x in ["return", "cagr", "drawdown", "turnover", "vol"]):
            p(f"  {k}: {v:.4f} ({v*100:.2f}%)")
        else:
            p(f"  {k}: {v:.4f}")
    else:
        p(f"  {k}: {v}")
p()

# Weight analysis
w_cols = [c for c in signals.columns if c.startswith("w_")]
p("=== Average Allocation Weights ===")
for c in w_cols:
    p(f"  {c}: {signals[c].mean():.4f}")
p()

p("=== Weight Stability (week-over-week) ===")
for c in w_cols:
    chg = signals[c].diff().dropna()
    p(f"  {c}: avg_abs_change={chg.abs().mean():.4f}, std={chg.std():.4f}")
p()

# Transitions
p("=== Regime Transitions ===")
transitions = (dominant != dominant.shift()).sum()
p(f"Total: {transitions}, Rate: {transitions/len(dominant):.4f} ({transitions/len(dominant)*52:.1f}/year)")
p()

# Year-by-year
p("=== Year-by-Year Dominant Regime ===")
regime_map = {"P_G": "G", "P_R": "R", "P_S": "S", "P_D": "D"}
for year in range(signals.index.year.min(), signals.index.year.max() + 1):
    mask = signals.index.year == year
    if mask.sum() == 0:
        continue
    yr_dom = dominant[mask].map(regime_map).value_counts()
    n = mask.sum()
    parts = [f"{r}={int(c/n*100)}%" for r, c in yr_dom.items()]
    p(f"  {year}: {', '.join(parts)}")
p()

# Crisis periods
p("=== Crisis Period Performance ===")
crisis_periods = {
    "GFC (2008-09 to 2009-03)": ("2008-09-01", "2009-03-31"),
    "Euro Crisis (2011-08 to 2011-11)": ("2011-08-01", "2011-11-30"),
    "COVID (2020-02 to 2020-04)": ("2020-02-15", "2020-04-15"),
    "Rate Hike (2022-01 to 2022-10)": ("2022-01-01", "2022-10-31"),
    "SVB (2023-03)": ("2023-03-01", "2023-03-31"),
}
for name, (start, end) in crisis_periods.items():
    mask = (signals.index >= start) & (signals.index <= end)
    if mask.sum() == 0:
        p(f"  {name}: no data")
        continue
    cd = signals[mask]
    avg_cp = cd["p_crisis"].mean()
    dom_r = cd[P_cols].idxmax(axis=1).map(regime_map).value_counts()
    n = len(cd)
    r_str = ", ".join([f"{r}={int(c/n*100)}%" for r, c in dom_r.items()])
    avg_L = cd["L"].mean()
    rm_val = cd["risk_momentum"].mean() if "risk_momentum" in cd.columns else 0
    p(f"  {name}:")
    p(f"    Regimes: {r_str}")
    p(f"    p_crisis={avg_cp:.3f}, L={avg_L:.3f}, risk_momentum={rm_val:.4f}")
p()

# Save outputs
signals.to_parquet("backtests/regime/auto/output/optimized_signals.parquet")
p("Saved: backtests/regime/auto/output/optimized_signals.parquet")

cfg_dict = dataclasses.asdict(cfg)
for k, v in cfg_dict.items():
    if isinstance(v, (set, frozenset)):
        cfg_dict[k] = list(v)
    elif isinstance(v, tuple):
        cfg_dict[k] = list(v)
json.dump(cfg_dict, open("backtests/regime/auto/output/optimized_config.json", "w"), indent=2)
p("Saved: backtests/regime/auto/output/optimized_config.json")

# Save diagnostics text
with open("backtests/regime/auto/output/full_diagnostics.txt", "w") as f:
    f.write("\n".join(out))
p("\nSaved: backtests/regime/auto/output/full_diagnostics.txt")
p("\n=== DONE ===")
