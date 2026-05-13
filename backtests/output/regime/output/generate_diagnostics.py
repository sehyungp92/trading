"""Generate full diagnostics for the optimized regime config."""
import json
import dataclasses
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Users/sehyu/Documents/Other/Projects/trading")

from backtests.regime._aliases import install
install()

state = json.load(open("C:/Users/sehyu/Documents/Other/Projects/trading/backtests/regime/auto/output/phase_state.json"))
muts = dict(state["cumulative_mutations"])
if "crisis_weights" in muts and isinstance(muts["crisis_weights"], list):
    muts["crisis_weights"] = tuple(muts["crisis_weights"])

from regime.config import MetaConfig
from backtests.regime.config import RegimeBacktestConfig
from backtests.regime.engine.portfolio_sim import simulate_portfolio
from backtests.regime.data.downloader import load_cached_data
from regime.engine import run_signal_engine

cfg = dataclasses.replace(MetaConfig(), **muts)

# Load data and regenerate signals with current mutations
from pathlib import Path
data_dir = Path("C:/Users/sehyu/Documents/Other/Projects/trading/backtests/regime/data/raw")
macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

print("Running signal engine with optimized config...", flush=True)
signals = run_signal_engine(
    macro_df=macro_df, strat_ret_df=strat_ret_df, market_df=market_df,
    growth_feature="GROWTH", inflation_feature="INFLATION", cfg=cfg,
)
signals.to_parquet("C:/Users/sehyu/Documents/Other/Projects/trading/backtests/regime/auto/output/optimized_signals.parquet")
print(f"Signals generated: {len(signals)} rows", flush=True)

sim_cfg = RegimeBacktestConfig(initial_equity=100_000, rebalance_cost_bps=5)
perf = simulate_portfolio(signals, strat_ret_df, sim_cfg)

out = []
def p(s=""):
    print(s)
    out.append(s)

p("=" * 70)
p("REGIME PREDICTOR - FULL DIAGNOSTICS (EMA-Only Optimization)")
p("=" * 70)
p()
p(f"Mutations ({len(muts)}):")
p(json.dumps(muts, indent=2))
p()

for ph, r in state["phase_results"].items():
    m = r.get("final_metrics", {})
    p(f"Phase {ph}: {r['baseline_score']:.4f} -> {r['final_score']:.4f} | rounds: {r['rounds']}")
    p(f"  Sharpe={m.get('sharpe',0):.3f} Sortino={m.get('sortino',0):.3f} MaxDD={m.get('max_drawdown_pct',0):.1%} CAGR={m.get('cagr',0):.2%} Return={m.get('total_return',0):.1%}")
p()

p(f"Signals: {len(signals)} rows, {signals.index.min().date()} to {signals.index.max().date()}")
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
entropy = -(P * np.log(P.clip(lower=1e-12))).sum(axis=1)
H_max = np.log(4)
p(f"Avg posterior entropy: {entropy.mean():.4f} (max={H_max:.4f})")
p(f"Normalized entropy: {(entropy / H_max).mean():.4f}")
p()

# Regime distribution
p("=== Regime Distribution ===")
rc = dominant.value_counts()
for r in P_cols:
    pct = rc.get(r, 0) / len(dominant) * 100
    avg_p = P[r].mean()
    mask = dominant == r
    avg_dom = dom_prob[mask].mean() if mask.sum() > 0 else 0
    p(f"  {r}: {pct:.1f}% dominant ({rc.get(r,0)} wks), avg_prob={avg_p:.4f}, AvgP(dom)={avg_dom:.4f}")
p()

# Confidence
p("=== Confidence ===")
c = signals["Conf"]
p(f"Avg={c.mean():.4f} Std={c.std():.4f} Min={c.min():.4f} Max={c.max():.4f}")
p()

# Momentum
if "risk_momentum" in signals.columns:
    rm = signals["risk_momentum"]
    p("=== Regime Momentum ===")
    p(f"Avg={rm.mean():.6f} Std={rm.std():.6f} Min={rm.min():.6f} Max={rm.max():.6f}")
    neg = int((rm < -0.05).sum())
    pos = int((rm > 0.05).sum())
    p(f"risk_mom < -0.05: {neg} ({neg/len(rm)*100:.1f}%)  |  risk_mom > +0.05: {pos} ({pos/len(rm)*100:.1f}%)")
    p()

# Crisis
p("=== Crisis Overlay ===")
crisis = signals["p_crisis"]
p(f"Avg={crisis.mean():.4f} Max={crisis.max():.4f}")
active = int((crisis > 0.5).sum())
p(f"Weeks p_crisis > 0.5: {active} ({active/len(crisis)*100:.1f}%)")
p()

# Leverage
Lv = signals["L"]
p("=== Leverage ===")
p(f"Avg={Lv.mean():.4f} Std={Lv.std():.4f} Min={Lv.min():.4f} Max={Lv.max():.4f}")
p()

# Portfolio performance
p("=== Portfolio Performance ===")
for k, v in dataclasses.asdict(perf).items():
    if isinstance(v, float):
        if any(x in k for x in ["return", "cagr", "drawdown", "turnover", "vol"]):
            p(f"  {k}: {v:.4f} ({v*100:.2f}%)")
        else:
            p(f"  {k}: {v:.4f}")
    else:
        p(f"  {k}: {v}")
p()

# Weight analysis
w_cols = [col for col in signals.columns if col.startswith("w_")]
p("=== Average Allocation Weights ===")
for col in w_cols:
    p(f"  {col}: {signals[col].mean():.4f}")
p()

p("=== Weight Stability (week-over-week) ===")
for col in w_cols:
    chg = signals[col].diff().dropna()
    p(f"  {col}: avg_abs_change={chg.abs().mean():.4f} std={chg.std():.4f}")
p()

# Transitions
transitions = int((dominant != dominant.shift()).sum())
p("=== Regime Transitions ===")
p(f"Total={transitions} Rate={transitions/len(dominant):.4f} ({transitions/len(dominant)*52:.1f}/year)")
p()

# Year-by-year
regime_map = {"P_G": "G", "P_R": "R", "P_S": "S", "P_D": "D"}
p("=== Year-by-Year Dominant Regime ===")
for year in range(signals.index.year.min(), signals.index.year.max() + 1):
    mask = signals.index.year == year
    if mask.sum() == 0:
        continue
    yr_dom = dominant[mask].map(regime_map).value_counts()
    n = mask.sum()
    parts = [f"{r}={int(cnt/n*100)}%" for r, cnt in yr_dom.items()]
    p(f"  {year}: {', '.join(parts)}")
p()

# Crisis periods
p("=== Crisis Period Performance ===")
crisis_periods = [
    ("GFC (2008-09 to 2009-03)", "2008-09-01", "2009-03-31"),
    ("Euro Crisis (2011-08 to 2011-11)", "2011-08-01", "2011-11-30"),
    ("COVID (2020-02 to 2020-04)", "2020-02-15", "2020-04-15"),
    ("Rate Hike (2022-01 to 2022-10)", "2022-01-01", "2022-10-31"),
    ("SVB (2023-03)", "2023-03-01", "2023-03-31"),
]
for name, start, end in crisis_periods:
    mask = (signals.index >= start) & (signals.index <= end)
    if mask.sum() == 0:
        p(f"  {name}: no data")
        continue
    cd = signals[mask]
    avg_cp = cd["p_crisis"].mean()
    dom_r = cd[P_cols].idxmax(axis=1).map(regime_map).value_counts()
    n = len(cd)
    r_str = ", ".join([f"{r}={int(cnt/n*100)}%" for r, cnt in dom_r.items()])
    avg_L = cd["L"].mean()
    rm_val = cd["risk_momentum"].mean() if "risk_momentum" in cd.columns else 0
    p(f"  {name}:")
    p(f"    Regimes: {r_str}")
    p(f"    p_crisis={avg_cp:.3f} L={avg_L:.3f} risk_momentum={rm_val:.4f}")
p()

# Save config
cfg_dict = dataclasses.asdict(cfg)
for k, v in cfg_dict.items():
    if isinstance(v, (set, frozenset)):
        cfg_dict[k] = list(v)
    elif isinstance(v, tuple):
        cfg_dict[k] = list(v)
json.dump(cfg_dict, open("C:/Users/sehyu/Documents/Other/Projects/trading/backtests/regime/auto/output/optimized_config.json", "w"), indent=2)
p("Saved: optimized_config.json")

with open("C:/Users/sehyu/Documents/Other/Projects/trading/backtests/regime/auto/output/full_diagnostics.txt", "w") as f:
    f.write("\n".join(out))
p("Saved: full_diagnostics.txt")
p()
p("=== DONE ===")
