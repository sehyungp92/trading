"""Full visual diagnostics & critical evaluation of the optimized regime predictor.

Loads the optimal config from phase_state.json, runs the full pipeline, generates
6 charts + text diagnostics + critical evaluation.

Usage:
    python backtests/regime/auto/output/evaluate_optimal.py
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ── project imports ──────────────────────────────────────────────────────
from regime.config import MetaConfig, REGIMES
from regime.engine import run_signal_engine
from regime.features import build_observation_matrix
from regime.hmm import fit_or_refit_hmm
from backtests.regime.auto.config_mutator import mutate_meta_config
from backtests.regime.auto.scoring import composite_score
from backtests.regime.auto.phase_scoring import compute_regime_stats
from backtests.regime.config import RegimeBacktestConfig
from backtests.regime.data.downloader import load_cached_data
from backtests.regime.engine.portfolio_sim import (
    simulate_benchmark_60_40,
    simulate_portfolio,
)
from backtests.regime.analysis.diagnostics import (
    generate_regime_diagnostics_report,
)

# ── constants ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("backtests/regime/auto/output")
PHASE_STATE = OUTPUT_DIR / "phase_state.json"

REGIME_COLORS = {"G": "#2e7d32", "R": "#e65100", "S": "#c62828", "D": "#1565c0"}
REGIME_NAMES = {"G": "Recovery", "R": "Reflation", "S": "Infl Hedge", "D": "Defensive"}

# Known crisis windows (start, end, label)
CRISIS_WINDOWS = [
    ("2007-10-01", "2009-03-31", "GFC"),
    ("2020-02-01", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "Inflation"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Data pipeline
# ═══════════════════════════════════════════════════════════════════════════


def _load_mutations() -> dict:
    with open(PHASE_STATE) as f:
        state = json.load(f)
    return state["cumulative_mutations"]


def _dominant_regime(signals: pd.DataFrame) -> pd.Series:
    cols = [f"P_{r}" for r in REGIMES]
    available = [c for c in cols if c in signals.columns]
    return signals[available].idxmax(axis=1).str.replace("P_", "")


def _run_pipeline():
    """Run the full regime pipeline with optimal mutations."""
    print("Loading mutations from phase_state.json ...")
    mutations = _load_mutations()
    mutations["rebalance_freq"] = "B"  # daily rebalancing for final eval

    print(f"  {len(mutations)} mutations loaded")
    cfg = mutate_meta_config(MetaConfig(), mutations)

    print("Loading cached data ...")
    macro_df, market_df, strat_ret_df = load_cached_data()

    print("Running signal engine ...")
    sim_cfg = RegimeBacktestConfig()
    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature=sim_cfg.growth_feature,
        inflation_feature=sim_cfg.inflation_feature,
        cfg=cfg,
    )
    print(f"  Signals: {len(signals)} rows, {signals.columns.tolist()[:8]} ...")

    print("Simulating portfolio ...")
    result = simulate_portfolio(signals, strat_ret_df, sim_cfg)

    print("Simulating 60/40 benchmark ...")
    benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)

    print("Computing scores ...")
    score = composite_score(result.metrics)
    regime_stats = compute_regime_stats(signals)

    # Extract HMM state feature means for diagnostics
    print("Extracting HMM state feature means ...")
    Xz, g_idx, i_idx = build_observation_matrix(
        macro_df, market_df, strat_ret_df, cfg,
        growth_feature=sim_cfg.growth_feature,
        inflation_feature=sim_cfg.inflation_feature,
    )
    feature_names = list(Xz.columns)

    if cfg.use_expanding_window:
        X_fit = Xz.values
    else:
        cutoff = Xz.index.max() - pd.DateOffset(years=cfg.rolling_window_years)
        X_fit = Xz.loc[cutoff:].values
        if len(X_fit) < 252:
            X_fit = Xz.values

    hmm_model, _ = fit_or_refit_hmm(
        X_train=X_fit, cfg=cfg,
        growth_idx=g_idx, infl_idx=i_idx,
        prev_model=None, first_fit=True,
    )
    hmm_state_means = pd.DataFrame(
        hmm_model.means_,
        index=REGIMES,
        columns=feature_names,
    )
    print(f"  Features: {feature_names}")
    print(f"  Fitted on {len(X_fit)} observations")

    return signals, result, benchmark, score, regime_stats, sim_cfg, market_df, strat_ret_df, hmm_state_means


# ═══════════════════════════════════════════════════════════════════════════
# Chart helpers
# ═══════════════════════════════════════════════════════════════════════════


def _add_regime_shading(ax, signals: pd.DataFrame, alpha: float = 0.12):
    """Add background colour bands for dominant regime."""
    dom = _dominant_regime(signals)
    prev_regime = None
    start_date = None

    for date, regime in dom.items():
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                ax.axvspan(start_date, date, color=REGIME_COLORS.get(prev_regime, "gray"),
                           alpha=alpha, linewidth=0)
            start_date = date
            prev_regime = regime

    # Final segment
    if prev_regime is not None and start_date is not None:
        ax.axvspan(start_date, dom.index[-1], color=REGIME_COLORS.get(prev_regime, "gray"),
                   alpha=alpha, linewidth=0)


def _add_crisis_bands(ax, alpha: float = 0.25):
    """Add vertical bands for known crisis periods."""
    for start, end, label in CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="red", alpha=alpha, linewidth=0)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1], f" {label}",
                fontsize=7, va="top", ha="left", color="red", alpha=0.8)


def _style_ax(ax, title: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _reconstruct_price(log_returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Reconstruct a price series from cumulative log returns."""
    return base * np.exp(log_returns.cumsum())


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 & 2: Asset price + regime overlay
# ═══════════════════════════════════════════════════════════════════════════


def chart_asset_regime(
    strat_ret_df: pd.DataFrame,
    signals: pd.DataFrame,
    market_df: pd.DataFrame,
    asset: str,
    output_path: Path,
):
    """Price chart with regime shading + p_crisis / VIX panel."""
    price = _reconstruct_price(strat_ret_df[asset])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1],
                                    sharex=True)
    fig.suptitle(f"{asset} with Regime Overlay", fontsize=13, fontweight="bold")

    # Top: price + regime
    ax1.plot(price.index, price.values, color="black", linewidth=0.8, label=f"{asset} price")
    _add_regime_shading(ax1, signals)
    _add_crisis_bands(ax1, alpha=0.08)
    _style_ax(ax1, ylabel="Price (rebased to 100)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

    # Legend for regimes
    for r in REGIMES:
        ax1.fill_between([], [], color=REGIME_COLORS[r], alpha=0.3,
                         label=f"{REGIME_NAMES[r]} ({r})")
    ax1.legend(loc="upper left", fontsize=7, ncol=3)

    # Bottom: p_crisis + VIX
    if "p_crisis" in signals.columns:
        ax2.plot(signals.index, signals["p_crisis"], color="red", linewidth=0.8,
                 label="p_crisis", alpha=0.9)
        ax2.axhline(0.3, color="orange", linestyle="--", linewidth=0.5, alpha=0.6)
        ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.5, alpha=0.6)
        ax2.axhline(0.7, color="darkred", linestyle="--", linewidth=0.5, alpha=0.6)
        ax2.set_ylim(-0.02, 1.02)
        _style_ax(ax2, ylabel="p_crisis")

    if "VIX" in market_df.columns:
        ax2b = ax2.twinx()
        vix = market_df["VIX"].dropna()
        ax2b.plot(vix.index, vix.values, color="purple", linewidth=0.5,
                  label="VIX", alpha=0.5)
        ax2b.set_ylabel("VIX", fontsize=9, color="purple")
        ax2b.tick_params(labelsize=7, colors="purple")

    _add_crisis_bands(ax2, alpha=0.08)
    ax2.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: Regime posterior evolution (stacked area)
# ═══════════════════════════════════════════════════════════════════════════


def chart_regime_posteriors(signals: pd.DataFrame, output_path: Path):
    fig, ax1 = plt.subplots(figsize=(16, 6))

    regime_cols = [f"P_{r}" for r in REGIMES]
    colors = [REGIME_COLORS[r] for r in REGIMES]
    labels = [f"{REGIME_NAMES[r]} ({r})" for r in REGIMES]

    data = signals[regime_cols].values.T
    ax1.stackplot(signals.index, data, colors=colors, labels=labels, alpha=0.75)
    ax1.set_ylim(0, 1)
    _style_ax(ax1, title="Regime Posterior Evolution", ylabel="Probability")
    ax1.legend(loc="upper left", fontsize=8, ncol=4)

    # Confidence overlay
    if "Conf" in signals.columns:
        ax2 = ax1.twinx()
        ax2.plot(signals.index, signals["Conf"], color="black", linewidth=0.7,
                 alpha=0.6, label="Confidence")
        ax2.set_ylabel("Confidence", fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.tick_params(labelsize=7)
        ax2.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: Crisis detection timeline
# ═══════════════════════════════════════════════════════════════════════════


def chart_crisis_detection(signals: pd.DataFrame, market_df: pd.DataFrame, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Crisis Detection Timeline", fontsize=13, fontweight="bold")

    # Top: p_crisis
    if "p_crisis" in signals.columns:
        ax1.fill_between(signals.index, 0, signals["p_crisis"], color="red", alpha=0.3)
        ax1.plot(signals.index, signals["p_crisis"], color="red", linewidth=0.8)
        for thresh, col in [(0.3, "orange"), (0.5, "red"), (0.7, "darkred")]:
            ax1.axhline(thresh, color=col, linestyle="--", linewidth=0.5, alpha=0.7,
                        label=f"t={thresh}")
        ax1.set_ylim(-0.02, 1.02)
        _style_ax(ax1, ylabel="p_crisis")
        ax1.legend(loc="upper right", fontsize=7)
        _add_crisis_bands(ax1, alpha=0.1)

    # Bottom: VIX
    if "VIX" in market_df.columns:
        vix = market_df["VIX"].dropna()
        ax2.plot(vix.index, vix.values, color="purple", linewidth=0.8)
        ax2.fill_between(vix.index, 0, vix.values, color="purple", alpha=0.15)
        ax2.axhline(30, color="orange", linestyle="--", linewidth=0.5, alpha=0.7, label="VIX=30")
        _style_ax(ax2, ylabel="VIX")
        ax2.legend(loc="upper right", fontsize=7)
        _add_crisis_bands(ax2, alpha=0.1)

    # Highlight last 3 months
    last_date = signals.index.max()
    recent_start = last_date - pd.DateOffset(months=3)
    for a in (ax1, ax2):
        a.axvspan(recent_start, last_date, color="yellow", alpha=0.1, linewidth=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 5: Portfolio performance (equity + drawdown)
# ═══════════════════════════════════════════════════════════════════════════


def chart_portfolio_performance(result, benchmark, signals: pd.DataFrame, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1], sharex=True)
    fig.suptitle("Portfolio Performance: Regime vs 60/40", fontsize=13, fontweight="bold")

    eq = result.equity_curve
    beq = benchmark.equity_curve

    # Top: equity curves
    ax1.plot(eq.index, eq.values, color="blue", linewidth=1.2, label="Regime Portfolio")
    ax1.plot(beq.index, beq.values, color="gray", linewidth=1.0, alpha=0.7,
             label="60/40 Benchmark")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _add_regime_shading(ax1, signals, alpha=0.08)
    _style_ax(ax1, ylabel="Equity (log scale)")
    ax1.legend(loc="upper left", fontsize=9)

    # Bottom: drawdown
    for curve, color, label in [(eq, "blue", "Regime"), (beq, "gray", "60/40")]:
        running_max = curve.cummax()
        dd = (curve - running_max) / running_max
        ax2.fill_between(dd.index, 0, dd.values, color=color, alpha=0.3, label=label)
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.6, alpha=0.7)

    _style_ax(ax2, ylabel="Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 6: Recent 6-month deep dive
# ═══════════════════════════════════════════════════════════════════════════


def chart_recent_6mo(
    signals: pd.DataFrame,
    result,
    strat_ret_df: pd.DataFrame,
    output_path: Path,
):
    last_date = signals.index.max()
    start = last_date - pd.DateOffset(months=6)

    sig_recent = signals.loc[signals.index >= start]
    if sig_recent.empty:
        print("  WARNING: No recent signals for 6mo chart")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"Recent 6 Months ({start.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')})",
                 fontsize=13, fontweight="bold")

    # Top: SPY price with regime
    spy_recent = strat_ret_df.loc[strat_ret_df.index >= start, "SPY"]
    spy_price = _reconstruct_price(spy_recent)
    ax1.plot(spy_price.index, spy_price.values, color="black", linewidth=1.0, label="SPY")
    _add_regime_shading(ax1, sig_recent, alpha=0.15)
    _style_ax(ax1, ylabel="SPY Price (rebased)")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.legend(loc="upper left", fontsize=8)

    # Middle: portfolio weights (stacked area)
    pi_cols = sorted(c for c in sig_recent.columns if c.startswith("pi_"))
    if pi_cols:
        sleeves = [c.replace("pi_", "") for c in pi_cols]
        sleeve_colors = {
            "SPY": "#1f77b4", "EFA": "#ff7f0e", "TLT": "#2ca02c",
            "GLD": "#d4af37", "IBIT": "#9467bd", "CASH": "#7f7f7f",
        }
        data = sig_recent[pi_cols].values.T
        colors = [sleeve_colors.get(s, "gray") for s in sleeves]
        ax2.stackplot(sig_recent.index, data, colors=colors, labels=sleeves, alpha=0.8)
        _style_ax(ax2, ylabel="Allocation Weight")
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.legend(loc="upper left", fontsize=7, ncol=6)

    # Bottom: p_crisis + leverage
    if "p_crisis" in sig_recent.columns:
        ax3.plot(sig_recent.index, sig_recent["p_crisis"], color="red", linewidth=1.0,
                 label="p_crisis")
        ax3.set_ylim(-0.02, 1.02)
        ax3.axhline(0.5, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        _style_ax(ax3, ylabel="p_crisis")
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    if "L" in sig_recent.columns:
        ax3b = ax3.twinx()
        ax3b.plot(sig_recent.index, sig_recent["L"], color="blue", linewidth=1.0,
                  alpha=0.6, label="Leverage")
        ax3b.set_ylabel("Leverage (L)", fontsize=9, color="blue")
        ax3b.tick_params(labelsize=7, colors="blue")
        ax3b.legend(loc="upper right", fontsize=7)

    ax3.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Critical evaluation
# ═══════════════════════════════════════════════════════════════════════════


def _evaluate_regime_distribution(signals: pd.DataFrame) -> list[str]:
    """Check regime dominance sanity across known market periods."""
    dom = _dominant_regime(signals)
    lines = [
        "1. REGIME DISTRIBUTION SANITY",
        "-" * 50,
    ]

    test_periods = [
        ("2003-2007 Bull", "2003-01-01", "2007-10-01", "G",
         "Goldilocks should dominate pre-GFC expansion"),
        ("2008-2009 GFC", "2007-10-01", "2009-03-31", "D",
         "Deflation should dominate during financial crisis"),
        ("2010-2012 Recovery", "2010-01-01", "2012-12-31", "G",
         "Goldilocks should dominate post-GFC recovery"),
        ("2013-2019 Expansion", "2013-01-01", "2019-12-31", "G",
         "Goldilocks should dominate long expansion"),
        ("2020 COVID Crash", "2020-02-01", "2020-04-30", "D",
         "Deflation should appear during pandemic crash"),
        ("2021 Reopening Bull", "2021-01-01", "2021-12-31", "G",
         "Goldilocks or Reflation should dominate"),
        ("2022 Inflation Bear", "2022-01-01", "2022-10-31", "S",
         "Stagflation should dominate rate-hike bear"),
        ("2023-2024 Recovery", "2023-01-01", "2024-12-31", "G",
         "Goldilocks should return during recovery"),
    ]

    for label, start, end, expected, rationale in test_periods:
        mask = (dom.index >= start) & (dom.index <= end)
        period_dom = dom[mask]
        if period_dom.empty:
            lines.append(f"  {label}: NO DATA")
            continue

        counts = period_dom.value_counts()
        actual_dom = counts.index[0] if len(counts) > 0 else "?"
        pct = counts.iloc[0] / len(period_dom) * 100 if len(counts) > 0 else 0

        status = "PASS" if actual_dom == expected else "MISMATCH"
        mix = ", ".join(f"{r}={counts.get(r, 0)}" for r in REGIMES if counts.get(r, 0) > 0)
        lines.append(
            f"  [{status}] {label}: dominant={actual_dom} ({pct:.0f}%), "
            f"expected={expected}"
        )
        lines.append(f"         Mix: {mix}")
        lines.append(f"         Rationale: {rationale}")
        lines.append("")

    return lines


def _evaluate_crisis_detection(signals: pd.DataFrame) -> list[str]:
    """Evaluate crisis detection accuracy during known crises."""
    lines = [
        "2. CRISIS DETECTION ACCURACY",
        "-" * 50,
    ]

    for start, end, label in CRISIS_WINDOWS:
        mask = (signals.index >= start) & (signals.index <= end)
        crisis_sig = signals.loc[mask]
        if crisis_sig.empty:
            lines.append(f"  {label}: NO DATA")
            continue

        if "p_crisis" in crisis_sig.columns:
            avg_pc = crisis_sig["p_crisis"].mean()
            max_pc = crisis_sig["p_crisis"].max()
            pct_above_50 = (crisis_sig["p_crisis"] > 0.5).mean() * 100
            pct_above_30 = (crisis_sig["p_crisis"] > 0.3).mean() * 100
        else:
            avg_pc = max_pc = pct_above_50 = pct_above_30 = 0

        dom = _dominant_regime(crisis_sig)
        counts = dom.value_counts()
        mix = ", ".join(f"{r}={counts.get(r, 0)}" for r in REGIMES if counts.get(r, 0) > 0)

        # Evaluate: crisis should have elevated p_crisis and D/S regime
        crisis_regime_pct = (counts.get("D", 0) + counts.get("S", 0)) / len(dom) * 100

        lines.append(f"  {label} ({start} to {end}):")
        lines.append(f"    p_crisis: avg={avg_pc:.3f}, max={max_pc:.3f}")
        lines.append(f"    p_crisis > 0.3: {pct_above_30:.0f}% of weeks")
        lines.append(f"    p_crisis > 0.5: {pct_above_50:.0f}% of weeks")
        lines.append(f"    Regime mix: {mix}")
        lines.append(f"    Distress regime (D+S) coverage: {crisis_regime_pct:.0f}%")

        if avg_pc > 0.3 and crisis_regime_pct > 50:
            lines.append(f"    Assessment: GOOD — crisis detected")
        elif avg_pc > 0.2 or crisis_regime_pct > 30:
            lines.append(f"    Assessment: PARTIAL — some detection")
        else:
            lines.append(f"    Assessment: MISSED — crisis not adequately flagged")
        lines.append("")

    return lines


def _evaluate_recent_market(signals: pd.DataFrame) -> list[str]:
    """Assess current regime and p_crisis for recent market conditions."""
    lines = [
        "3. RECENT MARKET ASSESSMENT",
        "-" * 50,
    ]

    last_date = signals.index.max()
    recent_3m = signals.loc[signals.index >= (last_date - pd.DateOffset(months=3))]
    recent_1m = signals.loc[signals.index >= (last_date - pd.DateOffset(months=1))]

    for period_name, period_signals in [("Last 3 months", recent_3m), ("Last 1 month", recent_1m)]:
        if period_signals.empty:
            lines.append(f"  {period_name}: NO DATA")
            continue

        dom = _dominant_regime(period_signals)
        counts = dom.value_counts()
        dominant = counts.index[0] if len(counts) > 0 else "?"
        mix = ", ".join(f"{r}={counts.get(r, 0)}" for r in REGIMES if counts.get(r, 0) > 0)

        avg_pc = period_signals["p_crisis"].mean() if "p_crisis" in period_signals.columns else 0
        avg_L = period_signals["L"].mean() if "L" in period_signals.columns else 0

        lines.append(f"  {period_name} (ending {last_date.strftime('%Y-%m-%d')}):")
        lines.append(f"    Dominant regime: {dominant} ({REGIME_NAMES.get(dominant, '?')})")
        lines.append(f"    Regime mix: {mix}")
        lines.append(f"    Avg p_crisis: {avg_pc:.3f}")
        lines.append(f"    Avg leverage: {avg_L:.3f}")

        # Latest signal
        latest = period_signals.iloc[-1]
        probs = {r: latest.get(f"P_{r}", 0) for r in REGIMES}
        lines.append(f"    Latest posteriors: " + ", ".join(f"{r}={v:.3f}" for r, v in probs.items()))
        if "p_crisis" in latest.index:
            lines.append(f"    Latest p_crisis: {latest['p_crisis']:.3f}")
        lines.append("")

    lines.append(
        "  Context: Recent tariff escalation and market volatility should show\n"
        "  elevated Stagflation/Deflation probabilities or rising p_crisis.\n"
        "  If Goldilocks dominates despite known macro stress, the model may\n"
        "  be slow to react or lacking recent-data sensitivity."
    )
    lines.append("")

    return lines


def _evaluate_predictive_value(result, benchmark) -> list[str]:
    """Compare regime-governed portfolio to 60/40 per-year."""
    lines = [
        "4. PREDICTIVE VALUE: REGIME vs 60/40",
        "-" * 50,
    ]

    r_daily = result.daily_returns
    b_daily = benchmark.daily_returns
    rm = result.metrics
    bm = benchmark.metrics

    lines.append(f"  Overall:")
    lines.append(f"    Regime:  CAGR={rm.cagr:.2%}, Sharpe={rm.sharpe:.3f}, MaxDD={rm.max_drawdown_pct:.1%}")
    lines.append(f"    60/40:   CAGR={bm.cagr:.2%}, Sharpe={bm.sharpe:.3f}, MaxDD={bm.max_drawdown_pct:.1%}")
    lines.append(f"    Alpha:   CAGR={rm.cagr - bm.cagr:+.2%}, Sharpe={rm.sharpe - bm.sharpe:+.3f}, "
                 f"DD reduction={bm.max_drawdown_pct - rm.max_drawdown_pct:+.1%}")
    lines.append("")

    # Per-year alpha
    years = sorted(r_daily.index.year.unique())
    lines.append(f"  {'Year':<6s} {'Regime':>8s} {'60/40':>8s} {'Alpha':>8s} {'Winner':>8s}")
    lines.append(f"  {'─' * 38}")
    n_regime_wins = 0
    n_total = 0
    for year in years:
        yr_r = float((1 + r_daily[r_daily.index.year == year]).prod() - 1)
        yr_b = float((1 + b_daily[b_daily.index.year == year]).prod() - 1)
        alpha = yr_r - yr_b
        winner = "Regime" if yr_r >= yr_b else "60/40"
        if yr_r >= yr_b:
            n_regime_wins += 1
        n_total += 1
        lines.append(f"  {year:<6d} {yr_r:>+7.1%} {yr_b:>+7.1%} {alpha:>+7.1%} {winner:>8s}")

    lines.append("")
    lines.append(f"  Win rate: {n_regime_wins}/{n_total} years ({n_regime_wins/n_total*100:.0f}%)")

    # Crisis drawdown reduction
    lines.append("")
    lines.append("  Crisis drawdown comparison:")
    eq = result.equity_curve
    beq = benchmark.equity_curve
    for start, end, label in CRISIS_WINDOWS:
        eq_period = eq.loc[(eq.index >= start) & (eq.index <= end)]
        beq_period = beq.loc[(beq.index >= start) & (beq.index <= end)]
        if eq_period.empty or beq_period.empty:
            continue
        r_dd = float((eq_period / eq_period.cummax() - 1).min())
        b_dd = float((beq_period / beq_period.cummax() - 1).min())
        lines.append(f"    {label}: Regime={r_dd:.1%}, 60/40={b_dd:.1%}, "
                     f"reduction={b_dd - r_dd:+.1%}")

    # Information ratio
    common = r_daily.index.intersection(b_daily.index)
    if len(common) > 20:
        tracking_diff = r_daily.reindex(common) - b_daily.reindex(common)
        te = float(tracking_diff.std() * np.sqrt(252))
        ir = float(tracking_diff.mean() * 252 / te) if te > 1e-10 else 0
        lines.append(f"\n  Information ratio: {ir:+.3f}")
        lines.append(f"  Tracking error (ann): {te:.3f}")

    lines.append("")
    return lines


def _suggest_improvements(signals, result, benchmark, regime_stats) -> list[str]:
    """Identify high-value improvements based on weaknesses."""
    lines = [
        "5. HIGH-VALUE IMPROVEMENT SUGGESTIONS",
        "-" * 50,
    ]

    rm = result.metrics
    bm = benchmark.metrics
    dom = _dominant_regime(signals)

    suggestions = []

    # 1. Check if crisis detection is too slow
    if "p_crisis" in signals.columns:
        # Check COVID response delay
        covid_start = pd.Timestamp("2020-02-20")  # market peak
        covid_mask = (signals.index >= "2020-02-01") & (signals.index <= "2020-04-30")
        covid_sig = signals.loc[covid_mask]
        if not covid_sig.empty:
            first_high = covid_sig[covid_sig["p_crisis"] > 0.5]
            if not first_high.empty:
                delay = (first_high.index[0] - covid_start).days
                if delay > 14:
                    suggestions.append(
                        f"CRISIS LAG: p_crisis took {delay} days to exceed 0.5 after COVID peak. "
                        "Consider faster-reacting crisis inputs (daily VIX change, credit spreads)."
                    )

    # 2. Regime stability
    tr = regime_stats.get("transition_rate", 0)
    if tr > 0.03:
        suggestions.append(
            f"REGIME CHURN: Transition rate={tr:.3f}/week is high. "
            "Consider higher sticky_diag or longer lookback to reduce whipsaws."
        )

    # 3. DD duration
    if rm.max_drawdown_duration > 400:
        suggestions.append(
            f"SLOW RECOVERY: Max DD duration={rm.max_drawdown_duration} days. "
            "Consider momentum overlay or faster rebalancing during drawdowns."
        )

    # 4. Check per-year losses
    r_daily = result.daily_returns
    years = sorted(r_daily.index.year.unique())
    losing_years = []
    for year in years:
        yr_ret = float((1 + r_daily[r_daily.index.year == year]).prod() - 1)
        if yr_ret < -0.02:
            losing_years.append((year, yr_ret))
    if losing_years:
        yr_str = ", ".join(f"{y} ({r:+.1%})" for y, r in losing_years)
        suggestions.append(
            f"LOSING YEARS: {yr_str}. Investigate regime assignments during these "
            "periods — may need more defensive allocation in transition periods."
        )

    # 5. CAGR vs benchmark gap
    if rm.cagr < bm.cagr:
        suggestions.append(
            "UNDERPERFORMANCE: Regime CAGR < 60/40 CAGR. The regime switching may be "
            "destroying value through turnover costs or poor timing. Consider reducing "
            "rebalance frequency or widening regime thresholds."
        )
    elif rm.cagr - bm.cagr < 0.01:
        suggestions.append(
            f"THIN ALPHA: Regime CAGR exceeds 60/40 by only {rm.cagr - bm.cagr:.2%}. "
            "The added complexity may not justify the marginal improvement."
        )

    # 6. Turnover
    if rm.avg_annual_turnover > 15:
        suggestions.append(
            f"HIGH TURNOVER: {rm.avg_annual_turnover:.1f}x annual turnover. "
            "Consider wider rebalance bands or slower allocation transitions."
        )

    # 7. Check if all regimes are actually useful
    dom_dist = regime_stats.get("dominant_dist", {})
    for r in REGIMES:
        if dom_dist.get(r, 0) < 0.03:
            suggestions.append(
                f"RARE REGIME: {REGIME_NAMES[r]} ({r}) appears only {dom_dist.get(r, 0):.1%} "
                "of the time. Consider merging with adjacent regime or recalibrating priors."
            )

    # 8. Walk-forward not available
    suggestions.append(
        "OVERFITTING RISK: No walk-forward validation available. Run time-series "
        "cross-validation to assess OOS performance degradation."
    )

    if not suggestions:
        suggestions.append("No critical weaknesses identified.")

    for i, s in enumerate(suggestions, 1):
        wrapped = textwrap.fill(s, width=80, initial_indent=f"  {i}. ", subsequent_indent="     ")
        lines.append(wrapped)
        lines.append("")

    return lines


def generate_critical_evaluation(
    signals, result, benchmark, score, regime_stats,
) -> str:
    """Generate the full critical evaluation text."""
    header = [
        "=" * 70,
        "  CRITICAL EVALUATION OF OPTIMIZED REGIME PREDICTOR",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"  Composite Score: {score.total:.4f}",
        f"  Sharpe: {result.metrics.sharpe:.3f}  |  Sortino: {result.metrics.sortino:.3f}",
        f"  CAGR: {result.metrics.cagr:.2%}  |  Max DD: {result.metrics.max_drawdown_pct:.1%}",
        f"  Calmar: {result.metrics.calmar:.3f}  |  Turnover: {result.metrics.avg_annual_turnover:.1f}x",
        "",
    ]

    sections = [
        _evaluate_regime_distribution(signals),
        _evaluate_crisis_detection(signals),
        _evaluate_recent_market(signals),
        _evaluate_predictive_value(result, benchmark),
        _suggest_improvements(signals, result, benchmark, regime_stats),
    ]

    parts = ["\n".join(header)]
    for s in sections:
        parts.append("\n".join(s))

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  Regime Predictor: Full Evaluation & Visual Diagnostics")
    print("=" * 60)
    print()

    # Run pipeline
    signals, result, benchmark, score, regime_stats, sim_cfg, market_df, strat_ret_df, hmm_state_means = (
        _run_pipeline()
    )

    # 1. Generate text diagnostics
    print("\nGenerating diagnostics report ...")
    report = generate_regime_diagnostics_report(
        signals, result, benchmark, score, sim_cfg,
        hmm_state_means=hmm_state_means,
    )
    diag_path = OUTPUT_DIR / "full_diagnostics_optimal.txt"
    diag_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {diag_path}")

    # 2. Generate charts
    print("\nGenerating charts ...")
    chart_asset_regime(strat_ret_df, signals, market_df, "SPY",
                       OUTPUT_DIR / "chart_spy_regime.png")
    chart_asset_regime(strat_ret_df, signals, market_df, "GLD",
                       OUTPUT_DIR / "chart_gld_regime.png")
    chart_regime_posteriors(signals, OUTPUT_DIR / "chart_regime_posteriors.png")
    chart_crisis_detection(signals, market_df, OUTPUT_DIR / "chart_crisis_detection.png")
    chart_portfolio_performance(result, benchmark, signals,
                                OUTPUT_DIR / "chart_portfolio_performance.png")
    chart_recent_6mo(signals, result, strat_ret_df,
                     OUTPUT_DIR / "chart_recent_6mo.png")

    # 3. Generate critical evaluation
    print("\nGenerating critical evaluation ...")
    eval_text = generate_critical_evaluation(signals, result, benchmark, score, regime_stats)
    eval_path = OUTPUT_DIR / "critical_evaluation.txt"
    eval_path.write_text(eval_text, encoding="utf-8")
    print(f"  Saved: {eval_path}")

    # 4. Summary
    m = result.metrics
    bm = benchmark.metrics
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Regime Portfolio:  CAGR={m.cagr:.2%}  Sharpe={m.sharpe:.3f}  MaxDD={m.max_drawdown_pct:.1%}")
    print(f"  60/40 Benchmark:   CAGR={bm.cagr:.2%}  Sharpe={bm.sharpe:.3f}  MaxDD={bm.max_drawdown_pct:.1%}")
    print(f"  Alpha:             CAGR={m.cagr - bm.cagr:+.2%}  Sharpe={m.sharpe - bm.sharpe:+.3f}")
    print(f"  Composite Score:   {score.total:.4f}")
    print(f"\n  Files saved to: {OUTPUT_DIR}/")
    print(f"    - full_diagnostics_optimal.txt")
    print(f"    - critical_evaluation.txt")
    print(f"    - chart_spy_regime.png")
    print(f"    - chart_gld_regime.png")
    print(f"    - chart_regime_posteriors.png")
    print(f"    - chart_crisis_detection.png")
    print(f"    - chart_portfolio_performance.png")
    print(f"    - chart_recent_6mo.png")
    print()


if __name__ == "__main__":
    main()
