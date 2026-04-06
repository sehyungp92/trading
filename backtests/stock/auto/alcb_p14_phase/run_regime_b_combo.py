"""Test regime_mult_b=0.7 + block_combined_regime_b=True.

Usage::
    python -m research.backtests.stock.auto.alcb_p14_phase.run_regime_b_combo
"""
from __future__ import annotations

import io
import sys
import time as _time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from research.backtests.stock._aliases import install

install()

from research.backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
from research.backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
from research.backtests.stock.auto.config_mutator import mutate_alcb_config
from research.backtests.stock.config_alcb import ALCBBacktestConfig
from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from research.backtests.stock.engine.research_replay import ResearchReplayEngine

from .run_p14_final_diagnostics import P14_FINAL_PLUS
from .time_utils import hydrate_time_mutations

VARIANTS: list[tuple[str, dict]] = [
    ("baseline_0.5", {"param_overrides.regime_mult_b": 0.5}),
    ("block_combined_0.5", {
        "param_overrides.regime_mult_b": 0.5,
        "param_overrides.block_combined_regime_b": True,
    }),
    ("regime_b_0.7_block_combined", {
        "param_overrides.regime_mult_b": 0.7,
        "param_overrides.block_combined_regime_b": True,
    }),
]


def _extract_metrics(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "wr": "0.0%", "mean_r": 0, "total_r": 0, "pnl": 0,
                "pf": 0, "tier_b_trades": 0, "tier_b_r": 0, "tier_b_pnl": 0,
                "tier_b_combined": 0, "tier_b_combined_r": 0, "tier_b_or": 0,
                "tier_b_or_r": 0, "tier_b_pdh": 0, "tier_b_pdh_r": 0}

    wins = [t for t in trades if t.r_multiple > 0]
    losses = [t for t in trades if t.r_multiple <= 0]
    total_r = sum(t.r_multiple for t in trades)
    gross_win = sum(t.r_multiple for t in wins)
    gross_loss = abs(sum(t.r_multiple for t in losses))
    pnl = sum(t.pnl_net for t in trades)

    tier_b = [t for t in trades if getattr(t, "regime_tier", "") == "B"]
    tier_b_r = sum(t.r_multiple for t in tier_b)
    tier_b_pnl = sum(t.pnl_net for t in tier_b)

    tier_b_combined = [t for t in tier_b if getattr(t, "entry_type", "") == "COMBINED_BREAKOUT"]
    tier_b_or = [t for t in tier_b if getattr(t, "entry_type", "") == "OR_BREAKOUT"]
    tier_b_pdh = [t for t in tier_b if getattr(t, "entry_type", "") == "PDH_BREAKOUT"]

    return {
        "trades": len(trades),
        "wr": f"{len(wins)/len(trades)*100:.1f}%",
        "mean_r": round(total_r / len(trades), 4),
        "total_r": round(total_r, 2),
        "pnl": round(pnl, 2),
        "pf": round(gross_win / max(gross_loss, 0.01), 2),
        "tier_b_trades": len(tier_b),
        "tier_b_r": round(tier_b_r, 2),
        "tier_b_pnl": round(tier_b_pnl, 2),
        "tier_b_combined": len(tier_b_combined),
        "tier_b_combined_r": round(sum(t.r_multiple for t in tier_b_combined), 2),
        "tier_b_or": len(tier_b_or),
        "tier_b_or_r": round(sum(t.r_multiple for t in tier_b_or), 2),
        "tier_b_pdh": len(tier_b_pdh),
        "tier_b_pdh_r": round(sum(t.r_multiple for t in tier_b_pdh), 2),
    }


def main() -> None:
    data_dir = Path("research/backtests/stock/data/raw")
    output_dir = Path("research/backtests/stock/auto/alcb_p14_phase/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading replay data...")
    replay = ResearchReplayEngine(data_dir=data_dir)
    replay.load_all_data()

    base_cfg = ALCBBacktestConfig(
        start_date="2024-01-01",
        end_date="2026-03-01",
        initial_equity=10_000.0,
        tier=2,
        data_dir=data_dir,
    )
    base_muts = hydrate_time_mutations(dict(P14_FINAL_PLUS))
    results: list[tuple[str, dict]] = []

    for name, variant_muts in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Running variant: {name}")
        print(f"{'='*60}")

        muts = {**base_muts, **variant_muts}
        cfg = mutate_alcb_config(base_cfg, muts)

        t0 = _time.time()
        engine = ALCBIntradayEngine(cfg, replay)
        shadow = ALCBShadowTracker()
        engine.shadow_tracker = shadow
        result = engine.run()
        elapsed = _time.time() - t0

        metrics = _extract_metrics(result.trades)
        results.append((name, metrics))
        print(f"  Completed in {elapsed:.0f}s -- {metrics['trades']} trades")

        # Save full diagnostics for the combo variant
        if "0.7" in name:
            report = alcb_full_diagnostic(
                result.trades,
                shadow_tracker=shadow,
                daily_selections=result.daily_selections,
            )
            out_path = output_dir / "regime_b_0.7_block_combined_diagnostics.txt"
            out_path.write_text(report, encoding="utf-8")
            print(f"  Diagnostics saved to {out_path}")

    print(f"\n\n{'='*80}")
    print("REGIME_MULT_B COMBO COMPARISON")
    print(f"{'='*80}\n")

    header = f"{'Variant':<35} {'Trades':>6} {'WR':>7} {'MeanR':>8} {'TotalR':>8} {'PnL':>10} {'PF':>6}"
    print(header)
    print("-" * len(header))
    for name, m in results:
        print(f"{name:<35} {m['trades']:>6} {m['wr']:>7} {m['mean_r']:>8} {m['total_r']:>8} {m['pnl']:>10} {m['pf']:>6}")

    print(f"\n{'Variant':<35} {'B_Trades':>8} {'B_R':>8} {'B_PnL':>10} {'B_CMB':>6} {'B_CMB_R':>8} {'B_OR':>5} {'B_OR_R':>7} {'B_PDH':>6} {'B_PDH_R':>8}")
    print("-" * 110)
    for name, m in results:
        print(
            f"{name:<35} {m['tier_b_trades']:>8} {m['tier_b_r']:>8} {m['tier_b_pnl']:>10} "
            f"{m['tier_b_combined']:>6} {m['tier_b_combined_r']:>8} "
            f"{m['tier_b_or']:>5} {m['tier_b_or_r']:>7} "
            f"{m['tier_b_pdh']:>6} {m['tier_b_pdh_r']:>8}"
        )


if __name__ == "__main__":
    main()
