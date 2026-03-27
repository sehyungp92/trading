"""Quick baseline test — run ALCB Tier 1 + Tier 2 and time each step."""
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

def main():
    print("=" * 60, flush=True)
    print("BASELINE TEST: ALCB Tier 1 + Tier 2", flush=True)
    print("=" * 60, flush=True)

    # 1. Load data
    print("\n[1/3] Loading data...", flush=True)
    t0 = time.time()

    from pathlib import Path
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path("research/backtests/stock/data/raw")
    replay = ResearchReplayEngine(data_dir)
    replay.load_all_data()

    print(f"  Data loaded in {time.time() - t0:.1f}s", flush=True)

    # 2. Run ALCB Tier 1 (daily) baseline
    print("\n[2/3] Running ALCB Tier 1 (daily) baseline...", flush=True)
    t1 = time.time()

    from research.backtests.stock.config_alcb import ALCBBacktestConfig
    from research.backtests.stock.engine.alcb_daily_engine import ALCBDailyEngine
    from research.backtests.stock.auto.scoring import composite_score, extract_metrics

    _10k_overrides = {
        "base_risk_fraction": 0.015,
        "min_adv_usd": 5_000_000.0,
        "heat_cap_r": 10.0,
        "min_containment": 0.70,
        "max_squeeze_metric": 1.30,
        "breakout_tolerance_pct": 0.10,
    }

    config_t1 = ALCBBacktestConfig(
        start_date="2024-01-01",
        end_date="2026-03-01",
        initial_equity=10_000.0,
        tier=1,
        data_dir=data_dir,
        param_overrides=_10k_overrides,
    )
    engine_t1 = ALCBDailyEngine(config_t1, replay)
    result_t1 = engine_t1.run()

    metrics_t1 = extract_metrics(result_t1.trades, result_t1.equity_curve,
                                  result_t1.timestamps, 10_000.0)
    score_t1 = composite_score(metrics_t1, 10_000.0)

    elapsed_t1 = time.time() - t1
    print(f"  Tier 1 done in {elapsed_t1:.1f}s", flush=True)
    print(f"  Trades: {metrics_t1.total_trades}", flush=True)
    print(f"  PF: {metrics_t1.profit_factor:.2f}", flush=True)
    print(f"  Max DD: {metrics_t1.max_drawdown_pct:.1%}", flush=True)
    print(f"  Net Profit: {metrics_t1.net_profit:.2f}", flush=True)
    print(f"  Score: {score_t1.total:.4f} (rejected={score_t1.rejected})", flush=True)
    if score_t1.rejected:
        print(f"  Reject reason: {score_t1.reject_reason}", flush=True)

    # 3. Run ALCB Tier 2 (intraday 30m) baseline
    print("\n[3/3] Running ALCB Tier 2 (intraday 30m) baseline...", flush=True)
    t2 = time.time()

    from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine

    config_t2 = ALCBBacktestConfig(
        start_date="2024-01-01",
        end_date="2026-03-01",
        initial_equity=10_000.0,
        tier=2,
        data_dir=data_dir,
        param_overrides=_10k_overrides,
    )
    engine_t2 = ALCBIntradayEngine(config_t2, replay)
    result_t2 = engine_t2.run()

    metrics_t2 = extract_metrics(result_t2.trades, result_t2.equity_curve,
                                  result_t2.timestamps, 10_000.0)
    score_t2 = composite_score(metrics_t2, 10_000.0)

    elapsed_t2 = time.time() - t2
    print(f"  Tier 2 done in {elapsed_t2:.1f}s", flush=True)
    print(f"  Trades: {metrics_t2.total_trades}", flush=True)
    print(f"  PF: {metrics_t2.profit_factor:.2f}", flush=True)
    print(f"  Max DD: {metrics_t2.max_drawdown_pct:.1%}", flush=True)
    print(f"  Net Profit: {metrics_t2.net_profit:.2f}", flush=True)
    print(f"  Score: {score_t2.total:.4f} (rejected={score_t2.rejected})", flush=True)
    if score_t2.rejected:
        print(f"  Reject reason: {score_t2.reject_reason}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)
    print(f"  Data load: {t1 - t0:.1f}s", flush=True)
    print(f"  Tier 1:    {elapsed_t1:.1f}s", flush=True)
    print(f"  Tier 2:    {elapsed_t2:.1f}s", flush=True)

    # Estimated pipeline time
    total_t2 = 124 * elapsed_t2
    total_t1 = 12 * elapsed_t1
    print(f"\n--- Estimated Pipeline Time ---", flush=True)
    print(f"124 T2 experiments x {elapsed_t2:.0f}s = {total_t2/60:.0f} min", flush=True)
    print(f" 12 T1 experiments x {elapsed_t1:.0f}s = {total_t1/60:.0f} min", flush=True)
    print(f"Total: ~{(total_t2 + total_t1)/60:.0f} min ({(total_t2 + total_t1)/3600:.1f} hr)", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
