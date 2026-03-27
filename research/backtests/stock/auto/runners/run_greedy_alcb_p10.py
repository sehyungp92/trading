"""Run P10 full re-optimization for ALCB T2 under new scoring + 2:1 leverage.

Tests ALL candidates from P1-P9 (~110+ deduplicated) via greedy forward selection,
starting from default config. The 2:1 leverage constraint is baked into StrategySettings
and the composite scoring normalization has been recalibrated (Calmar /50, Net Profit log(6)).

This finds the optimal config from scratch rather than building on prior phase bases.

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    PYTHONUNBUFFERED=1 python -u -m research.backtests.stock.auto.output.run_greedy_alcb_p10
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from research.backtests.stock.auto.greedy_optimize import (
    ALCB_T2_P10_BASE_MUTATIONS,
    ALCB_T2_P10_CANDIDATES,
    run_greedy,
    save_result,
)
from research.backtests.stock.engine.research_replay import ResearchReplayEngine

DATA_DIR = Path("research/backtests/stock/data/raw")
OUTPUT_DIR = Path("research/backtests/stock/auto/output")


def resolve_config(mutations: dict) -> dict:
    """Convert greedy mutations dict to optimal_config.json format."""
    params: dict = {}
    ablation: dict = {}
    for key, val in mutations.items():
        if key.startswith("param_overrides."):
            v = list(val) if isinstance(val, tuple) else val
            if isinstance(v, list):
                v = [list(x) if isinstance(x, tuple) else x for x in v]
            params[key.split(".", 1)[1]] = v
        elif key.startswith("ablation."):
            ablation[key.split(".", 1)[1]] = val
    result: dict = {}
    if ablation:
        result["ablation"] = dict(sorted(ablation.items()))
    result["param_overrides"] = dict(sorted(params.items()))
    return result


def main():
    print("=" * 70)
    print("ALCB T2 Phase 10 — Full Re-optimization (new scoring + 2:1 leverage)")
    print("=" * 70, flush=True)

    print(f"\nCandidate pool: {len(ALCB_T2_P10_CANDIDATES)} candidates (P1-P9 union)")
    print(f"Base mutations: {len(ALCB_T2_P10_BASE_MUTATIONS)} params (default config)\n")

    print("Loading data...", flush=True)
    t0 = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t0:.1f}s\n", flush=True)

    n_workers = max(1, (os.cpu_count() or 4) // 4)
    print(f"Using {n_workers} parallel workers\n", flush=True)

    checkpoint = OUTPUT_DIR / "greedy_checkpoint_alcb_p10.json"
    result = run_greedy(
        replay,
        strategy="alcb",
        tier=2,
        base_mutations=ALCB_T2_P10_BASE_MUTATIONS,
        candidates=ALCB_T2_P10_CANDIDATES,
        max_workers=n_workers,
        checkpoint_path=checkpoint,
    )

    # Save greedy result JSON
    save_result(result, OUTPUT_DIR / "greedy_optimal_alcb_p10.json")

    # Save optimal_config.json
    optimal = resolve_config(result.final_mutations)
    config_path = OUTPUT_DIR / "optimal_config_alcb_p10.json"
    config_path.write_text(
        json.dumps(optimal, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {config_path}")

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s ({total / 60:.1f}min)")


if __name__ == "__main__":
    main()
