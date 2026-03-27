"""Run P9 quick exit / MFE trailing / OR quality gate greedy optimization for ALCB T2.

Tests 19 candidates across 3 categories via forward selection:
  A. Two-stage quick exit (8 candidates — fix -113.65R QUICK_EXIT drag)
  B. MFE-triggered trailing stop (7 candidates — fix 31.5% MFE capture)
  C. OR_BREAKOUT quality gate (4 candidates — filter weak OR entries)

Base = P8 optimal config minus sector_mult_consumer_disc (overfitted).

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    PYTHONUNBUFFERED=1 python -u -m research.backtests.stock.auto.output.run_greedy_alcb_p9
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
    ALCB_T2_P9_BASE_MUTATIONS,
    ALCB_T2_P9_CANDIDATES,
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
    print("=" * 60)
    print("ALCB T2 Phase 9 Quick Exit / MFE Trailing / OR Gate Greedy")
    print("=" * 60, flush=True)

    print(f"\nCandidate pool: {len(ALCB_T2_P9_CANDIDATES)} candidates")
    print(f"Base mutations: {len(ALCB_T2_P9_BASE_MUTATIONS)} params (P8 optimal - sector_mult)\n")

    print("Loading data...", flush=True)
    t0 = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t0:.1f}s\n", flush=True)

    n_workers = max(1, (os.cpu_count() or 4) // 4)
    print(f"Using {n_workers} parallel workers\n", flush=True)

    result = run_greedy(
        replay,
        strategy="alcb",
        tier=2,
        base_mutations=ALCB_T2_P9_BASE_MUTATIONS,
        candidates=ALCB_T2_P9_CANDIDATES,
        max_workers=n_workers,
    )

    # Save greedy result JSON
    save_result(result, OUTPUT_DIR / "greedy_optimal_alcb_p9.json")

    # Save optimal_config.json
    optimal = resolve_config(result.final_mutations)
    config_path = OUTPUT_DIR / "optimal_config_alcb_p9.json"
    config_path.write_text(
        json.dumps(optimal, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {config_path}")

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s ({total / 60:.1f}min)")


if __name__ == "__main__":
    main()
