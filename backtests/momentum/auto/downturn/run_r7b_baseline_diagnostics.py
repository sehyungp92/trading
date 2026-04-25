from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backtests.diagnostic_snapshot import build_group_snapshot
from backtests.momentum._aliases import install

install()

from backtest.analysis.downturn_diagnostics import compute_downturn_metrics
from backtest.config_downturn import DownturnBacktestConfig
from backtest.engine.downturn_engine import DownturnEngine
from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
from backtests.momentum.auto.downturn.output.generate_r5_diagnostics import generate_expanded_report
from backtests.momentum.auto.downturn.output.generate_r7b_diagnostics import R7B_MUTATIONS
from backtests.momentum.cli import _load_downturn_data


DEFAULT_OUTPUT = Path("backtests/momentum/auto/downturn/output/r7b_baseline_full_diagnostics.txt")
DEFAULT_JSON = Path("backtests/momentum/auto/downturn/output/r7b_baseline_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    data_dir = Path("backtests/momentum/data/raw")
    data = _load_downturn_data("NQ", data_dir)
    base_cfg = DownturnBacktestConfig(initial_equity=10_000.0, max_notional_leverage=20.0)
    config = mutate_downturn_config(base_cfg, R7B_MUTATIONS)

    print(f"Running R7b baseline with {len(R7B_MUTATIONS)} mutations...")
    engine = DownturnEngine("NQ", config)
    result = engine.run(**data)
    metrics = compute_downturn_metrics(result, data["daily"])

    snapshot = build_group_snapshot(
        "Downturn Strength / Weakness Snapshot",
        result.trades,
        [
            ("engine tag", lambda trade: getattr(getattr(trade, "engine_tag", None), "value", None)),
            (
                "regime",
                lambda trade: getattr(getattr(trade, "composite_regime_at_entry", None), "value", None)
                or getattr(getattr(trade, "composite_regime_at_entry", None), "name", None),
            ),
            ("exit type", lambda trade: getattr(trade, "exit_type", None)),
        ],
        min_count=3,
    )
    full_report = snapshot + "\n\n" + generate_expanded_report(result, data, config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_report, encoding="utf-8")

    summary = {
        "baseline": "r7b_conservative",
        "mutations": R7B_MUTATIONS,
        "metrics": {
            "total_trades": metrics.total_trades,
            "profit_factor": metrics.profit_factor,
            "net_return_pct": metrics.net_return_pct,
            "max_dd_pct": metrics.max_dd_pct,
            "calmar": metrics.calmar,
            "sharpe": metrics.sharpe,
            "correction_pnl_pct": metrics.correction_pnl_pct,
            "correction_coverage": metrics.correction_coverage,
            "exit_efficiency": metrics.exit_efficiency,
            "bear_capture_ratio": metrics.bear_capture_ratio,
        },
    }
    json_path = Path(args.json_output)
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Saved diagnostics to {output_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
