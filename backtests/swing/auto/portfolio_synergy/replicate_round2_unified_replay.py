from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .plugin import PortfolioSynergyPlugin


DEFAULT_METRIC_KEYS: tuple[str, ...] = (
    "final_equity",
    "net_pnl",
    "net_return_pct",
    "total_trades",
    "entry_signals_fired",
    "entries_accepted_by_portfolio",
    "entries_blocked_by_portfolio",
    "active_trades_per_month",
    "total_r_per_month",
    "profit_factor",
    "win_rate",
    "max_drawdown_pct",
    "sharpe",
    "sortino",
    "calmar",
    "active_strategy_count",
    "trade_capture_ratio",
    "score_total",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replicate swing round 2 through the source-fingerprinted unified replay bundle.",
    )
    parser.add_argument("--round-dir", default="backtests/output/swing/portfolio_synergy/round_2")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--equity", type=float, default=25_000.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--abs-tolerance", type=float, default=1e-6)
    parser.add_argument("--rel-tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)

    report = replicate_round2_unified_replay(
        round_dir=Path(args.round_dir),
        data_dir=Path(args.data_dir),
        initial_equity=float(args.equity),
        max_workers=int(args.max_workers),
        abs_tolerance=float(args.abs_tolerance),
        rel_tolerance=float(args.rel_tolerance),
    )
    print(f"Round 2 unified replay matched: {report['matched']}")
    print(f"Architecture: {report['replay_architecture']}")
    print(f"Replay fingerprint: {report['replay_source_fingerprint']}")
    print(f"Wrote {report['output_path']}")


def replicate_round2_unified_replay(
    *,
    round_dir: Path,
    data_dir: Path,
    initial_equity: float = 25_000.0,
    max_workers: int = 2,
    abs_tolerance: float = 1e-6,
    rel_tolerance: float = 1e-9,
) -> dict[str, Any]:
    mutations = json.loads((round_dir / "optimized_config.json").read_text(encoding="utf-8"))
    expected_summary = json.loads((round_dir / "run_summary.json").read_text(encoding="utf-8"))
    expected_metrics = expected_summary["final_metrics"]

    plugin = PortfolioSynergyPlugin(
        data_dir=data_dir,
        initial_equity=initial_equity,
        max_workers=max_workers,
    )
    actual_metrics = plugin.compute_final_metrics(mutations)
    replay_bundle = plugin._ensure_bundle()
    comparisons = {
        key: _compare_metric(
            expected_metrics.get(key),
            actual_metrics.get(key),
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        for key in DEFAULT_METRIC_KEYS
    }
    matched = all(item["matched"] for item in comparisons.values())
    report = {
        "matched": matched,
        "round_dir": str(round_dir),
        "replay_architecture": "unified_swing_engine_source_fingerprinted_replay_bundle",
        "replay_source_fingerprint": replay_bundle.cache_source_fingerprint,
        "cache_key": replay_bundle.cache_key,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "comparisons": comparisons,
        "metrics": actual_metrics,
    }
    output_path = round_dir / "unified_replay_replication.json"
    output_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def _compare_metric(
    expected: object,
    actual: object,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> dict[str, Any]:
    expected_float = float(expected or 0.0)
    actual_float = float(actual or 0.0)
    abs_diff = abs(actual_float - expected_float)
    rel_diff = abs_diff / max(abs(expected_float), abs(actual_float), 1.0)
    return {
        "expected": expected_float,
        "actual": actual_float,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "matched": abs_diff <= abs_tolerance or rel_diff <= rel_tolerance,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
