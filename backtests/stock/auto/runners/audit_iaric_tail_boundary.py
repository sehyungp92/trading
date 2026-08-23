"""Audit the exact trades admitted at a suspicious IARIC score boundary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from backtests.stock.auto.config_mutator import mutate_iaric_config
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.data.replay_cache import load_research_replay_bundle
from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine


REPO_ROOT = Path(__file__).resolve().parents[4]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", default=str(REPO_ROOT / "backtests/output/stock/iaric/baseline_establishment/routes_ranking.json"))
    parser.add_argument("--candidate-id", default="routes_entry_score_35")
    parser.add_argument("--output", default=str(REPO_ROOT / "backtests/output/stock/iaric/baseline_establishment/tail_boundary_audit.json"))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def _trade_row(trade: Any) -> dict[str, Any]:
    metadata = dict(trade.metadata or {})
    return {
        "symbol": trade.symbol,
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "entry_price": float(trade.entry_price),
        "exit_price": float(trade.exit_price),
        "risk_per_share": float(trade.risk_per_share),
        "risk_pct": float(trade.risk_per_share / trade.entry_price) if trade.entry_price else 0.0,
        "r_multiple": float(trade.r_multiple),
        "pnl": float(trade.pnl),
        "exit_reason": str(trade.exit_reason),
        "route_family": str(metadata.get("route_family", "")),
        "intraday_score": float(metadata.get("intraday_score", 0.0) or 0.0),
        "daily_signal_score": float(metadata.get("daily_signal_score", 0.0) or 0.0),
        "stop_distance_pct": float(metadata.get("stop_distance_pct", 0.0) or 0.0),
        "mae_r": float(metadata.get("mae_r", 0.0) or 0.0),
        "mfe_r": float(metadata.get("mfe_r", 0.0) or 0.0),
        "entry_bar_index": int(metadata.get("entry_bar_index", -1) or -1),
    }


def main() -> None:
    args = _args()
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    ranking = json.loads(Path(args.ranking).resolve().read_text(encoding="utf-8"))
    candidate = next(row for row in ranking if row["id"] == args.candidate_id)
    data_dir = REPO_ROOT / "backtests/stock/data/raw"
    replay = load_research_replay_bundle(data_dir).data
    config = IARICBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=10_000.0,
        tier=3,
        data_dir=data_dir,
    )
    mutated = mutate_iaric_config(config, candidate["mutations"])
    result = IARICPullbackEngine(mutated, replay, collect_diagnostics=False).run()
    rows = [_trade_row(trade) for trade in result.trades]
    marginal = [row for row in rows if 35.0 <= row["intraday_score"] < 40.0]
    severe = [row for row in rows if row["r_multiple"] < -1.25]
    payload = {
        "candidate_id": args.candidate_id,
        "window": {"start": args.start, "end": args.end},
        "holdout_accessed": False,
        "trade_count": len(rows),
        "total_r": sum(row["r_multiple"] for row in rows),
        "marginal_35_to_40": sorted(marginal, key=lambda row: row["r_multiple"]),
        "severe_losses_below_minus_1_25r": sorted(severe, key=lambda row: row["r_multiple"]),
        "worst_20": sorted(rows, key=lambda row: row["r_multiple"])[:20],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "trade_count": payload["trade_count"],
        "total_r": payload["total_r"],
        "marginal_count": len(marginal),
        "marginal_total_r": sum(row["r_multiple"] for row in marginal),
        "severe_loss_count": len(severe),
        "worst": payload["worst_20"][:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
