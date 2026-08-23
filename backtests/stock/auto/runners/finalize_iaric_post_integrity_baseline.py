"""Freeze full diagnostics for the selected post-integrity IARIC reference."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path

from backtests.shared.diagnostics.snapshot import build_group_snapshot
from backtests.stock.analysis.iaric_pullback_diagnostics import (
    _trade_stats,
    pullback_full_diagnostic,
)
from backtests.stock.auto.config_mutator import mutate_iaric_config
from backtests.stock.auto.iaric.phase_scoring import (
    enrich_phase_score_metrics,
    merge_pullback_metrics,
)
from backtests.stock.auto.scoring import extract_metrics
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    _code_fingerprint,
    _replay_source_fingerprint,
)
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
from backtests.stock.engine.research_replay import ResearchReplayEngine


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIR = REPO_ROOT / "backtests/output/stock/iaric/baseline_establishment"
HOLDOUT_START = "2026-03-02"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", default=str(DEFAULT_DIR / "post_integrity_baseline_ranking.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.end >= HOLDOUT_START:
        raise ValueError(f"End date overlaps sealed holdout beginning {HOLDOUT_START}")
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = Path(args.output_dir).resolve()
    ranking = json.loads(Path(args.ranking).resolve().read_text(encoding="utf-8"))
    selected = ranking[0]
    mutations = dict(selected["mutations"])
    data_dir = REPO_ROOT / "backtests/stock/data/raw"
    replay = ResearchReplayEngine(data_dir=data_dir, require_bundle=not args.allow_legacy_data)
    replay.load_all_data()
    base = IARICBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=10_000.0,
        tier=3,
        data_dir=data_dir,
    )
    result = IARICPullbackEngine(
        mutate_iaric_config(base, mutations),
        replay,
        collect_diagnostics=True,
    ).run()
    performance = extract_metrics(result.trades, result.equity_curve, result.timestamps, 10_000.0)
    metrics = enrich_phase_score_metrics(
        merge_pullback_metrics(
            performance,
            result.trades,
            candidate_ledger=result.candidate_ledger,
            selection_attribution=result.selection_attribution,
        )
    )
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, Real)
    }
    diagnostic = pullback_full_diagnostic(
        result.trades,
        replay=replay,
        daily_selections=result.daily_selections,
        candidate_ledger=result.candidate_ledger,
        funnel_counters=result.funnel_counters,
        rejection_log=result.rejection_log,
        shadow_outcomes=result.shadow_outcomes,
        selection_attribution=result.selection_attribution,
        fsm_log=result.fsm_log,
    )
    snapshot = build_group_snapshot(
        "IARIC Post-Integrity Strength / Weakness Snapshot",
        result.trades,
        [
            ("symbol", lambda trade: getattr(trade, "symbol", None)),
            ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
        ],
        min_count=5,
    )
    eligible = bool(selected.get("full_period_eligible"))
    header = "\n".join(
        [
            "=" * 72,
            "  IARIC POST-INTEGRITY BASELINE DIAGNOSTICS",
            "=" * 72,
            f"  Candidate:       {selected['id']}",
            f"  Date range:      {args.start} -- {args.end}",
            "  Holdout accessed: False",
            "  Data authority:  legacy_diagnostic_only",
            f"  Promotion gate:  {'PASS' if eligible else 'FAIL'}",
            "  Status:          honest_reference_only" if not eligible else "  Status:          promotion_candidate",
            "  Integrity repairs: causal price-basis alignment; shared session ATR; shared V2 carry control",
            "  Execution:       completed 5m signal -> next 5m open; partial profit disabled; intraday only",
            "",
            "  Mutations:",
            *[f"    {key}: {value}" for key, value in sorted(mutations.items())],
        ]
    )
    report = header + "\n\n" + snapshot + "\n\n" + diagnostic
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "post_integrity_selected_config.json").write_text(
        json.dumps(mutations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "post_integrity_final_diagnostics.txt").write_text(report, encoding="utf-8")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "iaric",
        "candidate_id": selected["id"],
        "status": "promotion_candidate" if eligible else "honest_reference_only",
        "promotion_gate_passed": eligible,
        "date_range": {"start": args.start, "end": args.end},
        "holdout_start": HOLDOUT_START,
        "holdout_accessed": False,
        "data_authority": "legacy_diagnostic_only",
        "source_fingerprint": _replay_source_fingerprint(),
        "code_fingerprint": _code_fingerprint(),
        "mutations": mutations,
        "metrics": numeric_metrics,
        "trade_stats": _trade_stats(result.trades),
        "price_basis_adjustments": replay.price_basis_adjustments,
        "files": {
            "config": str((output_dir / "post_integrity_selected_config.json").resolve()),
            "diagnostics": str((output_dir / "post_integrity_final_diagnostics.txt").resolve()),
        },
    }
    (output_dir / "post_integrity_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_id": selected["id"],
                "trades": numeric_metrics.get("total_trades"),
                "total_r": numeric_metrics.get("expected_total_r"),
                "avg_r": numeric_metrics.get("avg_r"),
                "profit_factor": numeric_metrics.get("profit_factor"),
                "sharpe": numeric_metrics.get("sharpe"),
                "max_drawdown_pct": numeric_metrics.get("max_drawdown_pct"),
                "promotion_gate_passed": eligible,
                "adjusted_symbols": sorted(replay.price_basis_adjustments),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
