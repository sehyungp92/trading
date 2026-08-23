"""Exact-only follow-up for the IARIC representative residual baseline.

The approximate v5 screen omitted the economically pre-registered market
trend veto and standalone volume-exhaustion variants from exact replay.  This
runner reuses the official exact-98 selection panel, builds one shared frozen
market/sector/peer bundle, and evaluates only those missing variants.  Locked
validation and holdout data are never loaded.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.iaric.residual_phases import (
    run_exact_fold_evaluation,
    settings_from_discovery_candidate,
)
from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.auto.runners.run_iaric_residual_baseline_diagnostics import (
    SELECTED_REPRESENTATIVE_CONTRACT_ID,
    run as run_full_diagnostics,
)
from backtests.stock.engine.iaric_daily_residual_replay import (
    build_daily_residual_replay_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/round_4/residual_exact_followup_v5"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _status(output: Path, status: str, **details: Any) -> None:
    _write_json(
        output / "background_status.json",
        {
            "status": status,
            "max_workers": 2,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _candidate(
    candidate_id: str,
    *,
    score_components: tuple[str, ...],
    market_floor: float,
    holding: int = 10,
    stop_residual_r: float = 4.0,
) -> tuple[dict[str, Any], float]:
    return (
        {
            "candidate_id": candidate_id,
            "residual_z_floor": 1.0,
            "holding_sessions": holding,
            "max_positions": 10,
            "max_positions_per_sector": 2,
            "round_trip_cost_bps": 20.0,
            "formation_sessions": 1,
            "diagnostic_leg": "long_loser",
            "factor_model": "market_sector_peer",
            "score_components": list(score_components),
            "lane_id": "fresh_residual_volume_exhaustion_1d",
            "minimum_failed_continuation_r": 0.0,
            "minimum_sector_return_5d": -0.15,
            "minimum_market_trend_z_20d": market_floor,
        },
        stop_residual_r,
    )


def registered_followups() -> list[tuple[dict[str, Any], float]]:
    """A bounded mechanism grid derived only from the failed exact audit."""

    return [
        _candidate(
            "raw_volume__market_zm1__hold10__stop4",
            score_components=("volume_transition",),
            market_floor=-1.0,
        ),
        _candidate(
            "raw_volume__market_zm1__hold10__stop6",
            score_components=("volume_transition",),
            market_floor=-1.0,
            stop_residual_r=6.0,
        ),
        _candidate(
            "raw_volume__market_zm1__hold7__stop4",
            score_components=("volume_transition",),
            market_floor=-1.0,
            holding=7,
        ),
        _candidate(
            "volume_exhaustion__no_market_veto__hold10__stop4",
            score_components=("volume_exhaustion_quality",),
            market_floor=-8.0,
        ),
        _candidate(
            "volume_exhaustion__market_zm1__hold10__stop4",
            score_components=("volume_exhaustion_quality",),
            market_floor=-1.0,
        ),
        _candidate(
            "volume_exhaustion__market_zm1__hold10__stop6",
            score_components=("volume_exhaustion_quality",),
            market_floor=-1.0,
            stop_residual_r=6.0,
        ),
        _candidate(
            "volume_exhaustion_failed__market_zm1__hold10__stop4",
            score_components=("volume_exhaustion_quality", "failed_continuation"),
            market_floor=-1.0,
        ),
        _candidate(
            "raw_volume_failed__market_zm1__hold10__stop4",
            score_components=("volume_transition", "failed_continuation"),
            market_floor=-1.0,
        ),
    ]


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"trades", "equity_curves"}
    }


def run(output: Path, data_dir: Path, *, max_workers: int = 2) -> int:
    if max_workers != 2:
        raise ValueError("exact follow-up is registered for max-workers=2")
    output.mkdir(parents=True, exist_ok=True)
    _status(output, "loading_official_exact98_panel")
    close, open_, high, low, volume, sectors, paths = discovery._load_daily_panel(
        data_dir
    )
    fingerprint, _rows = discovery._selection_data_fingerprint(
        close, open_, high, low, volume, paths
    )
    _status(output, "building_shared_frozen_market_sector_peer_bundle")
    bundle = build_daily_residual_replay_bundle(
        close,
        open_,
        high,
        low,
        volume,
        sectors,
        factor_model="market_sector_peer",
        source_fingerprint=fingerprint,
    )
    registry = registered_followups()
    _write_json(
        output / "candidate_registry.json",
        [
            {"candidate": candidate, "catastrophic_stop_residual_r": stop}
            for candidate, stop in registry
        ],
    )
    _status(output, "running_exact_followups", candidates=len(registry))

    def evaluate(row: tuple[dict[str, Any], float]) -> dict[str, Any]:
        candidate, stop = row
        settings = replace(
            settings_from_discovery_candidate(candidate),
            daily_residual_catastrophic_stop_residual_r=stop,
        )
        exact = run_exact_fold_evaluation(
            replace(bundle, frozen_history_cache={}),
            settings,
            round_trip_cost_bps=20.0,
        )
        return {
            "candidate": candidate,
            "catastrophic_stop_residual_r": stop,
            "settings": exact["settings"],
            "exact": exact,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(evaluate, registry))
    results.sort(
        key=lambda row: (
            not bool(row["exact"]["research_anchor_eligible"]),
            -float(row["exact"]["immutable_score"]["score"]),
            row["candidate"]["candidate_id"],
        )
    )
    _write_json(
        output / "exact_followup_results.json",
        [
            {**row, "exact": _compact(row["exact"])}
            for row in results
        ],
    )
    eligible = [
        row for row in results if row["exact"]["research_anchor_eligible"]
    ]
    if not eligible:
        _status(
            output,
            "complete_no_representative_candidate",
            best_candidate_id=results[0]["candidate"]["candidate_id"],
            best_failed_gates=[
                name
                for name, passed in results[0]["exact"]["gates"].items()
                if not passed
            ],
        )
        return 2

    selected = eligible[0]
    selected_path = output / "selected_baseline_config.json"
    _write_json(
        selected_path,
        {
            "candidate": selected["candidate"],
            "catastrophic_stop_residual_r": selected[
                "catastrophic_stop_residual_r"
            ],
            "settings": selected["settings"],
        },
    )
    _status(
        output,
        "materializing_full_representative_baseline",
        candidate_id=selected["candidate"]["candidate_id"],
    )
    final_settings = replace(
        settings_from_discovery_candidate(selected["candidate"]),
        daily_residual_catastrophic_stop_residual_r=float(
            selected["catastrophic_stop_residual_r"]
        ),
    )
    diagnostics_output = output / "representative_baseline"
    payload = run_full_diagnostics(
        diagnostics_output,
        data_dir,
        settings=final_settings,
        contract_id=SELECTED_REPRESENTATIVE_CONTRACT_ID,
    )
    _status(
        output,
        "complete_representative_baseline",
        candidate_id=selected["candidate"]["candidate_id"],
        representative_alpha_baseline=payload["representative_alpha_baseline"],
        diagnostics_output=str(diagnostics_output),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=discovery.DEFAULT_DATA_DIR)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    return run(
        args.output_dir.resolve(),
        args.data_dir.resolve(),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
