"""Fast price/volume authority preflight for representative IARIC."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CONTRACT_VERSION,
    DISCOVERY_START,
    DOWNSTREAM_EXECUTION_CONTRACT,
    EXPERIMENT_REGISTRY,
    HOLDOUT_START,
    PHASE_ORDER,
    assess_input_authority,
    chronology_contract,
)
from backtests.stock.auto.iaric.input_authority import (
    DEFAULT_MANIFEST_RELATIVE,
    attest_input_authority,
)
from strategies.stock.iaric.core.mechanisms import SLEEVE_SPECS, validate_sleeve_specs


REPO_ROOT = Path(__file__).resolve().parents[4]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", default=DISCOVERY_START)
    parser.add_argument("--end-date", default=CALIBRATION_END)
    parser.add_argument(
        "--authority-manifest",
        default=str(REPO_ROOT / DEFAULT_MANIFEST_RELATIVE),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_preflight_payload(
    start_date: str,
    end_date: str,
    *,
    authority_manifest: str | Path | None = None,
    repository_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    attestation = attest_input_authority(
        root,
        manifest_path=(Path(authority_manifest).resolve() if authority_manifest else None),
        required_start=start_date,
        required_end=end_date,
    )
    authority = dict(attestation["input_authority"])
    assessment = assess_input_authority(authority)
    selection_window_valid = (
        str(start_date) == DISCOVERY_START
        and str(end_date) == CALIBRATION_END
        and str(end_date) < HOLDOUT_START
    )
    representative_eligible = bool(
        assessment["representative_reversion_baseline_eligible"]
        and selection_window_valid
    )
    programme_blockers = list(assessment["programme_blockers"])
    blockers = list(assessment["blockers"])
    if not selection_window_valid:
        message = (
            "representative selection window must exactly match the registered "
            f"{DISCOVERY_START} through {CALIBRATION_END} interval"
        )
        programme_blockers.append(message)
        blockers.insert(0, message)
    mechanism_contract = validate_sleeve_specs()
    parity_sleeves = [
        name
        for name, row in assessment["sleeve_readiness"].items()
        if row["ready"]
    ]
    return {
        "status": "complete_price_data_preflight",
        "strategy_input_scope": "price_volume_only",
        "news_or_earnings_required": False,
        "historical_quotes_or_order_imbalance_required": False,
        "forbidden_alpha_inputs": [
            "point_in_time_news",
            "point_in_time_earnings",
            "historical_quotes",
            "order_imbalance",
        ],
        "representative_contract_version": CONTRACT_VERSION,
        "window": {"start": start_date, "end": end_date},
        "selection_window_valid": selection_window_valid,
        "chronology": chronology_contract(),
        "phase_order": list(PHASE_ORDER),
        "experiment_registry": EXPERIMENT_REGISTRY,
        "holdout_accessed": False,
        "input_authority": authority,
        "input_authority_attestation": attestation,
        "sleeve_readiness": assessment["sleeve_readiness"],
        "ready_reversion_sleeves": assessment["ready_reversion_sleeves"],
        "ready_control_sleeves": assessment["ready_control_sleeves"],
        "disabled_sleeves": assessment["disabled_sleeves"],
        "mechanism_discovery_eligible": assessment["mechanism_discovery_eligible"],
        "representative_reversion_baseline_eligible": representative_eligible,
        "representative_reversion_baseline_blockers": blockers,
        "programme_blockers": programme_blockers,
        "mechanism_contract": {
            **mechanism_contract,
            "sleeves": {
                name: {
                    "role": spec.role,
                    "score_components": list(spec.score_components),
                    "score_component_count": len(spec.score_components),
                    "hard_vetoes": list(spec.hard_vetoes),
                    "entry_mechanisms": list(spec.entry_mechanisms),
                    "management_mechanisms": list(spec.management_mechanisms),
                    "diagnostic_legs": list(spec.diagnostic_legs),
                }
                for name, spec in SLEEVE_SPECS.items()
            },
        },
        "mechanism_atlas_complete": False,
        "mechanism_candidate_registry_complete": False,
        "qualified_sleeves": [],
        "optimization_launch_eligible": False,
        "economic_input_parity": {
            "passed": representative_eligible,
            "passed_sleeves": parity_sleeves,
            "reason": (
                "manifest-certified price/volume adapters and availability-time semantics"
                if representative_eligible
                else "representative price/volume authority or parity is incomplete"
            ),
        },
        "required_downstream_execution_contract": DOWNSTREAM_EXECUTION_CONTRACT,
        "downstream_execution_contract": "authority_only_no_execution",
    }


def main() -> int:
    args = _args()
    output = Path(args.output_dir).resolve()
    payload = build_preflight_payload(
        str(args.start_date),
        str(args.end_date),
        authority_manifest=args.authority_manifest,
    )
    _write_json(output / "atlas_summary.json", payload)
    _write_json(output / "progress.json", {
        "status": (
            "complete_representative_authority_ready"
            if payload["representative_reversion_baseline_eligible"]
            else "blocked_missing_authoritative_price_volume_inputs"
        ),
        "holdout_accessed": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(
        "representative input preflight: "
        + (
            "ready"
            if payload["representative_reversion_baseline_eligible"]
            else "blocked; replay imports skipped"
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
