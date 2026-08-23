"""Build the receipt-backed IARIC residual price-authority contract.

The command does not bless legacy Parquets.  It accepts only an immutable
bundle whose 98 stocks plus SPY and eleven sector ETFs were acquired directly
from IBKR as RTH ``ADJUSTED_LAST`` daily bars.  The bundle resolver rechecks
every object and acquisition receipt before this command emits an attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.iaric.representative_contract import (
    AUTHORITY_MANIFEST_VERSION,
    CALIBRATION_END,
    DISCOVERY_START,
    LOCKED_VALIDATION_END,
)
from backtests.stock.auto.runners.run_iaric_daily_residual_discovery import (
    SECTOR_ETFS,
    WARMUP_START,
)
from backtests.stock.data.bundle import verify_frozen_bundle
from backtests.stock.data.calendar import RTH_SESSION_POLICY, is_trading_day
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUNDLE = Path(
    "backtests/stock/data/authority/bundles/iaric_residual_adjusted_last.json"
)
DEFAULT_EVIDENCE = Path(
    "backtests/stock/data/authority/representative_reversion/price_authority_evidence.json"
)
DEFAULT_MANIFEST = Path(
    "backtests/stock/data/authority/representative_reversion/authority_manifest.json"
)
REQUIRED_CODE = (
    "strategies/stock/live_universe.py",
    "strategies/stock/volume_units.py",
    "strategies/stock/iaric/research_generator.py",
    "strategies/stock/iaric/daily_residual_selection.py",
    "strategies/stock/iaric/core/daily_residual.py",
    "strategies/stock/iaric/residual_engine.py",
    "backtests/stock/engine/iaric_daily_residual_replay.py",
    "backtests/stock/auto/runners/run_iaric_daily_residual_discovery.py",
    "backtests/stock/auto/runners/run_iaric_residual_phased_auto.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _sha256(path)}


def _input(
    *,
    source_id: str,
    schema_fingerprint: str,
    historical_adapter: str,
    live_adapter: str,
    semantics: str,
    artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "certified": True,
        "point_in_time": True,
        "availability_time_documented": True,
        "economic_input_parity_certified": True,
        "source_id": source_id,
        "schema_fingerprint": schema_fingerprint,
        "historical_adapter": historical_adapter,
        "live_adapter": live_adapter,
        "availability_time_semantics": semantics,
        "coverage": {"start": DISCOVERY_START, "end": LOCKED_VALIDATION_END},
        "selection_view_end": CALIBRATION_END,
        "artifacts": artifacts,
    }


def build_authority(
    bundle_path: Path,
    *,
    evidence_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    bundle_path = bundle_path.resolve()
    report = verify_frozen_bundle(
        bundle_path,
        repo_root=REPO_ROOT,
        require_clean=False,
        expected_universe=list(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        expected_session_policy_by_timeframe={"1d": RTH_SESSION_POLICY},
        expected_what_to_show_by_timeframe={"1d": "ADJUSTED_LAST"},
    )
    if not report["valid"]:
        raise ValueError(
            "IARIC residual bundle verification failed: "
            + "; ".join(report["errors"])
        )
    bundle = report["bundle"]
    expected = {
        *BACKTESTED_INTRADAY_STOCK_SYMBOLS,
        "SPY",
        *SECTOR_ETFS.values(),
    }
    entries = bundle.get("entries", [])
    observed = {str(row["symbol"]).upper() for row in entries}
    if observed != expected or len(entries) != len(expected):
        raise ValueError(
            "authority bundle must contain exactly 98 stocks plus SPY and 11 sector ETFs"
        )
    invalid_identity = [
        str(row["symbol"])
        for row in entries
        if row["dataset_identity"].get("what_to_show") != "ADJUSTED_LAST"
        or row["dataset_identity"].get("session_policy") != RTH_SESSION_POLICY
        or row["dataset_identity"].get("adjustment_policy")
        != "ibkr_adjusted_last_split_dividend_adjusted_v1"
    ]
    if invalid_identity:
        raise ValueError(
            "bundle contains a non-adjusted or non-RTH identity: "
            + ", ".join(invalid_identity)
        )
    required_last_session = date.fromisoformat(LOCKED_VALIDATION_END)
    while not is_trading_day(required_last_session):
        required_last_session -= timedelta(days=1)
    insufficient_coverage = []
    for row in entries:
        coverage = row.get("coverage", {})
        start = str(coverage.get("observed_start_utc", ""))[:10]
        end = str(coverage.get("observed_end_utc", ""))[:10]
        if (
            not start
            or start > WARMUP_START
            or not end
            or end < required_last_session.isoformat()
        ):
            insufficient_coverage.append(str(row["symbol"]))
    if insufficient_coverage:
        raise ValueError(
            "bundle lacks warmup-through-locked coverage: "
            + ", ".join(insufficient_coverage)
        )

    code = [_artifact(REPO_ROOT / relative) for relative in REQUIRED_CODE]
    evidence = {
        "schema_version": "iaric_residual_price_authority_evidence_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle["bundle_id"],
        "bundle_checksum": bundle["bundle_checksum"],
        "bundle_path": _relative(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "stock_universe_count": len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        "stock_universe": list(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        "non_traded_references": sorted(expected - set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)),
        "dataset_count": len(entries),
        "daily_what_to_show": "ADJUSTED_LAST",
        "session_policy": RTH_SESSION_POLICY,
        "adjustment_policy": "ibkr_adjusted_last_split_dividend_adjusted_v1",
        "volume_unit": "ibkr_us_equity_100_share_lots",
        "share_volume_multiplier": 100,
        "universe_semantics": (
            "predeclared fixed execution universe conditional on the 98 names with "
            "complete intraday data; never an index-membership claim"
        ),
        "availability_semantics": (
            "nightly selection consumes only sessions strictly before trade_date; "
            "selection reads end at calibration and locked validation is a separate one-shot view"
        ),
        "live_historical_parity": (
            "both adapters consume IBKR useRTH=true ADJUSTED_LAST daily OHLCV; "
            "shared selector and execution reducer are common"
        ),
        "news_quotes_or_order_imbalance_required": False,
        "code_artifacts": code,
    }
    _write_json(evidence_path, evidence)
    shared_artifacts = [
        _artifact(bundle_path),
        _artifact(evidence_path),
        *code,
    ]
    source_id = f"ibkr-adjusted-last:{bundle['bundle_id']}"
    schema = str(bundle["bundle_checksum"])
    inputs = {
        "daily_ohlcv": _input(
            source_id=source_id,
            schema_fingerprint=schema,
            historical_adapter="FrozenBundleResolver:ADJUSTED_LAST:RTH",
            live_adapter="iaric.research_generator:ADJUSTED_LAST:RTH",
            semantics=evidence["availability_semantics"],
            artifacts=shared_artifacts,
        ),
        "causal_universe_definition": _input(
            source_id="frozen-98-execution-universe",
            schema_fingerprint=_sha256(REPO_ROOT / "strategies/stock/live_universe.py"),
            historical_adapter="BACKTESTED_INTRADAY_STOCK_SYMBOLS",
            live_adapter="BACKTESTED_INTRADAY_STOCK_SYMBOLS",
            semantics=evidence["universe_semantics"],
            artifacts=shared_artifacts,
        ),
        "corporate_action_consistent_price_basis": _input(
            source_id=source_id,
            schema_fingerprint="ibkr-adjusted-last-split-dividend-v1",
            historical_adapter="FrozenBundleResolver:ADJUSTED_LAST",
            live_adapter="reqHistoricalData:ADJUSTED_LAST",
            semantics=(
                "IBKR ADJUSTED_LAST daily bars are split- and dividend-adjusted; "
                "the same basis is used by historical and live nightly signal construction"
            ),
            artifacts=shared_artifacts,
        ),
        "volume_unit_semantics": _input(
            source_id=source_id,
            schema_fingerprint="ibkr-us-equity-volume-100-share-lots-v1",
            historical_adapter="IBKR_SHARE_VOLUME_MULTIPLIER=100",
            live_adapter="IBKR_SHARE_VOLUME_MULTIPLIER=100",
            semantics="stored IBKR US equity volume is multiplied by 100 before dollar-volume use",
            artifacts=shared_artifacts,
        ),
        "completed_session_timestamps": _input(
            source_id=source_id,
            schema_fingerprint="completed-session-before-trade-date-v1",
            historical_adapter="authoritative bundle bounded daily loader",
            live_adapter="research_generator strict trade_date exclusion",
            semantics=evidence["availability_semantics"],
            artifacts=shared_artifacts,
        ),
        "historical_live_price_volume_parity": _input(
            source_id=source_id,
            schema_fingerprint="iaric-price-volume-parity-v1",
            historical_adapter="daily_residual_selection+shared execution core",
            live_adapter="daily_residual_selection+shared execution core",
            semantics=evidence["live_historical_parity"],
            artifacts=shared_artifacts,
        ),
    }
    manifest = {
        "manifest_version": AUTHORITY_MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_bundle": {
            "path": _relative(bundle_path),
            "sha256": _sha256(bundle_path),
            "bundle_id": bundle["bundle_id"],
            "bundle_checksum": bundle["bundle_checksum"],
            "daily_what_to_show": "ADJUSTED_LAST",
        },
        "inputs": inputs,
        "five_minute_sleeves_certified": False,
        "sealed_holdout_start": "2026-03-02",
        "holdout_accessed": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = build_authority(
        args.bundle.resolve(),
        evidence_path=args.evidence.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(args.manifest.resolve()),
                "bundle_id": manifest["selection_bundle"]["bundle_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
