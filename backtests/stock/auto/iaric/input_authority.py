"""Dynamic, fail-closed price/volume authority attestation for IARIC.

This module implements the optional receipt-backed deployment contract.  The
phased research runner separately accepts the project-designated official local
snapshot, so original acquisition logs are never an alpha-research blocker.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from backtests.stock.auto.iaric.representative_contract import (
    AUTHORITY_MANIFEST_VERSION,
    CALIBRATION_END,
    CURRENT_INPUT_AUTHORITY,
    DISCOVERY_START,
    HOLDOUT_START,
)


DEFAULT_MANIFEST_RELATIVE = Path(
    "backtests/stock/data/authority/representative_reversion/authority_manifest.json"
)

REQUIRED_ATTESTATION_FIELDS = (
    "certified",
    "point_in_time",
    "availability_time_documented",
    "economic_input_parity_certified",
    "source_id",
    "schema_fingerprint",
    "historical_adapter",
    "live_adapter",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _diagnostic_inventory(repository_root: Path) -> dict[str, Any]:
    raw = repository_root / "backtests/stock/data/raw"
    counts = {"daily_files": 0, "five_minute_files": 0, "thirty_minute_files": 0}
    if raw.exists():
        counts["daily_files"] = sum(1 for _ in raw.glob("*_1d.parquet"))
        counts["five_minute_files"] = sum(1 for _ in raw.glob("*_5m.parquet"))
        counts["thirty_minute_files"] = sum(1 for _ in raw.glob("*_30m.parquet"))
    return {
        **counts,
        "daily_ohlcv_available": counts["daily_files"] > 0,
        "five_minute_ohlcv_available": counts["five_minute_files"] > 0,
        "strategy_input_scope": "price_volume_only",
        "news_quotes_or_order_imbalance_required": False,
        "classification": "diagnostic_file_presence_only",
        "certifies_input_authority": False,
    }


def _artifact_assessment(
    repository_root: Path,
    artifacts: object,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    if not isinstance(artifacts, list) or not artifacts:
        return False, rows, ["missing fingerprinted authority artifacts"]
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, Mapping):
            failures.append(f"artifact[{index}] is not an object")
            continue
        relative = str(raw.get("path", "")).strip()
        expected = str(raw.get("sha256", "")).strip().lower()
        path = (repository_root / relative).resolve()
        row = {"path": relative, "expected_sha256": expected}
        if not relative or not _inside(repository_root, path):
            row["passed"] = False
            row["reason"] = "artifact path is missing or outside the repository"
            failures.append(f"invalid artifact path: {relative or '<missing>'}")
        elif not path.is_file():
            row["passed"] = False
            row["reason"] = "artifact is missing"
            failures.append(f"missing artifact: {relative}")
        elif len(expected) != 64:
            row["passed"] = False
            row["reason"] = "sha256 is missing or malformed"
            failures.append(f"invalid artifact sha256: {relative}")
        else:
            actual = _sha256(path)
            row["actual_sha256"] = actual
            row["passed"] = actual == expected
            if actual != expected:
                row["reason"] = "artifact checksum mismatch"
                failures.append(f"artifact checksum mismatch: {relative}")
        rows.append(row)
    return bool(rows) and not failures, rows, failures


def _input_assessment(
    name: str,
    raw: object,
    *,
    repository_root: Path,
    required_start: str,
    required_end: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "certified": False,
            "failures": ["missing input attestation"],
            "artifacts": [],
        }
    failures: list[str] = []
    for field in REQUIRED_ATTESTATION_FIELDS:
        value = raw.get(field)
        if field in {
            "certified",
            "point_in_time",
            "availability_time_documented",
            "economic_input_parity_certified",
        }:
            if value is not True:
                failures.append(f"{field} is not true")
        elif not str(value or "").strip():
            failures.append(f"{field} is missing")
    coverage = raw.get("coverage") or {}
    coverage_start = str(coverage.get("start", ""))
    coverage_end = str(coverage.get("end", ""))
    if not coverage_start or coverage_start > required_start:
        failures.append(
            f"coverage begins after required discovery start {required_start}"
        )
    if not coverage_end or coverage_end < required_end:
        failures.append(f"coverage ends before required calibration end {required_end}")
    if coverage_end and coverage_end >= HOLDOUT_START:
        # A source may contain later records, but the certified selection view
        # must name a bounded snapshot that cannot expose the sealed holdout.
        selection_end = str(raw.get("selection_view_end", ""))
        if selection_end != required_end:
            failures.append(
                "source extends into the sealed period without an exact calibration-bounded selection view"
            )
    artifacts_passed, artifacts, artifact_failures = _artifact_assessment(
        repository_root,
        raw.get("artifacts"),
    )
    failures.extend(artifact_failures)
    return {
        "certified": not failures and artifacts_passed,
        "source_id": str(raw.get("source_id", "")),
        "coverage": {"start": coverage_start, "end": coverage_end},
        "selection_view_end": str(raw.get("selection_view_end", required_end)),
        "schema_fingerprint": str(raw.get("schema_fingerprint", "")),
        "historical_adapter": str(raw.get("historical_adapter", "")),
        "live_adapter": str(raw.get("live_adapter", "")),
        "availability_time_semantics": str(raw.get("availability_time_semantics", "")),
        "artifacts": artifacts,
        "failures": failures,
        "input": name,
    }


def attest_input_authority(
    repository_root: Path,
    *,
    manifest_path: Path | None = None,
    required_start: str = DISCOVERY_START,
    required_end: str = CALIBRATION_END,
) -> dict[str, Any]:
    """Return certified authority, evidence and diagnostic availability."""

    root = repository_root.resolve()
    manifest = (manifest_path or (root / DEFAULT_MANIFEST_RELATIVE)).resolve()
    diagnostic = _diagnostic_inventory(root)
    authority = dict(CURRENT_INPUT_AUTHORITY)
    if not manifest.is_file():
        return {
            "manifest_found": False,
            "manifest_path": str(manifest),
            "manifest_version": None,
            "manifest_sha256": None,
            "input_authority": authority,
            "input_evidence": {
                name: {
                    "certified": False,
                    "failures": ["authority manifest is missing"],
                }
                for name in authority
            },
            "diagnostic_availability": diagnostic,
            "passed_inputs": [],
            "failed_inputs": list(authority),
            "failures": [f"missing authority manifest: {manifest}"],
        }

    failures: list[str] = []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "manifest_found": True,
            "manifest_path": str(manifest),
            "manifest_version": None,
            "manifest_sha256": _sha256(manifest),
            "input_authority": authority,
            "input_evidence": {},
            "diagnostic_availability": diagnostic,
            "passed_inputs": [],
            "failed_inputs": list(authority),
            "failures": [f"invalid authority manifest: {exc}"],
        }
    version = str(payload.get("manifest_version", ""))
    if version != AUTHORITY_MANIFEST_VERSION:
        failures.append(
            f"manifest version {version!r} does not match {AUTHORITY_MANIFEST_VERSION!r}"
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        failures.append("manifest inputs must be an object")
        inputs = {}
    evidence: dict[str, Any] = {}
    for name in authority:
        row = _input_assessment(
            name,
            inputs.get(name),
            repository_root=root,
            required_start=required_start,
            required_end=required_end,
        )
        evidence[name] = row
        authority[name] = bool(row["certified"]) and version == AUTHORITY_MANIFEST_VERSION
    selection_bundle = payload.get("selection_bundle")
    bundle_assessment: dict[str, Any] = {
        "certified": False,
        "path": "",
        "failures": [],
    }
    if not isinstance(selection_bundle, Mapping):
        bundle_assessment["failures"].append(
            "manifest selection_bundle is missing"
        )
    else:
        relative = str(selection_bundle.get("path", "")).strip()
        expected = str(selection_bundle.get("sha256", "")).strip().lower()
        price_basis = str(selection_bundle.get("daily_what_to_show", "")).upper()
        bundle_path = (root / relative).resolve()
        bundle_assessment["path"] = str(bundle_path)
        bundle_assessment["expected_sha256"] = expected
        bundle_assessment["daily_what_to_show"] = price_basis
        if not relative or not _inside(root, bundle_path):
            bundle_assessment["failures"].append(
                "selection bundle path is missing or outside the repository"
            )
        elif not bundle_path.is_file():
            bundle_assessment["failures"].append(
                f"selection bundle is missing: {relative}"
            )
        else:
            actual = _sha256(bundle_path)
            bundle_assessment["actual_sha256"] = actual
            if actual != expected:
                bundle_assessment["failures"].append(
                    "selection bundle checksum mismatch"
                )
        if price_basis != "ADJUSTED_LAST":
            bundle_assessment["failures"].append(
                "representative daily bundle must use ADJUSTED_LAST"
            )
        bundle_assessment["certified"] = not bundle_assessment["failures"]
    if not bundle_assessment["certified"]:
        for name in authority:
            authority[name] = False
        failures.extend(bundle_assessment["failures"])
    passed = [name for name, value in authority.items() if value]
    failed = [name for name, value in authority.items() if not value]
    return {
        "manifest_found": True,
        "manifest_path": str(manifest),
        "manifest_version": version,
        "manifest_sha256": _sha256(manifest),
        "input_authority": authority,
        "input_evidence": evidence,
        "selection_bundle": bundle_assessment,
        "diagnostic_availability": diagnostic,
        "passed_inputs": passed,
        "failed_inputs": failed,
        "failures": failures,
    }
