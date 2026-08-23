from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backtests.stock.auto.iaric.input_authority import attest_input_authority
from backtests.stock.auto.iaric.representative_contract import AUTHORITY_MANIFEST_VERSION


def _manifest_input(artifact: Path, repository_root: Path) -> dict[str, object]:
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return {
        "certified": True,
        "point_in_time": True,
        "availability_time_documented": True,
        "economic_input_parity_certified": True,
        "source_id": "test-source-snapshot",
        "schema_fingerprint": "schema-v1-sha256",
        "historical_adapter": "tests.fake:HistoricalAdapter",
        "live_adapter": "tests.fake:LiveAdapter",
        "availability_time_semantics": "available after the completed source timestamp",
        "coverage": {"start": "2024-03-01", "end": "2025-07-31"},
        "selection_view_end": "2025-07-31",
        "artifacts": [
            {
                "path": artifact.relative_to(repository_root).as_posix(),
                "sha256": digest,
            }
        ],
    }


def test_manifest_attestation_requires_real_fingerprinted_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "daily_inventory.json"
    artifact.write_text('{"rows": 10}\n', encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    bundle.write_text('{"accepted": true}\n', encoding="utf-8")
    manifest = tmp_path / "authority_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": AUTHORITY_MANIFEST_VERSION,
                "selection_bundle": {
                    "path": bundle.name,
                    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    "daily_what_to_show": "ADJUSTED_LAST",
                },
                "inputs": {"daily_ohlcv": _manifest_input(artifact, tmp_path)},
            }
        ),
        encoding="utf-8",
    )
    result = attest_input_authority(tmp_path, manifest_path=manifest)
    assert result["input_authority"]["daily_ohlcv"] is True
    assert result["input_authority"]["five_minute_ohlcv"] is False
    assert result["input_evidence"]["daily_ohlcv"]["artifacts"][0]["passed"] is True


def test_manifest_checksum_drift_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "daily_inventory.json"
    artifact.write_text('{"rows": 10}\n', encoding="utf-8")
    row = _manifest_input(artifact, tmp_path)
    bundle = tmp_path / "bundle.json"
    bundle.write_text('{"accepted": true}\n', encoding="utf-8")
    artifact.write_text('{"rows": 11}\n', encoding="utf-8")
    manifest = tmp_path / "authority_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": AUTHORITY_MANIFEST_VERSION,
                "selection_bundle": {
                    "path": bundle.name,
                    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    "daily_what_to_show": "ADJUSTED_LAST",
                },
                "inputs": {"daily_ohlcv": row},
            }
        ),
        encoding="utf-8",
    )
    result = attest_input_authority(tmp_path, manifest_path=manifest)
    assert result["input_authority"]["daily_ohlcv"] is False
    assert "checksum mismatch" in " ".join(
        result["input_evidence"]["daily_ohlcv"]["failures"]
    )
