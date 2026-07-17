"""Frozen standalone stock-data bundle construction and fail-closed loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.data.authority import (
    DEFAULT_AUTHORITY_ROOT,
    canonical_json_sha256,
    git_dirty_paths,
    git_revision,
    normalized_content_sha256,
    read_bar_frame,
    receipt_id,
    sha256_file,
    write_immutable_json,
)
from backtests.stock.data.calendar import RTH_SESSION_POLICY


BUNDLE_SCHEMA_VERSION = "stock_frozen_data_bundle_v1"


def universe_sha256(symbols: Iterable[str]) -> str:
    return canonical_json_sha256([str(symbol).upper().strip() for symbol in symbols])


def source_tree_sha256(repo_root: Path) -> tuple[str, int]:
    candidates = source_tree_paths(repo_root)
    repo_root = Path(repo_root)
    records = [
        {
            "path": path.resolve().relative_to(repo_root.resolve()).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in candidates
    ]
    return canonical_json_sha256(records), len(records)


def source_tree_paths(repo_root: Path) -> list[Path]:
    repo_root = Path(repo_root)
    candidates: list[Path] = []
    for root, suffixes in (
        (repo_root / "backtests" / "stock", {".py"}),
        (repo_root / "backtests" / "shared" / "data" / "ibkr", {".py"}),
        (repo_root / "strategies" / "stock", {".py"}),
        (repo_root / "config", {".json", ".yaml", ".yml", ".toml"}),
    ):
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    if (repo_root / "pyproject.toml").exists():
        candidates.append(repo_root / "pyproject.toml")
    return sorted(set(candidates))


def build_frozen_bundle(
    *,
    repo_root: Path,
    authority_root: Path = DEFAULT_AUTHORITY_ROOT,
    output_path: Path,
    intraday_symbols: list[str],
    daily_symbols: list[str],
    timeframes: tuple[str, ...] = ("1d", "30m", "5m"),
    session_policy_by_timeframe: dict[str, str] | None = None,
    require_clean: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    authority_root = _resolve(authority_root, repo_root)
    session_policy_by_timeframe = session_policy_by_timeframe or {
        timeframe: RTH_SESSION_POLICY for timeframe in timeframes
    }
    normalized_intraday = [symbol.upper().strip() for symbol in intraday_symbols]
    normalized_daily = list(dict.fromkeys(symbol.upper().strip() for symbol in daily_symbols))
    universe_source_path = repo_root / "strategies" / "stock" / "live_universe.py"
    requirements: list[dict[str, str]] = []
    for timeframe in timeframes:
        symbols = normalized_daily if timeframe in {"1d", "daily"} else normalized_intraday
        requirements.extend(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "session_policy": session_policy_by_timeframe[timeframe],
            }
            for symbol in symbols
        )

    references = _latest_references(authority_root)
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for requirement in requirements:
        matches = [
            reference
            for reference in references
            if _reference_key(reference) == _requirement_key(requirement)
        ]
        if len(matches) != 1:
            missing.append(
                f"{requirement['symbol']}:{requirement['timeframe']}:{requirement['session_policy']} "
                f"resolved to {len(matches)} latest references"
            )
            continue
        entries.append(_bundle_entry(matches[0], repo_root))

    code_config_sha256, code_config_file_count = source_tree_sha256(repo_root)
    input_paths = [
        _resolve(entry["object_path"], repo_root)
        for entry in entries
    ] + [
        _resolve(entry["receipt_path"], repo_root)
        for entry in entries
    ]
    dirty = git_dirty_paths(repo_root, [*input_paths, *source_tree_paths(repo_root)]) if require_clean else []
    blocking_reasons = [*missing]
    if dirty:
        blocking_reasons.append("dirty or untracked bundle inputs: " + ", ".join(sorted(dirty)))
    non_authoritative = [
        f"{entry['symbol']}:{entry['timeframe']}" for entry in entries if not entry["accepted"]
    ]
    if non_authoritative:
        blocking_reasons.append("non-authoritative receipts: " + ", ".join(non_authoritative))
    accepted = not blocking_reasons and len(entries) == len(requirements)

    stable_payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "accepted": accepted,
        "repository_revision": git_revision(repo_root),
        "code_config_sha256": code_config_sha256,
        "code_config_file_count": code_config_file_count,
        "calendar_versions": sorted(
            {entry["dataset_identity"]["calendar_version"] for entry in entries}
        ),
        "universe": {
            "source": "strategies.stock.live_universe.BACKTESTED_INTRADAY_STOCK_SYMBOLS",
            "source_path": "strategies/stock/live_universe.py",
            "source_sha256": sha256_file(universe_source_path),
            "ordered_symbols": normalized_intraday,
            "count": len(normalized_intraday),
            "sha256": universe_sha256(normalized_intraday),
        },
        "requirements": requirements,
        "entries": entries,
        "blocking_reasons": blocking_reasons,
    }
    checksum = canonical_json_sha256(stable_payload)
    bundle = {
        **stable_payload,
        "bundle_id": f"stockbundle1_{checksum[:24]}",
        "bundle_checksum": checksum,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if not accepted:
        raise ValueError("frozen stock bundle is blocked: " + "; ".join(blocking_reasons))
    write_immutable_json(output_path, bundle)
    return bundle


def verify_frozen_bundle(
    bundle_path: Path,
    *,
    repo_root: Path,
    require_clean: bool = True,
    expected_universe: list[str] | None = None,
    expected_session_policy_by_timeframe: dict[str, str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append("unsupported frozen bundle schema")
    stable_payload = {
        key: value
        for key, value in bundle.items()
        if key not in {"bundle_id", "bundle_checksum", "generated_at"}
    }
    recomputed = canonical_json_sha256(stable_payload)
    if recomputed != bundle.get("bundle_checksum"):
        errors.append("bundle checksum mismatch")
    if bundle.get("bundle_id") != f"stockbundle1_{recomputed[:24]}":
        errors.append("bundle id mismatch")
    if not bundle.get("accepted"):
        errors.append("bundle is not accepted")
    if bundle.get("blocking_reasons"):
        errors.append("bundle contains blocking reasons")

    if expected_universe is not None:
        expected = [symbol.upper().strip() for symbol in expected_universe]
        universe = bundle.get("universe", {})
        if universe.get("ordered_symbols") != expected:
            errors.append("bundle universe does not exactly match the canonical ordered universe")
        if universe.get("sha256") != universe_sha256(expected):
            errors.append("bundle universe checksum mismatch")
        source_path = _resolve(universe.get("source_path", ""), repo_root)
        if not source_path.exists() or sha256_file(source_path) != universe.get("source_sha256"):
            errors.append("canonical universe source file checksum mismatch")

    seen: set[tuple[str, str]] = set()
    input_paths: list[Path] = []
    for entry in bundle.get("entries", []):
        key = (str(entry.get("symbol", "")).upper(), str(entry.get("timeframe", "")).lower())
        if key in seen:
            errors.append(f"duplicate bundle dataset: {key[0]}:{key[1]}")
        seen.add(key)
        identity = entry.get("dataset_identity", {})
        if expected_session_policy_by_timeframe is not None:
            expected_policy = expected_session_policy_by_timeframe.get(key[1])
            if expected_policy and identity.get("session_policy") != expected_policy:
                errors.append(f"session policy mismatch for {key[0]}:{key[1]}")
        object_path = _resolve(entry.get("object_path", ""), repo_root)
        receipt_path = _resolve(entry.get("receipt_path", ""), repo_root)
        input_paths.extend([object_path, receipt_path])
        if not object_path.exists():
            errors.append(f"missing immutable object: {object_path}")
            continue
        if not receipt_path.exists():
            errors.append(f"missing acquisition receipt: {receipt_path}")
            continue
        if sha256_file(object_path) != entry.get("physical_sha256"):
            errors.append(f"physical checksum mismatch: {key[0]}:{key[1]}")
        try:
            frame = read_bar_frame(object_path)
            if normalized_content_sha256(frame) != entry.get("normalized_content_sha256"):
                errors.append(f"normalized content mismatch: {key[0]}:{key[1]}")
        except Exception as exc:
            errors.append(f"unreadable immutable object {key[0]}:{key[1]}: {exc}")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"unreadable acquisition receipt {key[0]}:{key[1]}: {exc}")
            continue
        if receipt_id(receipt) != receipt.get("receipt_id"):
            errors.append(f"receipt content hash mismatch: {key[0]}:{key[1]}")
        if receipt.get("receipt_id") != entry.get("receipt_id"):
            errors.append(f"receipt id mismatch: {key[0]}:{key[1]}")
        if receipt.get("dataset_identity_sha256") != entry.get("dataset_identity_sha256"):
            errors.append(f"receipt dataset identity mismatch: {key[0]}:{key[1]}")
        if not receipt.get("accepted"):
            errors.append(f"receipt is not accepted: {key[0]}:{key[1]}")

    requirement_keys = {_requirement_key(item) for item in bundle.get("requirements", [])}
    entry_keys = {
        (
            str(item.get("symbol", "")).upper(),
            str(item.get("timeframe", "")).lower(),
            str(item.get("dataset_identity", {}).get("session_policy", "")),
        )
        for item in bundle.get("entries", [])
    }
    if requirement_keys != entry_keys:
        errors.append("bundle entries do not exactly satisfy bundle requirements")

    current_tree_sha, current_tree_count = source_tree_sha256(repo_root)
    if current_tree_sha != bundle.get("code_config_sha256"):
        errors.append("code/config tree checksum differs from the frozen bundle")
    if current_tree_count != bundle.get("code_config_file_count"):
        errors.append("code/config file count differs from the frozen bundle")
    if require_clean:
        dirty = git_dirty_paths(repo_root, [*input_paths, *source_tree_paths(repo_root)])
        if dirty:
            errors.append("dirty or untracked bundle inputs: " + ", ".join(sorted(dirty)))
    return {
        "valid": not errors,
        "errors": errors,
        "bundle": bundle,
        "recomputed_bundle_checksum": recomputed,
    }


@dataclass(frozen=True)
class FrozenBundleResolver:
    repo_root: Path
    bundle_path: Path
    bundle: dict[str, Any]
    paths: dict[tuple[str, str], Path]

    @classmethod
    def load(
        cls,
        bundle_path: Path,
        *,
        repo_root: Path,
        require_clean: bool = True,
        expected_universe: list[str] | None = None,
        expected_session_policy_by_timeframe: dict[str, str] | None = None,
    ) -> "FrozenBundleResolver":
        report = verify_frozen_bundle(
            bundle_path,
            repo_root=repo_root,
            require_clean=require_clean,
            expected_universe=expected_universe,
            expected_session_policy_by_timeframe=expected_session_policy_by_timeframe,
        )
        if not report["valid"]:
            raise ValueError("frozen stock bundle verification failed: " + "; ".join(report["errors"]))
        bundle = report["bundle"]
        paths = {
            (str(entry["symbol"]).upper(), str(entry["timeframe"]).lower()): _resolve(
                entry["object_path"], Path(repo_root)
            )
            for entry in bundle["entries"]
        }
        return cls(Path(repo_root), Path(bundle_path), bundle, paths)

    @property
    def bundle_checksum(self) -> str:
        return str(self.bundle["bundle_checksum"])

    def bar_path(self, symbol: str, timeframe: str) -> Path:
        key = (symbol.upper(), timeframe.lower())
        if key not in self.paths:
            raise FileNotFoundError(f"frozen bundle has no dataset for {key[0]}:{key[1]}")
        return self.paths[key]


def _latest_references(authority_root: Path) -> list[dict[str, Any]]:
    root = Path(authority_root) / "refs" / "latest"
    if not root.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def _bundle_entry(reference: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    receipt_path = _resolve(reference["receipt_path"], repo_root)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = reference["dataset_identity"]
    return {
        "symbol": identity["symbol"],
        "timeframe": identity["timeframe"],
        "dataset_id": reference["dataset_id"],
        "dataset_identity": identity,
        "dataset_identity_sha256": reference["dataset_identity_sha256"],
        "object_path": reference["object_path"],
        "physical_sha256": reference["physical_sha256"],
        "normalized_content_sha256": reference["content_sha256"],
        "normalized_schema": receipt.get("object", {}).get("normalized_schema", {}),
        "receipt_id": reference["receipt_id"],
        "receipt_path": reference["receipt_path"],
        "accepted": bool(receipt.get("accepted")),
        "coverage": receipt.get("coverage", {}),
        "calendar_validation": receipt.get("calendar_validation", {}),
    }


def _reference_key(reference: dict[str, Any]) -> tuple[str, str, str]:
    identity = reference.get("dataset_identity", {})
    return (
        str(identity.get("symbol", "")).upper(),
        str(identity.get("timeframe", "")).lower(),
        str(identity.get("session_policy", "")),
    )


def _requirement_key(requirement: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(requirement.get("symbol", "")).upper(),
        str(requirement.get("timeframe", "")).lower(),
        str(requirement.get("session_policy", "")),
    )


def _resolve(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(repo_root) / candidate
