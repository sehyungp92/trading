"""Standalone immutable stock-data identities, objects, receipts, and bundles."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtests.stock.data.calendar import (
    CALENDAR_VERSION,
    EXTENDED_SESSION_POLICY,
    RTH_SESSION_POLICY,
    expected_bar_opens,
    session_dates,
)


AUTHORITY_SCHEMA_VERSION = "stock_data_authority_v1"
IDENTITY_VERSION = "stock_dataset_identity_v1"
CONTENT_HASH_VERSION = "normalized_bar_content_v1"
DEFAULT_AUTHORITY_ROOT = Path("backtests/stock/data/authority")
DEFAULT_ADJUSTMENT_POLICY = "ibkr_trades_raw_unadjusted_v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetIdentity:
    provider: str
    market: str
    symbol: str
    con_id: str
    timeframe: str
    what_to_show: str
    session_policy: str
    adjustment_policy: str
    calendar_version: str = CALENDAR_VERSION
    identity_version: str = IDENTITY_VERSION

    def __post_init__(self) -> None:
        missing = [key for key, value in asdict(self).items() if not str(value).strip()]
        if missing:
            raise ValueError("dataset identity fields must be non-empty: " + ", ".join(missing))
        if self.session_policy not in {RTH_SESSION_POLICY, EXTENDED_SESSION_POLICY}:
            raise ValueError(f"unsupported session policy: {self.session_policy}")

    @property
    def payload(self) -> dict[str, str]:
        return {
            "provider": self.provider.lower().strip(),
            "market": self.market.lower().strip(),
            "symbol": self.symbol.upper().strip(),
            "con_id": str(self.con_id).strip(),
            "timeframe": self.timeframe.lower().strip(),
            "what_to_show": self.what_to_show.upper().strip(),
            "session_policy": self.session_policy.strip(),
            "adjustment_policy": self.adjustment_policy.strip(),
            "calendar_version": self.calendar_version.strip(),
            "identity_version": self.identity_version.strip(),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.payload)

    @property
    def dataset_id(self) -> str:
        return f"stockds1_{self.identity_sha256[:32]}"


def identity_for_request(
    *,
    symbol: str,
    con_id: str,
    timeframe: str,
    what_to_show: str,
    use_rth: bool,
    adjustment_policy: str = DEFAULT_ADJUSTMENT_POLICY,
    provider: str = "ibkr",
    market: str = "us_equity",
    calendar_version: str = CALENDAR_VERSION,
) -> DatasetIdentity:
    return DatasetIdentity(
        provider=provider,
        market=market,
        symbol=symbol,
        con_id=con_id,
        timeframe=timeframe,
        what_to_show=what_to_show,
        session_policy=RTH_SESSION_POLICY if use_rth else EXTENDED_SESSION_POLICY,
        adjustment_policy=adjustment_policy,
        calendar_version=calendar_version,
    )


def normalize_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy and new Parquets to one timestamp-index schema."""
    normalized = frame.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        timestamp_column = next(
            (name for name in ("time", "timestamp", "timestamp_utc", "__index_level_0__") if name in normalized.columns),
            None,
        )
        if timestamp_column is not None:
            normalized.index = pd.to_datetime(normalized.pop(timestamp_column), utc=True)
        else:
            normalized.index = pd.to_datetime(normalized.index, utc=True)
    elif normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")
    else:
        normalized.index = normalized.index.tz_convert("UTC")
    normalized.index.name = "time"
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized


def read_bar_frame(path: Path) -> pd.DataFrame:
    return normalize_bar_frame(pd.read_parquet(path, engine="pyarrow"))


def normalized_schema(frame: pd.DataFrame) -> dict[str, Any]:
    normalized = normalize_bar_frame(frame)
    return {
        "schema_version": CONTENT_HASH_VERSION,
        "index": {
            "name": "time",
            "dtype": str(normalized.index.dtype),
            "timezone": str(normalized.index.tz),
        },
        "columns": [
            {"name": str(column), "dtype": str(normalized[column].dtype)}
            for column in normalized.columns
        ],
    }


def normalized_content_sha256(frame: pd.DataFrame) -> str:
    normalized = normalize_bar_frame(frame)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(normalized_schema(normalized)))
    if not normalized.empty:
        row_hashes = pd.util.hash_pandas_object(normalized, index=True, categorize=True)
        digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def write_immutable_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json_sha256(existing) != canonical_json_sha256(payload):
            raise ValueError(f"refusing to overwrite immutable JSON object: {path}")
        return path
    return _atomic_write_json(path, payload)


def object_path(authority_root: Path, identity: DatasetIdentity, content_sha256: str) -> Path:
    return Path(authority_root) / "objects" / identity.dataset_id / content_sha256 / "bars.parquet"


def write_immutable_bars(
    authority_root: Path,
    identity: DatasetIdentity,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    normalized = normalize_bar_frame(frame)
    content_sha256 = normalized_content_sha256(normalized)
    target = object_path(authority_root, identity, content_sha256)
    if target.exists():
        existing = read_bar_frame(target)
        if normalized_content_sha256(existing) != content_sha256:
            raise ValueError(f"immutable bar object content mismatch: {target}")
        return {
            "path": target,
            "content_sha256": content_sha256,
            "physical_sha256": sha256_file(target),
            "schema": normalized_schema(existing),
            "rows": len(existing),
        }

    staging = Path(authority_root) / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{identity.dataset_id}.{uuid.uuid4().hex}.parquet"
    normalized.to_parquet(temporary, engine="pyarrow", index=True)
    persisted = read_bar_frame(temporary)
    if normalized_content_sha256(persisted) != content_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError("Parquet round-trip changed normalized stock bar content")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)
    return {
        "path": target,
        "content_sha256": content_sha256,
        "physical_sha256": sha256_file(target),
        "schema": normalized_schema(persisted),
        "rows": len(persisted),
    }


def receipt_id(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_id"}
    return canonical_json_sha256(body)


def write_receipt(
    authority_root: Path,
    identity: DatasetIdentity,
    payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    receipt = {
        "schema_version": "stock_acquisition_receipt_v1",
        **payload,
        "dataset_identity": identity.payload,
        "dataset_identity_sha256": identity.identity_sha256,
    }
    receipt["receipt_id"] = receipt_id(receipt)
    path = Path(authority_root) / "receipts" / identity.dataset_id / f"{receipt['receipt_id']}.json"
    write_immutable_json(path, receipt)
    return path, receipt


def latest_reference_path(authority_root: Path, identity: DatasetIdentity) -> Path:
    return Path(authority_root) / "refs" / "latest" / f"{identity.dataset_id}.json"


def write_latest_reference(
    authority_root: Path,
    identity: DatasetIdentity,
    *,
    object_record: dict[str, Any],
    receipt_path: Path,
    receipt: dict[str, Any],
    repo_root: Path,
) -> Path:
    if not receipt.get("accepted"):
        raise ValueError("only accepted acquisitions may update latest")
    payload = {
        "schema_version": "stock_dataset_latest_v1",
        "dataset_id": identity.dataset_id,
        "dataset_identity": identity.payload,
        "dataset_identity_sha256": identity.identity_sha256,
        "object_path": _relative_path(Path(object_record["path"]), repo_root),
        "content_sha256": object_record["content_sha256"],
        "physical_sha256": object_record["physical_sha256"],
        "receipt_id": receipt["receipt_id"],
        "receipt_path": _relative_path(receipt_path, repo_root),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return _atomic_write_json(latest_reference_path(authority_root, identity), payload)


def validate_bar_frame(
    frame: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
    session_policy: str,
) -> dict[str, Any]:
    normalized = normalize_bar_frame(frame)
    errors: list[str] = []
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = sorted(required_columns.difference(normalized.columns))
    if missing_columns:
        errors.append("missing required columns: " + ", ".join(missing_columns))
    if normalized.empty:
        errors.append("no bars returned")
    if not normalized.index.is_monotonic_increasing:
        errors.append("timestamps are not monotonic")
    if not normalized.index.is_unique:
        errors.append("timestamps are not unique")

    expected = expected_bar_opens(start, end, timeframe, session_policy)
    actual = pd.DatetimeIndex(normalized.index)
    if timeframe.lower() in {"1d", "daily"}:
        actual_compare = pd.DatetimeIndex([pd.Timestamp(day, tz="UTC") for day in session_dates(actual, daily_labels=True)])
    else:
        actual_compare = actual
    missing = expected.difference(actual_compare)
    unexpected = actual_compare.difference(expected)
    if len(missing):
        errors.append(f"missing {len(missing)} expected session bars")
    if len(unexpected):
        errors.append(f"found {len(unexpected)} bars outside the declared session calendar")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "calendar_version": CALENDAR_VERSION,
        "session_policy": session_policy,
        "expected_bars": len(expected),
        "actual_bars": len(actual_compare),
        "missing_bars": [stamp.isoformat() for stamp in missing[:100]],
        "unexpected_bars": [stamp.isoformat() for stamp in unexpected[:100]],
        "coverage_ratio": (len(actual_compare) / len(expected)) if len(expected) else 0.0,
    }


def create_legacy_snapshot_inventory(
    *,
    repo_root: Path,
    data_dir: Path,
    output_path: Path,
    label: str,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    data_dir = Path(data_dir).resolve()
    records: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.parquet")):
        frame = read_bar_frame(path)
        stem = path.stem
        timeframe = next((value for value in ("1d", "30m", "5m") if stem.endswith(f"_{value}")), "unknown")
        symbol = stem[: -(len(timeframe) + 1)] if timeframe != "unknown" else stem
        session_policy = RTH_SESSION_POLICY if timeframe == "1d" else EXTENDED_SESSION_POLICY
        records.append(
            {
                "path": _relative_path(path, repo_root),
                "symbol": symbol,
                "timeframe": timeframe,
                "declared_session_policy": session_policy,
                "physical_sha256": sha256_file(path),
                "normalized_content_sha256": normalized_content_sha256(frame),
                "schema": normalized_schema(frame),
                "rows": len(frame),
                "observed_start": frame.index.min().isoformat() if not frame.empty else None,
                "observed_end": frame.index.max().isoformat() if not frame.empty else None,
                "acquisition_receipt_id": None,
                "authoritative": False,
            }
        )
    aggregate = canonical_json_sha256(records)
    payload = {
        "schema_version": "legacy_stock_snapshot_inventory_v1",
        "snapshot_id": f"legacy_stock_{aggregate[:24]}",
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_head_before_snapshot": git_revision(repo_root),
        "source_data_dir": _relative_path(data_dir, repo_root),
        "file_count": len(records),
        "aggregate_inventory_sha256": aggregate,
        "acquisition_provenance": "unknown_legacy_cache",
        "authoritative": False,
        "policy_note": (
            "Retrospective hashes prove retained bytes only. They are not provider or acquisition receipts. "
            "Intraday files are labelled extended-hours from the updater policy and observed slots."
        ),
        "files": records,
    }
    if Path(output_path).exists():
        existing = json.loads(Path(output_path).read_text(encoding="utf-8"))
        if existing.get("aggregate_inventory_sha256") == aggregate:
            return existing
    write_immutable_json(output_path, payload)
    return payload


def create_legacy_rth_projection(
    *,
    repo_root: Path,
    inventory_path: Path,
    authority_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Materialize a diagnostic RTH projection without creating acquisition receipts."""
    repo_root = Path(repo_root).resolve()
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    if inventory.get("authoritative") is not False:
        raise ValueError("legacy projection requires an explicitly non-authoritative inventory")
    sources = [
        source for source in inventory.get("files", []) if source.get("timeframe") in {"30m", "5m"}
    ]

    work = [
        (str(repo_root), str(Path(authority_root).resolve()), inventory["snapshot_id"], source)
        for source in sources
    ]
    workers = min(4, max(2, os.cpu_count() or 2))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_project_legacy_source_worker, work))
    aggregate = canonical_json_sha256(records)
    payload = {
        "schema_version": "legacy_eth_to_rth_projection_v1",
        "projection_id": f"legacy_rth_projection_{aggregate[:24]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_snapshot_id": inventory["snapshot_id"],
        "parent_inventory_sha256": inventory["aggregate_inventory_sha256"],
        "input_session_policy": EXTENDED_SESSION_POLICY,
        "output_session_policy": RTH_SESSION_POLICY,
        "calendar_version": CALENDAR_VERSION,
        "transformation": "09:30_to_versioned_exchange_close_filter_v1",
        "authoritative": False,
        "intended_use": "comparison_with_direct_ibkr_useRTH_true_only",
        "acquisition_receipt_id": None,
        "aggregate_projection_sha256": aggregate,
        "file_count": len(records),
        "files": records,
    }
    if Path(output_path).exists():
        existing = json.loads(Path(output_path).read_text(encoding="utf-8"))
        if existing.get("aggregate_projection_sha256") == aggregate:
            return existing
    write_immutable_json(output_path, payload)
    return payload


def project_rth(frame: pd.DataFrame) -> pd.DataFrame:
    from backtests.stock.data.calendar import rth_mask

    normalized = normalize_bar_frame(frame)
    if normalized.empty:
        return normalized
    return normalized.loc[rth_mask(normalized.index).to_numpy()].copy()


def _project_legacy_source_worker(
    work: tuple[str, str, str, dict[str, Any]],
) -> dict[str, Any]:
    repo_root_text, authority_root_text, snapshot_id, source = work
    repo_root = Path(repo_root_text)
    authority_root = Path(authority_root_text)
    base = (
        authority_root
        / "derived"
        / "legacy_eth_to_rth"
        / snapshot_id
        / source["symbol"]
        / source["timeframe"]
    )
    existing_targets = list(base.glob("*/bars.parquet")) if base.exists() else []
    if len(existing_targets) > 1:
        raise ValueError(f"multiple derived immutable projections for {source['symbol']} {source['timeframe']}")
    if existing_targets:
        target = existing_targets[0]
        projected = read_bar_frame(target)
        content_sha = target.parent.name
    else:
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if sha256_file(source_path) != source.get("physical_sha256"):
            raise ValueError(f"legacy source changed after inventory: {source_path}")
        projected = project_rth(read_bar_frame(source_path))
        content_sha = normalized_content_sha256(projected)
        target = base / content_sha / "bars.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        projected.to_parquet(temporary, engine="pyarrow", index=True)
        temporary.replace(target)
    return {
        "symbol": source["symbol"],
        "timeframe": source["timeframe"],
        "source_path": source["path"],
        "source_physical_sha256": source["physical_sha256"],
        "derived_path": _relative_path(target, repo_root),
        "derived_physical_sha256": sha256_file(target),
        "derived_normalized_content_sha256": content_sha,
        "schema": normalized_schema(projected),
        "rows": len(projected),
        "observed_start": projected.index.min().isoformat() if not projected.empty else None,
        "observed_end": projected.index.max().isoformat() if not projected.empty else None,
        "authoritative": False,
        "acquisition_receipt_id": None,
    }


def git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def git_dirty_paths(repo_root: Path, paths: Iterable[Path]) -> list[str]:
    resolved_root = Path(repo_root).resolve()
    relative = [_relative_path(Path(path).resolve(), resolved_root) for path in paths]
    if not relative:
        return []
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *relative],
        cwd=resolved_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


def _relative_path(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        return os.fspath(resolved)
