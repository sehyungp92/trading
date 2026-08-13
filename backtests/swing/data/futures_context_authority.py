"""Certified NQ/GC context data for TPC replay and promotion.

The physical 5-minute Panama series is the sole price parent.  Swing 1h and
RTH-daily context bars are deterministic children of that parent; independent
higher-timeframe futures downloads are not eligible for certification.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from backtests.shared.data.ibkr.bars import SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA
from backtests.shared.data.ibkr.store import ensure_utc_index, read_parquet_if_exists, write_manifest, write_parquet_atomic
from libs.market_data.futures_roll import root_spec
from libs.market_data.panama import stitch_panama

CONTEXT_SYMBOLS = ("NQ", "GC")
BASE_TIMEFRAME = "5m"
DERIVED_TIMEFRAMES = ("1h", "1d")
SOURCE_SCHEMA = "physical_futures_panama_source_v2"
DERIVED_SCHEMA = "swing_futures_context_derived_v1"
AUTHORITY_SCHEMA = "swing_futures_context_authority_v1"
DERIVATION_POLICIES = {
    "1h": "utc_hour_right_label_from_5m_ohlcv_v1",
    "1d": "new_york_rth_0930_1600_calendar_day_from_5m_ohlcv_v1",
}
_PROVENANCE_COLUMNS = (
    "source_contract_yyyymm",
    "source_contract_local_symbol",
    "source_contract_con_id",
)
_AUTHORITY_CACHE: dict[tuple[Any, ...], AuthorityReport] = {}


class FuturesContextAuthorityError(RuntimeError):
    pass


@dataclass
class AuthorityReport:
    data_dir: Path
    symbols: tuple[str, ...]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> "AuthorityReport":
        if not self.ok:
            detail = "\n - ".join(self.errors)
            raise FuturesContextAuthorityError(
                "TPC futures-context authority failed. Run the physical Swing futures sync and "
                f"derive/certify the children before strict optimisation or promotion:\n - {detail}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORITY_SCHEMA,
            "ok": self.ok,
            "data_dir": str(self.data_dir),
            "symbols": list(self.symbols),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "artifacts": self.artifacts,
        }


def manifest_path(data_dir: Path, symbol: str, timeframe: str) -> Path:
    return Path(data_dir) / f"{symbol.upper()}_{timeframe}.manifest.json"


def authority_path(data_dir: Path, symbol: str) -> Path:
    return Path(data_dir) / f"{symbol.upper()}_futures_context.manifest.json"


def derive_context_frame(base: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    base = ensure_utc_index(base)
    timeframe = timeframe.lower()
    if timeframe == "1h":
        return _aggregate(base, "1h", label="right", closed="left")
    if timeframe == "1d":
        idx_et = base.index.tz_convert(ZoneInfo("America/New_York"))
        minutes = idx_et.hour * 60 + idx_et.minute
        rth = base.loc[(minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)]
        return _aggregate(rth, "1D", label="left", closed="left")
    raise ValueError(f"Unsupported Swing futures-context timeframe: {timeframe!r}")


def validate_swing_futures_context(
    data_dir: Path,
    *,
    symbols: Iterable[str] = CONTEXT_SYMBOLS,
    verify_derivation: bool = True,
) -> AuthorityReport:
    base_dir = Path(data_dir)
    normalized_symbols = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    report = AuthorityReport(data_dir=base_dir, symbols=normalized_symbols)
    for symbol in normalized_symbols:
        _validate_symbol(base_dir, symbol, report, verify_derivation=verify_derivation)
    return report


def require_tpc_futures_context_authority(
    data_dir: Path,
    *,
    symbols: Iterable[str] = CONTEXT_SYMBOLS,
    verify_derivation: bool = True,
) -> AuthorityReport:
    normalized_symbols = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    cache_key = _authority_cache_key(Path(data_dir), normalized_symbols, verify_derivation)
    cached = _AUTHORITY_CACHE.get(cache_key)
    if cached is not None:
        return cached.require_ok()
    report = validate_swing_futures_context(
        data_dir,
        symbols=normalized_symbols,
        verify_derivation=verify_derivation,
    )
    if report.ok:
        _AUTHORITY_CACHE[cache_key] = report
    return report.require_ok()


def promote_derived_swing_futures_context(
    data_dir: Path,
    *,
    symbols: Iterable[str] = CONTEXT_SYMBOLS,
    backup_existing: bool = True,
) -> AuthorityReport:
    """Stage, validate, and atomically promote 1h/1d children for each symbol."""
    base_dir = Path(data_dir)
    normalized_symbols = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging_root = base_dir / "_staging" / "futures_context" / run_id
    backup_root = base_dir / "_uncertified_backup" / run_id

    source_report = AuthorityReport(data_dir=base_dir, symbols=normalized_symbols)
    staged: dict[str, dict[str, Path]] = {}
    for symbol in normalized_symbols:
        source_manifest = _validate_source(base_dir, symbol, source_report)
        if source_manifest is None:
            continue
        base_path = base_dir / f"{symbol}_{BASE_TIMEFRAME}.parquet"
        base = read_parquet_if_exists(base_path)
        symbol_stage = staging_root / symbol
        staged[symbol] = {}
        for timeframe in DERIVED_TIMEFRAMES:
            derived = derive_context_frame(base, timeframe)
            target = symbol_stage / f"{symbol}_{timeframe}.parquet"
            write_parquet_atomic(derived, target)
            payload = _derived_manifest_payload(
                symbol=symbol,
                timeframe=timeframe,
                derived_path=target,
                derived=derived,
                base_path=base_path,
                base=base,
                source_manifest_path=manifest_path(base_dir, symbol, BASE_TIMEFRAME),
                source_manifest=source_manifest,
            )
            target_manifest = symbol_stage / f"{symbol}_{timeframe}.manifest.json"
            write_manifest(target_manifest, payload)
            staged[symbol][timeframe] = target

            round_trip = read_parquet_if_exists(target)
            mismatch = _frame_mismatch(derived, round_trip)
            if mismatch:
                source_report.errors.append(f"{symbol} {timeframe} staged write is not stable: {mismatch}")

    source_report.require_ok()

    for symbol, paths in staged.items():
        for timeframe, staged_path in paths.items():
            final_path = base_dir / staged_path.name
            final_manifest = manifest_path(base_dir, symbol, timeframe)
            staged_manifest = staged_path.with_name(f"{staged_path.stem}.manifest.json")
            if backup_existing:
                for existing in (final_path, final_manifest):
                    if existing.exists():
                        destination = backup_root / symbol / existing.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(existing, destination)
            _atomic_copy(staged_path, final_path)
            _atomic_copy(staged_manifest, final_manifest)

        authority_payload = {
            "schema_version": AUTHORITY_SCHEMA,
            "usable_for_authoritative_validation": True,
            "symbol": symbol,
            "base": _artifact_receipt(base_dir / f"{symbol}_{BASE_TIMEFRAME}.parquet"),
            "base_manifest": _artifact_receipt(manifest_path(base_dir, symbol, BASE_TIMEFRAME)),
            "children": {
                timeframe: {
                    "data": _artifact_receipt(base_dir / f"{symbol}_{timeframe}.parquet"),
                    "manifest": _artifact_receipt(manifest_path(base_dir, symbol, timeframe)),
                }
                for timeframe in DERIVED_TIMEFRAMES
            },
        }
        write_manifest(authority_path(base_dir, symbol), authority_payload)

    return validate_swing_futures_context(base_dir, symbols=normalized_symbols, verify_derivation=True).require_ok()


def _validate_symbol(base_dir: Path, symbol: str, report: AuthorityReport, *, verify_derivation: bool) -> None:
    source_manifest = _validate_source(base_dir, symbol, report)
    base_path = base_dir / f"{symbol}_{BASE_TIMEFRAME}.parquet"
    if source_manifest is None or not base_path.exists():
        return
    base = read_parquet_if_exists(base_path)
    if base.empty:
        report.errors.append(f"{symbol} canonical {BASE_TIMEFRAME} parent is empty: {base_path}")
        return

    authority = _load_json(authority_path(base_dir, symbol), report, f"{symbol} futures-context authority")
    if authority is not None:
        _validate_authority_receipt(base_dir, symbol, authority, report)

    artifact = report.artifacts.setdefault(symbol, {})
    artifact["base_rows"] = len(base)
    artifact["base_start"] = base.index.min().isoformat()
    artifact["base_end"] = base.index.max().isoformat()
    for timeframe in DERIVED_TIMEFRAMES:
        child_path = base_dir / f"{symbol}_{timeframe}.parquet"
        child_manifest_path = manifest_path(base_dir, symbol, timeframe)
        if not child_path.exists():
            report.errors.append(f"{symbol} missing derived {timeframe} data: {child_path}")
            continue
        child_manifest = _load_json(child_manifest_path, report, f"{symbol} {timeframe}")
        if child_manifest is None:
            continue
        _validate_derived_manifest(
            base_dir,
            symbol,
            timeframe,
            child_path,
            child_manifest,
            base_path,
            source_manifest,
            report,
        )
        child = read_parquet_if_exists(child_path)
        if child.empty:
            report.errors.append(f"{symbol} derived {timeframe} is empty: {child_path}")
            continue
        if verify_derivation:
            expected = derive_context_frame(base, timeframe)
            mismatch = _frame_mismatch(expected, child)
            if mismatch:
                report.errors.append(f"{symbol} {timeframe} is not an exact child of {BASE_TIMEFRAME}: {mismatch}")
        artifact[f"{timeframe}_rows"] = len(child)
        artifact[f"{timeframe}_sha256"] = _sha256_file(child_path)


def _validate_source(base_dir: Path, symbol: str, report: AuthorityReport) -> dict[str, Any] | None:
    base_path = base_dir / f"{symbol}_{BASE_TIMEFRAME}.parquet"
    source_manifest_path = manifest_path(base_dir, symbol, BASE_TIMEFRAME)
    if not base_path.exists():
        report.errors.append(f"{symbol} missing canonical physical {BASE_TIMEFRAME} parent: {base_path}")
        return None
    source = _load_json(source_manifest_path, report, f"{symbol} {BASE_TIMEFRAME} source")
    if source is None:
        return None
    spec = root_spec(symbol)
    required = {
        "schema_version": SOURCE_SCHEMA,
        "source_kind": SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA,
        "symbol": symbol,
        "timeframe": BASE_TIMEFRAME,
        "contract_calendar_policy": spec.calendar_policy,
        "roll_policy": spec.roll_policy,
        "adjustment_policy": "deterministic_backward_panama_v1",
    }
    for key, expected in required.items():
        if source.get(key) != expected:
            report.errors.append(
                f"{symbol} source manifest {key}={source.get(key)!r}; expected {expected!r}"
            )
    if source.get("usable_for_authoritative_validation") is not True:
        report.errors.append(f"{symbol} source manifest is not marked authoritative")
    actual_hash = _sha256_file(base_path)
    if source.get("physical_sha256") != actual_hash:
        report.errors.append(f"{symbol} canonical parent hash does not match its source manifest")

    base = read_parquet_if_exists(base_path)
    missing_columns = [column for column in _PROVENANCE_COLUMNS if column not in base.columns]
    if missing_columns:
        report.errors.append(f"{symbol} parent lacks per-bar physical lineage columns: {missing_columns}")
        return source
    used_months = set(base["source_contract_yyyymm"].dropna().astype(str).unique())
    contracts = {
        str(item.get("yyyymm", "")): item
        for item in source.get("contracts", [])
        if isinstance(item, dict) and item.get("yyyymm")
    }
    raw_frames: dict[str, pd.DataFrame] = {}
    for month in sorted(used_months):
        contract = contracts.get(month)
        if contract is None:
            report.errors.append(f"{symbol} parent uses {month} but the contract is absent from the ledger")
            continue
        raw_path = _resolve_receipt_path(base_dir, contract.get("raw_path", ""))
        if raw_path is None or not raw_path.exists():
            report.errors.append(f"{symbol} {month} physical contract evidence is missing")
            continue
        if not contract.get("con_id"):
            report.errors.append(f"{symbol} {month} contract ledger is missing the IBKR conId")
        if contract.get("raw_sha256") != _sha256_file(raw_path):
            report.errors.append(f"{symbol} {month} raw physical-contract hash mismatch")
        raw_frames[month] = read_parquet_if_exists(raw_path)
    if len(raw_frames) == len(used_months) and used_months:
        parsed_rolls: list[tuple[date, str, str]] = []
        for item in source.get("rolls", []):
            if not isinstance(item, dict):
                continue
            try:
                parsed_rolls.append(
                    (
                        date.fromisoformat(str(item["roll_date"])),
                        str(item["old_month"]),
                        str(item["new_month"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                report.errors.append(f"{symbol} source manifest contains an invalid roll record: {item!r}")
        reproduced = stitch_panama(raw_frames, parsed_rolls, tick_size=spec.tick_size)
        if not reproduced.empty:
            reproduced = reproduced.loc[(reproduced.index >= base.index.min()) & (reproduced.index <= base.index.max())]
        mismatch = _frame_mismatch(base, reproduced)
        if mismatch:
            report.errors.append(f"{symbol} parent cannot be reproduced from its physical contract chain: {mismatch}")
    return source


def _validate_derived_manifest(
    base_dir: Path,
    symbol: str,
    timeframe: str,
    child_path: Path,
    child_manifest: dict[str, Any],
    base_path: Path,
    source_manifest: dict[str, Any],
    report: AuthorityReport,
) -> None:
    expected = {
        "schema_version": DERIVED_SCHEMA,
        "source_kind": "derived_from_certified_physical_futures_panama",
        "usable_for_authoritative_validation": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "parent_timeframe": BASE_TIMEFRAME,
        "derivation_policy": DERIVATION_POLICIES[timeframe],
        "parent_physical_sha256": _sha256_file(base_path),
        "physical_sha256": _sha256_file(child_path),
        "contract_calendar_policy": source_manifest.get("contract_calendar_policy"),
        "roll_policy": source_manifest.get("roll_policy"),
        "adjustment_policy": source_manifest.get("adjustment_policy"),
    }
    for key, value in expected.items():
        if child_manifest.get(key) != value:
            report.errors.append(
                f"{symbol} {timeframe} manifest {key}={child_manifest.get(key)!r}; expected {value!r}"
            )
    source_manifest_path = manifest_path(base_dir, symbol, BASE_TIMEFRAME)
    if child_manifest.get("parent_manifest_sha256") != _sha256_file(source_manifest_path):
        report.errors.append(f"{symbol} {timeframe} parent-manifest hash mismatch")


def _validate_authority_receipt(
    base_dir: Path,
    symbol: str,
    authority: dict[str, Any],
    report: AuthorityReport,
) -> None:
    if authority.get("schema_version") != AUTHORITY_SCHEMA:
        report.errors.append(f"{symbol} authority manifest has an unsupported schema")
    if authority.get("usable_for_authoritative_validation") is not True:
        report.errors.append(f"{symbol} authority manifest is not promotable")
    if authority.get("symbol") != symbol:
        report.errors.append(f"{symbol} authority manifest symbol mismatch")
    receipts = [
        ("base", authority.get("base"), base_dir / f"{symbol}_5m.parquet"),
        ("base manifest", authority.get("base_manifest"), manifest_path(base_dir, symbol, "5m")),
    ]
    children = authority.get("children", {}) if isinstance(authority.get("children"), dict) else {}
    for timeframe in DERIVED_TIMEFRAMES:
        child = children.get(timeframe, {}) if isinstance(children.get(timeframe), dict) else {}
        receipts.extend(
            [
                (f"{timeframe} data", child.get("data"), base_dir / f"{symbol}_{timeframe}.parquet"),
                (f"{timeframe} manifest", child.get("manifest"), manifest_path(base_dir, symbol, timeframe)),
            ]
        )
    for label, receipt, expected_path in receipts:
        if not isinstance(receipt, dict):
            report.errors.append(f"{symbol} authority manifest lacks the {label} receipt")
            continue
        if receipt.get("sha256") != _sha256_file(expected_path):
            report.errors.append(f"{symbol} authority {label} hash mismatch")


def _derived_manifest_payload(
    *,
    symbol: str,
    timeframe: str,
    derived_path: Path,
    derived: pd.DataFrame,
    base_path: Path,
    base: pd.DataFrame,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": DERIVED_SCHEMA,
        "source_kind": "derived_from_certified_physical_futures_panama",
        "usable_for_authoritative_validation": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": len(derived),
        "start": derived.index.min().isoformat() if not derived.empty else None,
        "end": derived.index.max().isoformat() if not derived.empty else None,
        "physical_sha256": _sha256_file(derived_path),
        "parent_timeframe": BASE_TIMEFRAME,
        "parent_path": str(base_path),
        "parent_rows": len(base),
        "parent_physical_sha256": _sha256_file(base_path),
        "parent_manifest_path": str(source_manifest_path),
        "parent_manifest_sha256": _sha256_file(source_manifest_path),
        "derivation_policy": DERIVATION_POLICIES[timeframe],
        "contract_calendar_policy": source_manifest.get("contract_calendar_policy"),
        "roll_policy": source_manifest.get("roll_policy"),
        "adjustment_policy": source_manifest.get("adjustment_policy"),
    }


def _aggregate(frame: pd.DataFrame, rule: str, *, label: str, closed: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in frame.columns:
        agg["volume"] = "sum"
    for column in _PROVENANCE_COLUMNS:
        if column in frame.columns:
            agg[column] = "last"
    result = frame.resample(rule, label=label, closed=closed).agg(agg)
    return result.dropna(subset=[column for column in ("open", "close") if column in result.columns])


def _frame_mismatch(expected: pd.DataFrame, actual: pd.DataFrame) -> str:
    expected = ensure_utc_index(expected)
    actual = ensure_utc_index(actual)
    if not expected.index.equals(actual.index):
        return f"timestamp index differs ({len(expected)} expected vs {len(actual)} actual rows)"
    columns = [column for column in expected.columns if column in actual.columns]
    missing = sorted(set(expected.columns) - set(actual.columns))
    if missing:
        return f"missing columns {missing}"
    try:
        pd.testing.assert_frame_equal(
            expected[columns],
            actual[columns],
            check_dtype=False,
            check_freq=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        return str(exc).splitlines()[0]
    return ""


def _load_json(path: Path, report: AuthorityReport, label: str) -> dict[str, Any] | None:
    if not path.exists():
        report.errors.append(f"{label} manifest is missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        report.errors.append(f"{label} manifest is unreadable: {exc}")
        return None
    if not isinstance(payload, dict):
        report.errors.append(f"{label} manifest must contain a JSON object")
        return None
    return payload


def _resolve_receipt_path(base_dir: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return base_dir / candidate


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256_file(path), "size": path.stat().st_size if path.exists() else 0}


def _authority_cache_key(data_dir: Path, symbols: tuple[str, ...], verify_derivation: bool) -> tuple[Any, ...]:
    paths: list[Path] = []
    for symbol in symbols:
        paths.extend(
            [
                data_dir / f"{symbol}_5m.parquet",
                manifest_path(data_dir, symbol, "5m"),
                data_dir / f"{symbol}_1h.parquet",
                manifest_path(data_dir, symbol, "1h"),
                data_dir / f"{symbol}_1d.parquet",
                manifest_path(data_dir, symbol, "1d"),
                authority_path(data_dir, symbol),
            ]
        )
        paths.extend(sorted((data_dir / "_physical_contracts" / symbol).glob("**/*.parquet")))
    stats = tuple(
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        if path.exists()
        else (str(path.resolve()), -1, -1)
        for path in paths
    )
    return (str(data_dir.resolve()), symbols, verify_derivation, stats)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
