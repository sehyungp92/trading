"""Authoritative, standalone IBKR stock downloader.

Unlike the legacy compatibility cache, this downloader never overwrites a Parquet.
Every successful acquisition produces a content-addressed object, an immutable receipt,
and a mutable ``latest`` JSON reference scoped by the complete dataset identity.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtests.shared.data.ibkr.bars import (
    CHUNK_DURATIONS,
    bars_to_frame,
    build_generic_contract,
    duration_to_timedelta,
    plan_bar_windows,
    request_bars_with_retry,
)
from backtests.shared.data.ibkr.models import BarDownloadRequest, DownloadResult
from backtests.shared.data.ibkr.pacing import RequestPacer
from backtests.shared.data.ibkr.store import merge_frames
from backtests.stock.data.authority import (
    DEFAULT_AUTHORITY_ROOT,
    DatasetIdentity,
    identity_for_request,
    latest_reference_path,
    normalize_bar_frame,
    normalized_content_sha256,
    read_bar_frame,
    receipt_id,
    sha256_file,
    validate_bar_frame,
    write_immutable_bars,
    write_latest_reference,
    write_receipt,
)


async def download_authoritative_stock_bars(
    ib: Any,
    request: BarDownloadRequest,
    *,
    repo_root: Path,
    authority_root: Path = DEFAULT_AUTHORITY_ROOT,
    pacer: RequestPacer | None = None,
    dry_run: bool = False,
    latest_only: bool = False,
) -> DownloadResult:
    if request.sec_type.upper() not in {"STK", "IND", "INDEX"}:
        raise ValueError("authoritative stock downloader supports STK and equity index requests only")
    start = _utc(request.start or datetime.now(timezone.utc) - duration_to_timedelta(request.duration))
    end = _utc(request.end or datetime.now(timezone.utc))
    if end <= start:
        raise ValueError("authoritative acquisition end must be after start")

    contract = await build_generic_contract(ib, request)
    contract_trace = await _contract_trace(ib, contract, request)
    identity = identity_for_request(
        symbol=request.symbol,
        con_id=contract_trace["con_id"],
        timeframe=request.timeframe,
        what_to_show=request.what_to_show,
        use_rth=request.use_rth,
        adjustment_policy=request.adjustment_policy,
        provider=request.provider,
        market=request.market,
        calendar_version=request.calendar_version,
    )

    existing, parent = _load_parent(
        repo_root=repo_root,
        authority_root=authority_root,
        identity=identity,
        latest_only=latest_only,
    )
    effective_start = start
    if not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(request.timeframe, "1 D"))
        effective_start = max(start, existing.index[-1].to_pydatetime() - overlap)
    windows = plan_bar_windows(effective_start, end, request.timeframe)
    if dry_run:
        return DownloadResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            dry_run=True,
            paths=[],
            messages=[
                f"{identity.dataset_id}: {len(windows)} immutable {identity.session_policy} requests"
            ],
            metadata={
                "dataset_identity": identity.payload,
                "dataset_identity_sha256": identity.identity_sha256,
                "planned_windows": [_window_payload(window) for window in windows],
                "parent": parent,
            },
        )

    pacer = pacer or RequestPacer()
    chunks: list[pd.DataFrame] = []
    trace_windows: list[dict[str, Any]] = []
    for window in windows:
        trace = _window_payload(window)
        trace_windows.append(trace)
        bars = await request_bars_with_retry(
            ib,
            contract,
            end_dt=window.end,
            duration=window.duration,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            use_rth=request.use_rth,
            pacer=pacer,
            trace_window=trace,
        )
        if bars:
            frame = bars_to_frame(bars)
            trace["observed_start_utc"] = frame.index.min().isoformat() if not frame.empty else None
            trace["observed_end_utc"] = frame.index.max().isoformat() if not frame.empty else None
            chunks.append(frame)

    merged = merge_frames(existing, *chunks)
    if not merged.empty:
        merged = normalize_bar_frame(merged)
        merged = merged[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]
    validation = validate_bar_frame(
        merged,
        start=start,
        end=end,
        timeframe=request.timeframe,
        session_policy=identity.session_policy,
    )
    failed_windows = [
        item for item in trace_windows if item.get("status") in {"failed", "timeout", "pacing"}
    ]
    contract_errors = _contract_trace_errors(contract_trace)
    accepted = validation["status"] == "passed" and not failed_windows and not contract_errors

    object_record: dict[str, Any] | None = None
    if not merged.empty:
        object_record = write_immutable_bars(authority_root, identity, merged)
    request_payload = {
        "provider": request.provider,
        "market": request.market,
        "symbol": request.symbol,
        "sec_type": request.sec_type,
        "exchange": request.exchange,
        "primary_exchange": request.primary_exchange,
        "trading_class": request.ib_trading_class,
        "currency": request.currency,
        "timeframe": request.timeframe,
        "bar_size_setting": _bar_size(request.timeframe),
        "what_to_show": request.what_to_show,
        "use_rth": request.use_rth,
        "format_date": request.format_date,
        "duration": request.duration,
        "requested_start_utc": start.isoformat(),
        "requested_end_utc": end.isoformat(),
        "adjustment_policy": request.adjustment_policy,
        "calendar_version": request.calendar_version,
        "latest_only": latest_only,
    }
    receipt_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "acquisition_class": "ibkr_direct_historical_data",
        "accepted": accepted,
        "request": request_payload,
        "resolved_contract": contract_trace,
        "chunk_windows": trace_windows,
        "parent_dataset_content_sha256": parent.get("content_sha256", ""),
        "parent_receipt_id": parent.get("receipt_id", ""),
        "merge_policy": "parent_plus_overlap_keep_latest_timestamp_v1",
        "duplicate_resolution_policy": "timestamp_keep_last_v1",
        "object": (
            {
                "path": _relative(Path(object_record["path"]), repo_root),
                "physical_sha256": object_record["physical_sha256"],
                "normalized_content_sha256": object_record["content_sha256"],
                "normalized_schema": object_record["schema"],
                "rows": object_record["rows"],
                "content_hash_version": "normalized_bar_content_v1",
            }
            if object_record
            else None
        ),
        "coverage": {
            "observed_start_utc": merged.index.min().isoformat() if not merged.empty else None,
            "observed_end_utc": merged.index.max().isoformat() if not merged.empty else None,
            "rows": len(merged),
        },
        "calendar_validation": validation,
        "transformation_lineage": [
            {
                "operation": "normalize_ibkr_bars",
                "input": "ibkr_historical_bars",
                "output": "utc_datetime_index_ohlcv",
                "session_policy": identity.session_policy,
            }
        ],
        "blocking_reasons": [*validation["errors"], *contract_errors]
        + ([f"{len(failed_windows)} request windows failed"] if failed_windows else []),
    }
    receipt_path, receipt = write_receipt(authority_root, identity, receipt_payload)
    paths = [receipt_path]
    if object_record:
        paths.insert(0, Path(object_record["path"]))
    if accepted and object_record:
        latest = write_latest_reference(
            authority_root,
            identity,
            object_record=object_record,
            receipt_path=receipt_path,
            receipt=receipt,
            repo_root=repo_root,
        )
        paths.append(latest)

    return DownloadResult(
        symbol=request.symbol,
        timeframe=request.timeframe,
        what_to_show=request.what_to_show,
        rows=len(merged),
        start=merged.index.min().to_pydatetime() if not merged.empty else None,
        end=merged.index.max().to_pydatetime() if not merged.empty else None,
        paths=paths,
        messages=[
            f"{identity.dataset_id} {'accepted' if accepted else 'blocked'}: {len(merged)} rows"
        ],
        metadata={
            "accepted": accepted,
            "dataset_identity": identity.payload,
            "dataset_identity_sha256": identity.identity_sha256,
            "receipt_id": receipt["receipt_id"],
            "receipt_path": str(receipt_path),
            "object": receipt_payload["object"],
            "validation": validation,
        },
    )


def _load_parent(
    *,
    repo_root: Path,
    authority_root: Path,
    identity: DatasetIdentity,
    latest_only: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not latest_only:
        return pd.DataFrame(), {}
    path = latest_reference_path(authority_root, identity)
    if not path.exists():
        return pd.DataFrame(), {}
    import json

    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("dataset_identity_sha256") != identity.identity_sha256:
        raise ValueError("latest reference identity mismatch")
    object_path = Path(reference["object_path"])
    if not object_path.is_absolute():
        object_path = Path(repo_root) / object_path
    if sha256_file(object_path) != reference.get("physical_sha256"):
        raise ValueError("latest reference physical checksum mismatch")
    frame = read_bar_frame(object_path)
    if normalized_content_sha256(frame) != reference.get("content_sha256"):
        raise ValueError("latest reference normalized content checksum mismatch")
    receipt_path = Path(reference["receipt_path"])
    if not receipt_path.is_absolute():
        receipt_path = Path(repo_root) / receipt_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt_id(receipt) != receipt.get("receipt_id"):
        raise ValueError("latest reference parent receipt content hash mismatch")
    if not receipt.get("accepted") or receipt.get("receipt_id") != reference.get("receipt_id"):
        raise ValueError("latest reference does not resolve to an accepted receipt")
    return frame, reference


async def _contract_trace(ib: Any, contract: Any, request: BarDownloadRequest) -> dict[str, str]:
    detail: Any | None = None
    try:
        details = await ib.reqContractDetailsAsync(contract)
        detail = next(
            (
                item
                for item in details or []
                if str(getattr(getattr(item, "contract", item), "conId", ""))
                == str(getattr(contract, "conId", ""))
            ),
            (details or [None])[0],
        )
    except Exception:
        detail = None
    detail_contract = getattr(detail, "contract", contract) if detail is not None else contract
    return {
        "con_id": str(getattr(detail_contract, "conId", "") or getattr(contract, "conId", "")),
        "symbol": str(getattr(detail_contract, "symbol", "") or request.symbol),
        "local_symbol": str(getattr(detail_contract, "localSymbol", "") or request.symbol),
        "primary_exchange": str(
            getattr(detail_contract, "primaryExchange", "")
            or request.primary_exchange
            or request.exchange
        ),
        "exchange": str(getattr(detail_contract, "exchange", "") or request.exchange),
        "currency": str(getattr(detail_contract, "currency", "") or request.currency),
        "timezone": str(getattr(detail, "timeZoneId", "") or "America/New_York"),
        "trading_hours": str(getattr(detail, "tradingHours", "") or "not_reported"),
        "liquid_hours": str(getattr(detail, "liquidHours", "") or "not_reported"),
        "resolution_method": (
            "qualifyContractsAsync+reqContractDetailsAsync"
            if detail is not None
            else "qualifyContractsAsync+contract_details_unavailable"
        ),
        "contract_details_resolved": str(detail is not None).lower(),
    }


def _contract_trace_errors(trace: dict[str, str]) -> list[str]:
    errors: list[str] = []
    if not trace.get("con_id"):
        errors.append("resolved contract conId is missing")
    if not trace.get("local_symbol"):
        errors.append("resolved contract local symbol is missing")
    if not trace.get("primary_exchange"):
        errors.append("resolved contract primary exchange is missing")
    if trace.get("contract_details_resolved") != "true":
        errors.append("IBKR contract details were not resolved")
    if trace.get("trading_hours") == "not_reported":
        errors.append("IBKR contract trading hours were not reported")
    if trace.get("liquid_hours") == "not_reported":
        errors.append("IBKR contract liquid hours were not reported")
    return errors


def _window_payload(window: Any) -> dict[str, Any]:
    return {
        "requested_start_utc": _utc(window.start).isoformat(),
        "requested_end_utc": _utc(window.end).isoformat(),
        "duration": window.duration,
        "status": "planned",
        "row_count": 0,
        "attempts": [],
    }


def _bar_size(timeframe: str) -> str:
    return {
        "1d": "1 day",
        "daily": "1 day",
        "5m": "5 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "1h": "1 hour",
    }.get(timeframe.lower(), timeframe)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
