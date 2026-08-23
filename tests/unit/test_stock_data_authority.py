from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backtests.shared.data.ibkr.bars import request_bars_with_retry
from backtests.shared.data.ibkr.models import BarDownloadRequest
from backtests.stock.data.authoritative_downloader import download_authoritative_stock_bars
from backtests.stock.data.authority import (
    create_legacy_snapshot_inventory,
    identity_for_request,
    write_immutable_bars,
    write_immutable_json,
    write_latest_reference,
    write_receipt,
)
from backtests.stock.data.bundle import build_frozen_bundle, verify_frozen_bundle
from backtests.stock.data.calendar import (
    EXTENDED_SESSION_POLICY,
    RTH_SESSION_POLICY,
    expected_bar_opens,
)
from backtests.stock.data.preprocessing import filter_rth
from backtests.stock.engine.research_replay import _build_date_index


def _rth_frame(day: str = "2026-05-04") -> pd.DataFrame:
    index = pd.date_range(f"{day} 13:30:00Z", f"{day} 19:55:00Z", freq="5min")
    return pd.DataFrame(
        {
            "open": range(len(index)),
            "high": [value + 1 for value in range(len(index))],
            "low": [value - 1 for value in range(len(index))],
            "close": [value + 0.5 for value in range(len(index))],
            "volume": [100] * len(index),
        },
        index=index,
    )


def test_dataset_identity_separates_rth_and_extended() -> None:
    rth = identity_for_request(
        symbol="MSFT", con_id="265598", timeframe="5m", what_to_show="TRADES", use_rth=True
    )
    extended = identity_for_request(
        symbol="MSFT", con_id="265598", timeframe="5m", what_to_show="TRADES", use_rth=False
    )
    assert rth.dataset_id != extended.dataset_id
    assert rth.identity_sha256 != extended.identity_sha256


def test_immutable_objects_create_new_versions_without_overwrite(tmp_path: Path) -> None:
    identity = identity_for_request(
        symbol="MSFT", con_id="265598", timeframe="5m", what_to_show="TRADES", use_rth=True
    )
    first = write_immutable_bars(tmp_path, identity, _rth_frame())
    second_frame = _rth_frame().copy()
    second_frame.iloc[-1, second_frame.columns.get_loc("close")] += 1
    second = write_immutable_bars(tmp_path, identity, second_frame)
    assert first["path"].exists()
    assert second["path"].exists()
    assert first["path"] != second["path"]
    assert first["content_sha256"] != second["content_sha256"]


def test_immutable_json_rejects_changed_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    write_immutable_json(path, {"value": 1})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_immutable_json(path, {"value": 2})


def test_filter_rth_starts_at_0930_and_models_early_close() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-11-29T14:00:00Z",  # 09:00 ET
            "2024-11-29T14:30:00Z",  # 09:30 ET
            "2024-11-29T17:55:00Z",  # 12:55 ET
            "2024-11-29T18:00:00Z",  # 13:00 ET early close
        ]
    )
    frame = pd.DataFrame({"close": [1, 2, 3, 4]}, index=index)
    filtered = filter_rth(frame)
    assert list(filtered["close"]) == [2, 3]


def test_replay_date_index_groups_standard_time_evening_by_et_session() -> None:
    frame = pd.DataFrame(
        {"close": [1, 2]},
        index=pd.DatetimeIndex(["2025-01-06T23:30:00Z", "2025-01-07T00:30:00Z"]),
    )
    dates, last_ilocs = _build_date_index(frame)
    assert [day.isoformat() for day in dates] == ["2025-01-06"]
    assert last_ilocs == [1]


def test_expected_rth_slots_honor_early_close() -> None:
    values = expected_bar_opens(
        datetime(2024, 11, 29, tzinfo=timezone.utc),
        datetime(2024, 11, 29, 23, 59, tzinfo=timezone.utc),
        "30m",
        RTH_SESSION_POLICY,
    )
    assert len(values) == 7
    assert values[-1] == pd.Timestamp("2024-11-29T17:30:00Z")


def test_expected_rth_slots_honor_2025_national_day_of_mourning() -> None:
    values = expected_bar_opens(
        datetime(2025, 1, 9, tzinfo=timezone.utc),
        datetime(2025, 1, 9, 23, 59, tzinfo=timezone.utc),
        "5m",
        RTH_SESSION_POLICY,
    )
    assert len(values) == 0


def test_legacy_inventory_is_explicitly_not_an_acquisition_receipt(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _rth_frame().to_parquet(data_dir / "MSFT_5m.parquet", engine="pyarrow", index=True)
    output = tmp_path / "legacy.json"
    inventory = create_legacy_snapshot_inventory(
        repo_root=tmp_path,
        data_dir=data_dir,
        output_path=output,
        label="legacy extended test",
    )
    assert inventory["authoritative"] is False
    assert inventory["files"][0]["acquisition_receipt_id"] is None
    assert inventory["files"][0]["declared_session_policy"] == EXTENDED_SESSION_POLICY


def test_frozen_bundle_verifies_bytes_receipt_identity_and_universe(tmp_path: Path) -> None:
    universe_source = tmp_path / "strategies" / "stock" / "live_universe.py"
    universe_source.parent.mkdir(parents=True)
    universe_source.write_text("BACKTESTED_INTRADAY_STOCK_SYMBOLS = ('MSFT',)\n", encoding="utf-8")
    authority_root = tmp_path / "authority"
    identity = identity_for_request(
        symbol="MSFT", con_id="265598", timeframe="5m", what_to_show="TRADES", use_rth=True
    )
    object_record = write_immutable_bars(authority_root, identity, _rth_frame())
    receipt_path, receipt = write_receipt(
        authority_root,
        identity,
        {
            "accepted": True,
            "request": {"use_rth": True},
            "resolved_contract": {"con_id": "265598"},
            "chunk_windows": [],
            "object": {
                "path": str(object_record["path"]),
                "physical_sha256": object_record["physical_sha256"],
                "normalized_content_sha256": object_record["content_sha256"],
                "normalized_schema": object_record["schema"],
            },
            "coverage": {"rows": object_record["rows"]},
            "calendar_validation": {"status": "passed"},
        },
    )
    write_latest_reference(
        authority_root,
        identity,
        object_record=object_record,
        receipt_path=receipt_path,
        receipt=receipt,
        repo_root=tmp_path,
    )
    bundle_path = tmp_path / "bundle.json"
    build_frozen_bundle(
        repo_root=tmp_path,
        authority_root=authority_root,
        output_path=bundle_path,
        intraday_symbols=["MSFT"],
        daily_symbols=["MSFT"],
        timeframes=("5m",),
        session_policy_by_timeframe={"5m": RTH_SESSION_POLICY},
        require_clean=False,
    )
    report = verify_frozen_bundle(
        bundle_path,
        repo_root=tmp_path,
        require_clean=False,
        expected_universe=["MSFT"],
        expected_session_policy_by_timeframe={"5m": RTH_SESSION_POLICY},
    )
    assert report["valid"] is True

    wrong_price_basis = verify_frozen_bundle(
        bundle_path,
        repo_root=tmp_path,
        require_clean=False,
        expected_universe=["MSFT"],
        expected_session_policy_by_timeframe={"5m": RTH_SESSION_POLICY},
        expected_what_to_show_by_timeframe={"5m": "ADJUSTED_LAST"},
    )
    assert wrong_price_basis["valid"] is False
    assert any(
        "what-to-show mismatch" in error
        for error in wrong_price_basis["errors"]
    )

    object_path = Path(object_record["path"])
    object_path.write_bytes(object_path.read_bytes() + b"tampered")
    report = verify_frozen_bundle(
        bundle_path,
        repo_root=tmp_path,
        require_clean=False,
        expected_universe=["MSFT"],
        expected_session_policy_by_timeframe={"5m": RTH_SESSION_POLICY},
    )
    assert report["valid"] is False
    assert any("physical checksum mismatch" in error for error in report["errors"])


@pytest.mark.asyncio
async def test_ibkr_request_trace_records_exact_attempt() -> None:
    class FakeIB:
        async def reqHistoricalDataAsync(self, _contract, **kwargs):
            assert kwargs["barSizeSetting"] == "5 mins"
            assert kwargs["useRTH"] is True
            assert kwargs["formatDate"] == 2
            return [object()]

    class Contract:
        conId = 265598
        symbol = "MSFT"
        lastTradeDateOrContractMonth = ""

    trace: dict[str, object] = {}
    result = await request_bars_with_retry(
        FakeIB(),
        Contract(),
        end_dt=datetime(2026, 5, 4, 20, 0, tzinfo=timezone.utc),
        duration="1 W",
        timeframe="5m",
        what_to_show="TRADES",
        use_rth=True,
        trace_window=trace,
    )
    assert len(result) == 1
    assert trace["status"] == "success"
    assert trace["attempts"] == [{"attempt": 1, "status": "success", "error": ""}]


@pytest.mark.asyncio
async def test_authoritative_downloader_writes_receipt_object_and_latest(tmp_path: Path) -> None:
    class FakeIB:
        async def qualifyContractsAsync(self, contract):
            contract.conId = 265598
            contract.localSymbol = "MSFT"
            contract.primaryExchange = "NASDAQ"
            return [contract]

        async def reqContractDetailsAsync(self, contract):
            return [
                SimpleNamespace(
                    contract=contract,
                    timeZoneId="America/New_York",
                    tradingHours="20260504:0930-1600",
                    liquidHours="20260504:0930-1600",
                )
            ]

        async def reqHistoricalDataAsync(self, _contract, **_kwargs):
            return [
                SimpleNamespace(
                    date=stamp.to_pydatetime(),
                    open=float(position),
                    high=float(position + 1),
                    low=float(position - 1),
                    close=float(position) + 0.5,
                    volume=100,
                    barCount=1,
                    average=float(position) + 0.25,
                )
                for position, stamp in enumerate(
                    pd.date_range("2026-05-04T13:30:00Z", "2026-05-04T19:55:00Z", freq="5min")
                )
            ]

    result = await download_authoritative_stock_bars(
        FakeIB(),
        BarDownloadRequest(
            symbol="MSFT",
            timeframe="5m",
            sec_type="STK",
            exchange="SMART",
            primary_exchange="NASDAQ",
            what_to_show="TRADES",
            use_rth=True,
            start=datetime(2026, 5, 4, 13, 30, tzinfo=timezone.utc),
            end=datetime(2026, 5, 4, 19, 55, tzinfo=timezone.utc),
        ),
        repo_root=tmp_path,
        authority_root=tmp_path / "authority",
    )
    assert result.metadata["accepted"] is True
    assert result.rows == 78
    assert len(list((tmp_path / "authority" / "objects").rglob("bars.parquet"))) == 1
    assert len(list((tmp_path / "authority" / "receipts").rglob("*.json"))) == 1
    latest = list((tmp_path / "authority" / "refs" / "latest").glob("*.json"))
    assert len(latest) == 1
    reference = json.loads(latest[0].read_text(encoding="utf-8"))
    assert reference["receipt_id"] == result.metadata["receipt_id"]
