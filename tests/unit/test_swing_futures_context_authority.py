from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backtests.shared.data.ibkr.bars import SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA
from backtests.shared.data.ibkr.store import write_manifest, write_parquet_atomic
from backtests.swing.data.futures_context_authority import (
    FuturesContextAuthorityError,
    _unexplained_futures_gaps,
    derive_context_frame,
    promote_derived_swing_futures_context,
    validate_swing_futures_context,
    validate_swing_futures_context_children,
)
from backtests.swing.data.replay_cache import load_tpc_replay_bundle, tpc_replay_source_artifacts
from libs.market_data.futures_roll import (
    GC_CALENDAR_POLICY,
    GC_ROLL_POLICY,
    generate_futures_contracts,
    root_spec,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_single_contract_source(data_dir: Path, symbol: str, month: str, con_id: str) -> None:
    index = pd.date_range("2026-01-05T14:30:00Z", periods=12 * 24 * 6, freq="5min")
    base = pd.DataFrame(
        {
            "open": range(len(index)),
            "high": [value + 1.0 for value in range(len(index))],
            "low": [value - 1.0 for value in range(len(index))],
            "close": [value + 0.5 for value in range(len(index))],
            "volume": [10.0] * len(index),
            "source_contract_yyyymm": [month] * len(index),
            "source_contract_local_symbol": [f"{symbol}{month}"] * len(index),
            "source_contract_con_id": [con_id] * len(index),
        },
        index=index,
    )
    raw_path = data_dir / "_physical_contracts" / symbol / month / "5m_trades.parquet"
    base_path = data_dir / f"{symbol}_5m.parquet"
    write_parquet_atomic(base, raw_path)
    write_parquet_atomic(base, base_path)
    spec = root_spec(symbol)
    write_manifest(
        data_dir / f"{symbol}_5m.manifest.json",
        {
            "schema_version": "physical_futures_panama_source_v3",
            "source_kind": SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA,
            "usable_for_authoritative_validation": True,
            "symbol": symbol,
            "timeframe": "5m",
            "physical_sha256": _sha256(base_path),
            "contract_calendar_policy": spec.calendar_policy,
            "roll_policy": spec.roll_policy,
            "adjustment_policy": "deterministic_backward_panama_v2",
            "contracts": [
                {
                    "yyyymm": month,
                    "raw_path": str(raw_path.relative_to(data_dir)),
                    "raw_sha256": _sha256(raw_path),
                    "con_id": con_id,
                }
            ],
            "rolls": [],
        },
    )


def test_gc_uses_comex_bimonthly_contract_calendar() -> None:
    contracts = generate_futures_contracts(
        "GC",
        start=date(2026, 1, 1),
        end=date(2026, 8, 1),
        include_buffer_contracts=False,
    )

    assert {contract.expiry.month for contract in contracts}.issubset({2, 4, 6, 8, 10, 12})
    assert all(contract.exchange == "COMEX" for contract in contracts)
    assert root_spec("GC").calendar_policy == GC_CALENDAR_POLICY
    assert root_spec("GC").roll_policy == GC_ROLL_POLICY
    assert root_spec("GC").tick_size == pytest.approx(0.1)
    assert root_spec("NQ").panama_min_gap_guard_points == pytest.approx(500.0)
    assert root_spec("GC").panama_min_gap_guard_points == pytest.approx(150.0)


def test_tpc_selection_artifacts_exclude_five_minute_parents(tmp_path: Path) -> None:
    artifacts = tpc_replay_source_artifacts(tmp_path)

    assert artifacts
    assert all("_5m." not in path.name for path in artifacts.values())
    assert tmp_path / "NQ_1h.manifest.json" in artifacts.values()
    assert tmp_path / "GC_futures_context.manifest.json" in artifacts.values()


def test_promote_context_derives_and_certifies_both_timeframes(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "NQ", "202603", "1001")

    report = promote_derived_swing_futures_context(tmp_path, symbols=("NQ",))

    assert report.ok
    assert (tmp_path / "NQ_1h.parquet").exists()
    assert (tmp_path / "NQ_1d.parquet").exists()
    assert (tmp_path / "NQ_futures_context.manifest.json").exists()
    assert validate_swing_futures_context(tmp_path, symbols=("NQ",)).ok


def test_promoted_hourly_context_uses_swing_start_timestamp_convention(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "NQ", "202603", "1001")
    base = pd.read_parquet(tmp_path / "NQ_5m.parquet")

    hourly = derive_context_frame(base, "1h")

    assert hourly.index[0] == pd.Timestamp("2026-01-05T14:00:00Z")
    assert hourly.index[1] == pd.Timestamp("2026-01-05T15:00:00Z")


def test_gc_daily_context_uses_comex_session_not_nq_cash_session() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-05T13:20:00Z"),  # 08:20 ET
            pd.Timestamp("2026-01-05T14:30:00Z"),  # 09:30 ET
            pd.Timestamp("2026-01-05T18:25:00Z"),  # 13:25 ET
            pd.Timestamp("2026-01-05T18:30:00Z"),  # 13:30 ET, excluded
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
            "close": [100.25, 101.25, 102.25, 103.25],
            "volume": [1, 2, 3, 4],
        },
        index=index,
    )

    daily = derive_context_frame(frame, "1d", symbol="GC")

    assert daily.iloc[0]["open"] == pytest.approx(100.0)
    assert daily.iloc[0]["close"] == pytest.approx(102.25)
    assert daily.iloc[0]["volume"] == pytest.approx(6.0)


def test_certified_children_remain_valid_without_local_parent(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "NQ", "202603", "1001")
    promote_derived_swing_futures_context(tmp_path, symbols=("NQ",))

    (tmp_path / "NQ_5m.parquet").unlink()
    (tmp_path / "NQ_5m.manifest.json").unlink()
    shutil.rmtree(tmp_path / "_physical_contracts")

    report = validate_swing_futures_context_children(tmp_path, symbols=("NQ",))

    assert report.ok, report.errors


def test_tpc_strict_loader_consumes_certified_children_without_local_parent(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "NQ", "202603", "1001")
    promote_derived_swing_futures_context(tmp_path, symbols=("NQ",))
    etf = pd.read_parquet(tmp_path / "NQ_5m.parquet")[["open", "high", "low", "close", "volume"]]
    etf.to_parquet(tmp_path / "QQQ_15m.parquet")
    derive_context_frame(etf, "1h").to_parquet(tmp_path / "QQQ_1h.parquet")
    derive_context_frame(etf, "1d").to_parquet(tmp_path / "QQQ_1d.parquet")
    (tmp_path / "NQ_5m.parquet").unlink()
    (tmp_path / "NQ_5m.manifest.json").unlink()
    shutil.rmtree(tmp_path / "_physical_contracts")

    bundle = load_tpc_replay_bundle(
        tmp_path,
        symbols=("QQQ",),
        require_context_authority=True,
    )

    assert bundle.data["QQQ"]["context_symbol"] == "NQ"
    assert bundle.data["QQQ"]["context_indicators"]


def test_tpc_loader_can_keep_etf_data_separate_from_context_authority(tmp_path: Path) -> None:
    authority_dir = tmp_path / "authority"
    etf_dir = tmp_path / "etf"
    _write_single_contract_source(authority_dir, "NQ", "202603", "1001")
    promote_derived_swing_futures_context(authority_dir, symbols=("NQ",))
    etf_dir.mkdir()
    etf = pd.read_parquet(authority_dir / "NQ_5m.parquet")[["open", "high", "low", "close", "volume"]]
    etf.to_parquet(etf_dir / "QQQ_15m.parquet")
    derive_context_frame(etf, "1h").to_parquet(etf_dir / "QQQ_1h.parquet")
    derive_context_frame(etf, "1d").to_parquet(etf_dir / "QQQ_1d.parquet")

    bundle = load_tpc_replay_bundle(
        etf_dir,
        symbols=("QQQ",),
        require_context_authority=True,
        context_data_dir=authority_dir,
    )
    artifacts = tpc_replay_source_artifacts(
        etf_dir,
        symbols=("QQQ",),
        require_context_authority=True,
        context_data_dir=authority_dir,
    )

    assert bundle.data["QQQ"]["context_indicators"]
    assert artifacts["NQ_1h"] == authority_dir / "NQ_1h.parquet"

    early = etf.iloc[[0]].copy()
    early.index = early.index - pd.Timedelta(hours=2)
    pd.concat([early, etf]).to_parquet(etf_dir / "QQQ_15m.parquet")
    with pytest.raises(FuturesContextAuthorityError, match="does not cover the requested"):
        load_tpc_replay_bundle(
            etf_dir,
            symbols=("QQQ",),
            require_context_authority=True,
            context_data_dir=authority_dir,
        )


def test_publisher_rejects_unexplained_in_session_parent_gap(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "NQ", "202603", "1001")
    base_path = tmp_path / "NQ_5m.parquet"
    raw_path = tmp_path / "_physical_contracts" / "NQ" / "202603" / "5m_trades.parquet"
    frame = pd.read_parquet(base_path)
    frame = frame.drop(frame.loc["2026-01-05T16:00:00Z":"2026-01-05T16:55:00Z"].index)
    write_parquet_atomic(frame, base_path)
    write_parquet_atomic(frame, raw_path)
    manifest_path = tmp_path / "NQ_5m.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["physical_sha256"] = _sha256(base_path)
    manifest["contracts"][0]["raw_sha256"] = _sha256(raw_path)
    write_manifest(manifest_path, manifest)

    with pytest.raises(FuturesContextAuthorityError, match="unexplained in-session gaps"):
        promote_derived_swing_futures_context(tmp_path, symbols=("NQ",))


def test_gap_gate_accepts_globex_maintenance_and_recurring_holiday_closures() -> None:
    expected_closures = (
        ("2025-08-21T20:55:00Z", "2025-08-21T22:05:00Z"),
        ("2025-11-28T19:40:00Z", "2025-11-30T23:00:00Z"),
        ("2026-01-19T19:25:00Z", "2026-01-19T23:00:00Z"),
        ("2026-04-02T20:55:00Z", "2026-04-05T22:00:00Z"),
    )

    for previous, current in expected_closures:
        index = pd.DatetimeIndex([pd.Timestamp(previous), pd.Timestamp(current)])
        assert _unexplained_futures_gaps(index) == []

    nq_closures = (
        ("2025-11-28T18:10:00Z", "2025-11-30T23:00:00Z"),
        ("2025-12-24T18:10:00Z", "2025-12-25T23:00:00Z"),
        ("2026-01-19T17:55:00Z", "2026-01-19T23:00:00Z"),
    )
    for previous, current in nq_closures:
        index = pd.DatetimeIndex([pd.Timestamp(previous), pd.Timestamp(current)])
        assert _unexplained_futures_gaps(index, symbol="NQ") == []

    weekday_hole = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-05T16:00:00Z"), pd.Timestamp("2026-01-05T17:05:00Z")]
    )
    assert len(_unexplained_futures_gaps(weekday_hole)) == 1


def test_validation_rejects_child_tampering(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "NQ", "202603", "1001")
    promote_derived_swing_futures_context(tmp_path, symbols=("NQ",))
    child_path = tmp_path / "NQ_1h.parquet"
    child = pd.read_parquet(child_path)
    child.iloc[0, child.columns.get_loc("close")] += 5.0
    write_parquet_atomic(child, child_path)

    report = validate_swing_futures_context(tmp_path, symbols=("NQ",))

    assert not report.ok
    assert any("hash" in error or "exact child" in error for error in report.errors)

    portable_report = validate_swing_futures_context_children(tmp_path, symbols=("NQ",))
    assert not portable_report.ok
    assert any("hash" in error for error in portable_report.errors)


def test_tpc_strict_loader_fails_before_using_unattested_context(tmp_path: Path) -> None:
    with pytest.raises(FuturesContextAuthorityError, match="TPC futures-context authority failed"):
        load_tpc_replay_bundle(tmp_path, require_context_authority=True)


def test_source_contract_receipt_requires_con_id(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "GC", "202602", "2002")
    manifest_path = tmp_path / "GC_5m.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contracts"][0]["con_id"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_swing_futures_context(tmp_path, symbols=("GC",))

    assert not report.ok
    assert any("conId" in error for error in report.errors)
