from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backtests.shared.data.ibkr.bars import SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA
from backtests.shared.data.ibkr.store import write_manifest, write_parquet_atomic
from backtests.swing.data.futures_context_authority import (
    FuturesContextAuthorityError,
    promote_derived_swing_futures_context,
    validate_swing_futures_context,
)
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
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
            "schema_version": "physical_futures_panama_source_v2",
            "source_kind": SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA,
            "usable_for_authoritative_validation": True,
            "symbol": symbol,
            "timeframe": "5m",
            "physical_sha256": _sha256(base_path),
            "contract_calendar_policy": spec.calendar_policy,
            "roll_policy": spec.roll_policy,
            "adjustment_policy": "deterministic_backward_panama_v1",
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


def test_promote_context_derives_and_certifies_both_timeframes(tmp_path: Path) -> None:
    _write_single_contract_source(tmp_path, "NQ", "202603", "1001")

    report = promote_derived_swing_futures_context(tmp_path, symbols=("NQ",))

    assert report.ok
    assert (tmp_path / "NQ_1h.parquet").exists()
    assert (tmp_path / "NQ_1d.parquet").exists()
    assert (tmp_path / "NQ_futures_context.manifest.json").exists()
    assert validate_swing_futures_context(tmp_path, symbols=("NQ",)).ok


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
