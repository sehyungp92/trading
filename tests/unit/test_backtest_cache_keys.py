from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtests.momentum.data.replay_cache import load_replay_bundle, replay_engine_kwargs
from backtests.shared.auto.cache_keys import (
    build_cache_key,
    fingerprint_paths,
    mutation_subset,
    stable_signature,
)
from backtests.stock.engine.research_replay import ResearchReplayEngine

UTC = timezone.utc


def test_stable_signature_is_order_insensitive_for_json_like_payloads() -> None:
    left = {"path": Path("foo/bar"), "when": datetime(2026, 4, 25, 12, 0, tzinfo=UTC), "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "when": datetime(2026, 4, 25, 12, 0, tzinfo=UTC), "path": Path("foo/bar")}

    assert stable_signature(left) == stable_signature(right)


def test_mutation_subset_extracts_exact_keys_and_prefixes() -> None:
    mutations = {
        "param_overrides.tp1": 1.5,
        "param_overrides.tp2": 2.0,
        "flags.use_filter": True,
        "risk.base_pct": 0.01,
    }

    subset = mutation_subset(
        mutations,
        exact_keys=("flags.use_filter",),
        prefixes=("param_overrides",),
    )

    assert subset == {
        "flags.use_filter": True,
        "param_overrides.tp1": 1.5,
        "param_overrides.tp2": 2.0,
    }


def test_build_cache_key_changes_when_source_fingerprint_changes() -> None:
    mutations = {"param_overrides.tp1": 1.5}

    key_a = build_cache_key("demo", source_fingerprint="abc", mutations=mutations)
    key_b = build_cache_key("demo", source_fingerprint="def", mutations=mutations)

    assert key_a != key_b


def test_load_replay_bundle_reuses_cached_preprocessing(monkeypatch, tmp_path) -> None:
    from backtests.momentum.data import cache as cache_mod
    from backtests.momentum.data import preprocessing as prep_mod

    five_min_path = tmp_path / "NQ_5m.parquet"
    daily_path = tmp_path / "NQ_1d.parquet"
    es_path = tmp_path / "ES_1d.parquet"
    for path in (five_min_path, daily_path, es_path):
        path.write_text("stub", encoding="utf-8")

    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100.0, 200.0],
        },
        index=pd.DatetimeIndex(
            [
                datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
                datetime(2026, 4, 25, 9, 35, tzinfo=UTC),
            ]
        ),
    )

    calls = {"load_bars": 0, "build_numpy_arrays": 0}

    def fake_load_bars(_path: Path) -> pd.DataFrame:
        calls["load_bars"] += 1
        return frame.copy()

    def fake_build_numpy_arrays(df: pd.DataFrame) -> dict[str, int]:
        calls["build_numpy_arrays"] += 1
        return {"rows": len(df)}

    monkeypatch.setattr(cache_mod, "load_bars", fake_load_bars)
    monkeypatch.setattr(prep_mod, "normalize_timezone", lambda df: df)
    monkeypatch.setattr(prep_mod, "filter_eth", lambda df: df)
    monkeypatch.setattr(prep_mod, "resample_5m_to_15m", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_30m", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_1h", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_4h", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_daily", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "build_numpy_arrays", fake_build_numpy_arrays)
    monkeypatch.setattr(prep_mod, "align_higher_tf_to_5m", lambda *_args: ("htf",))
    monkeypatch.setattr(prep_mod, "align_daily_to_5m", lambda *_args: ("daily",))

    first = load_replay_bundle("NQ", tmp_path, include_fifteen_min=True)
    second = load_replay_bundle("NQ", tmp_path, include_fifteen_min=True)
    third = load_replay_bundle("NQ", tmp_path, include_fifteen_min=False)

    assert first is second
    assert third is not first
    assert first["cache_source_fingerprint"] == fingerprint_paths(
        [five_min_path, daily_path, es_path],
        root=tmp_path,
    )
    assert calls["load_bars"] == 6
    assert calls["build_numpy_arrays"] == 13


def test_replay_engine_kwargs_strips_cache_metadata() -> None:
    bundle = {
        "five_min": {"rows": 10},
        "daily": {"rows": 5},
        "cache_key": "abc",
        "cache_source_fingerprint": "fp",
    }

    assert replay_engine_kwargs(bundle) == {
        "five_min": {"rows": 10},
        "daily": {"rows": 5},
    }


def test_research_replay_data_fingerprint_tracks_source_files(tmp_path) -> None:
    first = tmp_path / "AAA_1d.parquet"
    first.write_text("one", encoding="utf-8")

    replay = ResearchReplayEngine(data_dir=tmp_path)
    initial = replay.data_fingerprint()

    second = tmp_path / "BBB_5m.parquet"
    second.write_text("two", encoding="utf-8")
    replay._data_fingerprint = None

    assert replay.data_fingerprint() != initial
