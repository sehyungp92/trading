from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_tree
from backtests.shared.auto.replay_bundle import ReplayBundle

if TYPE_CHECKING:
    from backtests.stock.engine.research_replay import ResearchReplayEngine


_REPLAY_CACHE: dict[str, ReplayBundle[ResearchReplayEngine]] = {}


def load_research_replay_bundle(
    data_dir: Path,
    *,
    bundle_path: Path | None = None,
    require_bundle: bool | None = None,
) -> ReplayBundle[ResearchReplayEngine]:
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    base_dir = Path(data_dir)
    replay_kwargs: dict[str, object] = {"data_dir": base_dir}
    if bundle_path is not None:
        replay_kwargs["bundle_path"] = bundle_path
    if require_bundle is not None:
        replay_kwargs["require_bundle"] = require_bundle
    replay = ResearchReplayEngine(**replay_kwargs)
    source_fingerprint = (
        replay.data_fingerprint()
        if hasattr(replay, "data_fingerprint")
        else fingerprint_tree(base_dir, patterns=("*.parquet",))
    )
    cache_key = build_cache_key(
        "stock.research_replay_bundle",
        source_fingerprint=source_fingerprint,
        extra={"data_dir": str(base_dir.resolve())},
    )
    cached = _REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    replay.load_all_data()
    bundle = ReplayBundle(
        data=replay,
        cache_key=cache_key,
        cache_source_fingerprint=source_fingerprint,
    )
    _REPLAY_CACHE[cache_key] = bundle
    return bundle
