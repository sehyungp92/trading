"""Re-key IARIC escape evaluations after orchestration-only runner changes.

This migration is intentionally narrow: the replay source fingerprint must be
unchanged, every cache key must carry the recorded old code fingerprint, and a
backup is written before replacement.  It is used when scoring/gating changes
but the strategy and evaluation code that produced the cached trades did not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtests.stock.auto.runners.run_iaric_escape_round3 import (
    _code_fingerprint,
)
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    _replay_source_fingerprint,
    _write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    old_code = str(payload.get("code_fingerprint", ""))
    new_code = _code_fingerprint()
    current_source = _replay_source_fingerprint()
    if str(payload.get("source_fingerprint", "")) != current_source:
        raise RuntimeError("Replay source changed; cached evaluations cannot be migrated")
    if not old_code:
        raise RuntimeError("Cache has no prior code fingerprint")
    if old_code == new_code:
        return 0

    migrated: dict[str, object] = {}
    for key, value in dict(payload.get("evaluations", {})).items():
        parts = key.split("|")
        if len(parts) != 5 or parts[1] != old_code:
            raise RuntimeError(f"Unexpected cache key namespace: {key}")
        parts[1] = new_code
        migrated["|".join(parts)] = value

    backup_path = cache_path.with_name("evaluation_cache.pre_gate_repair.json")
    _write_json(backup_path, payload)
    payload["evaluations"] = migrated
    payload["code_fingerprint"] = new_code
    payload["orchestration_only_migration"] = {
        "from": old_code,
        "to": new_code,
        "reason": "baseline-relative drawdown gate repair",
    }
    _write_json(cache_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
