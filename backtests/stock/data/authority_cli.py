"""Commands for standalone stock-data preservation, bundles, and RTH comparison."""
from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtests.stock.data.authority import (
    DEFAULT_AUTHORITY_ROOT,
    canonical_json_sha256,
    create_legacy_snapshot_inventory,
    create_legacy_rth_projection,
    normalize_bar_frame,
    project_rth,
    read_bar_frame,
)
from backtests.stock.data.bundle import (
    FrozenBundleResolver,
    build_frozen_bundle,
    verify_frozen_bundle,
)
from backtests.stock.data.calendar import RTH_SESSION_POLICY
from backtests.stock.data.downloader import REFERENCE_SYMBOLS
from backtests.stock.data.update_intraday import IARIC_RESIDUAL_DAILY_REFERENCES
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS


REPO_ROOT = Path(__file__).resolve().parents[3]


def snapshot_legacy(args: argparse.Namespace) -> None:
    payload = create_legacy_snapshot_inventory(
        repo_root=REPO_ROOT,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        label=args.label,
    )
    print(json.dumps({key: payload[key] for key in ("snapshot_id", "file_count", "aggregate_inventory_sha256")}, indent=2))


def build_bundle(args: argparse.Namespace) -> None:
    intraday = list(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    daily = list(
        dict.fromkeys(
            [
                *intraday,
                *(
                    IARIC_RESIDUAL_DAILY_REFERENCES
                    if args.profile == "iaric-residual"
                    else REFERENCE_SYMBOLS
                ),
            ]
        )
    )
    timeframes = (
        ("1d",)
        if args.profile == "iaric-residual"
        else tuple(args.timeframes.split(","))
    )
    payload = build_frozen_bundle(
        repo_root=REPO_ROOT,
        authority_root=Path(args.authority_root),
        output_path=Path(args.output),
        intraday_symbols=intraday,
        daily_symbols=daily,
        timeframes=timeframes,
        session_policy_by_timeframe={
            timeframe: RTH_SESSION_POLICY for timeframe in timeframes
        },
        what_to_show_by_timeframe={
            timeframe: (
                "ADJUSTED_LAST"
                if args.profile == "iaric-residual" and timeframe == "1d"
                else "TRADES"
            )
            for timeframe in timeframes
        },
        require_clean=not args.allow_dirty,
    )
    print(json.dumps({key: payload[key] for key in ("bundle_id", "bundle_checksum", "accepted")}, indent=2))


def project_legacy_rth(args: argparse.Namespace) -> None:
    payload = create_legacy_rth_projection(
        repo_root=REPO_ROOT,
        inventory_path=Path(args.inventory),
        authority_root=Path(args.authority_root),
        output_path=Path(args.output),
    )
    print(
        json.dumps(
            {key: payload[key] for key in ("projection_id", "file_count", "aggregate_projection_sha256")},
            indent=2,
        )
    )


def verify_bundle(args: argparse.Namespace) -> None:
    report = verify_frozen_bundle(
        Path(args.bundle),
        repo_root=REPO_ROOT,
        require_clean=not args.allow_dirty,
        expected_universe=list(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        expected_session_policy_by_timeframe={
            "1d": RTH_SESSION_POLICY,
            "30m": RTH_SESSION_POLICY,
            "5m": RTH_SESSION_POLICY,
        },
    )
    print(json.dumps({"valid": report["valid"], "errors": report["errors"]}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def compare_rth(args: argparse.Namespace) -> None:
    resolver = FrozenBundleResolver.load(
        Path(args.bundle),
        repo_root=REPO_ROOT,
        require_clean=not args.allow_dirty,
        expected_universe=list(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        expected_session_policy_by_timeframe={"30m": RTH_SESSION_POLICY, "5m": RTH_SESSION_POLICY},
    )
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    results: list[dict[str, object]] = []
    for timeframe in ("30m", "5m"):
        for symbol in BACKTESTED_INTRADAY_STOCK_SYMBOLS:
            legacy_path = Path(args.legacy_data_dir) / f"{symbol}_{timeframe}.parquet"
            direct_path = resolver.bar_path(symbol, timeframe)
            projected = _window(project_rth(read_bar_frame(legacy_path)), start, end)
            direct = _window(read_bar_frame(direct_path), start, end)
            common = projected.index.intersection(direct.index)
            value_columns = [name for name in ("open", "high", "low", "close", "volume") if name in projected and name in direct]
            equal_values = bool(
                len(projected) == len(direct)
                and projected.index.equals(direct.index)
                and projected[value_columns].equals(direct[value_columns])
            )
            results.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "projected_rows": len(projected),
                    "direct_rows": len(direct),
                    "common_rows": len(common),
                    "projected_content_sha256": canonical_json_sha256(
                        pd.util.hash_pandas_object(projected[value_columns], index=True).astype(str).tolist()
                    ),
                    "direct_content_sha256": canonical_json_sha256(
                        pd.util.hash_pandas_object(direct[value_columns], index=True).astype(str).tolist()
                    ),
                    "exact_match": equal_values,
                }
            )
    payload = {
        "schema_version": "legacy_eth_to_direct_rth_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_checksum": resolver.bundle_checksum,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "authoritative_source": "direct_ibkr_useRTH_true",
        "derived_source": "legacy_extended_hours_deterministic_rth_projection",
        "all_exact": all(bool(item["exact_match"]) for item in results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": payload["all_exact"], "comparisons": len(results), "output": str(output)}, indent=2))


def readiness(args: argparse.Namespace) -> None:
    intraday = list(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    daily = list(dict.fromkeys([*intraday, *REFERENCE_SYMBOLS]))
    requirements = [
        *[(symbol, "1d", RTH_SESSION_POLICY) for symbol in daily],
        *[(symbol, "30m", RTH_SESSION_POLICY) for symbol in intraday],
        *[(symbol, "5m", RTH_SESSION_POLICY) for symbol in intraday],
    ]
    refs_root = Path(args.authority_root) / "refs" / "latest"
    references = []
    if refs_root.exists():
        references = [json.loads(path.read_text(encoding="utf-8")) for path in refs_root.glob("*.json")]
    available = {
        (
            str(ref.get("dataset_identity", {}).get("symbol", "")).upper(),
            str(ref.get("dataset_identity", {}).get("timeframe", "")).lower(),
            str(ref.get("dataset_identity", {}).get("session_policy", "")),
        )
        for ref in references
    }
    missing = [f"{symbol}:{timeframe}:{policy}" for symbol, timeframe, policy in requirements if (symbol, timeframe, policy) not in available]
    listener = _port_open(args.host, args.port)
    stable = {
        "schema_version": "stock_authority_readiness_v1",
        "status": "ready" if not missing and listener else "blocked",
        "ibkr_endpoint": {"host": args.host, "port": args.port, "listening": listener},
        "canonical_intraday_universe_count": len(intraday),
        "canonical_intraday_universe_sha256": canonical_json_sha256(intraday),
        "required_dataset_count": len(requirements),
        "accepted_latest_count": len(requirements) - len(missing),
        "missing_dataset_count": len(missing),
        "missing_datasets": missing,
        "legacy_snapshot_inventory": args.legacy_inventory,
        "legacy_projection_manifest": args.projection_manifest,
        "legacy_inputs_authoritative": False,
        "frozen_bundle_available": False if missing else Path(args.bundle).exists(),
        "blocking_reasons": [
            *( ["IBKR API endpoint is not listening"] if not listener else [] ),
            *( [f"{len(missing)} direct-RTH accepted datasets are missing"] if missing else [] ),
        ],
    }
    payload = {
        **stable,
        "readiness_sha256": canonical_json_sha256(stable),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "missing_dataset_count", "readiness_sha256")}, indent=2))


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(1.0)
        return client.connect_ex((host, port)) == 0


def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    normalized = normalize_bar_frame(frame)
    return normalized[(normalized.index >= start) & (normalized.index <= end)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot-legacy")
    snapshot.add_argument("--data-dir", default="backtests/stock/data/raw")
    snapshot.add_argument(
        "--output",
        default="backtests/stock/data/authority/legacy_snapshots/legacy_extended_through_2026-07-10.inventory.json",
    )
    snapshot.add_argument("--label", default="legacy extended-hours cache through 2026-07-10")
    snapshot.set_defaults(handler=snapshot_legacy)

    projection = subparsers.add_parser("project-legacy-rth")
    projection.add_argument(
        "--inventory",
        default="backtests/stock/data/authority/legacy_snapshots/legacy_extended_through_2026-07-10.inventory.json",
    )
    projection.add_argument("--authority-root", default=str(DEFAULT_AUTHORITY_ROOT))
    projection.add_argument(
        "--output",
        default="backtests/stock/data/authority/derived/legacy_eth_to_rth/legacy_extended_through_2026-07-10.projection.json",
    )
    projection.set_defaults(handler=project_legacy_rth)

    bundle = subparsers.add_parser("build-bundle")
    bundle.add_argument("--authority-root", default=str(DEFAULT_AUTHORITY_ROOT))
    bundle.add_argument("--output", required=True)
    bundle.add_argument("--timeframes", default="1d,30m,5m")
    bundle.add_argument("--profile", choices=["all", "iaric-residual"], default="all")
    bundle.add_argument("--allow-dirty", action="store_true")
    bundle.set_defaults(handler=build_bundle)

    verify = subparsers.add_parser("verify-bundle")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--allow-dirty", action="store_true")
    verify.set_defaults(handler=verify_bundle)

    compare = subparsers.add_parser("compare-rth")
    compare.add_argument("--bundle", required=True)
    compare.add_argument("--legacy-data-dir", default="backtests/stock/data/raw")
    compare.add_argument("--start", default="2026-05-01")
    compare.add_argument("--end", default="2026-06-30")
    compare.add_argument(
        "--output",
        default="backtests/stock/data/authority/comparisons/may_is_june_oos_rth_comparison.json",
    )
    compare.add_argument("--allow-dirty", action="store_true")
    compare.set_defaults(handler=compare_rth)

    ready = subparsers.add_parser("readiness")
    ready.add_argument("--authority-root", default=str(DEFAULT_AUTHORITY_ROOT))
    ready.add_argument("--host", default="127.0.0.1")
    ready.add_argument("--port", type=int, default=4002)
    ready.add_argument(
        "--legacy-inventory",
        default="backtests/stock/data/authority/legacy_snapshots/legacy_extended_through_2026-07-10.inventory.json",
    )
    ready.add_argument(
        "--projection-manifest",
        default="backtests/stock/data/authority/derived/legacy_eth_to_rth/legacy_extended_through_2026-07-10.projection.json",
    )
    ready.add_argument("--bundle", default="backtests/stock/data/authority/bundles/accepted.json")
    ready.add_argument(
        "--output",
        default="backtests/stock/data/authority/readiness/may_is_june_oos.json",
    )
    ready.set_defaults(handler=readiness)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
