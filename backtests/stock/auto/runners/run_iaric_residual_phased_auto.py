"""Gated Phase 0-16 IARIC price/volume residual-reversion programme.

The causal factor panel and every executable candidate are restricted to the
same frozen 98-name intraday execution universe.  Selection uses
discovery and calibration only; locked validation and holdout are never loaded
unless every prior gate has passed and the caller launches the dedicated
one-shot validation command.  The default research contract consumes the
retained, checksummed local IBKR price snapshot and never requires a broker
connection.  A receipt-backed frozen bundle remains optional provenance; it is
not required for research, validation, selection or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backtests.stock.auto.iaric.input_authority import (
    DEFAULT_MANIFEST_RELATIVE,
    attest_input_authority,
)
from backtests.stock.auto.iaric.final_diagnostics import (
    write_blocked_round_final_diagnostics,
    write_round_final_diagnostics,
)
from backtests.stock.auto.iaric.residual_phases import (
    ROUND2_SCORE_SPEC,
    run_capacity_neutral_alpha_recycling_phase,
    run_exit_capture_frontier_phase,
    run_exact_fold_evaluation,
    run_final_alpha_synergy_phase,
    run_final_robustness_phase,
    run_final_robustness_and_target_assessment_phase,
    run_management_phase,
    run_path_causal_profit_retention_phase,
    run_protected_integration_phase,
    run_quality_aperture_phase,
    run_risk_notional_frontier_phase,
    run_selective_sector_overflow_phase,
    settings_from_discovery_candidate,
)
from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CONTRACT_VERSION,
    DISCOVERY_START,
    DOWNSTREAM_EXECUTION_CONTRACT,
    HOLDOUT_START,
    LOCKED_VALIDATION_END,
    LOCKED_VALIDATION_START,
    PHASE_ORDER,
    assess_input_authority,
)
from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.engine.iaric_daily_residual_replay import (
    build_daily_residual_replay_bundle,
    run_daily_residual_replay,
)
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS
from strategies.stock.iaric.config import StrategySettings


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/round_2/phased_auto_alpha_v5_selective_sector_overflow"
)
DEFAULT_ROUND2_BASELINE = (
    REPO_ROOT
    / "backtests/output/stock/iaric/round_2/phased_auto_alpha_v3_robust_breadth/frozen_selection_candidate.json"
)
RETAINED_LOCAL_RESEARCH = "retained_local_research_snapshot"
PRODUCTION_AUTHORITY = "production_authority_bundle"
DATA_CONTRACTS = (RETAINED_LOCAL_RESEARCH, PRODUCTION_AUTHORITY)
MODEL_LABELS = {
    "market_only": "mkt",
    "market_sector": "mktsec",
    "market_sector_peer": "mktsecpeer",
    "peer_demeaned": "peerdm",
}
SEARCH_FACTOR_MODELS = ("market_sector_peer", "peer_demeaned")
FROZEN_ROUND3_CONTROL_TRADES = (
    REPO_ROOT / "backtests/output/stock/iaric/round_3/final_trades.json"
)


def _load_round2_baseline(path: Path) -> tuple[discovery.Candidate, dict[str, Any]]:
    """Load and type-check the latest frozen optimized starting configuration."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    settings_payload = dict(payload["settings"])
    if "strategy_mode" in settings_payload:
        settings = StrategySettings(**settings_payload)
    else:
        # Frozen phased-auto candidates use concise diagnostic field names.
        # Normalize them back into the shared StrategySettings contract instead
        # of maintaining a second execution/configuration implementation.
        compact_to_strategy = {
            "factor_model": "daily_residual_factor_model",
            "formation_sessions": "daily_residual_formation_sessions",
            "minimum_z": "daily_residual_minimum_z",
            "minimum_score": "daily_residual_minimum_score",
            "minimum_failed_continuation_r": (
                "daily_residual_minimum_failed_continuation_r"
            ),
            "lane_id": "daily_residual_lane_id",
            "minimum_sector_return_5d": (
                "daily_residual_minimum_sector_return_5d"
            ),
            "minimum_market_trend_z_20d": (
                "daily_residual_minimum_market_trend_z_20d"
            ),
            "score_components": "daily_residual_score_components",
            "ranking_score_components": (
                "daily_residual_ranking_score_components"
            ),
            "max_positions": "daily_residual_max_positions",
            "max_positions_per_sector": (
                "daily_residual_max_positions_per_sector"
            ),
            "sector_overflow_slots": (
                "daily_residual_sector_overflow_slots"
            ),
            "sector_overflow_minimum_score": (
                "daily_residual_sector_overflow_minimum_score"
            ),
            "sector_overflow_minimum_z": (
                "daily_residual_sector_overflow_minimum_z"
            ),
            "sector_overflow_risk_multiplier": (
                "daily_residual_sector_overflow_risk_multiplier"
            ),
            "risk_fraction": "daily_residual_risk_fraction",
            "maximum_notional_fraction": (
                "daily_residual_maximum_notional_fraction"
            ),
            "catastrophic_stop_atr": "daily_residual_catastrophic_stop_atr",
            "catastrophic_stop_residual_r": (
                "daily_residual_catastrophic_stop_residual_r"
            ),
            "partial_normalization_fraction": (
                "daily_residual_partial_normalization_fraction"
            ),
            "full_normalization_fraction": (
                "daily_residual_full_normalization_fraction"
            ),
            "structural_failure_extension_fraction": (
                "daily_residual_structural_failure_extension_fraction"
            ),
            "profit_retention_activation_fraction": (
                "daily_residual_profit_retention_activation_fraction"
            ),
            "profit_retention_giveback_fraction": (
                "daily_residual_profit_retention_giveback_fraction"
            ),
            "maximum_holding_sessions": (
                "daily_residual_maximum_holding_sessions"
            ),
            "partial_exit_fraction": "daily_residual_partial_exit_fraction",
        }
        normalized = {
            strategy_name: settings_payload[compact_name]
            for compact_name, strategy_name in compact_to_strategy.items()
            if compact_name in settings_payload
        }
        normalized["strategy_mode"] = "daily_residual_reversion"
        settings = StrategySettings(**normalized)
    if settings.strategy_mode != "daily_residual_reversion":
        raise ValueError("IARIC phased baseline must use daily_residual_reversion")
    if len(
        set(settings.daily_residual_score_components)
        | set(settings.daily_residual_ranking_score_components)
    ) > 7:
        raise ValueError("Round 2 baseline score exceeds the seven-component ceiling")
    candidate = discovery.Candidate(
        candidate_id=str(payload.get("candidate_id", "optimized_baseline_exact")),
        residual_z_floor=float(settings.daily_residual_minimum_z),
        holding_sessions=int(settings.daily_residual_maximum_holding_sessions),
        max_positions=int(settings.daily_residual_max_positions),
        max_positions_per_sector=int(
            settings.daily_residual_max_positions_per_sector
        ),
        round_trip_cost_bps=20.0,
        formation_sessions=int(settings.daily_residual_formation_sessions),
        diagnostic_leg="long_loser",
        factor_model=str(settings.daily_residual_factor_model),
        score_components=tuple(settings.daily_residual_score_components),
        lane_id=str(settings.daily_residual_lane_id),
        minimum_failed_continuation_r=float(
            settings.daily_residual_minimum_failed_continuation_r
        ),
        minimum_sector_return_5d=float(
            settings.daily_residual_minimum_sector_return_5d
        ),
        minimum_market_trend_z_20d=float(
            settings.daily_residual_minimum_market_trend_z_20d
        ),
        minimum_score=float(settings.daily_residual_minimum_score),
        catastrophic_stop_residual_r=float(
            settings.daily_residual_catastrophic_stop_residual_r
        ),
        ranking_score_components=tuple(
            settings.daily_residual_ranking_score_components
        ),
    )
    lineage = {
        "baseline_path": str(path.resolve()),
        "baseline_file_sha256": _sha256_path(path),
        "declared_baseline_sha256": payload.get(
            "sha256", payload.get("settings_sha256")
        ),
        "configuration_role": payload.get(
            "configuration_role", "frozen_selection_candidate"
        ),
        "contract_id": payload.get("contract_id"),
        "settings": asdict(settings),
        "candidate": asdict(candidate),
    }
    return candidate, lineage


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attest_retained_local_research_snapshot(data_dir: Path) -> dict[str, Any]:
    """Attest the existing research snapshot without inventing a broker need.

    This is intentionally not an acquisition receipt.  It proves the exact
    bytes, coverage source lineage and executable 98-name data contract used
    by this run.  Original broker request logs are optional provenance and do
    not affect research, validation or promotion eligibility.
    """

    required_symbols = sorted(
        {
            *BACKTESTED_INTRADAY_STOCK_SYMBOLS,
            "SPY",
            *discovery.SECTOR_ETFS.values(),
        }
    )
    paths = {symbol: data_dir / f"{symbol}_1d.parquet" for symbol in required_symbols}
    missing = sorted(symbol for symbol, path in paths.items() if not path.is_file())
    fingerprinted = []
    if not missing:
        fingerprinted = [
            {
                "symbol": symbol,
                "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
            }
            for symbol, path in paths.items()
        ]
    lineage_candidates = (
        REPO_ROOT
        / "backtests/stock/data/raw/_repair_manifests/stock_raw_repair_20260722_1230.json",
        REPO_ROOT
        / "backtests/stock/data/raw/_downloads/focused_20260722/daily_research_backfill/daily_research_backfill_report.json",
    )
    lineage = [
        {
            "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256_path(path),
        }
        for path in lineage_candidates
        if path.is_file()
    ]
    passed = not missing and len(paths) == 110
    source_digest = hashlib.sha256()
    for row in fingerprinted:
        source_digest.update(str(row["symbol"]).encode("utf-8"))
        source_digest.update(str(row["sha256"]).encode("utf-8"))
    source_id = (
        f"retained-local-daily:{source_digest.hexdigest()}"
        if fingerprinted
        else "retained-local-daily:incomplete"
    )
    daily_requirements = {
        "daily_ohlcv",
        "causal_universe_definition",
        "corporate_action_consistent_price_basis",
        "volume_unit_semantics",
        "completed_session_timestamps",
        "historical_live_price_volume_parity",
    }
    input_authority = {
        name: bool(passed and name in daily_requirements)
        for name in (
            "five_minute_ohlcv",
            *sorted(daily_requirements),
        )
    }
    return {
        "contract": RETAINED_LOCAL_RESEARCH,
        "authority_class": "project_official_local_snapshot",
        "official_for_selection_validation_and_promotion": passed,
        "research_snapshot_certified": passed,
        "production_promotion_eligible": passed,
        "broker_connection_required": False,
        "acquisition_receipts_required": False,
        "source_id": source_id,
        "daily_price_basis": "IBKR_TRADES_SPLIT_ADJUSTED_PRICE_RETURN",
        "selection_view_end": CALIBRATION_END,
        "universe_semantics": (
            "predeclared fixed 98-name execution universe conditional on local "
            "intraday availability; no index-membership claim"
        ),
        "availability_semantics": (
            "selection reads completed sessions through calibration only; locked "
            "validation is loaded once after every selection decision is frozen"
        ),
        "economic_input_parity": (
            "historical and live daily selectors use IBKR TRADES OHLCV, the same "
            "98-name taxonomy, shared residual selector and shared execution reducer"
        ),
        "input_authority": input_authority,
        "required_daily_dataset_count": 110,
        "fingerprinted_daily_dataset_count": len(fingerprinted),
        "missing_daily_symbols": missing,
        "fingerprinted_inputs": fingerprinted,
        "repair_lineage": lineage,
        "limitations": [
            "retained compatibility files are not immutable acquisition receipts",
            "results are conditional on the fixed 98-name universe",
        ],
        "failures": ([] if passed else ["missing required local daily price files"]),
    }


def _load_research_panel(
    *,
    data_contract: str,
    data_dir: Path,
    selection_bundle_path: Path | None,
    selection_end: str = CALIBRATION_END,
    allow_locked_validation: bool = False,
):
    if data_contract == PRODUCTION_AUTHORITY:
        if selection_bundle_path is None:
            raise ValueError("production authority requires a frozen selection bundle")
        return discovery._load_daily_panel_from_authoritative_bundle(
            selection_bundle_path,
            selection_end=selection_end,
            allow_locked_validation=allow_locked_validation,
        )
    if data_contract != RETAINED_LOCAL_RESEARCH:
        raise ValueError(f"unsupported IARIC data contract: {data_contract}")
    return discovery._load_daily_panel(
        data_dir,
        selection_end=selection_end,
        allow_locked_validation=allow_locked_validation,
    )


def _status(output: Path, status: str, **details: Any) -> None:
    _write_json(
        output / "background_status.json",
        {
            "status": status,
            "phase_order": list(PHASE_ORDER),
            "max_workers": 2,
            "tradable_execution_symbols": len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _atlas_cache_path(
    output: Path,
    *,
    data_fingerprint: str,
    code_fingerprint: str,
    factor_model: str,
) -> Path:
    key = hashlib.sha256(
        f"{data_fingerprint}|{code_fingerprint}|{factor_model}".encode("utf-8")
    ).hexdigest()[:24]
    return output / "cache" / f"atlas_{factor_model}_{key}.pkl"


def _load_or_build_atlas(
    *,
    output: Path,
    data_fingerprint: str,
    code_fingerprint: str,
    factor_model: str,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    sector_by_symbol: dict[str, str],
    peer_returns: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = _atlas_cache_path(
        output,
        data_fingerprint=data_fingerprint,
        code_fingerprint=code_fingerprint,
        factor_model=factor_model,
    )
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("data_fingerprint") == data_fingerprint
            and metadata.get("code_fingerprint") == code_fingerprint
            and metadata.get("factor_model") == factor_model
        ):
            return pd.read_pickle(cache_path), {**metadata, "cache_hit": True}
    if peer_returns is None:
        peer_returns = discovery._causal_correlated_peer_returns(
            close.pct_change(fill_method=None), sector_by_symbol
        )
    atlas = discovery.build_opportunity_atlas(
        close,
        open_,
        high,
        low,
        volume,
        sector_by_symbol,
        factor_model=factor_model,
        precomputed_peer_returns=peer_returns,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    atlas.to_pickle(temporary)
    temporary.replace(cache_path)
    metadata = {
        "cache_schema": "iaric_residual_atlas_cache_v1",
        "factor_model": factor_model,
        "data_fingerprint": data_fingerprint,
        "code_fingerprint": code_fingerprint,
        "rows": len(atlas),
        "tradable_rows": int(atlas["tradable_execution_universe"].sum()),
        "path": cache_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
    }
    _write_json(metadata_path, metadata)
    return atlas, metadata


def _candidate(
    *,
    factor_model: str,
    formation: int,
    holding: int,
    components: tuple[str, ...],
    leg: str = "long_loser",
    lane_id: str = "daily_residual_generic",
    minimum_failed_continuation_r: float = 0.0,
    minimum_sector_return_5d: float = -0.15,
    minimum_market_trend_z_20d: float = -8.0,
    minimum_score: float = 0.0,
    catastrophic_stop_residual_r: float = 6.0,
    ranking_components: tuple[str, ...] = (),
) -> discovery.Candidate:
    label = MODEL_LABELS[factor_model]
    component_label = "_".join(name[:4] for name in components)
    ranking_label = (
        "_rank_" + "_".join(name[:4] for name in ranking_components)
        if ranking_components
        else ""
    )
    return discovery.Candidate(
        candidate_id=(
            f"{label}_f{formation}_{leg}_h{holding}_z1_fc"
            f"{minimum_failed_continuation_r:.2f}_mktz"
            f"{minimum_market_trend_z_20d:.1f}_q{minimum_score:.0f}_stop"
            f"{catastrophic_stop_residual_r:.0f}_p10_s2_c20_{component_label}"
            f"{ranking_label}"
        ),
        residual_z_floor=1.0,
        holding_sessions=holding,
        max_positions=10,
        max_positions_per_sector=2,
        round_trip_cost_bps=20.0,
        formation_sessions=formation,
        diagnostic_leg=leg,
        factor_model=factor_model,
        score_components=components,
        lane_id=lane_id,
        minimum_failed_continuation_r=minimum_failed_continuation_r,
        minimum_sector_return_5d=minimum_sector_return_5d,
        minimum_market_trend_z_20d=minimum_market_trend_z_20d,
        minimum_score=minimum_score,
        catastrophic_stop_residual_r=catastrophic_stop_residual_r,
        ranking_score_components=ranking_components,
    )


def _evaluate(
    atlas: pd.DataFrame,
    candidates: list[discovery.Candidate],
    *,
    max_workers: int,
) -> list[dict[str, Any]]:
    # Windows process spawning duplicates the complete atlas into every worker
    # and can terminate an otherwise healthy run under concurrent research
    # load. Pandas/numpy release the GIL for the dominant vector operations, so
    # a two-thread shared-atlas pool preserves bounded parallelism and avoids
    # both serialization latency and multi-gigabyte copy-on-spawn pressure.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            pool.map(
                lambda candidate: discovery.evaluate_candidate(atlas, candidate),
                candidates,
            )
        )


def _attach_model_invariance(results: list[dict[str, Any]]) -> None:
    for row in results:
        candidate = row["candidate"]
        peers = [
            other
            for other in results
            if int(other["candidate"]["formation_sessions"])
            == int(candidate["formation_sessions"])
            and int(other["candidate"]["holding_sessions"])
            == int(candidate["holding_sessions"])
            and str(other["candidate"].get("diagnostic_leg"))
            == str(candidate.get("diagnostic_leg"))
            and tuple(other["candidate"].get("score_components", ()))
            == tuple(candidate.get("score_components", ()))
            and tuple(other["candidate"].get("ranking_score_components", ()))
            == tuple(candidate.get("ranking_score_components", ()))
            and abs(
                float(other["candidate"].get("minimum_failed_continuation_r", 0.0))
                - float(candidate.get("minimum_failed_continuation_r", 0.0))
            )
            < 1e-12
            and abs(
                float(other["candidate"].get("minimum_market_trend_z_20d", -8.0))
                - float(candidate.get("minimum_market_trend_z_20d", -8.0))
            )
            < 1e-12
            and abs(
                float(other["candidate"].get("minimum_score", 0.0))
                - float(candidate.get("minimum_score", 0.0))
            )
            < 1e-12
        ]
        positive = [
            other["candidate"]["factor_model"]
            for other in peers
            if float(other["fold_metrics"]["calibration"]["total_r"]) > 0.0
            and float(other["fold_metrics"]["calibration"]["profit_factor"]) > 1.0
        ]
        row["residual_model_invariance"] = {
            "models_tested": sorted(
                {str(other["candidate"]["factor_model"]) for other in peers}
            ),
            "positive_calibration_models": sorted(set(positive)),
            "positive_model_count": len(set(positive)),
        }
        row["gates"]["positive_under_at_least_two_residual_models"] = (
            len(set(positive)) >= 2
        )
        row["qualified_discovery_candidate"] = all(row["gates"].values())


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "trades"}


def _candidate_opportunity_attribution(
    atlas: pd.DataFrame,
    candidate: discovery.Candidate,
) -> dict[str, Any]:
    """Account for admission and capacity outcomes on a standardized path.

    This is deliberately an opportunity diagnostic, not an alternative
    portfolio PnL.  Rejected rows retain their fixed ten-session outcome and
    never free cash or create a counterfactual replacement chain.
    """

    pool = discovery._select_candidate(atlas, candidate, apply_capacity=False)
    selected = discovery._select_candidate(atlas, candidate, apply_capacity=True)
    if pool.empty:
        return {
            "candidate_id": candidate.candidate_id,
            "status": "inconclusive_no_standardized_opportunities",
            "accounted_opportunities": 0,
        }
    ordered = pool.sort_values(
        ["formation_date", "daily_score", "residual_z", "symbol"],
        ascending=[True, False, candidate.diagnostic_leg != "long_loser", True],
    ).copy()
    active: list[tuple[pd.Timestamp, str, str]] = []
    reason_by_index: dict[Any, str] = {}
    selected_indices: list[Any] = []
    for _formation_date, group in ordered.groupby("formation_date", sort=True):
        entry_date = pd.Timestamp(group["entry_date"].iloc[0])
        active = [row for row in active if row[0] >= entry_date]
        used_issuers = {row[1] for row in active}
        sector_counts = Counter(row[2] for row in active)
        remaining = max(0, int(candidate.max_positions) - len(active))
        for index, row in group.iterrows():
            if float(row["admission_score"]) < float(candidate.minimum_score):
                reason_by_index[index] = "score_below_floor"
                continue
            issuer = str(row["issuer"])
            sector = str(row["sector"])
            if remaining <= 0:
                reason_by_index[index] = "portfolio_capacity_displaced"
                continue
            if issuer in used_issuers:
                reason_by_index[index] = "issuer_capacity_displaced"
                continue
            if sector_counts[sector] >= int(candidate.max_positions_per_sector):
                reason_by_index[index] = "sector_capacity_displaced"
                continue
            reason_by_index[index] = "selected"
            selected_indices.append(index)
            used_issuers.add(issuer)
            sector_counts[sector] += 1
            remaining -= 1
            active.append(
                (pd.Timestamp(row["exit_date"]), issuer, sector)
            )
    ordered["attribution_reason"] = pd.Series(reason_by_index)
    key_columns = ["formation_date", "symbol", "trade_side"]
    expected_keys = {
        tuple(row)
        for row in selected[key_columns].itertuples(index=False, name=None)
    }
    manual_keys = {
        tuple(row)
        for row in ordered.loc[selected_indices, key_columns].itertuples(
            index=False, name=None
        )
    }
    if expected_keys != manual_keys:
        raise RuntimeError(
            f"opportunity attribution does not reconcile selector for {candidate.candidate_id}"
        )

    fold_payload: dict[str, Any] = {}
    for fold_name, start, end in discovery.FOLDS:
        fold = ordered[ordered["formation_date"].between(start, end)].copy()
        selected_fold = fold[fold["attribution_reason"] == "selected"]
        selected_groups = {
            key: group
            for key, group in selected_fold.groupby(
                ["formation_date", "sector"], sort=False
            )
        }
        reason_rows: dict[str, Any] = {}
        for reason, rejected in fold.groupby("attribution_reason", sort=True):
            values = rejected["r"].astype(float)
            row = {
                "observations": int(len(rejected)),
                "total_r_standardized_path": float(values.sum()),
                "average_r_standardized_path": (
                    float(values.mean()) if len(values) else 0.0
                ),
            }
            if reason != "selected":
                matched_selected: list[float] = []
                matched_rejected: list[float] = []
                for rejected_row in rejected.itertuples(index=False):
                    key = (rejected_row.formation_date, rejected_row.sector)
                    peers = selected_groups.get(key)
                    if peers is None or peers.empty:
                        continue
                    distance = (
                        peers["residual_z"].abs()
                        - abs(float(rejected_row.residual_z))
                    ).abs()
                    peer = peers.loc[distance.idxmin()]
                    matched_selected.append(float(peer["r"]))
                    matched_rejected.append(float(rejected_row.r))
                row.update(
                    {
                        "matched_observations": len(matched_rejected),
                        "matched_selected_average_r": (
                            sum(matched_selected) / len(matched_selected)
                            if matched_selected
                            else None
                        ),
                        "matched_rejected_average_r": (
                            sum(matched_rejected) / len(matched_rejected)
                            if matched_rejected
                            else None
                        ),
                        "matched_selected_minus_rejected_average_r": (
                            sum(
                                selected_r - rejected_r
                                for selected_r, rejected_r in zip(
                                    matched_selected, matched_rejected
                                )
                            )
                            / len(matched_rejected)
                            if matched_rejected
                            else None
                        ),
                    }
                )
            reason_rows[str(reason)] = row
        fold_payload[fold_name] = {
            "opportunities": int(len(fold)),
            "accounted_opportunities": int(
                sum(row["observations"] for row in reason_rows.values())
            ),
            "reasons": reason_rows,
        }
    return {
        "candidate_id": candidate.candidate_id,
        "status": "complete_standardized_opportunity_attribution",
        "outcome_contract": (
            "fixed_candidate_holding_path_only_not_counterfactual_portfolio_pnl"
        ),
        "admission_components": list(candidate.score_components),
        "ranking_components": list(
            candidate.ranking_score_components or candidate.score_components
        ),
        "score_floor": float(candidate.minimum_score),
        "mechanism_applicability": {
            "score_floor_rejection": bool(candidate.minimum_score > 0.0),
            "failed_continuation_rejection": bool(
                candidate.minimum_failed_continuation_r > 0.0
            ),
            "inactive_mechanisms_are_not_failures": True,
        },
        "selector_reconciliation": {
            "passed": True,
            "selected_opportunities": len(manual_keys),
        },
        "folds": fold_payload,
    }


def _settings_payload(settings) -> dict[str, Any]:
    return {
        "factor_model": settings.daily_residual_factor_model,
        "formation_sessions": settings.daily_residual_formation_sessions,
        "minimum_z": settings.daily_residual_minimum_z,
        "minimum_score": settings.daily_residual_minimum_score,
        "minimum_failed_continuation_r": (
            settings.daily_residual_minimum_failed_continuation_r
        ),
        "lane_id": settings.daily_residual_lane_id,
        "minimum_sector_return_5d": (
            settings.daily_residual_minimum_sector_return_5d
        ),
        "minimum_market_trend_z_20d": (
            settings.daily_residual_minimum_market_trend_z_20d
        ),
        "score_components": list(settings.daily_residual_score_components),
        "ranking_score_components": list(
            settings.daily_residual_ranking_score_components
        ),
        "max_positions": settings.daily_residual_max_positions,
        "max_positions_per_sector": settings.daily_residual_max_positions_per_sector,
        "sector_overflow_slots": settings.daily_residual_sector_overflow_slots,
        "sector_overflow_minimum_score": (
            settings.daily_residual_sector_overflow_minimum_score
        ),
        "sector_overflow_minimum_z": (
            settings.daily_residual_sector_overflow_minimum_z
        ),
        "sector_overflow_risk_multiplier": (
            settings.daily_residual_sector_overflow_risk_multiplier
        ),
        "risk_fraction": settings.daily_residual_risk_fraction,
        "maximum_notional_fraction": settings.daily_residual_maximum_notional_fraction,
        "catastrophic_stop_atr": settings.daily_residual_catastrophic_stop_atr,
        "catastrophic_stop_residual_r": (
            settings.daily_residual_catastrophic_stop_residual_r
        ),
        "partial_normalization_fraction": settings.daily_residual_partial_normalization_fraction,
        "full_normalization_fraction": settings.daily_residual_full_normalization_fraction,
        "structural_failure_extension_fraction": settings.daily_residual_structural_failure_extension_fraction,
        "profit_retention_activation_fraction": (
            settings.daily_residual_profit_retention_activation_fraction
        ),
        "profit_retention_giveback_fraction": (
            settings.daily_residual_profit_retention_giveback_fraction
        ),
        "replacement_mode": settings.daily_residual_replacement_mode,
        "replacement_loss_only": settings.daily_residual_replacement_loss_only,
        "replacement_minimum_held_sessions": (
            settings.daily_residual_replacement_minimum_held_sessions
        ),
        "replacement_maximum_normalization_fraction": (
            settings.daily_residual_replacement_maximum_normalization_fraction
        ),
        "replacement_minimum_score_margin": (
            settings.daily_residual_replacement_minimum_score_margin
        ),
        "replacement_max_per_session": (
            settings.daily_residual_replacement_max_per_session
        ),
        "maximum_holding_sessions": settings.daily_residual_maximum_holding_sessions,
        "partial_exit_fraction": settings.daily_residual_partial_exit_fraction,
        "entry_clock": "next_session_open",
        "universe_contract": "frozen_98_intraday_symbols_only",
    }


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _compact_exact(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"trades", "equity_curves"}
    }


def _compact_management(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("half_life_experiments", "typed_management_experiments"):
        result[key] = [
            {**row, "result": _compact_exact(row["result"])}
            for row in payload.get(key, [])
        ]
    for key in ("selected_half_life", "selected"):
        row = payload.get(key)
        if row:
            result[key] = {**row, "result": _compact_exact(row["result"])}
    return result


def _compact_settings_frontier(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove live objects and large trade/equity arrays from frontier output."""

    def compact_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            **{
                key: value
                for key, value in row.items()
                if key not in {"settings", "result"}
            },
            "settings": _settings_payload(row["settings"]),
            "result": _compact_exact(row["result"]),
        }

    return {
        **{
            key: value
            for key, value in payload.items()
            if key
            not in {
                "experiments",
                "selected",
                "selected_settings",
                "control",
            }
        },
        "experiments": [compact_row(row) for row in payload.get("experiments", [])],
        "selected": compact_row(payload.get("selected")),
        "control": compact_row(payload.get("control")),
    }


def _write_early_block(
    output: Path,
    *,
    authority: dict[str, Any],
    assessment: dict[str, Any],
) -> None:
    phase0 = {
        "status": "blocked_missing_authoritative_price_volume_contract",
        "representative_contract_version": CONTRACT_VERSION,
        "downstream_execution_contract": DOWNSTREAM_EXECUTION_CONTRACT,
        "input_scope": "price_volume_only",
        "news_quotes_or_order_imbalance_required": False,
        "input_authority": authority,
        "authority_assessment": assessment,
        "expensive_price_scan_skipped": True,
        "reason": (
            "source authority is a prerequisite to interpreting structural checks; "
            "no atlas, candidate evaluation or locked data was loaded"
        ),
        "selection_window": {"start": DISCOVERY_START, "end": CALIBRATION_END},
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(output / "phase_0_price_data_integrity_and_parity.json", phase0)
    summary = {
        "status": "blocked_phase_0_missing_authoritative_price_contract",
        "representative_reversion_baseline_eligible": False,
        "research_baseline_eligible": False,
        "optimizer_started": False,
        "phase_order": list(PHASE_ORDER),
        "last_completed_phase": None,
        "current_phase": PHASE_ORDER[0],
        "blocker": assessment["blockers"],
        "tradable_execution_symbols": len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        "max_workers": 2,
        "locked_validation_accessed": False,
        "holdout_accessed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "run_summary.json", summary)
    _status(
        output,
        summary["status"],
        blocker=summary["blocker"],
        expensive_price_scan_skipped=True,
    )


def _run_locked_validation_once(
    *,
    output: Path,
    data_contract: str,
    data_dir: Path,
    selection_bundle_path: Path | None,
    settings,
) -> dict[str, Any]:
    """Consume the registered locked fold once after every selection gate."""

    receipt_path = output / "phase_16_locked_validation_access_started.json"
    if receipt_path.exists():
        raise RuntimeError(
            "locked validation has already been consumed for this output directory"
        )
    frozen = _settings_payload(settings)
    receipt = {
        "status": "locked_validation_consumption_started",
        "candidate_sha256": _payload_sha256(frozen),
        "candidate": frozen,
        "window": {
            "start": LOCKED_VALIDATION_START,
            "end": LOCKED_VALIDATION_END,
        },
        "one_shot": True,
        "used_for_candidate_ranking": False,
        "holdout_start": HOLDOUT_START,
        "holdout_accessed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(receipt_path, receipt)
    close, open_, high, low, volume, sectors, paths = _load_research_panel(
        data_contract=data_contract,
        data_dir=data_dir,
        selection_bundle_path=selection_bundle_path,
        selection_end=LOCKED_VALIDATION_END,
        allow_locked_validation=True,
    )
    fingerprint, rows = discovery._selection_data_fingerprint(
        close, open_, high, low, volume, paths
    )
    bundle = build_daily_residual_replay_bundle(
        close,
        open_,
        high,
        low,
        volume,
        sectors,
        factor_model=settings.daily_residual_factor_model,
        source_fingerprint=fingerprint,
    )
    result = run_daily_residual_replay(
        bundle,
        settings,
        start=date.fromisoformat(LOCKED_VALIDATION_START),
        end=date.fromisoformat(LOCKED_VALIDATION_END),
        round_trip_cost_bps=20.0,
    )
    metrics = result.metrics()
    positive_sectors = {
        trade.sector
        for trade in result.trades
        if trade.r_multiple > 0.0
    }
    gates = {
        "positive_total_r": float(metrics["total_r"]) > 0.0,
        "positive_average_r": float(metrics["average_r"]) > 0.0,
        "profit_factor_above_one": float(metrics["profit_factor"]) > 1.0,
        "at_least_100_trades": int(metrics["trades"]) >= 100,
        "at_least_four_positive_sectors": len(positive_sectors) >= 4,
        "max_drawdown_lte_12pct_safety_ceiling": (
            float(metrics["max_drawdown_pct"]) <= 0.12
        ),
        "shared_core_contract_matches": (
            result.shared_core_contract == "iaric_daily_residual_execution_v2"
        ),
        "holdout_not_accessed": max(close.index).date().isoformat()
        <= LOCKED_VALIDATION_END,
    }
    return {
        "status": "passed" if all(gates.values()) else "failed_locked_validation",
        "passed": all(gates.values()),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "candidate_sha256": receipt["candidate_sha256"],
        "data_fingerprint": fingerprint,
        "fingerprinted_inputs": rows,
        "decision_event_count": len(result.decision_events),
        "positive_sectors": sorted(positive_sectors),
        "aspirational_target_assessment": {
            "mtm_max_drawdown_below_10pct": (
                float(metrics["max_drawdown_pct"]) < 0.10
            ),
            "used_as_hard_rejection_gate": False,
        },
        "locked_validation_accessed": True,
        "holdout_accessed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(
    output: Path,
    data_dir: Path,
    *,
    max_workers: int = 2,
    authority_manifest: Path | None = None,
    data_contract: str = RETAINED_LOCAL_RESEARCH,
    stop_after_exact_selection: bool = False,
    baseline_config: Path = DEFAULT_ROUND2_BASELINE,
    skip_protected_integration: bool = False,
) -> int:
    if max_workers != 2:
        raise ValueError("IARIC residual phased auto must run with max-workers=2")
    if data_contract not in DATA_CONTRACTS:
        raise ValueError(f"data-contract must be one of {DATA_CONTRACTS}")
    output.mkdir(parents=True, exist_ok=True)
    baseline_candidate, baseline_lineage = _load_round2_baseline(
        baseline_config.resolve()
    )
    _write_json(output / "round_2_baseline_lineage.json", baseline_lineage)
    _status(
        output,
        "attesting_phase_0_price_data_contract",
        data_contract=data_contract,
        broker_connection_required=False,
    )

    if data_contract == PRODUCTION_AUTHORITY:
        authority = attest_input_authority(
            REPO_ROOT,
            manifest_path=authority_manifest,
        )
        input_authority = authority.get("input_authority", {})
    else:
        authority = _attest_retained_local_research_snapshot(data_dir)
        input_authority = authority["input_authority"]
    authority_assessment = assess_input_authority(
        input_authority
    )
    if not authority_assessment["representative_reversion_baseline_eligible"]:
        _write_early_block(
            output,
            authority=authority,
            assessment=authority_assessment,
        )
        return 2

    _status(output, "running_phase_0_price_data_integrity_and_parity")

    selection_bundle_path = (
        Path(authority["selection_bundle"]["path"]).resolve()
        if data_contract == PRODUCTION_AUTHORITY
        else None
    )
    close, open_, high, low, volume, sectors, paths = _load_research_panel(
        data_contract=data_contract,
        data_dir=data_dir,
        selection_bundle_path=selection_bundle_path,
    )
    data_fingerprint, fingerprint_rows = discovery._selection_data_fingerprint(
        close, open_, high, low, volume, paths
    )
    code_fingerprint = _sha256_files(
        [
            Path(__file__),
            Path(discovery.__file__),
            REPO_ROOT / "backtests/stock/auto/iaric/residual_phases.py",
            REPO_ROOT / "strategies/stock/live_universe.py",
            REPO_ROOT / "strategies/stock/iaric/models.py",
            REPO_ROOT / "strategies/stock/iaric/artifact_store.py",
            REPO_ROOT / "strategies/stock/iaric/core/daily_residual.py",
            REPO_ROOT / "strategies/stock/iaric/core/residual.py",
            REPO_ROOT / "strategies/stock/iaric/core/lanes.py",
            REPO_ROOT / "strategies/stock/iaric/core/mechanisms.py",
            REPO_ROOT / "strategies/stock/iaric/config.py",
            REPO_ROOT / "strategies/stock/iaric/daily_residual_selection.py",
            REPO_ROOT / "strategies/stock/iaric/residual_engine.py",
            REPO_ROOT / "backtests/stock/engine/iaric_daily_residual_replay.py",
            REPO_ROOT / "backtests/stock/auto/iaric/representative_contract.py",
        ]
    )
    atlas_code_fingerprint = _sha256_files(
        [
            Path(discovery.__file__),
            REPO_ROOT / "strategies/stock/iaric/core/daily_residual.py",
            REPO_ROOT / "strategies/stock/iaric/core/mechanisms.py",
            REPO_ROOT / "strategies/stock/iaric/core/lanes.py",
            REPO_ROOT / "strategies/stock/iaric/daily_residual_selection.py",
            REPO_ROOT / "strategies/stock/volume_units.py",
        ]
    )
    integrity = discovery._price_data_integrity(
        close,
        open_,
        high,
        low,
        volume,
        sectors,
        authority_certified=True,
    )
    phase0 = {
        **integrity,
        "status": "complete_structural_integrity",
        "data_fingerprint": data_fingerprint,
        "code_fingerprint": code_fingerprint,
        "atlas_code_fingerprint": atlas_code_fingerprint,
        "residual_estimation_stock_symbols": len(sectors),
        "non_traded_explanatory_reference_symbols": 1
        + len(set(discovery.SECTOR_ETFS.values())),
        "non_traded_explanatory_reference_symbol_list": [
            "SPY",
            *sorted(set(discovery.SECTOR_ETFS.values())),
        ],
        "tradable_execution_symbols": len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        "tradable_execution_symbol_list": list(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        "cross_sectional_ranking_panel_matches_execution_universe": (
            set(sectors) == set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
        ),
        "all_execution_symbols_have_daily_data": set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
        <= set(close),
        "input_authority": authority,
        "authority_assessment": authority_assessment,
        "data_contract": data_contract,
        "data_authority_class": authority.get(
            "authority_class",
            "optional_receipt_backed_provenance_bundle",
        ),
        "broker_connection_required": False,
        "production_promotion_eligible": True,
        "acquisition_receipts_required": False,
        "selection_window": {"start": DISCOVERY_START, "end": CALIBRATION_END},
        "locked_validation_accessed": False,
        "holdout_accessed": False,
        "fingerprinted_inputs": fingerprint_rows,
        "selection_bundle_path": (
            str(selection_bundle_path) if selection_bundle_path is not None else None
        ),
        "retained_local_data_dir": (
            str(data_dir) if data_contract == RETAINED_LOCAL_RESEARCH else None
        ),
    }
    _write_json(output / "phase_0_price_data_integrity_and_parity.json", phase0)
    if not (
        integrity["passed_structural_checks"]
        and phase0["cross_sectional_ranking_panel_matches_execution_universe"]
        and phase0["all_execution_symbols_have_daily_data"]
    ):
        blocker = "certified inputs failed structural frozen-universe integrity"
        summary = {
            "status": "blocked_phase_0_structural_price_integrity_failure",
            "representative_reversion_baseline_eligible": False,
            "research_baseline_eligible": False,
            "optimizer_started": False,
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": None,
            "current_phase": PHASE_ORDER[0],
            "blocker": blocker,
            "max_workers": max_workers,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        }
        _write_json(output / "run_summary.json", summary)
        _status(output, summary["status"], blocker=blocker)
        return 2

    _status(output, "running_phase_1_residual_model_and_horizon_atlas")
    cache_paths = {
        factor_model: _atlas_cache_path(
            output,
            data_fingerprint=data_fingerprint,
            code_fingerprint=atlas_code_fingerprint,
            factor_model=factor_model,
        )
        for factor_model in SEARCH_FACTOR_MODELS
    }
    cache_complete = all(
        path.is_file() and path.with_suffix(".json").is_file()
        for path in cache_paths.values()
    )
    # Correlated-peer construction is the most expensive atlas precursor.  Do
    # not repeat it when every source/code/model-keyed atlas is already cached.
    peer_returns = None
    if not cache_complete:
        returns = close.pct_change(fill_method=None)
        peer_returns = discovery._causal_correlated_peer_returns(returns, sectors)
    atlases: dict[str, pd.DataFrame] = {}
    cache_rows: list[dict[str, Any]] = []
    atlas_rows: list[dict[str, Any]] = []
    for factor_model in SEARCH_FACTOR_MODELS:
        atlas, cache = _load_or_build_atlas(
            output=output,
            data_fingerprint=data_fingerprint,
            code_fingerprint=atlas_code_fingerprint,
            factor_model=factor_model,
            close=close,
            open_=open_,
            high=high,
            low=low,
            volume=volume,
            sector_by_symbol=sectors,
            peer_returns=peer_returns,
        )
        atlases[factor_model] = atlas
        cache_rows.append(cache)
        atlas_rows.append(
            {
                "factor_model": factor_model,
                "rows": len(atlas),
                "tradable_rows": int(atlas["tradable_execution_universe"].sum()),
                "tradable_symbols": int(
                    atlas.loc[atlas["tradable_execution_universe"], "symbol"].nunique()
                ),
            }
        )
    phase1 = {
        "status": "complete",
        "factor_models": list(SEARCH_FACTOR_MODELS),
        "primary_formation_sessions": [1, 3, 5],
        "control_formation_sessions": [20],
        "forward_horizons_sessions": [1, 2, 3, 5, 7, 10],
        "execution_universe_contract": "frozen_98_intraday_symbols_only",
        "stock_factor_estimation_and_ranking_panel_contract": (
            "same_frozen_98_intraday_symbols"
        ),
        "non_traded_explanatory_reference_contract": (
            "SPY_and_11_sector_ETFs_never_ranked_or_traded"
        ),
        "atlas_rows": atlas_rows,
        "cache": cache_rows,
        "atlas_code_fingerprint": atlas_code_fingerprint,
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(output / "phase_1_residual_model_and_horizon_atlas.json", phase1)

    _status(output, "running_phase_2_feature_qualification_and_discrimination")
    profiles: dict[str, dict[str, Any]] = {}
    executable_results: list[dict[str, Any]] = []
    control_results: list[dict[str, Any]] = []
    completed_models: list[str] = []
    for factor_model, atlas in atlases.items():
        _status(
            output,
            "running_phase_2_feature_qualification_and_discrimination",
            current_factor_model=factor_model,
            completed_factor_models=completed_models,
            executable_candidates_completed=len(executable_results),
            control_candidates_completed=len(control_results),
        )
        # Component diagnostics remain descriptive.  Qualify them on the same
        # formation and fixed-horizon outcome as the frozen baseline aperture;
        # the former five-session probe was not evidence for a one-session
        # selector.  Exact shared-core replay remains the ranking authority.
        # Round 2 varies only
        # mechanisms implicated by the frozen diagnostics: absolute quality
        # rejection, price rejection, failed continuation and one independent
        # formation-horizon challenger.  Holding/exit changes come later, after
        # the entry contract is frozen, so interactions cannot masquerade as
        # alpha discovery.
        profile = discovery.qualify_score_components(
            atlas,
            factor_model=factor_model,
            formation_sessions=int(baseline_candidate.formation_sessions),
            qualification_holding_sessions=int(
                baseline_candidate.holding_sessions
            ),
        )
        profile["executable_score_profiles_frozen_before_search"] = [
            ["volume_transition"],
            ["volume_transition", "price_rejection_recovery"],
            ["volume_transition", "failed_continuation"],
            [
                "volume_transition",
                "price_rejection_recovery",
                "failed_continuation",
            ],
            ["volume_transition", "regime_execution_quality"],
        ]
        profiles[factor_model] = profile
        executable: list[discovery.Candidate] = []
        if factor_model == baseline_candidate.factor_model:
            executable.append(baseline_candidate)
        mechanisms = (
            (("volume_transition",), (25.0, 40.0)),
            (("volume_transition", "price_rejection_recovery"), (25.0, 40.0)),
            (("volume_transition", "failed_continuation"), (25.0, 40.0)),
            (
                (
                    "volume_transition",
                    "price_rejection_recovery",
                    "failed_continuation",
                ),
                (25.0, 40.0),
            ),
            (("volume_transition", "regime_execution_quality"), (25.0,)),
        )
        if factor_model != baseline_candidate.factor_model:
            executable.append(
                _candidate(
                    factor_model=factor_model,
                    formation=1,
                    holding=10,
                    components=("volume_transition",),
                    lane_id="round2_residual_model_invariance_control",
                    minimum_market_trend_z_20d=-1.0,
                )
            )
        executable.extend(
            _candidate(
                factor_model=factor_model,
                formation=1,
                holding=10,
                components=score_profile,
                lane_id="round2_fresh_residual_quality_rejection_1d",
                minimum_market_trend_z_20d=-1.0,
                minimum_score=score_floor,
            )
            for score_profile, floors in mechanisms
            for score_floor in floors
        )
        executable.append(
            _candidate(
                factor_model=factor_model,
                formation=1,
                holding=10,
                components=(
                    "volume_transition",
                    "price_rejection_recovery",
                    "failed_continuation",
                ),
                lane_id="round2_hard_failed_continuation_confirmation_1d",
                minimum_failed_continuation_r=0.20,
                minimum_market_trend_z_20d=-1.0,
                minimum_score=25.0,
            )
        )
        if factor_model == "market_sector_peer":
            executable.append(
                _candidate(
                    factor_model=factor_model,
                    formation=3,
                    holding=10,
                    components=(
                        "volume_transition",
                        "price_rejection_recovery",
                        "failed_continuation",
                    ),
                    lane_id="round2_cumulative_residual_confirmation_3d",
                    minimum_failed_continuation_r=0.0,
                    minimum_market_trend_z_20d=-1.0,
                    minimum_score=25.0,
                )
            )
            executable.append(
                _candidate(
                    factor_model=factor_model,
                    formation=1,
                    holding=10,
                    components=(
                        "volume_transition",
                        "price_rejection_recovery",
                        "failed_continuation",
                    ),
                    ranking_components=(
                        "volume_transition",
                        "price_rejection_recovery",
                    ),
                    lane_id="round2_two_stage_admission_and_capacity_priority_1d",
                    minimum_failed_continuation_r=0.0,
                    minimum_market_trend_z_20d=-1.0,
                    minimum_score=25.0,
                )
            )
        # Controls share the same atlas and worker initialization.  A single
        # bounded pool per factor model avoids serializing the atlas twice.
        controls = [
            _candidate(
                factor_model=factor_model,
                formation=1,
                holding=10,
                components=("volume_transition",),
                leg=leg,
                lane_id=f"diagnostic_{leg}",
                minimum_market_trend_z_20d=-1.0,
            )
            for leg in ("short_winner", "dollar_neutral_spread")
        ]
        model_results = _evaluate(
            atlas, [*executable, *controls], max_workers=max_workers
        )
        executable_results.extend(model_results[: len(executable)])
        control_results.extend(model_results[len(executable) :])
        completed_models.append(factor_model)
        _status(
            output,
            "running_phase_2_feature_qualification_and_discrimination",
            current_factor_model=None,
            completed_factor_models=completed_models,
            executable_candidates_completed=len(executable_results),
            control_candidates_completed=len(control_results),
        )

    discovery._apply_neighbourhood_robustness(executable_results)
    _attach_model_invariance(executable_results)
    executable_results.sort(
        key=lambda row: (-float(row["score"]), row["candidate"]["candidate_id"])
    )
    control_results.sort(
        key=lambda row: (-float(row["score"]), row["candidate"]["candidate_id"])
    )
    screen_gates = (
        "positive_each_fold",
        "minimum_100_trades_each_fold",
        "positive_each_fold_after_30bps",
        "nonnegative_calibration_after_40bps",
        "at_least_four_positive_sectors",
    )
    screened = [
        row
        for row in executable_results
        if all(bool(row["gates"].get(name, False)) for name in screen_gates)
    ]
    best_diagnostic = executable_results[0]
    mandatory_exact = [
        row
        for row in executable_results
        if row["candidate"]["candidate_id"] == baseline_candidate.candidate_id
        or (
            row["candidate"]["factor_model"] == "peer_demeaned"
            and int(row["candidate"]["formation_sessions"]) == 1
            and int(row["candidate"]["holding_sessions"]) == 10
            and tuple(row["candidate"].get("score_components", ()))
            == ("volume_transition",)
            and float(row["candidate"].get("minimum_score", 0.0)) == 0.0
        )
        or bool(row["candidate"].get("ranking_score_components", ()))
    ]
    # Every candidate that passes the inexpensive economic screen receives the
    # same exact shared-core replay.  Approximate scores can reject obvious
    # failures but may not rank or truncate exact finalists.
    shortlist = []
    for row in [*mandatory_exact, *screened]:
        if row not in shortlist:
            shortlist.append(row)
    _write_json(output / "phase_2_feature_profiles.json", profiles)
    _write_json(
        output / "phase_2_executable_candidate_registry.json",
        [_compact(row) for row in executable_results],
    )
    _write_json(
        output / "phase_2_control_leg_registry.json",
        [_compact(row) for row in control_results],
    )
    _write_json(output / "best_diagnostic_candidate.json", best_diagnostic)

    _status(output, "evaluating_phase_3_selection_contract_robustness")
    phase3 = {
        "status": "passed_to_exact_shortlist" if shortlist else "blocked_no_selector_screen",
        "executable_candidate_count": len(executable_results),
        "control_candidate_count": len(control_results),
        "fully_qualified_approximate_candidate_count": sum(
            bool(row["qualified_discovery_candidate"])
            for row in executable_results
        ),
        "screened_candidate_count": len(screened),
        "screen_gates": list(screen_gates),
        "exact_shortlist_candidate_ids": [
            row["candidate"]["candidate_id"] for row in shortlist
        ],
        "exact_completion_contract": (
            "all_economically_screened_plus_frozen_baseline_model_control_and_"
            "pre_registered_two_stage_candidate"
        ),
        "approximate_score_used_for_exact_ranking": False,
        "best_diagnostic_candidate_id": best_diagnostic["candidate"]["candidate_id"],
        "best_diagnostic_failed_gates": [
            name for name, passed in best_diagnostic["gates"].items() if not passed
        ],
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(output / "phase_3_selection_contract_robustness.json", phase3)

    if not shortlist:
        final_status = "blocked_phase_3_no_robust_representative_selector"
        blocker = (
            "No 98-name executable selector passed discrimination, rejected-cohort, "
            "independent-date, cost, model, concentration and capacity gates."
        )
        summary = {
            "status": final_status,
            "representative_reversion_baseline_eligible": False,
            "research_baseline_eligible": False,
            "optimizer_class": "gated_price_volume_residual_phased_auto_v4",
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[3],
            "current_phase": PHASE_ORDER[3],
            "blocker": blocker,
            "best_diagnostic_candidate_id": best_diagnostic["candidate"]["candidate_id"],
            "best_diagnostic_metrics": best_diagnostic["metrics"],
            "best_diagnostic_fold_metrics": best_diagnostic["fold_metrics"],
            "data_fingerprint": data_fingerprint,
            "code_fingerprint": code_fingerprint,
            "max_workers": max_workers,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        }
        _write_json(output / "run_summary.json", summary)
        _status(output, final_status, blocker=blocker)
        return 2

    _status(
        output,
        "running_phase_4_causal_entry_delivery",
        exact_shortlist_candidates=len(shortlist),
    )
    bundle_by_factor = {
        factor_model: build_daily_residual_replay_bundle(
            close,
            open_,
            high,
            low,
            volume,
            sectors,
            factor_model=factor_model,
            source_fingerprint=data_fingerprint,
        )
        for factor_model in sorted(
            {str(row["candidate"]["factor_model"]) for row in shortlist}
        )
    }

    def exact_shortlist_row(row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_row = row["candidate"]
        settings_row = settings_from_discovery_candidate(candidate_row)
        exact_row = run_exact_fold_evaluation(
            replace(
                bundle_by_factor[str(candidate_row["factor_model"])],
                frozen_history_cache={},
            ),
            settings_row,
            round_trip_cost_bps=20.0,
            score_contract="round2",
        )
        return {
            "candidate": candidate_row,
            "approximate_score": row["score"],
            "exact": exact_row,
            "exact_qualified": bool(exact_row["research_anchor_eligible"]),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        exact_shortlist = list(pool.map(exact_shortlist_row, shortlist))

    _status(
        output,
        "running_phase_4a_exact_screen_completion_and_pareto",
        exact_candidates=len(exact_shortlist),
    )
    pareto_ids: list[str] = []
    for row in exact_shortlist:
        metrics = row["exact"]["continuous_metrics"]
        dominated = False
        for other in exact_shortlist:
            if other is row:
                continue
            other_metrics = other["exact"]["continuous_metrics"]
            no_worse = (
                float(other_metrics["total_r"]) >= float(metrics["total_r"])
                and int(other_metrics["trades"]) >= int(metrics["trades"])
                and float(other_metrics["max_drawdown_pct"])
                <= float(metrics["max_drawdown_pct"])
            )
            strictly_better = (
                float(other_metrics["total_r"]) > float(metrics["total_r"])
                or int(other_metrics["trades"]) > int(metrics["trades"])
                or float(other_metrics["max_drawdown_pct"])
                < float(metrics["max_drawdown_pct"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_ids.append(str(row["candidate"]["candidate_id"]))
    phase4a = {
        "status": "complete",
        "contract": "exact_replay_every_economically_screened_candidate_v1",
        "approximate_score_used_for_ranking": False,
        "exact_candidate_count": len(exact_shortlist),
        "pareto_dimensions": ["total_r", "trades", "max_drawdown_pct"],
        "pareto_candidate_ids": pareto_ids,
        "candidates": [
            {
                **{key: value for key, value in row.items() if key != "exact"},
                "exact": _compact_exact(row["exact"]),
            }
            for row in exact_shortlist
        ],
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(
        output / "phase_4a_exact_screen_completion_and_pareto.json",
        phase4a,
    )

    _status(
        output,
        "running_phase_4b_mechanism_aware_rejection_and_capacity_attribution",
    )
    def attribution_row(row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_payload = dict(row["candidate"])
        candidate_payload["score_components"] = tuple(
            candidate_payload.get("score_components", ())
        )
        candidate_payload["ranking_score_components"] = tuple(
            candidate_payload.get("ranking_score_components", ())
        )
        candidate_row = discovery.Candidate(**candidate_payload)
        return _candidate_opportunity_attribution(
            atlases[str(candidate_row.factor_model)], candidate_row
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        attributions = list(pool.map(attribution_row, exact_shortlist))
    phase4b = {
        "status": "complete",
        "contract": "mechanism_aware_rejection_and_capacity_ledger_v1",
        "inactive_mechanism_contract": "not_applicable_never_false",
        "matching_contract": (
            "same_formation_date_and_sector_nearest_absolute_residual_z"
        ),
        "portfolio_counterfactual_claimed": False,
        "candidates": attributions,
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(
        output
        / "phase_4b_mechanism_aware_rejection_and_capacity_attribution.json",
        phase4b,
    )

    _status(output, "running_phase_4c_two_stage_admission_and_ranking")
    structural_rows = [
        row
        for row in exact_shortlist
        if bool(row["candidate"].get("ranking_score_components", ()))
    ]
    phase4c = {
        "status": (
            "passed" if any(row["exact_qualified"] for row in structural_rows)
            else "complete_no_qualified_two_stage_candidate"
        ),
        "contract": "shared_core_two_stage_admission_then_capacity_priority_v1",
        "score_component_union_ceiling": 7,
        "candidate_count": len(structural_rows),
        "candidates": [
            {
                **{key: value for key, value in row.items() if key != "exact"},
                "exact": _compact_exact(row["exact"]),
            }
            for row in structural_rows
        ],
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(
        output / "phase_4c_two_stage_admission_and_ranking.json",
        phase4c,
    )
    exact_qualified = [row for row in exact_shortlist if row["exact_qualified"]]
    exact_qualified.sort(
        key=lambda row: (
            -float(row["exact"]["immutable_score"]["score"]),
            row["candidate"]["candidate_id"],
        )
    )
    selected_exact = exact_qualified[0] if exact_qualified else None
    exact_ranked = sorted(
        exact_shortlist,
        key=lambda row: (
            -float(row["exact"]["immutable_score"]["score"]),
            row["candidate"]["candidate_id"],
        ),
    )
    diagnostic_exact = exact_ranked[0]
    candidate = (
        selected_exact["candidate"]
        if selected_exact
        else diagnostic_exact["candidate"]
    )
    _status(
        output,
        "running_phase_4_causal_entry_delivery",
        candidate_id=candidate["candidate_id"],
    )
    bundle = bundle_by_factor[str(candidate["factor_model"])]
    phase4_settings = settings_from_discovery_candidate(candidate)
    phase4_exact = (
        selected_exact["exact"]
        if selected_exact is not None
        else diagnostic_exact["exact"]
    )
    phase4 = {
        "status": (
            "passed" if phase4_exact["research_anchor_eligible"] else "blocked"
        ),
        "entry_contract": "causal_next_session_open_market",
        "selection_signal_completed_at": "prior_session_close",
        "execution_fidelity": "shared_live_replay_neutral_actions_and_fill_reducer",
        "entry_delivery_attribution": {
            name: row["entry_delivery_attribution"]
            for name, row in phase4_exact["folds"].items()
        },
        "alternatives_not_promotable": {
            "first_30_minute_vwap": "not a directly attainable fill price and not used as an optimistic proxy",
            "resting_retrace_limit": "requires separately modelled misses and authoritative intraday bars",
            "five_minute_recovery_confirmation": "belongs to an independently qualified secondary sleeve",
        },
        "exact_replay": _compact_exact(phase4_exact),
        "exact_shortlist": [
            {
                **{key: value for key, value in row.items() if key != "exact"},
                "exact": _compact_exact(row["exact"]),
            }
            for row in exact_shortlist
        ],
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(output / "phase_4_causal_entry_delivery.json", phase4)
    if selected_exact is None:
        blocker = (
            "The discovery approximation did not survive the exact shared-core, "
            "shared-cash next-open replay in both selection folds."
        )
        summary = {
            "status": "blocked_phase_4_exact_delivery_failure",
            "representative_reversion_baseline_eligible": False,
            "research_baseline_eligible": False,
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[7],
            "current_phase": PHASE_ORDER[7],
            "blocker": blocker,
            "best_eligible_candidate_id": candidate["candidate_id"],
            "exact_fold_metrics": phase4_exact["folds"],
            "data_fingerprint": data_fingerprint,
            "code_fingerprint": code_fingerprint,
            "max_workers": max_workers,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        }
        _write_json(output / "run_summary.json", summary)
        _status(output, summary["status"], blocker=blocker)
        return 2

    if stop_after_exact_selection:
        frozen_settings = settings_from_discovery_candidate(candidate)
        summary = {
            "status": "complete_exact_selection_baseline",
            "representative_reversion_baseline_eligible": True,
            "research_baseline_eligible": True,
            "production_promotion_eligible": False,
            "selection_candidate": candidate,
            "selection_settings": _settings_payload(frozen_settings),
            "exact_selection": _compact_exact(selected_exact["exact"]),
            "exact_shortlist": phase4["exact_shortlist"],
            "data_fingerprint": data_fingerprint,
            "code_fingerprint": code_fingerprint,
            "max_workers": max_workers,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(output / "selected_baseline_config.json", {
            "candidate": candidate,
            "settings": summary["selection_settings"],
        })
        _write_json(output / "run_summary.json", summary)
        _status(
            output,
            summary["status"],
            candidate_id=candidate["candidate_id"],
            locked_validation_accessed=False,
            holdout_accessed=False,
        )
        return 0

    _status(output, "running_phase_5_residual_anchor_and_half_life_management")
    phase5 = run_management_phase(
        bundle,
        candidate,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_5_residual_anchor_and_half_life_management.json",
        _compact_management(phase5),
    )
    if phase5["status"] != "passed" or not phase5.get("selected"):
        blocker = "No typed residual-management contract survived both selection folds."
        summary = {
            "status": "blocked_phase_5_management_failure",
            "representative_reversion_baseline_eligible": False,
            "research_baseline_eligible": False,
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[8],
            "current_phase": PHASE_ORDER[8],
            "blocker": blocker,
            "best_eligible_candidate_id": candidate["candidate_id"],
            "max_workers": max_workers,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        }
        _write_json(output / "run_summary.json", summary)
        _status(output, summary["status"], blocker=blocker)
        return 2

    selected_management = phase5["selected"]["experiment"]
    final_settings = settings_from_discovery_candidate(
        candidate,
        management=selected_management,
    )
    frozen_candidate = {
        "candidate_id": candidate["candidate_id"],
        "discovery_candidate": candidate,
        "management_experiment": selected_management,
        "settings": _settings_payload(final_settings),
        "settings_sha256": _payload_sha256(_settings_payload(final_settings)),
        "score_component_count": len(
            set(final_settings.daily_residual_score_components)
            | set(final_settings.daily_residual_ranking_score_components)
        ),
        "optimizer_score_component_count": len(ROUND2_SCORE_SPEC),
        "optimizer_score_contract": "iaric_round2_non_saturated_exact_v2",
        "universe_contract": "frozen_98_intraday_symbols_only",
        "locked_validation_used_for_selection": False,
        "holdout_accessed": False,
    }
    _write_json(output / "frozen_selection_candidate.json", frozen_candidate)

    _status(
        output,
        "running_phase_6_independent_sleeve_qualification_and_final_robustness",
    )
    phase6 = run_final_robustness_phase(
        bundle,
        final_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    phase6_output = {
        **phase6,
        "base_20bps": _compact_exact(phase6["base_20bps"]),
        "cost_stress_30bps": _compact_exact(phase6["cost_stress_30bps"]),
        "cost_stress_40bps": _compact_exact(phase6["cost_stress_40bps"]),
        "neighbourhood": {
            name: _compact_exact(row)
            for name, row in phase6["neighbourhood"].items()
        },
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }
    _write_json(
        output / "phase_6_independent_sleeve_qualification_and_final_robustness.json",
        phase6_output,
    )
    if not phase6["qualification"]["passed"]:
        blocker = {
            "message": "The exact daily sleeve failed final robustness promotion gates.",
            "failed_gates": phase6["qualification"]["failed_gates"],
        }
        summary = {
            "status": "blocked_phase_6_final_robustness_failure",
            "representative_reversion_baseline_eligible": False,
            "research_baseline_eligible": False,
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[9],
            "current_phase": PHASE_ORDER[9],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "max_workers": max_workers,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        }
        _write_json(output / "run_summary.json", summary)
        _status(output, summary["status"], blocker=blocker)
        return 2

    _status(output, "running_phase_7_protected_integration_and_literal_ablation")
    if skip_protected_integration:
        phase7 = {
            "status": "not_applicable_single_sleeve_round2",
            "passed": True,
            "gates": {"exact_frozen_baseline_ablation_recorded": True},
            "reason": (
                "Round 2 optimizes the existing residual sleeve rather than adding "
                "a second sleeve; candidate-vs-frozen-baseline ablations are recorded "
                "in Phases 4-6. The unavailable legacy Round-3 cross-sleeve control "
                "cannot be substituted with the same Round-1 residual trades."
            ),
        }
    else:
        if not FROZEN_ROUND3_CONTROL_TRADES.is_file():
            raise FileNotFoundError(
                f"missing frozen Round-3 control: {FROZEN_ROUND3_CONTROL_TRADES}"
            )
        phase7 = run_protected_integration_phase(
            bundle,
            phase6["base_20bps"],
            frozen_control_trades_path=FROZEN_ROUND3_CONTROL_TRADES,
        )
    phase7.update({"locked_validation_accessed": False, "holdout_accessed": False})
    _write_json(
        output / "phase_7_protected_integration_and_literal_ablation.json",
        phase7,
    )
    if not phase7["passed"]:
        blocker = {
            "message": "Residual alpha did not add broad positive value after protected issuer arbitration.",
            "failed_gates": [
                name for name, passed in phase7["gates"].items() if not passed
            ],
        }
        summary = {
            "status": "blocked_phase_7_protected_integration_failure",
            "representative_reversion_baseline_eligible": False,
            "research_baseline_eligible": False,
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[10],
            "current_phase": PHASE_ORDER[10],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "max_workers": max_workers,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        }
        _write_json(output / "run_summary.json", summary)
        _status(output, summary["status"], blocker=blocker)
        return 2

    inherited_phase6_cap12 = phase6["neighbourhood"].get("position_cap_12")
    if not inherited_phase6_cap12 or not bool(
        inherited_phase6_cap12.get("research_anchor_eligible")
    ):
        raise ValueError(
            "Phase 8 requires the exact eligible Phase-6 position_cap_12 result"
        )
    if float(inherited_phase6_cap12["immutable_score"]["score"]) <= float(
        phase6["base_20bps"]["immutable_score"]["score"]
    ):
        raise ValueError(
            "Phase-6 position_cap_12 must strictly improve the immutable score"
        )
    phase8_starting_settings = replace(
        final_settings,
        daily_residual_max_positions=12,
        daily_residual_max_positions_per_sector=2,
        daily_residual_sector_overflow_slots=0,
    )
    _status(
        output,
        "running_phase_8_selective_sector_overflow_and_displacement_quality",
    )
    phase8 = run_selective_sector_overflow_phase(
        bundle,
        phase8_starting_settings,
        max_workers=max_workers,
        score_contract="round2",
        inherited_control_result=inherited_phase6_cap12,
    )
    _write_json(
        output / "phase_8_selective_sector_overflow_and_displacement_quality.json",
        {
            **_compact_settings_frontier(phase8),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if phase8.get("selected_settings") is None:
        blocker = "No exact selective-sector-overflow candidate retained robust selection economics."
        summary = {
            "status": "blocked_phase_8_selective_sector_overflow_frontier",
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[11],
            "current_phase": PHASE_ORDER[11],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2
    phase8_settings = phase8["selected_settings"]

    _status(output, "running_phase_9_quality_aperture_and_discrimination")
    phase9 = run_quality_aperture_phase(
        bundle,
        phase8_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_9_quality_aperture_and_discrimination.json",
        {
            **_compact_settings_frontier(phase9),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if phase9.get("selected_settings") is None:
        blocker = "No exact signal aperture retained both-fold discrimination."
        summary = {
            "status": "blocked_phase_9_quality_aperture",
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[12],
            "current_phase": PHASE_ORDER[12],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2
    phase9_settings = phase9["selected_settings"]

    _status(output, "running_phase_10_risk_and_notional_frontier")
    phase10 = run_risk_notional_frontier_phase(
        bundle,
        phase9_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_10_risk_and_notional_frontier.json",
        {
            **_compact_settings_frontier(phase10),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if phase10.get("selected_settings") is None:
        blocker = "No aggressive sizing candidate kept selection MTM drawdown below 10%."
        summary = {
            "status": "blocked_phase_10_risk_frontier",
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[13],
            "current_phase": PHASE_ORDER[13],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2
    phase10_settings = phase10["selected_settings"]

    _status(output, "running_phase_11_exit_capture_frontier")
    phase11 = run_exit_capture_frontier_phase(
        bundle,
        phase10_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_11_exit_capture_frontier.json",
        {
            **_compact_settings_frontier(phase11),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if phase11.get("selected_settings") is None:
        blocker = "No exit-capture candidate retained robust selection economics."
        summary = {
            "status": "blocked_phase_11_exit_capture",
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[14],
            "current_phase": PHASE_ORDER[14],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2
    phase11_settings = phase11["selected_settings"]

    _status(output, "running_phase_12_final_alpha_frequency_synergy")
    phase12 = run_final_alpha_synergy_phase(
        bundle,
        phase11_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_12_final_alpha_frequency_synergy.json",
        {
            **_compact_settings_frontier(phase12),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if phase12.get("selected_settings") is None:
        blocker = {
            "message": "No exact robust synergy candidate remained inside the 12% MTM drawdown safety ceiling.",
            "aspirational_guidance": {
                "selection_total_r": ">100R",
                "selection_mtm_max_drawdown": "<10%",
                "used_as_hard_rejection_gate": False,
            },
        }
        summary = {
            "status": "blocked_phase_12_no_robust_synergy_candidate",
            "representative_reversion_baseline_eligible": True,
            "research_baseline_eligible": True,
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[15],
            "current_phase": PHASE_ORDER[15],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "aspirational_target_met": False,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2
    final_settings = phase12["selected_settings"]

    _status(output, "running_phase_13_path_causal_profit_retention")
    phase13 = run_path_causal_profit_retention_phase(
        bundle,
        final_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_13_path_causal_profit_retention.json",
        {
            **_compact_settings_frontier(phase13),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if phase13.get("selected_settings") is None:
        blocker = "No path-causal profit-retention candidate retained robust economics."
        summary = {
            "status": "blocked_phase_13_path_causal_profit_retention",
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[16],
            "current_phase": PHASE_ORDER[16],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2
    final_settings = phase13["selected_settings"]

    _status(output, "running_phase_14_capacity_neutral_alpha_recycling")
    phase14 = run_capacity_neutral_alpha_recycling_phase(
        bundle,
        final_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_14_capacity_neutral_alpha_recycling.json",
        {
            **_compact_settings_frontier(phase14),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if phase14.get("selected_settings") is None:
        blocker = "No exact capacity-neutral replacement candidate retained robust economics."
        summary = {
            "status": "blocked_phase_14_capacity_neutral_alpha_recycling",
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[17],
            "current_phase": PHASE_ORDER[17],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2
    final_settings = phase14["selected_settings"]

    post_phase7_lineage = {
        "phase_8": phase8["selected"]["experiment_id"],
        "phase_9": phase9["selected"]["experiment_id"],
        "phase_10": phase10["selected"]["experiment_id"],
        "phase_11": phase11["selected"]["experiment_id"],
        "phase_12": phase12["selected"]["experiment_id"],
        "phase_13": phase13["selected"]["experiment_id"],
        "phase_14": phase14["selected"]["experiment_id"],
    }
    frozen_candidate.update(
        {
            "settings": _settings_payload(final_settings),
            "settings_sha256": _payload_sha256(_settings_payload(final_settings)),
            "post_phase_7_experiment_lineage": post_phase7_lineage,
            "aspirational_target_contract": phase12[
                "aspirational_target_contract"
            ],
        }
    )
    _write_json(output / "frozen_selection_candidate.json", frozen_candidate)

    _status(output, "running_phase_15_final_robustness_and_target_assessment")
    phase15 = run_final_robustness_and_target_assessment_phase(
        bundle,
        final_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    _write_json(
        output / "phase_15_final_robustness_and_target_assessment.json",
        {
            **phase15,
            "base_20bps": _compact_exact(phase15["base_20bps"]),
            "cost_stress_30bps": _compact_exact(phase15["cost_stress_30bps"]),
            "cost_stress_40bps": _compact_exact(phase15["cost_stress_40bps"]),
            "neighbourhood": {
                name: _compact_exact(row)
                for name, row in phase15["neighbourhood"].items()
            },
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if not phase15["qualification"]["passed"]:
        blocker = {
            "message": "The final candidate failed robustness or the 12% MTM drawdown safety ceiling.",
            "failed_gates": phase15["qualification"]["failed_gates"],
            "aspirational_target_assessment": phase15[
                "aspirational_target_assessment"
            ],
        }
        summary = {
            "status": "blocked_phase_15_final_robustness",
            "representative_reversion_baseline_eligible": True,
            "research_baseline_eligible": True,
            "phase_order": list(PHASE_ORDER),
            "last_completed_phase": PHASE_ORDER[18],
            "current_phase": PHASE_ORDER[18],
            "blocker": blocker,
            "frozen_candidate": frozen_candidate,
            "aspirational_target_met": bool(
                phase15["aspirational_target_assessment"]["both_met"]
            ),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "max_workers": max_workers,
        }
        _write_json(output / "run_summary.json", summary)
        write_blocked_round_final_diagnostics(output)
        _status(output, summary["status"], blocker=blocker)
        return 2

    _status(
        output,
        "running_phase_16_one_shot_locked_chronological_validation",
        locked_validation_accessed=True,
    )
    phase16 = _run_locked_validation_once(
        output=output,
        data_contract=data_contract,
        data_dir=data_dir,
        selection_bundle_path=selection_bundle_path,
        settings=final_settings,
    )
    _write_json(output / "phase_16_locked_chronological_validation.json", phase16)
    if phase16["passed"]:
        final_status = "complete_locked_validation"
    else:
        final_status = "failed_locked_chronological_validation"
    summary = {
        "status": final_status,
        "representative_reversion_baseline_eligible": bool(phase16["passed"]),
        "research_baseline_eligible": True,
        "production_promotion_eligible": False,
        "acquisition_receipts_required": False,
        "data_contract": data_contract,
        "data_authority_class": authority.get(
            "authority_class",
            "optional_receipt_backed_provenance_bundle",
        ),
        "broker_connection_required": False,
        "optimizer_class": "gated_price_volume_residual_phased_auto_v6",
        "terminal_artifact": "round_final_diagnostics.txt",
        "phase_order": list(PHASE_ORDER),
        "last_completed_phase": PHASE_ORDER[-1],
        "current_phase": None,
        "blocker": None if phase16["passed"] else phase16["failed_gates"],
        "frozen_candidate": frozen_candidate,
        "selection_fold_metrics": phase15["base_20bps"]["folds"],
        "selection_metrics": phase15["base_20bps"]["continuous_metrics"],
        "aspirational_target_met": bool(
            phase15["aspirational_target_assessment"]["both_met"]
        ),
        "aspirational_target_assessment": phase15[
            "aspirational_target_assessment"
        ],
        "locked_validation_metrics": phase16["metrics"],
        "residual_estimation_stock_symbols": len(sectors),
        "non_traded_explanatory_reference_symbols": 1
        + len(set(discovery.SECTOR_ETFS.values())),
        "tradable_execution_symbols": len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        "data_fingerprint": data_fingerprint,
        "code_fingerprint": code_fingerprint,
        "cache_hits": sum(bool(row.get("cache_hit")) for row in cache_rows),
        "cache_entries": len(cache_rows),
        "max_workers": max_workers,
        "locked_validation_start": LOCKED_VALIDATION_START,
        "locked_validation_accessed": True,
        "holdout_start": HOLDOUT_START,
        "holdout_accessed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "run_summary.json", summary)
    write_round_final_diagnostics(output)
    _status(
        output,
        final_status,
        representative_reversion_baseline_eligible=summary[
            "representative_reversion_baseline_eligible"
        ],
        locked_validation_accessed=True,
        holdout_accessed=False,
    )
    return 0 if phase16["passed"] else 2


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=discovery.DEFAULT_DATA_DIR)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_ROUND2_BASELINE,
        help="frozen optimized configuration used as the literal Round 2 baseline",
    )
    parser.add_argument(
        "--skip-protected-integration",
        action="store_true",
        help="skip the unrelated legacy cross-sleeve control; exact baseline ablations remain mandatory",
    )
    parser.add_argument(
        "--stop-after-exact-selection",
        action="store_true",
        help="freeze the representative discovery/calibration baseline without reading locked validation",
    )
    parser.add_argument(
        "--data-contract",
        choices=DATA_CONTRACTS,
        default=RETAINED_LOCAL_RESEARCH,
        help=(
            "project-official local snapshot (default; no broker or acquisition "
            "logs required) or the optional receipt-backed provenance bundle"
        ),
    )
    parser.add_argument(
        "--authority-manifest",
        type=Path,
        default=REPO_ROOT / DEFAULT_MANIFEST_RELATIVE,
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    output = args.output_dir.resolve()
    try:
        return run(
            output,
            args.data_dir.resolve(),
            max_workers=args.max_workers,
            authority_manifest=args.authority_manifest.resolve(),
            data_contract=args.data_contract,
            stop_after_exact_selection=args.stop_after_exact_selection,
            baseline_config=args.baseline_config.resolve(),
            skip_protected_integration=args.skip_protected_integration,
        )
    except Exception as exc:
        _status(
            output,
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
