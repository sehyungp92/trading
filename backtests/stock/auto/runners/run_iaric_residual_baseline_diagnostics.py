"""Establish the frozen exact-98 IARIC residual starting baseline.

This is a diagnostic replay, not an optimization.  It runs the pre-search
executable residual configuration over discovery plus calibration only, using
the shared live/replay selector, neutral-action execution core, shared cash,
next-session-open fills and explicit 20 bps round-trip costs.  Locked
validation and the sealed holdout are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CALIBRATION_START,
    DISCOVERY_END,
    DISCOVERY_START,
    HOLDOUT_START,
)
from backtests.stock.auto.iaric.residual_phases import (
    run_exact_fold_evaluation,
    settings_from_discovery_candidate,
)
from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.auto.runners.run_iaric_residual_phased_auto import (
    _attest_retained_local_research_snapshot,
)
from backtests.stock.engine.iaric_daily_residual_replay import (
    build_daily_residual_replay_bundle,
    run_daily_residual_replay,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core.lanes import issuer_key
from strategies.stock.iaric.core.daily_residual import DAILY_RESIDUAL_SLEEVE
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/round_4/residual_baseline_full_is"
)
BASELINE_CONTRACT_ID = "iaric_residual_presearch_exact98_v1"
REPRESENTATIVE_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/round_4/residual_representative_baseline_v5_official_anchor"
)
REPRESENTATIVE_CONTRACT_ID = (
    "iaric_residual_market_sector_peer_f1_h10_volume_exact98_v5"
)
SELECTED_REPRESENTATIVE_CONTRACT_ID = "iaric_residual_selected_exact98_v5"
INITIAL_EQUITY = 100_000.0
BASE_COST_BPS = 20.0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_settings() -> StrategySettings:
    """Return the literal pre-search executable strategy contract."""

    return StrategySettings(
        strategy_mode=DAILY_RESIDUAL_SLEEVE,
        daily_residual_factor_model="market_sector_peer",
        daily_residual_formation_sessions=3,
        daily_residual_minimum_z=1.0,
        daily_residual_minimum_failed_continuation_r=0.0,
        daily_residual_minimum_sector_return_5d=-0.15,
        daily_residual_lane_id="legacy_presearch_control",
        daily_residual_score_components=("residual_extremeness",),
        daily_residual_max_positions=10,
        daily_residual_max_positions_per_sector=2,
        daily_residual_risk_fraction=0.0035,
        daily_residual_maximum_notional_fraction=0.10,
        daily_residual_catastrophic_stop_atr=2.50,
        daily_residual_catastrophic_stop_residual_r=0.0,
        daily_residual_partial_normalization_fraction=0.50,
        daily_residual_full_normalization_fraction=1.00,
        daily_residual_structural_failure_extension_fraction=0.50,
        daily_residual_maximum_holding_sessions=7,
        daily_residual_partial_exit_fraction=0.50,
    )


def representative_settings() -> StrategySettings:
    """Return the literal executable form of the +74.55R diagnostic family."""

    return StrategySettings(
        strategy_mode=DAILY_RESIDUAL_SLEEVE,
        daily_residual_factor_model="market_sector_peer",
        daily_residual_formation_sessions=1,
        daily_residual_minimum_z=1.0,
        daily_residual_minimum_failed_continuation_r=0.0,
        daily_residual_minimum_sector_return_5d=-0.15,
        daily_residual_minimum_market_trend_z_20d=-8.0,
        daily_residual_score_components=("volume_transition",),
        daily_residual_lane_id="fresh_market_sector_peer_residual_1d",
        daily_residual_max_positions=10,
        daily_residual_max_positions_per_sector=2,
        daily_residual_risk_fraction=0.0035,
        daily_residual_maximum_notional_fraction=0.10,
        daily_residual_catastrophic_stop_atr=2.50,
        daily_residual_catastrophic_stop_residual_r=4.0,
        # Preserve the diagnostic fixed half-life before optimizing typed exits.
        daily_residual_partial_normalization_fraction=99.0,
        daily_residual_full_normalization_fraction=99.0,
        daily_residual_structural_failure_extension_fraction=99.0,
        daily_residual_maximum_holding_sessions=10,
        daily_residual_partial_exit_fraction=0.0,
    )


def _economic_code_paths() -> tuple[Path, ...]:
    """Files whose bytes can change selection, sizing, fills or management."""

    return (
        Path(__file__),
        Path(discovery.__file__),
        REPO_ROOT / "backtests/stock/auto/iaric/representative_contract.py",
        REPO_ROOT / "backtests/stock/auto/iaric/residual_phases.py",
        REPO_ROOT / "backtests/stock/engine/iaric_daily_residual_replay.py",
        REPO_ROOT / "strategies/stock/live_universe.py",
        REPO_ROOT / "strategies/stock/iaric/config.py",
        REPO_ROOT / "strategies/stock/iaric/models.py",
        REPO_ROOT / "strategies/stock/iaric/artifact_store.py",
        REPO_ROOT / "strategies/stock/iaric/daily_residual_selection.py",
        REPO_ROOT / "strategies/stock/iaric/residual_engine.py",
        REPO_ROOT / "strategies/stock/iaric/core/daily_residual.py",
        REPO_ROOT / "strategies/stock/iaric/core/residual.py",
        REPO_ROOT / "strategies/stock/iaric/core/lanes.py",
    )


def _economic_code_fingerprint() -> tuple[str, list[dict[str, str]]]:
    digest = hashlib.sha256()
    rows: list[dict[str, str]] = []
    for path in _economic_code_paths():
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        value = _sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
        rows.append({"path": relative, "sha256": value})
    return digest.hexdigest(), rows


def _settings_payload(settings: StrategySettings) -> dict[str, Any]:
    names = (
        "strategy_mode",
        "daily_residual_factor_model",
        "daily_residual_formation_sessions",
        "daily_residual_minimum_z",
        "daily_residual_minimum_failed_continuation_r",
        "daily_residual_minimum_sector_return_5d",
        "daily_residual_minimum_market_trend_z_20d",
        "daily_residual_lane_id",
        "daily_residual_score_components",
        "daily_residual_max_positions",
        "daily_residual_max_positions_per_sector",
        "daily_residual_risk_fraction",
        "daily_residual_maximum_notional_fraction",
        "daily_residual_catastrophic_stop_atr",
        "daily_residual_catastrophic_stop_residual_r",
        "daily_residual_partial_normalization_fraction",
        "daily_residual_full_normalization_fraction",
        "daily_residual_structural_failure_extension_fraction",
        "daily_residual_maximum_holding_sessions",
        "daily_residual_partial_exit_fraction",
    )
    return {name: getattr(settings, name) for name in names}


def _months(start: str = DISCOVERY_START, end: str = CALIBRATION_END) -> float:
    return max(
        (date.fromisoformat(end) - date.fromisoformat(start)).days / 30.4375,
        1.0,
    )


def _profit_factor(values: Iterable[float]) -> float:
    rows = list(values)
    gains = sum(value for value in rows if value > 0.0)
    losses = abs(sum(value for value in rows if value < 0.0))
    if losses > 0.0:
        return gains / losses
    return float("inf") if gains > 0.0 else 0.0


def _group_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["r_multiple"]) for row in rows]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    return {
        "trades": len(rows),
        "total_r": sum(values),
        "average_r": fmean(values) if values else 0.0,
        "median_r": median(values) if values else 0.0,
        "profit_factor": _profit_factor(values),
        "win_rate": len(wins) / len(values) if values else 0.0,
        "average_winner_r": fmean(wins) if wins else 0.0,
        "average_loser_r": fmean(losses) if losses else 0.0,
        "net_pnl": sum(float(row["net_pnl"]) for row in rows),
        "gross_pnl": sum(float(row["gross_pnl"]) for row in rows),
        "commission": sum(float(row["commission"]) for row in rows),
    }


def _grouped(
    rows: list[Mapping[str, Any]],
    key,
) -> dict[str, dict[str, Any]]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: _group_metrics(group) for name, group in sorted(groups.items())}


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = np.asarray(sorted(values), dtype=float)
    tail_count = max(1, int(math.ceil(len(ordered) * 0.05)))
    return {
        "minimum": float(ordered[0]),
        "p05": float(np.quantile(ordered, 0.05)),
        "p10": float(np.quantile(ordered, 0.10)),
        "p25": float(np.quantile(ordered, 0.25)),
        "median": float(np.quantile(ordered, 0.50)),
        "p75": float(np.quantile(ordered, 0.75)),
        "p90": float(np.quantile(ordered, 0.90)),
        "p95": float(np.quantile(ordered, 0.95)),
        "maximum": float(ordered[-1]),
        "standard_deviation": float(np.std(ordered)),
        "expected_shortfall_5pct": float(np.mean(ordered[:tail_count])),
    }


def _equity_diagnostics(equity_curve: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not equity_curve:
        return {}
    frame = pd.DataFrame(equity_curve)
    values = frame["mtm_equity"].astype(float).to_numpy()
    peaks = np.maximum.accumulate(values)
    drawdowns = values / np.maximum(peaks, 1e-9) - 1.0
    trough_index = int(np.argmin(drawdowns))
    peak_index = int(np.argmax(values[: trough_index + 1]))
    recovery_index = None
    peak_value = float(values[peak_index])
    for index in range(trough_index + 1, len(values)):
        if float(values[index]) >= peak_value:
            recovery_index = index
            break
    daily_returns = pd.Series(values).pct_change().dropna().to_numpy(dtype=float)
    daily_mean = float(np.mean(daily_returns)) if len(daily_returns) else 0.0
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    downside = daily_returns[daily_returns < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    elapsed_days = max(
        (date.fromisoformat(str(frame.iloc[-1]["date"])) - date.fromisoformat(str(frame.iloc[0]["date"]))).days,
        1,
    )
    total_return = float(values[-1] / values[0] - 1.0)
    cagr = (float(values[-1]) / float(values[0])) ** (365.25 / elapsed_days) - 1.0
    maximum_drawdown = abs(float(drawdowns[trough_index]))
    return {
        "start_equity": float(values[0]),
        "final_equity": float(values[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "maximum_drawdown_pct": maximum_drawdown,
        "drawdown_peak_date": str(frame.iloc[peak_index]["date"]),
        "drawdown_trough_date": str(frame.iloc[trough_index]["date"]),
        "drawdown_recovery_date": (
            str(frame.iloc[recovery_index]["date"]) if recovery_index is not None else None
        ),
        "drawdown_sessions": trough_index - peak_index,
        "recovery_sessions": (
            recovery_index - trough_index if recovery_index is not None else None
        ),
        "sharpe": daily_mean / daily_std * math.sqrt(252.0) if daily_std > 0.0 else 0.0,
        "sortino": daily_mean / downside_std * math.sqrt(252.0) if downside_std > 0.0 else 0.0,
        "calmar": cagr / maximum_drawdown if maximum_drawdown > 0.0 else 0.0,
        "maximum_open_positions": int(frame["open_positions"].max()),
        "average_open_positions": float(frame["open_positions"].mean()),
    }


def _path_diagnostics(bundle, rows: list[dict[str, Any]]) -> dict[str, Any]:
    mfe_values: list[float] = []
    mae_values: list[float] = []
    capture_values: list[float] = []
    positive_mfe_losers = 0
    for row in rows:
        start = pd.Timestamp(str(row["entry_date"]))
        end = pd.Timestamp(str(row["exit_date"]))
        path_high = bundle.high.loc[start:end, str(row["symbol"])].dropna()
        path_low = bundle.low.loc[start:end, str(row["symbol"])].dropna()
        risk_per_share = float(row["initial_risk_dollars"]) / max(int(row["qty_entry"]), 1)
        if path_high.empty or path_low.empty or risk_per_share <= 0.0:
            row["mfe_r"] = None
            row["mae_r"] = None
            row["mfe_capture_efficiency"] = None
            continue
        mfe = (float(path_high.max()) - float(row["entry_price"])) / risk_per_share
        mae = (float(path_low.min()) - float(row["entry_price"])) / risk_per_share
        capture = float(row["r_multiple"]) / max(mfe, 1e-9) if mfe > 0.0 else 0.0
        row["mfe_r"] = mfe
        row["mae_r"] = mae
        row["mfe_capture_efficiency"] = capture
        mfe_values.append(mfe)
        mae_values.append(mae)
        capture_values.append(capture)
        if float(row["r_multiple"]) < 0.0 and mfe > 0.30:
            positive_mfe_losers += 1
    losers = sum(float(row["r_multiple"]) < 0.0 for row in rows)
    return {
        "mfe_r": _distribution(mfe_values),
        "mae_r": _distribution(mae_values),
        "mfe_capture_efficiency": _distribution(capture_values),
        "losing_trades_with_mfe_above_0p30r": positive_mfe_losers,
        "share_of_losers_with_mfe_above_0p30r": (
            positive_mfe_losers / losers if losers else 0.0
        ),
    }


def _score_quintiles(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 25:
        return {"passed": False, "reason": "fewer_than_25_trades", "quintiles": {}}
    frame = pd.DataFrame(
        {"score": [float(row["score"]) for row in rows], "r": [float(row["r_multiple"]) for row in rows]}
    )
    frame["quintile"] = pd.qcut(frame["score"].rank(method="first"), 5, labels=False)
    values = {
        f"Q{int(index) + 1}": {
            "trades": len(group),
            "average_r": float(group["r"].mean()),
            "total_r": float(group["r"].sum()),
            "average_score": float(group["score"].mean()),
        }
        for index, group in frame.groupby("quintile")
    }
    passed = (
        values["Q5"]["average_r"] > values["Q3"]["average_r"]
        and values["Q5"]["average_r"] > values["Q1"]["average_r"] + 0.03
    )
    return {
        "passed": passed,
        "rank_correlation": float(frame[["score", "r"]].corr(method="spearman").iloc[0, 1]),
        "top_minus_bottom_r": values["Q5"]["average_r"] - values["Q1"]["average_r"],
        "top_minus_middle_r": values["Q5"]["average_r"] - values["Q3"]["average_r"],
        "quintiles": values,
    }


def _concentration(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if float(row["r_multiple"]) > 0.0]
    positive_total = sum(float(row["r_multiple"]) for row in positive)
    issuer_r: defaultdict[str, float] = defaultdict(float)
    sector_r: defaultdict[str, float] = defaultdict(float)
    for row in positive:
        issuer_r[issuer_key(str(row["symbol"]))] += float(row["r_multiple"])
        sector_r[str(row["sector"])] += float(row["r_multiple"])
    issuer_shares = {
        key: value / positive_total for key, value in issuer_r.items()
    } if positive_total else {}
    sector_shares = {
        key: value / positive_total for key, value in sector_r.items()
    } if positive_total else {}
    ordered = sorted((float(row["r_multiple"]) for row in rows), reverse=True)
    total = sum(ordered)
    return {
        "positive_r": positive_total,
        "top_positive_issuer_share": max(issuer_shares.values(), default=0.0),
        "top_positive_sector_share": max(sector_shares.values(), default=0.0),
        "issuer_positive_r_hhi": sum(value * value for value in issuer_shares.values()),
        "sector_positive_r_hhi": sum(value * value for value in sector_shares.values()),
        "top_positive_issuers": sorted(issuer_shares.items(), key=lambda item: (-item[1], item[0]))[:10],
        "top_positive_sectors": sorted(sector_shares.items(), key=lambda item: (-item[1], item[0])),
        "top_1_trade_total_r": sum(ordered[:1]),
        "top_5_trades_total_r": sum(ordered[:5]),
        "top_10_trades_total_r": sum(ordered[:10]),
        "top_20_trades_total_r": sum(ordered[:20]),
        "remainder_after_top_10_total_r": total - sum(ordered[:10]),
        "remainder_after_top_20_total_r": total - sum(ordered[:20]),
    }


def _render_report(payload: Mapping[str, Any]) -> str:
    metrics = payload["metrics"]
    folds = payload["fold_metrics"]
    score = payload["score_diagnostics"]
    concentration = payload["concentration"]
    path = payload["path_diagnostics"]
    equity = payload["equity_diagnostics"]
    opportunity = payload["opportunity_diagnostics"]
    exits = payload["exit_attribution"]
    monthly = payload["monthly"]
    costs = payload["cost_stress"]
    config = payload["baseline_config"]
    restart = payload.get("independent_restart_stress")
    failed_fold_gates = [name for name, passed in payload["fold_gates"].items() if not passed]
    positive_months = sum(float(row["total_r"]) > 0.0 for row in monthly.values())
    if opportunity.get("status") == "deferred_to_exact_phased_atlas":
        opportunity_lines = [
            "OPPORTUNITY AND REJECTION AUDIT",
            "Deferred to the factor-model-keyed phased atlas. The prior fast projection used a moving residual model and is not comparable with this frozen-model exact baseline.",
        ]
    else:
        opportunity_folds = opportunity["fold_metrics"]
        rejection_folds = opportunity["negative_rejection"]["folds"]
        opportunity_lines = [
            "OPPORTUNITY AND REJECTION AUDIT (FIXED-HORIZON PROJECTION)",
            f"Projected combined: {int(opportunity['metrics']['trades'])} trades, {opportunity['metrics']['total_r']:+.2f}R, avg {opportunity['metrics']['avg_r']:+.3f}R",
            f"Projected discovery: {int(opportunity_folds['discovery']['trades'])} trades, {opportunity_folds['discovery']['total_r']:+.2f}R, PF {opportunity_folds['discovery']['profit_factor']:.3f}",
            f"Projected calibration: {int(opportunity_folds['calibration']['trades'])} trades, {opportunity_folds['calibration']['total_r']:+.2f}R, PF {opportunity_folds['calibration']['profit_factor']:.3f}",
            f"Discovery accepted vs persistent-rejected avg: {rejection_folds['discovery']['accepted_avg_r']:+.3f}R vs {rejection_folds['discovery']['persistent_continuation_rejected_avg_r']:+.3f}R",
            f"Calibration accepted vs persistent-rejected avg: {rejection_folds['calibration']['accepted_avg_r']:+.3f}R vs {rejection_folds['calibration']['persistent_continuation_rejected_avg_r']:+.3f}R",
            f"Accepted beats rejected cohorts in both folds: {opportunity['negative_rejection']['passed_each_fold']}",
        ]
    lines = [
        "=" * 88,
        "IARIC RESIDUAL REVERSION — EXACT-98 FULL IN-SAMPLE STARTING BASELINE",
        "=" * 88,
        f"Contract: {payload['contract_id']}",
        f"Window: {DISCOVERY_START} through {CALIBRATION_END}",
        "Universe: exactly 98 traded stocks; SPY and 11 sector ETFs explanatory only",
        "Execution: shared live/replay core, next-session open, shared capital",
        f"Costs: {BASE_COST_BPS:.0f} bps round trip",
        "Locked validation accessed: no",
        "Sealed holdout accessed: no",
        "",
        "FROZEN PRE-SEARCH CONFIGURATION",
        f"Factor model: {config['daily_residual_factor_model']}",
        f"Formation / maximum hold: {config['daily_residual_formation_sessions']} / {config['daily_residual_maximum_holding_sessions']} sessions",
        f"Score components: {', '.join(config['daily_residual_score_components'])}",
        f"Minimum residual z: {config['daily_residual_minimum_z']:.2f}",
        f"Capacity: {config['daily_residual_max_positions']} positions, {config['daily_residual_max_positions_per_sector']} per sector, one per issuer",
        "",
        "HEADLINE PERFORMANCE",
        f"Trades: {metrics['trades']}",
        f"Trades/month: {metrics['trades_per_month']:.2f}",
        f"Total R: {metrics['total_r']:+.2f}",
        f"Average R: {metrics['average_r']:+.3f}",
        f"Profit factor: {metrics['profit_factor']:.3f}",
        f"Win rate: {metrics['win_rate']:.1%}",
        f"Net PnL: ${metrics['net_pnl']:,.2f}",
        f"Final equity: ${metrics['final_equity']:,.2f}",
        f"Return: {metrics['return_pct']:.2%}",
        f"CAGR: {equity['cagr']:.2%}",
        f"MTM max drawdown: {equity['maximum_drawdown_pct']:.2%}",
        f"Sharpe / Sortino / Calmar: {equity['sharpe']:.2f} / {equity['sortino']:.2f} / {equity['calmar']:.2f}",
        "",
        "CONTINUOUS-STATE CHRONOLOGICAL COHORTS (NO FOLD RESET)",
    ]
    for name in ("discovery", "calibration"):
        row = folds[name]
        lines.append(
            f"{name}: {int(row['trades'])} trades, {row['total_r']:+.2f}R, "
            f"avg {row['average_r']:+.3f}R, PF {row['profit_factor']:.3f}, "
            f"{row['trades_per_month']:.2f} trades/month, portfolio return "
            f"{row['return_pct']:+.2%}, carried-in positions "
            f"{int(row.get('carried_in_positions', 0))}, purged boundary entries "
            f"{int(row.get('purged_boundary_entry_count', 0))}"
        )
    lines.append(f"Failed exact-fold gates: {', '.join(failed_fold_gates) or 'none'}")
    if restart:
        lines.extend(
            [
                "",
                "INDEPENDENT-RESTART SENSITIVITY (NON-ECONOMIC SECONDARY TEST)",
                f"Continuous: {int(restart['continuous_trades'])} trades, {restart['continuous_total_r']:+.2f}R",
                f"Two reset folds: {int(restart['independently_reset_trades'])} trades, {restart['independently_reset_total_r']:+.2f}R",
                f"Reset minus continuous: {restart['reset_minus_continuous_r']:+.2f}R; restart stability {restart['restart_stability']:.3f}",
                f"Common entries: {int(restart['common_entry_trades'])}; continuous-only: {int(restart['continuous_unique_trades'])}; reset-only: {int(restart['reset_unique_trades'])}",
                "The reset total is never used as expected return or executable frequency.",
            ]
        )
    lines.extend(
        [
            "",
            "SIGNAL DISCRIMINATION",
            f"Score monotonicity passed: {score['passed']}",
            f"Score/outcome rank correlation: {score.get('rank_correlation', 0.0):+.3f}",
            f"Top-minus-bottom quintile: {score.get('top_minus_bottom_r', 0.0):+.3f}R",
            f"Top-minus-middle quintile: {score.get('top_minus_middle_r', 0.0):+.3f}R",
            "Combined score quintiles:",
            *[
                f"  {name}: {int(row['trades'])} trades, avg {row['average_r']:+.3f}R, total {row['total_r']:+.2f}R"
                for name, row in score["quintiles"].items()
            ],
            "",
            *opportunity_lines,
            "",
            "CONCENTRATION AND TAIL DEPENDENCE",
            f"Top issuer share of positive R: {concentration['top_positive_issuer_share']:.1%}",
            f"Top sector share of positive R: {concentration['top_positive_sector_share']:.1%}",
            f"Top 10 trades: {concentration['top_10_trades_total_r']:+.2f}R; remainder: {concentration['remainder_after_top_10_total_r']:+.2f}R",
            f"Top 20 trades: {concentration['top_20_trades_total_r']:+.2f}R; remainder: {concentration['remainder_after_top_20_total_r']:+.2f}R",
            "",
            "ENTRY AND MANAGEMENT",
            f"Average signal-close to next-open return: {payload['entry_delivery']['average_overnight_return']:+.3%}",
            f"Average open-to-final-exit return: {payload['entry_delivery']['average_post_open_return']:+.3%}",
            f"Losers that first achieved >+0.30R MFE: {path['losing_trades_with_mfe_above_0p30r']} ({path['share_of_losers_with_mfe_above_0p30r']:.1%} of losers)",
            "",
            "EXIT ATTRIBUTION",
            *[
                f"  {name}: {int(row['trades'])} trades, {row['total_r']:+.2f}R, avg {row['average_r']:+.3f}R, PF {row['profit_factor']:.3f}"
                for name, row in exits.items()
            ],
            "",
            "COST SENSITIVITY",
            f"  10 bps: {costs['10']['total_r']:+.2f}R, avg {costs['10']['average_r']:+.3f}R, PF {costs['10']['profit_factor']:.3f}",
            f"  20 bps: {metrics['total_r']:+.2f}R, avg {metrics['average_r']:+.3f}R, PF {metrics['profit_factor']:.3f}",
            f"  30 bps: {costs['30']['total_r']:+.2f}R, avg {costs['30']['average_r']:+.3f}R, PF {costs['30']['profit_factor']:.3f}",
            f"  40 bps: {costs['40']['total_r']:+.2f}R, avg {costs['40']['average_r']:+.3f}R, PF {costs['40']['profit_factor']:.3f}",
            "",
            "TEMPORAL STABILITY",
            f"Positive months: {positive_months}/{len(monthly)}",
            *[
                f"  {name}: {int(row['trades'])} trades, {row['total_r']:+.2f}R, avg {row['average_r']:+.3f}R"
                for name, row in monthly.items()
            ],
            "",
            "DRAWDOWN",
            f"Peak: {equity['drawdown_peak_date']}; trough: {equity['drawdown_trough_date']}; recovery: {equity['drawdown_recovery_date'] or 'not recovered'}",
            "",
            "BASELINE VERDICT",
            payload["verdict"],
            "The exit attribution separates fixed half-life or typed normalization value from catastrophic-stop and boundary cohorts. The baseline is representative only if the causal selector, entry delivery and management survive both folds after registered costs.",
            "",
            "This file is descriptive baseline evidence, not an optimized or promoted result.",
        ]
    )
    return "\n".join(lines) + "\n"


def _baseline_classification(fold_gates: Mapping[str, bool]) -> dict[str, Any]:
    """Separate a usable alpha baseline from a promotion-ready strategy.

    A baseline must establish positive, sufficiently broad, economically
    executable alpha on continuous state.  Imperfect score monotonicity is a
    weakness to optimize from, not a reason to pretend no baseline exists.
    Promotion remains blocked until every selection gate and the later locked
    validation contract pass.
    """

    baseline_gate_names = (
        "positive_each_fold",
        "positive_continuous_period",
        "positive_continuous_calibration_equity_return",
        "calibration_pf_gte_1p15",
        "calibration_average_r_gte_0p07",
        "at_least_100_trades_each_fold",
        "top_issuer_entry_risk_share_lte_15pct",
        "top_sector_entry_risk_share_lte_35pct",
    )
    representative = all(bool(fold_gates.get(name, False)) for name in baseline_gate_names)
    selection_gates_passed = all(bool(value) for value in fold_gates.values())
    if representative:
        verdict = (
            "This is the representative continuous-state alpha baseline for further "
            "optimization: economic breadth, calibration value, frequency and "
            "concentration gates pass. It is not promotion-ready; score discrimination "
            "and locked chronological validation remain explicit downstream gates."
        )
    else:
        verdict = (
            "The frozen configuration is auditable but not a representative alpha "
            "baseline because at least one continuous economic breadth, calibration, "
            "frequency or concentration gate fails."
        )
    return {
        "representative_alpha_baseline": representative,
        "selection_fold_qualification_complete": selection_gates_passed,
        "promotion_ready": False,
        "verdict": verdict,
    }


def run(
    output: Path,
    data_dir: Path,
    *,
    settings: StrategySettings | None = None,
    contract_id: str = BASELINE_CONTRACT_ID,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "progress.json",
        {
            "status": "loading_exact98_selection_panel",
            "contract_id": contract_id,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    authority = _attest_retained_local_research_snapshot(data_dir)
    if not authority["research_snapshot_certified"]:
        raise RuntimeError("local exact-98 daily snapshot failed attestation")
    close, open_, high, low, volume, sectors, paths = discovery._load_daily_panel(data_dir)
    if set(sectors) != set(BACKTESTED_INTRADAY_STOCK_SYMBOLS):
        raise RuntimeError("baseline panel is not the exact 98-name execution universe")
    data_fingerprint, fingerprint_rows = discovery._selection_data_fingerprint(
        close, open_, high, low, volume, paths
    )
    settings = settings or baseline_settings()
    config = _settings_payload(settings)
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    code_fingerprint, code_fingerprint_rows = _economic_code_fingerprint()
    bundle = build_daily_residual_replay_bundle(
        close,
        open_,
        high,
        low,
        volume,
        sectors,
        factor_model=settings.daily_residual_factor_model,
        source_fingerprint=data_fingerprint,
    )
    _write_json(
        output / "progress.json",
        {
            "status": "running_exact_shared_core_baseline",
            "contract_id": contract_id,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    combined = run_daily_residual_replay(
        bundle,
        settings,
        start=date.fromisoformat(DISCOVERY_START),
        end=date.fromisoformat(CALIBRATION_END),
        initial_equity=INITIAL_EQUITY,
        round_trip_cost_bps=BASE_COST_BPS,
    )
    exact_folds = run_exact_fold_evaluation(
        bundle,
        settings,
        round_trip_cost_bps=BASE_COST_BPS,
        include_independent_restart_stress=True,
        continuous_result=combined,
    )
    trade_rows = [asdict(trade) for trade in combined.trades]
    # Cost stress holds the causal trade set and sizing fixed and applies the
    # registered round-trip delta to entry notional. Replaying the identical
    # decision stream three more times added no information and doubled wall
    # time; exact cost-dependent replay remains part of Phase 6.
    cost_stress: dict[str, Any] = {}
    for cost in (10.0, 30.0, 40.0):
        delta = (float(cost) - BASE_COST_BPS) / 10_000.0
        stressed = []
        for row in trade_rows:
            risk_dollars = max(float(row["initial_risk_dollars"]), 1e-9)
            cost_delta = (
                delta
                * float(row["entry_price"])
                * int(row["qty_entry"])
                / risk_dollars
            )
            stressed.append(float(row["r_multiple"]) - cost_delta)
        gains = sum(value for value in stressed if value > 0.0)
        losses = abs(sum(value for value in stressed if value < 0.0))
        cost_stress[str(int(cost))] = {
            "method": "fixed_trade_set_round_trip_notional_delta",
            "trades": len(stressed),
            "total_r": sum(stressed),
            "average_r": fmean(stressed) if stressed else 0.0,
            "profit_factor": (
                gains / losses
                if losses > 0.0
                else (float("inf") if gains > 0.0 else 0.0)
            ),
        }
    path_diagnostics = _path_diagnostics(bundle, trade_rows)
    values = [float(row["r_multiple"]) for row in trade_rows]
    group = _group_metrics(trade_rows)
    metrics = {
        **combined.metrics(),
        **group,
        "trades_per_month": len(trade_rows) / _months(),
        "r_per_month": sum(values) / _months(),
        "r_distribution": _distribution(values),
        "average_holding_sessions": (
            fmean(float(row["held_sessions"]) for row in trade_rows) if trade_rows else 0.0
        ),
        "shared_core_contract": combined.shared_core_contract,
    }
    score_diagnostics = _score_quintiles(trade_rows)
    concentration = _concentration(trade_rows)
    equity_diagnostics = _equity_diagnostics(combined.equity_curve)
    entry_delivery = {
        "average_overnight_return": (
            fmean(float(trade.overnight_return) for trade in combined.trades)
            if combined.trades else 0.0
        ),
        "average_post_open_return": (
            fmean(float(trade.open_to_exit_return) for trade in combined.trades)
            if combined.trades else 0.0
        ),
        "average_signal_close_to_exit_return": (
            fmean(float(trade.signal_close_to_exit_return) for trade in combined.trades)
            if combined.trades else 0.0
        ),
    }
    decision_codes = dict(sorted(Counter(str(row["code"]) for row in combined.decision_events).items()))

    if contract_id == BASELINE_CONTRACT_ID:
        atlas = discovery.build_opportunity_atlas(
            close,
            open_,
            high,
            low,
            volume,
            sectors,
            factor_model=settings.daily_residual_factor_model,
        )
        approximate = discovery.evaluate_candidate(
            atlas,
            discovery.Candidate(
                candidate_id="presearch_exact98_diagnostic_projection",
                residual_z_floor=settings.daily_residual_minimum_z,
                holding_sessions=settings.daily_residual_maximum_holding_sessions,
                max_positions=settings.daily_residual_max_positions,
                max_positions_per_sector=settings.daily_residual_max_positions_per_sector,
                round_trip_cost_bps=BASE_COST_BPS,
                formation_sessions=settings.daily_residual_formation_sessions,
                diagnostic_leg="long_loser",
                factor_model=settings.daily_residual_factor_model,
                score_components=settings.daily_residual_score_components,
                lane_id=settings.daily_residual_lane_id,
                minimum_failed_continuation_r=(
                    settings.daily_residual_minimum_failed_continuation_r
                ),
                minimum_sector_return_5d=(
                    settings.daily_residual_minimum_sector_return_5d
                ),
                minimum_market_trend_z_20d=(
                    settings.daily_residual_minimum_market_trend_z_20d
                ),
            ),
        )
        opportunity_diagnostics = {
            key: value for key, value in approximate.items() if key != "trades"
        }
    else:
        opportunity_diagnostics = {
            "status": "deferred_to_exact_phased_atlas",
            "reason": (
                "the fast atlas uses a moving point-in-time model and is not "
                "used to judge the formation-frozen exact baseline"
            ),
        }
    monthly = _grouped(trade_rows, lambda row: str(row["entry_date"])[:7])
    symbols = _grouped(trade_rows, lambda row: row["symbol"])
    issuers = _grouped(trade_rows, lambda row: issuer_key(str(row["symbol"])))
    sectors_grouped = _grouped(trade_rows, lambda row: row["sector"])
    exits = _grouped(trade_rows, lambda row: row["exit_reason"])

    calibration = exact_folds["folds"]["calibration"]
    classification = _baseline_classification(exact_folds["gates"])
    representative = bool(classification["representative_alpha_baseline"])
    verdict = str(classification["verdict"])
    payload = {
        "contract_id": contract_id,
        "status": "complete_exact98_full_is_baseline",
        "representative_alpha_baseline": representative,
        "selection_fold_qualification_complete": classification[
            "selection_fold_qualification_complete"
        ],
        "promotion_ready": classification["promotion_ready"],
        "verdict": verdict,
        "window": {"start": DISCOVERY_START, "end": CALIBRATION_END},
        "locked_validation_start": "2025-08-01",
        "holdout_start": HOLDOUT_START,
        "locked_validation_accessed": False,
        "holdout_accessed": False,
        "traded_universe_count": len(sectors),
        "traded_universe": sorted(sectors),
        "non_traded_reference_count": 12,
        "data_fingerprint": data_fingerprint,
        "code_fingerprint": code_fingerprint,
        "fingerprinted_economic_code": code_fingerprint_rows,
        "config_sha256": config_sha,
        "baseline_config": config,
        "metrics": metrics,
        "fold_metrics": exact_folds["folds"],
        "fold_evaluation_contract": exact_folds["evaluation_contract"],
        "fold_boundary_diagnostics": exact_folds["fold_boundary_diagnostics"],
        "independent_restart_stress": exact_folds["independent_restart_stress"],
        "independent_restart_folds": exact_folds["independent_restart_folds"],
        "fold_score_quintiles": exact_folds["score_quintiles"],
        "fold_gates": exact_folds["gates"],
        "immutable_score": exact_folds["immutable_score"],
        "score_diagnostics": score_diagnostics,
        "concentration": concentration,
        "path_diagnostics": path_diagnostics,
        "entry_delivery": entry_delivery,
        "equity_diagnostics": equity_diagnostics,
        "cost_stress": cost_stress,
        "decision_event_codes": decision_codes,
        "opportunity_diagnostics": opportunity_diagnostics,
        "monthly": monthly,
        "exit_attribution": exits,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    _write_json(output / "baseline_config.json", {"contract_id": contract_id, "settings": config, "sha256": config_sha})
    _write_json(output / "baseline_data_contract.json", {**authority, "selection_fingerprint": data_fingerprint, "selection_inputs": fingerprint_rows})
    _write_json(output / "final_metrics.json", metrics)
    _write_json(output / "final_trades.json", trade_rows)
    _write_json(output / "final_equity_curve.json", combined.equity_curve)
    _write_json(output / "final_decision_events.json", combined.decision_events)
    _write_json(output / "final_fold_metrics.json", {
        "evaluation_contract": exact_folds["evaluation_contract"],
        "continuous_metrics": exact_folds["continuous_metrics"],
        "folds": exact_folds["folds"],
        "fold_boundary_diagnostics": exact_folds["fold_boundary_diagnostics"],
        "score_quintiles": exact_folds["score_quintiles"],
        "gates": exact_folds["gates"],
        "immutable_score": exact_folds["immutable_score"],
        "independent_restart_stress": exact_folds["independent_restart_stress"],
        "independent_restart_folds": exact_folds["independent_restart_folds"],
    })
    _write_json(output / "final_fold_trades.json", exact_folds["trades"])
    _write_json(output / "final_fold_equity_curves.json", exact_folds["equity_curves"])
    _write_json(
        output / "final_independent_restart_fold_trades.json",
        exact_folds["independent_restart_trades"],
    )
    _write_json(output / "final_monthly.json", monthly)
    _write_json(output / "final_symbols.json", symbols)
    _write_json(output / "final_issuers.json", issuers)
    _write_json(output / "final_sectors.json", sectors_grouped)
    _write_json(output / "final_exits.json", exits)
    _write_json(output / "final_score_diagnostics.json", score_diagnostics)
    _write_json(output / "final_concentration.json", concentration)
    _write_json(output / "final_path_diagnostics.json", path_diagnostics)
    _write_json(output / "final_entry_delivery.json", entry_delivery)
    _write_json(output / "final_drawdown_diagnostics.json", equity_diagnostics)
    _write_json(output / "final_cost_stress.json", cost_stress)
    _write_json(output / "final_opportunity_diagnostics.json", opportunity_diagnostics)
    _write_json(output / "run_summary.json", payload)
    report_path = output / "round_final_diagnostics.txt"
    _write_text(report_path, _render_report(payload))
    _write_json(
        output / "progress.json",
        {
            "status": "complete_exact98_full_is_baseline",
            "contract_id": contract_id,
            "representative_alpha_baseline": representative,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "completed_at_utc": payload["completed_at_utc"],
        },
    )
    finalize_artifact_manifest(output, contract_id=contract_id)
    return payload


def finalize_artifact_manifest(
    output: Path, *, contract_id: str = BASELINE_CONTRACT_ID
) -> None:
    """Hash only finalized artifacts; the manifest intentionally excludes itself."""

    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(
        output / "artifact_manifest.json",
        {
            "contract_id": contract_id,
            "artifacts": artifact_hashes,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )


def finalize_existing_report(output: Path) -> None:
    """Regenerate the human-readable report from an already completed replay."""

    payload = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    config_payload = json.loads((output / "baseline_config.json").read_text(encoding="utf-8"))
    payload["baseline_config"] = config_payload["settings"]
    payload["monthly"] = json.loads((output / "final_monthly.json").read_text(encoding="utf-8"))
    payload["exit_attribution"] = json.loads((output / "final_exits.json").read_text(encoding="utf-8"))
    classification = _baseline_classification(payload["fold_gates"])
    payload.update(classification)
    _write_json(output / "run_summary.json", payload)
    _write_text(output / "round_final_diagnostics.txt", _render_report(payload))
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress.update(
            {
                "representative_alpha_baseline": classification[
                    "representative_alpha_baseline"
                ],
                "selection_fold_qualification_complete": classification[
                    "selection_fold_qualification_complete"
                ],
                "promotion_ready": classification["promotion_ready"],
            }
        )
        _write_json(progress_path, progress)
    finalize_artifact_manifest(
        output,
        contract_id=str(payload.get("contract_id", BASELINE_CONTRACT_ID)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=discovery.DEFAULT_DATA_DIR)
    parser.add_argument("--representative-v2", action="store_true")
    parser.add_argument(
        "--selected-config",
        type=Path,
        help="selected_baseline_config.json produced by the phased exact selector",
    )
    args = parser.parse_args()
    if args.selected_config:
        selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
        settings = settings_from_discovery_candidate(selected["candidate"])
        contract_id = SELECTED_REPRESENTATIVE_CONTRACT_ID
    else:
        settings = (
            representative_settings() if args.representative_v2 else baseline_settings()
        )
        contract_id = (
            REPRESENTATIVE_CONTRACT_ID
            if args.representative_v2
            else BASELINE_CONTRACT_ID
        )
    output = args.output_dir or (
        REPRESENTATIVE_OUTPUT if args.representative_v2 else DEFAULT_OUTPUT
    )
    payload = run(
        output.resolve(),
        args.data_dir.resolve(),
        settings=settings,
        contract_id=contract_id,
    )
    print(json.dumps({
        "status": payload["status"],
        "representative_alpha_baseline": payload["representative_alpha_baseline"],
        "metrics": payload["metrics"],
        "fold_metrics": payload["fold_metrics"],
        "output": str(output.resolve()),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
