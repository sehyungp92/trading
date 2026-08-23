"""Run ALCB Phase 0: frozen-config replay validity and re-baselining.

This is intentionally not an optimization.  It reproduces the legacy
extended-hours cache semantics, switches only the replay session to the
versioned RTH policy, and then measures temporal and execution-cost stability.
Data after the consumed OOS end date is not evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin
from backtests.stock.data.calendar import (
    CALENDAR_VERSION,
    RAW_SESSION_POLICY,
    RTH_SESSION_POLICY,
    bar_open_in_session,
)


_ET = ZoneInfo("America/New_York")
_DEFAULT_CONFIG = REPO_ROOT / "backtests/output/stock/alcb/round_4/optimized_config.json"
_DEFAULT_OUTPUT = REPO_ROOT / "backtests/output/stock/alcb/phase_0_validity_20260816"
_ROUND4_METRICS = REPO_ROOT / "backtests/output/stock/alcb/round_4/final_metrics.json"
_READINESS = REPO_ROOT / "backtests/stock/data/authority/readiness/may_is_june_oos.json"
_PROJECTION = (
    REPO_ROOT
    / "backtests/stock/data/authority/derived/legacy_eth_to_rth"
    / "legacy_extended_through_2026-07-10.projection.json"
)

_IS = ("2024-03-25", "2026-03-01")
_CONSUMED_OOS = ("2026-03-02", "2026-05-01")
_FOLDS = (
    ("fold_1", "2024-03-25", "2024-09-30"),
    ("fold_2", "2024-10-01", "2025-03-31"),
    ("fold_3", "2025-04-01", "2025-09-30"),
    ("fold_4", "2025-10-01", "2026-03-01"),
)
_CORE_METRICS = (
    "total_trades",
    "trades_per_month",
    "win_rate",
    "expectancy",
    "expected_total_r",
    "expectancy_dollar",
    "net_profit",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe",
    "sortino",
    "mfe_capture_efficiency",
    "profit_protection",
    "signal_quality",
    "score_monotonicity",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _serialize_trade(trade) -> dict[str, Any]:
    payload = asdict(trade)
    payload["direction"] = int(trade.direction)
    payload["pnl_net"] = float(trade.pnl_net)
    payload["hold_hours"] = float(trade.hold_hours)
    payload["is_winner"] = bool(trade.is_winner)
    return payload


def _r_profit_factor(trades: Iterable) -> float:
    values = [float(trade.r_multiple) for trade in trades]
    gross_win = sum(value for value in values if value > 0)
    gross_loss = abs(sum(value for value in values if value < 0))
    return gross_win / gross_loss if gross_loss > 0 else math.inf


def _daily_r_summary(trades: list, trading_dates: list[date]) -> dict[str, float]:
    daily = {day: 0.0 for day in trading_dates}
    for trade in trades:
        day = trade.exit_time.astimezone(_ET).date()
        if day in daily:
            daily[day] += float(trade.r_multiple)
    values = np.asarray(list(daily.values()), dtype=float)
    if values.size == 0:
        return {
            "daily_r_mean": 0.0,
            "daily_r_std": 0.0,
            "worst_decile_daily_r_cvar": 0.0,
            "negative_day_rate": 0.0,
        }
    cutoff_n = max(1, int(math.ceil(values.size * 0.10)))
    worst = np.sort(values)[:cutoff_n]
    return {
        "daily_r_mean": float(values.mean()),
        "daily_r_std": float(values.std(ddof=0)),
        "worst_decile_daily_r_cvar": float(worst.mean()),
        "negative_day_rate": float((values < 0).mean()),
    }


def _group_summary(trades: list, key_fn) -> dict[str, dict[str, float]]:
    groups: dict[str, list] = defaultdict(list)
    for trade in trades:
        groups[str(key_fn(trade))].append(trade)
    payload: dict[str, dict[str, float]] = {}
    for key, rows in sorted(groups.items()):
        r_values = [float(row.r_multiple) for row in rows]
        payload[key] = {
            "trades": len(rows),
            "win_rate": sum(value > 0 for value in r_values) / len(rows),
            "avg_r": fmean(r_values),
            "total_r": sum(r_values),
            "net_profit": sum(float(row.pnl_net) for row in rows),
            "dollar_pf": (
                sum(max(0.0, float(row.pnl_net)) for row in rows)
                / abs(sum(min(0.0, float(row.pnl_net)) for row in rows))
                if any(float(row.pnl_net) < 0 for row in rows)
                else math.inf
            ),
            "r_pf": _r_profit_factor(rows),
        }
    return payload


def _feature_deciles(trades: list) -> dict[str, list[dict[str, float]]]:
    features = (
        "selection_score",
        "relative_strength_percentile",
        "accumulation_score",
        "signal_cpr",
        "daily_adx",
        "gap_pct",
        "avwap_distance_pct",
        "or_width_pct",
        "signal_range_r",
        "breakout_distance_r",
        "signal_risk_pct",
        "entry_signal_rvol",
        "orb_quality_score",
        "signal_minutes_et",
    )
    output: dict[str, list[dict[str, float]]] = {}
    for feature in features:
        rows = [
            (float(trade.metadata[feature]), float(trade.r_multiple))
            for trade in trades
            if feature in trade.metadata
            and math.isfinite(float(trade.metadata[feature]))
        ]
        if len(rows) < 20 or len({value for value, _ in rows}) < 2:
            continue
        rows.sort(key=lambda pair: pair[0])
        chunks = np.array_split(np.asarray(rows, dtype=float), min(10, len(rows)))
        deciles: list[dict[str, float]] = []
        for index, chunk in enumerate(chunks, start=1):
            if chunk.size == 0:
                continue
            r_values = chunk[:, 1]
            deciles.append({
                "bucket": index,
                "minimum": float(chunk[:, 0].min()),
                "maximum": float(chunk[:, 0].max()),
                "trades": int(len(chunk)),
                "win_rate": float((r_values > 0).mean()),
                "avg_r": float(r_values.mean()),
                "total_r": float(r_values.sum()),
            })
        output[feature] = deciles
    return output


def _analysis(context: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    metrics = context["metrics"]
    trades = context["trades"]
    replay = context["replay"]
    trade_days = replay.tradable_dates(date.fromisoformat(start), date.fromisoformat(end))
    summary = {key: float(metrics.get(key, 0.0) or 0.0) for key in _CORE_METRICS}
    summary["r_profit_factor"] = _r_profit_factor(trades)
    summary.update(_daily_r_summary(trades, trade_days))
    summary["first_entry_time"] = min((trade.entry_time for trade in trades), default=None)
    summary["last_exit_time"] = max((trade.exit_time for trade in trades), default=None)
    return {
        "window": {"start": start, "end": end, "trading_days": len(trade_days)},
        "summary": summary,
        "by_entry_type": _group_summary(trades, lambda trade: trade.entry_type),
        "by_exit_reason": _group_summary(trades, lambda trade: trade.exit_reason),
        "by_momentum_score": _group_summary(
            trades,
            lambda trade: trade.metadata.get("momentum_score", "missing"),
        ),
        "by_entry_hour_et": _group_summary(
            trades,
            lambda trade: trade.entry_time.astimezone(_ET).strftime("%H"),
        ),
        "by_symbol": _group_summary(trades, lambda trade: trade.symbol),
        "feature_deciles": _feature_deciles(trades),
        "shadow_funnel": (
            context["shadow_tracker"].funnel if context.get("shadow_tracker") else {}
        ),
        "shadow_rejections": (
            {
                gate: {
                    "opportunities": len(rows),
                    "avg_synthetic_r": fmean(float(row.simulated_r) for row in rows),
                    "positive_rate": sum(float(row.simulated_r) > 0 for row in rows) / len(rows),
                }
                for gate, rows in context["shadow_tracker"].get_filter_summary().items()
                if rows
            }
            if context.get("shadow_tracker")
            else {}
        ),
    }


def _delta(corrected: dict[str, Any], legacy: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key, corrected_value in corrected["summary"].items():
        if not isinstance(corrected_value, (int, float)):
            continue
        legacy_value = legacy["summary"].get(key)
        if not isinstance(legacy_value, (int, float)):
            continue
        absolute = float(corrected_value) - float(legacy_value)
        pct = absolute / abs(float(legacy_value)) if float(legacy_value) != 0 else 0.0
        result[key] = {
            "legacy": float(legacy_value),
            "rth": float(corrected_value),
            "absolute": absolute,
            "relative": pct,
        }
    return result


def _trade_overlap(corrected_trades: list, legacy_trades: list) -> dict[str, Any]:
    def keys(trades: list) -> set[tuple[str, str, int]]:
        return {
            (
                trade.entry_time.astimezone(_ET).date().isoformat(),
                trade.symbol,
                int(trade.reentry_sequence),
            )
            for trade in trades
        }

    corrected_keys = keys(corrected_trades)
    legacy_keys = keys(legacy_trades)
    union = corrected_keys | legacy_keys
    intersection = corrected_keys & legacy_keys
    return {
        "legacy_unique_trade_keys": len(legacy_keys),
        "rth_unique_trade_keys": len(corrected_keys),
        "shared_trade_keys": len(intersection),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "rth_only": len(corrected_keys - legacy_keys),
        "legacy_only": len(legacy_keys - corrected_keys),
    }


def _session_audit(replay) -> dict[str, Any]:
    checks = []
    for symbol in ("AMD", "NVDA", "AAPL"):
        for day in (date(2024, 3, 25), date(2024, 7, 3), date(2026, 4, 1)):
            raw = replay.get_5m_bar_objects_for_date(symbol, day)
            rth = [bar for bar in raw if bar_open_in_session(bar.start_time, RTH_SESSION_POLICY)]
            checks.append({
                "symbol": symbol,
                "date": day.isoformat(),
                "raw_bars": len(raw),
                "rth_bars": len(rth),
                "raw_first_et": raw[0].start_time.astimezone(_ET).isoformat() if raw else None,
                "raw_last_et": raw[-1].start_time.astimezone(_ET).isoformat() if raw else None,
                "rth_first_et": rth[0].start_time.astimezone(_ET).isoformat() if rth else None,
                "rth_last_et": rth[-1].start_time.astimezone(_ET).isoformat() if rth else None,
            })
    return {"calendar_version": CALENDAR_VERSION, "checks": checks}


def _run_context(
    plugin: ALCBP16Plugin,
    mutations: dict[str, Any],
    *,
    start: str,
    end: str,
    label: str,
    diagnostics: bool,
) -> dict[str, Any]:
    print(f"START {label}: {start} to {end}", flush=True)
    started = datetime.now(timezone.utc)
    context = plugin._run_config(
        mutations,
        start_date=start,
        end_date=end,
        store_context=diagnostics,
        collect_diagnostics=diagnostics,
    )
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    metrics = context["metrics"]
    print(
        f"DONE  {label}: trades={metrics['total_trades']:.0f} "
        f"R={metrics['expected_total_r']:+.2f} net=${metrics['net_profit']:+,.2f} "
        f"PF={metrics['profit_factor']:.3f} DD={metrics['max_drawdown_pct']:.2%} "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return context


def _score_spec(
    rth_is: dict[str, Any],
    fold_analyses: list[dict[str, Any]],
    cost_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = rth_is["summary"]
    worst_fold_avg_r = min(
        (fold["summary"]["expectancy"] for fold in fold_analyses),
        default=summary["expectancy"],
    )
    baselines = {
        "expected_total_r": summary["expected_total_r"],
        "net_profit": summary["net_profit"],
        "trades_per_month": summary["trades_per_month"],
        "profit_factor": summary["profit_factor"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "worst_decile_daily_r_loss": abs(min(0.0, summary["worst_decile_daily_r_cvar"])),
        "worst_fold_avg_r": worst_fold_avg_r,
        "cost_7p5_expected_total_r": next(
            row["analysis"]["summary"]["expected_total_r"]
            for row in cost_payloads
            if row["slip_bps"] == 7.5
        ),
    }
    return {
        "name": "alcb_rth_baseline_relative_tanh_v1",
        "formula": {
            "positive_ratio": "0.5 + 0.5*tanh(log(max(x,1e-9)/max(B,1e-9))/k)",
            "inverse_ratio": "0.5 - 0.5*tanh(log(max(x,1e-9)/max(B,1e-9))/k)",
            "signed_fold": "0.5 + 0.5*tanh((x-B)/0.10R)",
        },
        "components": {
            "expected_total_r": {"weight": 0.23, "k": 0.18, "direction": "positive_ratio"},
            "net_profit": {"weight": 0.12, "k": 0.20, "direction": "positive_ratio"},
            "trades_per_month": {"weight": 0.15, "k": 0.20, "direction": "positive_ratio"},
            "profit_factor": {"weight": 0.12, "k": 0.15, "direction": "positive_ratio"},
            "max_drawdown_pct": {"weight": 0.13, "k": 0.25, "direction": "inverse_ratio"},
            "worst_decile_daily_r_loss": {"weight": 0.07, "k": 0.25, "direction": "inverse_ratio"},
            "worst_fold_avg_r": {"weight": 0.10, "scale_r": 0.10, "direction": "signed_fold"},
            "cost_7p5_expected_total_r": {"weight": 0.08, "k": 0.30, "direction": "positive_ratio"},
        },
        "baseline_values": baselines,
        "baseline_score": 0.5,
        "hard_rejects": {
            "expected_total_r_floor": 0.85 * summary["expected_total_r"],
            "net_profit_floor": 0.80 * summary["net_profit"],
            "trades_per_month_floor": 0.90 * summary["trades_per_month"],
            "profit_factor_floor": max(1.10, 0.95 * summary["profit_factor"]),
            "max_drawdown_ceiling": min(0.10, 1.55 * summary["max_drawdown_pct"]),
            "minimum_fold_avg_r": -0.01,
            "cost_7p5_expected_total_r_floor": 0.0,
            "cost_7p5_profit_factor_floor": 1.02,
            "consumed_oos_expected_total_r_floor": 0.0,
        },
        "promotion_gates": {
            "cost_10_expected_total_r_floor": 0.0,
            "cost_10_profit_factor_floor": 1.0,
            "fresh_holdout_required": True,
            "direct_rth_bundle_required": True,
        },
        "notes": [
            "All scales are frozen to the corrected RTH Phase-0 baseline, not historical saturated ranges.",
            "The score is comparative; a value of 0.5 is the frozen baseline by construction.",
            "Hard rejects apply before scoring and prevent frequency/return gains from buying unacceptable tail risk.",
            "The 7.5 bps component is evaluated on a full causal replay, not a linear cost haircut.",
            "The baseline passes all search hard rejects; stricter 10 bps and fresh-holdout checks are promotion gates.",
        ],
    }


def _format_report(payload: dict[str, Any]) -> str:
    is_legacy = payload["runs"]["is_legacy"]["analysis"]["summary"]
    is_rth = payload["runs"]["is_rth"]["analysis"]["summary"]
    oos_legacy = payload["runs"]["oos_legacy"]["analysis"]["summary"]
    oos_rth = payload["runs"]["oos_rth"]["analysis"]["summary"]
    overlap = payload["comparisons"]["is_trade_overlap"]
    control = payload["comparisons"]["round4_control"]
    folds = payload["robustness"]["folds"]
    costs = payload["robustness"]["cost_stress"]

    def row(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"| {label} | {int(metrics['total_trades'])} | {metrics['trades_per_month']:.2f} | "
            f"{metrics['win_rate']:.1%} | {metrics['expectancy']:+.4f} | "
            f"{metrics['expected_total_r']:+.2f} | ${metrics['net_profit']:+,.2f} | "
            f"{metrics['profit_factor']:.3f} | {metrics['r_profit_factor']:.3f} | "
            f"{metrics['max_drawdown_pct']:.2%} |"
        )

    lines = [
        "# ALCB Phase 0 validity report",
        "",
        "## Scope and validity status",
        "",
        "The Round 4 optimized parameters were frozen. The only structural replay change was the session policy: the legacy control consumes the raw extended-hours cache, while the corrected run uses versioned US-equity RTH for both 30-minute research and 5-minute execution. The 2026-03-02 to 2026-05-01 window was already consumed during development and is labelled accordingly. No data after 2026-05-01 was evaluated.",
        "",
        f"Legacy control reproduction: {control['verdict']}. Maximum core-metric absolute error versus the saved Round 4 result was {control['max_core_metric_abs_error']:.8f}.",
        "",
        "## Headline results",
        "",
        "| Window | Trades | Trades/mo | Win rate | Avg R | Total R | Net | $ PF | R PF | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("IS legacy", is_legacy),
        row("IS corrected RTH", is_rth),
        row("Consumed OOS legacy", oos_legacy),
        row("Consumed OOS corrected RTH", oos_rth),
        "",
        "## Signal-set drift",
        "",
        f"The corrected IS run shares {overlap['shared_trade_keys']} symbol/date/re-entry keys with the legacy run (Jaccard {overlap['jaccard']:.1%}); {overlap['rth_only']} are RTH-only and {overlap['legacy_only']} are legacy-only. This is a structural signal-definition change, not a small execution adjustment.",
        "",
        "## Chronological stability",
        "",
        "| Fold | Trades | Trades/mo | Avg R | Total R | $ PF | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in folds:
        metrics = fold["analysis"]["summary"]
        lines.append(
            f"| {fold['name']} | {int(metrics['total_trades'])} | {metrics['trades_per_month']:.2f} | "
            f"{metrics['expectancy']:+.4f} | {metrics['expected_total_r']:+.2f} | "
            f"{metrics['profit_factor']:.3f} | {metrics['max_drawdown_pct']:.2%} |"
        )
    lines.extend([
        "",
        "## Cost stress on corrected IS",
        "",
        "| One-way slippage | Trades | Avg R | Total R | Net | $ PF | Max DD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for cost in costs:
        metrics = cost["analysis"]["summary"]
        lines.append(
            f"| {cost['slip_bps']:.1f} bps | {int(metrics['total_trades'])} | "
            f"{metrics['expectancy']:+.4f} | {metrics['expected_total_r']:+.2f} | "
            f"${metrics['net_profit']:+,.2f} | {metrics['profit_factor']:.3f} | "
            f"{metrics['max_drawdown_pct']:.2%} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Phase 1 implications",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["implications"])
    lines.extend([
        "",
        "## Immutable scoring",
        "",
        "The next phase should use the baseline-relative tanh score in `immutable_score.json`. Its corrected RTH baseline is exactly 0.5 by construction; improvements therefore remain sensitive around the current operating point instead of saturating obsolete ranges. Hard rejects are evaluated before the weighted score.",
        "",
        "## Artefact status",
        "",
        "These results are diagnostic because the cache is a deterministic projection of legacy extended-hours data, not a frozen direct-IBKR RTH acquisition. They are sufficient to invalidate or re-baseline the old signal semantics, but not to authorize production promotion. A direct-RTH bundle remains the final data-authority requirement.",
        "",
    ])
    return "\n".join(lines)


def _interpretation(payload: dict[str, Any]) -> tuple[str, list[str]]:
    legacy = payload["runs"]["is_legacy"]["analysis"]["summary"]
    rth = payload["runs"]["is_rth"]["analysis"]["summary"]
    oos = payload["runs"]["oos_rth"]["analysis"]["summary"]
    folds = [row["analysis"]["summary"] for row in payload["robustness"]["folds"]]
    r_delta = rth["expected_total_r"] - legacy["expected_total_r"]
    freq_delta = rth["trades_per_month"] - legacy["trades_per_month"]
    dd_delta = rth["max_drawdown_pct"] - legacy["max_drawdown_pct"]
    weakest_fold = min(folds, key=lambda row: row["expectancy"])

    if rth["expectancy"] <= 0 or rth["profit_factor"] <= 1:
        verdict = "The corrected session definition removes the apparent aggregate edge"
    elif rth["expected_total_r"] < 0.75 * legacy["expected_total_r"]:
        verdict = "A material part of the reported Round 4 alpha depends on contaminated session geometry"
    elif rth["expected_total_r"] > 1.10 * legacy["expected_total_r"]:
        verdict = "Correct RTH geometry reveals materially more alpha than the legacy replay"
    else:
        verdict = "The core edge survives the session correction, although the traded opportunity set changes materially"

    interpretation = (
        f"{verdict}. On IS, corrected RTH changes total R by {r_delta:+.2f}, frequency by "
        f"{freq_delta:+.2f} trades/month, and max drawdown by {dd_delta:+.2%}. "
        f"The weakest chronological fold has {weakest_fold['expectancy']:+.4f}R expectancy and "
        f"PF {weakest_fold['profit_factor']:.3f}. The already-consumed corrected OOS window records "
        f"{oos['expected_total_r']:+.2f}R at PF {oos['profit_factor']:.3f}; it is supporting evidence only, "
        "not a fresh confirmation."
    )

    implications = [
        "Treat the corrected RTH metrics, not Round 4's legacy metrics, as the immutable baseline for every later experiment.",
        "Do not tune the opening-range length immediately. First test structural signal discrimination on continuous features (RVOL, CPR, AVWAP distance, OR width, breakout distance, gap, ADX and selection strength) using broad buckets and chronological retention.",
        "Keep next-bar-open entry as the causal control. Test fill-time invalidation and pullback/reclaim alternatives as separate mechanisms; do not mix them into threshold sweeps.",
        "Use the newly attributed stop exits to isolate initial-stop, failure-stop, breakeven and adaptive-trail behaviour before changing management. Prior CLOSE_STOP aggregation could not identify which mechanism helped or hurt.",
        "Require improvements to survive all four IS folds, the consumed OOS diagnostic, and at least 7.5/10 bps cost stress. Preserve the untouched post-May-2026 interval for final confirmation.",
        "Keep synthetic rejected-signal outcomes diagnostic-only: they are now unique per armed opportunity, but still use a simplified 1.5R/stop counterfactual rather than full portfolio replay.",
    ]
    return interpretation, implications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "backtests/stock/data/raw")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Explicitly permit diagnostic-only legacy data. ResearchReplayEngine still
    # rejects it by default when production data authority is required.
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "0"

    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_mutations = json.loads(config_path.read_text(encoding="utf-8"))
    legacy_mutations = {**base_mutations, "intraday_session_policy": RAW_SESSION_POLICY}
    rth_mutations = {**base_mutations, "intraday_session_policy": RTH_SESSION_POLICY}

    plugin = ALCBP16Plugin(
        data_dir,
        start_date=_IS[0],
        end_date=_IS[1],
        initial_equity=10_000.0,
        max_workers=1,
    )

    is_legacy_ctx = _run_context(
        plugin, legacy_mutations, start=_IS[0], end=_IS[1], label="IS legacy control", diagnostics=True,
    )
    is_rth_ctx = _run_context(
        plugin, rth_mutations, start=_IS[0], end=_IS[1], label="IS corrected RTH", diagnostics=True,
    )
    oos_legacy_ctx = _run_context(
        plugin,
        legacy_mutations,
        start=_CONSUMED_OOS[0],
        end=_CONSUMED_OOS[1],
        label="consumed OOS legacy control",
        diagnostics=True,
    )
    oos_rth_ctx = _run_context(
        plugin,
        rth_mutations,
        start=_CONSUMED_OOS[0],
        end=_CONSUMED_OOS[1],
        label="consumed OOS corrected RTH",
        diagnostics=True,
    )

    contexts = {
        "is_legacy": (is_legacy_ctx, *_IS),
        "is_rth": (is_rth_ctx, *_IS),
        "oos_legacy": (oos_legacy_ctx, *_CONSUMED_OOS),
        "oos_rth": (oos_rth_ctx, *_CONSUMED_OOS),
    }
    runs: dict[str, Any] = {}
    for name, (context, start, end) in contexts.items():
        analysis = _analysis(context, start, end)
        runs[name] = {
            "analysis": analysis,
            "session_policy": context["config"].intraday_session_policy,
        }
        _write_json(output_dir / f"{name}_analysis.json", analysis)
        _write_json(
            output_dir / f"{name}_trades.json",
            [_serialize_trade(trade) for trade in context["trades"]],
        )
        diagnostic = alcb_full_diagnostic(
            context["trades"],
            shadow_tracker=context.get("shadow_tracker"),
            daily_selections=context.get("daily_selections"),
        )
        (output_dir / f"{name}_diagnostics.txt").write_text(diagnostic, encoding="utf-8")

    fold_payloads = []
    fold_analyses = []
    for name, start, end in _FOLDS:
        context = _run_context(
            plugin,
            rth_mutations,
            start=start,
            end=end,
            label=f"RTH {name}",
            diagnostics=False,
        )
        analysis = _analysis(context, start, end)
        fold_analyses.append(analysis)
        fold_payloads.append({"name": name, "analysis": analysis})

    cost_payloads = []
    for slip_bps in (7.5, 10.0, 15.0):
        mutations = {**rth_mutations, "slippage.slip_bps_normal": slip_bps}
        context = _run_context(
            plugin,
            mutations,
            start=_IS[0],
            end=_IS[1],
            label=f"RTH cost {slip_bps:.1f}bps",
            diagnostics=False,
        )
        cost_payloads.append({
            "slip_bps": slip_bps,
            "analysis": _analysis(context, *_IS),
        })

    saved_round4 = json.loads(_ROUND4_METRICS.read_text(encoding="utf-8"))
    legacy_summary = runs["is_legacy"]["analysis"]["summary"]
    control_errors = {
        key: float(legacy_summary[key]) - float(saved_round4[key])
        for key in (
            "total_trades",
            "trades_per_month",
            "win_rate",
            "expectancy",
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "max_drawdown_pct",
        )
    }
    max_error = max(abs(value) for value in control_errors.values())

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "ALCB Phase 0 frozen-config session validity and corrected RTH re-baseline",
        "optimization_performed": False,
        "holdout_policy": {
            "consumed_oos": {"start": _CONSUMED_OOS[0], "end": _CONSUMED_OOS[1]},
            "untouched_after": _CONSUMED_OOS[1],
            "post_may_2026_opened": False,
        },
        "provenance": {
            "optimized_config": str(config_path),
            "optimized_config_sha256": _sha256(config_path),
            "base_mutations": base_mutations,
            "data_dir": str(data_dir),
            "data_fingerprint": is_rth_ctx["cache_source_fingerprint"],
            "data_authoritative": bool(is_rth_ctx["replay"].authoritative_data),
            "data_bundle_context": is_rth_ctx["replay"].data_bundle_context(),
            "calendar_version": CALENDAR_VERSION,
            "legacy_session_policy": RAW_SESSION_POLICY,
            "corrected_session_policy": RTH_SESSION_POLICY,
            "readiness_file": str(_READINESS) if _READINESS.exists() else None,
            "readiness_sha256": _sha256(_READINESS) if _READINESS.exists() else None,
            "projection_manifest": str(_PROJECTION) if _PROJECTION.exists() else None,
            "projection_manifest_sha256": _sha256(_PROJECTION) if _PROJECTION.exists() else None,
        },
        "session_audit": _session_audit(is_rth_ctx["replay"]),
        "runs": runs,
        "comparisons": {
            "is_rth_vs_legacy": _delta(
                runs["is_rth"]["analysis"], runs["is_legacy"]["analysis"]
            ),
            "oos_rth_vs_legacy": _delta(
                runs["oos_rth"]["analysis"], runs["oos_legacy"]["analysis"]
            ),
            "is_trade_overlap": _trade_overlap(is_rth_ctx["trades"], is_legacy_ctx["trades"]),
            "round4_control": {
                "errors": control_errors,
                "max_core_metric_abs_error": max_error,
                "verdict": "PASS" if max_error < 1e-6 else "FAIL",
            },
        },
        "robustness": {
            "folds": fold_payloads,
            "cost_stress": cost_payloads,
        },
    }
    payload["immutable_score"] = _score_spec(
        runs["is_rth"]["analysis"],
        fold_analyses,
        cost_payloads,
    )
    interpretation, implications = _interpretation(payload)
    payload["interpretation"] = interpretation
    payload["implications"] = implications

    _write_json(output_dir / "phase_0_results.json", payload)
    _write_json(output_dir / "immutable_score.json", payload["immutable_score"])
    report = _format_report(payload)
    (output_dir / "phase_0_report.md").write_text(report, encoding="utf-8")
    print(f"SAVED {output_dir / 'phase_0_report.md'}", flush=True)


if __name__ == "__main__":
    main()
