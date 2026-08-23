"""Build a common-universe causal opportunity atlas for stock themes.

The atlas is intentionally portfolio-free.  It scans every symbol-day with
prior daily context and completed RTH 5-minute bars, detects pre-registered
reversion and breakout-reference events, and measures three causal entry
mechanisms under the same risk, costs, and exit convention.  This
answers whether IARIC's frequency gap originates in opportunity availability
or in its upstream generator/selector before another optimizer is allowed to
fit the portfolio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CONTRACT_VERSION,
    CURRENT_INPUT_AUTHORITY,
    DISCOVERY_END,
    DISCOVERY_START,
    DOWNSTREAM_EXECUTION_CONTRACT,
    HOLDOUT_START,
    assess_input_authority,
    chronology_contract,
)
from strategies.stock.iaric.bar_policy import completed_rth_5m_bars
from strategies.stock.iaric.core.opportunity import (
    BREAKOUT_REFERENCE_FAMILIES,
    OPPORTUNITY_SCORE_WEIGHTS,
    REVERSION_FAMILIES,
    DailyOpportunityContext,
    detect_completed_bar_opportunities,
    evaluate_standardized_entry_variants,
    prior_session_volume_expectations,
)
from strategies.stock.iaric.core.lanes import SCORE_PROFILES, issuer_key, score_from_components
from strategies.stock.iaric.core.residual import causal_relative_dislocation_atr
from backtests.stock.auto.runners.run_iaric_representative_preflight import (
    build_preflight_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
DEFAULT_OUTPUT = REPO_ROOT / "backtests/output/stock/opportunity_atlas/round_1"
START_DATE = DISCOVERY_START
END_DATE = CALIBRATION_END
IARIC_SUMMARY = REPO_ROOT / "backtests/output/stock/iaric/round_2/run_summary.json"
ALCB_SUMMARY = REPO_ROOT / "backtests/output/stock/alcb/round_2/run_summary.json"

FOLDS: tuple[tuple[str, date, date], ...] = (
    ("discovery", date.fromisoformat(DISCOVERY_START), date.fromisoformat(DISCOVERY_END)),
    ("calibration", date(2024, 12, 1), date.fromisoformat(CALIBRATION_END)),
)
ENTRY_VARIANTS = ("next_bar_open", "one_bar_confirmation", "resting_25pct_retrace")

# Honest coverage boundary: exhaustive over the causal hypotheses expressible
# by the current point-in-time price/volume cache, not over unavailable data.
DEFERRED_HYPOTHESES: dict[str, str] = {
    "earnings_news_catalyst_reversal": "no point-in-time event/news authority in the replay bundle",
    "stable_pairs_cointegration_reversion": "no frozen point-in-time pair membership or model authority",
    "borrow_short_interest_squeeze_reversion": "no point-in-time borrow or short-interest authority",
}

# This is a description of inputs actually consumed by this runner, not data
# that might exist elsewhere. Unknown informational state is deliberately not
# treated as "no news".
INPUT_AUTHORITY: dict[str, Any] = dict(CURRENT_INPUT_AUTHORITY)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--allow-legacy-data", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--bootstrap-simulations", type=int, default=5000)
    parser.add_argument(
        "--authority-preflight-only",
        action="store_true",
        help="write the zero-cost representative-input assessment without loading replay data",
    )
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        default=0,
        help="queue without consuming replay resources until this process exits",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _wait_for_pid(pid: int) -> None:
    if pid <= 0:
        return
    print(f"queued behind PID {pid}", flush=True)
    if os.name == "nt":
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue",
            ],
            check=False,
        )
        return
    import time
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(10.0)


def _code_fingerprint() -> str:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "strategies/stock/iaric/core/opportunity.py",
        REPO_ROOT / "strategies/stock/iaric/core/lanes.py",
        REPO_ROOT / "strategies/stock/iaric/core/residual.py",
        REPO_ROOT / "strategies/stock/iaric/bar_policy.py",
        REPO_ROOT / "backtests/stock/engine/research_replay.py",
        REPO_ROOT / "backtests/stock/data/price_basis.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _daily_context(replay: Any, symbol: str, prior_date: date) -> DailyOpportunityContext | None:
    result = replay._get_arrays_upto(symbol, prior_date, "daily")
    if result is None:
        return None
    arrays, end = result
    if end < 20:
        return None
    opens = arrays["open"]
    highs = arrays["high"]
    lows = arrays["low"]
    closes = arrays["close"]
    volumes = arrays["volume"]
    start = max(1, end - 13)
    true_ranges = [
        max(
            float(highs[index]) - float(lows[index]),
            abs(float(highs[index]) - float(closes[index - 1])),
            abs(float(lows[index]) - float(closes[index - 1])),
        )
        for index in range(start, end + 1)
    ]
    atr = fmean(true_ranges) if true_ranges else 0.0
    if atr <= 0:
        return None
    consecutive_down = 0
    for index in range(end, 0, -1):
        if float(closes[index]) < float(closes[index - 1]):
            consecutive_down += 1
        else:
            break
    prior_bars = completed_rth_5m_bars(
        list(replay.get_5m_bar_objects_for_date(symbol, prior_date))
    )
    expected_5m, expected_profile = prior_session_volume_expectations(
        prior_bars,
        fallback_daily_volume=float(volumes[end]),
    )
    five_day_return = (
        float(closes[end]) / float(closes[end - 5]) - 1.0
        if float(closes[end - 5]) > 0.0 else 0.0
    )
    sma20 = fmean(float(value) for value in closes[end - 19:end + 1])
    prior_sma20 = (
        fmean(float(value) for value in closes[end - 24:end - 4])
        if end >= 24 else sma20
    )
    return DailyOpportunityContext(
        prev_close=float(closes[end]),
        prev_high=float(highs[end]),
        prev_low=float(lows[end]),
        daily_atr=float(atr),
        consecutive_down_days=consecutive_down,
        expected_5m_volume=expected_5m,
        expected_5m_profile=expected_profile,
        five_day_return=five_day_return,
        sma20_slope_atr=(sma20 - prior_sma20) / atr,
    )


def _cross_sectional_returns(
    day_bars: dict[str, list[Any]],
    sector_map: dict[str, str],
) -> tuple[list[float], dict[str, list[float]]]:
    max_bars = max((len(bars) for bars in day_bars.values()), default=0)
    market: list[float] = []
    sectors: dict[str, list[float]] = defaultdict(list)
    sector_names = sorted(set(sector_map.values()))
    for bar_index in range(max_bars):
        market_values: list[float] = []
        sector_values: dict[str, list[float]] = defaultdict(list)
        for symbol, bars in day_bars.items():
            if bar_index >= len(bars) or float(bars[0].open) <= 0:
                continue
            value = float(bars[bar_index].close) / float(bars[0].open) - 1.0
            market_values.append(value)
            sector_values[sector_map.get(symbol, "")].append(value)
        market.append(fmean(market_values) if market_values else 0.0)
        for sector in sector_names:
            values = sector_values.get(sector, [])
            sectors[sector].append(fmean(values) if values else market[-1])
    return market, dict(sectors)


def _profit_factor(values: Iterable[float]) -> float:
    samples = [float(value) for value in values]
    gains = sum(max(value, 0.0) for value in samples)
    losses = -sum(min(value, 0.0) for value in samples)
    return gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)


def _aggregate(records: list[dict[str, Any]], eligible_symbol_days: int) -> dict[str, Any]:
    if not records:
        return {
            "events": 0,
            "events_per_1000_symbol_days": 0.0,
            "avg_stop_target_r": 0.0,
            "stop_target_profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_bar_12_r": 0.0,
            "avg_eod_r": 0.0,
            "r_per_1000_symbol_days_bar_12": 0.0,
            "avg_bars_to_terminal": 0.0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
        }
    stop_values = [float(record["stop_target_r"]) for record in records]
    event_rate = 1000.0 * len(records) / max(eligible_symbol_days, 1)
    avg_bar_12 = fmean(float(record["horizon_r"]["bar_12"]) for record in records)
    return {
        "events": len(records),
        "events_per_1000_symbol_days": event_rate,
        "avg_stop_target_r": fmean(stop_values),
        "stop_target_profit_factor": _profit_factor(stop_values),
        "win_rate": sum(value > 0 for value in stop_values) / len(stop_values),
        "avg_bar_12_r": avg_bar_12,
        "avg_eod_r": fmean(float(record["horizon_r"]["eod"]) for record in records),
        "r_per_1000_symbol_days_bar_12": event_rate * avg_bar_12,
        "avg_bars_to_terminal": fmean(float(record["bars_to_terminal"]) for record in records),
        "avg_mfe_r": fmean(float(record["mfe_r"]) for record in records),
        "avg_mae_r": fmean(float(record["mae_r"]) for record in records),
    }


def _fold_name(value: date) -> str:
    for name, start, end in FOLDS:
        if start <= value <= end:
            return name
    return "outside"


def _score_quintiles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: float(record["score"]))
    if not ordered:
        return []
    result = []
    for bucket in range(5):
        start = math.floor(bucket * len(ordered) / 5)
        end = math.floor((bucket + 1) * len(ordered) / 5)
        samples = ordered[start:end]
        if not samples:
            continue
        result.append({
            "quintile": bucket + 1,
            "count": len(samples),
            "score_min": min(float(item["score"]) for item in samples),
            "score_max": max(float(item["score"]) for item in samples),
            "avg_stop_target_r": fmean(float(item["stop_target_r"]) for item in samples),
            "avg_bar_12_r": fmean(float(item["horizon_r"]["bar_12"]) for item in samples),
        })
    return result


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {name: 0.0 for name in ("p00", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p100")}

    def value_at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        name: value_at(fraction)
        for name, fraction in (
            ("p00", 0.00), ("p05", 0.05), ("p10", 0.10),
            ("p25", 0.25), ("p50", 0.50), ("p75", 0.75),
            ("p90", 0.90), ("p95", 0.95), ("p100", 1.00),
        )
    }


def _score_integrity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Outcome-blind activation and component-integrity audit by family."""

    component_names = tuple(OPPORTUNITY_SCORE_WEIGHTS)
    component_rows: dict[str, dict[str, Any]] = {}
    for name in component_names:
        values = [
            float(record.get("score_components", {}).get(name, float("nan")))
            for record in records
        ]
        finite = [value for value in values if math.isfinite(value)]
        quantiles = _percentiles(finite)
        component_rows[name] = {
            "missing": len(values) - len(finite),
            "quantiles": quantiles,
            "low_saturation_fraction": (
                sum(value <= 0.01 for value in finite) / len(finite) if finite else 1.0
            ),
            "high_saturation_fraction": (
                sum(value >= 0.99 for value in finite) / len(finite) if finite else 1.0
            ),
            "nearly_constant": quantiles["p95"] - quantiles["p05"] < 0.05,
        }

    score_quantiles = _percentiles(record.get("score", 0.0) for record in records)
    profile_score_quantiles = {
        profile: _percentiles(
            score_from_components(record["score_components"], profile)
            for record in records
        )
        for profile in SCORE_PROFILES
    }
    room_values = [float(record.get("reversion_room_atr", 0.0)) for record in records]
    rr_values = [float(record.get("prospective_reward_risk", 0.0)) for record in records]
    room_rr_passes = sum(
        room >= 0.10 and reward_risk >= 0.60
        for room, reward_risk in zip(room_values, rr_values)
    )
    activation_floor = profile_score_quantiles["balanced"]["p90"]
    activation_passes = sum(
        room >= 0.10
        and reward_risk >= 0.60
        and float(record.get("score", 0.0)) >= activation_floor
        for record, room, reward_risk in zip(records, room_values, rr_values)
    )
    residual_nonzero = sum(
        abs(float(record.get("residual_dislocation_atr", 0.0))) > 1e-9
        for record in records
    )
    issuer_days = {
        (str(record.get("date", "")), issuer_key(str(record.get("symbol", ""))))
        for record in records
    }
    required = (
        "dislocation",
        "reclaim",
        "close_quality",
        "relative_volume",
        "residual_dislocation",
        "reversion_room",
    )
    checks = {
        "at_least_ten_events": len(records) >= 10,
        "all_seven_components_populated": all(
            component_rows[name]["missing"] == 0 for name in component_names
        ),
        "required_components_not_nearly_constant": all(
            not component_rows[name]["nearly_constant"] for name in required
        ),
        "required_components_not_excessively_high_saturated": all(
            component_rows[name]["high_saturation_fraction"] <= 0.50 for name in required
        ),
        "residual_input_populated": residual_nonzero >= max(1, math.ceil(0.50 * len(records))),
        "score_has_cross_sectional_spread": score_quantiles["p90"] - score_quantiles["p10"] >= 5.0,
        "room_rr_policy_has_positive_supply": room_rr_passes > 0,
        "combined_activation_passes_and_rejects": 0 < activation_passes < len(records),
    }
    return {
        "events": len(records),
        "unique_issuer_days": len(issuer_days),
        "components": component_rows,
        "score_quantiles": score_quantiles,
        "profile_score_quantiles": profile_score_quantiles,
        "remaining_room_atr_quantiles": _percentiles(room_values),
        "prospective_reward_risk_quantiles": _percentiles(rr_values),
        "room_rr_passes": room_rr_passes,
        "room_rr_rejects": len(records) - room_rr_passes,
        "activation_floor_p90": activation_floor,
        "activation_passes": activation_passes,
        "activation_rejects": len(records) - activation_passes,
        "residual_nonzero_fraction": residual_nonzero / len(records) if records else 0.0,
        "checks": checks,
        "activation_ready": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "floor_calibration": (
            "profile score quantiles are outcome-blind activation references; outcome R is forbidden"
        ),
    }


def _block_bootstrap(records: list[dict[str, Any]], simulations: int) -> dict[str, Any]:
    by_date: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_date[str(record["date"])].append(float(record["horizon_r"]["bar_12"]))
    daily = [fmean(by_date[key]) for key in sorted(by_date)]
    if not daily:
        return {"probability_positive": 0.0, "simulations": simulations, "days": 0}
    rng = random.Random(20260820)
    block = 5
    positive = 0
    for _ in range(simulations):
        sample: list[float] = []
        while len(sample) < len(daily):
            start = rng.randrange(len(daily))
            sample.extend(daily[(start + offset) % len(daily)] for offset in range(block))
        positive += fmean(sample[:len(daily)]) > 0.0
    return {
        "probability_positive": positive / simulations,
        "simulations": simulations,
        "days": len(daily),
        "block_days": block,
    }


def _concentration_robustness(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure whether a family survives removal of its easiest contributors."""

    if not records:
        return {
            "symbols": 0,
            "sectors": 0,
            "positive_sector_fraction": 0.0,
            "avg_bar_12_r_ex_top_3_symbols": 0.0,
            "worst_leave_one_sector_avg_bar_12_r": 0.0,
        }
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_symbol[str(record["symbol"])].append(record)
        by_sector[str(record["sector"] or "UNKNOWN")].append(record)
    ranked_symbols = sorted(
        by_symbol,
        key=lambda symbol: sum(float(item["horizon_r"]["bar_12"]) for item in by_symbol[symbol]),
        reverse=True,
    )
    excluded = set(ranked_symbols[:3])
    ex_top = [record for record in records if str(record["symbol"]) not in excluded]
    sector_means = {
        sector: fmean(float(item["horizon_r"]["bar_12"]) for item in samples)
        for sector, samples in by_sector.items()
    }
    leave_one_sector_means = [
        fmean(
            float(record["horizon_r"]["bar_12"])
            for record in records
            if str(record["sector"] or "UNKNOWN") != sector
        )
        for sector in by_sector
        if len(by_sector[sector]) < len(records)
    ]
    return {
        "symbols": len(by_symbol),
        "sectors": len(by_sector),
        "positive_sector_fraction": (
            sum(value > 0.0 for value in sector_means.values()) / len(sector_means)
        ),
        "avg_bar_12_r_ex_top_3_symbols": (
            fmean(float(record["horizon_r"]["bar_12"]) for record in ex_top)
            if ex_top else 0.0
        ),
        "worst_leave_one_sector_avg_bar_12_r": (
            min(leave_one_sector_means) if leave_one_sector_means else 0.0
        ),
    }


def _promotion_gate(row: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered evidence gate; it diagnoses and never mutates production."""

    folds = row["folds"]
    positive_folds = sum(float(folds[name]["avg_bar_12_r"]) > 0.0 for name, _, _ in FOLDS)
    quintiles = row["score_quintiles"]
    score_directional = (
        len(quintiles) == 5
        and float(quintiles[-1]["avg_bar_12_r"]) > 0.0
        and float(quintiles[-1]["avg_bar_12_r"]) > float(quintiles[0]["avg_bar_12_r"])
    )
    checks = {
        "at_least_200_events": int(row["events"]) >= 200,
        "positive_unconditional_12_bar_r": float(row["avg_bar_12_r"]) > 0.0,
        "bootstrap_probability_at_least_90pct": (
            float(row["bootstrap"]["probability_positive"]) >= 0.90
        ),
        "at_least_two_positive_folds": positive_folds >= 2,
        "calibration_fold_positive": float(folds["calibration"]["avg_bar_12_r"]) > 0.0,
        "score_top_quintile_positive_and_better_than_bottom": score_directional,
        "positive_after_removing_top_three_symbols": (
            float(row["concentration"]["avg_bar_12_r_ex_top_3_symbols"]) > 0.0
        ),
        "at_least_half_of_sectors_positive": (
            float(row["concentration"]["positive_sector_fraction"]) >= 0.50
        ),
    }
    return {
        "research_survivor": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "boundary": (
            "survival authorizes route-specific frozen replay, not production promotion or holdout access"
        ),
    }


def _baseline_reference(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["final_metrics"]
    return {
        key: float(metrics.get(key, 0.0))
        for key in (
            "total_trades",
            "trades_per_month",
            "expected_total_r",
            "profit_factor",
            "net_profit",
            "max_drawdown_pct",
            "avg_hold_hours",
            "avg_r",
            "expectancy",
        )
    }


def _render_report(
    summary: dict[str, Any],
    family_rows: dict[str, dict[str, Any]],
    theme_rows: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Common-Universe Stock Opportunity Atlas",
        "",
        f"Status: {summary['status']}",
        f"Window: {summary['window']['start']} through {summary['window']['end']}; holdout accessed: no.",
        f"Eligible symbol-days: {summary['coverage']['eligible_symbol_days']:,}.",
        "",
        "All events use completed-bar signals, three pre-registered causal entry mechanisms, a common "
        "0.5 daily-ATR risk unit, 10 bps round-trip friction, and the same conservative 1R stop/1R target "
        "evaluator. The next-bar-open route remains the unconditional control in this summary.",
        "Theme rows use the first event per theme/symbol/day. Family rows use one event per "
        "family/symbol/day, so overlapping families must not be added together as independent opportunities.",
        "",
        "## Theme comparison",
        "",
        "| Theme | Events | Events/1k symbol-days | Avg 12-bar R | R/1k symbol-days | Standard PF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for theme in ("reversion", "breakout_reference"):
        row = theme_rows[theme]
        lines.append(
            f"| {theme} | {row['events']} | {row['events_per_1000_symbol_days']:.1f} | "
            f"{row['avg_bar_12_r']:+.3f} | {row['r_per_1000_symbol_days_bar_12']:+.2f} | "
            f"{row['stop_target_profit_factor']:.2f} |"
        )
    lines.extend([
        "",
        "## Event families",
        "",
        "| Family | Events | Rate/1k | Avg 12-bar R | Avg EOD R | Standard PF | Bootstrap P(12-bar R>0) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family, row in sorted(family_rows.items()):
        lines.append(
            f"| {family} | {row['events']} | {row['events_per_1000_symbol_days']:.1f} | "
            f"{row['avg_bar_12_r']:+.3f} | {row['avg_eod_r']:+.3f} | "
            f"{row['stop_target_profit_factor']:.2f} | {row['bootstrap']['probability_positive']:.1%} |"
        )
    lines.extend([
        "",
        "## Outcome-blind score integrity",
        "",
        "| Family | Activation ready | Score p10/p50/p90 | Residual populated | Room/RR pass | Failed checks |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for family, row in sorted(family_rows.items()):
        integrity = row["score_integrity"]
        quantiles = integrity["score_quantiles"]
        failed = ", ".join(integrity["failed_checks"]) if integrity["failed_checks"] else "none"
        lines.append(
            f"| {family} | {'yes' if integrity['activation_ready'] else 'no'} | "
            f"{quantiles['p10']:.1f}/{quantiles['p50']:.1f}/{quantiles['p90']:.1f} | "
            f"{integrity['residual_nonzero_fraction']:.1%} | "
            f"{integrity['room_rr_passes']}/{integrity['events']} | {failed} |"
        )
    lines.extend([
        "",
        "## Pre-registered promotion gate",
        "",
        "| Family | Research survivor | Failed checks |",
        "| --- | ---: | --- |",
    ])
    for family, row in sorted(family_rows.items()):
        gate = row["promotion_gate"]
        failed = ", ".join(gate["failed_checks"]) if gate["failed_checks"] else "none"
        lines.append(f"| {family} | {'yes' if gate['research_survivor'] else 'no'} | {failed} |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These are standardized opportunity measurements, not a promoted portfolio. Breakout-reference "
        "events intentionally approximate the broad OR/PDH motifs; they are not claimed to reproduce ALCB's "
        "full selector and management. A production route is eligible for implementation only after positive "
        "folds, score stability, sector/symbol robustness, and realistic portfolio replay.",
        "The implemented registry covers every pre-registered price/volume reversion hypothesis supported "
        "by the current cache. Event/news, pairs, and positioning hypotheses remain explicitly deferred "
        "rather than silently treated as tested; this atlas never opens the holdout.",
    ])
    return "\n".join(lines) + "\n"


def _authority_preflight_payload(args: argparse.Namespace) -> dict[str, Any]:
    return build_preflight_payload(str(args.start_date), str(args.end_date))


def main() -> None:
    args = _parse_args()
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"end date overlaps sealed holdout beginning {HOLDOUT_START}")
    if not args.allow_legacy_data:
        raise ValueError("the current workspace requires --allow-legacy-data; atlas remains research-only")
    if len(OPPORTUNITY_SCORE_WEIGHTS) != 7:
        raise AssertionError("opportunity score must have exactly seven components")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_for_pid > 0:
        _write_json(output_dir / "queue_status.json", {
            "status": "queued",
            "waiting_for_pid": args.wait_for_pid,
            "queued_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        _wait_for_pid(args.wait_for_pid)
        _write_json(output_dir / "queue_status.json", {
            "status": "running",
            "waiting_for_pid": args.wait_for_pid,
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    if args.authority_preflight_only:
        payload = _authority_preflight_payload(args)
        _write_json(output_dir / "atlas_summary.json", payload)
        _write_json(output_dir / "progress.json", {
            "status": (
                "complete_authority_ready"
                if payload["representative_reversion_baseline_eligible"]
                else "blocked_missing_authoritative_reversion_inputs"
            ),
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(
            "representative input preflight: "
            + (
                "ready"
                if payload["representative_reversion_baseline_eligible"]
                else "blocked; replay atlas skipped"
            ),
            flush=True,
        )
        return

    from backtests.stock.data.replay_cache import load_research_replay_bundle

    bundle = load_research_replay_bundle(DATA_DIR, require_bundle=False)
    replay = bundle.data
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    dates = replay.tradable_dates(start, end)
    symbols = [symbol for symbol, _, _ in replay._universe]
    if args.max_dates > 0:
        dates = dates[:args.max_dates]
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    sector_map = dict(replay._sector_map)

    _write_json(output_dir / "run_spec.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "window": {"start": args.start_date, "end": args.end_date},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "data_authority": "legacy_research_only",
        "data_fingerprint": bundle.cache_source_fingerprint,
        "code_fingerprint": _code_fingerprint(),
        "symbols": len(symbols),
        "dates": len(dates),
        "score_components": OPPORTUNITY_SCORE_WEIGHTS,
        "risk_contract": {
            "entry_variants": list(ENTRY_VARIANTS),
            "control_entry": "next_5m_open_after_completed_signal",
            "risk_atr": 0.50,
            "roundtrip_bps": 10.0,
            "stop_r": 1.0,
            "target_r": 1.0,
            "same_bar_stop_target_order": "stop_first",
        },
        "event_families": sorted(REVERSION_FAMILIES | BREAKOUT_REFERENCE_FAMILIES),
        "hypothesis_coverage": {
            "scope": "data_feasible_price_volume_reversion_hypotheses",
            "implemented_reversion_families": sorted(REVERSION_FAMILIES),
            "deferred_unavailable": DEFERRED_HYPOTHESES,
        },
        "input_authority": INPUT_AUTHORITY,
        "representative_contract_version": CONTRACT_VERSION,
        "chronology": chronology_contract(),
        "representative_reversion_baseline_eligible": False,
        "mechanism_atlas_complete": False,
        "downstream_execution_contract": "price_volume_diagnostic_only",
    })

    records: list[dict[str, Any]] = []
    coverage = {
        "requested_symbol_days": len(symbols) * len(dates),
        "eligible_symbol_days": 0,
        "missing_daily_context": 0,
        "missing_or_short_5m": 0,
        "eligible_symbol_days_by_fold": {name: 0 for name, _, _ in FOLDS},
    }
    events_path = output_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as stream:
        for date_index, trade_date in enumerate(dates, 1):
            prior_date = replay.get_prev_trading_date(trade_date)
            if prior_date is None:
                continue
            day_bars: dict[str, list[Any]] = {}
            contexts: dict[str, DailyOpportunityContext] = {}
            for symbol in symbols:
                context = _daily_context(replay, symbol, prior_date)
                if context is None:
                    coverage["missing_daily_context"] += 1
                    continue
                bars = completed_rth_5m_bars(replay.get_5m_bar_objects_for_date(symbol, trade_date))
                if len(bars) < 2:
                    coverage["missing_or_short_5m"] += 1
                    continue
                contexts[symbol] = context
                day_bars[symbol] = bars
                coverage["eligible_symbol_days"] += 1
                fold = _fold_name(trade_date)
                if fold in coverage["eligible_symbol_days_by_fold"]:
                    coverage["eligible_symbol_days_by_fold"][fold] += 1
            relative_by_symbol = causal_relative_dislocation_atr(
                day_bars,
                sector_map,
                {symbol: context.daily_atr for symbol, context in contexts.items()},
            )
            for symbol, bars in day_bars.items():
                context = contexts[symbol]
                sector = sector_map.get(symbol, "")
                for event in detect_completed_bar_opportunities(
                    bars,
                    context,
                    relative_dislocation_atr=relative_by_symbol.get(symbol),
                ):
                    outcomes = evaluate_standardized_entry_variants(event, bars, context)
                    outcome = outcomes["next_bar_open"]
                    entry_variants = {
                        name: {
                            "entry_price": variant.entry_price,
                            "risk_per_share": variant.risk_per_share,
                            "cost_r": variant.cost_r,
                            "stop_target_r": variant.stop_target_r,
                            "bars_to_terminal": variant.bars_to_terminal,
                            "mfe_r": variant.mfe_r,
                            "mae_r": variant.mae_r,
                            "horizon_r": variant.horizon_r,
                        }
                        for name, variant in outcomes.items()
                    }
                    record = {
                        "date": trade_date.isoformat(),
                        "fold": _fold_name(trade_date),
                        "symbol": symbol,
                        "sector": sector,
                        "family": event.family,
                        "theme": event.theme,
                        "signal_bar_index": event.signal_bar_index,
                        "entry_bar_index": event.entry_bar_index,
                        "signal_time": event.signal_time,
                        "score": event.score,
                        "score_components": event.score_components,
                        "dislocation_atr": event.dislocation_atr,
                        "reclaim_atr": event.reclaim_atr,
                        "close_in_range": event.close_in_range,
                        "relative_volume": event.relative_volume,
                        "residual_dislocation_atr": event.residual_dislocation_atr,
                        "reversion_room_atr": event.reversion_room_atr,
                        "reversion_anchor": event.reversion_anchor,
                        "stop_anchor": event.stop_anchor,
                        "prospective_reward_risk": event.prospective_reward_risk,
                        "episode_start_bar_index": event.episode_start_bar_index,
                        "episode_sequence": event.episode_sequence,
                        "anchor_kind": event.anchor_kind,
                        "entry_price": outcome.entry_price,
                        "risk_per_share": outcome.risk_per_share,
                        "cost_r": outcome.cost_r,
                        "stop_target_r": outcome.stop_target_r,
                        "bars_to_terminal": outcome.bars_to_terminal,
                        "mfe_r": outcome.mfe_r,
                        "mae_r": outcome.mae_r,
                        "horizon_r": outcome.horizon_r,
                        "entry_variants": entry_variants,
                    }
                    records.append(record)
                    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            if date_index % 20 == 0 or date_index == len(dates):
                _write_json(output_dir / "progress.json", {
                    "status": "running",
                    "dates_complete": date_index,
                    "dates_total": len(dates),
                    "events": len(records),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })

    family_rows: dict[str, dict[str, Any]] = {}
    for family in sorted(REVERSION_FAMILIES | BREAKOUT_REFERENCE_FAMILIES):
        samples = [record for record in records if record["family"] == family]
        row = _aggregate(samples, coverage["eligible_symbol_days"])
        row["bootstrap"] = _block_bootstrap(samples, args.bootstrap_simulations)
        row["score_quintiles"] = _score_quintiles(samples)
        row["score_integrity"] = _score_integrity(samples)
        row["folds"] = {
            fold: _aggregate(
                [record for record in samples if record["fold"] == fold],
                int(coverage["eligible_symbol_days_by_fold"][fold]),
            )
            for fold, _, _ in FOLDS
        }
        row["concentration"] = _concentration_robustness(samples)
        row["promotion_gate"] = _promotion_gate(row)
        family_rows[family] = row

    # Causal first-event priority prevents multiple same-theme signals on one
    # symbol-day from exaggerating breadth.
    first_by_theme_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["theme"], record["date"], record["symbol"])
        current = first_by_theme_key.get(key)
        if current is None or int(record["signal_bar_index"]) < int(current["signal_bar_index"]):
            first_by_theme_key[key] = record
    theme_rows = {
        theme: _aggregate(
            [record for key, record in first_by_theme_key.items() if key[0] == theme],
            coverage["eligible_symbol_days"],
        )
        for theme in ("reversion", "breakout_reference")
    }

    authority_assessment = assess_input_authority(INPUT_AUTHORITY)
    summary = {
        "status": "complete_research_only",
        "representative_contract_version": CONTRACT_VERSION,
        "code_fingerprint": _code_fingerprint(),
        "data_fingerprint": bundle.cache_source_fingerprint,
        "window": {"start": args.start_date, "end": args.end_date},
        "holdout_accessed": False,
        "coverage": coverage,
        "hypothesis_coverage": {
            "scope": "bounded_exhaustive_within_current_point_in_time_price_volume_authority",
            "implemented_reversion_families": sorted(REVERSION_FAMILIES),
            "implemented_entry_variants": list(ENTRY_VARIANTS),
            "deferred_unavailable": DEFERRED_HYPOTHESES,
        },
        "input_authority": INPUT_AUTHORITY,
        "sleeve_readiness": authority_assessment["sleeve_readiness"],
        "representative_reversion_baseline_eligible": authority_assessment[
            "representative_reversion_baseline_eligible"
        ],
        "representative_reversion_baseline_blockers": authority_assessment["blockers"],
        "mechanism_atlas_complete": False,
        "mechanism_candidate_registry_complete": False,
        "typed_management_precedes_composition": False,
        "economic_input_parity": {
            "passed": False,
            "reason": "price/volume diagnostic atlas has no authoritative event/factor/liquidity parity",
        },
        "required_downstream_execution_contract": DOWNSTREAM_EXECUTION_CONTRACT,
        "downstream_execution_contract": "price_volume_diagnostic_only",
        "events": len(records),
        "family_results": family_rows,
        "theme_first_event_results": theme_rows,
        "current_portfolio_references": {
            "iaric_round2": _baseline_reference(IARIC_SUMMARY),
            "alcb_round2": _baseline_reference(ALCB_SUMMARY),
        },
    }
    _write_json(output_dir / "atlas_summary.json", summary)
    (output_dir / "report.md").write_text(
        _render_report(summary, family_rows, theme_rows),
        encoding="utf-8",
    )
    _write_json(output_dir / "progress.json", {
        "status": "complete",
        "dates_complete": len(dates),
        "dates_total": len(dates),
        "events": len(records),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    if args.wait_for_pid > 0:
        _write_json(output_dir / "queue_status.json", {
            "status": "complete",
            "waiting_for_pid": args.wait_for_pid,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    print(
        f"opportunity atlas complete: {coverage['eligible_symbol_days']} eligible symbol-days, "
        f"{len(records)} events; holdout accessed=no",
        flush=True,
    )


if __name__ == "__main__":
    main()
