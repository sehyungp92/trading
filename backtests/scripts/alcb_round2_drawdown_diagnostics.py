"""Deep drawdown and signal-funnel diagnostics for the ALCB Round-2 candidate.

The repaired legacy cache and previously consumed OOS window make every output
research-only.  This script does not mutate the production strategy config.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from backtests.scripts.alcb_round2_oos_robustness import (  # noqa: E402
    BASE_CONFIG_PATH,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    REPO_ROOT,
    _run_context,
    _load_json,
    _trade_to_dict,
    _write_json,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests"
    / "output"
    / "stock"
    / "alcb"
    / "round_2"
    / "oos_robustness_20260722"
    / "drawdown_diagnostics_20260723"
)
INITIAL_EQUITY = 10_000.0
ET = ZoneInfo("America/New_York")
BALANCED_PATCH: dict[str, Any] = {
    "param_overrides.rvol_threshold": 1.70,
    "param_overrides.opening_range_bars": 9,
    "param_overrides.adaptive_trail_late_distance_r": 0.04,
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _metadata(row: dict[str, Any], key: str, default: Any = None) -> Any:
    return (row.get("metadata") or {}).get(key, default)


def _iso_date(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def _quantile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    index = (len(clean) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return clean[lower]
    weight = index - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _group_rows(
    rows: list[dict[str, Any]],
    label: str,
    key_fn: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(row)
    output: list[dict[str, Any]] = []
    for key, group in buckets.items():
        rs = [_number(row.get("r_multiple")) for row in group]
        pnls = [_number(row.get("pnl_net")) for row in group]
        mfes = [_number(_metadata(row, "mfe_r")) for row in group]
        maes = [_number(_metadata(row, "mae_r")) for row in group]
        total_mfe = sum(max(0.0, value) for value in mfes)
        output.append(
            {
                label: key,
                "trades": len(group),
                "wins": sum(value > 0.0 for value in pnls),
                "win_rate": sum(value > 0.0 for value in pnls) / len(group),
                "avg_r": mean(rs),
                "median_r": median(rs),
                "total_r": sum(rs),
                "pnl_net": sum(pnls),
                "avg_mfe_r": mean(mfes),
                "avg_mae_r": mean(maes),
                "mfe_capture_ratio": sum(rs) / total_mfe if total_mfe > 0 else 0.0,
            }
        )
    return sorted(output, key=lambda item: (item["pnl_net"], item[label]))


def _daily_equity(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    pnl_by_day: dict[date, float] = defaultdict(float)
    r_by_day: dict[date, float] = defaultdict(float)
    count_by_day: dict[date, int] = defaultdict(int)
    for row in rows:
        exit_day = _iso_date(row["exit_time"])
        pnl_by_day[exit_day] += _number(row.get("pnl_net"))
        r_by_day[exit_day] += _number(row.get("r_multiple"))
        count_by_day[exit_day] += 1

    equity = INITIAL_EQUITY
    peak = equity
    peak_day = date.fromisoformat(start)
    output: list[dict[str, Any]] = []
    for day in _date_range(date.fromisoformat(start), date.fromisoformat(end)):
        equity += pnl_by_day.get(day, 0.0)
        if equity >= peak:
            peak = equity
            peak_day = day
        drawdown_dollar = peak - equity
        output.append(
            {
                "date": day.isoformat(),
                "equity": equity,
                "daily_pnl": pnl_by_day.get(day, 0.0),
                "daily_r": r_by_day.get(day, 0.0),
                "trades_closed": count_by_day.get(day, 0),
                "peak_equity": peak,
                "peak_date": peak_day.isoformat(),
                "drawdown_dollar": drawdown_dollar,
                "drawdown_pct": drawdown_dollar / peak if peak > 0 else 0.0,
            }
        )
    return output


def _drawdown_episodes(
    equity_rows: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for row in equity_rows:
        dd = _number(row["drawdown_pct"])
        if dd > 0 and active is None:
            active = {
                "peak_date": row["peak_date"],
                "peak_equity": row["peak_equity"],
                "start_date": row["date"],
                "trough_date": row["date"],
                "trough_equity": row["equity"],
                "max_drawdown_pct": dd,
                "max_drawdown_dollar": row["drawdown_dollar"],
                "recovery_date": None,
            }
        elif dd > 0 and active is not None and dd > active["max_drawdown_pct"]:
            active["trough_date"] = row["date"]
            active["trough_equity"] = row["equity"]
            active["max_drawdown_pct"] = dd
            active["max_drawdown_dollar"] = row["drawdown_dollar"]
        elif dd == 0 and active is not None:
            active["recovery_date"] = row["date"]
            episodes.append(active)
            active = None
    if active is not None:
        episodes.append(active)

    for episode in episodes:
        peak_day = date.fromisoformat(episode["peak_date"])
        trough_day = date.fromisoformat(episode["trough_date"])
        recovery_day = (
            date.fromisoformat(episode["recovery_date"])
            if episode.get("recovery_date")
            else date.fromisoformat(equity_rows[-1]["date"])
        )
        descent = [
            row
            for row in trades
            if peak_day < _iso_date(row["exit_time"]) <= trough_day
        ]
        full = [
            row
            for row in trades
            if peak_day < _iso_date(row["exit_time"]) <= recovery_day
        ]
        episode["calendar_days_to_trough"] = (trough_day - peak_day).days
        episode["calendar_days_to_recovery"] = (
            (recovery_day - peak_day).days if episode.get("recovery_date") else None
        )
        episode["descent_trade_count"] = len(descent)
        episode["descent_total_r"] = sum(_number(row.get("r_multiple")) for row in descent)
        episode["descent_pnl_net"] = sum(_number(row.get("pnl_net")) for row in descent)
        episode["descent_loss_count"] = sum(_number(row.get("pnl_net")) < 0 for row in descent)
        episode["descent_worst_trades"] = sorted(
            descent, key=lambda row: _number(row.get("pnl_net"))
        )[:15]
        episode["descent_by_exit_reason"] = _group_rows(
            descent, "exit_reason", lambda row: row.get("exit_reason")
        )
        episode["descent_by_entry_type"] = _group_rows(
            descent, "entry_type", lambda row: row.get("entry_type")
        )
        episode["descent_by_score"] = _group_rows(
            descent, "score", lambda row: _metadata(row, "momentum_score")
        )
        episode["full_episode_trade_count"] = len(full)
    return sorted(episodes, key=lambda item: item["max_drawdown_pct"], reverse=True)


def _loss_concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = sorted(
        (row for row in rows if _number(row.get("pnl_net")) < 0.0),
        key=lambda row: _number(row.get("pnl_net")),
    )
    gross_loss = -sum(_number(row.get("pnl_net")) for row in losses)

    def share(count: int) -> float:
        return (
            -sum(_number(row.get("pnl_net")) for row in losses[:count]) / gross_loss
            if gross_loss > 0
            else 0.0
        )

    return {
        "loss_count": len(losses),
        "gross_loss_dollar": gross_loss,
        "worst_1_share": share(1),
        "worst_3_share": share(3),
        "worst_5_share": share(5),
        "worst_10_share": share(10),
        "worst_20_share": share(20),
        "worst_trades": losses[:25],
        "loss_r_quantiles": {
            "p00": _quantile([_number(row.get("r_multiple")) for row in losses], 0.00),
            "p10": _quantile([_number(row.get("r_multiple")) for row in losses], 0.10),
            "p25": _quantile([_number(row.get("r_multiple")) for row in losses], 0.25),
            "p50": _quantile([_number(row.get("r_multiple")) for row in losses], 0.50),
            "p75": _quantile([_number(row.get("r_multiple")) for row in losses], 0.75),
            "p90": _quantile([_number(row.get("r_multiple")) for row in losses], 0.90),
        },
    }


def _losing_streaks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row.get("exit_time")))
    streaks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in ordered:
        if _number(row.get("pnl_net")) < 0:
            current.append(row)
        elif current:
            streaks.append(current)
            current = []
    if current:
        streaks.append(current)
    output = [
        {
            "losses": len(streak),
            "start": streak[0]["exit_time"],
            "end": streak[-1]["exit_time"],
            "total_r": sum(_number(row.get("r_multiple")) for row in streak),
            "pnl_net": sum(_number(row.get("pnl_net")) for row in streak),
            "trades": streak,
        }
        for streak in streaks
    ]
    return sorted(output, key=lambda item: (item["losses"], -item["pnl_net"]), reverse=True)


def _trade_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rvol_bucket(row: dict[str, Any]) -> str:
        value = _number(_metadata(row, "entry_signal_rvol"))
        if value < 1.8:
            return "1.70-1.79"
        if value < 2.0:
            return "1.80-1.99"
        if value < 2.5:
            return "2.00-2.49"
        if value < 3.0:
            return "2.50-2.99"
        if value < 4.0:
            return "3.00-3.99"
        return "4.00+"

    def hold_bucket(row: dict[str, Any]) -> str:
        bars = int(row.get("hold_bars", 0))
        if bars <= 4:
            return "00-04"
        if bars <= 9:
            return "05-09"
        if bars <= 15:
            return "10-15"
        if bars <= 24:
            return "16-24"
        if bars <= 48:
            return "25-48"
        return "49+"

    def mfe_bucket(row: dict[str, Any]) -> str:
        value = _number(_metadata(row, "mfe_r"))
        if value < 0.10:
            return "<0.10"
        if value < 0.20:
            return "0.10-0.19"
        if value < 0.40:
            return "0.20-0.39"
        if value < 0.75:
            return "0.40-0.74"
        if value < 1.50:
            return "0.75-1.49"
        return "1.50+"

    def mae_bucket(row: dict[str, Any]) -> str:
        value = _number(_metadata(row, "mae_r"))
        if value < 0.20:
            return "<0.20"
        if value < 0.40:
            return "0.20-0.39"
        if value < 0.70:
            return "0.40-0.69"
        if value < 1.00:
            return "0.70-0.99"
        return "1.00+"

    def signal_time_bucket(row: dict[str, Any]) -> str:
        value = str(_metadata(row, "signal_time", row.get("signal_time", "")))
        try:
            stamp = datetime.fromisoformat(value).astimezone(ET)
            hour = stamp.hour
            minute = stamp.minute
        except ValueError:
            return "unknown"
        total = hour * 60 + minute
        if total < 10 * 60 + 30:
            return "10:00-10:29"
        if total < 11 * 60:
            return "10:30-10:59"
        if total < 11 * 60 + 30:
            return "11:00-11:29"
        if total < 12 * 60:
            return "11:30-11:59"
        return "12:00+"

    def orb_quality_bucket(row: dict[str, Any]) -> str:
        value = _number(_metadata(row, "orb_quality_score"))
        if value < 55.0:
            return "<55"
        if value < 60.0:
            return "55-59"
        if value < 65.0:
            return "60-64"
        if value < 70.0:
            return "65-69"
        if value < 75.0:
            return "70-74"
        if value < 80.0:
            return "75-79"
        return "80+"

    def avwap_premium_bucket(row: dict[str, Any]) -> str:
        avwap = _number(_metadata(row, "avwap_at_entry"))
        premium = (_number(row.get("entry_price")) / avwap - 1.0) if avwap > 0 else 0.0
        if premium < 0.0025:
            return "<0.25%"
        if premium < 0.0050:
            return "0.25-0.49%"
        if premium < 0.0075:
            return "0.50-0.74%"
        if premium < 0.0100:
            return "0.75-0.99%"
        if premium < 0.0150:
            return "1.00-1.49%"
        return "1.50%+"

    def breakout_extension_bucket(row: dict[str, Any]) -> str:
        risk = _number(row.get("risk_per_share"))
        level = _number(_metadata(row, "breakout_level"))
        extension_r = (_number(row.get("entry_price")) - level) / risk if risk > 0 else 0.0
        if extension_r < 0.10:
            return "<0.10R"
        if extension_r < 0.20:
            return "0.10-0.19R"
        if extension_r < 0.35:
            return "0.20-0.34R"
        if extension_r < 0.50:
            return "0.35-0.49R"
        return "0.50R+"

    def score_rvol(row: dict[str, Any]) -> str:
        return f"score{_metadata(row, 'momentum_score')}|{rvol_bucket(row)}"

    def entry_time(row: dict[str, Any]) -> str:
        return f"{row.get('entry_type')}|{signal_time_bucket(row)}"

    def regime_score(row: dict[str, Any]) -> str:
        return f"{row.get('regime_tier')}|score{_metadata(row, 'momentum_score')}"

    return {
        "entry_type": _group_rows(rows, "entry_type", lambda row: row.get("entry_type")),
        "exit_reason": _group_rows(rows, "exit_reason", lambda row: row.get("exit_reason")),
        "symbol": _group_rows(rows, "symbol", lambda row: row.get("symbol")),
        "sector": _group_rows(rows, "sector", lambda row: row.get("sector")),
        "regime": _group_rows(rows, "regime", lambda row: row.get("regime_tier")),
        "momentum_score": _group_rows(
            rows, "momentum_score", lambda row: _metadata(row, "momentum_score")
        ),
        "signal_bar_index": _group_rows(
            rows, "signal_bar_index", lambda row: _metadata(row, "signal_bar_index")
        ),
        "signal_time_bucket": _group_rows(rows, "signal_time_bucket", signal_time_bucket),
        "rvol_bucket": _group_rows(rows, "rvol_bucket", rvol_bucket),
        "orb_quality_bucket": _group_rows(rows, "orb_quality_bucket", orb_quality_bucket),
        "avwap_premium_bucket": _group_rows(
            rows, "avwap_premium_bucket", avwap_premium_bucket
        ),
        "breakout_extension_bucket": _group_rows(
            rows, "breakout_extension_bucket", breakout_extension_bucket
        ),
        "score_x_rvol": _group_rows(rows, "score_x_rvol", score_rvol),
        "entry_type_x_time": _group_rows(rows, "entry_type_x_time", entry_time),
        "regime_x_score": _group_rows(rows, "regime_x_score", regime_score),
        "hold_bucket": _group_rows(rows, "hold_bucket", hold_bucket),
        "mfe_bucket": _group_rows(rows, "mfe_bucket", mfe_bucket),
        "mae_bucket": _group_rows(rows, "mae_bucket", mae_bucket),
        "day_of_week": _group_rows(
            rows,
            "day_of_week",
            lambda row: _iso_date(row.get("entry_time")).strftime("%A"),
        ),
        "month": _group_rows(rows, "month", lambda row: str(row.get("exit_time"))[:7]),
        "score_factors": {
            factor: _group_rows(
                rows,
                factor,
                lambda row, key=factor: bool(
                    (_metadata(row, "score_detail", {}) or {}).get(key)
                ),
            )
            for factor in (
                "above_pdh",
                "above_or",
                "bar_vol_surge",
                "strong_cpr",
                "above_avwap",
                "adx_trending",
                "gap_up",
            )
        },
    }


def _approximate_daily_stop(
    rows: list[dict[str, Any]],
    start: str,
    end: str,
    thresholds: tuple[float, ...] = (2.0, 2.35, 3.0, 3.5),
) -> list[dict[str, Any]]:
    """Approximate a realized-R daily entry lockout without replaying the portfolio.

    The approximation keeps the original fills, exits, and PnL of accepted trades.
    It only removes entries occurring after already-kept trades have realized losses
    beyond the threshold on the same ET session.  It therefore cannot model freed
    capital, simultaneous-position interactions, or altered exits and is deliberately
    reported as a directional parity diagnostic.
    """

    ordered = sorted(rows, key=lambda row: str(row.get("entry_time")))
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        kept: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        by_session: dict[date, list[dict[str, Any]]] = defaultdict(list)
        for row in ordered:
            entry_stamp = datetime.fromisoformat(str(row["entry_time"])).astimezone(ET)
            session = entry_stamp.date()
            realized_r = sum(
                _number(prior.get("r_multiple"))
                for prior in by_session[session]
                if datetime.fromisoformat(str(prior["exit_time"])).astimezone(ET) <= entry_stamp
            )
            if realized_r <= -threshold:
                skipped.append(row)
                continue
            kept.append(row)
            by_session[session].append(row)

        equity = _daily_equity(kept, start, end)
        max_dd = max((_number(row["drawdown_pct"]) for row in equity), default=0.0)
        output.append(
            {
                "threshold_r": threshold,
                "kept_trades": len(kept),
                "skipped_trades": len(skipped),
                "kept_total_r": sum(_number(row.get("r_multiple")) for row in kept),
                "kept_net_pnl": sum(_number(row.get("pnl_net")) for row in kept),
                "skipped_total_r": sum(_number(row.get("r_multiple")) for row in skipped),
                "skipped_net_pnl": sum(_number(row.get("pnl_net")) for row in skipped),
                "approx_max_drawdown_pct": max_dd,
            }
        )
    return output


def _shadow_payload(shadow: Any) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    for setup in shadow.completed if shadow is not None else []:
        payload = asdict(setup) if is_dataclass(setup) else dict(vars(setup))
        payload["direction"] = str(payload.get("direction"))
        payload["trade_date"] = str(payload.get("trade_date"))
        completed.append(payload)

    by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_gate[str(row.get("rejection_gate"))].append(row)
    gate_summary = []
    for gate, rows in by_gate.items():
        rs = [_number(row.get("simulated_r")) for row in rows]
        gate_summary.append(
            {
                "gate": gate,
                "setups": len(rows),
                "positive_rate": sum(value > 0 for value in rs) / len(rows),
                "target_hit_rate": sum(row.get("simulated_exit") == "TARGET_HIT" for row in rows)
                / len(rows),
                "stop_hit_rate": sum(row.get("simulated_exit") == "STOP_HIT" for row in rows)
                / len(rows),
                "avg_r": mean(rs),
                "median_r": median(rs),
                "total_r": sum(rs),
                "avg_mfe_r": mean(_number(row.get("mfe_r")) for row in rows),
                "avg_mae_r": mean(_number(row.get("mae_r")) for row in rows),
            }
        )
    return {
        "funnel": shadow.funnel if shadow is not None else {},
        "gate_summary": sorted(gate_summary, key=lambda item: item["total_r"], reverse=True),
        "completed_rejections": completed,
    }


def _selection_payload(daily_selections: dict[Any, Any]) -> dict[str, Any]:
    rows = []
    for day, artifact in sorted(daily_selections.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "date": str(day),
                "regime_tier": artifact.regime.tier,
                "items": len(artifact.items),
                "tradable": len(artifact.tradable),
                "overflow": len(artifact.overflow),
                "long_candidates": len(artifact.long_candidates),
                "short_candidates": len(artifact.short_candidates),
            }
        )
    return {
        "daily": rows,
        "regime_days": {
            tier: sum(row["regime_tier"] == tier for row in rows)
            for tier in sorted({row["regime_tier"] for row in rows})
        },
        "avg_items": mean(row["items"] for row in rows) if rows else 0.0,
        "avg_tradable": mean(row["tradable"] for row in rows) if rows else 0.0,
        "avg_overflow": mean(row["overflow"] for row in rows) if rows else 0.0,
    }


def _render_report(
    metrics: dict[str, Any],
    losses: dict[str, Any],
    episodes: list[dict[str, Any]],
    attribution: dict[str, Any],
    shadow: dict[str, Any],
    daily_stop: list[dict[str, Any]],
) -> str:
    max_episode = episodes[0] if episodes else {}
    lines = [
        "# ALCB Round 2 drawdown diagnostics",
        "",
        "Diagnostic-only repaired-cache analysis of RVOL 1.70 / OR 9 / late trail 0.04.",
        "",
        "## Headline",
        "",
        f"- Trades: {int(metrics.get('total_trades', 0))}; expected total R: {metrics.get('expected_total_r', 0):.2f}; "
        f"PF: {metrics.get('profit_factor', 0):.2f}; max DD: {100*metrics.get('max_drawdown_pct', 0):.2f}%.",
        f"- Losses: {losses.get('loss_count', 0)}; worst 1/3/5/10 shares of gross loss: "
        f"{100*losses.get('worst_1_share', 0):.1f}% / {100*losses.get('worst_3_share', 0):.1f}% / "
        f"{100*losses.get('worst_5_share', 0):.1f}% / {100*losses.get('worst_10_share', 0):.1f}%.",
    ]
    if max_episode:
        lines.extend(
            [
                f"- Max realized-equity drawdown: {100*max_episode['max_drawdown_pct']:.2f}% "
                f"from {max_episode['peak_date']} to {max_episode['trough_date']}; "
                f"{max_episode['descent_trade_count']} trades closed during the descent, "
                f"including {max_episode['descent_loss_count']} losses.",
                "",
            ]
        )

    lines.extend(["## Worst drawdown episodes", "", "| Peak | Trough | Recovery | DD | Descent trades | Losses | Descent R |", "|---|---|---|---:|---:|---:|---:|"])
    for row in episodes[:10]:
        lines.append(
            f"| {row['peak_date']} | {row['trough_date']} | {row.get('recovery_date') or 'unrecovered'} | "
            f"{100*row['max_drawdown_pct']:.2f}% | {row['descent_trade_count']} | "
            f"{row['descent_loss_count']} | {row['descent_total_r']:+.2f} |"
        )

    lines.extend(["", "## Lowest-value accepted cohorts", ""])
    for name in (
        "entry_type",
        "exit_reason",
        "momentum_score",
        "rvol_bucket",
        "orb_quality_bucket",
        "avwap_premium_bucket",
        "breakout_extension_bucket",
        "hold_bucket",
        "signal_time_bucket",
    ):
        rows = attribution.get(name, [])
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Cohort | Trades | Win rate | Avg R | Total R | Net PnL | Avg MFE | Avg MAE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        key = name
        for row in rows:
            lines.append(
                f"| {row[key]} | {row['trades']} | {100*row['win_rate']:.1f}% | "
                f"{row['avg_r']:+.3f} | {row['total_r']:+.2f} | ${row['pnl_net']:,.0f} | "
                f"{row['avg_mfe_r']:.2f} | {row['avg_mae_r']:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Approximate realized-R daily-stop parity",
            "",
            "| Stop | Kept | Skipped | Kept R | Skipped R | Kept net PnL | Approx DD |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in daily_stop:
        lines.append(
            f"| {row['threshold_r']:.2f}R | {row['kept_trades']} | {row['skipped_trades']} | "
            f"{row['kept_total_r']:+.2f} | {row['skipped_total_r']:+.2f} | "
            f"${row['kept_net_pnl']:,.0f} | {100*row['approx_max_drawdown_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "This is a directional counterfactual using original fills and exits, not a portfolio replay. "
            "It cannot model freed capital, simultaneous positions, or changed exits.",
            "",
        ]
    )

    lines.extend(
        [
            "## Rejected-signal shadow outcomes",
            "",
            "| Gate | Setups | Positive | Target hit | Stop hit | Avg R | Total R | Avg MFE | Avg MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in shadow.get("gate_summary", []):
        lines.append(
            f"| {row['gate']} | {row['setups']} | {100*row['positive_rate']:.1f}% | "
            f"{100*row['target_hit_rate']:.1f}% | {100*row['stop_hit_rate']:.1f}% | "
            f"{row['avg_r']:+.3f} | {row['total_r']:+.2f} | "
            f"{row['avg_mfe_r']:.2f} | {row['avg_mae_r']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Shadow outcomes use the tracker's simplified stop/1.5R target path and are directional diagnostics, not strategy PnL.",
            "",
            "No live configuration was modified.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=IS_START)
    parser.add_argument("--end", default=IS_END)
    parser.add_argument("--allow-legacy-data", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Regenerate attribution from persisted trades/shadows without replaying the engine.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.allow_legacy_data:
        raise SystemExit(
            "No authoritative frozen direct-RTH bundle is available. "
            "Pass --allow-legacy-data for diagnostic-only repaired-cache analysis."
        )
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    balanced_mutations = {**_load_json(BASE_CONFIG_PATH), **BALANCED_PATCH}
    if args.resume_existing and all(
        (output / name).exists()
        for name in ("metrics.json", "trades.json", "shadow_rejections.json", "selection_funnel.json")
    ):
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        trades = json.loads((output / "trades.json").read_text(encoding="utf-8"))
        shadow = json.loads((output / "shadow_rejections.json").read_text(encoding="utf-8"))
        selection = json.loads((output / "selection_funnel.json").read_text(encoding="utf-8"))
        context = None
    else:
        context = _run_context(balanced_mutations, args.start, args.end)
        trades = [_trade_to_dict(trade) for trade in context["trades"]]
        metrics = dict(context["metrics"])
        shadow = _shadow_payload(context.get("shadow_tracker"))
        selection = _selection_payload(context.get("daily_selections", {}))
    equity = _daily_equity(trades, args.start, args.end)
    episodes = _drawdown_episodes(equity, trades)
    losses = _loss_concentration(trades)
    attribution = _trade_attribution(trades)
    max_episode_attribution: dict[str, Any] = {}
    if episodes:
        peak_day = date.fromisoformat(episodes[0]["peak_date"])
        trough_day = date.fromisoformat(episodes[0]["trough_date"])
        max_episode_attribution = _trade_attribution(
            [
                row
                for row in trades
                if peak_day < _iso_date(row["exit_time"]) <= trough_day
            ]
        )
    daily_stop = _approximate_daily_stop(trades, args.start, args.end)
    _write_json(output / "run_spec.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"start": args.start, "end": args.end},
        "base_config": str(BASE_CONFIG_PATH),
        "candidate_patch": BALANCED_PATCH,
        "effective_mutation_count": len(balanced_mutations),
        "data_authority": "diagnostic-only repaired legacy filename cache",
        "promotion_authorized": False,
    })
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "trades.json", trades)
    _write_json(output / "daily_equity.json", equity)
    _write_json(output / "drawdown_episodes.json", episodes)
    _write_json(output / "loss_concentration.json", losses)
    _write_json(output / "losing_streaks.json", _losing_streaks(trades))
    _write_json(output / "trade_attribution.json", attribution)
    _write_json(output / "max_drawdown_attribution.json", max_episode_attribution)
    _write_json(output / "approximate_daily_stop.json", daily_stop)
    _write_json(output / "shadow_rejections.json", shadow)
    _write_json(output / "selection_funnel.json", selection)
    (output / "report.md").write_text(
        _render_report(metrics, losses, episodes, attribution, shadow, daily_stop),
        encoding="utf-8",
    )
    _write_json(output / "completion.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "trade_count": len(trades),
        "shadow_rejection_count": len(shadow["completed_rejections"]),
        "drawdown_episode_count": len(episodes),
        "promotion_authorized": False,
    })
    print(f"complete: {output}", flush=True)
    print(
        f"trades={len(trades)} max_dd={metrics.get('max_drawdown_pct', 0):.4%} "
        f"shadow_rejections={len(shadow['completed_rejections'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
