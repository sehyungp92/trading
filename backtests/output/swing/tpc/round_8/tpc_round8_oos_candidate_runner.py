"""OOS check for selected TPC round-8 candidates.

Evaluates the round-8 optimized baseline and the strongest in-sample smoke
leads on both the train window and the post-train holdout/OOS window.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.swing.auto.tpc.plugin import _extract_tpc_metrics
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
from backtests.swing.engine.tpc_engine import run_tpc_independent

ROUND_DIR = ROOT / "backtests" / "output" / "swing" / "tpc" / "round_8"
CONFIG_PATH = ROUND_DIR / "optimized_config.json"
DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
TRAIN_END = "2025-11-01"
INITIAL_EQUITY = 100_000.0

_TRAIN_DATA: dict[str, dict[str, Any]] | None = None
_FULL_DATA: dict[str, dict[str, Any]] | None = None
_DATA_DIR: Path | None = None
_OOS_WARMUP_15M = 1
_TRAIN_INDICATOR_CACHE: dict[Any, Any] = {}
_OOS_INDICATOR_CACHE: dict[Any, Any] = {}


@dataclass(frozen=True)
class Candidate:
    name: str
    mutations: dict[str, Any]
    thesis: str
    family: str = "selected"


def candidate_specs(base: dict[str, Any]) -> list[Candidate]:
    def c(name: str, extra: dict[str, Any], thesis: str, family: str = "selected") -> Candidate:
        muts = dict(base)
        muts.update(extra)
        return Candidate(name, muts, thesis, family)

    return [
        Candidate("baseline_round8", dict(base), "Latest round-8 optimized baseline.", "baseline"),
        c("pb30_duration_5_18", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18}, "Best clean signal-duration lead.", "signal"),
        c("qqq_t1_100", {"QQQ.t1_r": 1.00}, "Best clean QQQ management lead.", "management"),
        c("disable_mfe_giveback", {"all.mfe_giveback_trigger_r": 0.0}, "Remove MFE giveback, which improved train without DD increase.", "management"),
        c("trail_after_t1_6", {"all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "Slightly tighter post-T1 structure/VWAP trail.", "management"),
        c("max_extension_qqq_175", {"QQQ.max_extension_atr_mult": 1.75}, "Small clean QQQ extension filter.", "discrimination"),
        c("type_a_value_hits2", {"all.type_a_value_hits_min": 2}, "Higher-quality value interaction, lower frequency.", "discrimination"),
        c("t2_300", {"all.t2_r": 3.00}, "Larger T2 train-only lead.", "exit"),
        c("t1_stop_025", {"all.t1_stop_r": 0.25}, "Looser post-T1 stop lock.", "exit"),
        c("qqq_partial_060", {"QQQ.t1_partial_pct": 0.60}, "Let QQQ runners retain more size.", "exit"),
        c("pb30_value_touch_limit", {"all.pb30_ema20_value_touch_enabled": True, "all.pb30_ema20_value_touch_entry_order_model": "ema20_30m_value_touch_limit"}, "PB30 EMA20 value-touch limit route.", "entry"),
        c("addon_cautious", {"all.addon_enabled": True, "all.addon_trigger_r": 1.50, "all.addon_size_mult": 0.15, "all.addon_min_score": 17, "all.addon_requires_t1": True, "all.addon_require_vwap_hold": True, "all.addon_require_structure_hold": True, "all.addon_max_total_risk_pct": 0.03}, "Small continuation add-on.", "management"),
        c("risk_temper_020", {"all.max_risk_pct": 0.020, "all.max_position_notional_pct": 5.0, "all.risk_a_pct": 0.013, "all.risk_a_plus_pct": 0.020, "all.risk_b_pct": 0.0085}, "Lower risk stack for DD control.", "risk"),
        c("pb30_5_18_plus_qqq_t1_100", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00}, "Top two clean in-sample leads.", "combo"),
        c("pb30_5_18_plus_no_mfe", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "all.mfe_giveback_trigger_r": 0.0}, "PB30 duration plus no MFE giveback.", "combo"),
        c("qqq_t1_100_plus_no_mfe", {"QQQ.t1_r": 1.00, "all.mfe_giveback_trigger_r": 0.0}, "QQQ target plus no MFE giveback.", "combo"),
        c("qqq_t1_100_plus_trail6", {"QQQ.t1_r": 1.00, "all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "QQQ target plus tighter post-T1 trail.", "combo"),
        c("pb30_5_18_plus_trail6", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "PB30 duration plus tighter post-T1 trail.", "combo"),
        c("top3_pb30_qqq_no_mfe", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.mfe_giveback_trigger_r": 0.0}, "Best train stack: PB30 5-18, QQQ T1 1.0, no MFE giveback.", "combo"),
        c("top3_pb30_qqq_trail6", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "Strong train stack using tighter trail instead of no MFE giveback.", "combo"),
        c("top4_pb30_qqq_no_mfe_ext", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.mfe_giveback_trigger_r": 0.0, "QQQ.max_extension_atr_mult": 1.75}, "Best train stack plus QQQ extension control.", "combo"),
        c("top3_plus_risk_temper", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.mfe_giveback_trigger_r": 0.0, "all.max_risk_pct": 0.020, "all.max_position_notional_pct": 5.0, "all.risk_a_pct": 0.013, "all.risk_a_plus_pct": 0.020, "all.risk_b_pct": 0.0085}, "Best train stack with lower risk for DD.", "combo_risk"),
        c("top3_trail6_plus_risk_temper", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True, "all.max_risk_pct": 0.020, "all.max_position_notional_pct": 5.0, "all.risk_a_pct": 0.013, "all.risk_a_plus_pct": 0.020, "all.risk_b_pct": 0.0085}, "Trail-based train stack with lower risk for DD.", "combo_risk"),
        c("no_trail_profit_floor", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False, "all.profit_floor_ladder": ((1.25, 0.15), (1.75, 0.50), (2.50, 1.00))}, "High-return train lead but risky DD.", "risk_check"),
        c("disable_structure_trail", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False}, "Risky no-trail ablation.", "risk_check"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--train-end", default=TRAIN_END)
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir or ROUND_DIR / f"oos_candidate_check_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = read_json(CONFIG_PATH)
    candidates = dedupe(candidate_specs(base))
    print(f"[oos] evaluating {len(candidates)} candidates max_workers={args.max_workers} train_end={args.train_end}", flush=True)

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.max_workers)),
        initializer=_init_worker,
        initargs=(str(DATA_DIR), args.train_end),
    ) as pool:
        futures = {pool.submit(_score_worker, candidate): candidate for candidate in candidates}
        for idx, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "name": candidate.name,
                    "family": candidate.family,
                    "thesis": candidate.thesis,
                    "mutations": normalize(candidate.mutations),
                    "error": repr(exc),
                    "train": {},
                    "oos": {},
                }
            rows.append(row)
            if idx % 5 == 0 or idx == len(candidates):
                best = best_so_far(rows)
                print(f"[oos] {idx}/{len(candidates)} best={best.get('name')} oos_net={best.get('oos', {}).get('net_return_pct', 0):+.2f}%", flush=True)

    baseline = next(row for row in rows if row["name"] == "baseline_round8")
    scored = sorted((score_row(row, baseline) for row in rows), key=lambda row: row["objective"], reverse=True)
    recommended = choose_recommended(scored, baseline)
    write_json(output_dir / "oos_candidate_results.json", scored)
    write_csv(output_dir / "oos_candidate_results.csv", flatten(scored))
    write_json(output_dir / "recommended_config.json", recommended.get("mutations", base))
    report = format_report(scored, baseline, recommended, args.train_end, time.time() - started)
    (output_dir / "oos_candidate_report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[oos] wrote {output_dir.resolve()}", flush=True)


def _init_worker(data_dir_str: str, train_end: str) -> None:
    global _TRAIN_DATA, _FULL_DATA, _DATA_DIR, _OOS_WARMUP_15M
    global _TRAIN_INDICATOR_CACHE, _OOS_INDICATOR_CACHE
    _DATA_DIR = Path(data_dir_str)
    _TRAIN_DATA = load_tpc_replay_bundle(_DATA_DIR, end_date=train_end).data
    _FULL_DATA = load_tpc_replay_bundle(_DATA_DIR, end_date=None).data
    _OOS_WARMUP_15M = infer_holdout_warmup(_FULL_DATA, train_end)
    _TRAIN_INDICATOR_CACHE = {}
    _OOS_INDICATOR_CACHE = {}


def _score_worker(candidate: Candidate) -> dict[str, Any]:
    if _TRAIN_DATA is None or _FULL_DATA is None or _DATA_DIR is None:
        raise RuntimeError("worker not initialised")
    cfg_train = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=_DATA_DIR).with_overrides(candidate.mutations)
    train_result = run_tpc_independent(_TRAIN_DATA, cfg_train, indicator_cache=_TRAIN_INDICATOR_CACHE)

    oos_mutations = dict(candidate.mutations)
    oos_mutations["warmup_15m"] = _OOS_WARMUP_15M
    cfg_oos = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=_DATA_DIR).with_overrides(oos_mutations)
    oos_result = run_tpc_independent(_FULL_DATA, cfg_oos, indicator_cache=_OOS_INDICATOR_CACHE)

    return {
        "name": candidate.name,
        "family": candidate.family,
        "thesis": candidate.thesis,
        "mutations": normalize(candidate.mutations),
        "train": metrics_with_cohorts(train_result),
        "oos": metrics_with_cohorts(oos_result),
    }


def infer_holdout_warmup(data: dict[str, dict[str, Any]], train_end: str) -> int:
    if not data:
        return 1
    primary = max(data, key=lambda symbol: len(data[symbol]["bars_15m"].closes))
    times = pd.DatetimeIndex(data[primary]["bars_15m"].times)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    cutoff = pd.Timestamp(train_end)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    return int(np.searchsorted(times.values, cutoff.to_datetime64(), side="left"))


def metrics_with_cohorts(result: Any) -> dict[str, Any]:
    metrics = dict(_extract_tpc_metrics(result, INITIAL_EQUITY))
    trades = list(getattr(result, "trades", []))
    cohorts: dict[str, dict[str, float]] = {}
    specs: dict[str, list[Any]] = {
        "LONG": [t for t in trades if int(getattr(t, "direction", 0) or 0) > 0],
        "SHORT": [t for t in trades if int(getattr(t, "direction", 0) or 0) < 0],
    }
    for symbol in ("QQQ", "GLD"):
        specs[symbol] = [t for t in trades if str(getattr(t, "symbol", "") or "") == symbol]
    for name, items in specs.items():
        rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in items], dtype=float)
        pnls = np.asarray([float(getattr(t, "pnl_dollars", 0.0) or 0.0) for t in items], dtype=float)
        mfes = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in items], dtype=float)
        cohorts[name] = {
            "n": float(len(items)),
            "pnl": float(np.sum(pnls)) if pnls.size else 0.0,
            "avg_r": float(np.mean(rs)) if rs.size else 0.0,
            "win_rate": float(np.mean(rs > 0.0)) if rs.size else 0.0,
            "low_mfe_loss_rate": float(np.mean((mfes < 1.0) & (rs <= 0.0))) if rs.size else 0.0,
        }
    metrics["cohorts"] = cohorts
    return normalize(metrics)


def score_row(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    train = row.get("train", {}) or {}
    oos = row.get("oos", {}) or {}
    base_train = baseline.get("train", {}) or {}
    base_oos = baseline.get("oos", {}) or {}

    delta = {
        "train_net_return_pct": f(train, "net_return_pct") - f(base_train, "net_return_pct"),
        "train_trades": f(train, "total_trades") - f(base_train, "total_trades"),
        "train_avg_r": f(train, "avg_r") - f(base_train, "avg_r"),
        "train_max_dd_pct": f(train, "max_dd_pct") - f(base_train, "max_dd_pct"),
        "oos_net_return_pct": f(oos, "net_return_pct") - f(base_oos, "net_return_pct"),
        "oos_trades": f(oos, "total_trades") - f(base_oos, "total_trades"),
        "oos_trades_per_month": f(oos, "trades_per_month") - f(base_oos, "trades_per_month"),
        "oos_avg_r": f(oos, "avg_r") - f(base_oos, "avg_r"),
        "oos_max_dd_pct": f(oos, "max_dd_pct") - f(base_oos, "max_dd_pct"),
        "oos_low_mfe_loss_rate": f(oos, "low_mfe_loss_rate") - f(base_oos, "low_mfe_loss_rate"),
    }
    train_retention = safe_div(f(train, "net_return_pct"), max(f(base_train, "net_return_pct"), 1e-9))
    trade_retention = safe_div(f(train, "total_trades"), max(f(base_train, "total_trades"), 1e-9))
    oos_freq_retention = safe_div(f(oos, "total_trades"), max(f(base_oos, "total_trades"), 1e-9))
    # OOS leads, train acts as a sanity prior because the OOS window is small.
    objective = (
        0.34 * scale(f(oos, "net_return_pct"), -12.0, 8.0)
        + 0.20 * scale(f(oos, "avg_r"), -0.70, 0.30)
        + 0.14 * scale(f(oos, "trades_per_month"), 1.2, 3.8)
        + 0.12 * scale(f(oos, "dollar_profit_factor"), 0.30, 1.40)
        + 0.08 * scale(train_retention, 0.75, 1.25)
        + 0.06 * scale(trade_retention, 0.75, 1.10)
        + 0.06 * (1.0 - scale(f(oos, "max_dd_pct"), 5.0, 18.0))
        - 0.07 * scale(f(oos, "low_mfe_loss_rate"), 0.35, 0.70)
    )
    robust_gate = (
        f(oos, "net_return_pct") >= f(base_oos, "net_return_pct")
        and f(oos, "avg_r") >= f(base_oos, "avg_r") - 0.08
        and f(oos, "total_trades") >= max(10.0, f(base_oos, "total_trades") * 0.80)
        and f(oos, "max_dd_pct") <= min(15.0, f(base_oos, "max_dd_pct") + 1.0)
        and train_retention >= 0.85
        and trade_retention >= 0.90
    )
    out = dict(row)
    out["delta"] = delta
    out["retention"] = {
        "train_net": train_retention,
        "train_trades": trade_retention,
        "oos_trades": oos_freq_retention,
    }
    out["objective"] = objective
    out["robust_gate"] = robust_gate
    return out


def choose_recommended(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    robust = [row for row in rows if row.get("name") != "baseline_round8" and row.get("robust_gate")]
    if robust:
        return max(robust, key=lambda row: (f(row.get("oos", {}), "net_return_pct"), row["objective"]))
    # If nothing clears the OOS gate, preserve baseline rather than promoting a
    # train winner that worsens holdout behaviour.
    return next(row for row in rows if row.get("name") == "baseline_round8")


def best_so_far(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: f(row.get("oos", {}), "net_return_pct"))


def format_report(rows: list[dict[str, Any]], baseline: dict[str, Any], recommended: dict[str, Any], train_end: str, elapsed: float) -> str:
    base_train = baseline["train"]
    base_oos = baseline["oos"]
    top_oos = sorted([r for r in rows if r["name"] != "baseline_round8"], key=lambda r: f(r["oos"], "net_return_pct"), reverse=True)[:12]
    top_obj = sorted([r for r in rows if r["name"] != "baseline_round8"], key=lambda r: r["objective"], reverse=True)[:12]
    robust = [r for r in rows if r.get("robust_gate") and r["name"] != "baseline_round8"]
    lines = [
        "# TPC Round 8 OOS Candidate Check",
        "",
        f"Elapsed minutes: {elapsed / 60.0:.1f}",
        f"Split: train <= `{train_end}`, OOS after split with warmup at the cutoff.",
        "",
        "## Baseline",
        f"Train: net {f(base_train,'net_return_pct'):+.2f}%, avgR {f(base_train,'avg_r'):+.3f}, trades {f(base_train,'total_trades'):.0f}, trades/month {f(base_train,'trades_per_month'):.2f}, DD {f(base_train,'max_dd_pct'):.2f}%, PF$ {f(base_train,'dollar_profit_factor'):.2f}.",
        f"OOS: net {f(base_oos,'net_return_pct'):+.2f}%, avgR {f(base_oos,'avg_r'):+.3f}, trades {f(base_oos,'total_trades'):.0f}, trades/month {f(base_oos,'trades_per_month'):.2f}, DD {f(base_oos,'max_dd_pct'):.2f}%, PF$ {f(base_oos,'dollar_profit_factor'):.2f}, low-MFE {f(base_oos,'low_mfe_loss_rate'):.1%}.",
        "",
        "## Top OOS Net",
        "| Candidate | Family | Train Net | OOS Net | dOOS | OOS Trades | OOS AvgR | OOS DD | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_oos:
        lines.append(row_line(row))
    lines.extend([
        "",
        "## Top Blended Objective",
        "| Candidate | Family | Objective | Train Net | OOS Net | OOS Trades | OOS AvgR | OOS DD | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in top_obj:
        lines.append(obj_line(row))
    if robust:
        lines.extend([
            "",
            "## Robust-Gate Passes",
            "| Candidate | Train Net | OOS Net | OOS Trades | OOS AvgR | OOS DD | Thesis |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in sorted(robust, key=lambda r: f(r["oos"], "net_return_pct"), reverse=True):
            lines.append(
                f"| {row['name']} | {f(row['train'],'net_return_pct'):+.2f}% | {f(row['oos'],'net_return_pct'):+.2f}% | "
                f"{f(row['oos'],'total_trades'):.0f} | {f(row['oos'],'avg_r'):+.3f} | {f(row['oos'],'max_dd_pct'):.2f}% | {row['thesis']} |"
            )
    lines.extend([
        "",
        "## Recommendation",
        f"Selected: `{recommended['name']}`.",
        f"Train: net {f(recommended['train'],'net_return_pct'):+.2f}%, avgR {f(recommended['train'],'avg_r'):+.3f}, trades {f(recommended['train'],'total_trades'):.0f}, DD {f(recommended['train'],'max_dd_pct'):.2f}%.",
        f"OOS: net {f(recommended['oos'],'net_return_pct'):+.2f}%, avgR {f(recommended['oos'],'avg_r'):+.3f}, trades {f(recommended['oos'],'total_trades'):.0f}, DD {f(recommended['oos'],'max_dd_pct'):.2f}%.",
        "OOS is now selection data for these candidates, so the selected config still needs a fresh validation slice or paper-trade pass before promotion.",
    ])
    return "\n".join(lines) + "\n"


def row_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['name']} | {row['family']} | {f(row['train'],'net_return_pct'):+.2f}% | "
        f"{f(row['oos'],'net_return_pct'):+.2f}% | {row['delta']['oos_net_return_pct']:+.2f}% | "
        f"{f(row['oos'],'total_trades'):.0f} | {f(row['oos'],'avg_r'):+.3f} | {f(row['oos'],'max_dd_pct'):.2f}% | {row.get('robust_gate')} |"
    )


def obj_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['name']} | {row['family']} | {row['objective']:.3f} | {f(row['train'],'net_return_pct'):+.2f}% | "
        f"{f(row['oos'],'net_return_pct'):+.2f}% | {f(row['oos'],'total_trades'):.0f} | {f(row['oos'],'avg_r'):+.3f} | "
        f"{f(row['oos'],'max_dd_pct'):.2f}% | {row.get('robust_gate')} |"
    )


def flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        train = row.get("train", {})
        oos = row.get("oos", {})
        out.append(
            {
                "name": row.get("name"),
                "family": row.get("family"),
                "objective": row.get("objective"),
                "robust_gate": row.get("robust_gate"),
                "train_net_return_pct": f(train, "net_return_pct"),
                "train_avg_r": f(train, "avg_r"),
                "train_total_trades": f(train, "total_trades"),
                "train_trades_per_month": f(train, "trades_per_month"),
                "train_max_dd_pct": f(train, "max_dd_pct"),
                "train_dollar_profit_factor": f(train, "dollar_profit_factor"),
                "oos_net_return_pct": f(oos, "net_return_pct"),
                "oos_avg_r": f(oos, "avg_r"),
                "oos_total_trades": f(oos, "total_trades"),
                "oos_trades_per_month": f(oos, "trades_per_month"),
                "oos_max_dd_pct": f(oos, "max_dd_pct"),
                "oos_dollar_profit_factor": f(oos, "dollar_profit_factor"),
                "oos_low_mfe_loss_rate": f(oos, "low_mfe_loss_rate"),
                "delta_oos_net_return_pct": row.get("delta", {}).get("oos_net_return_pct", 0.0),
                "delta_oos_trades": row.get("delta", {}).get("oos_trades", 0.0),
                "thesis": row.get("thesis"),
            }
        )
    return out


def f(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key, 0.0) if isinstance(payload, dict) else 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def scale(value: float, low: float, high: float) -> float:
    if high <= low or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else 0.0


def dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        sig = json.dumps(normalize(candidate.mutations), sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(candidate)
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(normalize(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(normalize(rows))


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


if __name__ == "__main__":
    mp.freeze_support()
    main()
