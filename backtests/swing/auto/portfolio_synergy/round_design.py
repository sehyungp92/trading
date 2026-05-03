from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .phase_candidates import (
    INITIAL_EQUITY,
    PHASE_FOCUS,
    PHASE_GATES,
    RISK_STANCE,
    ROUND_NAME,
    ROUND_TARGETS,
    SCORE_WEIGHTS,
    SEED_PORTFOLIO_CONFIG,
    STRATEGY_ORDER,
    get_phase_candidates,
    phase_summary,
)


@dataclass(frozen=True)
class LatestRoundSource:
    key: str
    strategy_id: str
    round_name: str
    summary_path: str
    diagnostics_path: str


@dataclass(frozen=True)
class StrategyAssessment:
    key: str
    strategy_id: str
    round_name: str
    total_trades: float
    trades_per_month: float
    avg_r: float
    total_r_per_month: float
    net_return_pct: float
    profit_factor: float
    max_drawdown_pct: float
    win_rate: float
    sharpe: float | None
    primary_read: str
    risk_read: str
    allocation_bias: str
    summary_path: str
    diagnostics_path: str
    source_hashes: dict[str, str]
    extra: dict[str, Any]


LATEST_ROUNDS: tuple[LatestRoundSource, ...] = (
    LatestRoundSource(
        "atrss",
        "ATRSS",
        "r9_phase1",
        "backtests/swing/auto/atrss/output/phase_state.json",
        "backtests/swing/auto/atrss/output/r9_phase1_full_diagnostics.txt",
    ),
    LatestRoundSource(
        "helix_exit_alpha",
        "AKC_HELIX",
        "exit_alpha_optimized_baseline",
        "backtests/swing/auto/output/helix_exit_alpha_optimized_baseline_summary.json",
        "backtests/swing/auto/output/helix_exit_alpha_optimized_baseline_full_diagnostics.txt",
    ),
    LatestRoundSource(
        "brs",
        "BRS_R9",
        "r10",
        "backtests/swing/auto/brs/output/phase_state.json",
        "backtests/swing/auto/brs/output/round_final_diagnostics.txt",
    ),
    LatestRoundSource(
        "breakout",
        "SWING_BREAKOUT_V3",
        "optimized_seed",
        "backtests/swing/auto/output/breakout_optimized_seed_summary.json",
        "backtests/swing/auto/output/breakout_optimized_seed_full_diagnostics.txt",
    ),
    LatestRoundSource(
        "overlay",
        "OVERLAY",
        "portfolio_overlay_experiments",
        "backtests/swing/auto/output/results.tsv",
        "backtests/swing/auto/output/report.md",
    ),
)

STRATEGY_READS = {
    "atrss": (
        "High-frequency pullback core: 274 trades, 79.6% WR, 4.7 trades/month, and 2.29% DD in R9.",
        "Long-only and no breakout conversions; use as the frequency core but do not rely on it for short/bear coverage.",
        "primary_frequency_core",
    ),
    "helix_exit_alpha": (
        "Largest standalone return sleeve: exit-alpha Helix shows 348 synchronized trades, PF 2.80, and +252.6R.",
        "Right-then-stopped leakage remains material at 79 trades and 116.6R leaked; size needs leakage and stale-exit guards.",
        "primary_return_engine_capped",
    ),
    "brs": (
        "Bear-regime specialist: r10 reaches 101 trades, PF 5.32, 81.2% bear alpha, and 7.59 bear PF.",
        "Phase 4 improved alpha but regressed Sharpe and DD versus phase 3; keep it in bear/downturn lanes with regression watch.",
        "bear_alpha_sleeve",
    ),
    "breakout": (
        "Very low-drawdown opportunistic sleeve: 37 trades, 86.5% WR, PF 21.21, and only 0.32% DD.",
        "Frequency is thin at roughly 0.6 trades/month and edge decayed in the second half; expand carefully.",
        "low_dd_opportunistic",
    ),
    "overlay": (
        "Idle-cash sleeve, not a signal-quality engine; older tests favored overlay_max_70 over a larger cap.",
        "The overlay-off control scored much better historically, so this round keeps overlay but caps and throttles it.",
        "capped_idle_cash",
    ),
}


def build_round_design(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
    assessments = _load_strategy_assessments(root)
    assessment_map = {item.strategy_id: item for item in assessments}
    ordered_assessments = [assessment_map[strategy_id] for strategy_id in STRATEGY_ORDER]

    active_assessments = [item for item in ordered_assessments if item.strategy_id != "OVERLAY"]
    isolated_trades_per_month = sum(item.trades_per_month for item in active_assessments)
    isolated_total_r_per_month = sum(item.total_r_per_month for item in active_assessments)
    expected_selected_trades_per_month = isolated_trades_per_month * 0.85
    expected_selected_r_per_month = isolated_total_r_per_month * 0.78

    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "family": "swing",
        "strategy": "portfolio_synergy",
        "round": ROUND_NAME,
        "status": "designed_not_executed",
        "generated_at_utc": generated_at,
        "description": (
            "Five-strategy swing phase-auto round design using the latest ATRSS, AKC Helix, "
            "BRS, Swing Breakout, and overlay diagnostics."
        ),
        "risk_stance": RISK_STANCE,
        "initial_equity": INITIAL_EQUITY,
        "starting_equity_assumption": (
            "$25,000 is used as the controlled-aggressive starter book. The prior swing diagnostics "
            "mostly use $10,000, but a five-sleeve ETF portfolio plus overlay needs more room so "
            "active heat, idle-cash deployment, and drawdown governors can operate without forcing "
            "fragile all-or-nothing sizing."
        ),
        "diagnostic_sources": [asdict(source) for source in LATEST_ROUNDS],
        "diagnostic_assessments": [asdict(item) for item in ordered_assessments],
        "diagnostic_conflicts": [
            "ATRSS uses the later r9 synchronized diagnostic, not the older negative optimized-baseline artifact.",
            "Helix uses the exit-alpha optimized baseline as the latest completed result; leakage-control code exists but has no final output yet.",
            "BRS round_final_diagnostics reports PASS, while phase_state still records a phase-4 no-regression DD failure; the round treats BRS as high-alpha with regression watch.",
            "Overlay-off was historically the strongest control, but the user requested overlay participation, so the design caps rather than removes it.",
        ],
        "diagnostic_portfolio_baseline": {
            "isolated_active_trades_per_month": isolated_trades_per_month,
            "isolated_total_r_per_month_proxy": isolated_total_r_per_month,
            "expected_selected_active_trades_per_month": expected_selected_trades_per_month,
            "expected_selected_total_r_per_month_proxy": expected_selected_r_per_month,
            "design_target_active_trades_per_month": ROUND_TARGETS["min_active_trades_per_month"],
            "design_target_total_r_per_month": ROUND_TARGETS["min_total_r_per_month"],
            "design_target_max_drawdown_pct": ROUND_TARGETS["target_max_drawdown_pct"],
            "design_hard_max_drawdown_pct": ROUND_TARGETS["hard_max_drawdown_pct"],
        },
        "seed_portfolio_config": _jsonable(SEED_PORTFOLIO_CONFIG),
        "scoring_weights": SCORE_WEIGHTS,
        "round_targets": ROUND_TARGETS,
        "phase_design": [
            {
                "phase": phase,
                "focus": PHASE_FOCUS[phase],
                "gate": PHASE_GATES[phase],
                "candidates": _jsonable(get_phase_candidates(phase)),
            }
            for phase in sorted(PHASE_FOCUS)
        ],
        "phase_summary": phase_summary(),
        "acceptance_policy": {
            "score_component_limit": len(SCORE_WEIGHTS),
            "min_delta": 0.003,
            "max_rounds_per_phase": 10,
            "hard_rejects": {
                "all_five_sleeves_required": True,
                "reject_if_max_drawdown_pct_gt": ROUND_TARGETS["hard_max_drawdown_pct"],
                "reject_if_active_trades_per_month_lt": ROUND_TARGETS["min_active_trades_per_month"],
                "reject_if_profit_factor_lt": ROUND_TARGETS["min_profit_factor"],
                "reject_if_single_strategy_risk_share_gt": ROUND_TARGETS["max_single_strategy_risk_share"],
                "reject_overlay_off_except_as_control": True,
            },
        },
        "replay_notes": [
            "This artifact is a phase-auto design seed, not an optimized result.",
            "Execution uses synchronized all-five portfolio replay before adopting live risk changes.",
            "BRS is evaluated as a first-class unified swing sleeve, not as a post-hoc equity merge.",
            "Candidates that raise return by starving Breakout, BRS, or overlay participation should fail the synergy gates.",
        ],
    }


def render_markdown(design: dict[str, Any]) -> str:
    baseline = design["diagnostic_portfolio_baseline"]
    lines = [
        "# Swing Portfolio Synergy Round 1 Design",
        "",
        f"Status: {design['status']}",
        f"Generated: {design['generated_at_utc']}",
        f"Initial equity: ${design['initial_equity']:,.0f}",
        f"Risk stance: {design['risk_stance']}",
        "",
        "## Diagnostic Read",
        "",
        "| Strategy | Round | Trades | Trades/Mo | Avg R | R/Mo Proxy | PF | Max DD | Allocation Read |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in design["diagnostic_assessments"]:
        dd = "n/a" if item["strategy_id"] == "OVERLAY" else f"{item['max_drawdown_pct']:.1%}"
        lines.append(
            "| {strategy_id} | {round_name} | {total_trades:.0f} | {trades_per_month:.2f} | "
            "{avg_r:.3f} | {total_r_per_month:.2f} | {profit_factor:.2f} | {dd} | "
            "{allocation_bias} |".format(dd=dd, **item)
        )
    lines.extend(
        [
            "",
            "## Portfolio Baseline",
            "",
            f"Isolated active frequency proxy: {baseline['isolated_active_trades_per_month']:.2f} trades/month.",
            f"Expected selected active frequency: {baseline['expected_selected_active_trades_per_month']:.2f} trades/month.",
            f"Expected selected R/month proxy: {baseline['expected_selected_total_r_per_month_proxy']:.2f}R.",
            f"Design gate: at least {baseline['design_target_active_trades_per_month']:.2f} active trades/month, "
            f"at least {baseline['design_target_total_r_per_month']:.2f}R/month proxy, "
            f"target DD at or below {baseline['design_target_max_drawdown_pct']:.1%}, "
            f"hard DD stop at {baseline['design_hard_max_drawdown_pct']:.1%}.",
            "",
            "## Seed Allocation",
            "",
            "| Sleeve | Risk/Cap | Max Heat | Daily Stop | Priority | Role |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    allocations = design["seed_portfolio_config"]["strategy_allocations"]
    for strategy_id in STRATEGY_ORDER:
        alloc = allocations[strategy_id]
        if strategy_id == "OVERLAY":
            risk = f"{alloc['max_equity_pct']:.0%} cap"
            max_heat = "n/a"
            daily_stop = "n/a"
        else:
            risk = f"{alloc['unit_risk_pct']:.2%}"
            max_heat = f"{alloc['max_heat_R']:.2f}R"
            daily_stop = f"{alloc['daily_stop_R']:.2f}R"
        lines.append(
            f"| {strategy_id} | {risk} | {max_heat} | {daily_stop} | "
            f"{alloc['priority']} | {alloc['role']} |"
        )
    lines.extend(["", "## Phase Auto Design", ""])
    for phase in design["phase_design"]:
        lines.append(
            f"{phase['phase']}. {phase['focus']} - {len(phase['candidates'])} candidates; "
            f"gate {phase['gate']}"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- All five sleeves start enabled: ATRSS, AKC_HELIX, BRS_R9, SWING_BREAKOUT_V3, and OVERLAY.",
            "- Alpha return and trade frequency are deliberately overweighted, but DD hard-rejects above 12%.",
            "- Overlay-off is a control candidate only; adoption would violate this round's overlay requirement.",
            "- BRS is replayed inside the synchronized portfolio path before the final blend is considered executable.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_round_design(output_dir: Path, repo_root: Path | None = None) -> dict[str, Path]:
    design = build_round_design(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_spec": output_dir / "run_spec.json",
        "seed_config": output_dir / "seed_portfolio_config.json",
        "candidate_space": output_dir / "candidate_space.json",
        "assessment": output_dir / "diagnostic_assessment.json",
        "evaluation": output_dir / "round_evaluation.txt",
    }
    paths["run_spec"].write_text(json.dumps(design, indent=2), encoding="utf-8")
    paths["seed_config"].write_text(
        json.dumps(design["seed_portfolio_config"], indent=2),
        encoding="utf-8",
    )
    paths["candidate_space"].write_text(
        json.dumps(design["phase_design"], indent=2),
        encoding="utf-8",
    )
    paths["assessment"].write_text(
        json.dumps(design["diagnostic_assessments"], indent=2),
        encoding="utf-8",
    )
    paths["evaluation"].write_text(render_markdown(design), encoding="utf-8")
    return paths


def _load_strategy_assessments(repo_root: Path) -> list[StrategyAssessment]:
    raw: list[tuple[LatestRoundSource, dict[str, Any], dict[str, Any]]] = []
    observed_months: list[float] = []
    for source in LATEST_ROUNDS:
        metrics, extra = _load_metrics_for_source(repo_root, source)
        raw.append((source, metrics, extra))
        total_trades = _metric(metrics, "total_trades")
        trades_per_month = _metric(metrics, "trades_per_month")
        if total_trades > 0 and trades_per_month > 0:
            observed_months.append(total_trades / trades_per_month)

    backtest_months = max(observed_months) if observed_months else 58.0
    assessments: list[StrategyAssessment] = []
    for source, metrics, extra in raw:
        total_trades = _metric(metrics, "total_trades")
        trades_per_month = _metric(metrics, "trades_per_month")
        if trades_per_month <= 0 and total_trades > 0 and backtest_months > 0:
            trades_per_month = total_trades / backtest_months
        total_r = _metric(metrics, "total_r")
        avg_r = _metric(metrics, "avg_r")
        if avg_r <= 0 and total_trades > 0 and total_r:
            avg_r = total_r / total_trades
        total_r_per_month = _metric(metrics, "total_r_per_month")
        if total_r_per_month <= 0 and trades_per_month > 0:
            total_r_per_month = trades_per_month * avg_r
        primary_read, risk_read, allocation_bias = STRATEGY_READS[source.key]
        summary_path = repo_root / source.summary_path
        diagnostics_path = repo_root / source.diagnostics_path
        assessments.append(
            StrategyAssessment(
                key=source.key,
                strategy_id=source.strategy_id,
                round_name=source.round_name,
                total_trades=total_trades,
                trades_per_month=trades_per_month,
                avg_r=avg_r,
                total_r_per_month=total_r_per_month,
                net_return_pct=_metric(metrics, "net_return_pct", "return_pct"),
                profit_factor=_metric(metrics, "profit_factor"),
                max_drawdown_pct=_metric(metrics, "max_drawdown_pct", "max_dd_pct"),
                win_rate=_metric(metrics, "win_rate", "win_rate_pct"),
                sharpe=_optional_metric(metrics, "sharpe"),
                primary_read=primary_read,
                risk_read=risk_read,
                allocation_bias=allocation_bias,
                summary_path=source.summary_path,
                diagnostics_path=source.diagnostics_path,
                source_hashes={
                    "summary": _file_sha256(summary_path),
                    "diagnostics": _file_sha256(diagnostics_path),
                },
                extra=extra,
            )
        )
    return assessments


def _load_metrics_for_source(repo_root: Path, source: LatestRoundSource) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = repo_root / source.summary_path
    diagnostics_path = repo_root / source.diagnostics_path
    if source.key == "atrss":
        state = _load_json(summary_path)
        metrics = dict(state.get("phase_results", {}).get("1", {}).get("final_metrics", {}))
        optimizer_pf = metrics.get("profit_factor")
        fee_net_pf = _extract_after_marker(diagnostics_path, r"--- AGGREGATE SUMMARY ---", r"Profit factor:\s+([\d.]+)")
        if fee_net_pf is not None:
            metrics["profit_factor"] = fee_net_pf
        return metrics, {
            "kept_features": state.get("phase_results", {}).get("1", {}).get("kept_features", []),
            "diagnostic_note": "R9 phase 1 is the latest completed ATRSS synchronized diagnostic.",
            "optimizer_reference_profit_factor": optimizer_pf,
            "fee_net_aggregate_profit_factor": fee_net_pf,
        }
    if source.key == "brs":
        state = _load_json(summary_path)
        metrics = dict(state.get("phase_results", {}).get("4", {}).get("final_metrics", {}))
        total_r = _extract_float(diagnostics_path, r"captured\s+([\d.]+)R")
        if total_r is not None:
            metrics["total_r"] = total_r
            trades = _metric(metrics, "total_trades")
            if trades:
                metrics["avg_r"] = total_r / trades
        return metrics, {
            "phase4_gate_passed_in_state": bool(state.get("phase_gate_results", {}).get("4", {}).get("passed")),
            "diagnostic_gate_passed": "Result: PASS" in diagnostics_path.read_text(encoding="utf-8", errors="replace"),
        }
    if source.key == "overlay":
        overlay_rows = _load_overlay_rows(summary_path)
        return {
            "total_trades": 0.0,
            "trades_per_month": 0.0,
            "avg_r": 0.0,
            "total_r_per_month": 0.0,
            "net_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
        }, {
            "overlay_experiment_rows": overlay_rows,
            "recommended_starting_cap": 0.70,
        }

    data = _load_json(summary_path)
    metrics = dict(data)
    if "win_rate_pct" in metrics and "win_rate" not in metrics:
        metrics["win_rate"] = float(metrics["win_rate_pct"]) / 100.0
    if "total_r" in metrics and "total_trades" in metrics and "avg_r" not in metrics:
        trades = float(metrics.get("total_trades") or 0.0)
        if trades:
            metrics["avg_r"] = float(metrics["total_r"]) / trades
    return metrics, {}


def _load_overlay_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            experiment_id = row.get("experiment_id", "")
            if not experiment_id.startswith("pf_overlay"):
                continue
            rows[experiment_id] = {
                "delta_pct": _safe_float(row.get("delta_pct")),
                "experiment_score": _safe_float(row.get("experiment_score")),
                "status": row.get("status", ""),
                "description": row.get("description", ""),
            }
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return _safe_float(value, default=default)
    return default


def _optional_metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_float(path: Path, pattern: str) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    if not match:
        return None
    return _safe_float(match.group(1), default=0.0)


def _extract_after_marker(path: Path, marker_pattern: str, value_pattern: str) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = re.search(marker_pattern, text)
    if not marker:
        return None
    match = re.search(value_pattern, text[marker.end():])
    if not match:
        return None
    return _safe_float(match.group(1), default=0.0)


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
