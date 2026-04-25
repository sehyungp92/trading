from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_manifest_path() -> Path:
    return repo_root() / "tests" / "fixtures" / "backtest_baselines" / "manifest.json"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or default_manifest_path()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_repo_path(relative_path: str, *, root: Path | None = None) -> Path:
    return (root or repo_root()) / Path(relative_path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_summary_source(entry: dict[str, Any], *, root: Path | None = None) -> Any | None:
    source = entry.get("summary_source")
    if source:
        source_path = resolve_repo_path(source["path"], root=root)
        kind = source.get("kind", "json")
        if kind != "json":
            raise ValueError(f"Unsupported summary source kind: {kind}")
        return json.loads(source_path.read_text(encoding="utf-8"))

    artifact_path = resolve_repo_path(entry["artifact_path"], root=root)
    phase_manifest = artifact_path.parent / "phase_run_manifest.json"
    if phase_manifest.exists():
        return json.loads(phase_manifest.read_text(encoding="utf-8"))
    return None


def collect_baseline_snapshot(
    entry: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    base_root = root or repo_root()
    artifact_path = resolve_repo_path(entry["artifact_path"], root=base_root)
    summary_data = load_summary_source(entry, root=base_root)
    text = artifact_path.read_text(encoding="utf-8")
    return {
        "sha256": sha256_file(artifact_path),
        "metrics": parse_diagnostic_metrics(
            text,
            parser_kind=entry["parser_kind"],
            summary_data=summary_data,
        ),
    }


def parse_diagnostic_metrics(
    text: str,
    *,
    parser_kind: str,
    summary_data: Any | None = None,
) -> dict[str, float]:
    if parser_kind == "swing_aggregate_summary":
        block = _section_between(text, "--- AGGREGATE SUMMARY ---", ["PER-SYMBOL SUMMARY"])
        return {
            "total_trades": _extract_labeled_number(block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(block, "Win rate"),
            "profit_factor": _extract_labeled_number(block, "Profit factor"),
            "total_pnl": _extract_labeled_number(block, "Total PnL"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max drawdown"),
        }
    if parser_kind == "swing_helix_summary":
        block = _section_between(text, "Fee-net trade count:", ["A) PER-SYMBOL TRADE SUMMARY"])
        return {
            "total_trades": _extract_labeled_number(block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(block, "Win Rate"),
            "profit_factor": _extract_labeled_number(block, "Profit Factor"),
            "total_pnl": _extract_labeled_number(block, "Total PnL"),
            "net_return_pct": _extract_labeled_number(block, "Net Return"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max Drawdown"),
        }
    if parser_kind == "brs_topline":
        block = _section_between(text, "A) Topline", ["B) Strength / Weakness Snapshot"])
        return {
            "campaigns": _extract_labeled_number(block, "Campaigns"),
            "fee_net_pnl": _extract_labeled_number(block, "Fee-net PnL"),
            "profit_factor": _extract_labeled_number(block, "Profit factor"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max drawdown"),
            "composite_score": _extract_labeled_number(block, "Composite score"),
        }
    if parser_kind == "momentum_performance_summary":
        summary_block = _section_between(text, "PERFORMANCE SUMMARY", ["FUNNEL"])
        return {
            "composite_score": _extract_labeled_number(text, "Composite Score"),
            "total_trades": _extract_labeled_number(summary_block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(summary_block, "Win rate"),
            "profit_factor": _extract_labeled_number(summary_block, "Profit factor"),
            "net_profit": _extract_labeled_number(summary_block, "Net profit"),
            "max_drawdown_pct": _extract_labeled_number(summary_block, "Max drawdown"),
        }
    if parser_kind == "momentum_performance_report":
        return {
            "total_trades": _extract_labeled_number(text, "Total trades"),
            "win_rate_pct": _extract_labeled_number(text, "Win rate"),
            "profit_factor": _extract_labeled_number(text, "Profit factor"),
            "net_profit": _extract_labeled_number(text, "Net profit"),
            "max_drawdown_pct": _extract_labeled_number(text, "Max drawdown"),
        }
    if parser_kind == "downturn_summary":
        block = _section_between(text, "--- Summary ---", ["--- Per-Engine Breakdown ---"])
        return {
            "total_trades": _extract_labeled_number(block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(block, "Win rate"),
            "profit_factor": _extract_labeled_number(block, "Profit factor"),
            "net_return_pct": _extract_labeled_number(block, "Net return"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max drawdown"),
            "correction_pnl_pct": _extract_labeled_number(text, "Correction PnL"),
        }
    if parser_kind == "stock_alcb_round_summary":
        core = _require_match(
            re.search(
                r"Core:\s*trades=(?P<trades>[+\-]?\d+(?:\.\d+)?),\s*"
                r"net_profit=(?P<pnl>[+\-]?\d+(?:,\d{3})*(?:\.\d+)?),.*?"
                r"pf=(?P<pf>[+\-]?\d+(?:\.\d+)?),\s*dd=(?P<dd>[+\-]?\d+(?:\.\d+)?)%",
                text,
            ),
            "ALCB core summary",
        )
        overview = _section_between(text, "1. Overview", ["2. Signal Funnel"])
        return {
            "total_trades": _to_number(core.group("trades")),
            "total_pnl": _to_number(core.group("pnl")),
            "profit_factor": _to_number(core.group("pf")),
            "max_drawdown_pct": _to_number(core.group("dd")),
            "win_rate_pct": _extract_labeled_number(overview, "Win Rate"),
        }
    if parser_kind == "stock_iaric_phase_summary":
        if not isinstance(summary_data, dict) or "live_metrics" not in summary_data:
            raise ValueError("IARIC baseline requires summary JSON with live_metrics")
        live_metrics = summary_data["live_metrics"]
        return {
            "total_trades": _to_number(live_metrics["n"]),
            "win_rate_pct": _to_number(live_metrics["wr"]) * 100.0,
            "profit_factor": _to_number(live_metrics["pf"]),
            "total_pnl": _to_number(live_metrics["pnl"]),
            "max_drawdown_r": _extract_labeled_number(text, "Max drawdown"),
        }
    raise ValueError(f"Unsupported parser kind: {parser_kind}")


def _section_between(text: str, start_marker: str, end_markers: list[str]) -> str:
    start_index = text.find(start_marker)
    if start_index == -1:
        raise ValueError(f"Missing section start marker: {start_marker}")
    section = text[start_index + len(start_marker):]
    end_positions = [section.find(marker) for marker in end_markers if section.find(marker) != -1]
    if end_positions:
        section = section[: min(end_positions)]
    return section


def _extract_labeled_number(text: str, label: str) -> float:
    match = _require_match(
        re.search(
            rf"{re.escape(label)}\s*[:=]\s*\$?(?P<value>[+\-]?\d+(?:,\d{{3}})*(?:\.\d+)?)%?",
            text,
            flags=re.IGNORECASE,
        ),
        label,
    )
    return _to_number(match.group("value"))


def _to_number(value: Any) -> float:
    return float(str(value).replace(",", "").strip())


def _require_match(match: re.Match[str] | None, label: str) -> re.Match[str]:
    if match is None:
        raise ValueError(f"Could not parse metric block: {label}")
    return match
