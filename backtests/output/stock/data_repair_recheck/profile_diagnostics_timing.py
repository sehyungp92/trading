from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TRADING_REQUIRE_FROZEN_DATA", "false")

from backtests.shared.auto.phase_state import load_phase_state
from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis
from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
from backtests.stock.analysis.iaric_pullback_diagnostics import pullback_full_diagnostic
from backtests.stock.auto.alcb.phase_scoring import merge_alcb_metrics
from backtests.stock.auto.alcb.time_utils import hydrate_time_mutations
from backtests.stock.auto.config_mutator import mutate_alcb_config, mutate_iaric_config
from backtests.stock.auto.iaric.phase_scoring import enrich_phase_score_metrics, merge_pullback_metrics
from backtests.stock.auto.scoring import extract_metrics
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
from backtests.stock.engine.research_replay import ResearchReplayEngine

DATA_DIR = REPO_ROOT / "backtests" / "stock" / "data" / "raw"
OUT_DIR = REPO_ROOT / "backtests" / "output" / "stock" / "data_repair_recheck"
PROFILE_DIR = OUT_DIR / f"profile_diagnostics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
START = os.environ.get("PROFILE_START", "2024-03-25")
END = os.environ.get("PROFILE_END", "2024-04-30")


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


@contextmanager
def timed(label: str, rows: list[dict[str, Any]]):
    t0 = time.perf_counter()
    log(f"start {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        rows.append({"label": label, "elapsed_seconds": round(elapsed, 3)})
        log(f"done {label}: {elapsed:.3f}s")


def profiled(label: str, rows: list[dict[str, Any]], func: Callable[[], Any]) -> Any:
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    log(f"start {label}")
    profiler.enable()
    try:
        return func()
    finally:
        profiler.disable()
        elapsed = time.perf_counter() - t0
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("cumtime").print_stats(40)
        (PROFILE_DIR / f"{label.replace('/', '_')}.profile.txt").write_text(s.getvalue(), encoding="utf-8")
        rows.append({"label": label, "elapsed_seconds": round(elapsed, 3)})
        log(f"done {label}: {elapsed:.3f}s")


def metric_headline(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("total_trades", "net_profit", "profit_factor", "max_drawdown_pct", "sharpe", "expected_total_r")
    return {key: metrics.get(key) for key in keys if key in metrics}


def run_iaric(replay: ResearchReplayEngine, collect_diagnostics: bool, rows: list[dict[str, Any]]) -> dict[str, Any]:
    state = load_phase_state(REPO_ROOT / "backtests" / "output" / "stock" / "iaric" / "round_1" / "phase_state.json")
    config = mutate_iaric_config(
        IARICBacktestConfig(
            start_date=START,
            end_date=END,
            initial_equity=10_000.0,
            tier=3,
            data_dir=DATA_DIR,
        ),
        state.cumulative_mutations,
    )
    label = f"iaric/engine_collect_{int(collect_diagnostics)}"
    result = profiled(
        label,
        rows,
        lambda: IARICPullbackEngine(config, replay, collect_diagnostics=collect_diagnostics).run(),
    )
    with timed(f"iaric/metrics_collect_{int(collect_diagnostics)}", rows):
        perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, 10_000.0)
        metrics = enrich_phase_score_metrics(
            merge_pullback_metrics(
                perf,
                result.trades,
                candidate_ledger=result.candidate_ledger,
                selection_attribution=result.selection_attribution,
            )
        )
    render_len = 0
    if collect_diagnostics:
        with timed("iaric/full_diagnostic_render", rows):
            text = pullback_full_diagnostic(
                result.trades,
                metrics=metrics,
                replay=replay,
                daily_selections=result.daily_selections,
                candidate_ledger=result.candidate_ledger,
                funnel_counters=result.funnel_counters,
                rejection_log=result.rejection_log,
                shadow_outcomes=result.shadow_outcomes,
                selection_attribution=result.selection_attribution,
                fsm_log=result.fsm_log,
            )
            render_len = len(text)
    return {
        "strategy": "iaric",
        "collect_diagnostics": collect_diagnostics,
        "trades": len(result.trades),
        "diagnostic_text_chars": render_len,
        "metrics": metric_headline(metrics),
    }


def run_alcb(replay: ResearchReplayEngine, collect_diagnostics: bool, rows: list[dict[str, Any]]) -> dict[str, Any]:
    state = load_phase_state(REPO_ROOT / "backtests" / "output" / "stock" / "alcb" / "round_2" / "phase_state.json")
    mutations = hydrate_time_mutations(state.cumulative_mutations)
    config = mutate_alcb_config(
        ALCBBacktestConfig(
            start_date=START,
            end_date=END,
            initial_equity=10_000.0,
            tier=2,
            data_dir=DATA_DIR,
        ),
        mutations,
    )
    tracker = ALCBShadowTracker() if collect_diagnostics else None

    def _run():
        engine = ALCBIntradayEngine(config, replay)
        if tracker is not None:
            engine.shadow_tracker = tracker
        return engine.run()

    label = f"alcb/engine_collect_{int(collect_diagnostics)}"
    result = profiled(label, rows, _run)
    with timed(f"alcb/metrics_collect_{int(collect_diagnostics)}", rows):
        perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, 10_000.0)
        metrics = merge_alcb_metrics(perf, result.trades)
    render_len = 0
    if collect_diagnostics:
        with timed("alcb/full_diagnostic_render", rows):
            text = "\n\n".join(
                [
                    alcb_full_diagnostic(
                        result.trades,
                        shadow_tracker=tracker,
                        daily_selections=result.daily_selections,
                    ),
                    qe_replacement_analysis(result.trades, max_positions=int(config.param_overrides.get("max_positions", 10))),
                ]
            )
            render_len = len(text)
    return {
        "strategy": "alcb",
        "collect_diagnostics": collect_diagnostics,
        "trades": len(result.trades),
        "diagnostic_text_chars": render_len,
        "metrics": metric_headline(metrics),
    }


def main() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    log(f"profile window {START}->{END}")
    with timed("replay/construct", rows):
        replay = ResearchReplayEngine(DATA_DIR, require_bundle=False)
    with timed("replay/fingerprint", rows):
        fingerprint = replay.data_fingerprint()
    with timed("replay/load_all_data", rows):
        replay.load_all_data()
    summaries.append(
        {
            "strategy": "replay",
            "fingerprint": fingerprint,
            "daily_symbols": len(replay._daily_cache),
            "thirty_min_symbols": len(replay._intraday_30m_cache),
            "trading_dates": len(replay.trading_dates),
        }
    )
    summaries.append(run_iaric(replay, False, rows))
    summaries.append(run_iaric(replay, True, rows))
    summaries.append(run_alcb(replay, False, rows))
    summaries.append(run_alcb(replay, True, rows))
    payload = {
        "window": {"start": START, "end": END},
        "timings": rows,
        "summaries": summaries,
    }
    output = PROFILE_DIR / "profile_summary.json"
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
