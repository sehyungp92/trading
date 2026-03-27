"""CLI for regime backtesting and optimization.

Usage:
    python -m research.backtests.regime.cli download
    python -m research.backtests.regime.cli run [--diagnostics]
    python -m research.backtests.regime.cli optimize [--max-rounds N] [--max-workers N]
    python -m research.backtests.regime.cli walk-forward [--test-years 2]
    python -m research.backtests.regime.cli phase-run --phase N [--max-rounds 20]
    python -m research.backtests.regime.cli phase-gate --phase N
    python -m research.backtests.regime.cli phase-diagnostics --phase N
    python -m research.backtests.regime.cli historical-validate [--mutations-json ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root on path
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    """Download and cache all data from FRED + yfinance."""
    from research.backtests.regime.data.downloader import build_all_data

    data_dir = Path(args.data_dir)
    print(f"Downloading data to {data_dir}...")
    macro_df, market_df, strat_ret_df = build_all_data(data_dir)

    print(f"\n=== Data Download Complete ===")
    print(f"\nmacro_df:")
    print(f"  Columns: {list(macro_df.columns)}")
    print(f"  Date range: {macro_df.index.min().date()} to {macro_df.index.max().date()}")
    print(f"  Rows: {len(macro_df):,}")
    print(f"  NaN: {macro_df.isna().sum().to_dict()}")

    print(f"\nmarket_df:")
    print(f"  Columns: {list(market_df.columns)}")
    print(f"  Date range: {market_df.index.min().date()} to {market_df.index.max().date()}")
    print(f"  Rows: {len(market_df):,}")
    print(f"  NaN: {market_df.isna().sum().to_dict()}")

    print(f"\nstrat_ret_df:")
    print(f"  Columns: {list(strat_ret_df.columns)}")
    print(f"  Date range: {strat_ret_df.index.min().date()} to {strat_ret_df.index.max().date()}")
    print(f"  Rows: {len(strat_ret_df):,}")
    print(f"  NaN: {strat_ret_df.isna().sum().to_dict()}")

    print(f"\nFiles saved to: {data_dir}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single backtest with default or custom MetaConfig."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from research.backtests.regime.auto.scoring import composite_score
    from research.backtests.regime.config import RegimeBacktestConfig
    from research.backtests.regime.data.downloader import load_cached_data
    from research.backtests.regime.engine.portfolio_sim import (
        simulate_benchmark_60_40,
        simulate_portfolio,
    )

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    # Load mutations if provided
    mutations = {}
    if args.mutations_json:
        with open(args.mutations_json) as f:
            data = json.load(f)
        if "accepted_mutations" in data:
            mutations = data["accepted_mutations"]
        else:
            mutations = data
        print(f"Loaded {len(mutations)} mutations from {args.mutations_json}")

    # Build config
    from research.backtests.regime.auto.config_mutator import mutate_meta_config
    cfg = mutate_meta_config(MetaConfig(), mutations) if mutations else MetaConfig()

    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    print(f"Running regime backtest...")
    print(f"  Equity: ${args.equity:,.0f}")
    print(f"  Cost: {args.cost_bps} bps")
    print(f"  Data: {strat_ret_df.index.min().date()} to {strat_ret_df.index.max().date()}")

    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
    score = composite_score(result.metrics)

    _print_metrics(result, score)

    if args.diagnostics:
        from research.backtests.regime.analysis.diagnostics import (
            generate_regime_diagnostics_report,
        )

        benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)
        report = generate_regime_diagnostics_report(
            signals=signals,
            result=result,
            benchmark=benchmark,
            score=score,
            sim_cfg=sim_cfg,
        )
        print(report)

        output_dir = Path("research/backtests/regime/auto/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "diagnostics_latest.txt"
        output_path.write_text(report)
        print(f"\nDiagnostics saved to {output_path}")


def _print_metrics(result, score) -> None:
    """Print portfolio metrics and composite score."""
    m = result.metrics
    print(f"\n=== Portfolio Metrics ===")
    print(f"  Total return:     {m.total_return:>10.1%}")
    print(f"  CAGR:             {m.cagr:>10.2%}")
    print(f"  Sharpe:           {m.sharpe:>10.3f}")
    print(f"  Sortino:          {m.sortino:>10.3f}")
    print(f"  Calmar:           {m.calmar:>10.3f}")
    print(f"  Max drawdown:     {m.max_drawdown_pct:>10.1%}")
    print(f"  Max DD duration:  {m.max_drawdown_duration:>10d} days")
    print(f"  Avg annual TO:    {m.avg_annual_turnover:>10.2f}")
    print(f"  Rebalances:       {m.n_rebalances:>10d}")

    print(f"\n=== Composite Score ===")
    print(f"  Sharpe component:  {score.sharpe_component:.4f}  (w=0.25)")
    print(f"  Calmar component:  {score.calmar_component:.4f}  (w=0.25)")
    print(f"  Inv DD component:  {score.inv_dd_component:.4f}  (w=0.20)")
    print(f"  CAGR component:    {score.cagr_component:.4f}  (w=0.15)")
    print(f"  Sortino component: {score.sortino_component:.4f}  (w=0.15)")
    print(f"  TOTAL:             {score.total:.4f}")
    if score.rejected:
        print(f"  REJECTED: {score.reject_reason}")


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

def cmd_optimize(args: argparse.Namespace) -> None:
    """Run greedy forward selection."""
    import json as _json

    from research.backtests.regime.auto.candidates import get_all_candidates
    from research.backtests.regime.auto.greedy_optimize import (
        run_greedy,
        save_greedy_result,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path("research/backtests/regime/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "greedy_optimal.json"

    candidates = get_all_candidates()

    # Optimization uses daily rebalance + zero costs by default to measure
    # pure prediction quality. Production deployment decides cadence separately.
    base_mutations = {"rebalance_freq": "B"}
    if args.resume:
        resume_path = output_path if args.resume == "auto" else Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                prev = _json.load(f)
            base_mutations = {"rebalance_freq": "B", **prev.get("accepted_mutations", {})}
            accepted_ids = {r["candidate_id"] for r in prev.get("rounds", [])}
            before = len(candidates)
            candidates = [(n, m) for n, m in candidates if n not in accepted_ids]
            print(f"Resuming from {resume_path.name}: "
                  f"base score={prev.get('final_score', 0):.4f}, "
                  f"{len(accepted_ids)} already accepted, "
                  f"{before - len(candidates)} candidates removed")
        else:
            print(f"No previous result at {resume_path}, starting fresh")

    print(f"Loaded {len(candidates)} candidates")

    result = run_greedy(
        candidates=candidates,
        data_dir=data_dir,
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        prune_threshold=args.prune_threshold,
        base_mutations=base_mutations,
        verbose=True,
    )

    save_greedy_result(result, output_path)


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run expanding-window walk-forward validation."""
    import pandas as pd

    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from research.backtests.regime.auto.candidates import get_all_candidates
    from research.backtests.regime.auto.config_mutator import mutate_meta_config
    from research.backtests.regime.auto.greedy_optimize import run_greedy
    from research.backtests.regime.auto.scoring import composite_score
    from research.backtests.regime.config import RegimeBacktestConfig
    from research.backtests.regime.data.downloader import load_cached_data
    from research.backtests.regime.engine.portfolio_sim import simulate_portfolio

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    test_years = args.test_years

    # Build folds dynamically: expanding train, fixed-size test window
    data_start_year = macro_df.index.min().year
    data_end_year = macro_df.index.max().year
    # First test window starts after a minimum 4-year training period
    first_test_start = data_start_year + 5
    folds = []
    test_start = first_test_start
    while test_start + test_years - 1 <= data_end_year:
        train_end = test_start - 1
        test_end = test_start + test_years - 1
        folds.append((
            f"{data_start_year}-01-01",
            f"{train_end}-12-31",
            f"{test_start}-01-01",
            f"{test_end}-12-31",
        ))
        test_start += test_years
    if not folds:
        print("Not enough data for walk-forward validation.")
        return

    print(f"=== Walk-Forward Validation ===")
    print(f"  Folds: {len(folds)}")
    print(f"  Test window: {test_years} years")

    fold_results = []
    candidates = get_all_candidates()

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n--- Fold {i + 1}: Train {train_start}-{train_end}, "
              f"Test {test_start}-{test_end} ---")

        ts_train_end = pd.Timestamp(train_end)
        ts_test_start = pd.Timestamp(test_start)
        ts_test_end = pd.Timestamp(test_end)

        # Filter data for training period
        train_macro = macro_df.loc[:ts_train_end]
        train_market = market_df.loc[:ts_train_end]
        train_strat = strat_ret_df.loc[:ts_train_end]

        # Save filtered training data temporarily for greedy
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_macro.to_parquet(tmp_path / "macro_df.parquet")
            train_market.to_parquet(tmp_path / "market_df.parquet")
            train_strat.to_parquet(tmp_path / "strat_ret_df.parquet")

            # Run greedy on training period
            train_result = run_greedy(
                candidates=list(candidates),
                data_dir=tmp_path,
                initial_equity=args.equity,
                rebalance_cost_bps=args.cost_bps,
                max_workers=args.max_workers,
                max_rounds=args.max_rounds,
                min_delta=args.min_delta,
                verbose=False,
            )

        is_score = train_result.final_score
        optimal_muts = train_result.accepted_mutations

        print(f"  IS score: {is_score:.4f} ({len(train_result.rounds)} mutations accepted)")

        # Evaluate on test period (using full data but scoring only test window)
        cfg = mutate_meta_config(MetaConfig(), optimal_muts) if optimal_muts else MetaConfig()
        sim_cfg = RegimeBacktestConfig(
            initial_equity=args.equity,
            rebalance_cost_bps=args.cost_bps,
            data_dir=data_dir,
        )

        signals = run_signal_engine(
            macro_df=macro_df,
            strat_ret_df=strat_ret_df,
            market_df=market_df,
            growth_feature="GROWTH",
            inflation_feature="INFLATION",
            cfg=cfg,
        )

        oos_result = simulate_portfolio(
            signals, strat_ret_df, sim_cfg,
            start_date=ts_test_start,
            end_date=ts_test_end,
        )
        oos_score = composite_score(oos_result.metrics)

        # OOS regime distribution
        oos_signals = signals.loc[ts_test_start:ts_test_end]
        if not oos_signals.empty:
            _rcols = ["P_G", "P_R", "P_S", "P_D"]
            _dom = oos_signals[_rcols].idxmax(axis=1).str.replace("P_", "")
            oos_regime_dist = _dom.value_counts(normalize=True).to_dict()
        else:
            oos_regime_dist = {}

        print(f"  OOS score: {oos_score.total:.4f}")
        print(f"  OOS Sharpe: {oos_result.metrics.sharpe:.3f}")
        print(f"  OOS CAGR: {oos_result.metrics.cagr:.2%}")
        print(f"  OOS Max DD: {oos_result.metrics.max_drawdown_pct:.1%}")

        fold_results.append({
            "fold": i + 1,
            "train": f"{train_start}-{train_end}",
            "test": f"{test_start}-{test_end}",
            "is_score": is_score,
            "oos_score": oos_score.total,
            "oos_sharpe": oos_result.metrics.sharpe,
            "oos_cagr": oos_result.metrics.cagr,
            "oos_max_dd": oos_result.metrics.max_drawdown_pct,
            "n_mutations": len(train_result.rounds),
            "mutations": list(optimal_muts.keys()),
            "oos_regime_dist": oos_regime_dist,
        })

    # Summary
    print(f"\n=== Walk-Forward Summary ===")
    is_scores = [f["is_score"] for f in fold_results]
    oos_scores = [f["oos_score"] for f in fold_results]

    import numpy as np
    avg_is = np.mean(is_scores)
    avg_oos = np.mean(oos_scores)
    stability = avg_oos / avg_is if avg_is > 0 else 0

    print(f"  Avg IS score:  {avg_is:.4f}")
    print(f"  Avg OOS score: {avg_oos:.4f}")
    print(f"  OOS/IS ratio:  {stability:.3f}")
    print(f"  Positive OOS:  {sum(1 for s in oos_scores if s > 0)}/{len(oos_scores)}")

    print(f"\n  Per-fold:")
    for f in fold_results:
        print(f"    Fold {f['fold']}: IS={f['is_score']:.4f}  "
              f"OOS={f['oos_score']:.4f}  "
              f"Sharpe={f['oos_sharpe']:.3f}  "
              f"CAGR={f['oos_cagr']:.2%}  "
              f"DD={f['oos_max_dd']:.1%}")

    # Save results
    output_dir = Path("research/backtests/regime/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "walk_forward_results.json"
    with open(output_path, "w") as fp:
        json.dump(fold_results, fp, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    # Full diagnostics if requested
    if getattr(args, "diagnostics", False) and fold_results:
        from research.backtests.regime.analysis.diagnostics import (
            generate_regime_diagnostics_report,
        )
        from research.backtests.regime.engine.portfolio_sim import (
            simulate_benchmark_60_40,
        )

        # Use last fold's signals (full-period) and config
        full_result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
        full_score = composite_score(full_result.metrics)
        benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)

        report = generate_regime_diagnostics_report(
            signals=signals,
            result=full_result,
            benchmark=benchmark,
            score=full_score,
            sim_cfg=sim_cfg,
            walk_forward_results=fold_results,
        )
        print(report)

        diag_path = output_dir / "diagnostics_walk_forward.txt"
        diag_path.write_text(report)
        print(f"\nDiagnostics saved to {diag_path}")


# ---------------------------------------------------------------------------
# phase-run
# ---------------------------------------------------------------------------

def cmd_phase_run(args: argparse.Namespace) -> None:
    """Run a single phase of the multi-phase optimization."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from research.backtests.regime.auto.config_mutator import mutate_meta_config
    from research.backtests.regime.auto.greedy_optimize import (
        run_greedy,
        save_greedy_result,
    )
    from research.backtests.regime.auto.phase_candidates import get_phase_candidates
    from research.backtests.regime.auto.phase_scoring import compute_regime_stats
    from research.backtests.regime.auto.phase_state import (
        PhaseState,
        load_phase_state,
        save_phase_state,
    )
    from research.backtests.regime.auto.scoring import composite_score
    from research.backtests.regime.config import RegimeBacktestConfig
    from research.backtests.regime.data.downloader import load_cached_data
    from research.backtests.regime.engine.portfolio_sim import (
        simulate_benchmark_60_40,
        simulate_portfolio,
    )

    phase = args.phase
    data_dir = Path(args.data_dir)
    output_dir = Path("research/backtests/regime/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "phase_state.json"

    # Load or create phase state
    if state_path.exists():
        state = load_phase_state(state_path)
        print(f"Loaded phase state: completed phases {state.completed_phases}")
    else:
        state = PhaseState()
        print("Starting fresh phase state")

    # Get cumulative mutations from prior phases
    base_muts = dict(state.cumulative_mutations)
    base_muts["rebalance_freq"] = "B"  # daily rebalance for optimization

    # Get candidates for this phase (Phase 4 gets prior mutations for adaptive narrowing)
    prior_diag = {"cumulative_mutations": base_muts} if phase == 4 else None
    candidates = get_phase_candidates(phase, prior_diagnostics=prior_diag)
    print(f"Phase {phase}: {len(candidates)} candidates")

    # Run greedy with phase-aware scoring
    result = run_greedy(
        candidates=candidates,
        data_dir=data_dir,
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        prune_threshold=args.prune_threshold,
        base_mutations=base_muts,
        verbose=True,
        phase=phase,
    )

    # Save greedy result
    result_path = output_dir / f"phase_{phase}_result.json"
    save_greedy_result(result, result_path)

    # Update phase state
    new_muts = {k: v for k, v in result.accepted_mutations.items()
                if k != "rebalance_freq"}
    state.advance_phase(phase, new_muts, {
        "baseline_score": result.baseline_score,
        "final_score": result.final_score,
        "n_rounds": len(result.rounds),
        "max_rounds": args.max_rounds,
        "rounds": [r.candidate_id for r in result.rounds],
        "elapsed_seconds": result.elapsed_seconds,
    })
    save_phase_state(state, state_path)

    # Run diagnostics
    print(f"\n--- Running phase {phase} diagnostics ---")
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)
    cfg = mutate_meta_config(MetaConfig(), state.cumulative_mutations)
    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    sim_result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
    score = composite_score(sim_result.metrics)
    regime_stats = compute_regime_stats(signals)

    # Generate full diagnostics report
    from research.backtests.regime.analysis.diagnostics import (
        generate_regime_diagnostics_report,
    )

    benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)
    report = generate_regime_diagnostics_report(
        signals=signals,
        result=sim_result,
        benchmark=benchmark,
        score=score,
        sim_cfg=sim_cfg,
    )

    # Add phase-specific diagnostics
    from research.backtests.regime.auto.phase_diagnostics import (
        generate_phase_diagnostics,
    )

    phase_report = generate_phase_diagnostics(
        phase=phase,
        regime_stats=regime_stats,
        metrics=sim_result.metrics,
        greedy_result=result,
        state=state,
    )

    full_report = report + "\n\n" + phase_report
    diag_path = output_dir / f"phase_{phase}_diagnostics.txt"
    diag_path.write_text(full_report)
    print(full_report)
    print(f"\nDiagnostics saved to {diag_path}")

    # Print regime stats summary
    print(f"\n=== Regime Stats ===")
    print(f"  Active regimes: {regime_stats['n_active_regimes']}")
    print(f"  Regime entropy: {regime_stats['regime_entropy']:.4f}")
    print(f"  Transition rate: {regime_stats['transition_rate']:.4f}")
    print(f"  Distribution: {regime_stats['dominant_dist']}")
    print(f"  Crisis response: {regime_stats['crisis_response']:.4f}")


# ---------------------------------------------------------------------------
# phase-gate
# ---------------------------------------------------------------------------

def cmd_phase_gate(args: argparse.Namespace) -> None:
    """Check the success gate for a completed phase."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from research.backtests.regime.auto.config_mutator import mutate_meta_config
    from research.backtests.regime.auto.phase_gates import check_phase_gate
    from research.backtests.regime.auto.phase_scoring import compute_regime_stats
    from research.backtests.regime.auto.phase_state import (
        load_phase_state,
        save_phase_state,
    )
    from research.backtests.regime.config import RegimeBacktestConfig
    from research.backtests.regime.data.downloader import load_cached_data
    from research.backtests.regime.engine.portfolio_sim import simulate_portfolio

    phase = args.phase
    data_dir = Path(args.data_dir)
    output_dir = Path("research/backtests/regime/auto/output")
    state_path = output_dir / "phase_state.json"

    if not state_path.exists():
        print("No phase state found. Run phase-run first.")
        return

    state = load_phase_state(state_path)
    if phase not in state.completed_phases:
        print(f"Phase {phase} not completed yet. Completed: {state.completed_phases}")
        return

    # Run evaluation with cumulative mutations
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)
    cfg = mutate_meta_config(MetaConfig(), state.cumulative_mutations)
    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    sim_result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
    regime_stats = compute_regime_stats(signals)

    # Check gate
    greedy_data = state.phase_results.get(phase, {})
    gate = check_phase_gate(phase, sim_result.metrics, regime_stats, greedy_data)

    # Record gate result
    from dataclasses import asdict
    state.record_gate(phase, {
        "passed": gate.passed,
        "criteria": [asdict(c) for c in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": gate.recommendations,
    })
    save_phase_state(state, state_path)

    # Print results
    status = "PASSED" if gate.passed else "FAILED"
    print(f"\n=== Phase {phase} Gate: {status} ===")
    for c in gate.criteria:
        mark = "[PASS]" if c.passed else "[FAIL]"
        print(f"  {mark} {c.name}: {c.actual:.4f} (target: {c.target:.4f})")

    if not gate.passed:
        print(f"\n  Failure category: {gate.failure_category}")
        print(f"  Recommendations:")
        for r in gate.recommendations:
            print(f"    - {r}")


# ---------------------------------------------------------------------------
# phase-diagnostics
# ---------------------------------------------------------------------------

def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    """Print diagnostics for a completed phase."""
    output_dir = Path("research/backtests/regime/auto/output")
    diag_path = output_dir / f"phase_{args.phase}_diagnostics.txt"

    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}. Run phase-run first.")
        return

    print(diag_path.read_text())


# ---------------------------------------------------------------------------
# historical-validate
# ---------------------------------------------------------------------------

def cmd_historical_validate(args: argparse.Namespace) -> None:
    """Run historical validation against known regime timeline."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from research.backtests.regime.auto.config_mutator import mutate_meta_config
    from research.backtests.regime.auto.historical_validation import (
        compute_historical_alignment,
        compute_transition_latency,
    )
    from research.backtests.regime.data.downloader import load_cached_data

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    # Load mutations
    mutations = {}
    if args.mutations_json:
        with open(args.mutations_json) as f:
            data = json.load(f)
        if "cumulative_mutations" in data:
            mutations = data["cumulative_mutations"]
        elif "accepted_mutations" in data:
            mutations = data["accepted_mutations"]
        else:
            mutations = data
        print(f"Loaded {len(mutations)} mutations from {args.mutations_json}")
    else:
        # Try phase state
        state_path = Path("research/backtests/regime/auto/output/phase_state.json")
        if state_path.exists():
            from research.backtests.regime.auto.phase_state import load_phase_state
            state = load_phase_state(state_path)
            mutations = state.cumulative_mutations
            print(f"Using cumulative mutations from phase state ({len(mutations)} params)")

    cfg = mutate_meta_config(MetaConfig(), mutations) if mutations else MetaConfig()

    print("Running signal engine...")
    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    alignment = compute_historical_alignment(signals)
    latency = compute_transition_latency(signals)

    print(f"\n=== Historical Validation ===")
    print(f"  Alignment score:  {alignment['overall']:.4f} (target: >0.5)")
    print(f"  Per-period:")
    for name, score in alignment.get("per_period", {}).items():
        print(f"    {name}: {score:.4f}")

    print(f"\n  Transition latency:")
    for name, weeks in latency.items():
        target = "<8 weeks"
        print(f"    {name}: {weeks:.1f} weeks ({target})")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="regime-backtest",
        description="Regime Prediction Backtesting & Optimization",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Shared args
    def add_common(p):
        p.add_argument("--data-dir", default="research/backtests/regime/data/raw",
                       help="Path to data directory")
        p.add_argument("--equity", type=float, default=100_000.0,
                       help="Initial equity (default: 100,000)")
        p.add_argument("--cost-bps", type=float, default=5.0,
                       help="Rebalance cost in bps (default: 5)")

    # download
    dl = sub.add_parser("download", help="Download data from FRED + yfinance")
    add_common(dl)

    # run
    run = sub.add_parser("run", help="Run single backtest")
    add_common(run)
    run.add_argument("--diagnostics", action="store_true",
                     help="Print extended diagnostics")
    run.add_argument("--mutations-json", default=None,
                     help="Path to mutations JSON (e.g., greedy_optimal.json)")

    # optimize
    opt = sub.add_parser("optimize", help="Greedy forward selection")
    add_common(opt)
    opt.set_defaults(cost_bps=0.0)  # zero costs: measure prediction quality, not trading profit
    opt.add_argument("--max-rounds", type=int, default=50,
                     help="Max greedy rounds (default: 50)")
    opt.add_argument("--max-workers", type=int, default=None,
                     help="Parallel workers (default: cpu_count - 1)")
    opt.add_argument("--min-delta", type=float, default=0.001,
                     help="Min score improvement to accept (default: 0.001)")
    opt.add_argument("--prune-threshold", type=float, default=-0.10,
                     help="Prune candidates with delta below this; also prunes failures (default: -0.10)")
    opt.add_argument("--resume", nargs="?", const="auto", default=None,
                     help="Resume from previous result (default: greedy_optimal.json, or pass path)")

    # walk-forward
    wf = sub.add_parser("walk-forward", help="Walk-forward validation")
    add_common(wf)
    wf.add_argument("--test-years", type=int, default=2,
                    help="Test window in years (default: 2)")
    wf.add_argument("--max-rounds", type=int, default=50,
                    help="Max greedy rounds per fold (default: 50)")
    wf.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers (default: cpu_count - 1)")
    wf.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    wf.add_argument("--diagnostics", action="store_true",
                    help="Generate full diagnostics report with WF summary")

    # phase-run
    pr = sub.add_parser("phase-run", help="Run single phase of multi-phase optimization")
    add_common(pr)
    pr.set_defaults(cost_bps=0.0)
    pr.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                    help="Phase number (1-4)")
    pr.add_argument("--max-rounds", type=int, default=20,
                    help="Max greedy rounds (default: 20)")
    pr.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers")
    pr.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    pr.add_argument("--prune-threshold", type=float, default=-0.10,
                    help="Prune candidates below this delta (default: -0.10)")

    # phase-gate
    pg = sub.add_parser("phase-gate", help="Check phase success gate")
    add_common(pg)
    pg.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                    help="Phase number (1-4)")

    # phase-diagnostics
    pd_cmd = sub.add_parser("phase-diagnostics", help="Print phase diagnostics")
    pd_cmd.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Phase number (1-4)")

    # historical-validate
    hv = sub.add_parser("historical-validate", help="Historical regime validation")
    add_common(hv)
    hv.add_argument("--mutations-json", default=None,
                    help="Path to mutations JSON (phase_state.json or greedy_optimal.json)")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "walk-forward":
        cmd_walk_forward(args)
    elif args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    elif args.command == "historical-validate":
        cmd_historical_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
