"""CLI interface for the stock backtest framework.

Usage::

    python -m research.backtests.stock download --timeframes 1d
    python -m research.backtests.stock run --strategy alcb --tier 1 --start 2024-01-01
    python -m research.backtests.stock run --strategy iaric --tier 1
    python -m research.backtests.stock run --strategy alcb --tier 2
    python -m research.backtests.stock portfolio --tier 1
    python -m research.backtests.stock optimize --strategy alcb --objective sharpe
    python -m research.backtests.stock walk-forward --strategy alcb
    python -m research.backtests.stock auto --strategy all
    python -m research.backtests.stock auto --strategy alcb --skip-robustness
    python -m research.backtests.stock auto --experiments abl_alcb_stale_exit abl_alcb_regime_gate
    python -m research.backtests.stock auto --resume
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows to avoid cp949 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace) -> None:
    """Download historical bar data."""
    from research.backtests.stock.data.downloader import download_stock_universe

    timeframes = args.timeframes.split(",") if args.timeframes else ["1d"]
    output_dir = Path(args.data_dir)

    async def _run():
        return await download_stock_universe(
            timeframes=timeframes,
            duration=args.duration,
            output_dir=output_dir,
            host=args.host,
            port=args.port,
            skip_existing=not args.force,
        )

    result = asyncio.run(_run())
    for tf, symbols in result.items():
        print(f"  {tf}: {len(symbols)} symbols downloaded")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single-strategy backtest."""
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    if args.strategy == "alcb":
        _run_alcb(args, replay)
    elif args.strategy == "iaric":
        _run_iaric(args, replay)
    else:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        sys.exit(1)


def _run_alcb(args: argparse.Namespace, replay) -> None:
    from research.backtests.stock.analysis.reports import full_report
    from research.backtests.stock.config_alcb import ALCBBacktestConfig

    # Parse --param key=value overrides
    param_overrides: dict = {}
    for p in getattr(args, "param", []):
        k, _, v = p.partition("=")
        if not v:
            print(f"Invalid --param format: {p} (expected key=value)", file=sys.stderr)
            sys.exit(1)
        # Auto-cast numeric values
        try:
            v_parsed: object = int(v)
        except ValueError:
            try:
                v_parsed = float(v)
            except ValueError:
                v_parsed = v
        param_overrides[k] = v_parsed

    config = ALCBBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=args.tier,
        data_dir=Path(args.data_dir),
        verbose=args.verbose,
        param_overrides=param_overrides,
    )

    # Shadow tracker (Tier 2 only)
    shadow_tracker = None
    if getattr(args, "shadow", False) and args.tier == 2:
        from research.backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
        shadow_tracker = ALCBShadowTracker()

    if args.tier == 1:
        from research.backtests.stock.engine.alcb_daily_engine import ALCBDailyEngine

        engine = ALCBDailyEngine(config, replay)
        result = engine.run()
        report = full_report(
            result.trades, result.equity_curve, result.timestamps,
            config.initial_equity, strategy="ALCB Tier 1",
            daily_selections=result.daily_selections,
        )
        print(report)
    elif args.tier == 2:
        from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine

        engine = ALCBIntradayEngine(config, replay)
        if shadow_tracker:
            engine.shadow_tracker = shadow_tracker
        result = engine.run()
        report = full_report(
            result.trades, result.equity_curve, result.timestamps,
            config.initial_equity, strategy="ALCB Tier 2",
            daily_selections=result.daily_selections,
        )
        print(report)

    # Deep diagnostics
    if getattr(args, "diagnostics", False):
        from research.backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic

        diag = alcb_full_diagnostic(
            result.trades,
            shadow_tracker=shadow_tracker,
            daily_selections=result.daily_selections,
        )
        if args.report_file:
            Path(args.report_file).write_text(diag, encoding="utf-8")
            print(f"  Diagnostics saved to {args.report_file}")
        print(diag)

    if args.save_charts:
        _save_charts(result, f"ALCB_Tier{args.tier}", args.output_dir)


def _run_iaric(args: argparse.Namespace, replay) -> None:
    from research.backtests.stock.analysis.reports import full_report
    from research.backtests.stock.config_iaric import IARICBacktestConfig

    config = IARICBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=args.tier,
        data_dir=Path(args.data_dir),
        verbose=args.verbose,
    )

    # Shadow tracker (Tier 2 only)
    shadow_tracker = None
    if getattr(args, "shadow", False) and args.tier == 2:
        from research.backtests.stock.analysis.iaric_shadow_tracker import IARICShadowTracker
        shadow_tracker = IARICShadowTracker()

    if args.tier == 1:
        from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine

        engine = IARICDailyEngine(config, replay)
        result = engine.run()
        report = full_report(
            result.trades, result.equity_curve, result.timestamps,
            config.initial_equity, strategy="IARIC Tier 1",
            daily_selections=result.daily_selections,
        )
        print(report)
    elif args.tier == 2:
        from research.backtests.stock.engine.iaric_intraday_engine_v2 import IARICIntradayEngineV2

        engine = IARICIntradayEngineV2(config, replay, shadow_tracker=shadow_tracker)
        result = engine.run()
        report = full_report(
            result.trades, result.equity_curve, result.timestamps,
            config.initial_equity, strategy="IARIC Tier 2 v2",
            daily_selections=result.daily_selections,
        )
        print(report)

    # Deep diagnostics
    if getattr(args, "diagnostics", False):
        from research.backtests.stock.analysis.iaric_diagnostics import iaric_full_diagnostic

        diag = iaric_full_diagnostic(
            result.trades,
            fsm_log=getattr(result, 'fsm_log', None),
            rejection_log=getattr(result, 'rejection_log', None),
            shadow_tracker=shadow_tracker,
            daily_selections=result.daily_selections,
        )
        print(diag)
        if args.report_file:
            Path(args.report_file).write_text(diag, encoding="utf-8")
            print(f"  Diagnostics saved to {args.report_file}")

    if args.save_charts:
        _save_charts(result, f"IARIC_Tier{args.tier}", args.output_dir)


def _save_charts(result, prefix: str, output_dir: str) -> None:
    from research.backtests.stock.analysis.charts import (
        plot_equity_curve,
        plot_monthly_returns,
        plot_sector_attribution,
        plot_trade_distribution,
    )

    out = Path(output_dir)
    plot_equity_curve(result.equity_curve, result.timestamps, f"{prefix} Equity Curve", out / f"{prefix}_equity.png")
    plot_trade_distribution(result.trades, f"{prefix} Trade Distribution", out / f"{prefix}_distribution.png")
    plot_monthly_returns(result.trades, f"{prefix} Monthly Returns", out / f"{prefix}_monthly.png")
    plot_sector_attribution(result.trades, f"{prefix} Sector Attribution", out / f"{prefix}_sectors.png")
    print(f"  Charts saved to {out}/")


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Run portfolio backtest (both strategies)."""
    from research.backtests.stock.analysis.reports import full_report
    from research.backtests.stock.config_portfolio import PortfolioBacktestConfig
    from research.backtests.stock.engine.portfolio_engine import StockPortfolioEngine
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    pf_config = PortfolioBacktestConfig(
        data_dir=data_dir,
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=args.tier,
        verbose=args.verbose,
    )
    alcb_config, iaric_config = pf_config.build_strategy_configs()

    # Run individual strategies
    print("Running ALCB...")
    if args.tier == 1:
        from research.backtests.stock.engine.alcb_daily_engine import ALCBDailyEngine
        from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine

        alcb_result = ALCBDailyEngine(alcb_config, replay).run()
        print("Running IARIC...")
        iaric_result = IARICDailyEngine(iaric_config, replay).run()
    elif args.tier == 2:
        from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine
        from research.backtests.stock.engine.iaric_intraday_engine_v2 import IARICIntradayEngineV2

        alcb_result = ALCBIntradayEngine(alcb_config, replay).run()
        print("Running IARIC...")
        iaric_result = IARICIntradayEngineV2(iaric_config, replay).run()
    else:
        print(f"Unknown tier: {args.tier}", file=sys.stderr)
        sys.exit(1)

    # Portfolio merge
    print("Merging with portfolio rules...")
    engine = StockPortfolioEngine(pf_config)
    pf_result = engine.run(alcb_result.trades, iaric_result.trades)

    # Reports
    print("\n" + "=" * 60)
    print("ALCB individual:")
    print(full_report(
        alcb_result.trades, alcb_result.equity_curve, alcb_result.timestamps,
        pf_config.initial_equity, strategy=f"ALCB Tier {args.tier}",
    ))
    print("\n" + "=" * 60)
    print("IARIC individual:")
    print(full_report(
        iaric_result.trades, iaric_result.equity_curve, iaric_result.timestamps,
        pf_config.initial_equity, strategy=f"IARIC Tier {args.tier}",
    ))
    print("\n" + "=" * 60)
    print("Portfolio combined:")
    print(full_report(
        pf_result.trades, pf_result.equity_curve, pf_result.timestamps,
        pf_config.initial_equity, strategy="Stock Family Portfolio",
    ))
    print(f"  Blocked trades: {len(pf_result.blocked_trades)}")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run grid search optimization."""
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine
    from research.backtests.stock.optimization.param_space import ALCB_PARAM_SPACE, IARIC_PARAM_SPACE, grid_size

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    if args.strategy == "alcb":
        from research.backtests.stock.config_alcb import ALCBBacktestConfig
        from research.backtests.stock.optimization.runner import optimize_alcb

        param_space = ALCB_PARAM_SPACE
        print(f"ALCB grid search: {grid_size(param_space)} combinations")
        config = ALCBBacktestConfig(
            start_date=args.start,
            end_date=args.end,
            initial_equity=args.equity,
            data_dir=data_dir,
        )
        summary = optimize_alcb(replay, config, param_space, args.objective)
    elif args.strategy == "iaric":
        from research.backtests.stock.config_iaric import IARICBacktestConfig
        from research.backtests.stock.optimization.runner import optimize_iaric

        param_space = IARIC_PARAM_SPACE
        print(f"IARIC grid search: {grid_size(param_space)} combinations")
        config = IARICBacktestConfig(
            start_date=args.start,
            end_date=args.end,
            initial_equity=args.equity,
            data_dir=data_dir,
        )
        summary = optimize_iaric(replay, config, param_space, args.objective)
    else:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print(f"\nOptimization complete: {len(summary.results)} valid runs")
    if summary.best:
        m = summary.best.metrics
        print(f"Best params: {summary.best.params}")
        print(f"  Sharpe={m.sharpe:.2f}, CAGR={m.cagr*100:.2f}%, PF={m.profit_factor:.2f}")
        print(f"  Trades={m.total_trades}, Win%={m.win_rate*100:.1f}%, MaxDD={m.max_drawdown_pct*100:.1f}%")

    print(f"\nTop {min(10, len(summary.results))}:")
    for i, r in enumerate(summary.top_n(10)):
        print(f"  {i+1}. {r.params} → Sharpe={r.metrics.sharpe:.2f} PF={r.metrics.profit_factor:.2f}")


def cmd_auto(args: argparse.Namespace) -> None:
    """Run automated experiment harness."""
    from research.backtests.stock.auto.harness import AutoBacktestHarness

    harness = AutoBacktestHarness(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        verbose=args.verbose,
    )

    harness.run_all(
        strategy_filter=args.strategy,
        experiment_ids=args.experiments,
        skip_robustness=args.skip_robustness,
        resume=args.resume,
    )


def cmd_greedy(args: argparse.Namespace) -> None:
    """Run greedy forward selection for optimal config."""
    from research.backtests.stock.auto.greedy_optimize import (
        ALCB_T2_BASE_MUTATIONS,
        ALCB_T2_CANDIDATES,
        IARIC_T1_BASE_MUTATIONS,
        IARIC_T1_CANDIDATES,
        IARIC_T2_BASE_MUTATIONS,
        IARIC_T2_CANDIDATES,
        run_greedy,
        save_result,
    )
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    tier = getattr(args, "tier", 1)

    if args.strategy == "iaric":
        if tier == 2:
            base_mutations = IARIC_T2_BASE_MUTATIONS
            candidates = IARIC_T2_CANDIDATES
        else:
            base_mutations = IARIC_T1_BASE_MUTATIONS
            candidates = IARIC_T1_CANDIDATES
    elif args.strategy == "alcb":
        base_mutations = ALCB_T2_BASE_MUTATIONS
        candidates = ALCB_T2_CANDIDATES
        tier = 2  # ALCB momentum is always T2
    else:
        print(f"Greedy selection not yet configured for {args.strategy}", file=sys.stderr)
        sys.exit(1)

    result = run_greedy(
        replay=replay,
        strategy=args.strategy,
        tier=tier,
        base_mutations=base_mutations,
        candidates=candidates,
        initial_equity=args.equity,
        start_date=args.start,
        end_date=args.end,
        data_dir=args.data_dir,
    )

    output_path = Path(args.output_dir) / f"greedy_optimal_{args.strategy}_t{tier}.json"
    save_result(result, output_path)


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run walk-forward optimization."""
    from research.backtests.stock.engine.research_replay import ResearchReplayEngine
    from research.backtests.stock.optimization.param_space import ALCB_PARAM_SPACE, IARIC_PARAM_SPACE

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    if args.strategy == "alcb":
        from research.backtests.stock.optimization.walk_forward import walk_forward_alcb

        result = walk_forward_alcb(
            replay, ALCB_PARAM_SPACE,
            start_date=args.start, end_date=args.end,
            is_months=args.is_months, oos_months=args.oos_months,
            step_months=args.step_months,
            initial_equity=args.equity, objective=args.objective,
        )
    elif args.strategy == "iaric":
        from research.backtests.stock.optimization.walk_forward import walk_forward_iaric

        result = walk_forward_iaric(
            replay, IARIC_PARAM_SPACE,
            start_date=args.start, end_date=args.end,
            is_months=args.is_months, oos_months=args.oos_months,
            step_months=args.step_months,
            initial_equity=args.equity, objective=args.objective,
        )
    else:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print(f"\nWalk-forward complete: {len(result.windows)} windows")
    print(f"Efficiency (OOS/IS Sharpe): {result.efficiency:.2f}")
    for w in result.windows:
        is_s = w.is_metrics.sharpe if w.is_metrics else 0
        oos_s = w.oos_metrics.sharpe if w.oos_metrics else 0
        print(f"  Window {w.window_id}: IS [{w.is_start}, {w.is_end}] Sharpe={is_s:.2f}"
              f" → OOS [{w.oos_start}, {w.oos_end}] Sharpe={oos_s:.2f}")
        print(f"    Params: {w.best_params}")

    if result.combined_metrics:
        m = result.combined_metrics
        print(f"\nCombined OOS: {len(result.combined_oos_trades)} trades")
        print(f"  Sharpe={m.sharpe:.2f}, CAGR={m.cagr*100:.2f}%, PF={m.profit_factor:.2f}")
        print(f"  Win%={m.win_rate*100:.1f}%, MaxDD={m.max_drawdown_pct*100:.1f}%")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-backtest",
        description="Stock family backtesting framework (ALCB + IARIC)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download historical bar data")
    dl.add_argument("--timeframes", default="1d", help="Comma-separated timeframes (default: 1d)")
    dl.add_argument("--duration", default="5 Y", help="IBKR duration string")
    dl.add_argument("--data-dir", default="research/backtests/stock/data/raw")
    dl.add_argument("--host", default="127.0.0.1")
    dl.add_argument("--port", type=int, default=7496)
    dl.add_argument("--force", action="store_true", help="Re-download existing files")

    # run
    run = sub.add_parser("run", help="Run a backtest")
    run.add_argument("--strategy", choices=["alcb", "iaric"], required=True)
    run.add_argument("--tier", type=int, default=1, choices=[1, 2])
    run.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    run.add_argument("--end", default="2026-03-01", help="End date (YYYY-MM-DD)")
    run.add_argument("--equity", type=float, default=10_000.0)
    run.add_argument("--data-dir", default="research/backtests/stock/data/raw")
    run.add_argument("--save-charts", action="store_true", help="Save chart PNGs")
    run.add_argument("--output-dir", default="research/backtests/stock/output")
    run.add_argument("--diagnostics", action="store_true",
                     help="Run deep 27-section diagnostics report")
    run.add_argument("--shadow", action="store_true",
                     help="Enable shadow tracker for rejected setup analysis")
    run.add_argument("--report-file", type=str, default=None,
                     help="Write diagnostics report to file")
    run.add_argument("--param", action="append", default=[],
                     help="Override param: key=value (e.g. --param opening_range_bars=3)")

    # portfolio
    pf = sub.add_parser("portfolio", help="Run portfolio backtest (both strategies)")
    pf.add_argument("--tier", type=int, default=1, choices=[1, 2])
    pf.add_argument("--start", default="2024-01-01")
    pf.add_argument("--end", default="2026-03-01")
    pf.add_argument("--equity", type=float, default=10_000.0)
    pf.add_argument("--data-dir", default="research/backtests/stock/data/raw")

    # optimize
    opt = sub.add_parser("optimize", help="Run grid search optimization")
    opt.add_argument("--strategy", choices=["alcb", "iaric"], required=True)
    opt.add_argument("--objective", default="sharpe",
                     choices=["sharpe", "calmar", "profit_factor", "expectancy", "cagr", "net_profit"])
    opt.add_argument("--start", default="2024-01-01")
    opt.add_argument("--end", default="2026-03-01")
    opt.add_argument("--equity", type=float, default=10_000.0)
    opt.add_argument("--data-dir", default="research/backtests/stock/data/raw")

    # auto
    auto = sub.add_parser("auto", help="Run automated experiment harness")
    auto.add_argument("--strategy", choices=["alcb", "iaric", "all"], default="all")
    auto.add_argument("--experiments", nargs="*", help="Specific experiment IDs to run")
    auto.add_argument("--skip-robustness", action="store_true",
                       help="Skip robustness checks for fast ablation scan")
    auto.add_argument("--resume", action="store_true", help="Skip completed experiments")
    auto.add_argument("--output-dir", default="research/backtests/stock/auto/output")
    auto.add_argument("--start", default="2024-01-01")
    auto.add_argument("--end", default="2026-03-01")
    auto.add_argument("--equity", type=float, default=10_000.0)
    auto.add_argument("--data-dir", default="research/backtests/stock/data/raw")

    # greedy
    gr = sub.add_parser("greedy", help="Greedy forward selection for optimal config")
    gr.add_argument("--strategy", choices=["alcb", "iaric"], required=True)
    gr.add_argument("--tier", type=int, default=1, choices=[1, 2])
    gr.add_argument("--data-dir", default="research/backtests/stock/data/raw")
    gr.add_argument("--output-dir", default="research/backtests/stock/auto/output")
    gr.add_argument("--start", default="2024-01-01")
    gr.add_argument("--end", default="2026-03-01")
    gr.add_argument("--equity", type=float, default=10_000.0)

    # walk-forward
    wf = sub.add_parser("walk-forward", help="Run walk-forward optimization")
    wf.add_argument("--strategy", choices=["alcb", "iaric"], required=True)
    wf.add_argument("--objective", default="sharpe",
                    choices=["sharpe", "calmar", "profit_factor", "expectancy", "cagr", "net_profit"])
    wf.add_argument("--start", default="2024-01-01")
    wf.add_argument("--end", default="2026-03-01")
    wf.add_argument("--equity", type=float, default=10_000.0)
    wf.add_argument("--is-months", type=int, default=6, help="In-sample window (months)")
    wf.add_argument("--oos-months", type=int, default=3, help="Out-of-sample window (months)")
    wf.add_argument("--step-months", type=int, default=3, help="Step between windows (months)")
    wf.add_argument("--data-dir", default="research/backtests/stock/data/raw")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "download":
        cmd_download(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "portfolio":
        cmd_portfolio(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "auto":
        cmd_auto(args)
    elif args.command == "greedy":
        cmd_greedy(args)
    elif args.command == "walk-forward":
        cmd_walk_forward(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
