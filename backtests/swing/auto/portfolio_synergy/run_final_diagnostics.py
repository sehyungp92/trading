from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plugin import PortfolioSynergyPlugin


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate final swing portfolio synergy diagnostics from an optimized config.",
    )
    parser.add_argument("--run-dir", default="backtests/output/swing/portfolio_synergy/round_2")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--equity", type=float, default=25_000.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "round_final_diagnostics.txt"
    mutations = json.loads((run_dir / "optimized_config.json").read_text(encoding="utf-8"))

    plugin = PortfolioSynergyPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=float(args.equity),
        max_workers=int(args.max_workers),
    )
    metrics = plugin.compute_final_metrics(mutations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        plugin._format_diagnostics("FINAL PORTFOLIO SYNERGY DIAGNOSTICS", metrics, None),
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
