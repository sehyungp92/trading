# Swing Trader

Multi-strategy IBKR swing trading system. Python 3.12, async, Pydantic models.

## Shell Commands
- Always use `mkdir -p` instead of `mkdir` when creating directories
- Never use bare `mkdir` — it errors if the directory already exists

## Project Layout
- `strategy/` — Strategy 1 (ATRSS): ATR-based swing strategy
- `strategy_2/` — Strategy 2 (AKC-Helix): mean-reversion helix
- `strategy_3/` — Strategy 3 (Breakout): breakout entries
- `strategy_4/` — Strategy 4: additional strategy
- `shared/` — Shared infra: `ibkr_core/` (IBKR client), `oms/` (order management), `overlay/` (overlay engine)
- `instrumentation/` — Trade telemetry: `src/kit.py` (facade), trade logger, drawdown/gap/session trackers
- `backtest/` — Backtesting framework
- `relay/` — Signal relay
- `infra/` — Docker, systemd, DB init, deployment
- `config/` — YAML configs (IBKR profiles, contracts, routing)
- `main_multi.py` — Unified launcher: runs all strategies in one process with shared IBKR session and OMS

## Each Strategy Module
Contains `engine.py` (async strategy loop), `models.py` (Pydantic state models), and `tests/`.

## Key Conventions
- Async-first: engines are `async def` loops driven by IBKR market data
- Pydantic v2 models for all state and config
- `from __future__ import annotations` in every file
- InstrumentationKit (`instrumentation/src/kit.py`) wraps all telemetry — never crashes trading
- All IBKR interaction goes through `shared/ibkr_core/`; order flow through `shared/oms/`

## Testing
- pytest, run from repo root: `python -m pytest`
- Strategy tests: `strategy*/tests/`
- Instrumentation tests: `instrumentation/tests/`
- Integration tests: `tests/`

## Dependencies
numpy, pandas, ib_async, asyncpg, pydantic, pyyaml (see `requirements.txt`)
