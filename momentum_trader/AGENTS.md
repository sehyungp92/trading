## Project Overview

Multi-strategy NQ/MNQ momentum trading system. Three live strategies + shared OMS + instrumentation + backtesting. Python 3.12, asyncio, Interactive Brokers.

## Structure

- `strategy/` — Helix v4.0 (trend-following)
- `strategy_2/` — NQDTC v2.1 (compression→displacement breakout)
- `strategy_3/` — Vdubus v4.2 (VWAP pullback)
- `shared/oms/` — Order management, risk gateway, portfolio rules
- `shared/ibkr_core/` — IB Gateway adapter
- `backtest/` — CLI backtesting framework (sync, bar-by-bar)
- `instrumentation/` — Event logging, trade/missed-opportunity tracking, sidecar relay
- `infra/` — Docker compose, deployment docs
- `config/` — IBKR profiles, contracts, routing

## Commands

```bash
# Backtesting
python -m backtest run --strategy helix --diagnostics
python -m backtest run --strategy nqdtc
python -m backtest run --strategy vdubus --diagnostics

# Tests
pytest instrumentation/tests/ -v

# Live (inside Docker containers)
python -m strategy       # Helix
python -m strategy_2     # NQDTC
python -m strategy_3     # Vdubus
```

## Shell Commands

- Always use `mkdir -p` instead of `mkdir` when creating directories
- Never use bare `mkdir` — it errors if the directory already exists

## Key Conventions

- **Config is code**: Strategy constants in `strategy*/config.py` (frozen). Ablation flags in `backtest/config_*.py`. Portfolio presets in `shared/oms/config/portfolio_config.py` (active: `make_10k_v6_config()`).
- **Backtest ≠ live sizing**: `fixed_qty=10` in backtest configs bypasses all sizing multipliers. Changes to `SIZE_MULT_M`, `SESSION_SIZE_MULT` only affect live.
- **Live = async, backtest = sync**: Live engines use `async def`/`await` with `ib_async`. Backtest engines are synchronous bar-by-bar loops. Shared logic lives in pure functions (`signals.py`, `gates.py`, `risk.py`).
- **Intent flow**: `Intent` → `IntentHandler` → `RiskGateway` (heat/daily/weekly stops) → `ExecutionAdapter`
- **Instrumentation is non-fatal**: All instrumentation calls wrapped in try/except. Facade pattern via `InstrumentationKit`. Graceful degradation — never block trading.
- **All new schema fields must be `Optional[...] = None`** for backward-compatible JSONL.
- **Never trust shadow sim for shorts** — shadow sim overestimates short-side performance.
- **Additive testing only**: Test changes against clean baseline. Subtractive ablation misleads.
