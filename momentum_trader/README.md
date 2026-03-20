# momentum_trader

Multi-strategy NQ/MNQ momentum trading system. Three independent strategies share a unified order management system, portfolio-level risk controls, and instrumentation layer. Python 3.12, asyncio, Interactive Brokers.

## Strategies

### Helix v4.0 — 1H Trend-Following Pullback

**Edge**: Higher-timeframe trend alignment + structural pullback exhaustion on the 1H chart. When Daily and 4H EMAs agree on direction, Helix waits for a clean higher-low pivot structure on the 1H and enters on the breakout above the intermediate swing high. The MACD momentum reclaim confirms the pullback is exhausting — not reversing — before entry. Trailing exits are the core alpha: 83% WR on trailed exits with +1.333R average, capturing extended moves that trend-following is designed for. Monday and Wednesday blocks remove FOMC-correlated chop days. Short entries are heavily suppressed (NQ's structural upward bias makes shorts net-negative at lower conviction thresholds).

| Metric | Value |
|--------|-------|
| Timeframe | 1H bars |
| Signal | Class M pivot-breakout + Class T 4H continuation |
| Entry filter | Alignment score (Daily + 4H EMA), extension gate, spike filter, vol cap |
| Exits | R-based trailing (`mult = max(1.5, 3.0 - R/5)`), partials at +1R/+1.5R, MFE ratchet at 65% floor |

### NQDTC v2.1 — 30m Compression-to-Displacement Breakout

**Edge**: Volatility compression precedes expansion. NQDTC detects when NQ consolidates within an adaptive box (16–40 bars depending on ATR regime) and enters only when the breakout shows genuine displacement — measured as distance from the box-anchored VWAP relative to ATR, exceeding the 10th percentile of historical displacements. The evidence scorecard (relative volume, squeeze tightness, trend alignment, body decisiveness) filters for high-conviction breakouts. A DIRTY state machine catches false breakouts that re-enter the box, suspending entries until the box re-qualifies. The strategy is direction-neutral — it captures expansion in either direction — making it complementary to the directionally-biased Helix and Vdubus.

| Metric | Value |
|--------|-------|
| Timeframe | 30m boxes, 5m entry evaluation |
| Signal | Box breakout with displacement threshold + evidence scorecard (>1.5 score) |
| Entry filter | Regime classification (Aligned/Neutral/Caution/Counter), MACD slope, RVOL, breakout quality |
| Exits | Tiered chandelier trail, TP1 at +1.5R (25%), measured move continuation mode, ratchet floor at 25% |

### Vdubus v4.2 — 15m VWAP Pullback Reclaim

**Edge**: VWAP is the volume-weighted average price for the session — the price where institutional size has transacted. When price pulls back to VWAP and reclaims it (close above), it signals that buyers are defending their average cost. Vdubus trades this reclaim exclusively in the direction of the macro trend (ES SMA200), ensuring entries align with the dominant institutional flow. A Predator overlay adds divergence confirmation: structural higher lows with declining momentum indicate trapped shorts being unwound. The wide midday dead zone (10:45–15:00) removes the low-conviction grind period, concentrating entries in the OPEN, late-CORE, CLOSE, and EVENING windows where VWAP reclaims carry the most follow-through.

| Metric | Value |
|--------|-------|
| Timeframe | 15m bars |
| Signal | VWAP pullback + close reclaim, momentum slope confirmation, predator divergence overlay |
| Entry filter | ES SMA200 daily trend gate, 1H EMA50 alignment, choppiness index, vol state |
| Exits | Progressive MFE ratchet, VWAP failure exit (2-bar close below), early kill, 15:50 "Earn the Hold" overnight decision gate |

## Architecture

```
momentum_trader/
├── strategy/          Helix v4.0 (async live engine)
├── strategy_2/        NQDTC v2.1 (async live engine)
├── strategy_3/        Vdubus v4.2 (async live engine)
├── shared/
│   ├── oms/           Order management, risk gateway, portfolio rules
│   └── ibkr_core/     IB Gateway adapter (ib_async)
├── backtest/          CLI backtesting framework (sync, bar-by-bar)
├── instrumentation/   Event logging, sidecar relay, telemetry
├── config/            IBKR profiles, contracts, routing
└── infra/             Docker compose, deployment
```

### Order Flow

```
Strategy Engine → Intent → IntentHandler → RiskGateway → ExecutionAdapter → IB Gateway
                                              │
                                    ┌─────────┴──────────┐
                                    │  8 sequential      │
                                    │  risk checks:      │
                                    │  1. Global halt    │
                                    │  2. Event blackout │
                                    │  3. Session block  │
                                    │  4. Daily stop     │
                                    │  5. Portfolio stop │
                                    │  6. Weekly stop    │
                                    │  7. Max orders     │
                                    │  8. Heat cap       │ 
                                    │  + Portfolio rules │
                                    └────────────────────┘
```

### Portfolio-Level Risk

All three strategies route through a shared `RiskGateway` with cross-strategy coordination:

| Rule | Description |
|------|-------------|
| Proximity cooldown | Helix and NQDTC cannot both enter during 09:45–11:30 ET within 120 min of each other |
| NQDTC direction filter | If NQDTC traded today, Vdubus is blocked from opposing direction, boosted for same direction |
| Directional cap | Max 3.5R same-direction risk across all strategies |
| Drawdown tiers | Portfolio drawdown >8/12/15% triggers 50/75/100% size reduction |
| Weekly halt | Combined weekly loss exceeding 12R halts all strategies |

### Live vs Backtest

Live engines are async (`ib_async`, real-time data from IB Gateway). Backtest engines are synchronous bar-by-bar loops over historical Parquet data. Shared signal logic lives in pure functions (`signals.py`, `gates.py`, `risk.py`) used by both.

`fixed_qty=10` MNQ contracts in backtest configs bypasses all sizing multipliers — changes to `SIZE_MULT_M`, `SESSION_SIZE_MULT`, `DOW_SIZE_MULT` only affect live trading.

## Commands

```bash
# Backtesting
python -m backtest.cli run --strategy helix --diagnostics
python -m backtest.cli run --strategy nqdtc
python -m backtest.cli run --strategy vdubus --diagnostics

# Ablation testing
python -m backtest.cli ablation --strategy vdubus --filter daily_trend_gate

# Parameter optimization
python -m backtest.cli optimize --strategy nqdtc --n-coarse 500 --n-refine 200

# Walk-forward analysis
python -m backtest.cli walk-forward --strategy vdubus --test-months 6

# Tests
pytest instrumentation/tests/ -v

# Live (inside Docker containers)
python -m strategy       # Helix
python -m strategy_2     # NQDTC
python -m strategy_3     # Vdubus
```

## Instrumentation

All instrumentation is non-fatal (wrapped in `try/except`, never blocks trading). Events are logged to date-partitioned JSONL files and forwarded via a sidecar relay to a central service.

Tracked events: trade entries/exits with full signal context, missed opportunities with block reasons, filter decisions with threshold margins, order lifecycle, indicator snapshots, config parameter changes, heartbeats, and daily performance snapshots.

## Key Design Principles

- **Config is code**: Strategy constants live in `strategy*/config.py` (frozen dataclasses). Ablation flags in `backtest/config_*.py`. Portfolio presets in `shared/oms/config/portfolio_config.py`.
- **Instrumentation is non-fatal**: Facade pattern via `InstrumentationKit`. Graceful degradation — never block trading.
- **All new schema fields must be `Optional[...] = None`** for backward-compatible JSONL.
- **Additive testing only**: Test changes against clean baseline. Subtractive ablation misleads.
