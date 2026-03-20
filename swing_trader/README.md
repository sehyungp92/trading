# Swing Trader

Multi-strategy IBKR swing trading system. Runs 5 strategy instances across 4 strategy modules in a single process with a shared Interactive Brokers session, order management system, and cross-strategy coordination.

## Strategies

Strategies are prioritised and coordinated through a shared OMS. Cross-strategy rules include Helix stop tightening on ATRSS entries and position size boosts when strategies agree on direction.

### Strategy 1 — ATRSS (ATR Swing Strategy)

**Edge:** Trend-following pullback-to-value. In confirmed trending markets (ADX + dual EMA regime filter), price tends to continue in the trend direction after retracing to a dynamic support level. The strategy enters on the recovery rather than chasing breakouts, capturing favorable risk:reward at the pullback point.

**Signals:**
- **Pullback-to-value:** price touches the "pull" EMA (adaptive period: 34 bars in strong trends, 50 otherwise) and recovers. Bread-and-butter entry.
- **Breakout-arm-then-pullback:** in strong trends only, a Donchian channel breakout arms a 24-hour window, then enters on a 30-50% retracement of the breakout bar.
- **Stop-and-reverse:** on a daily trend direction flip, flattens the existing position and re-enters in the new direction.

All entries pass a quality gate (ADX, DI alignment, EMA separation, touch distance, momentum — minimum 4/7 score) and are submitted as stop-limit orders expiring after 18 hours.

**Exits:** Layered — catastrophic cap at -2R, partial profit-taking at 1R and 2R (33% each), stall exits after 36 bars, time decay after 480 bars, and a Chandelier trailing stop that activates at 1.25R MFE with regime-adaptive multipliers.

**Instruments:** QQQ, GLD (ETF mode, default); also supports micro and full-size futures (MNQ/NQ, MCL/CL, MGC/GC, MBT/BRR).

**Priority:** 0 (highest)

---

### Strategy 2 — AKC-Helix (MACD Divergence + Break of Structure)

**Edge:** Multi-class MACD divergence system anchored to structural pivots. Hidden divergence (price making higher lows while MACD makes lower lows) signals latent trend strength after a pullback. Classic divergence signals exhaustion and reversal. Each setup requires a break-of-structure (BoS) between two pivots as the entry trigger, adding directional confirmation before committing capital.

**Signal classes (by priority):**
- **Class A — 4H hidden divergence continuation:** highest priority. Two 4H pivots with hidden divergence, BoS trigger between them. No regime gate — takes setups in any regime but sizes down in chop/counter-trend.
- **Class C — 4H classic divergence reversal:** fires when price is extended (>1.5 ATR from fast EMA) or in chop regime. Catches exhaustion reversals.
- **Class B — 1H hidden divergence continuation:** same logic on the hourly timeframe, but gated by ADX >= 20 and must be trend-aligned.
- **Class D — 1H momentum continuation:** pure MACD momentum (no divergence required), trend-only, lowest priority.

**Exits:** Break-even at +1R (4H) or +0.75R (1H), 50% partial at +2.5R, 25% partial at +5R. Chandelier trailing stop with adaptive multiplier (2.0-4.0x ATR) that tightens on regime deterioration, stalled MFE, or declining MACD histogram. Immediate flatten on regime flip against position.

**Instruments:** QQQ, USO, GLD, IBIT (ETF mode); also supports micro/full futures.

**Priority:** 4 (lowest)

---

### Strategy 3 — Breakout (Post-Compression Breakout)

**Edge:** Exploits post-compression breakout momentum — the tendency for assets consolidating in a tight range to exhibit a strong directional move once they escape. Combines two conditions: (1) a volatility squeeze (rolling range compressed relative to long-run ATR) and (2) a displacement pass (close far from anchored VWAP, confirmed by historical quantile). The strategy waits for compression to build, then enters on the breakout with structural confirmation.

**Entry flow:**
- **Daily layer:** detects compression (70%+ containment, squeeze metric below adaptive ceiling), confirms structural breakout above/below box, checks displacement vs AVWAP, rejects anomalous spike bars, and computes a multi-factor evidence score (volume, squeeze quality, regime, consecutive closes, ATR expansion).
- **Hourly layer (3 entry types):** AVWAP retest + reclaim (highest priority), sweep + reclaim (price sweeps below AVWAP then recovers), or 2-bar hold above AVWAP (limit order).

**Exits:** Regime-dependent TP targets (TP1 at 0.10-0.20R, TP2 at 0.25-0.50R depending on regime alignment), then a 4H ATR trailing stop on the runner. Stale exit after 10 days underwater. Gap-through-stop protection at the open.

**Campaign state machine:** INACTIVE -> COMPRESSION -> BREAKOUT -> POSITION_OPEN -> CONTINUATION -> DIRTY/EXPIRED/INVALIDATED. Manages re-entry after failed breakouts and continuation entries after measured moves.

**Instruments:** QQQ, GLD (active); USO, IBIT (defined but inactive).

**Priority:** 3

---

### Strategy 4 — Keltner Momentum Breakout (KMB)

**Edge:** Momentum and mean-reversion on liquid ETFs using Keltner Channels as a volatility-normalised envelope. Price escaping the ATR-scaled band signals sustained momentum; midline pullback recovery signals continuation. RSI and rate-of-change filters confirm genuine momentum rather than noise.

**Entry modes (configured per instance):**
- **Breakout:** close beyond Keltner upper/lower band + RSI + ROC alignment.
- **Pullback:** price crosses back above the Keltner midline (mean-reversion continuation).
- **Dual:** tries breakout first, then pullback. Maximum trade frequency.

All entries require volume above its 20-period SMA. Daily timeframe only — fires at 16:15 ET after market close.

**Exits:** ATR trailing stop (1.5x ATR) activates after 1R of profit. Optional midline or reversal exit modes.

**Runs as two instances:**
- **S5_PB** (priority 1): IBIT only, pullback mode, EMA(10), 1.5x ATR stop.
- **S5_DUAL** (priority 2): GLD + IBIT, dual mode, longs only, EMA(15), 2.0x ATR stop.

---

## Portfolio & Overlay

### Cross-Strategy Coordination

The shared OMS and `StrategyCoordinator` enforce portfolio-level rules:
- **Priority-based allocation:** ATRSS (0) > S5_PB (1) > S5_DUAL (2) > Breakout (3) > Helix (4). Higher priority strategies get first claim on capital and position slots.
- **Cross-strategy signals:** ATRSS entries tighten Helix stops. When strategies agree on direction for the same symbol, Helix boosts size by 25%.
- **Portfolio heat caps:** total open risk is capped at the portfolio level (3-6% depending on strategy), with per-instrument limits.
- **Correlated position penalties:** correlated same-direction positions (e.g. QQQ + IBIT) incur sizing penalties.

### Overlay Engine

The overlay is an **idle-capital deployer**, not a regime filter. It puts equity to work in ETFs when alpha strategies aren't using it.

**Mechanism:** Daily EMA crossover allocation. For each overlay symbol (QQQ, GLD), if the fast EMA is above the slow EMA, allocate capital; otherwise, go flat. Capital is split equally among bullish symbols, capped at 85% of NAV.

**EMA periods:** QQQ uses 10/21, GLD uses 13/21.

**Independence:** The overlay bypasses the OMS entirely and places orders directly via the IB API. Its EMA signals are exposed to the instrumentation layer for regime tagging on trade logs, but do not gate or modify alpha strategy entries.

---

## Project Structure

```
main_multi.py              Unified launcher — all strategies, one process
strategy/                  Strategy 1 (ATRSS)
strategy_2/                Strategy 2 (Helix)
strategy_3/                Strategy 3 (Breakout)
strategy_4/                Strategy 4 (KMB — runs as S5_PB and S5_DUAL)
shared/
  ibkr_core/               IBKR client, contract mapping, execution adapter
  oms/                     Order management system with risk calculator
  overlay/                 Overlay engine (idle-capital EMA crossover)
  market_calendar.py       Market hours / holidays
config/                    YAML configs (IBKR profiles, contracts, routing)
instrumentation/           Trade telemetry and analytics
  src/kit.py               InstrumentationKit facade
  src/trade_logger.py      Trade event logging (JSONL)
  src/sidecar.py           Relay forwarding with HMAC auth
  src/missed_opportunity.py  Counterfactual analysis
  src/post_exit_tracker.py   Post-exit price tracking
relay/                     Signal relay server (FastAPI)
backtest/                  Backtesting framework
infra/                     Docker, systemd, DB init, deployment
tests/                     Integration tests
```

## Requirements

- Python 3.12+
- Interactive Brokers TWS or IB Gateway
- PostgreSQL (for state persistence)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Configure IBKR connection and contracts in `config/`.

## Running

```bash
python main_multi.py
```

This starts all strategies with a shared IBKR session and OMS. Strategies run as async loops driven by IBKR market data.

## Testing

```bash
python -m pytest
```

## Key Design Decisions

- **Async-first**: all strategy engines are `async def` loops
- **Pydantic v2**: all state and config models
- **Single process**: shared IBKR connection avoids rate limits and simplifies coordination
- **InstrumentationKit**: wraps all telemetry — designed to never crash trading
- **Relay architecture**: sidecar forwards trade events to a central relay for analysis
