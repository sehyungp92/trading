# Trading Monorepo

Systematic trading system running **11 strategies** across **3 families** — swing (ETFs), momentum (NQ futures), and stock (US equities) — on a single-VPS deployment with unified OMS, risk gateway, and instrumentation pipeline.

---

## Strategy Families

### Swing Family (4 strategies + overlay)

Trades ETFs (QQQ, GLD) on hourly/daily timeframes. All strategies share a single OMS instance with coordinated position management.

| Strategy | Edge |
|----------|------|
| **AKC Helix** | Trend-following with regime filtering across two ETFs. EMA-based trend alignment with ATR stops. |
| **ATRSS** | ATR-calibrated multi-leg breakout/reentry with chandelier trailing and stall detection. Quality gate filters low-conviction setups. |
| **Swing Breakout** | Campaign-based entries scored by AVWAP + multi-timeframe EMA alignment + displacement quantile. Only high-momentum, low-chop breakouts are taken. |
| **BRS** | Regime-gated multi-arm entry with 4H structural classification and daily bias alignment. Pyramiding with signal-confirmed adds. |
| **Overlay** | Deploys idle cash via daily EMA crossover rebalancing. Zero additional margin cost — uses capital not committed to swing risk. |

### Momentum Family (4 strategies)

Trades MNQ (Micro E-mini Nasdaq) futures on intraday timeframes (5m/15m/1H). Each strategy runs its own OMS instance.

| Strategy | Edge |
|----------|------|
| **NQDTC** | Compression-breakout on 5m bars with precise stop placement at box boundary. Chop detection (ADX) blocks ranging conditions. Ablation-optimized partials. |
| **VdubusNQ** | VWAP as dynamic support/resistance with quantile-based displacement filtering. Session window restrictions concentrate entries in highest-probability time blocks. |
| **AKC Helix** | Class-based signal hierarchy (Momentum > Fade > Trend) with session-aware sizing and DOW filtering. Apex chandelier trail captures extended moves. |
| **Downturn Dominator** | Short-only specialist with 5 bear-regime override modes. Conviction scoring + fast-crash detection increase aggression during genuine downturns. VWAP-failure exit cuts losers early. |

### Stock Family (3 strategies)

Trades US equities (S&P 500 universe + scanner-driven). Each strategy runs its own OMS instance.

| Strategy | Instruments | Edge |
|----------|-------------|------|
| **IARIC** | S&P 500 watchlist | Nightly pullback selection ranked by 7-trigger score with 5 intraday entry routes (opening reclaim, VWAP bounce, afternoon retest, etc.). AVWAP provides objective entry-level validation. |
| **ALCB** | S&P 500 candidates | Opening range momentum continuation with ADX + RVOL confirmation. Sector-based size penalty prevents concentration. Flow reversal trailing stop. |
| **US ORB** | IB scanner (gap + volume) | Pure first-hour momentum capture. Scanner dynamically selects the day's strongest movers; breadth filter prevents entries during weak tape. |

---

## Regime System

A macro regime classifier runs weekly and governs risk budgets, position limits, and allocation weights across all three families.

### Classification

A Gaussian HMM fitted on 6-9 z-scored macro/market features (growth, inflation, yield curve slope, credit spreads, equity-bond correlation, momentum breadth, VIX, commodities) classifies the environment into four states:

| Regime | Label | Growth | Inflation | Posture |
|--------|-------|--------|-----------|---------|
| **G** | Recovery | Rising | Falling | Full risk, long-biased |
| **R** | Reflation | Rising | Rising | Moderate risk, balanced |
| **S** | Inflationary Hedge | Falling | Rising | Reduced risk, short-tolerant |
| **D** | Defensive | Falling | Falling | Minimum risk, capital preservation |

### Downstream Impact

Each coordinator receives a `RegimeContext` and atomically updates its live `PortfolioRulesConfig`:

- **Swing** — Directional cap and unit risk scale smoothly across regimes. Overlay idle-capital allocation shifts from QQQ-heavy to GLD-only as conditions deteriorate.

  | Regime | `directional_cap_R` | `regime_unit_risk_mult` | Overlay QQQ/GLD |
  |:------:|:-------------------:|:----------------------:|:---------------:|
  | G | 6.0 | 1.0x | 60 / 40 |
  | R | 5.0 | 0.9x | 50 / 50 |
  | S | 4.0 | 0.8x | 20 / 80 |
  | D | 3.0 | 0.6x | 0 / 100 |

- **Momentum** — Asymmetric long/short caps rebalance exposure: long-only in Growth, balanced long/short in Defensive. NQDTC opposing-direction multiplier (`nqdtc_oppose_size_mult`) activates at 0.5x in S/D regimes (blocked in G/R). Contract capacity scales with `max_contracts_scale`. DownturnDominator is disabled in G/R (trend-following only), enabled in S/D.

  | Regime | `cap_long_R` | `cap_short_R` | `unit_risk_mult` | `contracts_scale` | `oppose_mult` |
  |:------:|:------------:|:-------------:|:----------------:|:-----------------:|:-------------:|
  | G | 3.5 | 0.0 | 1.0x | 1.0x | 0.0 (block) |
  | R | 3.0 | 1.0 | 0.9x | 0.9x | 0.0 (block) |
  | S | 2.0 | 2.0 | 0.7x | 0.7x | 0.5x |
  | D | 1.5 | 2.5 | 0.5x | 0.5x | 0.5x |

- **Stock** — Directional cap tightens, symbol collision action escalates from `half_size` (allow at reduced size) to `block` (deny entry) in S/D. Priority headroom for IARIC+ALCB narrows. US_ORB disabled in S/D regimes. Per-strategy max positions reduce progressively.

  | Regime | `directional_cap_R` | `unit_risk_mult` | `collision` | `headroom_R` | ALCB max pos | IARIC max pos | US_ORB |
  |:------:|:-------------------:|:----------------:|:-----------:|:------------:|:------------:|:-------------:|:------:|
  | G | 8.0 | 1.0x | half_size | 3.0 | 8 | 8 | on |
  | R | 6.0 | 0.9x | half_size | 3.0 | 6 | 5 | on |
  | S | 5.0 | 0.7x | block | 2.0 | 4 | 3 | off |
  | D | 4.0 | 0.5x | block | 1.5 | 3 | 2 | off |

A **drawdown ladder** further tightens sizing within each regime. Thresholds compress in S/D to trigger earlier:

| Tier | G/R threshold | S threshold | D threshold | Size multiplier |
|:----:|:------------:|:-----------:|:-----------:|:---------------:|
| Full | < 8% | < 7% | < 6% | 1.00x |
| Half | 8-12% | 7-11% | 6-10% | 0.50x |
| Quarter | 12-15% | 11-14% | 10-13% | 0.25x |
| Halt | > 15% | > 14% | > 13% | 0.00x |

---

## Phased Greedy Optimisation Framework

All strategies are optimised through a **phased greedy optimisation** framework (`backtests/shared/auto/`), directly leveraging ideas from Karpathy's AutoResearch framework.

### Core Mechanics

1. **Phased search** — Optimisation is split into focused phases (e.g., exits, signal discrimination, timing, fine-tuning). Each phase explores a curated domain rather than the full space.
2. **Greedy forward selection** — Candidates are evaluated sequentially against a cumulative baseline. Accepted if the composite score improves; mutations accumulate across phases.
3. **Composite scoring** — Immutable weighted sum of 7-8 normalised components (profit factor, Calmar, Sortino, capture efficiency, drawdown, trade frequency, etc.) with hard-reject thresholds that short-circuit degenerate configs.
4. **Gate + retry** — Each phase has pass/fail criteria. Failures trigger automatic retries with adjusted scoring weights or enhanced diagnostics (up to 2 retries per phase).
5. **Ablation verification** — A final flat greedy pass tests removal of each accepted mutation to confirm genuine additivity and catch interaction effects.

### Adaptive Analysis Loop

After each phase completes, a full diagnostic pass runs on the current optimal config. The analysis evaluates:

- **Progress toward goal** — Whether the phase moved metrics in the intended direction and by how much.
- **Diagnostic gaps** — Whether additional diagnostics would expose hidden weaknesses or confirm emerging strengths, and adds them for the next run if so.
- **Experiment generation** — Whether the results suggest new candidate mutations worth exploring in subsequent phases.

Based on the diagnostic output, the framework decides to:

1. **Re-score and rerun** — If the scoring function is not driving the desired behaviour, redesign an optimised immutable score and rerun the phase.
2. **Enhance diagnostics and rerun** — Add new diagnostic dimensions and re-evaluate the phase's optimal configs under richer analysis.
3. **Seed next phase** — Feed discovered experiments forward as candidates for the next greedy phase.

After all phases complete, a comprehensive diagnostic runs on the final optimised config.

### Infrastructure

State is JSON-persisted so interrupted runs resume from the last checkpoint. Each strategy implements a `StrategyPlugin` interface wiring its own candidates, gates, scoring, and backtest worker into the shared `PhaseRunner`.

---

## Evidence Pipeline

Each strategy engine emits structured events via an `InstrumentationKit` facade during live trading. A per-family sidecar forwards these events (HMAC-signed, priority-queued) to a central relay, which persists them as raw JSONL partitioned by date and bot.

### What Gets Captured

| Layer | Events | Examples |
|-------|--------|----------|
| **Trade lifecycle** | `trade_entry`, `trade_exit`, `post_exit` | Fill prices, R-multiple, MFE/MAE excursion, hold time, exit reason |
| **Decision context** | `filter_decision`, `missed_opportunity`, `indicator_snapshot` | Why a signal was taken/rejected, indicator values at decision time, scored but skipped setups |
| **Execution quality** | `order`, `orderbook_context`, `stop_adjustment` | Order lifecycle (submit/fill/cancel), book depth at entry, every stop move with trigger reason |
| **Risk & coordination** | `portfolio_rule_check`, `coordinator_action`, `parameter_change` | Rule block/pass with details, regime-driven config changes, live parameter mutations |
| **Health** | `daily_snapshot`, `market_snapshot`, `process_quality`, `heartbeat` | Equity curve, per-strategy NAV, process scoring, liveness |

### Data Reduction

A `DailyMetricsBuilder` reduces raw events into **28 curated JSON files** per bot per day (e.g., `exit_efficiency.json`, `factor_attribution.json`, `slippage_stats.json`, `stop_adjustment_analysis.json`). Portfolio-level files (`rule_blocks_summary.json`, `sector_exposure.json`, `macro_regime_analysis.json`) are computed across families.

### Trading Assistant Consumption

The trading assistant loads curated files via a `PromptAssembler` and runs daily/weekly analysis cycles:

1. **Daily triage** — Surfaces the highest-impact anomalies (outsized losses, filter failures, slippage spikes, regime mismatches) and produces root-cause analysis grounded in the actual decision context.
2. **Weekly review** — Aggregates cross-day patterns (exit efficiency trends, signal health decay, parameter drift) and proposes concrete, evidence-backed improvements: parameter mutations, filter adjustments, sizing changes, or structural experiments.
3. **Outcome measurement** — Tracks whether accepted suggestions improved performance, building a feedback loop that refines future proposals.

The pipeline ensures every suggestion is traceable to specific trades, fills, and market conditions rather than generic heuristics.

---

## Project Structure

```
apps/
  runtime/        # RuntimeShell — loads all families via --family flag
  relay/          # Webhook event ingestion (HMAC auth, rate limiting)
  dashboard/      # Next.js portfolio viewer
strategies/
  swing/          # 4 strategies + overlay + coordinator + instrumentation
  momentum/       # 4 strategies + coordinator + instrumentation
  stock/          # 3 strategies + coordinator + instrumentation
libs/
  oms/            # Unified OMS (models, engine, risk gateway, persistence)
  broker_ibkr/    # IBKR adapter layer
  services/       # Bootstrap, heartbeat, trade recorder
  config/         # YAML loader with env-var substitution
backtests/
  shared/auto/    # PhaseRunner, greedy optimizer, plugin protocol
  swing/          # Swing backtest engines + auto-optimisation
  momentum/       # Momentum backtest engines + auto-optimisation
  stock/          # Stock backtest engines + auto-optimisation
config/
  strategies.yaml # All 13 strategies with symbols, risk params, flags
infra/            # Docker, migrations, cron, deployment
```
