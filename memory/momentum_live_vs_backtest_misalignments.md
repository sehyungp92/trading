# Momentum: Live vs Backtest Portfolio Rules Comparison

Generated 2026-03-25. Sources:
- **Live**: `strategies/momentum/coordinator.py`, `libs/oms/risk/portfolio_rules.py`, strategy `config.py` files
- **Backtest**: `research/backtests/momentum/engine/portfolio_engine.py`, `research/backtests/momentum/config_portfolio.py`
- **v7 Optimal**: `libs/oms/config/portfolio_config.py` `make_10k_v7_config()`, `research/backtests/momentum/auto/output/greedy_portfolio_optimal.json`

---

## Master Comparison Table

| # | Parameter | Live (coordinator + portfolio_rules.py) | Backtest (portfolio_engine + config_portfolio) | v7 Optimal (make_10k_v7_config) | Match? |
|---|---|---|---|---|---|
| 1 | **DD tiers** | `(0.08,1.0), (0.12,0.5), (0.15,0.25), (1.0,0.0)` — default in `PortfolioRulesConfig` :54-59 | `(0.08,1.0), (0.12,0.5), (0.15,0.25), (1.0,0.0)` — from `PortfolioConfig` :49-54 | Same | MATCH |
| 2 | **DD calculation basis** | `(initial_equity - current) / initial_equity` — `portfolio_rules.py:131-136` | `(peak_equity - current) / peak_equity` — `portfolio_engine.py:~555-559` | N/A | **MISMATCH** |
| 3 | **Directional cap** | `6.0R` — hardcoded in `coordinator.py:143` | `3.5R` — from `make_10k_v7_config()` via PortfolioConfig :655 | `3.5R` | **MISMATCH** |
| 4 | **Portfolio daily stop** | `1.5R` (Helix), `1.5R` (NQDTC, Vdub — from `coordinator.py:154,309,325`) | `1.5R` — from `make_10k_v7_config()` :653 | `1.5R` | MATCH |
| 5 | **Per-strategy daily stop: Helix** | `1.5R` — `helix_v40/config.py:33` (DAILY_STOP_R), passed via coordinator:267 | `2.0R` — `make_10k_v7_config()` Helix alloc :683 `daily_stop_R=2.0` | `2.0R` | **MISMATCH** |
| 6 | **Per-strategy daily stop: NQDTC** | `2.5R` — hardcoded `coordinator.py:307` | `2.5R` — `make_10k_v7_config()` NQDTC alloc :670 | `2.5R` | MATCH |
| 7 | **Per-strategy daily stop: Vdub** | `2.5R` — hardcoded `coordinator.py:323` | `2.5R` — `make_10k_v7_config()` Vdubus alloc :661 | `2.5R` | MATCH |
| 8 | **Heat cap (portfolio)** | `3.5R` — from `PortfolioConfig` (used by backtest); live uses per-strategy `heat_cap_R` passed to OMS | `3.5R` — `make_10k_v7_config()` :652 | `3.5R` | See note below |
| 9 | **Heat cap: Helix** | `3.0R` — `helix_v40/config.py:34`, passed via coordinator:267 | N/A (portfolio-level 3.5R only) | N/A | STRUCTURAL DIFF |
| 10 | **Heat cap: NQDTC** | `3.5R` — hardcoded `coordinator.py:308` | N/A (portfolio-level 3.5R only) | N/A | STRUCTURAL DIFF |
| 11 | **Heat cap: Vdub** | `3.5R` — hardcoded `coordinator.py:324` | N/A (portfolio-level 3.5R only) | N/A | STRUCTURAL DIFF |
| 12 | **nqdtc_direction_filter** | `enabled=False` — `coordinator.py:146` | `enabled=False` — `make_10k_v7_config()` :688 | `False` | MATCH |
| 13 | **nqdtc_agree_size_mult** | `1.50` — default in `PortfolioRulesConfig:40` (moot since filter disabled) | `1.50` — `make_10k_v7_config()` :689 | `1.50` | MATCH (moot) |
| 14 | **nqdtc_oppose_size_mult** | `0.0` — default in `PortfolioRulesConfig:41` (moot) | `0.0` — `make_10k_v7_config()` :690 | `0.0` | MATCH (moot) |
| 15 | **helix_nqdtc_cooldown** | `120 min` — default in `PortfolioRulesConfig:33` | `120 min` — `make_10k_v7_config()` :686 | `120 min` | MATCH |
| 16 | **cooldown_session_only** | `True` — default in `PortfolioRulesConfig:34` | `True` — `make_10k_v7_config()` :687 (inherited from v6 base) | `True` | MATCH |
| 17 | **helix_veto** | **NOT IMPLEMENTED** in `portfolio_rules.py` — no field, no check | **Implemented** in `portfolio_engine.py:~500-507` — `helix_veto_enabled=True`, `window=120min` | `helix_veto_enabled=True`, `window=120min` | **MISMATCH** |
| 18 | **Symbol collision** | `half_size` — `coordinator.py:145` | Not applicable (momentum trades NQ only) | N/A | N/A |
| 19 | **family_strategy_ids** | All 3 IDs — `coordinator.py:109,144` | N/A (backtest uses PortfolioConfig directly) | N/A | STRUCTURAL DIFF |
| 20 | **Base risk: Helix** | `0.02` (2.0%) — `helix_v40/config.py:32` | `0.02` (2.0%) — `make_10k_v7_config()` :680 | `0.02` | MATCH |
| 21 | **Base risk: NQDTC** | `0.008` (0.8%) — `nqdtc/config.py:52` | `0.008` (0.8%) — `make_10k_v7_config()` :668 | `0.008` | MATCH |
| 22 | **Base risk: Vdub** | `0.01` (1.0%) — `vdub/config.py:49` | `0.01` (1.0%) — `make_10k_v7_config()` :660 | `0.01` | MATCH |
| 23 | **Weekly portfolio stop** | **NOT IMPLEMENTED** in `portfolio_rules.py` | `12.0R` — `PortfolioBacktestConfig.portfolio_weekly_stop_R` default | N/A | **MISMATCH** |
| 24 | **Monthly portfolio stop** | NOT IMPLEMENTED | NOT IMPLEMENTED | N/A | MATCH (both absent) |
| 25 | **Max total positions** | NOT in `PortfolioRulesConfig` — each OMS checks independently | `3` — `make_10k_v7_config()` :654 | `3` | STRUCTURAL DIFF |
| 26 | **Max concurrent per strategy** | Each OMS checks its own `MAX_CONCURRENT_POSITIONS` | Per-alloc: Helix=2, NQDTC=1, Vdub=1 | Helix=2, NQDTC=1, Vdub=1 | See note |
| 27 | **Continuation sizing** | NOT in `PortfolioRulesConfig` — no NQDTC continuation check | NQDTC: `continuation_half_size=True`, `continuation_size_mult=0.70` | Same | **MISMATCH** |
| 28 | **Reversal only** | NOT in `PortfolioRulesConfig` | NQDTC: `reversal_only=False` (v7) | `False` | N/A |
| 29 | **NQDTC chop throttle** | `enabled=False` — `PortfolioRulesConfig:48` | NOT in backtest engine | N/A | MATCH (both disabled) |
| 30 | **Trade re-sizing** | Live OMS uses `RiskCalculator.compute_unit_risk_dollars()` per strategy | Backtest: `risk_per_trade = equity * base_risk_pct`, then `raw_qty = risk / (stop_dist * point_value)` | N/A | STRUCTURAL DIFF |
| 31 | **Priority/ordering** | No priority — each OMS is independent, first-come-first-served | Backtest merges all trades chronologically, checks rules in order | N/A | STRUCTURAL DIFF |

---

## Critical Misalignments (Ranked by Impact)

### 1. Directional Cap: 6.0R (live) vs 3.5R (backtest/v7)
- **Live**: `coordinator.py:143` — `directional_cap_R=6.0` with comment "family-wide cap (vs 3.5 per-strategy default)"
- **Backtest/v7**: `make_10k_v7_config()` :655 — `directional_cap_R=3.5`
- **Impact**: SEVERE. Live allows nearly 2x the same-direction risk exposure. A 6R directional position is catastrophic if the market reverses. The backtest results assume 3.5R cap.
- **Fix**: Change `coordinator.py:143` from `6.0` to `3.5`.

### 2. Helix Veto: Missing in Live
- **Live**: `portfolio_rules.py` has NO `helix_veto_enabled` field or check. The `PortfolioRuleChecker.check_entry()` does not implement rule #8.
- **Backtest**: `portfolio_engine.py:~500-507` — if `alloc.helix_veto_enabled` and Helix traded within `helix_veto_window_minutes` (120 min), NQDTC entry is denied.
- **v7 Optimal**: `helix_veto_enabled=True`, `helix_veto_window_minutes=120` (round 3 of greedy, +0.52% score)
- **Impact**: HIGH. Without helix_veto, NQDTC enters when Helix just traded — exactly the scenario the optimizer found harmful.
- **Fix**: Add `helix_veto_enabled`, `helix_veto_window_minutes` fields to `PortfolioRulesConfig` and implement the check in `PortfolioRuleChecker.check_entry()`.

### 3. Drawdown Calculation: initial_equity (live) vs peak_equity (backtest)
- **Live**: `portfolio_rules.py:131-136` — `dd_pct = (initial - equity) / initial`
- **Backtest**: `portfolio_engine.py:~555-559` — `dd_pct = (peak_equity - equity) / peak_equity`
- **Impact**: MEDIUM-HIGH. After equity grows (e.g., $10K -> $12K), the live system sees 0% DD while the backtest would see DD from the $12K peak. If equity then drops to $11K, live sees 0% DD (still above initial) while backtest sees 8.3% DD and triggers half-sizing. The backtest DD tiers are more conservative in bull runs.

### 4. Helix Daily Stop: 1.5R (live) vs 2.0R (backtest/v7)
- **Live**: `helix_v40/config.py:33` — `DAILY_STOP_R = 1.5`
- **Backtest/v7**: `make_10k_v7_config()` Helix alloc :683 — `daily_stop_R=2.0`
- **Impact**: MEDIUM. Live Helix stops 0.5R earlier per day than backtest assumes. This clips winning streaks on trending days.

### 5. Weekly Portfolio Stop: Missing in Live
- **Live**: `portfolio_rules.py` has no weekly stop logic.
- **Backtest**: `config_portfolio.py` — `portfolio_weekly_stop_R=12.0`, checked at `portfolio_engine.py:~479-483`.
- **Impact**: LOW (12R weekly stop rarely triggers). But if a catastrophic week occurs, the backtest would halt while live keeps trading.

### 6. NQDTC Continuation Sizing: Missing in Live
- **Live**: `PortfolioRulesConfig` has no `continuation_half_size` or `continuation_size_mult` fields. No check in `check_entry()`.
- **Backtest**: `portfolio_engine.py:~532-538` — NQDTC continuation trades get `size_mult *= 0.70` when `continuation_half_size=True`.
- **Impact**: MEDIUM. Live NQDTC continuation trades enter at full size (1.0x) vs backtest's 0.70x. This increases risk on the lower-quality continuation signals.

### 7. Max Total Positions: Not Cross-Strategy in Live
- **Live**: Each OMS independently checks its own max concurrent. No cross-strategy total cap.
- **Backtest**: `portfolio_engine.py:~455-457` — `max_total_positions=3` checked across all 3 strategies.
- **Impact**: LOW-MEDIUM. In theory, all 3 strategies could have 1 position each (3 total, matching the cap). But Helix allows `max_concurrent=2` in v7, so live could have 4 total positions (2 Helix + 1 NQDTC + 1 Vdub).

---

## Structural Differences (By Design)

### Heat Cap Architecture
- **Live**: Per-strategy `heat_cap_R` passed to each OMS (Helix=3.0, NQDTC=3.5, Vdub=3.5). Each OMS checks its own heat independently.
- **Backtest**: Single portfolio-level `heat_cap_R=3.5` applied across all open positions.
- These are fundamentally different architectures. The live system cannot exceed 3.0R on Helix alone but could have 3.0+3.5+3.5=10R total heat across all three strategies. The backtest caps total portfolio heat at 3.5R.

### Trade Sizing
- **Live**: `RiskCalculator.compute_unit_risk_dollars(nav, unit_risk_pct)` computes 1R in dollars. OMS then computes qty from stop distance.
- **Backtest**: `risk_per_trade = equity * base_risk_pct`, then `raw_qty = risk_per_trade * size_mult / (stop_distance * point_value)`, minimum 1 contract.
- Both arrive at the same result for MNQ trades, but the live system uses per-strategy allocated_nav (from `bootstrap_capital`) while the backtest uses shared portfolio equity.

### Capital Allocation
- **Live**: `bootstrap_capital()` allocates NAV per strategy from `config/strategies.yaml`. Each strategy's unit risk is computed from its allocated portion.
- **Backtest**: All strategies share the same equity pool. Risk is computed from total portfolio equity * per-strategy base_risk_pct.

### Priority / Ordering
- **Live**: Independent OMS instances — no priority, first signal wins.
- **Backtest**: Chronological timeline with priority ordering (Vdubus=0 > NQDTC=1 > Helix=2) when trades overlap.
