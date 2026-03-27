# ALCB P9 vs P10 Comprehensive Comparison

Generated: 2026-03-26

---

## D) Performance Comparison

| Metric | P9 Baseline | P9 Optimal | P10 Baseline | P10 Optimal |
|--------|-------------|------------|--------------|-------------|
| **Score** | 0.8941 | 0.9011 | 0.1697 | 0.7078 |
| **Improvement** | — | +0.79% | — | +317% |
| **Trades** | (inherited from P8) | 1,159 | (defaults) | 1,575 |
| **Profit Factor** | — | 3.30 | — | 2.08 |
| **Max Drawdown** | — | 4.55% | — | 1.47% |
| **Return** | — | 1,366% | — | 152% |
| **Rounds Tested** | 4 (3 kept) | — | 18 (17 kept) | — |
| **Candidate Pool** | 20 | — | 118 (union P1-P9) | — |

### Key Differences in Setup

- **P9**: Started from the P8 optimal base (already highly optimized with 20+ params baked in). Only 20 new candidates tested. Incremental refinement.
- **P10**: Started from **empty base** (raw StrategySettings defaults) with 2:1 intraday leverage baked in. 118 candidates (full union of all P1-P9 candidates). Complete re-optimization from scratch under a **new scoring function**.

### Why the numbers look so different

P9 and P10 use **different composite scoring functions**. P10 introduced a new scoring formula (likely with different weights for PF, DD, return, R-multiples), so the raw scores are not directly comparable. P9's 0.90 score and P10's 0.71 score do NOT mean P9 is better -- they are on different scales.

P9's 1,366% return vs P10's 152% return is also misleading: P9 uses `base_risk_fraction: 0.02` (2% risk per trade) while P10 uses the default `0.005` (0.5% risk per trade). P10 has 4x less risk per trade but achieves a **much tighter drawdown** (1.47% vs 4.55%).

**Risk-adjusted comparison**: P10 at 0.5% risk achieves PF 2.08 with 1.47% DD. If scaled to P9's 2% risk, the return would be roughly 4x higher (~608%) with proportionally higher DD (~5.9%). P9 gets 1,366% return but at 4.55% DD. P10 is arguably more capital-efficient per unit of risk.

---

## A) Common Features (Kept in Both P9 and P10)

These features survived greedy selection in both optimization runs:

| Feature | P9 Value | P10 Value | What It Does |
|---------|----------|-----------|--------------|
| `rvol_max` | 5.0 | 5.0 | **RVOL ceiling**: Reject entries where relative volume exceeds 5x average. Extremely high RVOL stocks are erratic — RVOL 3.0+ had PF=0.78 and destroyed -65R in diagnostics. |
| `regime_mult_b` | 0.0 | 0.0 | **Kill Regime B entries**: Set position sizing multiplier for Regime B (mixed/unclear market) to zero, completely blocking entries. Tier B was -74R with PF=0.69. |
| `max_positions_per_sector` | 3 | 3 | **Sector concentration cap**: Max 3 positions per sector (up from default 2). Allows more diversification within strong sectors. |
| `heat_cap_r` | 10.0 | 10.0 | **Portfolio heat ceiling**: Maximum total open risk in R-multiples raised to 10R (from default 6R). Allows larger portfolio exposure. |
| `momentum_score_min` | 2 | 2 | **Lower score threshold**: Accept entries with momentum score >= 2 (from default 3). Widens the funnel to capture more setups while relying on other filters. |
| `carry_min_cpr` | 0.4 | 0.4 | **Easier carry qualification**: Lower the minimum close-price-ratio for overnight carry to 0.4 (from 0.6). More positions qualify to hold overnight. |
| `or_quality_gate` (ablation) | Enabled | Enabled | **OR breakout quality gate**: Apply a separate minimum momentum score specifically for OR_BREAKOUT entries (distinct from the global minimum). Filters weak OR signals. |
| `fr_trailing` (concept) | 0.5/0.4 | 0.3/0.3 | **Flow reversal trailing stop**: Instead of killing positions on flow reversal, convert it to a trailing stop that activates after MFE reaches a threshold. Both keep this concept but at different tightness. |

---

## B) P9-Only Features (In P9 optimal but NOT in P10 optimal)

These features were in P9's final config but P10's greedy did not select them (or selected different values):

| Feature | P9 Value | Default | What It Does |
|---------|----------|---------|--------------|
| `base_risk_fraction` | 0.02 | 0.005 | **4x risk per trade**: Risk 2% of equity per trade instead of 0.5%. This is the single biggest driver of P9's high return AND high drawdown. P10 did NOT select this — it stayed at default 0.5%. |
| `use_partial_takes` | **False** | True | **Disable partial profit-taking**: P9 turns OFF the partial exit at TP1. All-or-nothing on each trade. P10 RE-ENABLES partials (see P10-only). |
| `use_avwap_filter` | **False** | True | **Disable AVWAP filter**: P9 removes the requirement that price be above/below session AVWAP. P10 does not set this ablation at all (uses default = enabled). |
| `opening_range_bars` | 3 | 6 | **Shorter opening range**: Use first 3 bars (15 min) instead of 6 bars (30 min) to define the opening range. Triggers earlier entries. P10 selected 12 bars instead. |
| `stop_atr_multiple` | 2.0 | 1.0 | **Wider stop**: Stop at 2x ATR below entry instead of 1x. Gives more room but more risk per share. P10 did not select this. |
| `carry_min_r` | 0.0 | 0.5 | **Zero R carry threshold**: Allow overnight carry even with 0R profit (breakeven). P10 did not override this. |
| `entry_window_end` | 11:00 | 12:00 | **Earlier entry cutoff**: Stop accepting new entries after 11:00 AM ET instead of noon. P10 did not select this. |
| `late_entry_score_min` | 5 | 0 | **Late entry quality gate**: Require momentum score >= 5 for entries after the late_entry_cutoff. P10 did not select this. |
| `use_time_based_quick_exit` | **False** | (base had True) | **Disable quick exit entirely**: P9's greedy REMOVED the P8 quick exit mechanism. The `qe_disabled` mutation was the 2nd feature selected. P10 RE-ENABLES it. |
| `use_avwap_distance_cap` | True | False | **AVWAP distance cap**: Block entries where price is >1% above AVWAP. P10 did not select this. |
| `avwap_distance_cap_pct` | 0.01 | 0.0 | Cap at 1% above AVWAP. |
| `flow_reversal_min_hold_bars` | 24 | 0 | **FR grace period**: Don't check for flow reversal until 24 bars (2 hours) have passed. P10 did not explicitly set this (uses default 0). |
| `or_breakout_score_min` | **5** | **4** | Both have OR quality gate but P9 requires score >= 5, P10 only >= 4. |
| `fr_trailing_activate_r` | **0.5** | **0.3** | P9 activates trailing at 0.5R MFE; P10 activates earlier at 0.3R. |
| `fr_trailing_distance_r` | **0.4** | **0.3** | P9 trails 0.4R behind; P10 trails tighter at 0.3R. |
| `quick_exit_max_bars` | 6 (in base, then disabled) | **12** | P9 had 6-bar QE in base but disabled it. P10 uses 12-bar QE. |
| `use_quick_exit_stage1` | True (in base) | True | P9 base had stage1 enabled but then disabled all QE. P10 keeps stage1 active. |

---

## C) P10-Only Features (In P10 optimal but NOT in P9 optimal)

These features were selected by P10's greedy but were not part of P9's final config:

| Feature | P10 Value | Default | What It Does |
|---------|-----------|---------|--------------|
| `use_time_based_quick_exit` | **True** | False | **Re-enable time-based quick exit**: If a position hasn't reached +0.2R after 12 bars (1 hour), exit it. Cuts losers early. P9 disabled this entirely. |
| `quick_exit_max_bars` | 12 | 0 | **12-bar quick exit window**: Extended from P8's 6 bars to 12. Gives trades more time to develop before cutting them. |
| `quick_exit_min_r` | 0.2 | 0.2 | Minimum R to survive the quick exit check. |
| `max_positions` | **10** | 5 | **Larger portfolio**: Allow up to 10 simultaneous positions (from default 5). P9 used 8. More diversification, more opportunity capture. |
| `use_combined_quality_gate` | True | False | **COMBINED_BREAKOUT quality gate**: Require momentum score >= 6 specifically for COMBINED_BREAKOUT entries (price breaks both OR high AND prior day high). These had a 10% lower win rate than pure OR breakouts. |
| `combined_breakout_score_min` | 6 | 0 | Score threshold for COMBINED entries. |
| `use_breakout_distance_cap` | True | False | **Breakout distance cap**: Reject entries where price has already moved >1.5R from the opening range high. Prevents chasing extended moves. |
| `breakout_distance_cap_r` | 1.5 | 0.0 | Max distance in R-multiples from OR high. |
| `use_quick_exit_stage1` | True | False | **Ultra-early quick exit (Stage 1)**: If position is below -0.5R after just 2 bars (10 minutes), exit immediately. Kills clearly failed entries very fast. |
| `qe_stage1_bars` | 2 | 0 | Stage 1 check at bar 2. |
| `qe_stage1_min_r` | -0.5 | -0.3 | Exit if below -0.5R at the stage 1 check. |
| `use_partial_takes` | **True** | True | **Re-enable partial profit-taking**: Take 33% off at 1.25R. P9 had disabled this entirely. |
| `partial_r_trigger` | 1.25 | 1.5 | Take partial profit at 1.25R instead of 1.5R (earlier lock-in). |
| `partial_fraction` | 0.33 | 0.50 | Take 33% off (conservative) instead of 50%. |
| `fr_max_hold_bars` | 48 | 0 | **FR timeout**: Disable flow reversal checking after 48 bars (4 hours). Late-session positions are better served by the trailing stop than FR logic. |
| `fr_mfe_grace_r` | 0.3 | 0.0 | **FR MFE grace**: Skip flow reversal exit if position ever reached 0.3R MFE. If it showed profit, FR is likely a false signal. 934 trades had MFE > 0.3R before FR killed them. |
| `opening_range_bars` | **12** | 6 | **Longer opening range**: Use first 12 bars (1 hour) instead of 6 (30 min). More stable/reliable range definition. Opposite direction from P9's 3 bars. |
| `sector_sz_weighted` | Full set | 1.0 each | **Sector-weighted sizing**: Consumer Discretionary 1.1x, Financials 0.8x, Communication 0.8x, Industrials 0.5x. Size up in historically strong sectors, down in weak. |
| `or_breakout_score_min` | 4 | 0 | OR breakout minimum score of 4 (vs P9's 5). Less restrictive but still filters the worst. |
| `fr_trailing_activate_r` | **0.3** | 0.0 | Tighter trailing: activate at just 0.3R MFE (vs P9's 0.5R). More aggressive profit protection. |
| `fr_trailing_distance_r` | **0.3** | 0.5 | Trail only 0.3R behind (vs P9's 0.4R). Tighter trailing stop. |
| `carry_min_cpr` | 0.4 | 0.6 | Easier carry qualification (same as P9). |

---

## E) What Each Kept Feature Actually Does (Plain English)

### P10 Kept Features (in selection order, most impactful first)

1. **fr_trail_tighter** (Round 1, +178% score): Convert the destructive flow-reversal exit into a tight trailing stop. Instead of dumping the position when flow reverses, activate a trailing stop at 0.3R profit that follows 0.3R behind. This was the single most impactful change in both P9 and P10.

2. **rvol_max_5** (Round 2, +26%): Cap entry RVOL at 5.0x. Stocks with extreme volume spikes (>5x normal) are erratic and unpredictable. Blocking them removes a large pool of losing trades.

3. **quick_exit_12bar_02r** (Round 3, +6%): If a trade hasn't reached +0.2R profit after 12 bars (1 hour), cut it. This is a "prove yourself" gate -- if momentum can't produce profit in an hour, the thesis is wrong.

4. **regime_b_0** (Round 4, +3.9%): Completely skip entries when the market regime is "B" (mixed signals). These trades were net losers (PF=0.69, -74R). Zero tolerance for ambiguous regimes.

5. **bigger_portfolio** (Round 5, +2%): Expand from 5 to 10 max positions, 10R heat cap, 3 per sector. More diversification captures more of the opportunity set.

6. **sector_sz_weighted** (Round 6, +1.6%): Size up 10% in Consumer Discretionary (historically strong sector), size down in Financials (-20%), Communication (-20%), and Industrials (-50%).

7. **or_12bars** (Round 7, +0.4%): Define the opening range using the first 12 bars (1 hour) instead of 6 bars (30 minutes). A wider, more established range produces more reliable breakout signals.

8. **fr_max_hold_48** (Round 8, +0.7%): Stop checking for flow reversal after 48 bars (4 hours). Late in the session, flow reversal signals are noisy -- let the trailing stop handle it instead.

9. **max_pos_10** (Round 9, +0.03%): Marginal bump from 8 to 10 positions (building on bigger_portfolio).

10. **fr_mfe_grace_03** (Round 10, +0.5%): If a position ever reached 0.3R profit (MFE), skip the flow reversal exit. The trade showed it was working -- FR is likely a false signal in this case.

11. **score_min_2** (Round 11, +0.07%): Lower momentum score threshold from 3 to 2. Widens the entry funnel to capture more setups, trusting other filters to maintain quality.

12. **combined_score_6** (Round 12, +1.5%): Require momentum score >= 6 specifically for COMBINED_BREAKOUT entries (price breaks both OR high AND prior day high simultaneously). These dual-breakouts had a 10% lower win rate than single OR breakouts -- the quality gate fixes this.

13. **breakout_dist_cap_15r** (Round 13, +0.4%): Reject entries where price has already run >1.5R from the OR high. Don't chase extended moves -- the risk/reward is worse when you enter late.

14. **early_qe_2bar_n05** (Round 14, +0.09%): Ultra-early kill switch: if a trade is at -0.5R or worse after just 2 bars (10 minutes), exit immediately. These are clearly failed entries that won't recover.

15. **partial_125r_033f** (Round 15, +0.03%): Take 33% profit at 1.25R. Lock in gains on a portion of the position to reduce give-back risk on the remaining runner.

16. **or_score_min_4** (Round 16, +0.02%): Require momentum score >= 4 specifically for OR_BREAKOUT entries. Filters the weakest ~219 OR trades that were dragging down the signal quality.

17. **carry_cpr_04** (Round 17, +0.3%): Lower the carry CPR threshold to 0.4 (from 0.6). More positions qualify for overnight carry, which is a major alpha source (+726R identified in diagnostics).

### P9 Kept Features (in selection order)

1. **mfe_trail_05_04** (Round 1, +0.3%): Adjust the FR trailing stop from 0.5/0.5 to 0.5/0.4 (activate at 0.5R MFE, trail 0.4R behind). Tightened the trail distance to reduce give-back.

2. **qe_disabled** (Round 2, +0.16%): Completely disable the time-based quick exit mechanism. P8's 6-bar quick exit was destroying -113.65R across 626 trades (35.8% WR, PF=0.10). Removing it entirely was better.

3. **or_score_min_5** (Round 3, +0.33%): Require momentum score >= 5 for OR_BREAKOUT entries. Aggressive filter that cuts ~219 weak OR trades.

---

## F) Current Live Config Parameters That Would Need to Change

### Current StrategySettings defaults (from config.py):

| Parameter | Current Live | P9 Optimal | P10 Optimal | Notes |
|-----------|-------------|------------|-------------|-------|
| `base_risk_fraction` | 0.005 | **0.02** | 0.005 (unchanged) | P9 = 4x risk; P10 keeps default |
| `max_positions` | 5 | **8** | **10** | Both increase |
| `max_positions_per_sector` | 2 | **3** | **3** | Both increase |
| `heat_cap_r` | 6.0 | **10.0** | **10.0** | Both increase |
| `momentum_score_min` | 3 | **2** | **2** | Both lower |
| `opening_range_bars` | 6 | **3** | **12** | Opposite directions! |
| `stop_atr_multiple` | 1.0 | **2.0** | 1.0 (unchanged) | P9 widens; P10 keeps default |
| `carry_min_r` | 0.5 | **0.0** | 0.5 (unchanged) | P9 removes threshold |
| `carry_min_cpr` | 0.6 | **0.4** | **0.4** | Both lower |
| `regime_mult_b` | 0.8 | **0.0** | **0.0** | Both kill Regime B |
| `rvol_max` | 999.0 | **5.0** | **5.0** | Both cap at 5.0 |
| `flow_reversal_min_hold_bars` | 0 | **24** | 0 (unchanged) | P9 adds grace; P10 doesn't |
| `fr_trailing_activate_r` | 0.0 | **0.5** | **0.3** | Both enable; P10 more aggressive |
| `fr_trailing_distance_r` | 0.5 | **0.4** | **0.3** | Both tighten; P10 tighter |
| `fr_mfe_grace_r` | 0.0 | 0.0 (unchanged) | **0.3** | P10 only |
| `fr_max_hold_bars` | 0 | 0 (unchanged) | **48** | P10 only |
| `quick_exit_max_bars` | 0 | 0 (disabled) | **12** | P10 enables; P9 disables |
| `quick_exit_min_r` | 0.2 | 0.2 | **0.2** | P10 uses default |
| `partial_r_trigger` | 1.5 | (partials disabled) | **1.25** | P10 takes earlier partials |
| `partial_fraction` | 0.50 | (partials disabled) | **0.33** | P10 takes smaller partials |
| `entry_window_end` | 12:00 | **11:00** | 12:00 (unchanged) | P9 narrows window |
| `late_entry_score_min` | 0 | **5** | 0 (unchanged) | P9 adds late gate |
| `sector_mult_consumer_disc` | 1.0 | 1.0 | **1.1** | P10 only |
| `sector_mult_financials` | 1.0 | 1.0 | **0.8** | P10 only |
| `sector_mult_communication` | 1.0 | 1.0 | **0.8** | P10 only |
| `sector_mult_industrials` | 1.0 | 1.0 | **0.5** | P10 only |
| `combined_breakout_score_min` | 0 | 0 | **6** | P10 only |
| `use_combined_quality_gate` | (ablation) | — | **True** | P10 only |
| `breakout_distance_cap_r` | 0.0 | 0.0 | **1.5** | P10 only |
| `use_breakout_distance_cap` | (ablation) | — | **True** | P10 only |
| `qe_stage1_bars` | 0 | 0 | **2** | P10 only |
| `qe_stage1_min_r` | -0.3 | -0.3 | **-0.5** | P10 only |
| `use_quick_exit_stage1` | (ablation) | — | **True** | P10 only |
| `or_breakout_score_min` | 0 | **5** | **4** | Both enable; different thresholds |
| `use_or_quality_gate` | (ablation) | **True** | **True** | Both enable |
| `use_partial_takes` | (ablation) | **False** | **True** | Opposite! P9 disables, P10 enables |
| `use_avwap_filter` | (ablation) | **False** | (default=True) | P9 disables |
| `use_time_based_quick_exit` | (ablation) | **False** | **True** | Opposite! |
| `use_avwap_distance_cap` | (ablation) | **True** | (default=False) | P9 enables |

---

## Summary Assessment

### P9 Philosophy: "Fewer, bigger bets with wider stops"
- 4x risk per trade (2% vs 0.5%)
- Wider stops (2x ATR), longer grace periods (24-bar FR hold)
- Disables partial takes and quick exits — let winners run fully
- Narrower entry window (before 11 AM only)
- Aggressive OR quality gate (score >= 5)
- Higher return, higher drawdown

### P10 Philosophy: "More trades, tighter management, systematic filtering"
- Default risk (0.5%) with 2:1 leverage
- Tighter trailing stops (0.3/0.3), early kill switches (2-bar and 12-bar)
- Re-enables partials at 1.25R, adds sector weighting
- Wider entry funnel (score >= 2, 12-bar OR, no entry time restriction)
- Multiple quality gates (combined score >= 6, breakout distance cap, OR score >= 4)
- Lower return per unit capital, but much tighter drawdown

### Deployment Recommendation Considerations

1. **If risk tolerance is high**: P9's 2% risk / 4.55% DD might be acceptable for the return profile
2. **If capital preservation matters**: P10's 0.5% risk / 1.47% DD is dramatically safer
3. **P10 has more features = more potential points of failure** in live trading, but also more guardrails
4. **P10 was optimized on a new scoring function** that presumably better captures what matters for live trading
5. **P10 started from scratch** — it independently rediscovered most P9 features (rvol_max, regime_b_0, carry_cpr, FR trailing, OR gate) which validates those features as robust
6. **The biggest disagreement** is on quick exit: P9 says remove it entirely, P10 says use a refined version (12-bar + 2-bar stage1). P10's approach is more nuanced.
7. **Opening range definition**: P9 uses 15 min (faster triggers), P10 uses 60 min (more stable signals). This is a fundamental philosophical difference.
