# TPC Round 7 Holdout Alpha Absence And Repair Evaluation

## Executive conclusion

The holdout alpha is not absent in the broad market-movement sense; it is mostly absent in the specific shape this strategy is allowed to trade. GLD still captures usable continuation alpha in the holdout. QQQ is the real failure: the holdout contains movement, but very little of it passes the implemented `4h trend regime -> 1h value pullback -> 15m/30m confirmation -> context -> execution` chain.

The mutation search did not find a robust OOS repair that also preserves in-sample performance. Every material OOS booster either increased bad QQQ supply or overfit one holdout trade while damaging the training distribution. No mutation should be promoted as an OOS repair from this pass.

## Market context

- train QQQ: buy-hold +88.58%, annualized daily vol 22.8%, max DD 35.6%, above daily SMA20 63.5%, above daily SMA50 65.9%.
- oos QQQ: buy-hold +6.66%, annualized daily vol 18.1%, max DD 11.8%, above daily SMA20 46.8%, above daily SMA50 25.0%.
- train GLD: buy-hold +114.62%, annualized daily vol 15.4%, max DD 21.0%, above daily SMA20 59.7%, above daily SMA50 60.3%.
- oos GLD: buy-hold +14.75%, annualized daily vol 32.5%, max DD 19.2%, above daily SMA20 57.3%, above daily SMA50 34.7%.

Interpretation: QQQ OOS was positive but spent only 25.0% of days above the 50-day average versus 65.9% in training. GLD OOS was strong but much more volatile than training, with a 19.2% buy-hold drawdown. Those are tradable markets, but not necessarily clean continuation-pullback markets.

## Signal funnel evidence

### train QQQ
- Shadow bars: 74,047; in-session bars: 17,264; entry-ready bars: 154; ready/all 0.208%; ready/in-session 0.892%.
- Top first failures:
  - session_filter: 56,783 bars (76.7% of all denominator)
  - regime_ma_rsi_conflict: 15,338 bars (88.8% of in-session denominator)
  - direction_short_not_a_plus: 606 bars (3.5% of in-session denominator)
  - regime_long_trend_quality: 432 bars (2.5% of in-session denominator)
  - regime_extended_from_value: 233 bars (1.3% of in-session denominator)
  - pullback_none: 172 bars (1.0% of in-session denominator)
  - regime_short_trend_quality: 168 bars (1.0% of in-session denominator)
  - entry_ready: 124 bars (0.7% of in-session denominator)
  - pullback_none: 61 bars (0.4% of in-session denominator)
  - pullback_none: 28 bars (0.2% of in-session denominator)
  - entry_ready: 23 bars (0.1% of in-session denominator)
  - asset_context: 18 bars (0.1% of in-session denominator)

### train GLD
- Shadow bars: 73,527; in-session bars: 27,457; entry-ready bars: 292; ready/all 0.397%; ready/in-session 1.063%.
- Top first failures:
  - session_filter: 46,070 bars (62.7% of all denominator)
  - regime_ma_rsi_conflict: 21,909 bars (79.8% of in-session denominator)
  - regime_extended_from_value: 2,749 bars (10.0% of in-session denominator)
  - regime_long_trend_quality: 835 bars (3.0% of in-session denominator)
  - regime_short_trend_quality: 529 bars (1.9% of in-session denominator)
  - pullback_none: 408 bars (1.5% of in-session denominator)
  - pullback_none: 305 bars (1.1% of in-session denominator)
  - entry_ready: 144 bars (0.5% of in-session denominator)
  - pullback_none: 110 bars (0.4% of in-session denominator)
  - entry_ready: 101 bars (0.4% of in-session denominator)
  - pullback_none: 77 bars (0.3% of in-session denominator)
  - confirmation_count: 43 bars (0.2% of in-session denominator)

### oos QQQ
- Shadow bars: 7,905; in-session bars: 1,800; entry-ready bars: 4; ready/all 0.051%; ready/in-session 0.222%.
- Top first failures:
  - session_filter: 6,105 bars (77.2% of all denominator)
  - regime_ma_rsi_conflict: 1,494 bars (83.0% of in-session denominator)
  - direction_short_not_a_plus: 79 bars (4.4% of in-session denominator)
  - pullback_none: 65 bars (3.6% of in-session denominator)
  - regime_long_trend_quality: 48 bars (2.7% of in-session denominator)
  - regime_short_trend_quality: 26 bars (1.4% of in-session denominator)
  - pullback_none: 21 bars (1.2% of in-session denominator)
  - asset_context: 21 bars (1.2% of in-session denominator)
  - pullback_none: 17 bars (0.9% of in-session denominator)
  - confirmation_combo: 5 bars (0.3% of in-session denominator)
  - regime_extended_from_value: 5 bars (0.3% of in-session denominator)
  - confirmation_combo: 4 bars (0.2% of in-session denominator)

### oos GLD
- Shadow bars: 7,385; in-session bars: 2,700; entry-ready bars: 20; ready/all 0.271%; ready/in-session 0.741%.
- Top first failures:
  - session_filter: 4,685 bars (63.4% of all denominator)
  - regime_ma_rsi_conflict: 2,160 bars (80.0% of in-session denominator)
  - regime_extended_from_value: 249 bars (9.2% of in-session denominator)
  - regime_long_trend_quality: 127 bars (4.7% of in-session denominator)
  - pullback_none: 64 bars (2.4% of in-session denominator)
  - regime_short_trend_quality: 24 bars (0.9% of in-session denominator)
  - pullback_none: 24 bars (0.9% of in-session denominator)
  - daily_room: 15 bars (0.6% of in-session denominator)
  - entry_ready: 9 bars (0.3% of in-session denominator)
  - entry_ready: 8 bars (0.3% of in-session denominator)
  - confirmation_combo: 5 bars (0.2% of in-session denominator)
  - confirmation_count: 4 bars (0.1% of in-session denominator)

The QQQ drop is severe: entry-ready/in-session falls from 0.892% in training to 0.222% OOS. GLD falls from 1.064% to 0.741%, which is a much smaller degradation. This is why the strategy still finds GLD trades but almost no QQQ trades.

## Implementation critique

- The code captures a narrower alpha than the written thesis. `regime_direction` does not compute true moving-average slope; it compares the MA value to recent closes. That behaves more like a reclaim/rejection geometry test than the spec's "rising/falling 4h MA" test. It can be useful, but it is not the same theoretical alpha and can starve transition regimes.
- The written thesis names SPY, breadth, semiconductors, DXY, real yields, miners, and economic calendar context. The executable context currently uses only proxy futures plus self DI/MA votes, and the news filter is a stub. So some theoretical context alpha is not captured at all.
- Pullback detection only checks 1h EMA20, EMA50, and VWAP value hits plus a simple fib depth. It does not model anchored VWAP, prior breakout shelves, support/resistance flips, or trendline zones from the spec. That can miss real pullbacks that are visually valid but not represented by these three value levels.
- The daily-room filter is not the main OOS bottleneck. Relaxing QQQ daily room to 1.2R or 1.0R changed nothing in the tested holdout.
- Execution is not the main bottleneck either. Switching QQQ to confirmation-close added one holdout trade but reduced OOS net by -4.86%, so the missed QQQ signal was not simply a stop-limit fill problem.

## Mutation search result

Tested 40 targeted single/combo mutations plus 10 refined QQQ-valid-short variants. Guardrail required material OOS improvement without material train deterioration in net return, avgR, dollar PF, drawdown, or QQQ excellent supply.

### OOS boosters that failed guardrails
- qqq_allow_valid_shorts: OOS net delta +2.77%, trades +1, QQQ trades +1, QQQ excellent +1; train net -36.14%, avgR -0.159, PF -0.518, DD +8.64%.
- combo_qqq_short_fibwide: OOS net delta +0.23%, trades +5, QQQ trades +5, QQQ excellent +3; train net -73.70%, avgR -0.292, PF -0.833, DD +10.79%.
- qqq_context_off: OOS net delta -10.07%, trades +6, QQQ trades +6, QQQ excellent +1; train net -4.22%, avgR -0.019, PF -0.050, DD +0.00%.
- gld_t1_090_partial70: OOS net delta +0.83%, trades +0, QQQ trades +0, QQQ excellent +0; train net -28.16%, avgR -0.137, PF -0.171, DD +0.00%.
- gld_mfe_trigger150: OOS net delta -5.07%, trades +0, QQQ trades +0, QQQ excellent +0; train net -1.91%, avgR -0.019, PF +0.007, DD +0.00%.
- qqq_entry_confirmation_close: OOS net delta -4.86%, trades +1, QQQ trades +1, QQQ excellent +0; train net -27.85%, avgR -0.070, PF -0.320, DD +5.14%.
- qqq_valid_shorts_min14: OOS net delta +2.77%, QQQ excellent +1; train net -7.22%, avgR -0.060, PF -0.232, DD +4.87%.
- qqq_valid_shorts_min14_t1_075_stop05: OOS net delta +3.49%, QQQ excellent +2; train net -11.97%, avgR -0.068, PF -0.249, DD +4.87%.

### Mutations that helped training but did not repair OOS
- qqq_fib_a_high090: OOS unchanged; train net +16.58%, trades +12, avgR -0.035, PF -0.017, QQQ excellent +6.
- gld_trail_after_t1_6: OOS unchanged; train net +4.16%, trades +1, avgR +0.004, PF +0.009, QQQ excellent +0.
- qqq_fib_high090_gld_trail: OOS unchanged; train net +21.05%, trades +13, avgR -0.031, PF -0.008, QQQ excellent +6.

## Recommendation

Do not promote any OOS-repair mutation from this pass. The only material OOS improvement is non-A+ QQQ shorts, and the training evidence says that lane is structurally weak. The stricter score-14 version still fails guardrails, so this is not a robust missing-alpha fix.

The next useful work is structural research, not a looser mutation pass: implement a diagnostic module that labels rejected bars with forward MFE/MAE, add true MA-slope/transition-regime features separately from the existing reclaim geometry, and add the missing context inputs from the spec. Only then should a new mutation round test a distinct QQQ transition-pullback module rather than loosening the current trend-continuation core.
