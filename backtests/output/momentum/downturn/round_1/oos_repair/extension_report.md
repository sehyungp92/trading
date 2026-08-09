# Downturn Round 1 OOS Repair Extension

Disposition: **SHADOW_RESEARCH_ONLY**. The observed validation interval was used for comparison and cannot support production promotion.

| Configuration | IS trades | IS return | IS PF | IS DD | Validation trades | Validation return | Validation PF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous recommendation | 127 | 107.08% | 2.76 | 4.81% | 9 | 20.45% | 11.85 |
| Extended recommendation | 139 | 118.84% | 3.09 | 4.41% | 10 | 24.81% | 33.12 |

## Decision

The prior candidate was not the best available point. The extended candidate raises return and frequency in both windows while improving IS PF and drawdown. It adds a four-entry daily cap, longer entry TTL, earlier breakeven, and default-off profit protection during min hold.

The 1.5R floor is the balanced knee. A 1.25R trigger is more interior but gives up IS return; 1.6R raises IS further while giving back about 0.9 percentage points of validation return.

## Robustness

- New unique configurations: 751; total corrected-split repair configurations: 1276.
- TTL/BE surface passing IS >=110% and OOS >=24%: 19/24.
- Floor/lock surface passing IS >=110% and OOS >=24%: 22/38.
- Selected OOS bootstrap probability of positive PnL: 100.0%.
- Commission, slippage, and spread stresses remain strong; one-bar latency remains the principal fragility.

## Caveats

- Validation still contains only ten trades and three active days; all three active-day PnLs and every leave-one-active-day-out result are positive.
- Validation win rate is 90.0% versus 47.5% IS. This gap is not treated as a durable uplift; it is primarily a ten-trade sampling effect.
- The inherited 1.8R emerging TP remains sensitive above approximately 1.85R; it was retained rather than retuned on observed validation.
- The min-hold protection flag is default-off and requires live/core parity implementation and fresh shadow validation before promotion.
