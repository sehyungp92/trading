# TPC Round 8 OOS Optimal Configuration Decision

Generated: 2026-05-09

## Scope

- Train window: through `2025-11-01`
- OOS window: after `2025-11-01`, with warmup starting at the cutoff
- Candidates tested: `25`
- Worker cap: `max_workers=4`
- Baseline config: `backtests/output/swing/tpc/round_8/optimized_config.json`
- Candidate result files:
  - `oos_candidate_results.json`
  - `oos_candidate_results.csv`
  - `recommended_config.json`

The OOS sample is small: the baseline has only `10` OOS trades. The decision therefore uses OOS as a veto/confirmation layer and requires train-side return, drawdown, and frequency to remain strong.

## Baseline

| Window | Net | AvgR | Trades | Trades/mo | Max DD | Dollar PF | Low-MFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | `+133.01%` | `+0.681` | `127` | `2.30` | `13.68%` | `2.08` | `33.1%` |
| OOS | `+11.00%` | `+0.661` | `10` | `1.69` | `3.46%` | `3.28` | `20.0%` |

## Selected Configuration

Selected candidate: `top3_pb30_qqq_no_mfe`

Mutations versus the Round 8 optimized baseline:

- `all.pb30_pullback_min_bars_30m = 5`
- `all.pb30_pullback_max_bars_30m = 18`
- `QQQ.t1_r = 1.0`
- `all.mfe_giveback_trigger_r = 0.0`

Result:

| Window | Net | AvgR | Trades | Trades/mo | Max DD | Dollar PF | Low-MFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | `+160.59%` | `+0.733` | `126` | `2.28` | `13.85%` | `2.28` | `32.5%` |
| OOS | `+11.32%` | `+0.679` | `10` | `1.69` | `3.46%` | `3.34` | `20.0%` |

Versus baseline:

- Train net improves by `+27.58%`.
- Train avgR improves by `+0.052R`.
- Train frequency is essentially preserved: `126` vs `127` trades.
- Train max DD increases only slightly: `13.85%` vs `13.68%`.
- OOS net improves by `+0.32%`.
- OOS avgR improves by `+0.018R`.
- OOS trades and max DD are unchanged: `10` trades, `3.46%` DD.

## Why This Wins

This is the best balanced candidate because it improves expected return on both train and OOS while preserving frequency and not increasing OOS drawdown. The OOS edge over baseline is small, but it does not require sacrificing OOS trade count, and the train-side improvement is large enough to suggest the mechanism is not just a one-trade OOS accident.

The `top4_pb30_qqq_no_mfe_ext` candidate produced identical train and OOS metrics, but the extra `QQQ.max_extension_atr_mult=1.75` filter was non-binding. The simpler `top3_pb30_qqq_no_mfe` version is preferred because it avoids carrying an inactive extra rule.

## Alternatives Rejected

`type_a_value_hits2` has the highest OOS net at `+12.82%` and lower train DD at `10.81%`, but it cuts train trades from `127` to `97` and OOS trades from `10` to `9`. It is a defensive low-frequency variant, not the optimal return/frequency configuration.

`risk_temper_020` and the risk-tempered stacks reduce OOS DD to `2.22%` and train DD to roughly `9.2%`, but net return collapses materially. Example: `top3_plus_risk_temper` falls to `+93.95%` train and `+8.29%` OOS. This is useful only if capital preservation dominates return.

`top3_pb30_qqq_trail6` has the best blended objective and the highest train net at `+161.80%`, but OOS return is exactly baseline at `+11.00%`. It remains a strong backup, but the selected no-MFE-giveback stack has better OOS avgR and net with the same OOS DD and trade count.

`disable_structure_trail` still fails as a robust choice. It has very high train net, but train DD jumps to `22.68%` and OOS underperforms the baseline.

## Recommendation

Use `recommended_config.json` from this folder as the OOS-selected research configuration:

`backtests/output/swing/tpc/round_8/oos_candidate_check_20260509_final_4w/recommended_config.json`

This should be considered the best current research config, not a final production promotion. OOS has now been used for model selection, so the next validation step should be a fresh untouched slice, anchored walk-forward, or paper-trade shadow before live adoption.
