# TPC Round 8 Strategy Evaluation - In-Sample Smoke

Generated: 2026-05-09

## Inputs

- Requested file check: `backtests/output/swing/tpc/round_8/full_diagnostics_in_sample.txt` is not present on disk.
- Round 8 train-only sources used instead:
  - `backtests/output/swing/tpc/round_8/round_final_diagnostics.txt`
  - `backtests/output/swing/tpc/round_8/run_summary.json`
  - `backtests/output/swing/tpc/round_8/optimized_config.json`
  - `backtests/output/swing/tpc/round_8/phase_1_diagnostics.txt`
  - `backtests/output/swing/tpc/round_8/phase_2_diagnostics.txt`
  - `backtests/output/swing/tpc/round_8/phase_3_diagnostics.txt`
  - `backtests/output/swing/tpc/round_8/phase_4_diagnostics.txt`
- Closest full in-sample diagnostic used as context:
  - `backtests/output/swing/tpc/round_6/full_diagnostics_in_sample.txt`
- Fresh combined smoke report:
  - `backtests/output/swing/tpc/round_8/in_sample_smoke_20260509_fresh_4w/smoke_report.md`

All smoke tests are train-only through `2025-11-01` and used `max_workers=4`.

## Baseline

Round 8 optimized train-only baseline:

- Net return: `+133.01%`
- AvgR / TotalR: `+0.681R` / `+86.47R`
- Trades: `127`, `2.30/month`
- Dollar PF: `2.08`
- Max DD: `13.68%`
- MFE capture: `33.8%`
- Avg MFE / MAE: `+2.96R` / `0.93R`
- Never-worked rate: `23.6%`
- Low-MFE loss rate: `33.1%`
- Right-then-lost rate: `7.1%`
- Symbol mix: `94 GLD`, `33 QQQ`; GLD trade share `74.0%`
- Additive PB30 lane: `6 trades`, `+0.701R avg`, `+4.21R total`

## Strengths

The strategy has real continuation edge. The baseline is profitable, has usable frequency, keeps max drawdown below the round guardrail, and finds meaningful excursion: average MFE is almost `3R`. The QQQ sleeve is especially clean, with a much higher excellent-rate than GLD. The round-8 PB30 additive lane is small but directionally positive, and disabling it removes four trades and slightly reduces net.

The strategy is also not dependent on one giant winner in dollar terms: dollar top-5 winner share is `24.0%`, which is acceptable for this style. Worst-year behavior is not great but is not catastrophic in sample: `2023` is the weak year at roughly `-2.25%`.

## Weaknesses

The largest weakness is not raw signal absence; it is realization. MFE capture is only `33.8%`, so a lot of found movement is not monetized. This is confirmed by the ablation where disabling the structure trail explodes train return to `+208.86%` and avgR to `+2.089R`, but max DD jumps to `22.68%`, so the latent alpha exists but is risky to harvest naively.

False positives are still material. `23.6%` of trades never reach `0.5R` MFE, and `33.1%` are low-MFE losses. GLD is the main quality drag: it supplies most of the trades but has a lower excellent-rate than QQQ.

The additive alpha sample is thin. PB30 contributes only `6` actual trades. It passes the round gates, but most EMA20/PB30 route variants either do nothing, underperform, or fail route-level sample gates.

## Signal Extraction

The signal is capturing real alpha but not maximum alpha. Average MFE near `3R` versus realized avgR `0.681R` means there is unharvested excursion. The best completed signal probe was:

- `pb30_duration_5_18`: net `+145.38%`, dNet `+12.37%`, avgR `+0.698`, trades `127`, DD `13.85%`, low-MFE `32.3%`.

This is the strongest train-only research lead. It suggests the 30m pullback duration window is leaving some alpha on the table when it starts at 6 bars and stretches to 20.

Most broader supply tests were negative:

- `pb30_relax_orderly`: dNet `-9.78%`, trades `130`, avgR down.
- `enable_gld_type_b`: dNet `-7.54%`, DD `14.00%`.
- `all_type_b_not_aplus`: dNet `-8.15%`, DD `14.00%`.
- `pb30_duration_7_22`: more trades, worse score and higher DD.

Conclusion: add supply only through narrowly controlled PB30 duration research; broad shallow or relaxed pullback expansion dilutes edge.

## Discrimination

The current filters are moderately discriminatory but not fully surgical. Many stricter score/confirmation/trend filters were no-ops, meaning the current accepted set already sits above those thresholds or those gates do not bind in the replay.

The clearest high-quality discrimination variant was:

- `type_a_value_hits2`: avgR `+0.759`, DD `10.81%`, but net `-11.22%` and trades fall to `97`.

That is useful as a drawdown-control lead, not as the main return/frequency config. The best non-starving discriminator was:

- `max_extension_qqq_175`: net `+135.51%`, dNet `+2.50%`, avgR `+0.687`, trades `125`, DD unchanged.

Hard VWAP-only confirmation is too restrictive:

- `require_vwap`: trades `83`, DD `7.48%`, rejected for `min_valid_trades`.

Conclusion: the strategy rejects many weak signals already, but GLD false positives remain. Tightening all signals is too expensive; symbol-specific QQQ extension and GLD-specific quality research are better than global clamps.

## Entry

The existing entry mechanism is the best tested balance. The structure-stop baseline materially outperformed market or retest alternatives.

Rejected/weak entry alternatives:

- `entry_market_next_bar`: net `+103.66%`, DD `20.02%`, rejected for max DD.
- `entry_structure_stop_market`: net `+59.98%`, avgR `+0.231`, rejected.
- `entry_adaptive_structure_stop`: trades `76`, rejected for sample.
- `entry_vwap_retest_limit`: trades `37`, rejected for sample.
- `entry_midpoint_retest_limit`: net only `+11.50%`, rejected.
- `gld_structure_stop`: trades `64`, rejected for sample.

Widening the stop-limit cap to `0.12` or `0.20` was a no-op. Conclusion: do not replace the current entry model. Further entry work should focus on pre-entry signal quality and per-route activation, not generic market/retest replacements.

## Trade Management And Exits

Best train-only management leads:

- `QQQ.t1_r=1.0`: net `+142.67%`, dNet `+9.66%`, avgR `+0.698`, trades `126`, DD `13.68%`.
- `trail_after_t1_30m_bars=6`: net `+137.90%`, dNet `+4.89%`, trades `128`, DD `13.63%`.
- `t2_r=3.0`: net `+136.37%`, dNet `+3.36%`, DD `13.78%`.
- `disable_mfe_giveback`: net `+137.30%`, dNet `+4.29%`, avgR `+0.698`, DD unchanged.

However, Round 8 already removed `t2_r=3.0` after holdout attribution, so it should be treated as train-only evidence only. The `disable_mfe_giveback` ablation implies the giveback exit may be trimming winners or duplicating protection already provided by the structure trail, but removing protection needs a fresh holdout/live-path check before adoption.

The profit-floor ladder did not work: more trades and lower low-MFE rate came with weaker additive quality and rejection. Add-ons slightly increased net but reduced avgR, so they look like leverage/monetization rather than new signal alpha.

## Drawdown

Max DD can be minimized, but not for free.

Best DD reducers:

- `risk_temper_020`: DD `9.25%`, but net drops by `-52.37%`.
- `type_a_value_hits2`: DD `10.81%`, but trades fall to `97` and net drops by `-11.22%`.
- `pb30_value_touch_limit`: DD `12.86%` and net `+2.21%`, but rejected for insufficient PB30 EMA20 route sample.

Best balanced DD/return lead:

- `trail_after_t1_30m_bars=6`: DD improves slightly to `13.63%` while net improves `+4.89%` and frequency is preserved.

Conclusion: drawdown reduction below roughly `10-11%` is possible only by reducing risk or starving frequency. The best current goal is not hard DD minimization; it is preserving the `13-14%` DD band while improving expected return and route quality.

## Recommended Research Leads

Do not promote directly from this in-sample report. Treat these as next candidates for untouched holdout and live-path parity checks:

1. `all.pb30_pullback_min_bars_30m=5`, `all.pb30_pullback_max_bars_30m=18`
2. `QQQ.t1_r=1.0`
3. `all.trail_after_t1_30m_bars=6`
4. `QQQ.max_extension_atr_mult=1.75`
5. Drawdown-only alternate: `all.type_a_value_hits_min=2`, only if the objective explicitly accepts lower frequency and lower net.

Avoid broad supply expansions for now: relaxed PB30 orderly rules, GLD Type B, global Type B, generic market entries, and generic retest entries all diluted the edge or failed sample/risk gates.

## Smoke Scope

Completed smoke scope:

- `112` variants
- `42` smoke-promotable by guardrails
- `23` hard rejected
- all train-only to `2025-11-01`
- all with `max_workers=4`

An optional additional targeted stack-combo pass was attempted after the completed report, but it exceeded the one-hour timeout and wrote no artifacts. It is excluded from the conclusions above.
