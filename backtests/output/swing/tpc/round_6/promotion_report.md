# TPC Round 6 Promotion Report

Promoted config revalidated after ETF engine accounting correction: `2026-05-06T17:26:55.400477+00:00`

## Accepted Combination
- Base: cumulative round 8 manifest mutations, including accepted round 6 and round 7 mutations.
- Stop-floor repair: `all.min_stop_atr_mult = 0.15`.
- Candidate overlay: `alpha7_balanced_pair_ma100_030_no_di__t1r090_partial70_stop035`.
- Structural overlay: QQQ NQ context gate, A+ gated QQQ shorts, and soft GLD GC context scoring.

## Accounting Correction
The earlier `+998.95%` in-sample headline was not a real return. A stale same-bar `MFE_GIVEBACK` flatten on the GLD long entered `2022-10-26 13:15 UTC` was submitted for the original quantity after a T1 partial had already reduced the position. The trade record clipped to the remaining shares, but the equity cash ledger credited the full stale flatten order. The engine now clips exit cash and fill quantity to the remaining open quantity.

## Corrected Validation Snapshot
- Train: net +120.94%, trades 127, avgR +0.634, dollar PF 2.01, max DD 13.68%, excellent 70.
- Train QQQ: trades 30, avgR +0.641, pnl $+51,324, win 76.7%.
- Train GLD: trades 97, avgR +0.631, pnl $+69,613, win 52.6%.
- Selection OOS: net +11.00%, trades 10, avgR +0.661, dollar PF 3.28, max DD 3.46%, excellent 6.
- OOS QQQ: trades 1, avgR +0.168, pnl $+305, win 100.0%.
- OOS GLD: trades 9, avgR +0.716, pnl $+10,693, win 66.7%.

## Decision Notes
The promoted structure still treats QQQ and GLD differently because the validation weakness was not symmetric. QQQ needed an external NQ context gate before accepting continuation/reversal setups; GLD lost too many good trades when GC context was used as a hard gate, so context remains available to the score and diagnostics without raising its minimum threshold. The score model remains the seven-component `alpha7` form requested in the implementation constraints.
