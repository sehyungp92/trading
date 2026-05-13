# TPC Round 6 Alpha Audit

Output directory: `C:\Users\sehyu\Documents\Other\Projects\trading\backtests\output\swing\tpc\round_6\alpha_audit_20260506_160248`
Elapsed minutes: 1.2

## Headline
Train: trades 126, net +96.90%, avgR +2.353, win 53.2%, max DD 13.75%.
OOS: trades 10, net +3.58%, avgR +0.383, win 50.0%, max DD 4.45%.

## Symbol Cohorts
- Train QQQ: trades 20, pnl $+44,173, avgR +0.654, win 70.0%, excellent 14/20.
- OOS QQQ: trades 2, pnl $-5,101, avgR -1.024, win 0.0%, excellent 0/2.
- Train GLD: trades 106, pnl $+52,727, avgR +2.673, win 50.0%, excellent 50/106.
- OOS GLD: trades 8, pnl $+8,679, avgR +0.735, win 62.5%, excellent 5/8.

## Six-Month Context
OOS net-return rank versus rolling train six-month trade windows: 20.4%.
OOS trade-count rank versus rolling train six-month trade windows: 26.5%.
Rolling train six-month trade-count percentiles: {'p10': 8.0, 'p25': 10.0, 'p50': 13.0, 'p75': 18.0, 'p90': 19.0}.

## Shadow Setup Supply
QQQ train entry-ready bar rate 0.139%; QQQ OOS 0.215%.
GLD train entry-ready bar rate 0.472%; GLD OOS 0.298%.

## Interpretation Flags
- Train R is top-winner concentrated: True (top-5 R share 77.9%).
- QQQ OOS sample too small for decisive rejection: True (n=2).
- Probability of zero QQQ excellent trades in OOS if train excellent rate persisted: 9.0%.
- Asset context/news are currently stubbed in the executable code, so the implementation captures a narrower edge than the `swing_3.md` thesis.

## Follow-Up Probes
QQQ entry relaxation is not the fix. `QQQ.entry_order_model = market_next_bar` filled all QQQ OOS requests, but OOS net fell to -11.07% and QQQ OOS average R fell to -1.249 across 9 QQQ trades. Adaptive/stop-market/wider stop-limit probes were unchanged from baseline in this config. See `alpha_audit_entry_probe.csv`.

The strongest structural cleanup is a stop-denominator floor, not a QQQ alpha mutation. `all.min_stop_atr_mult = 0.15` removed the 273R GLD in-sample denominator artifact, reduced train top-5 R winner share from 77.9% to 37.4%, and slightly improved train net (+100.73%) and OOS net (+3.63%). QQQ OOS remained unchanged, so this should be treated as a robustness repair for forward validation rather than a solved QQQ edge. See `alpha_audit_stop_floor_probe.csv`.
