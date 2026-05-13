# TPC Round 6 Alpha Audit

Output directory: `C:\Users\sehyu\Documents\Other\Projects\trading\backtests\output\swing\tpc\round_7\alpha_audit_holdout_diagnostic`
Elapsed minutes: 1.1

## Headline
Train: trades 123, net +132.72%, avgR +0.681, win 60.2%, max DD 13.68%.
OOS: trades 10, net +11.00%, avgR +0.661, win 70.0%, max DD 3.46%.

## Symbol Cohorts
- Train QQQ: trades 32, pnl $+59,612, avgR +0.659, win 78.1%, excellent 25/32.
- OOS QQQ: trades 1, pnl $+305, avgR +0.168, win 100.0%, excellent 0/1.
- Train GLD: trades 91, pnl $+73,111, avgR +0.689, win 53.8%, excellent 45/91.
- OOS GLD: trades 9, pnl $+10,693, avgR +0.716, win 66.7%, excellent 6/9.

## Six-Month Context
OOS net-return rank versus rolling train six-month trade windows: 40.8%.
OOS trade-count rank versus rolling train six-month trade windows: 34.7%.
Rolling train six-month trade-count percentiles: {'p10': 8.0, 'p25': 10.0, 'p50': 13.0, 'p75': 18.0, 'p90': 18.200000000000003}.

## Shadow Setup Supply
QQQ train entry-ready bar rate 0.208%; QQQ OOS 0.051%.
GLD train entry-ready bar rate 0.397%; GLD OOS 0.271%.

## Interpretation Flags
- Train R is top-winner concentrated: False (top-5 R share 35.6%).
- QQQ OOS sample too small for decisive rejection: True (n=1).
- Probability of zero QQQ excellent trades in OOS if train excellent rate persisted: 21.9%.
- Asset context is now executable via completed-bar proxy context; the news-risk input remains a stub, so that part of the `swing_3.md` thesis is still not captured.
