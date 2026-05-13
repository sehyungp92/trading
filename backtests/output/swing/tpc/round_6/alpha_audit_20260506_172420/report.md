# TPC Round 6 Alpha Audit

Output directory: `C:\Users\sehyu\Documents\Other\Projects\trading\backtests\output\swing\tpc\round_6\alpha_audit_20260506_172420`
Elapsed minutes: 1.8

## Headline
Train: trades 127, net +120.94%, avgR +0.634, win 58.3%, max DD 13.68%.
OOS: trades 10, net +11.00%, avgR +0.661, win 70.0%, max DD 3.46%.

## Symbol Cohorts
- Train QQQ: trades 30, pnl $+51,324, avgR +0.641, win 76.7%, excellent 23/30.
- OOS QQQ: trades 1, pnl $+305, avgR +0.168, win 100.0%, excellent 0/1.
- Train GLD: trades 97, pnl $+69,613, avgR +0.631, win 52.6%, excellent 47/97.
- OOS GLD: trades 9, pnl $+10,693, avgR +0.716, win 66.7%, excellent 6/9.

## Six-Month Context
OOS net-return rank versus rolling train six-month trade windows: 44.9%.
OOS trade-count rank versus rolling train six-month trade windows: 32.7%.
Rolling train six-month trade-count percentiles: {'p10': 8.0, 'p25': 10.0, 'p50': 13.0, 'p75': 18.0, 'p90': 20.0}.

## Shadow Setup Supply
QQQ train entry-ready bar rate 0.194%; QQQ OOS 0.051%.
GLD train entry-ready bar rate 0.413%; GLD OOS 0.271%.

## Interpretation Flags
- Train R is top-winner concentrated: False (top-5 R share 35.3%).
- QQQ OOS sample too small for decisive rejection: True (n=1).
- Probability of zero QQQ excellent trades in OOS if train excellent rate persisted: 23.3%.
- Asset context is now executable via completed-bar proxy context; the news-risk input remains a stub, so that part of the `swing_3.md` thesis is still not captured.
