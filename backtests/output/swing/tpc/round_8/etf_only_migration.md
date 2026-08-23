# TPC Round 8 ETF-only migration

## Decision

TPC is now explicitly an ETF-only system trading QQQ and GLD. It consumes only
the traded ETFs' 15-minute, 1-hour, and daily bars; 30-minute and 4-hour views
are derived from those ETF bars. NQ, GC, futures manifests, and futures context
authority are not part of the live, replay, optimization, or promotion data
contract.

The retained `etf_context` score is self-context, not cross-asset context. It
uses the traded ETF's completed 4-hour DI alignment and 4-hour MA alignment.
QQQ keeps the Round 8 minimum score of -0.1. GLD keeps the score enabled with
the permissive default minimum of -1.0.

## In-sample verification

Window: 2021-02-08 through 2025-11-01. Initial equity: $100,000.

| Metric | Former NQ/GC-enabled result | ETF-only result | ETF-only delta |
|---|---:|---:|---:|
| Trades | 123 | 125 | +2 |
| Net return | 152.805285% | 152.483585% | -0.321700 pp |
| Net PnL | $152,805.28 | $152,483.58 | -$321.70 |
| Dollar profit factor | 2.266821 | 2.217867 | -0.048955 |
| Average R | 0.740693 | 0.724491 | -0.016203 |
| Total R | 91.1053 | 90.5614 | -0.5439 |
| Win rate | 59.3496% | 59.2000% | -0.1496 pp |
| Max drawdown | 13.846465% | 13.846465% | 0.000000 pp |
| QQQ / GLD trades | 28 / 95 | 30 / 95 | +2 / 0 |

Authoritative ETF-only replay output: `etf_only_train_summary.json`.

## Why this is the optimal removal

The futures lane had only a marginal in-sample contribution: 0.5439R and
0.3217 percentage points of return, with no maximum-drawdown improvement. It
also changed semantics partway through the training history because NQ/GC was
unavailable before February 2024 and required a separate certified-data path
for later periods.

The migration removes that non-uniform dependency without retuning the remaining
parameters against the same in-sample result. Retuning after seeing the ablation
would add selection bias for a very small headline difference. Keeping the
existing ETF self-context preserves the strategy's intended 4-hour trend-quality
signal while making every date use the same market-data definition.

Older Round 8 diagnostics and `nq_gc_context_requalification.md` are retained as
pre-migration research lineage. They are not the current TPC specification.
