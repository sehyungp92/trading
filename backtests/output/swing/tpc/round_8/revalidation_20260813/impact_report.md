# TPC round 8 impact revalidation — 2026-08-13

## Verdict

TPC round 8 is numerically unchanged on its original evaluation window through
2025-11-01. The phased-auto runner itself cannot currently be rerun or promoted
under strict certification because certified NQ/GC 5-minute roots and their
derivation manifests do not yet exist.

Extending NQ/GC context beyond its previous 2026-02-06 endpoint has a small,
measurable effect in the later Swing portfolio window through 2026-05-01. It
adds three long QQQ TPC trades whose combined result is -1.70984R. This reduces
the portfolio return by 0.84045 percentage points, while leaving maximum
drawdown and portfolio rule behavior unchanged.

ATRSS and AKC Helix are not behaviorally affected.

## TPC round 8: exact-window replay

| Metric | Saved round 8 | Current replay | Delta |
|---|---:|---:|---:|
| Trades | 126 | 126 | 0 |
| Net return | 160.591175% | 160.591175% | 0 |
| Total R | 92.359035 | 92.359035 | 0 |
| Average R | 0.733008 | 0.733008 | 0 |
| Dollar profit factor | 2.279612 | 2.279612 | 0 |
| Max drawdown | 13.846465% | 13.846465% | 0 |
| Win rate | 59.523810% | 59.523810% | 0 |

The stored and current values match at full serialized precision. See
`current_code_train_summary.json`.

## Later-window futures-context impact

The promoted Swing portfolio round 3 was replayed through its historical
2026-05-01 cutoff under two conditions:

1. current QQQ/GLD data with NQ/GC context capped at the old 2026-02-06 endpoint;
2. current QQQ/GLD data with NQ/GC context extended through 2026-05-01.

The capped-context arm reproduces the saved portfolio result to floating-point
noise. The extended-context arm produces the following change:

| Metric | Old/capped context | Extended context | Delta |
|---|---:|---:|---:|
| Portfolio trades | 716 | 719 | +3 |
| Net return | 509.768698% | 508.928246% | -0.840453 pp |
| Profit factor | 3.354873 | 3.340115 | -0.014758 |
| Max drawdown | 8.555451% | 8.555451% | 0 |
| Sharpe | 2.124567 | 2.122696 | -0.001871 |
| TPC sleeve trades | 134 | 137 | +3 |
| TPC sleeve total R | 79.270645 | 77.560439 | -1.710206R |

Portfolio rule behavior is identical in both arms: 249 rule events, one block,
248 sizing events, and three portfolio daily-stop activations.

### Newly enabled TPC trades

All three are long QQQ `classic_38_62` / `structure_stop` trades:

| Entry | Exit | Result | MFE | Assessment |
|---|---|---:|---:|---|
| 2026-03-10 14:30 | 2026-03-10 17:30 | +0.33154R | 1.11571R | Small winner with giveback |
| 2026-04-01 16:45 | 2026-04-01 18:00 | -1.02709R | 0.24266R | Clean stop loss; little favorable excursion |
| 2026-04-06 14:00 | 2026-04-06 14:30 | -1.01430R | 0.00000R | Immediate clean stop loss |

Their combined result is -1.70984R. These are ordinary bounded losses rather
than tail-loss edge cases, and they do not expand portfolio drawdown. Small
additional differences in later common GLD trade dollar sizing arise from the
changed prior equity path; they do not represent different GLD signals or exits.

## Other Swing strategies

### ATRSS round 3

The exact 2026-05-01 replay matches all saved final metrics, including 264
trades, 238.9710945% net return, 6.582510158 profit factor, 11.178399873% max
drawdown, 234.781969R, and 1.832806280 Sharpe.

### AKC Helix round 5

The phase-runner valuation matches all decision-grade saved fields: 325 trades,
114.684184% net return, 2.158638120 profit factor, 8.733122R max R drawdown,
49.846154% win rate, and identical regime/side profit factors. Derived R and
Sharpe values differ only around 1e-5 to 1e-7, below any material threshold.

The full synchronized replay also reproduces the historical raw trade count,
dollar P&L, final equity, and percentage drawdown exactly. The existing Helix
`run_summary.json` contains a pre-existing inconsistency between the independent
phase-runner valuation and the synchronized full-diagnostics valuation; this
predates and is unrelated to the futures-context changes.

### Dependency isolation

ATRSS and Helix use QQQ/GLD ETF bars and do not import or consume the NQ/GC roll
or futures-context authority path. The shared replay-cache code change starts in
the TPC loaders; the ATRSS and Helix loader bodies are unchanged.

## Current full-data observation

This is not a like-for-like certification comparison, because the data now
extends through 2026-07-10:

- TPC round-8 config: 147 trades, 171.167695% return, 96.649626R, 1.956141
  dollar profit factor, and unchanged 13.846465% max drawdown.
- Swing portfolio: 754 trades, 527.717272% return, 3.324447 profit factor, and
  unchanged 8.555451% max drawdown.

Relative to the saved May portfolio, the additional period adds 38 trades and
17.94857 percentage points of return, with lower profit factor. This is an
extended-window observation, not evidence of an optimization-code uplift.

## Strict certification status

Starting round 8 from phase 1 now fails before candidate evaluation, as designed,
because these authority inputs are absent for both NQ and GC:

- certified explicit-contract `5m.parquet` root;
- 5-minute source manifest;
- certified 1-hour derivation manifest;
- certified 1-day derivation manifest;
- aggregate futures-context manifest.

Existing uncertified NQ/GC 1-hour and 1-day files remain usable for ordinary
research replay, but cannot support strict phased-auto certification or
promotion. No round-8 historical artifact was overwritten by the failed strict
run.

## Verification

The focused Swing regression suite completed with **137 passed** tests. A
same-process A/B experiment also exposed an existing TPC source-replay cache-key
weakness: different data roots can collide unless the cache is cleared. The
reported causal A/B result was obtained in isolated processes / with an explicit
cache clear, so it is not contaminated by that issue.
