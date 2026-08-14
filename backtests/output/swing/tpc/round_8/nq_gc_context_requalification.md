# Swing NQ/GC context requalification

## Scope and comparability

This audit compares the latest Swing configurations while holding ETF execution data and code constant and changing only the NQ/GC context lane.

- Common replay window: 2025-08-01 through 2026-05-02.
- TPC OOS reporting window: entries on or after 2025-11-01.
- Old lane: the former independently sourced `backtests/swing/data/raw/NQ|GC_1h|1d.parquet` files.
- New lane: the certified explicit-contract, backward-Panama 5m parents and their deterministic 1h/1d children in `backtests/swing/data/authority/oos_20260502`.
- Post-recovery lane: the new certified data plus `QQQ.asset_context_block_opposed_daily=true`.

The certified authority starts in August 2025, so a like-for-like new-data replay of the complete 2021-2026 round history is not possible. The comparisons below therefore do not overwrite or masquerade as the saved full-history round headline metrics.

## Final results

### Recovery replay across the official IS/OOS split

The selected recovery was rerun with `QQQ.asset_context_block_opposed_daily`
disabled and enabled while holding every other round-8 mutation and the current
code constant. The official split is 2025-11-01.

The complete historical in-sample replay necessarily uses the legacy/raw NQ/GC
context because the certified authority does not begin until 2025-08-01. Its
baseline exactly reproduces the saved round-8 in-sample headline (126 trades,
+160.591175%, PF 2.279612, avgR 0.733008, max DD 13.846465%), confirming that
this is the correct round-8 training replay rather than a different sample.

| Complete historical IS | Trades | Net return | Net PnL | PF | Avg R | Total R | Win rate | Max DD | QQQ / GLD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 126 | 160.591175% | $160,591.17 | 2.279612 | 0.733008 | 92.3590 | 59.524% | 13.846465% | 31 / 95 |
| Recovery enabled | 123 | 152.805285% | $152,805.28 | 2.266821 | 0.740693 | 91.1053 | 59.350% | 13.846465% | 28 / 95 |
| Recovery minus baseline | -3 | -7.785890 pp | -$7,785.89 | -0.012791 | +0.007685 | -1.2537 | -0.174 pp | 0.000000 pp | -3 / 0 |

On IS, the veto removes three QQQ longs: +1.1780R on 2024-04-26,
+1.0852R on 2025-03-24, and -1.0104R on 2025-04-14. Their direct baseline PnL
is +$5,104.21; the rest of the $7,785.89 headline reduction comes from the
lower subsequent equity/risk-sizing path after the two earlier winners are
removed. The recovery therefore improves average R slightly and leaves maximum
drawdown unchanged, but it is not IS-return neutral.

For the strict certified lane, one continuous 2025-08-01 through 2026-05-02
bundle was replayed and trades were partitioned at the cutoff. This preserves
the required indicator warm-up and the actual equity/sizing path into OOS.

| Strict certified partition | Lane | Trades | Net PnL | PF | Avg R | Total R | Win rate | Trade-path max DD | QQQ / GLD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS overlap, Aug-Oct 2025 | Baseline | 3 | -$1,200.50 | 0.666608 | -0.170185 | -0.5106 | 33.333% | $3,600.87 | 0 / 3 |
| IS overlap, Aug-Oct 2025 | Recovery | 3 | -$1,200.50 | 0.666608 | -0.170185 | -0.5106 | 33.333% | $3,600.87 | 0 / 3 |
| OOS, Nov 2025-May 2026 | Baseline | 14 | $461.94 | 1.031355 | 0.192580 | 2.6961 | 50.000% | $6,913.60 | 5 / 9 |
| OOS, Nov 2025-May 2026 | Recovery | 11 | $8,330.02 | 2.132287 | 0.524620 | 5.7708 | 63.636% | $2,757.20 | 2 / 9 |
| OOS recovery delta | Recovery minus baseline | -3 | +$7,868.08 | +1.100932 | +0.332040 | +3.0747 | +13.636 pp | -$4,156.40 | -3 / 0 |

The certified IS overlap contains no QQQ trades, so its zero delta is expected
and is not evidence that the QQQ veto is neutral on the wider IS. In certified
OOS the veto removes three losing QQQ longs (2026-02-18, 2026-04-01, and
2026-04-06), each roughly -1R. Their direct PnL is -$7,490.59; preserved equity
for subsequent sizing accounts for the remainder of the +$7,868.08 uplift.

The balanced conclusion is that the recovery is a large OOS repair with a
bounded but real IS opportunity cost: IS trade count falls 2.38%, total R falls
1.36%, and net return falls 4.85% relative to the baseline, while avgR improves
1.05%, PF changes by only -0.56%, and max drawdown is unchanged. Because the
rule was selected after examining this OOS period, these figures support the
mechanism but do not turn that period back into an untouched holdout.

### TPC round 8, OOS only

| Lane | Trades | Net PnL | PF | Avg R | Win rate | QQQ / GLD |
|---|---:|---:|---:|---:|---:|---:|
| Old context | 13 | $6,256.46 | 1.622593 | 0.390896 | 61.538% | 4 / 9 |
| New certified context | 14 | $461.94 | 1.031355 | 0.192580 | 50.000% | 5 / 9 |
| New context, post recovery | 11 | $8,330.02 | 2.132287 | 0.524620 | 63.636% | 2 / 9 |

Final deltas:

| Comparison | Trades | Net PnL | PF | Avg R | Win-rate pp |
|---|---:|---:|---:|---:|---:|
| New minus old | +1 | -$5,794.52 | -0.591238 | -0.198316 | -11.538 |
| Recovered minus new | -3 | +$7,868.08 | +1.100932 | +0.332040 | +13.636 |
| Recovered minus old | -2 | +$2,073.56 | +0.509694 | +0.133724 | +2.098 |

### TPC round 8, complete common window

| Lane | Trades | Net return | Net PnL | PF | Avg R | Win rate | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old context | 16 | 5.055960% | $5,055.96 | 1.370403 | 0.285693 | 56.250% | 6.320020% |
| New certified context | 17 | -0.738560% | -$738.56 | 0.959715 | 0.128563 | 47.059% | 7.422366% |
| New context, post recovery | 14 | 7.129520% | $7,129.52 | 1.650641 | 0.375734 | 57.143% | 5.654413% |

### Portfolio-synergy round 3, complete common window

The portfolio headline uses its configured static-initial-strategy-risk return basis.

| Lane | Trades | Static-risk return | Static-risk PnL | Compounded MTM return | PF | Max DD | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old context | 65 | 52.442637% | $26,221.32 | 54.155663% | 2.839241 | 9.154679% | 1.783989 |
| New certified context | 66 | 51.304479% | $25,652.24 | 53.017504% | 2.736705 | 9.520068% | 1.750651 |
| New context, post recovery | 64 | 52.275103% | $26,137.55 | 53.988129% | 2.833153 | 9.154679% | 1.779375 |

Portfolio deltas:

| Comparison | Trades | Static-risk return pp | Static-risk PnL | PF | Max-DD pp | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| New minus old | +1 | -1.138159 | -$569.08 | -0.102536 | +0.365388 | -0.033338 |
| Recovered minus new | -2 | +0.970625 | +$485.31 | +0.096448 | -0.365388 | +0.028724 |
| Recovered minus old | -1 | -0.167534 | -$83.77 | -0.006088 | 0.000000 | -0.004614 |

The recovery restores 85.3% of the portfolio static-risk PnL lost in the data transition and restores the old-data max drawdown. The small residual difference is the March QQQ timing change described below.

## Which strategies changed

- ATRSS round 3 has no NQ/GC inputs. Its common-window replay is identical across all three lanes: 19 trades, 7.114903% return, PF 3.679863, and 14.078893R.
- Helix round 5 has no NQ/GC inputs. Its common-window standalone replay is identical across all three lanes: 33 trades, 18.440618% return, PF 3.099523, and 28.755153R.
- TPC round 8 is the only standalone strategy affected because QQQ consumes NQ and GLD consumes GC as context.
- Portfolio-synergy round 3 changes only through its TPC source trades and the resulting shared-equity/risk interactions. ATRSS's portfolio component is exactly unchanged. Helix's signals and standalone result are unchanged; its portfolio component moves only slightly through shared-ledger sizing after TPC timing changes.

## Why this was not a relative Panama-level effect

TPC's trend vote tests `close >= SMA20 >= SMA50` (or the inverse) and the sign of a trailing return. A constant additive Panama translation preserves the moving-average ordering and the return sign while prices remain positive. A regression test now locks in this property.

The old and certified series are not related by a constant translation:

| Series | Spread standard deviation | First-difference correlation | Bar-direction disagreement | Trend-vote disagreement |
|---|---:|---:|---:|---:|
| NQ 1h | 198.26 points | 0.908294 | 10.448% | 2.277% |
| NQ 1d | 205.73 points | 0.992276 | 1.596% | 7.857% |
| GC 1h | 38.69 points | 0.966442 | 6.295% | 1.471% |
| GC 1d | 43.91 points | 0.948661 | 9.043% | 6.429% |

The former hourly files also contain material sparse/flat-bar contamination. In the common window, old NQ 1h has 7.437% zero-volume bars and 11.393% flat OHLC bars, versus 0.226% and 0.226% in the certified child. Old GC 1h has 2.022% zero-volume and 3.363% flat OHLC, versus 0.226% and 0.226%. The certified files carry explicit source-contract provenance and use liquid physical-contract volume; the old files do not.

## Signal-level attribution

The material standalone delta is concentrated in QQQ. GLD's OOS trade set is unchanged.

- 2026-02-18: the old NQ hourly vote was neutral and the total context score was -0.20, so the long was rejected. The certified hourly series was fully aligned while daily context was opposed, lifting the score to +0.20 and admitting a -1.0327R loss.
- 2026-03-10: the old hourly context was already aligned at the earlier setup, producing a 14:30 entry and +0.3315R. In the certified series, the earlier hourly vote was neutral; the next completed hour became aligned, producing a 15:00 entry and -1.0209R.
- 2026-04-01 and 2026-04-06: both lanes admitted daily-opposed QQQ longs and both lost about 1R. The standalone recovery removes these structurally weak trades as well. In the portfolio, the 1 April entry was already rejected by the TPC heat ceiling.

The portfolio source-trade cache initially masked this difference because its key used timestamp spans and indicator names but not the source fingerprint. The cache key now includes the TPC replay-source fingerprint, preventing old trades from being reused for a new authority with the same shape.

## Recovery selection

Several alternatives were tested:

- Raising the aggregate QQQ context threshold above 0.20 improved this short OOS window but discarded most QQQ trades. A threshold above 0.35 left only one QQQ OOS trade, an obvious concentration/overfit risk.
- Requiring a newly aligned hourly trend to persist did not remove the losing campaign reliably; it merely delayed eligibility and slightly reduced performance.
- Disabling context or loosening its threshold did not recover performance.
- Hard-vetoing only an explicitly opposed completed daily context directly targets the repeated failure mode while preserving neutral daily context and the March signal family.

The daily-opposition rule also passed a broader old-context check from 2024-02-01 through 2026-05-02: trades changed from 46 to 41, net PnL improved from $78,236.92 to $81,974.49, and PF improved from 3.0261 to 4.1129. The five filtered old-lane QQQ observations contained two winners and three losses totaling -0.7886R; the three filtered certified observations were all losses totaling about -3.07R. This is supportive evidence, but not a substitute for a future untouched holdout.

## Implemented controls

- Added the optional `asset_context_block_opposed_daily` TPC policy with an explicit `asset_context_daily_opposed` rejection reason.
- Enabled it for QQQ in TPC round 8, the live TPC symbol configuration, and portfolio-synergy round 3.
- Added certified TPC context-directory and strict-authority routing to unified portfolio loading.
- Included the selected context authority and manifests in unified replay fingerprints.
- Included the TPC replay-source fingerprint in the portfolio source-trade cache key.
- Added tests for the veto, instrumentation routing, Panama translation invariance, context-authority loading, live/portfolio configuration parity, and cache separation.

The post-recovery figures are requalification results selected with knowledge of this OOS window. They should not be relabelled as untouched OOS performance. The next period after 2026-05-02 must remain a fresh holdout for promotion evidence.
