# Standalone stock-data authority

This repository owns its stock acquisition and replay evidence. It has no runtime or
data dependency on `trading_assistant_data`.

## Data classes

- `backtests/stock/data/raw/*.parquet` is the retained legacy compatibility cache.
  Its inventory proves retained bytes only and is never an acquisition receipt.
- `backtests/stock/data/authority/objects/` contains direct, content-addressed IBKR
  objects. RTH and extended data have different dataset identities.
- `backtests/stock/data/authority/receipts/` contains immutable acquisition receipts.
- `backtests/stock/data/authority/refs/latest/` contains the only mutable dataset
  pointers. A blocked acquisition cannot update one.
- `backtests/stock/data/authority/derived/` contains diagnostic transformations such
  as the legacy ETH-to-RTH projection. Derived legacy data is not production authority.

## Operator sequence

1. Preserve and verify the legacy cache:

   ```powershell
   python -m backtests.stock.data.authority_cli snapshot-legacy
   python -m backtests.stock.data.authority_cli project-legacy-rth
   ```

2. Start TWS or IB Gateway with read-only API access, then acquire direct RTH data for
   the canonical 98-symbol universe and all daily reference inputs:

   ```powershell
   python -m backtests.stock.data.update_intraday --session rth --start 2025-03-21
   ```

   Use `--latest` only after an accepted parent exists. To retain extended data for a
   separate strategy requirement, run `--session extended`; it writes a separate identity.

3. Run tests and commit the immutable objects, receipts, references, code, configuration,
   and canonical universe used for the run. Bundle construction rejects dirty inputs.

4. Build and verify a frozen bundle:

   ```powershell
   python -m backtests.stock.data.authority_cli build-bundle `
     --output backtests/stock/data/authority/bundles/objects/may_is_june_oos.json
   python -m backtests.stock.data.authority_cli verify-bundle `
     --bundle backtests/stock/data/authority/bundles/objects/may_is_june_oos.json
   ```

5. Compare the deterministic legacy projection with direct RTH authority:

   ```powershell
   python -m backtests.stock.data.authority_cli compare-rth `
     --bundle backtests/stock/data/authority/bundles/objects/may_is_june_oos.json
   ```

6. Supply the verified bundle to every official backtest or optimization. Legacy data
   requires the explicit diagnostic-only `--allow-legacy-data` override:

   ```powershell
   python -m backtests.stock run --strategy alcb `
     --bundle backtests/stock/data/authority/bundles/objects/may_is_june_oos.json `
     --start 2026-05-01 --end 2026-06-30
   ```

The CLI writes a run data-context manifest containing the bundle checksum, dataset
identities, session policies, receipt IDs, object checksums, universe version, and
code/config checksum before execution begins.
