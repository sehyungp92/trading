# Metabase Dashboard Setup

After starting the infrastructure with `docker-compose up -d`, configure Metabase:

## 1. Initial Setup

1. Open http://localhost:3000
2. Create admin account
3. Add database connection:
   - Type: PostgreSQL
   - Host: postgres (or localhost if accessing externally)
   - Port: 5432
   - Database: trading
   - User: trading_reader
   - Password: (from .env)

## 2. Create Dashboard: "Trading Operations"

### Panel A: Live Positions
- **Type:** Table
- **Source:** `v_live_positions`
- **Columns:** instrument_symbol, net_qty, avg_price, stale_minutes
- **Sort:** instrument_symbol ASC

### Panel B: Working Orders
- **Type:** Table
- **Source:** `v_working_orders`
- **Columns:** strategy_id, instrument_symbol, role, side, quantity, filled_qty, state, age_minutes
- **Sort:** age_minutes DESC
- **Highlight:** state = 'WORKING' (yellow), age_minutes > 10 (red)

### Panel C: Today's Realized R by Strategy
- **Type:** Bar Chart
- **Query:**
```sql
SELECT strategy_id, daily_realized_r
FROM risk_daily_strategy
WHERE trade_date = CURRENT_DATE
ORDER BY strategy_id
```

### Panel D: Portfolio Realized R (Today)
- **Type:** Number
- **Query:**
```sql
SELECT COALESCE(daily_realized_r, 0) as value
FROM risk_daily_portfolio
WHERE trade_date = CURRENT_DATE
```
- **Display:** Large number with +/- coloring

### Panel E: Active Halts
- **Type:** Table
- **Source:** `v_active_halts`
- **Columns:** halt_level, entity, halt_reason, last_update_at
- **Conditional:** Hide if empty, show alert styling if rows exist

### Panel F: Strategy Health
- **Type:** Table
- **Source:** `v_strategy_health`
- **Columns:** strategy_id, mode, health_status, heartbeat_age_sec, heat_r, daily_pnl_r
- **Conditional formatting:**
  - health_status = 'OK': green
  - health_status = 'STALE': yellow
  - health_status = 'HALTED': red

### Panel G: Adapter Health
- **Type:** Table
- **Source:** `v_adapter_health`
- **Columns:** adapter_id, connected, health_status, heartbeat_age_sec, disconnect_count_24h
- **Conditional formatting:** same as strategy health

### Panel H: Recent Fills (24h)
- **Type:** Table
- **Source:** `v_recent_fills`
- **Columns:** strategy_id, instrument_symbol, side, quantity, price, fill_ts, slippage
- **Sort:** fill_ts DESC
- **Limit:** 50 rows

### Panel I: Today's Trades
- **Type:** Table
- **Source:** `v_today_trades`
- **Columns:** strategy_id, instrument_symbol, direction, entry_price, exit_price, realized_r, exit_reason, duration_minutes
- **Sort:** entry_ts DESC

### Panel J: Slippage Summary (7 days)
- **Type:** Bar Chart
- **Query:**
```sql
SELECT
    instrument_symbol,
    ROUND(AVG(slippage)::numeric, 4) as avg_slippage,
    COUNT(*) as fill_count
FROM v_recent_fills
WHERE fill_ts >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY instrument_symbol
```

## 3. Set Auto-Refresh

- Set dashboard auto-refresh to 30 seconds for real-time monitoring
