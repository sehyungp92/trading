import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import {
  SYSTEM_ORDER,
  getSystem,
  setRegistryCache,
  type PortfolioData,
  type StrategyData,
  type PositionRow,
  type TradeRow,
  type OrderRow,
  type StrategyHealthRow,
  type AdapterHealthRow,
  type HaltRow,
  type HealthData,
  type SystemPnlSummary,
  type LiveBatchResponse,
} from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // 8 parallel DB queries (down from 11 across 6 endpoints)
    // Note: strategies + health used to both query v_strategy_health — now we query it once
    const [
      portfolioRows,
      unrealizedRows,
      strategyRows,
      adapterRows,
      haltRows,
      positionRows,
      tradeRows,
      orderRows,
      registryRows,
    ] = await Promise.all([
      // Portfolio realized (aggregated view handles multi-family)
      query<{
        daily_realized_r: number;
        daily_realized_usd: number;
        portfolio_open_risk_r: number;
        halted: boolean;
        halt_reason: string | null;
      }>(
        `SELECT daily_realized_r, daily_realized_usd, portfolio_open_risk_r, halted, halt_reason
         FROM v_portfolio_daily_summary
         WHERE trade_date = (now() AT TIME ZONE 'America/New_York')::date`
      ),
      // Portfolio unrealized + heat
      query<{ unrealized_pnl: number; heat_r: number }>(
        `SELECT
           COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
           COALESCE(SUM(open_risk_r), 0) AS heat_r
         FROM positions
         WHERE net_qty != 0`
      ),
      // Strategies (v_strategy_health + risk_daily_strategy — single query serves both strategies and health)
      query<StrategyData>(
        `SELECT
           sh.strategy_id,
           sh.mode,
           sh.last_heartbeat_ts,
           sh.heartbeat_age_sec,
           sh.health_status,
           sh.heat_r,
           sh.daily_pnl_r,
           sh.last_error,
           sh.last_error_ts,
           COALESCE(rds.daily_realized_r, 0)   AS daily_realized_r,
           COALESCE(rds.daily_realized_usd, 0)  AS daily_realized_usd,
           COALESCE(rds.open_risk_r, 0)         AS open_risk_r,
           COALESCE(rds.filled_entries, 0)      AS filled_entries,
           COALESCE(rds.halted, FALSE)          AS halted,
           rds.halt_reason
         FROM v_strategy_health sh
         LEFT JOIN risk_daily_strategy rds
           ON sh.strategy_id = rds.strategy_id AND rds.trade_date = (now() AT TIME ZONE 'America/New_York')::date
         ORDER BY sh.strategy_id`
      ),
      // Adapter health
      query<AdapterHealthRow>(
        `SELECT * FROM v_adapter_health ORDER BY adapter_id`
      ),
      // Active halts
      query<HaltRow>(
        `SELECT halt_level, entity, halt_reason, last_update_at
         FROM v_active_halts
         ORDER BY halt_level, entity`
      ),
      // Positions
      query<PositionRow>(
        `SELECT
           account_id, instrument_symbol, strategy_id, net_qty, avg_price,
           unrealized_pnl, realized_pnl, open_risk_dollars, open_risk_r,
           last_update_at,
           EXTRACT(EPOCH FROM (now() - last_update_at)) / 60 AS stale_minutes
         FROM positions
         WHERE net_qty != 0
         ORDER BY last_update_at DESC`
      ),
      // Trades
      query<TradeRow>(
        `SELECT * FROM v_today_trades LIMIT 50`
      ),
      // Orders
      query<OrderRow>(
        `SELECT * FROM v_working_orders ORDER BY created_at DESC`
      ),
      // Registry: strategy→family mapping from DB (drives getSystem())
      query<{ strategy_id: string; family_id: string }>(
        `SELECT DISTINCT ON (strategy_id) strategy_id, family_id
         FROM risk_daily_strategy
         WHERE family_id IS NOT NULL
           AND family_id != 'unknown'
         ORDER BY strategy_id, trade_date DESC`
      ),
    ]);

    // Assemble portfolio
    const portfolioRaw = portfolioRows[0] ?? {
      daily_realized_r: 0,
      daily_realized_usd: 0,
      portfolio_open_risk_r: 0,
      halted: false,
      halt_reason: null,
    };
    const unrealized = unrealizedRows[0] ?? { unrealized_pnl: 0, heat_r: 0 };
    const portfolio: PortfolioData = {
      daily_realized_r: portfolioRaw.daily_realized_r,
      daily_realized_usd: portfolioRaw.daily_realized_usd,
      portfolio_open_risk_r: portfolioRaw.portfolio_open_risk_r,
      unrealized_pnl: unrealized.unrealized_pnl,
      halted: portfolioRaw.halted,
      halt_reason: portfolioRaw.halt_reason,
      heat_r: unrealized.heat_r,
    };

    // Derive health from strategy rows (no redundant v_strategy_health query)
    const healthStrategies: StrategyHealthRow[] = strategyRows.map(s => ({
      strategy_id: s.strategy_id,
      mode: s.mode,
      last_heartbeat_ts: s.last_heartbeat_ts,
      heartbeat_age_sec: s.heartbeat_age_sec,
      health_status: s.health_status,
      heat_r: s.heat_r,
      daily_pnl_r: s.daily_pnl_r,
      last_error: s.last_error,
      last_error_ts: s.last_error_ts,
    }));

    const health: HealthData = {
      strategies: healthStrategies,
      adapters: adapterRows,
      halts: haltRows,
    };

    // Populate registry cache from DB so getSystem() uses live family mappings
    if (registryRows.length > 0) {
      const familyMap: Record<string, string> = {};
      for (const r of registryRows) familyMap[r.strategy_id] = r.family_id;
      setRegistryCache(familyMap);
    }

    // Compute per-system P&L summaries
    const systemPnl: SystemPnlSummary[] = SYSTEM_ORDER.map(sys => {
      const strats = strategyRows.filter(s => getSystem(s.strategy_id) === sys);
      return {
        system: sys,
        daily_realized_r: strats.reduce((sum, s) => sum + s.daily_realized_r, 0),
        daily_realized_usd: strats.reduce((sum, s) => sum + s.daily_realized_usd, 0),
        heat_r: strats.reduce((sum, s) => sum + s.heat_r, 0),
        filled_entries: strats.reduce((sum, s) => sum + s.filled_entries, 0),
        strategy_count: strats.length,
        healthy_count: strats.filter(s => s.health_status === 'OK').length,
      };
    }).filter(sp => sp.strategy_count > 0);

    const data: LiveBatchResponse = {
      portfolio,
      strategies: strategyRows,
      positions: positionRows,
      trades: tradeRows,
      orders: orderRows,
      health,
      systemPnl,
      serverTime: new Date().toISOString(),
    };

    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/live] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
