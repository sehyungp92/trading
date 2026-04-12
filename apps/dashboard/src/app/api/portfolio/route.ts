import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { PortfolioData } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const [portfolioRows, unrealizedRows] = await Promise.all([
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
      query<{ unrealized_pnl: number; heat_r: number }>(
        `SELECT
           COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
           COALESCE(SUM(open_risk_r), 0) AS heat_r
         FROM positions
         WHERE net_qty != 0`
      ),
    ]);

    const portfolio = portfolioRows[0] ?? {
      daily_realized_r: 0,
      daily_realized_usd: 0,
      portfolio_open_risk_r: 0,
      halted: false,
      halt_reason: null,
    };

    const unrealized = unrealizedRows[0] ?? { unrealized_pnl: 0, heat_r: 0 };

    const data: PortfolioData = {
      daily_realized_r: portfolio.daily_realized_r,
      daily_realized_usd: portfolio.daily_realized_usd,
      portfolio_open_risk_r: portfolio.portfolio_open_risk_r,
      unrealized_pnl: unrealized.unrealized_pnl,
      halted: portfolio.halted,
      halt_reason: portfolio.halt_reason,
      heat_r: unrealized.heat_r,
    };

    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/portfolio] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
