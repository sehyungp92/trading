import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { EquityCurvePoint, DailyPnlPoint, ChartBatchResponse } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const [equityRows, dailyRows] = await Promise.all([
      query<EquityCurvePoint>(
        `SELECT
           trade_date,
           daily_realized_r,
           SUM(daily_realized_r) OVER (ORDER BY trade_date) AS cumulative_r
         FROM risk_daily_portfolio
         WHERE trade_date >= CURRENT_DATE - INTERVAL '90 days'
         ORDER BY trade_date`
      ),
      query<DailyPnlPoint>(
        `SELECT trade_date, daily_realized_r
         FROM risk_daily_portfolio
         WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
         ORDER BY trade_date`
      ),
    ]);

    const data: ChartBatchResponse = {
      equityCurve: equityRows,
      dailyPnl: dailyRows,
      serverTime: new Date().toISOString(),
    };

    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/charts]', err);
    return NextResponse.json({ error: 'Database error' }, { status: 500 });
  }
}
