import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { DailyPnlPoint } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const rows = await query<DailyPnlPoint>(
      `SELECT trade_date, daily_realized_r
       FROM risk_daily_portfolio
       WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
       ORDER BY trade_date`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/daily-pnl]', err);
    return NextResponse.json({ error: 'Database error' }, { status: 500 });
  }
}
