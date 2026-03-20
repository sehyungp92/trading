import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { OrderRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // v_working_orders filters to active statuses and computes age_minutes
    const rows = await query<OrderRow>(
      `SELECT * FROM v_working_orders ORDER BY created_at DESC`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/orders]', err);
    return NextResponse.json({ error: 'Database error' }, { status: 500 });
  }
}
