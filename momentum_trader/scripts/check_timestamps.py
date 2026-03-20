"""Check 5m bar timestamps and ETH window boundaries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.cli import _load_apex_data
import numpy as np
from datetime import datetime, timezone

data = _load_apex_data("NQ", Path("backtest/data/raw"))

times = data["minute_bars"].times
print(f"Total 5m bars: {len(times)}")
print(f"First bar: {times[0]}")
print(f"Last bar:  {times[-1]}")
print()

# Show first 20 bar times
print("First 20 bar times:")
for i in range(min(20, len(times))):
    t = times[i]
    if hasattr(t, 'astype'):
        t = t.astype('datetime64[ms]').astype(datetime)
    print(f"  [{i}] {t}")
print()

# Check for bar times around 16:15 ET (20:15 UTC in summer)
# In summer (EDT), UTC = ET + 4. 16:15 ET = 20:15 UTC
# In winter (EST), UTC = ET + 5. 16:15 ET = 21:15 UTC
print("Bars near session boundaries (checking for 16:15 ET):")
count = 0
for i in range(len(times)):
    t = times[i]
    if hasattr(t, 'astype'):
        dt = t.astype('datetime64[ms]').astype(datetime)
    else:
        dt = t
    # Check UTC hours 20-21 and minutes around 15
    if hasattr(dt, 'hour'):
        h, m = dt.hour, dt.minute
    else:
        # numpy datetime64
        ts = (t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        dt2 = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        h, m = dt2.hour, dt2.minute
    if h in (20, 21) and m in (10, 15, 20):
        if count < 10:
            print(f"  [{i}] {times[i]}  UTC h={h} m={m}")
            count += 1

print()

# Show bars for 2025-09-09 (when orders were placed)
print("Bars on 2025-09-09:")
count = 0
for i in range(len(times)):
    t = times[i]
    if hasattr(t, 'astype'):
        ts = (t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    else:
        ts = t.timestamp() if hasattr(t, 'timestamp') else 0
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if dt.year == 2025 and dt.month == 9 and dt.day == 9:
        if count < 5 or (dt.hour >= 20 and count < 20):
            print(f"  [{i}] {dt}")
        count += 1
    if count > 0 and dt.day != 9:
        print(f"  ... total bars on 2025-09-09: {count}")
        break

# Check hourly index map
h_map = data["hourly_idx_map"]
print(f"\nHourly index map: length={len(h_map)}, min={h_map.min()}, max={h_map.max()}")
print(f"First 20 values: {h_map[:20]}")
