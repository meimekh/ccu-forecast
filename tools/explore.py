"""Scratch exploration of the kit's practice cases and comparables (dev aid, not the entry).

Prints, for every practice case, the hour-1 print and which launches its comparables hold; and for every comparable,
its hour-1 value and its viewers at hours 2..24 relative to hour 1 (the ratio profile the forecast is built from).
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

KIT = os.environ.get("KIT", "C:/script/x/kits/gta6-twitch-ccu-forecast-01/gta6-twitch-ccu-forecast-01-kit")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def day_hours(comp):
    """The comparable's viewers at hours 1..24 of its own day_start, or None where the source has no point."""
    start = ts(comp["day_start"])
    by_hour = {}
    for p in comp["hourly"]:
        delta = (ts(p["from"]) - start).total_seconds() / 3600.0
        if abs(delta - round(delta)) < 1e-6:
            h = int(round(delta)) + 1
            if 1 <= h <= 24:
                by_hour[h] = p["viewers"]
    return [by_hour.get(h) for h in range(1, 25)]


def main():
    inp = os.path.join(KIT, "input")
    for name in sorted(os.listdir(inp)):
        if not name.endswith(".in"):
            continue
        case = json.load(open(os.path.join(inp, name), encoding="utf-8"))
        h1 = case["known"][0]
        print("=" * 100)
        print(f"{name}  id={case['id']}  game={case['game']['name']}  day_start={case['day_start']}")
        print(f"  hour1 print={h1['viewers']} channels={h1['channels']} reads={h1['reads']}")
        names = [c["name"] for c in case["comparables"]]
        print(f"  comparables n={len(names)}")
        for n in names:
            print(f"    - {n}")
        print("  history_hourly tail:")
        for r in case["history_hourly"][-6:]:
            print("   ", r)
    # Ratio profiles of every comparable in p00 (the final holds all 42; practice cases leave out the 4 launches).
    case = json.load(open(os.path.join(inp, "p00.in"), encoding="utf-8"))
    print("=" * 100)
    print("comparable ratio profiles (viewers at hour h / hour 1), h=1..24")
    print(f"{'name':42s} {'h1':>9s} " + " ".join(f"{h:>6d}" for h in range(2, 25)))
    for c in case["comparables"]:
        hrs = day_hours(c)
        if hrs[0] is None:
            print(f"{c['name'][:42]:42s}  (no hour-1 point)")
            continue
        base = hrs[0]
        cells = []
        for h in range(2, 25):
            v = hrs[h - 1]
            cells.append("  n/a " if v is None else f"{v / base:6.2f}")
        print(f"{c['name'][:42]:42s} {base:9.0f} " + " ".join(cells))
    print("=" * 100)
    for c in case["comparables"][:6]:
        print(f"--- {c['name']} unlock: {c['unlock'][:300]}")


if __name__ == "__main__":
    sys.exit(main())
