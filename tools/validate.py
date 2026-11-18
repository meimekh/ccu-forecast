"""Contract, determinism, timing and edge-case checks for main.py (dev tool, not part of the entry).

Runs the entry exactly the way the grader does - one process, one case per input directory, nothing but the standard
library - and checks what the rules require of the output:

  * every <id>.out is a regular file of at most 65,536 bytes of UTF-8 JSON, one object with exactly the key "hours",
    a list of exactly 23 rows, hours 2 to 24 each exactly once, each row exactly {"hour", "q"} with five finite
    numbers above 0 and at most 1e9, non-decreasing;
  * two runs of the same case are byte for byte the same;
  * each single-case run is far inside the 8 second budget;
  * edge cases: no comparables (the persistence fallback), a zero first hour, a first hour so large the clamp binds,
    and a case with no history at all.

Usage: python tools/validate.py
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = Path(os.environ.get("KIT", "C:/script/x/kits/gta6-twitch-ccu-forecast-01/gta6-twitch-ccu-forecast-01-kit"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def run_entry(input_dir: Path, output_dir: Path) -> float:
    """One run of the entry over `input_dir`, returning its wall time in seconds."""
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(input_dir), MAKEFUN_OUTPUT_DIR=str(output_dir))
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")], env=env,
                          capture_output=True, text=True, timeout=60)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"main.py exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return elapsed


def contract_errors(path: Path) -> list:
    """Every way `path` breaks the output contract (empty list = it holds)."""
    errs = []
    if path.is_symlink() or not path.is_file():
        return [f"{path.name}: not a regular file"]
    size = path.stat().st_size
    if size > 65536:
        errs.append(f"{path.name}: {size} bytes, over 65536")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or list(doc.keys()) != ["hours"]:
        return errs + [f"{path.name}: keys are {list(doc.keys()) if isinstance(doc, dict) else type(doc).__name__}"]
    rows = doc["hours"]
    if not isinstance(rows, list) or len(rows) != 23:
        return errs + [f"{path.name}: {len(rows) if isinstance(rows, list) else 'not a list'} rows"]
    if sorted(r.get("hour") for r in rows) != list(range(2, 25)):
        errs.append(f"{path.name}: hours are not 2..24 exactly once")
    for row in rows:
        if set(row.keys()) != {"hour", "q"}:
            errs.append(f"{path.name} hour {row.get('hour')}: keys {sorted(row.keys())}")
            continue
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            errs.append(f"{path.name} hour {row['hour']}: q is not five values")
            continue
        for i, v in enumerate(q):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                errs.append(f"{path.name} hour {row['hour']} q{i}: not a finite number")
            elif not (0 < v <= 1e9):
                errs.append(f"{path.name} hour {row['hour']} q{i}: {v} outside (0, 1e9]")
        for i in range(1, 5):
            if not (q[i] >= q[i - 1]):
                errs.append(f"{path.name} hour {row['hour']}: q{i} below q{i-1}")
    return errs


def synthetic_cases() -> dict:
    """Small hand-built cases that exercise the fallback, zero and clamp paths (no comparables needed)."""
    day = "2026-11-19T00:00:00Z"
    rows = [{"hour": h, "from": f"2026-11-19T{h - 1:02d}:00:00Z", "until": f"2026-11-19T{h:02d}:00:00Z"}
            for h in range(2, 25)]
    def case(name, viewers, history):
        return {
            "schema": "makefun/twitch-ccu-case/v1", "id": name,
            "game": {"name": "Grand Theft Auto VI", "twitch_id": "1867279146"},
            "day_start": day, "checkpoint": "2026-11-19T01:00:00Z", "levels": LEVELS,
            "known": [{"hour": 1, "from": day, "until": "2026-11-19T01:00:00Z", "viewers": viewers,
                       "channels": 100, "reads": 60}],
            "forecast": rows, "history": [], "history_hourly": history, "streams": None,
            "categories_hourly": None, "comparables": [], "notes": "synthetic, for the contract checks",
        }
    hist = [{"from": f"2026-11-18T{h:02d}:00:00Z", "viewers": 1000 + 10 * h, "channels": 5, "reads": 60}
            for h in range(1, 24)]
    return {
        "no_comparables": case("no_comparables", 100000, hist),
        "zero_hour_one": case("zero_hour_one", 0, hist),
        "huge_hour_one": case("huge_hour_one", 5_000_000_000, hist),
        "empty_history": case("empty_history", 250000, []),
    }


def final_like_case() -> dict:
    """A case shaped like h00: a first hour far above every comparable, a slide into it, the real comparable list.

    GTA VI's hour 1 cannot resemble any past launch's: the launch is bigger than the whole comparable set and its first
    UTC hour carries the European and UK unlocks. This case puts the model in that corner - a first hour of 2,000,000
    viewers, above the largest comparable (ELDEN RING's 875,155) - to check the kernel does not collapse onto one
    template and that the interval stays honest.
    """
    real = load_practice_doc()
    day = real["day_start"]
    viewers = {off: round(900_000 + (2_600_000 - 900_000) * (off + 24) / 24.0) for off in range(-24, 0)}
    hist = [{"from": iso_from(day, off), "viewers": viewers[off], "channels": 400, "reads": 1}
            for off in range(-24, 0)]
    return {
        "schema": "makefun/twitch-ccu-case/v1", "id": "h00_like",
        "game": {"name": "Grand Theft Auto VI", "twitch_id": "1867279146"},
        "day_start": day, "checkpoint": "2026-11-19T01:00:00Z", "levels": LEVELS,
        "known": [{"hour": 1, "from": day, "until": "2026-11-19T01:00:00Z", "viewers": 2_000_000,
                   "channels": 12000, "reads": 60}],
        "forecast": [{"hour": h, "from": iso_from(day, h - 1), "until": iso_from(day, h)} for h in range(2, 25)],
        "history": [], "history_hourly": hist, "streams": None, "categories_hourly": None,
        "comparables": real["comparables"], "notes": "synthetic, shaped like the final's case",
    }


def iso_from(day: str, offset: int) -> str:
    """An ISO instant `offset` hours from a day start such as 2026-11-19T00:00:00Z."""
    from datetime import datetime, timedelta, timezone
    start = datetime.fromisoformat(day.replace("Z", "+00:00"))
    return (start + timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_practice_doc(name: str = "p00") -> dict:
    return json.loads((KIT / "input" / f"{name}.in").read_text(encoding="utf-8"))


def final_like_checks(tmp: Path, ok: bool) -> bool:
    """The h00-shaped case and the leakage guard, on the model's own functions."""
    sys.path.insert(0, str(ROOT))
    import main as M

    case = final_like_case()
    rows = M.forecast_hours(case)
    errs = contract_rows(rows)
    ok &= not errs
    p1 = 2_000_000
    print(f"\nfinal-like case (hour 1 = {p1:,}, above every comparable's): {'OK' if not errs else '; '.join(errs)}")
    print("  median ratio: " + " ".join(f"{r['q'][2] / p1:5.2f}" for r in rows))
    print("  5%-95% span : " + " ".join(f"{math.log(r['q'][4] / r['q'][0]):5.2f}" for r in rows))

    # How much weight the kernel leaves on how many templates (a collapse onto one would hollow out the spread), and
    # how much the spread floor widens the raw weighted empirical span.
    ctx = M.target_context(case)
    comps = case["comparables"]
    contexts = M.comparable_contexts(comps)
    feats = M.profile_features(ctx)
    day = M.epoch(case["day_start"])
    fits = []
    for comp, c in zip(comps, contexts):
        if not c or c.get(0) is None or c[0] <= 0:
            continue
        if any(day <= t < day + 24 * 3600.0 for t in M.horizon_times(comp)) or abs(day - M.epoch(comp["day_start"])) < 1.0:
            continue
        f = M.profile_features(c)
        ratios = {h: c[h - 1] / c[0] for h in range(2, 25) if c.get(h - 1) is not None and c[h - 1] > 0}
        if f is None or len(ratios) < 6:
            continue
        fits.append((M.weight(feats, f, M.CONFIG), ratios, comp["name"]))
    total = sum(w for w, _, _ in fits)
    heaviest = sorted(((w / total, n) for w, _, n in fits), reverse=True)
    print(f"  {len(fits)} templates, effective number {M.ess([w for w, _, _ in fits]):.1f}; heaviest: "
          + ", ".join(f"{n[:24]} {w:.2f}" for w, n in heaviest[:4]))
    print("  raw weighted empirical 5%-95% span, before the floor holds it:")
    for h in (2, 6, 10, 14, 18, 22, 24):
        pairs = [(math.log(r[h]), w) for w, r, _ in fits if h in r]
        lo = M.weighted_quantiles(pairs, 0.05, "interp")
        hi = M.weighted_quantiles(pairs, 0.95, "interp")
        print(f"    hour {h:2d}: {hi - lo:4.2f} nats")

    # The leakage guard: a comparable that carries the target's own day must be ignored. Take the h00-like case, add a
    # "self" comparable holding the target's real hours 2..24, and check the forecast does not move.
    control = json.loads(json.dumps(case))
    leaky = json.loads(json.dumps(case))
    hourly = [{"from": iso_from(case["day_start"], off), "viewers": v, "channels": 400}
              for off, v in sorted(
                  {off: [row for row in case["history_hourly"] if row["from"] == iso_from(case["day_start"], off)][0]
                   ["viewers"] for off in range(-24, 0)}.items())]
    hourly.append({"from": case["day_start"], "viewers": 2_000_000, "channels": 12000})
    for h in range(2, 25):  # the "answer" the leak would repeat, deliberately wild so using it would show
        hourly.append({"from": iso_from(case["day_start"], h - 1), "viewers": 2_000_000 * (1 + h), "channels": 1})
    leaky["comparables"] = case["comparables"] + [{
        "name": "Grand Theft Auto VI launch day (the target itself)", "day_start": case["day_start"],
        "unlock": "midnight local", "platforms": "PS5", "source": "synthetic", "hourly": hourly}]
    a = json.dumps(M.forecast_hours(control), separators=(",", ":"))
    b = json.dumps(M.forecast_hours(leaky), separators=(",", ":"))
    same = a == b
    ok &= same
    print(f"  leakage guard: forecast unchanged when the target's own day is offered as a comparable: {same}")
    return ok


def contract_rows(rows: list) -> list:
    """The contract violations of a forecast row list (empty = it holds)."""
    errs = []
    if len(rows) != 23 or sorted(r["hour"] for r in rows) != list(range(2, 25)):
        return ["rows are not hours 2..24"]
    for row in rows:
        if set(row.keys()) != {"hour", "q"} or len(row["q"]) != 5:
            errs.append(f"hour {row['hour']} shape")
            continue
        q = row["q"]
        if any(not (0 < v <= 1e9) for v in q) or any(q[i] < q[i - 1] for i in range(1, 5)):
            errs.append(f"hour {row['hour']} quantiles")
    return errs


def main() -> int:
    inp = KIT / "input"
    tmp = Path(tempfile.mkdtemp(prefix="ccu-validate-"))
    ok = True
    try:
        # The four practice cases, run twice end to end.
        times = []
        digests = []
        for run in (1, 2):
            out = tmp / f"run{run}"
            out.mkdir()
            times.append(run_entry(inp, out))
            digests.append({p.name: p.read_bytes() for p in sorted(out.glob("*.out"))})
        same = digests[0] == digests[1]
        print(f"practice run over {len(digests[0])} cases: {times[0]:.2f}s, then {times[1]:.2f}s; "
              f"byte-identical across runs: {same}")
        ok &= same
        for name, blob in sorted(digests[0].items()):
            path = tmp / "run1" / name
            errs = contract_errors(path)
            ok &= not errs
            print(f"  {name:10s} {len(blob):6d} bytes  {'OK' if not errs else '; '.join(errs)}")

        # One case per sandbox, as the grader runs it: the cold single-case time.
        single = tmp / "single-in"
        single.mkdir()
        shutil.copy(inp / "p02.in", single / "p02.in")
        out = tmp / "single-out"
        out.mkdir()
        cold = run_entry(single, out)
        print(f"single-case cold run (p02, 1.0 MB case): {cold:.3f}s  (budget 8s)")
        ok &= cold < 8.0

        # Edge cases.
        edge_in = tmp / "edge-in"
        edge_in.mkdir()
        cases = dict(synthetic_cases())
        cases["h00_like"] = final_like_case()
        for name, doc in cases.items():
            (edge_in / f"{name}.in").write_text(json.dumps(doc), encoding="utf-8")
        edge_out = tmp / "edge-out"
        edge_out.mkdir()
        edge_time = run_entry(edge_in, edge_out)
        print(f"edge cases: {edge_time:.3f}s")
        for name in cases:
            path = edge_out / f"{name}.out"
            errs = contract_errors(path) if path.exists() else ["no output written"]
            ok &= not errs
            doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"hours": []}
            q05 = doc["hours"][0]["q"][0] if doc["hours"] else None
            q95 = doc["hours"][0]["q"][4] if doc["hours"] else None
            print(f"  {name:15s} hour-2 band [{q05}, {q95}]  {'OK' if not errs else '; '.join(errs)}")
        # The fallback must be exactly the house baseline's arithmetic, all five quantiles.
        fb = json.loads((edge_out / "no_comparables.out").read_text(encoding="utf-8"))
        worst = 0.0
        for row in fb["hours"]:
            h = row["hour"]
            for z, q in zip(Z, row["q"]):
                worst = max(worst, abs(q - round(100000 * math.exp(z * 0.15 * math.sqrt(h - 1)), 6)))
        close = worst < 1e-6
        ok &= close
        print(f"  no-comparables fallback == the house baseline (worst deviation {worst:.2e}): {close}")
        # The clamp: a first hour of 5e9 must still leave every quantile at or under 1e9.
        huge = json.loads((edge_out / "huge_hour_one.out").read_text(encoding="utf-8"))
        print(f"  huge_hour_one max quantile: {max(max(r['q']) for r in huge['hours']):.0f} (clamped at 1e9)")

        # The median multiplier profile each practice case gets, for a reading of the shape.
        print("\nmedian count as a multiple of hour 1 (the model's own profile), hours 2..24:")
        for name in ("p00", "p01", "p02", "p03"):
            doc = json.loads((tmp / "run1" / f"{name}.out").read_text(encoding="utf-8"))
            case = json.loads((inp / f"{name}.in").read_text(encoding="utf-8"))
            p1 = next(r for r in case["known"] if r["hour"] == 1)["viewers"]
            ratios = [row["q"][2] / p1 for row in doc["hours"]]
            print(f"  {name} p1={p1:7d}  " + " ".join(f"{r:5.2f}" for r in ratios))

        ok = final_like_checks(tmp, ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL CHECKS PASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
