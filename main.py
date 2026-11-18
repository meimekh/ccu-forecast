"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's launch-day Twitch category viewers, hour by hour.

Standard library only, deterministic, no network. Reads every ``*.in`` in ``$MAKEFUN_INPUT_DIR`` (else /input) and writes
one ``<id>.out`` per case to ``$MAKEFUN_OUTPUT_DIR`` (else /output): ``{"hours": [{"hour": h, "q": [5 quantiles]}, ...]}``
for the hours 2 to 24, the quantiles at the levels 0.05, 0.25, 0.5, 0.75, 0.95.

The model
=========
A case gives hour 1's count ``p1`` (the mean over the day's first UTC hour) and a ``comparables`` list: past big launches,
each with its hourly category counts from 14 days before its own release day to 3 days after. The forecast is a
distribution over each later hour's count as a *multiple of p1*.

1.  Each comparable's release-day window is aligned to whole-hour offsets from its own ``day_start``; its hour 1 is the
    offset 1 and its ratios are ``r[h] = viewers[h] / viewers[1]``.
2.  A comparable is a template for the target only as far as the target resembles it. Two launches whose first UTC hour
    plays the same role in their own launch draw the same kind of day. The model measures that role on the *pre-checkpoint*
    data only, which every case has for itself and for every comparable: ``scale`` (hour 1's size, log viewers) and
    ``rel0`` (hour 1 against the hour before it, in log). A Gaussian kernel on those differences weights each comparable;
    kernel width 0 means "every comparable, equally".
3.  For each hour h the model takes the weighted empirical quantiles, in log space, of the templates' ``ln r[h]``, and
    scales them back by p1. A power ``alpha`` pulls the quantiles towards (alpha < 1) or away from (alpha > 1) their
    median, and a light geometric smoothing over h damps single-hour noise. Disagreement between templates widens the
    interval; that is the point of a spread a rank rewards.

No case name, id or game name is read: only the numbers and times in the case document. Nothing after the checkpoint is
read, so the forecast can never peek at the hours it is scored on. Everything is deterministic.

Outside the grader's container:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The levels scored, in the order each row lists them, and the standard normal's quantile at each.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
# The persistence baseline's spread in log units per square-root hour (the house baseline: the fallback profile).
BASELINE_SIGMA = 0.15
# Viewers are clamped into the contract's range: a finite number above 0 and at most 1e9.
VIEWER_MIN = 1.0
VIEWER_MAX = 1e9

# The model's parameters. Chosen by leave-one-launch-out over the kit's comparables (tools/eval.py, QA_WORKER.md):
# every comparable in turn is held out as a target, the model is refitted on the rest, and the pinball loss is measured
# on the held-out launch's real later hours, on three pools (all holds, holds with a first hour of 100k or more, and
# holds that resemble the kit's four practice launches). These are the values that minimised the pools' mean loss.
CONFIG = {
    # Gaussian kernel widths, in the feature's own units, 0 = do not use the feature at all (weight 1). A missing
    # feature value on either side leaves that dimension out of the kernel.
    "bw_scale": 0.9,    # log of hour 1's viewers: how big the launch's first hour was
    "bw_rel0": 0.6,     # hour 1 against the hour before it, in log: is hour 1 a peak, a slide, or a rise?
    "bw_relprev": 0.0,  # hour 1 against the six hours before it: measured, but it did not beat the two above
    # Spread calibration: q_tau <- median * (q_tau / median) ** alpha (1.0 = the raw empirical spread).
    "alpha": 1.0,
    # Geometric smoothing over h of each quantile curve: [1, k, 1] / (k + 2), k = 0 disables it.
    "smooth_k": 1,
    # The weighted quantiles: "step" (an order statistic) or "interp" (the weighted CDF read as a line).
    "qmethod": "interp",
    # A kernel that narrows onto one or two launches would collapse the spread to nothing; below this effective number
    # of templates the weights are mixed with a constant until they hold it (0 disables the floor).
    "ess_floor": 0.0,
    # A floor on the 5% to 95% log span of each hour's quantiles, in nats: an honest interval is never narrower.
    "min_log_spread": 1.0,
    # The fewest usable templates a case needs before its own empirical spread is trusted; below it, fall back to the
    # persistence profile.
    "min_templates": 6,
    # A comparable's template is used only when its first hour is at least this many viewers (a ratio to a near-zero
    # first hour says nothing about a launch whose first hour is already large).
    "min_base": 1.0,
}


def epoch(iso: str) -> float:
    """The POSIX time of an ISO 8601 UTC instant such as ``2026-11-19T00:00:00Z``."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def hour_offsets(points, start: float, lo: int = -24, hi: int = 23) -> dict:
    """``{whole-hour offset from `start`: viewers}`` for the points at offsets `lo` to `hi`.

    A point at offset ``o`` covers the hour ``[start + o h, start + o + 1 h)``, which is the case's hour ``o + 1``: the
    day's own first hour is offset 0, the hour before it is -1, and hour 24 is offset 23.
    """
    out = {}
    for p in points:
        try:
            off = (epoch(p["from"]) - start) / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        near = round(off)
        if abs(off - near) > 1e-6 or not (lo <= near <= hi):
            continue
        viewers = p.get("viewers")
        if isinstance(viewers, (int, float)) and math.isfinite(viewers):
            out[near] = float(viewers)
    return out


def profile_features(ctx: dict):
    """The pre-checkpoint features of one launch's own context (`ctx`: offsets to viewers about its day start).

    They say what role hour 1 played in that launch, from data before the checkpoint alone: how large it was and whether
    the launch was already rising into it, sliding out of the evening before, or crossing its unlock inside the hour.
    """
    v1 = ctx.get(0)
    if v1 is None or v1 <= 0:
        return None
    feats = {"scale": math.log(max(1.0, v1))}
    v0 = ctx.get(-1)
    if v0 is not None and v0 > 0:
        feats["rel0"] = math.log(v1) - math.log(v0)
    prev = [ctx[o] for o in range(-6, 0) if ctx.get(o) is not None and ctx[o] > 0]
    if prev:
        feats["relprev"] = math.log(v1) - math.log(sum(prev) / len(prev))
    return feats


def weight(feats_a: dict, feats_b: dict, cfg: dict) -> float:
    """The kernel weight of a template `feats_b` for a target `feats_a` (each dimension skipped when either lacks it)."""
    total = 0.0
    for key in ("scale", "rel0", "relprev"):
        bw = cfg.get("bw_" + key, 0.0)
        if bw <= 0 or key not in feats_a or key not in feats_b:
            continue
        total += ((feats_a[key] - feats_b[key]) / bw) ** 2
    return math.exp(-0.5 * total)


def comparable_contexts(comparables: list) -> list:
    """Each comparable's own hourly context, aligned to its own day start (the parse is done once per comparable)."""
    out = []
    for comp in comparables:
        try:
            cstart = epoch(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            out.append(None)
            continue
        out.append(hour_offsets(comp.get("hourly") or [], cstart, -24, 23))
    return out


def templates(comparables: list, target_day: float, feats_target: dict, cfg: dict, contexts=None) -> list:
    """Every usable comparable as ``(weight, {hour: ratio to its own hour 1})``, in the case's own order.

    `contexts` is comparable_contexts(comparables) when the caller has it already (tools/eval.py reuses it across
    folds). The leakage guard drops any template whose own window reaches the target's launch day: no template may
    carry the day it is predicting.
    """
    if feats_target is None:
        return []
    if contexts is None:
        contexts = comparable_contexts(comparables)
    rows = []
    # The target's scored window: hours 1 to 24 of its own day, offsets 0 to 23. A template reaching into it knows the
    # answer, and a template of the very same day is the target itself.
    day_lo, day_hi = target_day, target_day + 24 * 3600.0
    for comp, ctx in zip(comparables, contexts):
        if ctx is None or not ctx:
            continue
        if any(day_lo <= t < day_hi for t in horizon_times(comp)) or abs(target_day - epoch(comp["day_start"])) < 1.0:
            continue
        feats = profile_features(ctx)
        if feats is None:
            continue
        base = ctx[0]
        if base < cfg.get("min_base", 1.0):
            continue
        ratios = {h: ctx[h - 1] / base for h in range(2, 25) if ctx.get(h - 1) is not None and ctx[h - 1] > 0}
        if len(ratios) < 6:
            continue
        rows.append((weight(feats_target, feats, cfg), ratios))
    return rows


def horizon_times(comp) -> list:
    """The instants a comparable's own points cover, used only for the overlap (leakage) guard."""
    out = []
    for p in comp.get("hourly") or []:
        try:
            out.append(epoch(p["from"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def target_context(case: dict) -> dict:
    """The target's own context: its history_hourly and its hour-1 row, which reach only our checkpoint (offset 0)."""
    return pre_context(case["day_start"], list(case.get("history_hourly") or []), case.get("known") or [])


def pre_context(day_start: str, points: list, known: list = ()) -> dict:
    """A launch's context before the checkpoint: the points at offsets -24 to 0 about `day_start`, hour 1 included."""
    merged = list(points)
    for row in known:
        if row.get("hour") == 1 and row.get("from"):
            merged.append({"from": row["from"], "viewers": row.get("viewers")})
    return hour_offsets(merged, epoch(day_start), -24, 0)


def weighted_quantiles(pairs: list, tau: float, method: str = "step") -> float:
    """The weighted empirical `tau` quantile of ``pairs`` (value, weight).

    ``step``: the smallest value whose cumulative weight reaches ``tau`` of the total, the plain weighted order
    statistic. ``interp``: the same CDF read as a piecewise-linear function, each value plotted at its mid-rank
    ``(C_i - w_i / 2) / W`` and interpolated - less jumpy at the extreme levels, where a plain order statistic of a
    few dozen launches is one launch away from moving anywhere.
    """
    usable = [(v, w) for v, w in pairs if w > 0.0 and math.isfinite(v)]
    if not usable:
        return 0.0
    usable.sort(key=lambda vw: vw[0])
    total = sum(w for _, w in usable)
    if total <= 0.0:
        return usable[0][0]
    target = tau * total
    if method != "interp" or len(usable) < 3:
        cum = 0.0
        for value, w in usable:
            cum += w
            if cum >= target:
                return value
        return usable[-1][0]
    cum, ranks = 0.0, []
    for value, w in usable:
        cum += w
        ranks.append(((cum - w / 2.0) / total, value))
    if tau <= ranks[0][0]:
        return ranks[0][1]
    for i in range(1, len(ranks)):
        p1, v1 = ranks[i - 1]
        p2, v2 = ranks[i]
        if tau <= p2:
            if p2 <= p1:
                return v2
            return v1 + (v2 - v1) * (tau - p1) / (p2 - p1)
    return ranks[-1][1]


def ess(weights: list) -> float:
    """Kish's effective sample size of a weight list."""
    total = sum(weights)
    squares = sum(w * w for w in weights)
    return (total * total / squares) if squares > 0 else 0.0


def floored_weights(pairs: list, floor: float) -> list:
    """`pairs` with a constant added to every weight, just enough to lift the effective sample size to `floor`."""
    if floor <= 0 or len(pairs) < 2 or ess([w for _, w in pairs]) >= floor:
        return pairs
    lo, hi = 0.0, 1.0
    top = max(w for _, w in pairs)
    while ess([w + hi for _, w in pairs]) < floor and hi < 1e6 * (top + 1.0):
        hi *= 4.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if ess([w + mid for _, w in pairs]) >= floor:
            hi = mid
        else:
            lo = mid
    return [(v, w + hi) for v, w in pairs]


def widen(curve: list, minimum: float) -> list:
    """Widen a five-quantile log curve about its median until its 5% to 95% span reaches `minimum` nats."""
    span = curve[4] - curve[0]
    if minimum <= 0 or span >= minimum:
        return curve
    mid = curve[2]
    factor = minimum / span if span > 0 else 1.0
    return [mid + (x - mid) * factor for x in curve]


def forecast_from(target_ctx: dict, comparables: list, p1: float, cfg: dict = None, contexts=None, target_day: float = None):
    """The 23 rows from a target context, its comparables, and hour 1's print (tools/eval.py calls this per fold)."""
    cfg = dict(CONFIG if cfg is None else cfg)
    feats_target = profile_features(target_ctx)
    rows = templates(comparables, target_day if target_day is not None else 0.0, feats_target, cfg, contexts)
    usable = [(w, r) for w, r in rows if w > 0.0]
    if len(usable) < cfg.get("min_templates", 6):
        # Too few templates to trust a spread of its own: the persistence profile, q_tau = p1 * exp(z_tau * 0.15 sqrt(h-1)).
        quantiles = [[z * BASELINE_SIGMA * math.sqrt(h - 1) for z in Z] for h in range(2, 25)]
    else:
        curves = []
        for h in range(2, 25):
            pairs = [(math.log(r[h]), w) for w, r in usable if h in r]
            if len(pairs) < 3:
                curves.append([z * BASELINE_SIGMA * math.sqrt(h - 1) for z in Z])
                continue
            pairs = floored_weights(pairs, cfg.get("ess_floor", 0.0))
            q = [weighted_quantiles(pairs, t, cfg.get("qmethod", "step")) for t in LEVELS]
            mid = q[2]
            # A power on the deviations from the median: the empirical spread of a few dozen launches is noisy.
            alpha = cfg.get("alpha", 1.0)
            curves.append([mid + alpha * (x - mid) for x in q])
        quantiles = _smooth(curves, cfg.get("smooth_k", 1))
        span = cfg.get("min_log_spread", 0.0)
        if span > 0:
            quantiles = [widen(c, span) for c in quantiles]
    return _rows_from_quantiles(p1, quantiles)


def forecast_hours(case: dict, cfg: dict = None):
    """The 23 rows for a case: hour, five quantiles, built from hour 1 and the case's comparables."""
    first = next((row for row in case.get("known") or [] if row.get("hour") == 1), None)
    p1 = max(VIEWER_MIN, float(first["viewers"]) if first and isinstance(first.get("viewers"), (int, float)) else 1.0)
    return forecast_from(target_context(case), case.get("comparables") or [], p1, cfg, target_day=epoch(case["day_start"]))


def _smooth(curves: list, k: int):
    """A light [1, k, 1] geometric smoothing over hour of each level's log-quantile curve."""
    if k <= 0 or len(curves) <= 2:
        return curves
    out = []
    for i in range(len(curves)):
        acc = []
        for level in range(5):
            total, weight_sum = 0.0, 0.0
            for j in range(max(0, i - 1), min(len(curves), i + 2)):
                w = 1.0 if j == i else float(k)
                total += w * curves[j][level]
                weight_sum += w
            acc.append(total / weight_sum)
        out.append(acc)
    return out


def _rows_from_quantiles(p1: float, quantiles: list):
    """The output rows from the 23 log-ratio quantile rows: clamp, order, and scale by hour 1's print."""
    rows = []
    for i, qi in enumerate(quantiles):
        hour = i + 2
        # Clamp each quantile into the contract's range before ordering, so the five stay sorted and finite.
        vals = [min(VIEWER_MAX, max(VIEWER_MIN, p1 * math.exp(x))) for x in qi]
        for j in range(1, 5):
            if vals[j] < vals[j - 1]:
                vals[j] = vals[j - 1]
        rows.append({"hour": hour, "q": [round(v, 6) for v in vals]})
    return rows


def forecast_case(case: dict, cfg: dict = None) -> dict:
    return {"hours": forecast_hours(case, cfg)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    sys.exit(main())
