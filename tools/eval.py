"""Leave-one-launch-out validation of main.py against the kit's comparables (dev tool, not part of the entry).

The kit ships the practice cases as *inputs* only: their answers (hours 2..24) live in the server's scorer, which the
rules say stays on the server, so a true practice pinball loss cannot be computed locally. What can be computed, on
data the kit does hold, is the same scored measure under the same protocol the entry must survive on h00: for every
comparable launch, hide it, refit the model on the other comparables of the same case, forecast its own hours 2..24 and
score against its real counts. That is a genuine out-of-sample test: no fold's training pool holds the launch it
predicts (the model's own overlap guard would drop it if it did), and the held-out hours are never read.

The house baseline (the kit's starter, q = p1 * exp(z * 0.15 sqrt(h-1))) is scored on exactly the same items.

Two pools are reported:
  all       every held-out comparable that has at least 18 of its 24 hours (37 folds here);
  big       the folds whose hour 1 is at least 100,000 viewers (the launches that look like the final: GTA VI's hour 1
            cannot be small - the category's own pre-release levels already ran to hundreds of thousands).
The configuration is chosen on the big pool's loss, with the all pool reported beside it so a choice that wins on one
pool by wrecking the other is visible. That selection is on the same held-out folds the loss is then reported on, so the
reported number is mildly optimistic; it is still out-of-sample (nothing is fitted on the launch being scored).

Usage:
    python tools/eval.py              # feature table, the two-pass search, the report (writes tools/eval-results.json)
    python tools/eval.py --check      # only the scorer self-check against RULES.txt's worked example
    python tools/eval.py --features   # only the feature table for the comparables and the practice cases
"""

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main as M  # noqa: E402  (the entry itself, imported the way the harness runs it)

# The search refits the model hundreds of times over the same comparables: parse each comparable's point times once.
_HORIZON = {}
_raw_horizon_times = M.horizon_times


def _memo_horizon_times(comp):
    key = comp.get("name")
    if key not in _HORIZON:
        _HORIZON[key] = _raw_horizon_times(comp)
    return _HORIZON[key]


M.horizon_times = _memo_horizon_times

KIT = Path(os.environ.get("KIT", "C:/script/x/kits/gta6-twitch-ccu-forecast-01/gta6-twitch-ccu-forecast-01-kit"))
LEVELS = M.LEVELS
Z = M.Z
BIG_POOL_VIEWERS = 100_000


# ----------------------------------------------------------------------------------------------------------- the score
def pinball(y: float, q: float, tau: float) -> float:
    """The rules' item loss: 2 x max(tau x d, (tau - 1) x d), d = ln(y) - ln(q), on the log scale."""
    d = math.log(y) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(rows: list, actual: dict) -> dict:
    """The pooled loss and the board's two informational numbers for one forecast over one launch."""
    items, abs_miss, covered, hours = [], [], 0, 0
    for row in rows:
        h = row["hour"]
        if h not in actual:
            continue
        y = max(1.0, float(actual[h]))
        hours += 1
        for tau, q in zip(LEVELS, row["q"]):
            items.append(pinball(y, q, tau))
        abs_miss.append(abs(math.log(y) - math.log(row["q"][2])))
        covered += 1 if row["q"][0] <= y <= row["q"][4] else 0
    if not items:
        return {"items": 0}
    return {
        "items": len(items),
        "loss": sum(items) / len(items),
        "typical_miss": 100.0 * (math.exp(sum(abs_miss) / len(abs_miss)) - 1.0),
        "coverage": covered / hours,
        "hours": hours,
    }


def baseline_rows(p1: float) -> list:
    """The house baseline (the kit's starter), which the entry must beat: q = p1 * exp(z * 0.15 sqrt(h-1))."""
    return [
        {"hour": h, "q": [max(1.0, p1 * math.exp(z * M.BASELINE_SIGMA * math.sqrt(h - 1))) for z in Z]}
        for h in range(2, 25)
    ]


def scorer_self_check() -> bool:
    """RULES.txt's worked example: two hours, both rows [600000, 800000, 900000, 1100000, 1500000], prints 1e6 and 7e5
    must give loss 0.112545, typical miss 19.52% and coverage 1."""
    rows = [{"hour": 2, "q": [600000, 800000, 900000, 1100000, 1500000]},
            {"hour": 3, "q": [600000, 800000, 900000, 1100000, 1500000]}]
    got = score(rows, {2: 1_000_000, 3: 700_000})
    ok = (abs(got["loss"] - 0.112545) < 1e-6 and abs(round(got["typical_miss"], 2) - 19.52) < 0.005
          and got["coverage"] == 1.0)
    print(f"scorer self-check: loss={got['loss']:.6f} (want 0.112545)  typical_miss={got['typical_miss']:.2f}% "
          f"(want 19.52%)  coverage={got['coverage']:.2f} (want 1.00)  -> {'PASS' if ok else 'FAIL'}")
    # The invalid-case rule: every quantile equals hour 1's print -> the mean absolute log error.
    bad = [{"hour": 2, "q": [1_200_000] * 5}, {"hour": 3, "q": [1_200_000] * 5}]
    got_bad = score(bad, {2: 1_000_000, 3: 700_000})
    want_bad = (abs(math.log(1_000_000) - math.log(1_200_000)) + abs(math.log(700_000) - math.log(1_200_000))) / 2
    print(f"invalid-case self-check: loss={got_bad['loss']:.6f} (want {want_bad:.6f})  -> "
          f"{'PASS' if abs(got_bad['loss'] - want_bad) < 1e-12 else 'FAIL'}")
    return ok and abs(got_bad["loss"] - want_bad) < 1e-12


# ----------------------------------------------------------------------------------------------------------- the folds
def load_practice(name: str = "p00") -> dict:
    with open(KIT / "input" / f"{name}.in", encoding="utf-8") as fh:
        return json.load(fh)


def pre_context_of(comp_or_case) -> dict:
    """A launch's own pre-checkpoint context: the 'hourly' points up to offset 0, or a case's history_hourly + known."""
    if "history_hourly" in comp_or_case:
        return M.target_context(comp_or_case)
    return M.pre_context(comp_or_case["day_start"], comp_or_case["hourly"])


def folds_from(case: dict) -> list:
    """One fold per comparable that can be a target: its own pre-checkpoint context, its real hours 2..24, and the other
    comparables of the same case as its training pool (the held-out launch itself never enters its own pool)."""
    comps = case["comparables"]
    contexts = M.comparable_contexts(comps)
    folds = []
    for i, comp in enumerate(comps):
        ctx = contexts[i]
        if not ctx or ctx.get(0) is None or ctx[0] <= 0:
            continue
        actual = {h: ctx[h - 1] for h in range(2, 25) if ctx.get(h - 1) is not None}
        if len(actual) < 18:  # too few real hours to be scored on
            continue
        keep = [j for j in range(len(comps)) if j != i]
        folds.append({
            "name": comp["name"],
            "day_start": comp["day_start"],
            "p1": ctx[0],
            "target_ctx": pre_context_of(comp),
            "training": [comps[j] for j in keep],
            "training_contexts": [contexts[j] for j in keep],
            "actual": actual,
        })
    return folds


def pool_loss(folds: list, cfg: dict, baseline: bool = False) -> dict:
    """The pooled loss of one configuration over `folds` (the baseline's when `baseline`)."""
    items, per_fold, hours = 0, {}, 0
    total = 0.0
    coverage = 0.0
    for fold in folds:
        rows = baseline_rows(fold["p1"]) if baseline else M.forecast_from(
            fold["target_ctx"], fold["training"], fold["p1"], cfg,
            contexts=fold["training_contexts"], target_day=M.epoch(fold["day_start"]))
        got = score(rows, fold["actual"])
        if not got["items"]:
            continue
        per_fold[fold["name"]] = got
        items += got["items"]
        total += got["loss"] * got["items"]
        hours += got["hours"]
        coverage += got["coverage"] * got["hours"]
    return {"items": items, "loss": total / items if items else float("nan"),
            "coverage": coverage / hours if hours else float("nan"), "hours": hours, "per_fold": per_fold}


def feature_table() -> None:
    """What each launch looks like before its checkpoint, and what each practice case looks like: the model's inputs."""
    case = load_practice("p00")
    print(f"{'launch':44s} {'hour1':>9s} {'scale':>7s} {'rel0':>7s} {'relprev':>8s}")
    for comp in case["comparables"]:
        ctx = pre_context_of(comp)
        feats = M.profile_features(ctx)
        if feats is None:
            print(f"{comp['name'][:44]:44s} {'-':>9s} (no hour-1 point: not a template)")
            continue
        print(f"{comp['name'][:44]:44s} {ctx[0]:9.0f} {feats['scale']:7.2f} {feats.get('rel0', float('nan')):7.2f} "
              f"{feats.get('relprev', float('nan')):8.2f}")
    for name in ("p00", "p01", "p02", "p03"):
        case = load_practice(name)
        ctx = pre_context_of(case)
        feats = M.profile_features(ctx)
        known = next(r for r in case["known"] if r["hour"] == 1)
        print(f"{'CASE ' + name + ' ' + case['game']['name']:44s} {known['viewers']:9.0f} "
              f"{feats['scale']:7.2f} {feats.get('rel0', float('nan')):7.2f} {feats.get('relprev', float('nan')):8.2f}")


def near_pool(folds: list, threshold: float = 0.35) -> list:
    """The folds that resemble the kit's four practice launches.

    Those four are the only launches in the kit built exactly like h00: a console game on a rolling midnight-local
    unlock, its own UTC release day, the same source. Their pre-checkpoint features are known even though their answers
    are not, so a fold close to them in feature space is the closest thing to a held-out practice case.
    """
    practice = []
    for name in ("p00", "p01", "p02", "p03"):
        feats = M.profile_features(pre_context_of(load_practice(name)))
        if feats:
            practice.append(feats)
    bw = {"bw_scale": 1.0, "bw_rel0": 0.5, "bw_relprev": 0.5}
    keep = []
    for fold in folds:
        feats = M.profile_features(fold["target_ctx"])
        if feats is None:
            continue
        if max(M.weight(feats, pf, bw) for pf in practice) >= threshold:
            keep.append(fold)
    return keep


def grid_search(folds: list, pools: dict) -> dict:
    """Pass 1 over the kernel widths and the spread calibration, pass 2 refining the leaders with the guards."""
    def evaluate(cfg):
        got = {name: pool_loss(pool, cfg)["loss"] for name, pool in pools.items()}
        got["cfg"] = cfg
        got["obj"] = sum(got[name] for name in pools) / len(pools)
        return got

    t0 = time.time()
    pass1 = []
    for bw_scale in (0.0, 0.6, 0.9, 1.25, 1.6, 2.5):
        for bw_rel0 in (0.0, 0.3, 0.6, 1.0):
            for alpha in (0.6, 0.75, 0.85, 1.0, 1.15):
                for smooth_k in (0, 1, 2):
                    pass1.append(evaluate({"bw_scale": bw_scale, "bw_rel0": bw_rel0, "bw_relprev": 0.0,
                                           "alpha": alpha, "smooth_k": smooth_k,
                                           "ess_floor": 0.0, "min_log_spread": 0.0,
                                           "min_templates": 6, "min_base": 1.0}))
    pass1.sort(key=lambda r: r["obj"])
    print(f"\npass 1: {len(pass1)} configurations in {time.time() - t0:.0f}s, ranked by the pools' mean loss")
    print(f"{'obj':>9s} {'all':>9s} {'big':>9s} {'near':>9s} {'bw_scale':>8s} {'bw_rel0':>7s} {'alpha':>6s} {'smooth':>6s}")
    for r in pass1[:8]:
        c = r["cfg"]
        print(f"{r['obj']:9.6f} {r['all']:9.6f} {r['big']:9.6f} {r['near']:9.6f} {c['bw_scale']:8.2f} {c['bw_rel0']:7.2f} "
              f"{c['alpha']:6.2f} {c['smooth_k']:6d}")

    t0 = time.time()
    pass2 = []
    for leader in pass1[:6]:
        for bw_relprev in (0.0, 0.5, 1.0):
            for ess_floor in (0.0, 4.0, 8.0):
                for min_log_spread in (0.0, 0.6, 1.0):
                    cfg = dict(leader["cfg"])
                    cfg.update({"bw_relprev": bw_relprev, "ess_floor": ess_floor, "min_log_spread": min_log_spread})
                    pass2.append(evaluate(cfg))
    pass2.sort(key=lambda r: r["obj"])
    print(f"\npass 2: {len(pass2)} configurations in {time.time() - t0:.0f}s, ranked by the pools' mean loss")
    print(f"{'obj':>9s} {'all':>9s} {'big':>9s} {'near':>9s} {'bw_scale':>8s} {'bw_rel0':>7s} {'bw_relprev':>10s} "
          f"{'alpha':>6s} {'smooth':>6s} {'ess':>5s} {'spread':>7s}")
    for r in pass2[:8]:
        c = r["cfg"]
        print(f"{r['obj']:9.6f} {r['all']:9.6f} {r['big']:9.6f} {r['near']:9.6f} {c['bw_scale']:8.2f} {c['bw_rel0']:7.2f} "
              f"{c['bw_relprev']:10.2f} {c['alpha']:6.2f} {c['smooth_k']:6d} {c['ess_floor']:5.1f} "
              f"{c['min_log_spread']:7.2f}")
    t0 = time.time()
    pass3 = []
    for leader in pass2[:5]:
        for qmethod in ("step", "interp"):
            for alpha in (1.0, 1.15, 1.3):
                for min_log_spread in (1.0, 1.5, 2.0, 2.5):
                    for smooth_k in (1, 2):
                        cfg = dict(leader["cfg"])
                        cfg.update({"qmethod": qmethod, "alpha": alpha, "min_log_spread": min_log_spread,
                                    "smooth_k": smooth_k})
                        pass3.append(evaluate(cfg))
    pass3.sort(key=lambda r: r["obj"])
    print(f"\npass 3: {len(pass3)} configurations in {time.time() - t0:.0f}s, ranked by the pools' mean loss")
    print(f"{'obj':>9s} {'all':>9s} {'big':>9s} {'near':>9s} {'qmethod':>8s} {'alpha':>6s} {'spread':>7s} {'smooth':>6s}")
    for r in pass3[:8]:
        c = r["cfg"]
        print(f"{r['obj']:9.6f} {r['all']:9.6f} {r['big']:9.6f} {r['near']:9.6f} {c['qmethod']:>8s} {c['alpha']:6.2f} "
              f"{c['min_log_spread']:7.2f} {c['smooth_k']:6d}")
    return {"pass1": pass1, "pass2": pass2, "pass3": pass3, "chosen": pass3[0]}


def report(folds: list, pools: dict, chosen: dict, baselines: dict) -> dict:
    """The chosen configuration's fold-by-fold numbers beside the baseline's, on every pool."""
    got_all = pool_loss(folds, chosen)
    print(f"\nchosen: {json.dumps(chosen)}")
    for name, pool in pools.items():
        got = pool_loss(pool, chosen)
        base = baselines[name]
        print(f"  {name:5s} pool: model {got['loss']:.6f} vs baseline {base['loss']:.6f} "
              f"({got['loss'] - base['loss']:+.6f}, {got['items']} items, 90% coverage {got['coverage']:.3f})")
    base_all = baselines["all"]
    print(f"\n{'held-out launch':44s} {'hour1':>9s} {'loss':>8s} {'base':>8s} {'miss':>8s} {'base':>8s} {'cov':>5s}")
    for fold in folds:
        m = got_all["per_fold"][fold["name"]]
        b = base_all["per_fold"][fold["name"]]
        print(f"{fold['name'][:44]:44s} {fold['p1']:9.0f} {m['loss']:8.4f} {b['loss']:8.4f} {m['typical_miss']:7.1f}% "
              f"{b['typical_miss']:7.1f}% {m['coverage']:5.2f}")
    return {"pools": {name: {k: v for k, v in pool_loss(pool, chosen).items() if k != "per_fold"}
                      for name, pool in pools.items()},
            "baselines": {name: base["loss"] for name, base in baselines.items()},
            "per_fold": got_all["per_fold"], "baseline_per_fold": baselines["all"]["per_fold"]}


def main() -> int:
    if "--check" in sys.argv:
        return 0 if scorer_self_check() else 1
    if "--features" in sys.argv:
        feature_table()
        return 0
    scorer_self_check()
    case = load_practice("p00")
    folds = folds_from(case)
    pools = {
        "all": folds,
        "big": [f for f in folds if f["p1"] >= BIG_POOL_VIEWERS],
        "near": near_pool(folds),
    }
    print(f"\n{len(folds)} leave-one-launch-out folds ({len(pools['big'])} with hour 1 >= {BIG_POOL_VIEWERS:,} "
          f"viewers, {len(pools['near'])} practice-like), {sum(len(f['actual']) for f in folds)} real scored hours:\n"
          f"  practice-like: {', '.join(f['name'] for f in pools['near'])}")
    baselines = {name: pool_loss(pool, None, baseline=True) for name, pool in pools.items()}
    for name, base in baselines.items():
        print(f"  house baseline on the {name:5s} pool: {base['loss']:.6f} ({base['items']} items)")
    search = grid_search(folds, pools)
    chosen = search["chosen"]["cfg"]
    body = report(folds, pools, chosen, baselines)
    out = {"folds": len(folds), "big_folds": len(pools["big"]), "near_folds": len(pools["near"]),
           "baseline": {name: base["loss"] for name, base in baselines.items()},
           "search": {"pass1": search["pass1"][:20], "pass2": search["pass2"][:20], "pass3": search["pass3"][:20],
                      "chosen": chosen},
           "report": body}
    (ROOT / "tools" / "eval-results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote tools/eval-results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
