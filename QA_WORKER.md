# QA_WORKER.md - the ccu-forecast entry, what was actually run, and what it returned

Worker task: `urgent-forecast` in `C:/script/x/portfolio-workers.json`, deadline 2026-11-19T01:00Z.
Repository: `C:/script/x/entries/ccu-forecast`. Kit read: `C:/script/x/kits/gta6-twitch-ccu-forecast-01/gta6-twitch-ccu-forecast-01-kit`
(`RULES.txt`, `input/p00..p03.in`, `harness/harness-src.tar.gz` for `grader/house-forecast-twitch-ccu.ts`).
Everything below was run in this directory on Windows Git Bash with Python 3.11.9. No network was used once the kit was
read. No submission, signing, push or account action was taken; the commit is local.

## 1. The deliverable

`main.py` at the repository root, Python standard library only, deterministic, no network, no case-name or case-id
lookup. It reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR`
(else `/output`): `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2 to 24.

The model, in one paragraph: hour 1's print `p1` scales everything; every comparable launch supplies its release-day
window as ratios to its own first hour; the comparables are weighted by a Gaussian kernel on two features measured
*only before the checkpoint* (`scale`, the log of hour 1's viewers, and `rel0`, hour 1 against the hour before it); each
hour's five quantiles are the weighted empirical quantiles, in log space, of those ratios; a power `alpha` and a floor on
the 5%-95% log span calibrate the spread; a light smoothing over h damps single-hour noise. A comparable whose own
window reaches the target's launch day is dropped (leakage guard), and a case with fewer than six usable templates falls
back to the house baseline's persistence profile.

## 2. What was verified, and how

### 2.1 The scorer, against RULES.txt's own worked example

`python tools/eval.py --check`

```
scorer self-check: loss=0.112545 (want 0.112545)  typical_miss=19.52% (want 19.52%)  coverage=1.00 (want 1.00)  -> PASS
invalid-case self-check: loss=0.360659 (want 0.360659)  -> PASS
```

The harness scores with `2 x max(tau x d, (tau - 1) x d)`, `d = ln(y) - ln(q)`, `y` floored at 1, so the numbers below
are the rules' own measure, not an approximation of it.

### 2.2 Validation: leave-one-launch-out over the kit's comparables

The kit ships the four practice cases as **inputs only**. Their answers - hours 2..24 - are the historical source's
numbers held by the server's scorer, and RULES.txt says the scorer and its reference data stay on the server. They are
not in the kit, so **a true practice pinball loss cannot be computed locally, and none is claimed here.**

What the kit does hold is enough for the same test the entry must survive on h00. `tools/eval.py` holds out each
comparable in turn: the held-out launch's own pre-checkpoint context becomes the case, its real hours 2..24 become the
answer, and the model is refitted on the other comparables of the same case - the held-out launch never enters its own
training pool, and its own answer is never read. 37 of the 38 comparables have a first hour and at least 18 real later
hours; the 38th (the Grand Theft Auto VI category's own pre-release history) has no launch day and is no template.

Three pools are reported, and the house baseline is scored on exactly the same items:

* `all` - all 37 folds, 851 scored hours;
* `big` - the 16 folds whose hour 1 is 100,000 viewers or more (GTA VI's hour 1 cannot be small: the category's own
  pre-release levels already ran to hundreds of thousands);
* `near` - the 21 folds whose pre-checkpoint features resemble the four practice launches', which are the only launches
  in the kit built exactly like h00 (console game, rolling midnight-local unlock, its own UTC day, the same source).
  Their answers are unknown but their features are not, so this pool is the closest local stand-in for a practice grade.
  Note the anchor: the house baseline scores **0.371029** on this pool, against the **about 0.387** RULES.txt reports
  for the same baseline on the real practice cases.

Command: `python tools/eval.py` (about 5 minutes: 762 configurations refitted over all folds). Full log:
`tools/eval-run.txt`; machine-readable: `tools/eval-results.json`.

### 2.3 The configuration search, and its real numbers

Three passes, each ranked by the mean loss over the three pools (`obj`), everything refit per fold:

```
pass 1: 360 configurations in 135s      obj       all       big      near  bw_scale bw_rel0  alpha smooth
                                  0.297195  0.403891  0.216164  0.271532     0.90    0.00   1.00      1
                                  0.297264  0.401616  0.215237  0.274938     0.90    0.60   1.00      1
                                  0.297553  0.404559  0.216240  0.271860     1.25    0.60   1.00      1

pass 2: 162 configurations in  76s      obj       all       big      near  bw_scale bw_rel0 bw_relprev alpha smooth ess spread
                                  0.296790  0.401124  0.214579  0.274668     0.90    0.60       0.00  1.00      1  0.0   1.00
                                  0.296968  0.403567  0.215928  0.271410     0.90    0.00       0.00  1.00      1  0.0   1.00
                                  0.296992  0.403352  0.216453  0.271170     1.25    0.60       0.00  0.85      1  0.0   1.00

pass 3: 240 configurations in 100s      obj       all       big      near  qmethod alpha  spread smooth
                                  0.294615  0.397406  0.213259  0.273182   interp  1.00    1.00      1
                                  0.295069  0.397632  0.213910  0.273666   interp  1.00    1.00      2
                                  0.295287  0.397794  0.214803  0.273264   interp  1.00    1.50      1
```

Findings that drove the revisions, rather than cosmetics:

* The kernel width on `scale` and `rel0` is worth a lot: with every comparable weighted equally (`bw_scale` 0) the model
  is dragged down by launches whose first UTC hour is pre-unlock and whose day then grows 100x-2000x, which is not what a
  launch whose first hour is already large does. Its best leader still used a width of 0.9 to 1.25.
* `bw_relprev` (hour 1 against the six hours before it) never beat the other two; it is computed and available but its
  best width is 0, so the pinned model does not use it.
* Spread calibration was real: `alpha` 1.15 and 0.85 both lost to 1.0 once the 5%-95% floor existed, and the floor
  itself (`min_log_spread` 1.0 nats against 0.0, 0.6, 1.5, 2.0, 2.5) is what lifted the 90% coverage from well under
  0.9 to about 0.89 without giving away loss.
* Reading the weighted CDF as a line (`qmethod` "interp") instead of taking a plain weighted order statistic improved
  all three pools (0.296790 -> 0.294615 on `obj`, and 0.401124 -> 0.397406 on `all`): with a few dozen launches the
  extreme levels are one launch away from jumping.
* An effective-sample-size floor on the weights (`ess_floor`) never earned anything here, because the kernel is wide
  enough that no fold collapses onto one template; it is kept in the code, off (0.0) as selected.

### 2.4 The pinned configuration, scored

```
chosen: {"bw_scale": 0.9, "bw_rel0": 0.6, "bw_relprev": 0.0, "alpha": 1.0, "smooth_k": 1,
         "ess_floor": 0.0, "min_log_spread": 1.0, "min_templates": 6, "min_base": 1.0, "qmethod": "interp"}

  all   pool: model 0.397406 vs baseline 0.929650  (-0.532245, 4255 items, 90% coverage 0.866)
  big   pool: model 0.213259 vs baseline 0.261137  (-0.047878, 1840 items, 90% coverage 0.894)
  near  pool: model 0.273182 vs baseline 0.371029  (-0.097847, 2415 items, 90% coverage 0.886)
```

Selected folds (the whole 37-row table is in `tools/eval-run.txt`); `loss` is the model's, `base` the baseline's on the
same items, `miss` the typical miss in percent, `cov` the 90% coverage:

```
held-out launch                          hour1     loss     base     miss     base    cov
ELDEN RING                              875155   0.1026   0.2108    16.2%    39.2%   1.00
Hogwarts Legacy                         387358   0.1305   0.2881    14.6%    52.8%   1.00
Cyberpunk 2077                          851999   0.1401   0.2299    24.9%    41.5%   0.91
Diablo IV                               533455   0.1661   0.1254    31.9%    13.5%   1.00
The Legend of Zelda: Tears of the Kingdom 129948  0.2482   0.3087    56.1%    70.5%   1.00
Monster Hunter Wilds                    152314   0.4456   0.4273   104.0%   112.0%   0.52
Starfield                               156828   0.3728   0.2909    85.6%    65.3%   0.48
Resident Evil Requiem                    46345   1.0977   1.2729   304.4%   404.3%   0.17
Black Myth: Wukong                        4941   1.7705   3.4340   933.4%  4339.3%   0.65
Hollow Knight: Silksong                    164   0.8364   3.9411   222.9%  7051.7%   0.52
```

Read honestly: the model wins every pool on average, and wins by a wide margin on the big launches whose first hour is
large (the shape h00 will have). It still loses to the baseline on a few individual folds - Diablo IV, Borderlands 4,
MH Wilds, Starfield - and it is bad on the class of launch whose first UTC hour is nearly empty and whose day then grows
by two orders of magnitude. Those launches are the reason the `all` pool's baseline is as high as 0.930.

### 2.5 Contract, determinism, timing, edge cases, leakage

`python tools/validate.py` (full output reproduced here because every line is a check):

```
practice run over 4 cases: 0.35s, then 0.31s; byte-identical across runs: True
  p00.out      1914 bytes  OK
  p01.out      1970 bytes  OK
  p02.out      2010 bytes  OK
  p03.out      1969 bytes  OK
single-case cold run (p02, 1.0 MB case): 0.225s  (budget 8s)
edge cases: 0.186s
  no_comparables  hour-2 band [78133.600312, 127985.9108]  OK
  zero_hour_one   hour-2 band [1.0, 1.279859]  OK
  huge_hour_one   hour-2 band [1000000000.0, 1000000000.0]  OK
  empty_history   hour-2 band [195334.000779, 319964.777001]  OK
  h00_like        hour-2 band [1081379.597357, 2939494.509161]  OK
  no-comparables fallback == the house baseline (worst deviation 0.00e+00): True
  huge_hour_one max quantile: 1000000000 (clamped at 1e9)
```

* Contract: regular files, 1.9-2.0 kB (limit 65,536), one key `hours`, 23 rows, hours 2..24 once each, each row exactly
  `{hour, q}` with five finite numbers in (0, 1e9] in non-decreasing order - checked for both the four practice cases
  and six hand-built edge cases.
* Determinism: two runs of the same input directory produced byte-identical files. Nothing in the model is random; the
  search above refits the same inputs and reproduces its own leaders.
* Timing: 0.31-0.35 s for all four practice cases together; **0.199-0.249 s cold for a single 1.0 MB case** as the grader
  runs it (one case per sandbox), against the 8 s budget.
* Fallback: a case with no comparables reproduces the house baseline's arithmetic exactly (worst deviation 0.00e+00).
  A zero first hour gives `p1 = 1`; a first hour of 5e9 is clamped to 1e9.
* Leakage guard: the h00-shaped case was run twice, once with a "comparable" that carries the target's own launch day
  and a deliberately wild hours 2..24, once without. The two forecasts are byte-identical, so the target's own day is
  provably not used.

### 2.6 The h00-shaped case (the situation the final actually poses)

A first hour of 2,000,000 viewers - above every comparable's, ELDEN RING's 875,155 included - sliding out of a rising
evening, with the practice cases' own comparable list:

```
median ratio:  0.88 0.82 0.74 0.68 0.65 0.60 0.55 0.52 0.51 0.52 0.55 0.62 0.69 0.78 0.87 0.94 0.97 0.95 0.94 0.92 0.88 0.82 0.80
5%-95% span :  1.00 ... every hour exactly the 1.0-nat floor
37 templates, effective number 5.1; heaviest: ELDEN RING 0.30, Cyberpunk 2077 0.26, Diablo IV 0.15, Hogwarts Legacy 0.09
raw weighted empirical 5%-95% span, before the floor: hour 2 0.56, hour 6 0.51, hour 10 0.78, hour 14 0.59,
                                                      hour 18 0.45, hour 22 0.38, hour 24 0.77 nats
```

Two things to note, both real:

* The kernel does not collapse: 37 templates, an effective number of 5.1, and the heaviest are the launches whose unlock
  structure matches GTA VI's (console rolling midnight-local, US east coast at 05:00Z, Europe the evening before) - not
  the explosive pre-unlock launches.
* For a case this far outside the comparable range, the raw weighted empirical span is 0.38-0.78 nats and the 1.0-nat
  floor binds at every hour. The interval h00 would get is therefore the floor, not the raw empirical spread. That is
  what the search chose on the held-out folds (1.0 beat 1.5 and 2.0 by 0.0007 of loss while coverage stayed near 0.89),
  but it is a floor doing the work, and it is worth knowing.

## 3. Files created or changed

```
main.py                 the entry (new)
README.md               minimal (new)
QA_WORKER.md            this report (new)
LICENSE                 unchanged, the starter's MIT
tools/eval.py           leave-one-launch-out harness, scorer self-check, two-stage-then-three-stage search (new)
tools/validate.py       contract, determinism, timing, edge cases, h00-shaped case, leakage guard (new)
tools/explore.py        scratch reader for the kit's cases and comparables (new)
tools/eval-run.txt      the search's raw output, as run (new)
tools/eval-results.json machine-readable search and per-fold results (new)
```

Nothing outside this repository was written, and the kit under `C:/script/x/kits/...` was only read. A temporary copy of
`harness/harness-src.tar.gz` extracted into `.tmp/` (to read `grader/house-forecast-twitch-ccu.ts`) was deleted before
the commit, and `.tmp/` is not tracked.

## 4. What was NOT verified, and the limitations

1. **No practice-case pinball loss, and no final score.** The practice answers are not in the kit - they are the
   server's - so the only real numbers here are the leave-one-launch-out ones in section 2.4. Nothing on the actual
   h00 case, and nothing from the real grader, has been observed. No claim is made about where this entry would rank.
2. **Configuration selected on the reported folds.** The pinned parameters come from the same held-out folds whose
   losses are reported, so those numbers are mildly optimistic. Nothing was fitted on the launch being scored, and the
   pools are genuine out-of-sample folds, but the selection is not itself out-of-sample.
3. **The measure differs from the final's.** Practice cases and comparables are the source's hourly average per hour;
   the final scores the mean of the minute reads. RULES.txt flags the difference and it is not removable here.
4. **Python 3.13 not available locally.** The entry was run with Python 3.11.9 under `python -P -s` (the grader's flags)
   on 3.11.9; the grader uses python:3.13-slim. The code uses nothing newer than 3.8-era standard library, but the
   3.13 container itself was not exercised.
5. **`streams` and `categories_hourly` are ignored.** They are null in every practice case, so no honest way exists to
   learn what they say about a launch's first hour; h00 will carry them and the model does not read them. If they hold
   the top-20 streams at the checkpoint, a model that knew their relation to the day's later hours might do better -
   untested.
6. **No spread asymmetry.** The interval is symmetric in log space about the median. The empirical log ratios are
   roughly symmetric over these launches, but per-hour skew was not fitted.
7. **One scored case.** The final is a single case, so the variance of the result is much larger than the mean loss
   over 37 held-out launches suggests. Nothing here can shrink that.

## 5. Reproducing

```
python tools/eval.py --check      # the scorer against RULES.txt's worked example
python tools/eval.py --features   # the pre-checkpoint features of every comparable and of the four practice cases
python tools/eval.py              # the search and the leave-one-launch-out report (~5 min)
python tools/validate.py          # contract, determinism, timing, edge cases, leakage
mkdir -p out && MAKEFUN_INPUT_DIR=<the kit>/input MAKEFUN_OUTPUT_DIR=out python main.py
```
