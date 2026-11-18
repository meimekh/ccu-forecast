# ccu-forecast

A forecast of GTA VI's launch-day Twitch category viewers, hour by hour: for each of hours 2 to 24 of the UTC release
day, five quantiles of that hour's viewer count, on the levels 0.05, 0.25, 0.5, 0.75 and 0.95.

Python 3, standard library only, no network. `main.py` is the entry: it reads every `*.in` in `$MAKEFUN_INPUT_DIR`
(else `/input`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python main.py
```

The model: hour 1's print `p1` scales the answer, and each comparable launch's release-day window supplies a profile of
ratios to its own first hour. The comparables are weighted by how well their pre-checkpoint first hour played the same
role the target's does (size in log viewers, and whether hour 1 was rising, sliding or crossing its unlock), the
quantiles of the weighted log ratios are taken per hour, and the spread is calibrated. The details, the validation and
its measured numbers are in `QA_WORKER.md`.

Files:

* `main.py` - the entry.
* `tools/eval.py` - leave-one-launch-out validation against the kit's comparables (the pinned configuration search and
  the pinned numbers; writes `tools/eval-results.json`).
* `tools/validate.py` - output contract, determinism, timing and edge cases.
* `tools/explore.py` - a scratch reader for the kit's cases and comparables.
* `QA_WORKER.md` - what was run, what it returned, and what was not verified.
* `LICENSE` - MIT.
