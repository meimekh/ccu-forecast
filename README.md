# GTA VI launch-day Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/meimekh/ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/meimekh/ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day,
2026-11-19 (UTC), after the first: for hours 2 to 24, five quantiles of that hour's concurrent
viewers. Written for the make.fun bounty
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01), $250 in USDC
over twenty places.

`main.py` is Python standard library only, needs no network, and is deterministic. It reads the
case file the grader mounts, learns the shape of a launch hour by hour from the case's own
comparables (past big console launches, weighted by how closely each one's unlock structure matches
the case's), and writes five quantiles per hour in log space, pinned to the case's own first-hour
count.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<the kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs it as `python3 /bot/main.py` with one case in `/input` and a writable `/output`,
twice, both runs needing byte-identical forecasts.
