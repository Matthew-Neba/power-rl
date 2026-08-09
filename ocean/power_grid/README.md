# Alberta-driven IEEE-118 topology control

This environment trains a topology-control policy with fast DC power flow and replays the same
policy under AC physics. The electrical network is the canonical MATPOWER IEEE 118-bus case; its
chronological demand, renewable, and weather trajectories come from Alberta data. It is an
Alberta-driven synthetic benchmark, not a reconstruction of the Alberta transmission network.

## Model

- 118 substations, represented by 236 independently configurable busbars.
- 186 switchable branches, 54 generators, 99 loads, and a 100 MVA base.
- 525 movable branch/generator/load terminals and 118 controllable ideal couplers.
- 830 discrete actions: no-op, branch toggles, terminal transfers, and coupler toggles.
- 1,985 observations covering line state/loading/stress, busbar assignments, couplers,
  injections, weather/rating state, voltage, and generator reactive output.

`build_ieee118_case.py` downloads and verifies canonical MATPOWER `case118.m`, places its slack
generator first, identifies the nine bridge branches excluded from randomized contingencies, and
generates `power_grid_ieee118_data.h`. The canonical case marks every branch rating as unlimited.
This benchmark therefore assigns documented synthetic voltage-class ratings: 225 MVA for
138/161 kV corridors and 550 MVA for 345 kV corridors and transformers. Rebuild with:

```sh
python ocean/power_grid/build_ieee118_case.py
```

The DC solver enforces active-power balance, KCL/KVL, zero flow on open branches, and connectivity
of every load and generator to the bus-69 slack component. The AC solver adds resistance, charging,
transformer taps, bus shunts, voltage/reactive limits, losses, worst-end MVA loading, and a simple
inverse-time thermal protection model. A 200% loading trips in one step, 150% in four, and 120% in
about 25; safe loading cools accumulated stress.

## Alberta trajectories

The checked-in scenario cache contains 365 training days from 2019 and a disjoint 366-day
validation pool from 2020. Each complete day combines:

- hourly Alberta Internal Load, wind, and solar generation from AESO;
- Edmonton-area ERA5 temperature, 10 m wind speed, and shortwave radiation from Open-Meteo.

The builder averages these into twelve correlated two-hour periods. Demand scales the 4,242 MW
IEEE-118 base load. Alberta renewable capacity factors drive 500 MW solar at bus 80 and 700 MW
wind at bus 89; canonical online generators redispatch while the slack retains approximately nine
percent of demand. Weather adjusts every synthetic branch rating from 0.90x to 1.35x using a
steady-state IEEE 738 calculation with one reference conductor. This does not supply real route,
conductor, sag, or transient-temperature data.

Training defaults to 90% complete 2019 days and 10% synthetic regional/renewable stress ramps.
Validation always selects 2020. Runtime uses a compiled read-only cache—no files, parsing,
allocation, Python, or network access in the environment hot path. Rebuild the cache with:

```sh
python ocean/power_grid/build_offline_scenarios.py
```

## Outages and objectives

Half of training episodes schedule three distinct persistent line outages at distinct later
periods. Candidate lines must not be bridges, and each selected three-line combination must leave
the normal topology connected. The outage count, rather than just its probability, is configurable
with `random_outage_count`. AC feasibility after arbitrary policy actions is intentionally evaluated,
not guaranteed by the DC connectivity certificate.

For a valid state:

```text
reward = - squared DC congestion - 0.001 per physical switch + 0.20 if overload-free
```

An invalid topology or failed solve returns `-5` and terminates. AC voltage violations and thermal
trips are evaluation metrics; they do not alter the DC training reward. `perf` is the secure-step
fraction and is forced to zero after catastrophic failure. Evaluation also reports demand served,
outage completion/survival, thermal trips, peak stress/loading, overload exposure, and switching.
Reward/episode return is intentionally omitted from evaluation reports.

## Validation, training, and benchmarks

```sh
uv run --with pytest pytest -q \
  tests/test_power_grid_solver.py tests/test_power_grid_scenarios.py
PATH="$PWD/.venv/bin:$PATH" ./build.sh power_grid
puffer train power_grid
```

Replay one checkpoint under matched held-out DC and AC trajectories:

```sh
python ocean/power_grid/evaluate.py --checkpoint latest --rollouts 2
```

Compare PPO with no-action, uniform-random, random-line, safe-random, greedy-line, and greedy-all
baselines on identical seeds:

```sh
python ocean/power_grid/benchmark.py --checkpoint latest --episodes 256 \
  --physics ac --random-event-probability 1 --random-outages 3
```

Greedy baselines use DC one-step candidate scoring even during final AC evaluation. This matches
the learned policy's training information and avoids granting greedy an expensive AC oracle.
Safe-random uses the same DC solvability screen. These are diagnostic baselines, not multi-step
planning oracles.

The compact renderer uses a deterministic 12-column layout because IEEE-118 has no geographic
coordinates. Branch color communicates loading, dotted gray branches are open, small upper/lower
markers show generator/load terminals, and the sidebar reports the hottest branches and AC state.
