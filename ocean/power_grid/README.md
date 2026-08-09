# Power Grid

This environment trains with a DC adaptation of the MATPOWER IEEE 14-bus case and can replay
the same policy through a busbar-aware AC evaluation solver. It has 20 switchable
branches, two independently configurable busbars, and a controllable bus coupler at each of 14
substations. A closed coupler merges the two busbars into one electrical node; opening it restores
their independent topology. The implementation
is deliberately split into layers:

- `power_grid_solver.[ch]`: dependency-free grid data, topology actions and validation, DC power
  flow, loading, and operating profiles.
- `power_grid_ac.[ch]`: Newton-Raphson AC power flow, reactive limits, voltage checks, losses,
  MVA loading, and thermal-protection support.
- `power_grid_scenarios_data.h`: the generated, compile-time historical scenario cache used by
  `power_grid.h`.
- `build_offline_scenarios.py`: the network-enabled offline data preparation tool. It is never
  imported or called by training.
- `power_grid.h`: the 221-observation, 91-action environment, episode logic, and Raylib renderer.
- `binding.c`: vectorized PufferLib registration.

## Physics represented

The solver balances active generation and demand with the generator at bus 1 as the slack. It fixes
that busbar's phase angle to zero, solves the nodal susceptance system, and computes every closed
branch flow from its phase-angle difference, reactance, and transformer tap. Consequently:

- Kirchhoff's current law holds at each individual busbar: injection equals net outgoing flow.
- Kirchhoff's voltage law holds around connected cycles through consistent busbar phase angles.
- Open branches carry zero power.
- Loads and generators must remain in the single component containing the slack generator.

AC evaluation uses MATPOWER resistance, line charging, transformer taps, bus-9 shunt, generator
voltage setpoints and Q limits. It calculates voltage magnitude/angle, P/Q branch-end flows,
losses, and loading from the larger apparent power at either branch end. Synthetic branch limits
remain scenario parameters because IEEE-14 supplies no usable facility ratings. Scheduled active
generator-limit violations are logged as scenario-data faults rather than blamed on topology. The
AC 1-2 rating is 160 MVA instead of the 155 MW DC proxy so nominal reactive flow preserves the
required safe initial state; all remaining synthetic rating numbers are shared.

AC-mode inverse-time protection accumulates squared overload exposure. Sustained 200%
loading trips a line in one step, 150% in four, and 120% in about 25; loading at or below 100%
cools it. A tripped line is locked out for the episode and AC power flow is recalculated. This is
a thermal proxy, not a conductor-temperature or manufacturer relay model.

Episode logs use `perf` for the fraction of overload-free steps, forced to zero after a
catastrophic failure. `score` is the fraction of actions that were no-ops, including in failed
episodes. Hyperparameter sweeps maximize `perf`.

Episodes contain 12 six-step operating periods. By default, reset makes one reproducible RNG draw:
90% of episodes use one complete day from the compile-time historical cache, while 10% retain the
deliberately congested synthetic curriculum. Set `offline_scenario_probability` anywhere from 0 to
1 to control that mix. A synthetic episode uniformly selects one of stress profiles `P1` through
`P14`, then smoothly ramps from nominal `P0` to that operating point and back down over its twelve
periods. This keeps changes temporally related while retaining profile coverage across episodes.
Scenario identity is not observed because injections, flows, and weather-adjusted loading expose
the actionable current condition.

## Offline historical domain randomization

The checked-in cache contains 365 training days from 2019 and a disjoint 366-day validation pool
from 2020. Each day combines:

- actual hourly Alberta Internal Load, wind generation, and solar generation published by the
  [Alberta Electric System Operator](https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-ail-smp-wind-generation-and-solar-generation-data-for-2016-to-2020/);
- Edmonton-area hourly ERA5 temperature, 10 m wind speed, and shortwave radiation from
  the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

The builder averages those correlated local-time observations into the episode's twelve two-hour
periods. AESO load is normalized to the IEEE-14 total demand, and annual renewable capacity factors
drive the bus-3 solar and bus-6 wind generators. Ambient temperature, wind speed, and irradiance
feed the steady-state IEEE 738 heat balance, normalized to median 2019 weather and capped between
0.90x and 1.35x because the underlying IEEE-14 ratings are synthetic. The model uses the published
26/7 Drake ACSR example parameters, Edmonton elevation, perpendicular wind, and measured
irradiance as effective incident flux. Real deployment still requires conductor and route geometry,
wind attack angle, calibrated facility ratings, and transient conductor-temperature or sag limits.

`power_grid_scenarios_data.h` records SHA-256 hashes of both downloaded inputs. Rebuild it with:

```sh
python ocean/power_grid/build_offline_scenarios.py
```

Downloads are cached under the ignored `.scenario_sources` directory. Pass `--refresh` to fetch
new copies. The resulting header is compiled into the environment. A historical reset only selects
a pointer into this read-only table, so it performs no API calls, file reads, parsing, allocation,
or per-environment copy. Use `offline_scenario_validation = True` to sample only 2020; do not use
that pool for training if it is serving as held-out validation.

The observation exposes normalized weather, the adjusted rating scale, per-branch availability and
thermal stress, all 28 busbar voltage magnitudes, and generator reactive output. DC mode supplies
`1` for active busbars, `0` for inactive busbars, and zero reactive output, so the layout is
identical in DC and AC. This expansion is
incompatible with checkpoints trained on the older 144-value contract.

## Random event layer

An episode can independently layer one persistent line outage onto a historical day or synthetic
profile. `random_event_probability` controls the fraction of episodes that schedule the event. Its
line and later period are randomized, and the failed line remains locked open. Line availability is
already part of each line's observation, so the policy sees the outage without changing the layout.

Certification is intentionally minimal: from the normal topology, the outage must leave the DC
power flow solvable. It does not need to cause an overload, and no recovery is required or supplied.
The outage is exogenous and still applies if the agent has already changed topology; that combined
state may fail because only the normal topology is certified. Events use the same scheduling and
line-selection logic in DC and AC modes. The environment excludes the radial 7-8 branch; tests
verify every other single-line outage remains DC-solvable against every cached period and synthetic
profile. AC mode models the outage directly, but the DC certificate does not promise AC convergence
or voltage feasibility after it occurs.

The hot training path samples one period and line from a compact bit mask. Metrics report whether
the outage was reached and survived, full-demand service, line overload exposure, peak loading,
thermal stress, and trips.

No historical period or synthetic profile is required to have a safe topology within a bounded
number of actions. The six actions in each operating period are an agent response budget, not a
promise that recovery exists. Terminal failures are divided into mutually exclusive
`connectivity_failure` and `solver_failure` counters. Invalid actions or scenario data are treated
as solver failures because they should only indicate a bug or corrupted input during normal use.
`total_failure` is the exact sum. `random_events` separately reports applied outages, while
`event_failure` reports only an
immediate terminal failure from applying and re-solving a scheduled outage. It is a diagnostic tag,
not another category in the total.

## Tests

Run the dependency-free C checks directly through pytest:

```sh
uv run --with pytest pytest -q \
  tests/test_power_grid_solver.py tests/test_power_grid_scenarios.py
```

Build the CUDA PufferLib extension and standalone renderer with:

```sh
./build.sh power_grid
./build.sh power_grid --fast
```

Train with fast DC power flow (the default), then replay a checkpoint under AC physics:

```sh
puffer train power_grid
puffer eval power_grid --load-model-path latest --env.ac-power-flow True \
  --env.offline-scenario-validation True
```

AC voltage violations and thermal trips are reported as dedicated evaluation metrics; they do not
add terms to the reward. The AC `perf` security metric still treats a voltage-violating step as
unsafe. Replay never updates the policy.

For a headless quantitative comparison on recorded 2020 validation days under both solvers:

```sh
python ocean/power_grid/evaluate.py --checkpoint latest --rollouts 2
```

The comparison reports security, return, failure, switching, and event metrics.

Compare a trained checkpoint with six baselines spanning no-action, random, safe-random, and
greedy policies on the same held-out seed range:

```sh
python ocean/power_grid/benchmark.py --checkpoint latest --episodes 1024 \
  --physics ac --random-event-probability 1
```

The greedy policies score every immediately solvable candidate using the environment reward. One
searches only no-op and branch toggles; the other searches all 91 actions. Safe-random uniformly
samples from actions whose immediate topology remains solvable. These are diagnostic baselines,
not recovery guarantees or multi-step planning oracles.

The AC renderer shows voltage range, P/Q generation and demand, active/reactive losses, MVA line
loading, Q-limit conversions, P-limit data faults, Newton convergence, thermal stress, and trips.

The `--cpu` build exposes only the lightweight vector environment interface and is intended for
the PyTorch `--slowly` backend. Normal `puffer train power_grid` requires the CUDA build above,
which provides `create_pufferl`.
