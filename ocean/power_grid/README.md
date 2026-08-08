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
- `power_grid_baselines.[ch]`: do-nothing, seeded-random, one-step greedy, and bounded search
  controllers for validation without learning.
- `power_grid.h`: the 144-observation, 91-action environment, episode logic, and Raylib renderer.
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

Evaluation-only inverse-time protection accumulates squared overload exposure. Sustained 200%
loading trips a line in one step, 150% in four, and 120% in about 25; loading at or below 100%
cools it. A tripped line is locked out for the episode and AC power flow is recalculated. This is
a thermal proxy, not a conductor-temperature or manufacturer relay model.

Episode logs use `perf` for the fraction of overload-free steps, forced to zero after a
catastrophic failure. `score` is the fraction of actions that were no-ops, including in failed
episodes. Hyperparameter sweeps maximize `perf`.

Episodes contain 12 six-step operating periods. The first is always the safe nominal `P0`; the
other 11 are independent samples from stress profiles `P1` through `P14`, with replacement. The
sampling uses Puffer's per-environment RNG state, so a seed is reproducible and vector environments
do not share RNG state. Profile IDs are not added to observations because injections and line flows
already describe the current electrical condition.

The held-out `evaluation_scenarios` mode replaces random profiles with a repeatable 24-hour load,
synthetic bus-3 solar, and bus-6 wind trajectory. It also takes line 9-14 out for maintenance from
08:00 through 16:00. The headless report enables this for both DC and AC so only the solver physics
change between columns. These renewable injections are exogenous; curtailment remains out of scope.

## Tests

Run the dependency-free C checks directly through pytest:

```sh
uv run --with pytest pytest -q tests/test_power_grid_solver.py
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
  --env.evaluation-scenarios True
```

For a headless quantitative comparison of the same checkpoint under both solvers:

```sh
python ocean/power_grid/evaluate.py --checkpoint latest --rollouts 2
```

The AC renderer shows voltage range, P/Q generation and demand, active/reactive losses, MVA line
loading, Q-limit conversions, P-limit data faults, Newton convergence, thermal stress, and trips.

The `--cpu` build exposes only the lightweight vector environment interface and is intended for
the PyTorch `--slowly` backend. Normal `puffer train power_grid` requires the CUDA build above,
which provides `create_pufferl`.
