# Power Grid

This environment is a DC adaptation of the MATPOWER IEEE 14-bus case. It has 20 switchable
branches, two independently configurable busbars, and a controllable bus coupler at each of 14
substations. A closed coupler merges the two busbars into one electrical node; opening it restores
their independent topology. The implementation
is deliberately split into layers:

- `power_grid_solver.[ch]`: dependency-free grid data, topology actions and validation, DC power
  flow, loading, and operating profiles.
- `power_grid_baselines.[ch]`: do-nothing, seeded-random, one-step greedy, and bounded search
  controllers for validation without learning.
- `power_grid.h`: the 144-observation, 91-action PufferLib environment contract and renderer.
- `binding.c`: vectorized PufferLib registration.

## Physics represented

The solver balances active generation and demand with the generator at bus 1 as the slack. It fixes
that busbar's phase angle to zero, solves the nodal susceptance system, and computes every closed
branch flow from its phase-angle difference, reactance, and transformer tap. Consequently:

- Kirchhoff's current law holds at each individual busbar: injection equals net outgoing flow.
- Kirchhoff's voltage law holds around connected cycles through consistent busbar phase angles.
- Open branches carry zero power.
- Loads and generators must remain in the single component containing the slack generator.

This is intentionally not AC power flow. Voltage magnitude, reactive power, resistance, losses,
frequency, protection dynamics, and generator dispatch limits are not modeled in version one.

Episode logs use `perf` for the fraction of overload-free steps. `score` is the episode return
divided by the theoretical `0.20`-per-step maximum and clipped to `[0, 1]`; catastrophic failures
always score zero.

Episodes contain six four-step operating periods. The first is always the safe nominal `P0`; the
other five are independent samples from stress profiles `P1` through `P14`, with replacement. The
sampling uses Puffer's per-environment RNG state, so a seed is reproducible and vector environments
do not share RNG state. Profile IDs are not added to observations because injections and line flows
already describe the current electrical condition.

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

The `--cpu` build exposes only the lightweight vector environment interface and is intended for
the PyTorch `--slowly` backend. Normal `puffer train power_grid` requires the CUDA build above,
which provides `create_pufferl`.
