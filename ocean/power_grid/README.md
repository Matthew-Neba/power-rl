# Alberta-driven IEEE-14 topology control

This environment trains a topology-control policy with fast DC power flow and evaluates the same
policy under AC physics. The electrical network is the canonical MATPOWER IEEE 14-bus case.
Chronological demand, renewable, and weather trajectories come from Alberta data, so this is an
Alberta-driven synthetic benchmark rather than a reconstruction of Alberta's transmission grid.

## Model

- 14 substations represented by 28 independently configurable busbars.
- 20 switchable branches, 5 generators, 11 loads, and a 100 MVA base.
- 56 movable branch/generator/load terminals and 14 ideal busbar couplers.
- 91 unmasked actions: no-op, line toggles, terminal transfers, and coupler toggles.
- 221 observations: line loading/state/stress, terminal and coupler state, injections,
  weather/rating state, busbar voltage or DC activity, and generator reactive output.

`power_grid_solver.c` retains the IEEE-118-era optimized implementation: union-find topology
validation, topology caches, precomputed branch coefficients, a minimum-degree sparse Cholesky
factorization cache, float training solves with double residual refinement, and no-op solution and
observation reuse. AC Newton-Raphson is integrated in the same solver and adds resistance, charging,
transformer taps, the bus-9 shunt, voltage/reactive limits, losses, worst-end MVA loading, and
inverse-time thermal protection.

Every load and generator must remain connected to the bus-1 slack component. Random-event
eligibility guarantees that removing a line leaves the normal topology connected and DC-solvable;
it does not guarantee AC convergence, absence of overload, or security after later policy actions.

## Scenarios and outages

The checked-in scenario cache contains all 365 days of 2019 for training and all 366 days of 2020
as a held-out validation pool. Each day combines AESO Alberta Internal Load, wind, and solar with
Edmonton-area ERA5 temperature, wind speed, and solar irradiance in twelve correlated two-hour
periods.

Demand scales IEEE-14's 259 MW base load. The cached 500 MW solar and 700 MW wind traces are treated
as capacity-factor references and mapped at runtime to 30 MW solar at bus 6 and 45 MW wind at bus 3.
The bus-2 conventional generator uses available headroom and the bus-1 slack balances the residual.
Weather scales every synthetic line rating from 0.90x to 1.35x using the retained steady-state
IEEE 738 approximation. No files, parsing, allocation, Python, or network calls occur in the
environment hot path.

Training mixes 65% historical days with 35% synthetic global, regional-transfer, and renewable
stress ramps. Every standard training episode schedules one or two persistent outages at distinct
later periods. The radial 7-8 line is excluded, and each sampled multi-line set is validated
together before use.

## Reward and metrics

The DC training reward is:

```text
-congestion_cost * congestion_cost_weight
+signed congestion-progress reward
-switch_penalty when a physical switch occurs
+safe_step_reward when all lines are within their limits
+recovery_reward on a physical action that restores security after an outage
```

Invalid topology or solver failure terminates with `failure_reward`. The current defaults are
failure `-1.0`, safe step `1.0`, recovery `0.5`, congestion weight `0.01`, switch penalty `0.002`,
secure-state switch penalty `0.1`, and congestion progress `1.0`. These coefficients remain
configurable; Puffer clips individual training rewards to `[-1, 1]`.
AC voltage violations and thermal trips remain evaluation metrics because AC
power flow is disabled during high-throughput training.

Connectivity failure and numerical solver failure are logged separately. Evaluation also reports
secure-step performance, demand served, outage completion and survival, switching, thermal trips,
peak stress/loading, and overload exposure. Evaluation reports intentionally omit reward/return.

## Build, test, train, and evaluate

```sh
uv run --with pytest pytest -q \
  tests/test_power_grid_solver.py tests/test_power_grid_scenarios.py
PATH="$PWD/.venv/bin:$PATH" ./build.sh power_grid
PATH="$PWD/.venv/bin:$PATH" puffer train power_grid
PATH="$PWD/.venv/bin:$PATH" puffer eval power_grid
```

Compare a checkpoint against deterministic baselines on held-out 2020 scenarios:

```sh
PATH="$PWD/.venv/bin:$PATH" python ocean/power_grid/benchmark.py \
  --checkpoint latest --episodes 1024 --physics dc \
  --random-event-probability 1.0 --random-outages 2
```

The checked-in 256x3 MinGRU PPO checkpoint was frozen after model selection on 2019 training data.
On the fixed 256-episode held-out 2020 set (seed offset 50,000), its final results are:

| Scenario / inference | PPO perf | PPO failures | Greedy perf | Lookahead perf |
|---|---:|---:|---:|---:|
| N-1 argmax | 0.9065 | 0.00% | 0.9153 | 0.9171 |
| N-1 stochastic | 0.9063 | 0.00% | 0.9153 | 0.9171 |
| N-2 argmax | 0.7978 | 0.78% | 0.8026 | 0.8049 |
| N-2 stochastic | 0.7898 | 1.56% | 0.8026 | 0.8049 |

These are unrestricted 91-action results: no action mask, rollback, fallback, forced no-op, or
line-only policy is used. The PPO policy did not beat greedy or two-step lookahead on secure-step
performance. Its useful result is lower deterministic N-2 failure (0.78% versus 1.95% for both
baselines), not a performance-baseline win. Argmax is the stronger deployment mode on this gate;
stochastic sampling remains available in the interactive app for direct comparison.

For environment-only throughput:

```sh
clang -std=c11 -O3 -march=native -Iocean/power_grid \
  tests/benchmark_power_grid_throughput.c -lm -o /tmp/power-grid-bench
/tmp/power-grid-bench mixed 128 8192
```

On the current machine, the migrated environment measured about 7.43M no-op SPS, 1.17M mixed SPS,
and 652K all-random SPS. A real 5M-step CUDA PPO probe using a 256x3 policy, 1,024 agents,
16 environment threads, cached scenarios, and randomized outages sustained roughly 308K-328K SPS
and ended at 314K SPS. The active sweep configuration may use a different fixed policy or thread
count.

## Limitations

IEEE-14 is a compact research benchmark, not Alberta topology. Ratings, renewable placement,
weather exposure, two-busbar layouts, dispatch, and outage distributions are assumptions. Real
deployment requires authenticated network cases and facility ratings, route/conductor geometry,
dispatch and remedial-action rules, protection/interlocks, N-1/N-k and transient/frequency studies,
operator review, and deployment monitoring.

The policy also remains below simple planning baselines in held-out secure-step performance and was
trained with the DC solver. AC validation checks feasibility but is not evidence that the policy is
ready for real-grid control. N-2 generalization is the principal learned-policy limitation.
