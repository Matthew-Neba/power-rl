# Alberta-driven IEEE-14 topology control

This environment trains a topology-control policy with fast DC power flow and evaluates the same
policy under AC physics. The electrical network is the canonical MATPOWER IEEE 14-bus case.
Chronological demand, renewable, and weather trajectories come from Alberta data, so this is an
Alberta-driven synthetic benchmark rather than a reconstruction of Alberta's transmission grid.

## Model

- 14 substations represented by 28 independently configurable busbars.
- 20 switchable branches, 5 generators, 11 loads, and a 100 MVA base.
- 56 movable branch/generator/load terminals and 14 ideal busbar couplers.
- 106 unmasked actions: no-op, line toggles, terminal transfers, coupler toggles,
  11 irreversible load sheds, and 4 irreversible non-slack generator trips.
- 236 observations: line loading/state/stress, terminal and coupler state, injections,
  weather/rating state, busbar voltage or DC activity, generator reactive output, and the
  connected state of controllable loads and generators.

`power_grid_solver.c` retains the IEEE-118-era optimized implementation: union-find topology
validation, topology caches, precomputed branch coefficients, a minimum-degree sparse Cholesky
factorization cache, float training solves with double residual refinement, and no-op solution and
observation reuse. AC Newton-Raphson is integrated in the same solver and adds resistance, charging,
transformer taps, the bus-9 shunt, voltage/reactive limits, losses, worst-end MVA loading, and
inverse-time thermal protection.

Connected equipment must remain in an energized island. Deliberately shed loads, deliberately
tripped generators, and de-energized islands with no injection are represented explicitly. Outage
requests are never rejected merely because they island the network; AC convergence, thermal
security, and continued service are outcomes measured after the request.

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
stress ramps. Every standard training episode schedules one or two distinct persistent outages at
later periods. All 20 lines are eligible, including radial line 7-8, and multi-line sets are not
prefiltered for feasibility.

## Reward and metrics

The DC training reward is:

```text
-congestion_cost * congestion_cost_weight
+signed congestion-progress reward
-unserved_load_fraction * unserved_load_cost_weight
-switch_penalty when a physical switch occurs
+safe_step_reward when all lines are within their limits
+recovery_reward on a physical action that restores security after an outage
```

Invalid topology or solver failure terminates with `failure_reward`. The current defaults are
failure `-1.0`, safe step `1.0`, recovery `0.5`, congestion weight `0.01`, unserved-load weight
`5.0`, switch penalty `0.002`, secure-state switch penalty `0.1`, and congestion progress `1.0`.
These coefficients remain configurable; Puffer clips individual training rewards to `[-1, 1]`.
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

The plain evaluation command follows `config/power_grid.ini`: DC physics with automatic N-1/N-2
outages. To evaluate a named checkpoint with AC physics and an outage in every episode, use:

```sh
PATH="$PWD/.venv/bin:$PATH" puffer eval power_grid \
  --load-model-path resources/power_grid/policy.bin \
  --env.ac-power-flow True \
  --env.random-events True \
  --env.random-event-probability 1 \
  --env.random-outage-count 2 \
  --env.random-outage-count-min 1
```

This samples episodes; it does not replace the exhaustive 210-combination browser-contract QA.

The deployed research prototype is available at
<https://matthew-neba.github.io/power-rl/>. Policy inference and AC validation
run locally in WebAssembly; at most two persistent user clicks are the only
outage source.

Build and serve the browser demo using the same Ocean C-to-WebAssembly path as the upstream
PufferLib demos:

```sh
source /path/to/emsdk/emsdk_env.sh
./build.sh power_grid --web
python -m http.server 8000 --directory build/web/power_grid
```

Open `http://localhost:8000/game.html`. The browser runs the embedded 256x3 MinGRU policy and AC
solver locally. It draws held-out 2020 demand, renewable, and weather conditions; automatic
outages are disabled, so clicking up to two lines is the only contingency source. Argmax is the
default inference mode; keys `1` and `2` select stochastic and argmax inference respectively.
Infeasible user outages remain applied and are counted as failures rather than being rejected or
rolled back.

The exhaustive browser-contract QA enumerates all 20 N-1 and 190 N-2 line sets, alternates both
N-2 click orders, and rotates outage time and held-out context:

```sh
clang -std=c11 -O3 -Wall -Wextra -Werror -pedantic \
  -Iocean/power_grid -Isrc -Ivendor ocean/power_grid/qa_user_outages.c \
  -lm -o build/qa_user_outages
./build/qa_user_outages resources/power_grid/policy.bin 8 8
```

The final argument is a context offset. Contexts 0-7 are the 2020 validation
subset used during policy selection; contexts 8-15 are the separate confirmation
subset shown below.

The QA report keeps physical feasibility separate from policy performance and defines a handled
trial as issuing no pre-click emergency command, completing the day without solver/connectivity
failure, keeping at least 95% of post-click steps thermally and voltage secure, and serving at
least 90% of demand on average. The first and last gates prevent anticipatory or indiscriminate
shedding from counting as recovery. Impossible islanding combinations remain in the denominator
rather than being hidden behind a request filter.

The default configuration samples one or two outages in every training episode
(`random_outage_count_min = 1`, `random_outage_count = 2`). A plain
`puffer train power_grid` therefore trains on a mixture of N-1 and N-2 events,
but its dashboard averages the two cases and is not an exact N-1 or N-2 gate.

To reproduce the current emergency-control curriculum from the 91-action checkpoint recorded by
commit `fcef678b`, run:

```sh
ocean/power_grid/train_honest.sh
```

The script first expands the old 221/91 checkpoint to 236/106 without changing its legacy argmax
behavior. Its fixed-seed AC PPO curriculum then progresses through general reset recovery,
one-step N-1 recovery, intact-grid service preservation, balanced intact/N-1/N-2 recovery, and
full-day batches that mix reset-time, naturally delayed, and consecutive-click N-1/N-2 outages.
The intact stages were added after browser QA caught a policy that shed load before the first
click; the mixed stages prevent recovery training from forgetting the recurrent transition after
a click. Sparse PPO still discovered too few corrective actions, so two stages fit the same
decoder to unrestricted AC recovery actions generated offline from 2019 states. A final
aggregation stage uses set-valued supervision: when several existing actions independently meet
the AC thermal, voltage, and demand gates, any of them is a correct target instead of forcing an
arbitrary tie-break. The final stage unrolls those complete trajectories through the unchanged
MinGRU and fine-tunes its encoder and decoder, allowing the recurrent transition after consecutive
clicks to receive the same supervision. The 2020 data is never used by training.
The exported MinGRU acts by itself without action masks, rollback, planners, forced no-op,
recurrent-state reset on click, or runtime fallbacks. The offline teacher is not compiled into the
web app. The script writes only beneath `checkpoints/power_grid/honest-surpass-run/` and never
replaces the resource policy.

The promoted checkpoint is `resources/power_grid/policy.bin`, SHA-256
`3e703a790b660330da8c0d77b1890c6fe6fb570e2f70b9838a844fa01b231e65`. Exhaustive held-out 2020
AC confirmation used contexts 40-47 for every combination (1,680 trials total), after the policy
lineage and training parameters had been evaluated on contexts 0-39:

| Contingency | Survival | Handled | Secure steps | Demand served |
|---|---:|---:|---:|---:|
| N-1 (20 sets) | 93.13% | 86.88% | 91.96% | 98.00% |
| N-2 (190 sets) | 80.99% | 67.50% | 78.69% | 96.00% |
| Combined | 82.14% | 69.35% | 79.96% | 96.19% |

No tested case issued an emergency command before its first click. The earlier emergency policy
had higher apparent handled performance under the old metric, but browser QA showed it shed load
before every request; under the corrected product contract its handled rate was 0%. The current
policy improves the earlier honest checkpoint's fresh-slice handled rate from 54.29% to 69.35%
while preserving high demand service, but it does **not**
meet the requested 95% handled gate. It is a research/demo checkpoint, not a real-grid deployment
policy.

Compare a checkpoint against deterministic baselines on held-out 2020 scenarios:

```sh
PATH="$PWD/.venv/bin:$PATH" python ocean/power_grid/benchmark.py \
  --checkpoint latest --episodes 1024 --physics dc \
  --random-event-probability 1.0 --random-outages 2
```

The previous protected 256x3 MinGRU PPO checkpoint at commit `c80d9d83` was frozen after model
selection on 2019 training data. On the fixed 256-episode held-out 2020 set (seed offset 50,000),
its final results were:

| Scenario / inference | PPO perf | PPO failures | Greedy perf | Lookahead perf |
|---|---:|---:|---:|---:|
| N-1 argmax | 0.9065 | 0.00% | 0.9153 | 0.9171 |
| N-1 stochastic | 0.9063 | 0.00% | 0.9153 | 0.9171 |
| N-2 argmax | 0.7978 | 0.78% | 0.8026 | 0.8049 |
| N-2 stochastic | 0.7898 | 1.56% | 0.8026 | 0.8049 |

These historical results used unrestricted 91-action inference: no action mask, rollback, fallback, forced
no-op, or line-only policy was used when the checkpoint ran. Repository history shows that this
previous checkpoint's earlier training lineage did use action shielding and a lookahead-action
reward, however, so it is not evidence of fully unassisted PPO training. The honest recovery
curriculum above deliberately uses neither mechanism. The archived policy did not beat greedy or
two-step lookahead on secure-step performance. Its useful result is lower deterministic N-2
failure, not a performance-baseline win. Argmax is the stronger deployment mode on this gate;
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

The legacy policy remained below simple planning baselines in held-out secure-step performance.
The current emergency policy was fine-tuned with AC power flow and is validated under AC, but that
still does not establish transient, frequency, protection, or real-grid safety. N-2 recovery and
service preservation remain the principal learned-policy limitations.
