# Power-Grid Topology Control Design

## Implementation status (audited 2026-08-08)

Completed:

- Dependency-free DC IEEE-14 solver, busbar topology model, 91-action environment contract,
  rewards, episode logging, renderer, and vectorized PufferLib binding.
- AC replay with voltage and reactive-power checks, losses, Q-limit handling, thermal-stress
  protection, scheduled maintenance, and DC/AC checkpoint comparison tooling.
- Offline domain randomization from actual correlated AESO load/wind/solar and ERA5 temperature,
  wind-speed, and irradiance data. The checked-in cache contains 365 training days from 2019 and
  366 held-out days from 2020, with twelve two-hour operating periods per day.
- Runtime scenario selection is deterministic from each environment RNG and uses compile-time
  read-only data only: no network, files, parsing, heap allocation, copies, or new Python runtime
  dependency. Training defaults to 75% cached 2019 days and 25% synthetic curriculum; validation
  mode always selects the disjoint 2020 pool.
- A steady-state IEEE 738 heat balance varies synthetic branch limits from 0.90x to 1.35x using
  published Drake ACSR example parameters. The cache records source hashes and can be reproduced
  with `build_offline_scenarios.py`.
- Automated checks cover electrical invariants, action and reward bookkeeping, deterministic
  scenario selection, train/validation separation, randomized environment lifecycle, and every
  one of the 8,772 cached operating periods. Every normal operating point converges in AC without
  voltage violations, and every random-event line outage remains DC-solvable. A focused regression
  verifies that topology persists across periods.

Audit corrections completed:

- Rejected attempts to close a maintenance-disabled or thermally tripped line no longer count as
  physical switching operations.
- A failure caused by advancing to new exogenous injections emits and logs exactly the `-5`
  failure reward instead of retaining the preceding state's provisional reward.
- Held-out validation selection no longer depends on the training-mixture probability.
- Scenario-mixture configuration is passed directly to the environment without a hidden fallback.
- Weather, rating scale, line availability/thermal stress, busbar voltage, and generator reactive
  output are observed. DC marks active busbars at unit voltage and supplies zero reactive output,
  so one stable contract supports DC training and optional AC policy training.

Not yet completed or proven:

- IEEE 738 currently uses one reference conductor, perpendicular wind, effective incident
  irradiance, and one scale for all branches because IEEE-14 has no physical route or asset data.
  It is steady-state ampacity, not a sag or transient conductor-temperature model.
- Random events are checked for DC solvability from normal topology. Continuous-day recoverability
  from every topology an arbitrary policy could create is not required. Full N-1 security is also
  out of scope without redispatch, shedding, or revised scenario constraints.
- AC convergence and voltage feasibility are exhaustively checked for every cached normal point,
  while AC thermal overload remains an evaluation signal rather than a guaranteed-easy condition.
- PPO performance, policy robustness on held-out 2020 data, real facility
  ratings, and deployment-grade protection behavior still require empirical validation.

## 1. Goal and safety

Train on fast DC power flow, then replay policies under AC physics and chronological scenarios. The agent must relieve congestion with as little switching as possible while keeping every load and generator connected to the main grid. A disconnected terminal, unintended island, invalid topology, or failed solve is terminal failure; overload alone is recoverable.

Priority: connectivity and solvability, thermal limits, then switching cost. Version one excludes load shedding, intentional islands, and dispatch control.

## 2. Grid and code structure

Use a DC adaptation of MATPOWER IEEE-14: 14 substations, 20 switchable branches (17 lines and 3 transformers), 5 generators, 11 loads, and a 100 MVA base. Every substation has two busbars and an ideal controllable coupler. All 40 branch endpoints and all generator/load terminals can move independently between busbars. Normal topology puts every terminal on busbar 1, closes every branch, and opens every coupler.

A terminal's stable index identifies its equipment and substation; its state bit chooses busbar 1 or 2. A closed coupler electrically merges both busbars. Build connectivity from busbar nodes, branch endpoints, terminal assignments, and couplers—not substation adjacency.

Synthetic DC branch limits in branch order are `[155,145,130,80,70,50,80,50,35,65,25,25,30,25,50,20,15,20,20,25]` MW. They are scenario parameters, not real IEEE ratings, and should eventually be calibrated from approved scenarios and target utilization.

- `power_grid_solver.[ch]`: grid data, topology actions/validation, DC flow, profiles, shared dense solve.
- `power_grid_ac.[ch]`: Newton-Raphson AC replay, electrical constraints, losses, and thermal support.
- `power_grid.h`: environment state, contract, episodes, logging, and renderer.
- `binding.c`: PufferLib registration; `power_grid.c`: standalone playback; `evaluate.py`: DC/AC checkpoint comparison.

## 3. Electrical simulation

Inputs are exogenous active generation and demand. Generator bus 1 is slack: it supplies the remaining active-power balance, and its busbar angle is the zero-angle reference. For valid connected topology, solve `B theta = P`; compute each closed-branch flow from endpoint angle difference, reactance, 100 MVA base, and transformer tap. This enforces KCL at each busbar, KVL through consistent phase angles, zero flow on open branches, and total active-power balance.

For branch `l`, `rho_l = abs(F_l) / Fmax_l` and congestion `C = sum(max(0, rho_l - 1)^2)`. Before solving, reject equipment outside the slack component, multiple energized components, islands, singular/non-finite solves, and invalid input.

Step order: observe current solved state; receive one action; apply it; rebuild and validate topology; solve with unchanged injections; reward/log/terminate; then advance and solve the next injection period.

## 4. Environment contract

Observation: 221 unclipped floats in stable order.

| Count | Encoding |
|---:|---|
| 100 | Per branch: signed `F/Fmax`, `rho`, closed bit, available bit, and thermal stress |
| 40 | Branch-endpoint busbar bits: busbar 1=`0`, busbar 2=`1` |
| 5 / 11 | Generator/load terminal busbar bits |
| 14 | Coupler bits: open=`0`, closed=`1` |
| 14 | Net substation injection `P_MW / 100` |
| 1 | Current weather-adjusted branch-rating scale |
| 3 | Ambient temperature `/50`, wind speed `/10`, and irradiance `/1000` |
| 28 | Per-busbar AC voltage magnitude; active=`1`, inactive=`0` in DC mode |
| 5 | Generator reactive output `/100`; neutral `0` in DC mode |

Continuous values are normalized but not clipped, preserving overload severity. Binary fields losslessly encode two-state categories. Do not observe timestep, remaining fraction, previous action, or switch count; the recurrent policy handles history.

Action: one categorical value from 0 through 90: `0` no-op; `1..20` toggle branches; `21..60` move branch endpoints; `61..65` move generators; `66..76` move loads; `77..90` toggle couplers. Only one item changes per step. Unsafe actions are not masked. A safe busbar transfer can close the coupler, move terminals, then reopen it.

For valid states, `safe = (max_rho <= 1)` and `reward = -C - 0.001*N_switch + 0.20*safe`, with no lower reward cap. A safety/solver failure gives exactly `-5` and terminates. Episodes normally end after 72 actions. `perf` is the overload-free step fraction, forced to zero after catastrophe; `score` is the fraction of actions that were no-ops. Log return, length, switches, and failure causes. Hyperparameter sweeps maximize `perf`.

## 5. Episodes, profiles, and validation

Each episode has 12 six-action periods. Default training uses a 75/25 mixture of complete days from the precomputed 2019 AESO/ERA5 cache and the deliberately congested original curriculum; the ratio is configurable. Historical episodes preserve correlated two-hour load, solar, wind, temperature, wind-speed, and irradiance trajectories. Weather adjusts synthetic branch ratings through a bounded steady-state IEEE 738 calculation. A disjoint 2020 cache is held out for validation. The runtime performs one RNG draw and no network, filesystem, parsing, allocation, or data-copy work. Synthetic episodes start with safe nominal `P0`, then sample every later period independently and uniformly from `P1..P14` with replacement. Scenario IDs are hidden because injections, flows, weather, and normalized loading expose the actionable current state.

- `P1`: 1.10 load/gen-2 scale, overload 1-2; validated bus-2 split moves endpoints 2-4 and 2-5.
- `P2/P3`: shift load bus 3→14 / bus 3→4; P2 overloads 9-14, P3 teaches restoration after the P1 split.
- `P4/P5/P6`: shift load bus 3→6/9/10; overload 5-6 / 4-7 / 9-10.
- `P7`: 90 MW at generator bus 6; overload 10-11.
- `P8`: shift load bus 9→13; overload 9-14 with a distinct distribution.
- `P9`: 170 MW at generator bus 3; overload 3-4.
- `P10/P11`: add load at bus 4/5 and generation at bus 2; overload 4-5 / 2-5.
- `P12/P13`: add load at bus 11/12 and generation at bus 6; overload 6-11 / 6-13.
- `P14`: add 30 MW demand at bus 2; overload 1-2 differently from P1.

Tests verify reference flows, profile stress, slack balance, KCL/KVL, taps, topology failures, and
AC reference values. Profiles are not required to have a known safe topology or bounded recovery
sequence.

Held-out evaluation uses a repeatable 00:00–22:00 demand curve, synthetic solar at generator bus 3, wind at bus 6, and a 9-14 maintenance outage from 08:00–16:00. Compare identical DC and AC trajectories. Curriculum progresses from safe no-op and one-switch cases through busbar sequences, unsafe alternatives, multi-action solutions, restoration, and broader randomization while retaining earlier cases.

## 6. AC evaluation and future work

AC replay keeps the same observations/actions but solves voltage magnitudes/angles, P/Q, resistance, taps, charging, bus-9 shunt, losses, generator Q-limit PV→PQ conversion, and worst-end MVA loading. The first AC rating is 160 MVA so nominal reactive flow preserves the safe start; other synthetic values match DC. Log voltage/P-limit violations, Q-limit events, losses, convergence, stress, and trips. The thermal proxy trips at 200% after one step, 150% after four, about 120% after 25, cools at safe loading, and locks tripped lines out for the episode.

DC cannot detect voltage collapse, reactive shortages, frequency or transient instability, short circuits, synchronization hazards, or real protection behavior. Future work: dynamics and frequency control; full IEEE 738 conductor temperature with route geometry, emergency ratings, and calibrated relays; transformer thermal state; breaker/disconnector interlocks and protection zones; real facility ratings; N-1 security; redispatch and DC/AC OPF; topology-dependent renewable curtailment; heavily penalized emergency shedding; validated viable islands; differentiated switching costs; randomized safe starts; and larger grids.

## 7. Ultimate architecture

Offline domain randomization across grid conditions and operating scenarios for robust policy learning.
Offline PPO training from validated trajectories, with AC replay used for physics-faithful evaluation.
Runtime safety guardrails enforce topology validity, solver health, thermal limits, and controlled fallback actions.
