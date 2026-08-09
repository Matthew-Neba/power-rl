# Power-Grid Topology Control Design

## Status (audited 2026-08-08)

Implemented: a dependency-free IEEE-14 DC solver; two-busbar topology model; 91-action PufferLib environment; rewards, logs, renderer, and vectorization; AC replay with voltage, reactive-power, loss, Q-limit, and thermal-stress checks; random outages; and DC/AC checkpoint comparison.

Offline scenarios combine correlated AESO load/wind/solar and ERA5 weather. The reproducible cache contains 365 training days from 2019 and 366 held-out days from 2020, each with twelve two-hour periods. Runtime selection is deterministic from the environment RNG and uses compile-time, read-only data: no network, files, parsing, allocation, copies, or added Python dependency. Training defaults to 90% historical and 10% synthetic curriculum; validation always uses 2020.

Weather adjusts synthetic limits from 0.90x to 1.35x using a steady-state IEEE 738 heat balance and published Drake ACSR parameters. Tests cover electrical invariants, bookkeeping, deterministic selection, train/validation separation, lifecycle, topology persistence, and all 8,772 cached periods. Normal cached points converge in AC without voltage violations; random-event outages are DC-solvable from normal topology.

Audit fixes ensure rejected closes are not counted as switches, period-advance failures return exactly `-5`, validation ignores the training mixture, configuration reaches the environment directly, and weather/equipment/AC state is observable through one stable DC/AC contract.

Unproven or out of scope: the rating model uses one reference conductor, perpendicular wind, effective irradiance, and one scale because IEEE-14 lacks route and asset data; it does not model sag or transient conductor temperature. Event states are not guaranteed AC-feasible, arbitrary policy-created topologies need not remain recoverable, and full N-1 security requires redispatch, shedding, or tighter scenarios. AC overload remains an evaluation signal. PPO quality, held-out robustness, real ratings, and deployment protection need empirical validation.

## 1. Goal and safety

Train quickly with DC power flow, then replay policies under AC physics and chronological scenarios. Relieve congestion with minimal switching while keeping all loads and generators in the slack-connected grid. A disconnected terminal, island, invalid topology, or failed solve is terminal; overload is recoverable. Priorities are connectivity/solvability, thermal limits, then switching cost. V1 excludes shedding, intentional islands, and dispatch control.

## 2. Grid and code

Use MATPOWER IEEE-14: 14 substations, 20 switchable branches (17 lines, 3 transformers), 5 generators, 11 loads, and a 100 MVA base. Each substation has two busbars and a controllable ideal coupler. All 40 branch endpoints and generator/load terminals move independently. Normal topology assigns terminals to busbar 1, closes branches, and opens couplers. Connectivity is built from busbar nodes, terminal assignments, branches, and couplers—not substation adjacency.

Synthetic DC limits, in branch order, are `[155,145,130,80,70,50,80,50,35,65,25,25,30,25,50,20,15,20,20,25]` MW. They are scenario parameters, not real ratings.

- `power_grid_solver.[ch]`: data, topology, DC flow, profiles, dense solve.
- `power_grid_ac.[ch]`: AC replay, constraints, losses, thermal support.
- `power_grid.h`: environment contract, episodes, logging, renderer.
- `binding.c`, `power_grid.c`, `evaluate.py`: PufferLib binding, playback, DC/AC comparison.

## 3. Electrical simulation

Active generation and demand are exogenous. Generator bus 1 is slack and its busbar angle is zero. For connected topology solve `B theta = P`, then compute closed-branch flow from angle difference, reactance, 100 MVA base, and tap. This enforces KCL/KVL, zero open-line flow, and active-power balance. Before solving, reject equipment outside the slack component, multiple energized components, islands, singular/non-finite solves, and invalid inputs.

For branch `l`, `rho_l = abs(F_l) / Fmax_l`; congestion is `C = sum(max(0, rho_l - 1)^2)`. Step order: observe solved state; apply one action; rebuild/validate; solve unchanged injections; reward/log/terminate; advance and solve the next injection period.

## 4. Environment contract

Observation: 221 unclipped floats in stable order.

| Count | Encoding |
|---:|---|
| 100 | Per branch: signed `F/Fmax`, `rho`, closed, available, thermal stress |
| 40 | Branch-end busbar bits; busbar 1=`0`, busbar 2=`1` |
| 5 / 11 | Generator/load busbar bits |
| 14 | Coupler bits; open=`0`, closed=`1` |
| 14 | Net substation injection `P_MW/100` |
| 1 / 3 | Rating scale; temperature `/50`, wind `/10`, irradiance `/1000` |
| 28 / 5 | Busbar voltage (DC active=`1`, inactive=`0`); generator Q `/100` (DC=`0`) |

Do not expose timestep, remaining fraction, prior action, or switch count; recurrence carries history. Action `0` is no-op; `1..20` toggle branches; `21..60` move branch ends; `61..65` move generators; `66..76` move loads; `77..90` toggle couplers. Actions are unmasked and change one item. Safe transfer may close a coupler, move terminals, then reopen it.

For valid states, `safe = (max_rho <= 1)` and `reward = -C - 0.001*N_switch + 0.20*safe`, without a lower cap. Safety/solver failure returns exactly `-5` and terminates. Episodes normally last 72 actions. `perf` is overload-free step fraction, forced to zero after catastrophe; `score` is no-op fraction. Mutually exclusive connectivity and solver failures sum to `total_failure`; invalid input is folded into solver failure, and `event_failure` separately tags only an immediate failure while applying an outage. Log return, length, switches, and failure causes; sweeps maximize `perf`.

## 5. Episodes and validation

Each episode has 12 six-action periods. Historical episodes retain correlated two-hour demand, generation, and weather trajectories. Synthetic episodes uniformly choose `P1..P14` and follow a smooth nominal-stress-nominal daily ramp. Scenario IDs stay hidden because observations expose actionable state.

- `P1`: scale load/gen-2 by 1.10, overload 1-2; validated split moves ends 2-4 and 2-5.
- `P2/P3`: move load 3→14 / 3→4; overload 9-14 / restore after P1 split.
- `P4/P5/P6`: move load 3→6/9/10; overload 5-6 / 4-7 / 9-10.
- `P7/P8/P9`: gen 6=90 MW / load 9→13 / gen 3=170 MW; overload 10-11 / 9-14 / 3-4.
- `P10/P11`: add load at 4/5 and generation at 2; overload 4-5 / 2-5.
- `P12/P13`: add load at 11/12 and generation at 6; overload 6-11 / 6-13.
- `P14`: add 30 MW demand at 2; overload 1-2 differently from P1.

Tests cover reference flows and AC values, profile stress, slack balance, KCL/KVL, taps, and topology failures; profiles need not have a known safe recovery. Held-out evaluation uses 2020 trajectories. DC/AC comparisons share scenarios, and either mode may enable outages. Curriculum grows from safe/no-op and one-switch cases through busbar sequences, unsafe alternatives, multi-action solutions, restoration, and broader randomization while retaining earlier cases.

## 6. AC evaluation and roadmap

AC replay preserves observations/actions while solving voltage magnitude/angle, P/Q, resistance, taps, charging, bus-9 shunt, losses, generator Q-limit PV→PQ conversion, and worst-end MVA loading. The first AC rating is 160 MVA; other synthetic values match DC. Detailed AC diagnostics remain in the renderer instead of the training dashboard. Voltage violations still make AC `perf` unsafe, while thermal trips affect the simulated topology. The thermal proxy trips at 200% after one step, 150% after four, about 120% after 25, cools when safe, and locks tripped lines for the episode.

DC omits voltage collapse, reactive shortage, frequency/transient stability, faults, synchronization, and protection. Roadmap: dynamics/frequency control; full IEEE 738 temperature, geometry, ratings, and relays; transformer thermal state; switching interlocks/protection zones; real ratings; N-1; redispatch and DC/AC OPF; curtailment; emergency shedding; validated islands; switching costs; randomized safe starts; larger grids.

## 7. Target architecture

Offline domain randomization produces robust policies from validated trajectories; PPO trains on fast DC physics; AC replay provides physics-faithful evaluation; runtime guardrails enforce topology, solver, and thermal safety with controlled fallback actions.
