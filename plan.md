# Power-Grid Topology Control Design

## 1. Objective and safety order

Build a fixed-grid RL environment where time-varying generation and demand create congestion and the agent safely reconfigures lines and substation busbars.

Priority order:

1. Keep every load and generator connected to the main grid; never create an unintended island.
2. Maintain a valid, solvable electrical topology.
3. Keep every closed branch at or below its thermal limit.
4. Minimize line and busbar switching.

Version one does not permit load shedding or viable intentional islands. Safety failures terminate immediately; overload alone does not terminate because the agent must have time to correct it.

## 2. Architecture and grid model

Implementation layers:

- `power_grid_solver.[ch]`: fixed grid data, topology actions, graph validation, DC solve, flows, loading, and operating profiles. It has no PufferLib or rendering dependency.
- `power_grid_baselines.[ch]`: do-nothing, seeded-random, greedy, and bounded-search controllers.
- `power_grid.h`: PufferLib environment state, observations, actions, rewards, episode logic, logs, and Raylib renderer.
- `binding.c`: vectorized PufferLib registration; `power_grid.c`: interactive standalone renderer.

Use a DC adaptation of the MATPOWER IEEE 14-bus case:

- 14 fixed substations, each containing busbars 1 and 2 plus a controllable bus coupler.
- 20 switchable branches: 17 lines and 3 transformer branches.
- 5 generator terminals, 11 load terminals, and a 100 MVA calculation base.
- 40 independently movable branch endpoints plus all generator and load terminals.
- A closed coupler electrically merges its two busbars; an open coupler keeps them independent.
- Normal topology: all terminals on busbar 1, all branches closed, and all couplers open.

Each terminal's parent equipment and substation are fixed metadata implied by its stable index. Its dynamic bit selects busbar 1 or 2. The electrical graph is built from busbar nodes, closed branches, and terminal assignments—not merely from substation adjacency.

MATPOWER does not provide usable IEEE-14 facility ratings. Current MW-equivalent limits are provisional synthetic scenario parameters in branch order:

`[155,145,130,80,70,50,80,50,35,65,25,25,30,25,50,20,15,20,20,25]`

Recalibrate them using approved normal scenarios and a target normal utilization, then regenerate and search-validate the stress profiles. Do not present these values as real IEEE equipment ratings.

## 3. Deterministic DC simulator

Inputs are exogenous active-power generation and demand. Bus 1 is the reference/slack generator and supplies the remaining active-power balance; dispatch is not agent-controlled.

For a connected topology, solve the linear nodal system `B * theta = P` after fixing the slack-busbar angle to zero. Closed-branch flow is calculated from endpoint angle difference, reactance, the 100 MVA base, and transformer tap ratio. This enforces:

- Kirchhoff's current law at every individual busbar: injection equals net outgoing flow.
- Kirchhoff's voltage law through consistent busbar phase angles and branch angle drops.
- Zero flow on every open branch.
- Total active generation equal to total active demand through slack balancing.

For each branch:

`rho = abs(F) / Fmax`

`C = sum(max(0, rho - 1)^2)`

Before solving, reject a load or generator outside the slack component, multiple energized components, and unintended islands. Also reject singular, failed, non-finite, or invalid-input solves. With valid fixed data, a connected DC network should solve; a numerical solve failure is primarily a diagnostic guard.

Timestep order:

1. Solve and observe the current injections and topology.
2. Receive one action.
3. Apply the toggle and rebuild/validate the busbar-level graph.
4. Re-solve with the same injections.
5. Calculate reward, termination, and logs.
6. Advance the injection profile and solve the next state.

## 4. Environment contract

### Observation: 144 unclipped floats

Stable layout:

| Positions | Count | Encoding |
|---|---:|---|
| Branch features | 60 | For each of 20 branches: `F/Fmax`, `abs(F)/Fmax`, closed=`1`/open=`0` |
| Branch-endpoint busbars | 40 | One bit per endpoint: busbar 1=`0`, busbar 2=`1` |
| Generator busbars | 5 | One bit per terminal: busbar 1=`0`, busbar 2=`1` |
| Load busbars | 11 | One bit per terminal: busbar 1=`0`, busbar 2=`1` |
| Bus couplers | 14 | One bit per substation: open=`0`, closed=`1` |
| Substation injections | 14 | Net active injection in per-unit form: `P_MW / 100` |

Continuous values are normalized but not clipped: an overloaded line at 130% is observed as `rho=1.30`, and a 219 MW net injection is `2.19`. Binary fields are lossless encodings of two-state categories, not ordinal multi-class IDs. Fixed vector positions identify equipment and parent substations, so static metadata is not repeated.

Do not include timestep, remaining episode fraction, previous action, or cumulative switch count. The recurrent LSTM handles history. There is no useful egocentric transform for one fixed control-room agent observing a fixed indexed grid.

### Action: one categorical value from 0 through 90

| Actions | Meaning |
|---|---|
| `0` | Do nothing |
| `1..20` | Toggle one branch open/closed |
| `21..60` | Toggle one branch endpoint between busbars |
| `61..65` | Toggle one generator terminal between busbars |
| `66..76` | Toggle one load terminal between busbars |
| `77..90` | Toggle one substation bus coupler open/closed |

At most one element changes per step. PPO produces 91 categorical logits. Dynamically unsafe actions are not masked; the agent learns their consequences. Couplers make energized transfers possible: close the coupler, move terminals, then reopen it to establish a split topology. The solver models an ideal closed coupler by merging both busbars into one electrical node, not as an artificial near-zero-reactance branch.

### Reward and termination

For a valid post-action state:

`safe = 1 if max_rho <= 1 else 0`

`raw_reward = -1.0*C - 0.01*N_switch + 0.20*safe`

`reward = max(raw_reward, -4.0)`

`N_switch=0` for no-op and `1` for a line, terminal, or coupler toggle. A safe no-op earns `0.20`; a safe switch earns `0.19`. Valid rewards lie in `[-4.0, 0.2]`.

A disconnected load or generator, island, invalid topology, or failed solve receives exactly `-5.0` and terminates. Episode exhaustion is normal completion after 48 steps.

Log `perf` as the overload-free step fraction. Log `score` as normalized episode return: `clamp(episode_return / (0.20 * episode_length), 0, 1)`, forced to zero after any catastrophic failure. This makes `perf` directly describe grid security while `score` combines congestion, switching, and safety through the actual reward. Also log switch counts, failure causes, and episode return/length.

## 5. Randomized episodes and validation

Use 48 actions with injections held for four actions, producing 12 periods. Every episode begins with safe `P0`; independently sample each of the remaining 11 periods uniformly from `P1..P14` with replacement. Puffer initializes a separate RNG state for each vector environment, and reset advances that state, giving reproducible but varied schedules. The profile label is not observed: injections and resulting flows already expose the physical state.

- `P0`: nominal MATPOWER active loads, generator 2 at 40 MW, and bus 1 balancing; normal topology is safe.
- `P1`: multiply loads and generator 2 by 1.10; overloads 1-2. A validated bus-2 split moves the 2-4 and 2-5 endpoints to busbar 2.
- `P2`: move 10 MW of load from bus 3 to bus 14; overloads 9-14.
- `P3`: move 45 MW of load from bus 3 to bus 4; safe normally but overloads 3-4 if the P1 split remains, teaching restoration.
- `P4`: move 40 MW of load from bus 3 to bus 6; overloads 5-6.
- `P5`: move 50 MW of load from bus 3 to bus 9; overloads 4-7.
- `P6`: move 20 MW of load from bus 3 to bus 10; overloads 9-10.
- `P7`: schedule 90 MW at generator bus 6; overloads 10-11 while bus 1 balances the change.
- `P8`: move 20 MW of load from bus 9 to bus 13; overloads 9-14 under a different distribution than P2.
- `P9`: schedule 170 MW at generator bus 3; overloads 3-4.
- `P10`: add 60 MW at load bus 4 and schedule 120 MW at generator bus 2; overloads 4-5.
- `P11`: add 80 MW at load bus 5 and schedule 140 MW at generator bus 2; overloads 2-5.
- `P12`: add 20 MW at load bus 11 and schedule 60 MW at generator bus 6; overloads 6-11.
- `P13`: add 30 MW at load bus 12 and schedule 60 MW at generator bus 6; overloads 6-13.
- `P14`: add a local 30 MW demand peak at bus 2; overloads 1-2 through a different injection pattern than P1.

All fifteen profiles have a connected overload-free topology. Bounded search reaches one from normal topology in at most three switches. Tests cover profile stress locations and solvability, reference flows, slack balance, KCL/KVL, transformer taps, topology failures, and baseline actions.

Do not train against changed grid data until bounded search proves every intended profile has a connected overload-free solution and records the minimum switch depth. PPO must beat random and do-nothing and should approach the search baseline without exploiting termination or reward clipping.

Curriculum order: safe no-op, one line toggle, short busbar sequence, unsafe alternatives, multiple candidate actions, multi-action solutions, time-varying restoration, then broader bounded injection/timing randomization. Retain earlier scenarios to prevent safety forgetting.

## 6. Future work and DC-model limitations

DC power flow models active power and phase-angle differences only. It cannot detect voltage collapse, reactive-power shortages, voltage-magnitude violations, frequency instability, transient or small-signal instability, short circuits, protection behavior, or unsafe breaker synchronization. A connected topology accepted by this environment is therefore not proof that the corresponding real switching action is safe.

Future electrical-fidelity work:

- Add AC power flow with voltage magnitudes, reactive power, generator Q limits, losses, and AC non-convergence handling.
- Add dynamic simulation for frequency, inertia, governor response, transient stability, synchronization, and switching transients.
- Model individual breakers and disconnectors, interlocks, detailed switching sequences, faults, relays, and automatic line trips.
- Add overload duration, conductor/transformer temperature, emergency ratings, cascading failures, and blackout propagation.
- Replace synthetic limits with voltage-, conductor-, transformer-, weather-, terminal-equipment-, and contingency-based facility ratings.

Future operational scope:

- N-1 security analysis and contingency-aware observations/rewards.
- Generator limits, redispatch, merit order, and DC/AC optimal power flow.
- Renewable availability and curtailment penalties based on topology-dependent delivered energy; never reward exogenous production directly.
- Emergency load shedding with unserved-energy penalties far larger than congestion costs.
- Carefully validated viable islands with generation-demand balance and frequency control.
- Differentiated switching costs, restricted terminal controls, randomized safe initial topologies, larger grids, and held-out evaluation scenarios.
