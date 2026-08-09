# IEEE-118 Power-Grid Topology Control Design

## Status

The IEEE-14 environment has been ported to canonical MATPOWER IEEE-118 while retaining its
two-busbar topology actions, fast DC training, AC replay, chronological Alberta scenarios,
weather-adjusted ratings, inverse-time thermal protection, and deterministic baseline harness.
This branch is intentionally isolated from `main` until benchmark review and explicit merge approval.

## Electrical model

- 118 substations, 186 branches, 54 generators, 99 loads, 100 MVA base.
- Two ideal busbars at every substation; all 525 equipment terminals move independently.
- 118 couplers merge or separate local busbars.
- Canonical case118 resistance, reactance, charging, taps, shunts, voltage targets, and Q/P limits.
- Synthetic ratings of 225 MVA at 138/161 kV and 550 MVA at 345 kV/transformers because the
  canonical source declares its ratings unlimited.
- Bus 69 is the slack/reference generator.

DC solves `B theta = P`, rejects equipment outside the slack component, and computes branch MW
loading. AC Newton-Raphson evaluation performs PV-to-PQ conversion at Q limits, branch-end P/Q/MVA,
losses, voltage checks, and thermal protection. DC omits voltage/reactive, frequency, transient,
fault, synchronization, and detailed protection dynamics.

## Environment contract

Observation size is 1,985 floats:

| Count | Encoding |
|---:|---|
| 930 | 186 branches × signed loading, rho, closed, available, thermal stress |
| 525 | equipment-terminal busbar bits |
| 118 | coupler bits |
| 118 | substation net injections divided by 100 MVA |
| 4 | rating scale and normalized temperature/wind/irradiance |
| 236 | busbar voltage or DC active bit |
| 54 | generator Q/100, zero in DC |

There are 830 unmasked discrete actions: no-op, 186 line toggles, 525 terminal transfers, and 118
coupler toggles. Valid reward is squared-overload cost plus a 0.001 switching penalty and 0.20 safe
bonus. Invalid topology/solve returns -5 and terminates. AC-only voltage and trip outcomes remain
evaluation metrics and do not change DC learning.

## Scenarios and outages

The compiled cache preserves twelve two-hour periods for every day of 2019 and 2020. Training uses
2019, validation uses held-out 2020. AESO load scales the 4,242 MW base demand; Alberta wind/solar
capacity factors drive 700/500 MW generators; canonical online generators redispatch; ERA5 weather
scales ratings from 0.90x to 1.35x. Ten synthetic profiles supply global, regional-shift, and
renewable stresses.

Training mixes 90% historical and 10% synthetic episodes. Half schedule three distinct outages at
distinct periods. The nine canonical bridge branches are excluded, and each selected three-line set
must leave the normal topology connected. Policy-created combinations and AC security are measured,
not assumed safe.

## Evaluation

PPO and all baselines receive the same held-out days, environment seeds, outage lines, and outage
periods. Primary results are AC `perf`, catastrophic failure, full-demand service, outage completion,
all-outage survival, voltage safety through `perf`, thermal-trip incidence/count, peak loading/stress,
overload exposure, and switch count. Evaluation return is deliberately omitted.

Baselines are no-action, uniform random, random lines, DC-screened safe random, greedy lines, and
greedy all. Greedy candidate scoring is DC even when the episode is evaluated in AC, preventing an
unfair AC-model oracle. The main report separates no-outage control, matched 50%/three-outage
distribution, and forced three-outage stress suites.

## Limitations

IEEE-118 is not Alberta topology. Ratings, weather exposure, two-busbar layouts, generator mapping,
and regional synthetic profiles are benchmark assumptions. Real deployment requires authenticated
network cases, facility ratings, route/conductor geometry, dispatch rules, remedial-action schemes,
protection/interlocks, transient/frequency validation, and operational review.
