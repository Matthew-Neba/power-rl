# IEEE-14 Power-Grid Topology Control Design

## Status

The environment is scaled back to canonical MATPOWER IEEE-14 while retaining the optimized
IEEE-118-era solver, cached historical/synthetic scenario curriculum, randomized persistent line
failures, dynamic ratings, AC validation, reward sweep controls, security metrics, renderer
improvements, and deterministic baseline tooling.

## Contract

| Item | IEEE-14 value |
|---|---:|
| Substations / electrical nodes | 14 / 28 |
| Branches / generators / loads | 20 / 5 / 11 |
| Equipment terminals / couplers | 56 / 14 |
| Discrete unmasked actions | 91 |
| Float observations | 221 |
| Episode length | 72 steps / 12 periods |

The action space remains no-op plus line, terminal-busbar, and coupler toggles. No safety mask is
applied; unsafe topology must be learned through consequence. Connectivity failure and numerical
solver failure remain distinct.

## Physics and performance

DC training retains topology validation and factorization caches, branch-constant precomputation,
minimum-degree sparse Cholesky, float factorization with residual refinement, and no-op reuse.
AC evaluation retains Newton-Raphson, PV/PQ switching, voltage and reactive limits, branch-end MVA,
losses, and inverse-time thermal trips.

Correctness gates cover dense/optimized DC parity in double and float modes, KCL/KVL, all
single-line contingencies, randomized topology and injection changes, canonical AC voltage/angle
agreement, complex-power balance, scenario selection, multi-outage scheduling, reward controls,
and incremental/full observation parity.

Measured throughput on the current system:

- C environment only: 7.43M no-op, 1.17M mixed, 652K all-random SPS.
- Full CUDA PPO, 256x3 network, 1,024 agents, 16 threads: approximately 308K-328K SPS,
  ending a 5M-step probe at 314K SPS.

## Scenario and evaluation policy

The read-only cache keeps 2019 training and disjoint 2020 validation days. AESO/ERA5 traces retain
their chronological correlation. Demand scales the 259 MW IEEE-14 case; source renewable traces are
converted to capacity factors and mapped to 30 MW solar and 45 MW wind. Synthetic profiles and
weather-adjusted ratings remain mixed into training.

Event episodes use three distinct non-bridge outages at distinct later periods, and the combined
set must preserve normal-topology DC connectivity. AC security is evaluated rather than assumed.
Baseline tooling compares PPO with no-op, random, safe-random, and greedy policies under identical
seeds and held-out scenarios.
