# IEEE-14 Power-Grid RL Environment

## Status

The IEEE-14 environment, solver audit, scenario randomization, outage recovery controls, and
continuous interactive runtime are implemented. The checked-in policy is trained with DC physics;
AC is retained as an explicit validation mode.

## Contract

| Item | IEEE-14 value |
|---|---:|
| Substations / electrical nodes | 14 / 28 |
| Branches / generators / loads | 20 / 5 / 11 |
| Equipment terminals / couplers | 56 / 14 |
| Environment actions | 91 |
| PPO actions | 21 (no-op + 20 line switches) |
| Float observations | 221 |
| Training episode | 72 steps / 12 periods |
| Interactive runtime | continuous; periods wrap without reset |

The general environment retains line, terminal-busbar, and coupler actions. PPO is intentionally
restricted to no-op and line switching. Invalid or non-improving recovery commands are rejected
transactionally, so a bad command does not turn a recoverable contingency into a blackout.

## Physics and performance

DC training retains topology validation, branch-constant precomputation, minimum-degree sparse
Cholesky, float factorization with residual refinement, and no-op reuse.
AC evaluation retains Newton-Raphson, PV/PQ switching, voltage and reactive limits, branch-end MVA,
losses, and inverse-time thermal trips.

Correctness gates cover dense/optimized DC parity in double and float modes, KCL/KVL, all
single-line contingencies, randomized topology and injection changes, canonical AC voltage/angle
agreement, complex-power balance, scenario selection, multi-outage scheduling, reward controls,
and incremental/full observation parity.

Previously measured throughput on this system:

- C environment only: 7.43M no-op, 1.17M mixed, 652K all-random SPS.
- Full CUDA PPO, 256x3 network, 1,024 agents, 16 threads: approximately 308K-328K SPS,
  ending a 5M-step probe at 314K SPS.

## Scenario and evaluation policy

The read-only cache keeps 2019 training and disjoint 2020 validation days. Training samples 65%
recorded days and 35% synthetic stress profiles. AESO/ERA5 demand, renewable, temperature, wind,
and solar traces retain their daily correlation and drive injections and IEEE-738-style ratings.

Training now guarantees two distinct non-islanding outages per episode to match the live N-2 limit.
The interactive controller in `power_grid_user.h` accepts at most two user-selected outages,
rejects invalid combinations, and restores the pre-contingency topology when the final outage is
cleared. This state and mouse input are separate from the training environment. The app selects an
unbounded episode limit so time and operating conditions continue indefinitely.

Held-out 2020 evaluation of the deployed checkpoint (4,096 episodes):

- N-1: PPO perf 0.9186 vs greedy 0.9184; zero failures and 100% demand served.
- N-2: PPO perf 0.8137 vs greedy 0.8172; zero failures and 100% demand served.
- Repeated exhaustive DC outage windows: PPO safe-step fraction 0.5591 vs greedy 0.5531,
  with no terminal failures across 2,184 steps for either policy.

The remaining limitation is N-2 efficiency: PPO survives but is still slightly below greedy on the
independent held-out N-2 perf metric. Do not present that case as a PPO win.
