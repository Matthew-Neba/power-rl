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
| PPO actions | 91 (all environment actions) |
| Float observations | 221 |
| Training episode | 72 steps / 12 periods |
| Interactive runtime | continuous; periods wrap without reset |

PPO can select every line, terminal-busbar, and coupler action. Invalid topology commands and
outage-sensitive topology choices have their real episode-ending consequences; the environment
does not silently replace them with no-op or roll them back for the agent.

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
cleared. This state and mouse input are separate from the training environment. The app retains
automatic outages and uses ordinary 72-step episodes. A terminal state starts a fresh environment
episode and a newly initialized policy instance.

Held-out 2020 evaluation of the deployed checkpoint (4,096 episodes):

- N-1: PPO perf 0.9186 vs greedy 0.9184; zero failures and 100% demand served.
- N-2: PPO perf 0.8137 vs greedy 0.8172; zero failures and 100% demand served.
- Repeated exhaustive DC outage windows: PPO safe-step fraction 0.5591 vs greedy 0.5531,
  with no terminal failures across 2,184 steps for either policy.

These results belong to the superseded 21-action phase-1 checkpoint. Phase 2 restores all 91
actions and must be re-trained and re-evaluated before new performance claims are made. The web app
offers both stochastic Puffer-style sampling and deterministic argmax inference.

## Phase-2 training audit

The unrestricted 91-action PPO was trained and screened with direct N-1/N-2, staged N-1 then N-2,
recorded-day adaptation, congestion shaping, and an early-outage recovery curriculum. The strongest
checkpoints were specialists rather than one policy that achieved both results:

- fixed 256-episode N-1 gate: specialist PPO `0.9098` perf with zero failures, greedy `0.9153`,
  and lookahead `0.9171`; this checkpoint scored `0.7525` on N-2 with `6.64%` failures;
- fixed 256-episode N-2 gate: specialist PPO `0.7918` perf with `0.78%` failures, greedy `0.8026`,
  and lookahead `0.8049`, both with `1.95%` failures; this checkpoint scored `0.9039` on N-1
  with zero failures.

These are deterministic-argmax results. No action masking, rollback, fallback, forced no-op, or
line-only restriction was used. PPO was more reliable on N-2 but remained behind both baselines on
safe-step performance, so no phase-2 checkpoint was deployed as a baseline-beating policy.

The old phase-1 checkpoint was also retested under the current honest transition rules. Its N-1
perf fell to `0.5908` with `30.9%` failures, confirming that it must not be reused as a phase-2
result. Temporary checkpoints and benchmark binaries were removed after the audit.
