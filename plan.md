# IEEE-14 Power-Grid RL Environment

## Status

The IEEE-14 environment, solver audit, scenario randomization, outage recovery controls, and
interactive web runtime are implemented. The checked-in policy is trained with DC physics; AC is
retained as an explicit validation mode.

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
| Interactive runtime | 72-step episodes; fresh environment and policy after terminal |

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

Training now guarantees one or two non-islanding outages per episode, covering N-1 and N-2 while
matching the live N-2 limit.
The interactive controller in `power_grid_user.h` accepts at most two user-selected outages,
rejects invalid combinations, and restores the pre-contingency topology when the final outage is
cleared. This state and mouse input are separate from the training environment. The app retains
automatic outages and uses ordinary 72-step episodes. A terminal state starts a fresh environment
episode and a newly initialized policy instance.

The web app offers both stochastic Puffer-style sampling and deterministic argmax inference. It
keeps automatic N-2 outages enabled, reports an automatic outage that immediately invalidates the
topology, and starts a new policy instance only when the episode terminates. Training likewise
preserves memory across rollout boundaries and clears only a vector slot whose episode terminated.

## Phase-2 training audit

The original phase-2 checkpoint was one 256x3 recurrent PPO policy with all 91 actions. It was
selected after unrestricted PPO curricula, reward-scale experiments, standard on-policy
fine-tuning, logit-temperature tests, and full-action imitation probes. Selection used 2019 data;
the fixed 2020 gate was reserved for finalists.

After recurrent-state handling was corrected, an additional mixed N-1/N-2 continuation and an
overloaded recovery curriculum were screened. Both were rejected because they improved neither
held-out performance nor the overall N-1/N-2 tradeoff, so that checkpoint was not replaced during
the phase-2 audit.

Fixed 256-episode held-out 2020 results (seed offset 50,000):

| Scenario / inference | PPO perf | PPO failures | Greedy perf | Lookahead perf |
|---|---:|---:|---:|---:|
| N-1 argmax | 0.9065 | 0.00% | 0.9153 | 0.9171 |
| N-1 stochastic | 0.9063 | 0.00% | 0.9153 | 0.9171 |
| N-2 argmax | 0.7978 | 0.78% | 0.8026 | 0.8049 |
| N-2 stochastic | 0.7898 | 1.56% | 0.8026 | 0.8049 |

The N-2 greedy and lookahead baselines each failed in 1.95% of episodes, so deterministic PPO was
more reliable but did not beat their safe-step performance. No masking, rollback, fallback,
forced no-op, or line-only restriction is used. The phase-2 model is therefore an honest strong
unrestricted result, not a baseline-beating claim.

## Fine-tuned incumbent

The phase-2 resource policy was subsequently fine-tuned by the reproducible unrestricted
curriculum in `ocean/power_grid/train_honest.sh`: 30M N-1 steps, 20M N-2 steps, and a 5M N-1
polish, all on the 2019 pool with fixed seeds and low learning rates. The curriculum uses all 91
actions without masks, rollback, greedy targets, or lookahead rewards. Its held-out 2020 AC and
DC results are recorded in `ocean/power_grid/README.md`.

## Final validation

The incumbent checkpoint is `resources/power_grid/policy.bin` (256 hidden units, three MinGRU
layers, SHA-256 `48b48f0194a4b3a2e19b5cf28332cbf0431e1edb80abad6d9f35e7b43d5a82f8`).
The phase-2 gate reran all four inference/baseline rows above on the fixed held-out set. The complete
power-grid Python/C suite passed (16 tests), and both the native CUDA float build and Emscripten web
build completed successfully. Temporary training, benchmark, web-build, and downloaded dependency
artifacts were removed after validation; unrelated user-owned sweep and tool directories were not
modified.
