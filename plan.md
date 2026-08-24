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
| Environment actions | 106 |
| PPO actions | 106 (all environment actions) |
| Float observations | 236 |
| Training episode | 72 steps / 12 periods |
| Interactive runtime | 72-step episodes; fresh environment and policy after terminal |

PPO can select every line, terminal-busbar, coupler, load-shed, and non-slack-generator-trip
action. Invalid topology commands and
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

Training now guarantees one or two outages per episode, covering N-1 and N-2 while matching the
live N-2 limit. No line or pair is removed from the outage distribution for feasibility.
The interactive controller in `power_grid_user.h` accepts at most two user-selected outages,
applies them as exogenous events, and never rolls an infeasible request back. Clearing an outage
makes its line available without overwriting topology changes made by the policy. This state and
mouse input are separate from the training environment. The app uses ordinary 72-step episodes.
A terminal state starts a fresh environment episode and a newly initialized policy instance.

The web app offers both stochastic Puffer-style sampling and deterministic argmax inference. It
runs held-out 2020 days with AC physics, defaults to argmax, and disables automatic outages so the
one or two clicked lines are the complete contingency. It reports infeasible clicks as grid
failures instead of silently filtering them. Training likewise preserves memory across rollout
boundaries and clears only a vector slot whose episode terminated.

## Phase-2 training audit

The original phase-2 checkpoint was one 256x3 recurrent PPO policy with all 91 legacy actions. It was
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

## Emergency-control incumbent

The phase-2 policy was expanded deterministically from 221/91 to 236/106, adding explicit load and
generator connected-state inputs plus agent-selected emergency controls. Browser QA then caught
anticipatory load shedding, so the reproducible `ocean/power_grid/train_honest.sh` AC curriculum
now includes intact-grid preservation, balanced N-1/N-2 recovery, full-day batches mixing
reset-time with naturally delayed outages, and consecutive-click training. An offline AC teacher
then fits the same decoder on 2019 recovery states and one aggregated policy-state pass. The
deployed model still uses every action without masks, rollback, a runtime planner, forced no-op,
click-triggered recurrent reset, or fallback control.

## Final validation

The incumbent checkpoint is `resources/power_grid/policy.bin` (256 hidden units, three MinGRU
layers, SHA-256 `6a91a3a65f1675a8bd300417d6535929a3291c4ea0505eb7f29c23e21a75c0c2`).
Its exhaustive confirmation gate uses 8 untouched 2020 contexts after selection on a disjoint
8-context subset and covers all 20 N-1 and 190 N-2 requests. Combined survival is 79.94%, handled
rate is 56.25%, secure-step rate is 73.51%, and demand served is 97.47%; no trial issues a
pre-click emergency command.
Therefore the web prototype is implemented, but the requested 95% handled performance gate remains
open. The current 14-test Python/C suite, native build, Emscripten build, exhaustive AC QA, and
headless-browser click test pass. GitHub Pages deploys the research prototype at
`https://matthew-neba.github.io/power-rl/`; the deployed HTTPS/WASM build also passed a held-click
7-8 bridge-outage test with secure AC recovery, 100% load served, and no browser errors. Product
readiness remains open while the 95% policy-performance gate is unmet.
