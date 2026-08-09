# IEEE-118 Power Grid policy benchmark — 2026-08-08

The PPO policy was trained from scratch with the IEEE-118 environment for a configured 40 million
DC agent steps. Training reached the final rollout boundary at 39,976,960 steps in 2,550 seconds
(42m30s). The final checkpoint is:

```text
checkpoints/power_grid/1786239474232/0000000039976960.bin
```

The run used 1,024 agents, a 128-step rollout horizon, a 512-unit three-layer policy, the cached
2019 training pool, a 90% historical / 10% synthetic curriculum, and a 50% probability of three
distinct persistent line outages. Its post-training DC evaluation over 10,083 episodes reported
`perf = 0.6755`, `total_failure = 0.2313`, and 64.21 physical switches per episode.

## Comparison protocol

Both comparisons below use:

- the same held-out 2020 historical operating days and environment seeds 0–63;
- one 72-step episode per seed;
- AC power flow for the environment's realized trajectory and reported security metrics;
- no evaluation return metric;
- DC candidate screening for greedy and safe-random baselines, so they do not receive an expensive
  AC-solver oracle unavailable to PPO; and
- eight deterministic CPU shards whose weighted results were checked against an unsharded run.

`perf` is the fraction of AC-secure steps, forced to zero after a catastrophic termination. `score`
is the no-action fraction. `failure` is the fraction of episodes terminated by an invalid,
disconnected, islanded, singular, non-finite, or otherwise unsolved topology. `outage completion`
is the fraction of the three scheduled outages reached before termination, and `all survived` is
the fraction of episodes that remained alive through all three.

## Nominal AC control: no scheduled outages

| Policy | Perf | Score | Failure | Switches | Demand fulfilled | Trip episode | Peak rho | Overloaded line-steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO agent | 0.3579 | 0.0000 | 0.6094 | 44.578 | 61.07% | 0.00% | 0.998 | 0.057% |
| No action | 0.6367 | 1.0000 | 0.0156 | 0.000 | 99.67% | 9.38% | 1.097 | 0.387% |
| Uniform random | 0.0000 | 0.0000 | 1.0000 | 4.625 | 5.03% | 1.56% | 0.826 | 0.011% |
| Random lines | 0.0000 | 0.0015 | 1.0000 | 10.453 | 13.17% | 18.75% | 1.146 | 0.039% |
| Safe random | 0.0247 | 0.0003 | 0.9531 | 33.141 | 44.75% | 60.94% | 1.595 | 0.206% |
| Greedy lines | 0.7747 | 0.9644 | 0.0000 | 2.562 | 100.00% | 3.12% | 1.060 | 0.196% |
| Greedy all | 0.7747 | 0.9644 | 0.0000 | 2.562 | 100.00% | 3.12% | 1.060 | 0.196% |

PPO underperforms no-action by 0.2788 perf and greedy by 0.4168. Its 60.94% failure rate and
44.58 switches per episode show that it learned a highly active policy that frequently destroys a
valid topology. Its low peak loading is therefore not evidence of successful congestion control:
many episodes terminate before serving their full demand trajectory.

## Forced AC stress: three scheduled outages in every episode

| Policy | Perf | Failure | Event failure | Switches | Applied outages | Demand fulfilled | Outage completion | All survived | Trip episode | Peak rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO agent | 0.3032 | 0.6406 | 0.0469 | 42.391 | 1.781 | 59.14% | 59.38% | 35.94% | 1.56% | 1.007 |
| No action | 0.5677 | 0.0625 | 0.0156 | 0.000 | 2.953 | 98.05% | 98.44% | 93.75% | 14.06% | 1.131 |
| Uniform random | 0.0000 | 1.0000 | 0.0156 | 4.594 | 0.047 | 5.01% | 1.56% | 0.00% | 1.56% | 0.827 |
| Random lines | 0.0000 | 1.0000 | 0.0156 | 10.406 | 0.281 | 13.15% | 9.38% | 0.00% | 18.75% | 1.144 |
| Safe random | 0.0126 | 0.9844 | 0.1094 | 30.844 | 1.172 | 41.78% | 39.06% | 1.56% | 64.06% | 1.587 |
| Greedy lines | 0.6758 | 0.0781 | 0.0469 | 2.641 | 2.938 | 97.48% | 97.92% | 92.19% | 6.25% | 1.083 |
| Greedy all | 0.6758 | 0.0781 | 0.0469 | 2.641 | 2.938 | 97.48% | 97.92% | 92.19% | 6.25% | 1.083 |

Under forced contingencies, PPO underperforms no-action by 0.2645 perf and greedy by 0.3726. PPO
reaches only 59.38% of scheduled outages and survives all three in 35.94% of episodes. No-action
and greedy both exceed 92% full-outage survival. Greedy obtains the best secure-step performance
and cuts AC thermal-trip episodes from no-action's 14.06% to 6.25%, although its 7.81% total
failure rate is slightly higher than no-action's 6.25% on this 64-seed sample.

Greedy-lines and greedy-all are identical in both suites, meaning the immediate greedy objective
never preferred a busbar-terminal or coupler operation over its branch-only choices on these
scenarios.

## Conclusion

The IEEE-118 transfer and scaled contingency machinery work, but the inherited IEEE-14 PPO setup
does not transfer successfully in this single 40-million-step run. The action space grew to 830
choices, and the learned policy converged toward near-continuous switching rather than preserving a
valid topology. On held-out AC operation, no-action is a substantially stronger baseline and the
DC-screened greedy branch policy is the clear winner. A subsequent training iteration should first
address action validity and switching behavior (for example action masking or a hierarchical
no-op/branch-selection policy) and should retain intermediate checkpoints; simply extending this
final policy's evaluation does not change the observed failure mechanism.

Reproduce the two suites from the IEEE-118 worktree with:

```sh
uv run ./build.sh power_grid
uv run python ocean/power_grid/benchmark.py \
  --checkpoint checkpoints/power_grid/1786239474232/0000000039976960.bin \
  --episodes 64 --physics ac --random-event-probability 0 --random-outages 3 --jobs 8
uv run python ocean/power_grid/benchmark.py \
  --checkpoint checkpoints/power_grid/1786239474232/0000000039976960.bin \
  --episodes 64 --physics ac --random-event-probability 1 --random-outages 3 --jobs 8
```
