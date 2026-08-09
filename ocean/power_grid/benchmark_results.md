# IEEE-118 Power Grid policy benchmark — 2026-08-08

The reference PPO policy was trained from scratch with the IEEE-118 environment for a configured 40
million DC agent steps. Training reached the final rollout boundary at 39,976,960 steps in 2,550
seconds (42m30s). It used the original low-credit-assignment settings `gamma = 0.8` and
`gae_lambda = 0.6119008620196786`; its checkpoint is:

```text
checkpoints/power_grid/1786239474232/0000000039976960.bin
```

The run used 1,024 agents, a 128-step rollout horizon, a 512-unit three-layer policy, the cached
2019 training pool, a 90% historical / 10% synthetic curriculum, and a 50% probability of three
distinct persistent line outages. Its post-training DC evaluation over 10,083 episodes reported
`perf = 0.6755`, `total_failure = 0.2313`, and 64.21 physical switches per episode.

The controlled long-horizon experiment changed only those two values to `gamma = 0.99` and
`gae_lambda = 0.95`. It also reached 39,976,960 steps, taking 2,518 seconds (41m58s). Its
checkpoint is:

```text
checkpoints/power_grid/1786243504654/0000000039976960.bin
```

Its post-training DC evaluation over 11,123 episodes reported `perf = 0.6115`,
`total_failure = 0.3201`, and 58.72 physical switches per episode.

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

## Credit-assignment experiment: long-horizon PPO

The long-horizon checkpoint was replayed on the same 64 held-out seeds and AC settings as the
reference checkpoint. The invariant baselines above therefore remain the comparison points.

| AC suite | Reference PPO (`gamma=.8`, `lambda=.6119`) | Long-horizon PPO (`gamma=.99`, `lambda=.95`) | Change |
|---|---:|---:|---:|
| Nominal perf | 0.3579 | **0.4891** | +0.1312 |
| Nominal failure | 60.94% | **45.31%** | −15.63 pp |
| Nominal switches | 44.58 | 55.13 | +10.55 |
| Forced-outage perf | 0.3032 | **0.4388** | +0.1356 |
| Forced-outage failure | 64.06% | **50.00%** | −14.06 pp |
| Forced-outage completion | 59.38% | **71.87%** | +12.49 pp |
| Full three-outage survival | 35.94% | **50.00%** | +14.06 pp |
| Forced-outage switches | 42.39 | 52.88 | +10.48 |

Favoring longer returns helped substantially on both AC suites, especially contingency completion
and survival. It did not make PPO competitive with the invariant baselines: nominal no-action and
greedy perf remain 0.6367 and 0.7747, while forced-outage no-action and greedy remain 0.5677 and
0.6758. The improvement came with roughly ten additional switches per episode, and the training
dashboard showed unusually high KL values (roughly 0.4–0.6) and entropy collapse. This suggests
that `gamma`/GAE were part of the problem, but the unchanged learning rate is now too aggressive for
the longer-horizon targets.

## Conclusion

## Best screened PPO: horizon 64, conservative optimizer

The best verified checkpoint was trained for 39,976,960 saved steps with
`horizon=64`, `gamma=.99`, `gae_lambda=.95`, `learning_rate=.0025`,
`ent_coef=.0001`, and training event probability `.25`:
`checkpoints/power_grid/1786250866399/0000000039976960.bin`.

| AC suite | PPO perf | Failure | Switches | Demand fulfilled | Outage completion | All survived |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | **0.7613** | 3.12% | 71.19 | 98.83% | — | — |
| Forced outages | **0.6595** | 10.94% | 69.17 | 95.96% | 96.88% | 89.06% |

This beats no-action on both suites (0.6367 / 0.5677) and is close to, but does
not yet exceed, greedy (0.7747 / 0.6758) on the 64 held-out seeds. Dashboard
timing stayed CPU/environment dominated: approximately 71–76% environment,
19–21% train/forward, and 2% GPU evaluation.

The IEEE-118 transfer and scaled contingency machinery work. Moving to longer-horizon credit
assignment improved PPO materially, but the inherited optimizer still produces near-continuous
switching and remains behind no-action and greedy on held-out AC. The next controlled experiment
should retain `gamma = 0.99`/`gae_lambda = 0.95` while lowering the learning rate and/or adding
action validity masking; changing credit assignment alone is not enough to make PPO competitive.

Reproduce the two suites from the IEEE-118 worktree with:

```sh
uv run ./build.sh power_grid
uv run python ocean/power_grid/benchmark.py \
  --checkpoint checkpoints/power_grid/1786243504654/0000000039976960.bin \
  --episodes 64 --physics ac --random-event-probability 0 --random-outages 3 --jobs 8
uv run python ocean/power_grid/benchmark.py \
  --checkpoint checkpoints/power_grid/1786243504654/0000000039976960.bin \
  --episodes 64 --physics ac --random-event-probability 1 --random-outages 3 --jobs 8
```
