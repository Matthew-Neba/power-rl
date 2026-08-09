# Power Grid policy benchmark — 2026-08-08

PPO was configured for 40 million steps and finished at the final rollout boundary of 39,976,960
steps. It trained for 2m23s on the 65% historical / 35% synthetic DC curriculum with a 25% chance
of an exogenous persistent line outage. The corrected-event checkpoint is:

```text
checkpoints/power_grid/1786231242192/0000000039976960.bin
```

Every comparison below uses AC physics, the same environment seeds 0–511, held-out 2020 days, and
exactly one 72-step episode per vector slot. `perf` is the fraction of AC-secure steps, forced to
zero after catastrophe. `score` is the no-op fraction, and return is the raw environment return.

## AC control: no random outage

| Policy | Perf | Score | Return | Failure | Event failure | Switches |
|---|---:|---:|---:|---:|---:|---:|
| PPO agent | 0.9582 | 0.9394 | 13.758 | 0.0039 | 0.0000 | 4.346 |
| No action | 0.9230 | 1.0000 | 13.236 | 0.0195 | 0.0000 | 0.000 |
| Uniform random | 0.0000 | 0.0074 | -5.017 | 1.0000 | 0.0000 | 3.484 |
| Random lines | 0.0000 | 0.0418 | -5.108 | 1.0000 | 0.0000 | 4.393 |
| Safe random | 0.0006 | 0.0114 | -5.452 | 0.9980 | 0.0000 | 11.461 |
| Greedy lines | 0.9650 | 0.9940 | 13.889 | 0.0078 | 0.0000 | 0.428 |
| Greedy all | 0.9662 | 0.9938 | 13.908 | 0.0078 | 0.0000 | 0.439 |

## AC stress: every episode schedules a random outage

The failed line and its later period are randomized from the shared seed. The event applies even
after a policy changes topology, preventing policies from avoiding the contingency by switching
early. Policies that fail before the scheduled period can still record no applied event.

| Policy | Perf | Score | Return | Failure | Event failure | Switches | Applied events |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO agent | 0.7186 | 0.8502 | 10.575 | 0.1992 | 0.0234 | 7.574 | 0.9980 |
| No action | 0.6748 | 1.0000 | 9.369 | 0.2168 | 0.0117 | 0.000 | 0.9980 |
| Uniform random | 0.0000 | 0.0074 | -5.019 | 1.0000 | 0.0020 | 3.480 | 0.0215 |
| Random lines | 0.0000 | 0.0418 | -5.110 | 1.0000 | 0.0039 | 4.383 | 0.0156 |
| Safe random | 0.0000 | 0.0114 | -5.466 | 1.0000 | 0.0566 | 10.539 | 0.1289 |
| Greedy lines | 0.7623 | 0.9818 | 11.060 | 0.1621 | 0.0156 | 1.105 | 0.9980 |
| Greedy all | 0.7619 | 0.9806 | 11.064 | 0.1660 | 0.0176 | 1.168 | 0.9980 |

On the no-outage control, PPO improves perf by 0.0352 over no-action but remains 0.0080 below
greedy-all. Under guaranteed random outages, PPO improves perf by 0.0438 and failure rate by 0.0176
over no-action. Greedy-lines remains strongest: 0.0437 higher perf, 0.0371 fewer failures, and 6.47
fewer switches per episode than PPO.

The original checkpoint trained with policy-dependent event skipping scored 0.6726 on the forced
AC outage suite. Retraining with exogenous events raised this to 0.7186, a 0.0460 absolute gain.
The remaining gap indicates that PPO has learned useful contingency response, but its switching
policy is still less efficient and less reliable than immediate greedy branch control on this grid.

Reproduce both suites with:

```sh
uv run ./build.sh power_grid
uv run python ocean/power_grid/benchmark.py \
  --checkpoint checkpoints/power_grid/1786231242192/0000000039976960.bin \
  --episodes 512 --physics ac --random-event-probability 0
uv run python ocean/power_grid/benchmark.py \
  --checkpoint checkpoints/power_grid/1786231242192/0000000039976960.bin \
  --episodes 512 --physics ac --random-event-probability 1
```
