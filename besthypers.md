# Best Power Grid Hypersweep Configuration

Source: `logs/power_grid/1787946504819.json` from the completed 1,200-run
Protein sweep on 2026-08-28. The sweep trained with DC physics and optimized
`env/perf`. This was the best final sweep observation.

## Winning hyperparameters

| Parameter | Value |
|---|---:|
| `train.total_timesteps` | `46680868` |
| `policy.hidden_size` | `512` |
| `policy.num_layers` | `1` |
| `train.learning_rate` | `0.0173757567694621` |
| `train.ent_coef` | `0.2` |
| `train.gamma` | `0.999762479114525` |
| `train.gae_lambda` | `0.854647393728266` |
| `env.curriculum_safe_probability` | `0.1` |
| `env.curriculum_sequence_probability` | `0.165270979465442` |
| `env.reward_congestion_cost_weight` | `100` |
| `env.reward_switch_penalty` | `0.000274060799681267` |

The remaining PPO settings came from `config/power_grid.ini`: MinGRU,
`minibatch_size=18432`, `clip_coef=0.916769218350675`, `vf_coef=0.1`,
`vf_clip_coef=0.01`, `max_grad_norm=5`, `beta1=0.999`,
`beta2=0.990076949045561`, `eps=6.13270494313886e-05`,
`prio_alpha=0.193588313321192`, and `prio_beta0=1`.

## Results

- DC sweep observation: `perf=0.8765375`, `score=0.5610115`, 46.67M steps.
- From-scratch full-AC retrain checkpoint:
  `ac_best_hypers/checkpoints/power_grid/1787954758254/0000000046669824.bin`.
- Held-out AC N-1 (10,000 episodes): `perf=0.7121`, survival `80.41%`.
- Held-out AC N-2 (10,000 episodes): `perf=0.4267`, survival `52.77%`.
- AC N-2 greedy-all baseline: `perf=0.4866`, survival `58.45%`.

The AC numbers are from a fresh seed-equivalent retrain, not the unsaved sweep
weights. Stage two targets the remaining N-2 gap using from-scratch training.
