# Full-AC multi-seed sweep confirmation

The five configurations with the highest final `env/perf` in the completed 1,200-trial DC Protein
sweep were retrained from scratch with full AC physics. Each configuration used three independent
training/environment seeds (6101, 6102, and 6103) and retained its sweep-selected step budget.
Every final checkpoint was evaluated on the same held-out 2020 scenarios: 2,000 N-1 and 2,000 N-2
episodes with evaluation seeds 0 through 1,999.

| Sweep run | N-1 perf mean | N-2 perf mean | Combined perf mean | Combined sample SD |
|---|---:|---:|---:|---:|
| 1787946504819 | 0.6733 | 0.3888 | **0.5311** | **0.0241** |
| 1787944230806 | 0.6376 | 0.3582 | 0.4979 | 0.0883 |
| 1787943742161 | 0.6365 | 0.3524 | 0.4944 | 0.0651 |
| 1787933403357 | 0.5982 | 0.3287 | 0.4635 | 0.0775 |
| 1787935057966 | 0.5809 | 0.3218 | 0.4514 | 0.0929 |

The original sweep winner, 1787946504819, remains selected. It has both the highest mean held-out
AC performance and the lowest seed sensitivity. Its exact hyperparameters are stored in
`config/power_grid.ini`, now with `ac_power_flow = True` so plain training uses full AC physics.

Training JSON and checkpoints are under `confirmation_ac/logs` and `confirmation_ac/checkpoints`.
Per-run held-out reports are under `confirmation_ac/eval`.
