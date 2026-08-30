# Fatal bug: stale CUDA backend produced an all-zero policy

## Summary

On 2026-08-27, power-grid GPU training and hypersweep results were found to be invalid because the loaded native CUDA extension was stale. The compiled `build/bindings.o` did not represent the current `src/bindings.cu` implementation. Fresh GPU policies were consequently saved with every parameter exactly zero and always selected action 0 (no-op).

This was not caused by the power-grid reward function or by an inability of the environment to execute switch actions. Recompiling the current CUDA source and relinking `pufferlib/_C.cpython-312-x86_64-linux-gnu.so` restored nonzero initialization, stochastic actions, switching, and learning.

## User-visible symptoms

- Every trained agent selected no-op.
- Sweep trials reported zero switches even after long training runs.
- Changing reward coefficients, policy size, layer count, and training duration did not restore switching.
- A fresh native-policy checkpoint contained 276,736 floats, all exactly zero.
- The behavior appeared superficially plausible because training continued at high SPS and emitted metrics without crashing.

## Root cause

The Python process imports the compiled native extension, not the C/CUDA source files directly. The extension had been linked from an out-of-date CUDA object. Source edits therefore did not affect the executable training path.

The stale backend's initialization path produced an all-zero policy. Equal zero logits make deterministic discrete action selection choose the first action, which is the power-grid no-op action. This made the problem look like a reward-design or exploration failure even though it occurred before meaningful learning could begin.

The normal project build path could not be used unchanged on this machine because:

- `clang` was unavailable;
- `ccache` was unavailable; and
- `/usr/include/python3.12/Python.h` was unavailable.

The usable Python headers were installed with the managed Python runtime at:

```text
/home/mastrmatt/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12
```

CUDA 13.3's compiler was available at:

```text
/usr/local/cuda-13.3/bin/nvcc
```

No changes were made to `build.sh` or to PufferLib core.

## Evidence that isolated the fault

### Before rebuilding

- Fresh checkpoint size: 276,736 float parameters.
- Nonzero parameters: 0.
- GPU rollout sample: 155,150 episodes.
- GPU score: 1.0 under the then-current metric behavior.
- GPU switches per episode: 0.
- CPU stochastic inference using the same zero weights sampled switch actions, proving the power-grid environment accepted and processed switches.

The CPU/GPU contrast localized the problem to the native GPU policy/execution path rather than the environment transition implementation.

### After rebuilding

- Fresh checkpoint parameters: 276,736.
- Nonzero parameters: 276,736.
- Parameter standard deviation: 0.040611383.
- Fresh GPU rollout sample: 94,412 episodes.
- Switches per episode: 3.0069 before training.
- Failure rate: 0.7657 before training, consistent with an untrained stochastic policy rather than forced no-op.

An 80-million-step corrected training run then produced:

- perf: 0.2476732;
- switches per episode: 5.3130;
- curriculum recovery success: 0.0914797; and
- total failure: 0.1676647.

The corrected hypersweep subsequently produced real switching trials. At 22 completed post-fix trials, its leader had:

- perf: 0.6274227;
- switches per episode: 5.0758;
- curriculum recovery success: 0.3504; and
- total failure: 0.0461.

## Repair performed

`src/bindings.cu` was compiled directly with CUDA 13.3 using the managed Python 3.12 headers. The resulting current `build/bindings.o` was linked into the Python native extension against the CUDA 13 libraries.

After repair, artifact timestamps confirmed the expected order:

```text
src/bindings.cu                         1786567699
build/bindings.o                        1787879042
pufferlib/_C.cpython-312-x86_64-linux-gnu.so 1787879058
```

The object is newer than the CUDA source and the linked extension is newer than the object.

Temporary compiler compatibility helpers were created only under `/tmp`; they are not part of the repository.

## Impact on experiment results

All power-grid GPU training and hypersweep conclusions obtained with the stale extension are invalid, especially conclusions that the agent preferred no-op or that reward changes could not motivate switching. Those trials measured a broken all-zero execution path, not the learning problem.

Only experiments started after the 2026-08-27 rebuild should be compared or used to select hyperparameters. The corrected sweep was restarted from scratch with `puffer sweep power_grid`.

## Validation performed

The corrected backend was checked with fresh-weight statistics and GPU rollout action diversity. The targeted power-grid correctness suite also passed 16/16 tests, covering:

- double- and float-precision DC solving;
- AC solving;
- double- and float-precision environment behavior;
- the user controller;
- deployed policy architecture;
- scenario behavior; and
- Puffer training memory behavior.

Because this machine lacks `clang`, the native test wrappers were run through a temporary GCC compatibility wrapper with the same strict C flags. The only additional GCC suppression was for an existing unused parameter in Puffer core that the source already suppresses through a Clang-specific pragma.

## Prevention and mandatory preflight

After any change to native C/CUDA code, rebuild before training. Do not assume importing successfully means the extension is current.

Before starting an expensive train or sweep, verify all of the following:

1. The CUDA object is newer than every native source it contains.
2. The linked `_C` extension is newer than its object files.
3. A newly initialized checkpoint has finite parameters, nonzero standard deviation, and is not all zero.
4. A short stochastic GPU rollout emits more than one action and includes non-no-op actions.
5. Power-grid rollout metrics report nonzero switch actions for an untrained stochastic policy.
6. The targeted native and Python test suite passes.

Useful timestamp check:

```bash
stat -c '%Y %n' src/bindings.cu build/bindings.o pufferlib/_C*.so
```

This timestamp check is necessary but not sufficient: a newly built yet behaviorally broken binary can still be newer. The fresh-weight and action-diversity smoke tests are the decisive checks.

## Recommended automated guard

Add a pre-sweep smoke test outside Puffer core that fails immediately when any of these conditions is true:

- a native source is newer than the loaded extension;
- initialized parameters are non-finite;
- every initialized parameter is zero;
- initialized parameter standard deviation is effectively zero; or
- a sufficiently large stochastic GPU action sample contains only action 0.

The smoke test should run before allocating hours to a sweep and should print the loaded extension path and artifact timestamps so that a stale or shadowed module is immediately visible.

## Additional audit findings (2026-08-27)

### Confirmed: replacing the native extension during a live sweep can stall it

While auditing the build toolchain, `build.sh` relinked `pufferlib/_C.cpython-312-x86_64-linux-gnu.so` in place while a power-grid sweep parent was still running. The already-running worker remained valid because it had mapped the previous extension, but a newly spawned trial attempted to import the file while it was being replaced. That worker exited with an extension import failure, while the sweep parent remained blocked waiting for a result that could never arrive.

The parent process was stopped, the completed 16 trials were preserved in `logs/power_grid`, the final linked extension was import-tested and GPU-smoke-tested, and the sweep was restarted. The restarted sweep completed 21 valid trials before this audit entry was written.

Prevention: never rebuild or overwrite `_C.so` while a train or sweep using that checkout is active. Stop the process, rebuild, run the fresh-weight/action smoke test, and then restart. A more robust build system should link to a temporary output and atomically rename the completed extension into place, and the sweep controller should detect worker death instead of waiting forever.

### Confirmed: two reporting metrics use the wrong denominator for short episodes

`power_grid_finish_episode` divides both `demand_fulfilled` and `overloaded_line_fraction` by the fixed `POWER_GRID_EPISODE_STEPS` value of 72. Curriculum episodes and failed episodes often terminate earlier, so these fields are underreported relative to their actual episode duration.

A deterministic four-step overloaded episode produced:

```text
episode steps:             4
overloaded line-steps:     4
logged overload fraction:  0.002777778
actual episode fraction:   0.050000000
logged / actual:           0.0556 (18x too small)
```

This was a dashboard/reporting defect. It did not affect rewards, policy inputs, failure handling, the primary `perf` metric, or hypersweep selection. It was fixed by normalizing both episode-fraction metrics by the actual guarded episode step count.

### Cross-checkout audit after copying repositories

The other native PufferLib checkouts were rebuilt from their current source and behavior-smoke-tested:

| Checkout | Fresh parameters | Nonzero | Finite | Standard deviation |
|---|---:|---:|---:|---:|
| `kaggriculture` | 128,288 | 128,288 | yes | 0.056214 |
| `conformer_generation` | 3,283,968 | 3,283,968 | yes | 0.024513 |
| `pufferlib_vs_torsionnet/pufferlib` | 3,527,680 | 3,527,680 | yes | 0.023855 |

None reproduced the all-zero policy defect.

Both conformer checkouts did contain copied, machine-specific generated state:

- CMake caches hard-coded to the missing laptop path `/home/mastrmatt/learn-puffer/.micromamba/envs/rdkit-cpp`;
- RDKit molecule pickles serialized by version 16.3, which the installed version-15.0 reader could not safely deserialize.

Those generated caches were moved to `/tmp` rather than deleting source or user results. Compatible CMake state and molecule caches were regenerated locally, and both conformer environments completed one-million-step GPU smoke runs with finite, nonzero checkpoints.

### Global build prerequisites installed

The system now has the host-side dependencies needed by the unmodified Puffer build path: build-essential, Clang 18, ccache, CMake, Python 3.12 headers, OpenMP, Ninja, pkg-config, CUDA 13.3 development libraries, cuDNN development files, NCCL development files, RDKit development files, Boost development files, Eigen, CoordGen, MaeParser, and system NumPy headers.

The unmodified command below now completes successfully in the power-RL checkout:

```bash
./build.sh power_grid
```
