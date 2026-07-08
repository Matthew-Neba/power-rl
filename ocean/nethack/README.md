# NetHack environment for PufferLib

C-level NLE binding for the PufferLib RL framework. Each env owns a
per-env `nle_ctx_t` holding all of NetHack's mutable game state, with a
private 64 MB memory arena. `libnethack.so` is linked directly (no
dlopen). Auto-dismiss for prompts, fixed 1070-byte observation (glyph crop
+ blstats + cooldown/prev-action/inventory stats), reward shaping,
multi-threaded OMP stepping.

---

## Full setup (from scratch on any HPC)

### 1. Clone PufferLib + vendored NLE

```bash
git clone https://github.com/PufferAI/PufferLib.git && cd PufferLib
git checkout 4.0

# Clone the modified NLE into vendor/nle
git clone https://github.com/liujonathan24/NetHack.git vendor/nle
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[nethack]"    # installs pufferlib + nethack deps (torch, numpy, wandb, etc.)
```

If `[nethack]` extras aren't defined yet, use:
```bash
pip install -e .
pip install bz2file           # if system libbz2 headers missing
```

### 3. System modules (cluster-specific)

You need a C compiler (clang preferred, gcc works), CUDA toolkit, and
an OpenMP runtime. On Princeton Della:

```bash
module load cudatoolkit/12.8 intel-oneapi/2024.2
```

On other clusters, find equivalents for:
- **CUDA**: `nvcc` for GPU training backend
- **OpenMP**: `libiomp5.so` (Intel) or `libgomp.so` (GCC) — needed at link and runtime
- **clang** (optional but preferred): `build.sh` uses clang flags by default. Set `CC=gcc` if clang isn't available, but note `-ferror-limit` must be removed from `build.sh`.

### 4. Build libnethack.so

```bash
# First-time only: configure cmake
cd vendor/nle/src
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cd ../../../..

# Build (always from repo root)
make -C vendor/nle/src/build nethack -j$(nproc)
```

Produces `vendor/nle/src/build/libnethack.so` and data files in
`vendor/nle/src/build/dat/` (including `nhdat`).

If cmake fails with missing `bz2`, install libbz2-dev
(`apt install libbz2-dev` or `yum install bzip2-devel`).

### 5. Build PufferLib C extension (_C.so)

```bash
bash build.sh nethack          # auto-detects CUDA; falls back to CPU
```

Produces `pufferlib/_C.cpython-*.so`. If you get linker errors about
`-liomp5`, make sure the Intel OpenMP module is loaded.

### 6. Set NETHACKDIR

Point to the directory containing `nhdat`:

```bash
export NETHACKDIR=$(pwd)/vendor/nle/src/build/dat
```

### 7. Verify

```bash
# Quick training test (runs on login node if GPU available)
puffer train nethack \
    --vec.total-agents 64 --vec.num-buffers 1 --vec.num-threads 4 \
    --train.gpus 1 --train.total-timesteps 1000000 --train.minibatch-size 4096

# Interactive play
./live_view -i

# Random agent viewer
./live_view --random --steps 200
```

### 8. SLURM training

```bash
sbatch sweep_nethack.slurm     # hyperparameter sweep
# or single run:
puffer train nethack --wandb \
    --vec.total-agents 4096 --vec.num-buffers 1 --vec.num-threads 16 \
    --train.gpus 1 --train.total-timesteps 1000000000
```

### Rebuild after code changes

```bash
make -C vendor/nle/src/build nethack -j$(nproc)   # if vendor/nle changed
bash build.sh nethack                               # always (relinks _C.so)
```

### Pulling NLE updates

The modified NLE lives in a separate repo. To pull latest changes:

```bash
cd vendor/nle
git pull origin main
cd ../..
make -C vendor/nle/src/build nethack -j$(nproc)
bash build.sh nethack
```

To push NLE changes (after editing files under `vendor/nle/`):

```bash
cd vendor/nle
git add -A && git commit -m "description of changes"
git push origin main
cd ../..
```

Note: `vendor/nle/` has its own `.git` — it is NOT tracked by the
PufferLib repo. Add `vendor/nle` to PufferLib's `.gitignore` if it
isn't already.

### Troubleshooting

| Error | Fix |
|---|---|
| `clang: command not found` | `module load llvm` or set `CC=gcc` |
| `cannot find -liomp5` | `module load intel-oneapi/2024.2` (or equivalent) |
| `libiomp5.so: cannot open shared object file` | Same module, also at runtime |
| `cannot allocate memory in static TLS block` | Too many `__thread` vars — rebuild libnethack |
| `NETHACKDIR is misconfigured` | `export NETHACKDIR=$(pwd)/vendor/nle/src/build/dat` |
| `ZeroDivisionError` in train | Need `--train.gpus 1` (even on CPU-only, the CUDA build requires it) |
| Core dump / segfault at T>1 | Check that libnethack.so was rebuilt after latest source changes |

---

## Observation (fixed, 1070 bytes)

A single flat `ByteTensor` per agent:

| Slice     | Offset | Bytes | Encoding                                    |
|-----------|-------:|------:|---------------------------------------------|
| `glyphs`  |      0 |   882 | 21x21 egocentric crop, int16 LE per cell     |
| `blstats` |    882 |   108 | 27 stats, int64 truncated to int32 LE        |
| `extra`   |    990 |    80 | 20 stats, int32 LE (see below)               |

The crop is centered on the agent and padded with `NO_GLYPH` (5976)
outside the map. The extra segment carries 20 env-derived stats: the
prayer cooldown (`u.ublesscnt`, exported by a vendored NLE patch —
makes pray timing learnable instead of superstition), the previous
action index (-1 at episode start; a 64-dim recurrent trunk otherwise
has to reverse-engineer what it just did from obs deltas), and 18
per-object-class inventory item counts (stack = 1; tells the policy
*when* WEAR/EAT are worth pressing — the macros choose the item). This
layout is exactly what the custom CUDA encoder in `src/ocean.cu`
expects (glyph embedding -> 2 convs, blstats+extra normalized/expanded
to 88 features -> linear, concat -> projection); it is gradient-checked
by `tests/test_nethack_encoder.py`. `chars`, `message`, and
`inv_glyphs/letters/oclasses` are additionally bound to side buffers
for the standalone tools, the prompt heuristic, and the item macros,
but never enter the obs tensor raw (`inv_strs` stays unbound: it alone
triggers NetHack's doname() string formatting).

Three alternative encoders are selected via `NETHACK_ENCODER` at train
time (default = conv), all gradient-checked by
`tests/test_nethack_mixer.py`:

- `patch`: embed(16) -> shared non-overlapping 3x3 patch linear ->
  flatten || blstats -> projection. Speed parity with the cuDNN conv
  path with far less machinery (train phase 707 vs 715 ms/epoch at
  minibatch 8192); single-tap embedding backward. Compile-time knobs
  NHM_D/NHM_C trade capacity for further speed.
- `minpatch`: patch stem + learned token pool (49 spatial tokens ->
  16) -> flatten || blstats -> projection. Preserves the local symbolic
  patch primitive while shrinking spatial features from 3136 to 1024.
- `mixer`: patch stem + one norm-free MLP-Mixer block (full-crop
  token mixing). Trains correctly but ~30% lower SPS than conv —
  bandwidth-bound elementwise passes; exists for score-per-sample
  A/B, not speed.

Checkpoints are not interchangeable between encoders, and the CPU
demo only supports the conv encoder.

## Action space (24 actions)

```
 0  N            8  N_RUN         16  >  (down)
 1  S            9  S_RUN         17  <  (up)
 2  W           10  W_RUN        18  kick (^D)
 3  E           11  E_RUN        19  search (s)
 4  NW          12  NW_RUN       20  pray (M-p)
 5  NE          13  NE_RUN       21  engrave Elbereth (macro)
 6  SW          14  SW_RUN       22  wear armor (macro)
 7  SE          15  SE_RUN       23  eat (macro)
```

The 8 cardinal/intercardinal moves use vi-keys (kjhl ynbu). Long
"run" versions are uppercase (KJHL YNBU). Kick prompts "In what
direction?"; that prompt is left live, so the agent's next action (a
movement key) answers it — kick-through-a-locked-door is a learnable
two-step sequence. Search checks adjacent squares once for hidden
doors/passages (repeated searching raises the find chance, as in the
real game). Pray's "Are you sure?" confirm is auto-answered 'y'; the
gods heal low HP and cure starving, but praying too often angers them
— when to pray is the thing to learn. Engrave-Elbereth drives the full
key sequence (`E`, `-` fingertip, "Elbereth", RET) in one env step;
most melee monsters won't attack while the agent stands on a legible
Elbereth square, but attacking from it smudges the text — a panic
button, not a fortress.

Wear and eat are getobj macros: the command's own prompt ("What do you
want to wear? [bc or ?*]") already lists only the items valid for the
verb, so the macro answers with a random listed letter and lets the
game resolve it. Random, not first: an item the game refuses (a second
cloak, a blocked slot) stays in the list, and a first-letter pick would
retry it forever. Eat is gated on hunger: pressing it while Satiated is
a no-op (the game step is skipped entirely) because eating past
Satiated is NetHack's choke() death — ungated, EAT-happy policies
choke themselves in a handful of presses. Corpses are never eaten:
floor offers ("There is a lichen corpse here; eat it?") are declined
and autopickup'd corpses are filtered out of the candidate list by
glyph — corpse age is invisible and old or poisonous ones kill.
Nothing wearable/edible -> no prompt -> clean no-op, no penalty.
Autopickup is set to `$[%` (gold, armor, food) to feed both macros.
The wear macro doesn't rank items — a body-armor suit can get worn
(monks fight worse in suits) and cursed gear sticks; both are rare and
part of the game. No wield action on purpose: monks fight better
bare-handed, so a weapon slot would be a trap button.

## Reward shaping (config-tunable via nethack.ini)

```
reward = exp_coef      * dlog1p(exp points)             # dense per-kill, tail-compressed
       + descent_coef  * (new episode-max depth only)   # yo-yo-proof descent bonus
       + xp_coef       * (new episode-max XL only)      # power progression before descent
       + scout_coef    * (new_tile_this_level)          # exploration bonus
       + hp_coef       * (hp - prev_hp)                 # HP potential (dense survival signal)
       + illegal_penalty * (illegal_action)             # sub-prompt penalty
```

The terminal step of a game-over (not truncation) carries
`death_penalty - hp_coef * prev_hp` instead of the shaped terms — the
HP potential cashes out, so dying at high HP forfeits more.

NetHack's score is deliberately NOT rewarded: `botl_score()` is
gold + exp + 50*depth, a fixed mix that overrode the coefs above
(depth at 50x descent_coef) and saturated the trainer's [-1,1] reward
clamp on every eventful step, silently truncating any term riding on a
kill or descent. Its useful component (exp) is the explicit `exp_coef`
term — the log-delta form is potential-based (phi = log1p(exp)) and
keeps one big kill from hitting the clamp. Score remains the `perf`
log metric. `reward_saturated` tracks the fraction of steps at |r| > 1;
it should stay ~0 — if it climbs, a reward term is being clipped again.

All coefficients are set in `config/nethack.ini` under `[env]`.

## Auto-dismiss hook

After each agent action, the harness inspects `misc[]`
(`in_yn_function`, `in_getlin`, `xwaitingforspace`) plus a heuristic
message-ends-in-`?` check:

- `yn_function` prompts: auto-answered `y` (commit to the action), except
  the dlvl-1 "no return" climb confirm and "Really attack?" (peacefuls),
  which are dismissed with ESC
- `getlin` prompts: auto-dismissed with ESC
- `--More--` / `xwaitforspace`: auto-dismissed with space/ESC

Capped at `NETHACK_AUTODISMISS_MAX=64` iterations. Episodes also
auto-reset after `NETHACK_MAX_EPISODE_STEPS=10000` steps.

## Per-episode log entries

| Key                | Meaning                                                |
|--------------------|--------------------------------------------------------|
| `perf` / `score`   | Final NetHack score                                    |
| `depth`            | Final dungeon level                                    |
| `episode_return`   | Sum of shaped reward                                   |
| `episode_length`   | Number of c_steps                                      |
| `valid_moves`      | c_steps where NetHack's turn counter advanced          |
| `illegal_actions`  | c_steps where the agent triggered a sub-prompt         |
| `new_tiles`        | Unique tiles entered this episode                      |
| `max_depth`        | Deepest level reached (depth-at-death under-reports)   |
| `prayers`          | Prayer actions per episode                             |
| `prayers_low_hp`   | Prayers issued at <=25% max HP (learned panic button)  |
| `searches`         | Search actions per episode                             |
| `engraves`         | Elbereth macro actions per episode                     |
| `wears`            | Wear macro presses that chose an item, per episode     |
| `eats`             | Eat macro presses that chose an item, per episode      |
| `damage_taken`     | Sum of HP lost per episode                             |
| `reward_saturated` | Fraction of steps with \|reward\| > 1 (trainer clamps) |
| `game_time`        | NetHack turns survived (steps can compress many turns) |
| `max_xp_level`     | Highest experience level reached                       |
| `death_combat`     | Fraction of episodes ended by DIED (monsters, traps)   |
| `death_starved`    | Fraction of episodes ended by STARVING                 |
| `death_other`      | Fraction ended any other way (drowning, poison, ...)   |
| `truncated`        | Fraction that hit the 10,000-step cap                  |
| `reach_mines`      | Fraction that entered the Gnomish Mines                |
| `reach_minetown`   | Fraction that reached Mines level 3+ (Minetown band)   |
| `reach_deep_mines` | Fraction that reached Mines level 5+ (past Minetown)   |
| `reach_main_d5`    | Fraction that reached depth 5+ in the main dungeon     |
| `reach_sokoban`    | Fraction that entered Sokoban                          |

## Standalone tools

The `nethack` binary (from `ocean/nethack/nethack.c`) plays the trained
policy by default — a CPU puffernet port of the CUDA encoder that loads
`resources/nethack/nethack_weights.bin` (copy any checkpoint there — the
hidden size and layer count are inferred from the file):

```bash
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./nethack               # policy demo
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./nethack 2000 0       # headless, 2000 steps
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./nethack random       # random policy
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./nethack bench 100000 # steps/sec
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./nethack record traj.txt 500
```

```bash
# Live viewer (interactive play)
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./live_view -i

# Live viewer (random agent)
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./live_view --random --steps 500

# Live viewer (replay a recorded trajectory)
NETHACKDIR=$(pwd)/vendor/nle/src/build/dat ./live_view --replay path/to/recording.bin
```

Build the nethack tool (same flags, swap the source file for live_view):
```bash
clang -O2 -I vendor/nle/src/include -I vendor/nle/src/build/include \
    -I vendor/nle/src/third_party/deboost.context/include \
    -DDEFAULT_WINDOW_SYS=\"rl\" -DDLB -DNLE_ALLOW_SEEDING \
    -DNLE_PER_ENV_FILES=1 -DNLE_PER_ENV_FLAGS=1 -DNLE_USE_ARENA_FREE=1 \
    -DNLE_USE_TILES -DNOCLIPPING -DNOCWD_ASSUMPTIONS -DNOMAIL -DNOTPARMDECL \
    ocean/nethack/nethack.c -o nethack \
    -L./vendor/nle/src/build -lnethack -lm -lbz2 -lpthread \
    -Wl,-rpath=$(pwd)/vendor/nle/src/build
```

Build live_view from source:
```bash
clang -O2 -I vendor/nle/src/include -I vendor/nle/src/build/include \
    -I vendor/nle/src/third_party/deboost.context/include \
    -DDEFAULT_WINDOW_SYS=\"rl\" -DDLB -DNLE_ALLOW_SEEDING \
    -DNLE_PER_ENV_FILES=1 -DNLE_PER_ENV_FLAGS=1 -DNLE_USE_ARENA_FREE=1 \
    -DNLE_USE_TILES -DNOCLIPPING -DNOCWD_ASSUMPTIONS -DNOMAIL -DNOTPARMDECL \
    ocean/nethack/live_view.c -o live_view \
    -L./vendor/nle/src/build -lnethack -lm -lbz2 -lpthread \
    -Wl,-rpath=$(pwd)/vendor/nle/src/build
```

## Performance

| Configuration | SPS | Notes |
|---|---|---|
| N=512 T=4 (4M model) | 40K | GPU-bound (train=68%) |
| N=4096 T=4 (4M model) | 128K | Balanced (env=66%, train=30%) |
| N=4096 T=16 (4M model) | ~400K | More threads = more env throughput |
| N=8192 T=64 B=2 (sweep) | ~1M | Full utilization with double-buffering |
