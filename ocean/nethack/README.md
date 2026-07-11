# NetHack environment for PufferLib

C-level binding for the PufferLib RL framework against
[fast-nle](https://github.com/FinlaySanders/fast-nle). Each env owns a
per-env `nle_ctx_t` holding all of NetHack's mutable game state.
`libnethack.so` is linked directly (no dlopen). Auto-dismiss for prompts,
fixed 1070-byte observation (glyph crop + blstats +
cooldown/prev-action/inventory stats), reward shaping, multi-threaded OMP
stepping.

---

## Full setup (from scratch on any HPC)

### 1. Clone PufferLib + vendored NLE

```bash
git clone https://github.com/PufferAI/PufferLib.git && cd PufferLib
git checkout 4.0

# Clone fast-nle into vendor/fast-nle
git clone https://github.com/FinlaySanders/fast-nle.git vendor/fast-nle
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
cmake -S vendor/fast-nle -B vendor/fast-nle/build -DCMAKE_BUILD_TYPE=Release

# Build (always from repo root)
cmake --build vendor/fast-nle/build --target nethack -j$(nproc)
```

Produces `vendor/fast-nle/build/libnethack.so` and data files in
`vendor/fast-nle/build/dat/` (including `nhdat`).

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
export NETHACKDIR=$(pwd)/vendor/fast-nle/build/dat
```

### 7. Verify

```bash
# Quick training test (runs on login node if GPU available)
puffer train nethack \
    --vec.total-agents 64 --vec.num-buffers 1 --vec.num-threads 4 \
    --train.gpus 1 --train.total-timesteps 1000000 --train.minibatch-size 4096
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
cmake --build vendor/fast-nle/build --target nethack -j$(nproc)   # if vendor/fast-nle changed
bash build.sh nethack                               # always (relinks _C.so)
```

### Pulling NLE updates

fast-nle lives in its own repo. To pull latest changes:

```bash
cd vendor/fast-nle
git pull origin main
cd ../..
cmake --build vendor/fast-nle/build --target nethack -j$(nproc)
bash build.sh nethack
```

To push NLE changes (after editing files under `vendor/fast-nle/`):

```bash
cd vendor/fast-nle
git add -A && git commit -m "description of changes"
git push origin main
cd ../..
```

Note: `vendor/fast-nle/` has its own `.git` — it is NOT tracked by the
PufferLib repo (it is in `.gitignore`).

### Troubleshooting

| Error | Fix |
|---|---|
| `clang: command not found` | `module load llvm` or set `CC=gcc` |
| `cannot find -liomp5` | `module load intel-oneapi/2024.2` (or equivalent) |
| `libiomp5.so: cannot open shared object file` | Same module, also at runtime |
| `cannot allocate memory in static TLS block` | Too many `__thread` vars — rebuild libnethack |
| `NETHACKDIR is misconfigured` | `export NETHACKDIR=$(pwd)/vendor/fast-nle/build/dat` |
| `ZeroDivisionError` in train | Need `--train.gpus 1` (even on CPU-only, the CUDA build requires it) |
| Core dump / segfault at T>1 | Check that libnethack.so was rebuilt after latest source changes |

---

## Observation (fixed, 3616 bytes)

A single flat `ByteTensor` per agent:

| Slice     | Offset | Bytes | Encoding                                     |
|-----------|-------:|------:|----------------------------------------------|
| `glyphs`  |      0 |  3318 | full 79x21 grid, int16 LE per cell (map memory included; the GPU encoder derives the 9x9 egocentric crop from blstats x,y) |
| `blstats` |   3318 |   108 | 27 stats, int64 truncated to int32 LE        |
| `extra`   |   3426 |    80 | 20 stats, int32 LE (see below)               |
| `inv`     |   3506 |   110 | 55 inventory slot glyphs, int16 LE, in slot order (the item action head indexes these positions); empty slots = `NO_GLYPH` |

The crop is centered on the agent and padded with `NO_GLYPH` (5976)
outside the map. The hero's own tile shows what the hero is standing
on (top object, else terrain) instead of the hero glyph — a vendored,
settings-gated engine patch (`underfoot_glyphs`, off by default so
golden replays are unaffected). NetHack's display occludes exactly the
items an agent must act on (corpses to eat, stairs), the hero position
is already in blstats, and stock NLE has the same blind spot in every
spatial channel. The extra segment carries 20 env-derived stats: the
prayer cooldown (`u.ublesscnt`, exported by a vendored NLE patch;
vestigial since pray left the action space, kept to preserve the obs
layout), the previous
action index (-1 at episode start; a 64-dim recurrent trunk otherwise
has to reverse-engineer what it just did from obs deltas), and 18
per-object-class inventory item counts (stack = 1; tells the policy
*when* WEAR/EAT are worth pressing — the macros choose the item). This
layout is exactly what the custom CUDA encoder in `src/nethack.cu`
expects (glyph embedding -> 2 convs, blstats+extra normalized/expanded
to 88 features -> linear, concat -> projection); it is gradient-checked
by `tests/test_nethack_encoder.py`. `chars`, `message`, and
`inv_glyphs/letters/oclasses` are additionally bound to side buffers
for the standalone tools, the prompt heuristic, and the item macros,
but never enter the obs tensor raw (`inv_strs` stays unbound: it alone
triggers NetHack's doname() string formatting).

## Action space (MultiDiscrete {24, 55})

Two heads, sampled jointly each step. Head 0 is the verb:

```
 0  N            8  N_RUN         16  >  (down)
 1  S            9  S_RUN         17  <  (up)
 2  W           10  W_RUN        18  kick (^D)
 3  E           11  E_RUN        19  search (s)
 4  NW          12  NW_RUN       20  engrave Elbereth (macro)
 5  NE          13  NE_RUN       21  wear armor (slot arg)
 6  SW          14  SW_RUN       22  eat (slot arg)
 7  SE          15  SE_RUN       23  quaff potion (slot arg)
```

Head 1 is the item argument: an inventory position 0-54, matching the
obs `inv` block one-to-one. It is consumed only by WEAR/EAT/QUAFF (ignored
otherwise). The env answers the verb's getobj prompt with that slot's
letter if the game lists it as valid; otherwise the prompt is ESC'd and
the step counts as an illegal action (illegal_penalty applies). An
action mask (`MY_ACTION_MASK`, applied at sampling and in the training
recompute) keeps the lottery small: slots are legal only when holding
armor, food, or a potion, and EAT is masked out while Satiated — residual
mismatches the mask can't see (worn armor, carried corpses) still go
through the ESC path.

The 8 cardinal/intercardinal moves use vi-keys (kjhl ynbu). Long
"run" versions are uppercase (KJHL YNBU). Kick prompts "In what
direction?"; that prompt is left live, so the agent's next action (a
movement key) answers it — kick-through-a-locked-door is a learnable
two-step sequence. Search checks adjacent squares once for hidden
doors/passages (repeated searching raises the find chance, as in the
real game). Pray was removed from the action space: policies burned a
large share of episodes on god smites before learning prayer timing
(accumulated god anger is unobservable). Engrave-Elbereth drives the full
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
choke themselves in a handful of presses. Floor offers ("There is a
lichen corpse here; eat it?") are accepted — with pray removed, fresh
kills are the sustainable food source, and when the rot gamble is worth
taking is the policy's to learn. The cockatrice family is the exception
(declined: eating one is instant petrification). Carried corpses are
still filtered out of the candidate list by glyph — they age invisibly
in the pack, so old or poisonous ones kill.
Nothing wearable/edible -> no prompt -> clean no-op, no penalty.
Autopickup is set to `$[%!` (gold, armor, food, potions) to feed the item verbs,
with an `AUTOPICKUP_EXCEPTION` for corpses so they stay on the floor
where the eat offer (and NetHack's own freshness clock) lives — the
exception is a config-file-only directive, so the env writes a small
rc file per process and passes `@<path>` as the options string.
The wear macro doesn't rank items — a body-armor suit can get worn
(monks fight worse in suits) and cursed gear sticks; both are rare and
part of the game. No wield action on purpose: monks fight better
bare-handed, so a weapon slot would be a trap button.

## Reward shaping (config-tunable via nethack.ini)

```
reward = gold_coef     * d(gold net of start, floor 0)  # as botl_score counts it
       + exp_coef      * max(0, d exp points)           # dense per-kill (positive part)
       + descent_coef  * (new episode-max depth only)   # yo-yo-proof descent bonus
       + xp_coef       * (new episode-max XL only)      # power progression before descent
       + scout_coef    * (new_tile_this_level)          # exploration bonus
       + hp_coef       * (hp - prev_hp)                 # HP potential (dense survival signal)
       + hunger_coef   * (prev_hunger - hunger)         # hunger potential (see below)
       + illegal_penalty * (illegal_action)             # sub-prompt penalty
```

The terminal step of a game-over (not truncation) carries
`death_penalty - hp_coef * prev_hp` instead of the shaped terms — the
HP potential cashes out, so dying at high HP forfeits more.

The hunger potential is linear over the hunger blstat clamped to
[NotHungry..Starved] — Satiated counts as NotHungry, so overeating
toward NetHack's choke death earns nothing. Eating pays at the meal
(from Weak: +2 levels), hunger decay charges one level at a time, and
the potential form is farm-proof (any eat/digest cycle nets zero). It
exists to bridge the credit gap that kept eating reactive: the payoff
for opportunistic fresh-corpse eating otherwise lands thousands of
steps later, far outside the effective credit horizon.

gold/exp/descent decompose NetHack's score into separable knobs:
`botl_score()` = gold (net of starting purse, floored at 0) + `u.urexp`
+ 50*(deepest-1), where urexp accrues 4x exp points on kills and never
drops on level drain (hence the positive-part exp delta). So
`score_coef * dscore` is reproduced exactly by `gold_coef = score_coef`,
`exp_coef = 4 * score_coef`, and adding `50 * score_coef` to
`descent_coef` — same signal, separable knobs. The current config does
exactly that with the last known-good score run's values (gold 0.125 /
exp 0.5 / descent 6.305), probe-verified step-for-step against the old
reward. This deliberately rides the trainer's [-1,1] clamp the way that
run did: descents and most kills saturate to +1, so `reward_saturated`
is EXPECTED to be well above zero under this config. Score itself
remains the `perf` log metric.

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
| `searches`         | Search actions per episode                             |
| `engraves`         | Elbereth macro actions per episode                     |
| `wears`            | Wear macro presses that chose an item, per episode     |
| `eats`             | Eat macro presses that chose an item, per episode      |
| `floor_eats`       | Eats that accepted a floor "eat it?" offer             |
| `quaffs`           | Quaff presses that drank a potion                      |
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

## Policy demo

The `nethack` binary (from `ocean/nethack/nethack.c`) plays the trained
policy — a CPU puffernet port of the CUDA encoder that loads
`resources/nethack/nethack_weights.bin` (copy any checkpoint there — the
hidden size and layer count are inferred from the file):

```bash
NETHACKDIR=$(pwd)/vendor/fast-nle/build/dat ./nethack          # policy demo
NETHACKDIR=$(pwd)/vendor/fast-nle/build/dat ./nethack 2000 0   # headless, 2000 steps
```

Build the nethack tool:
```bash
clang -O2 -I vendor/fast-nle/include -I vendor/fast-nle/build/include \
    -I vendor/fast-nle/build/_deps/deboost_context-src/include \
    -DDEFAULT_WINDOW_SYS=\"rl\" -DDLB -DNLE_ALLOW_SEEDING \
    -DNLE_PER_ENV_FILES=1 -DNLE_PER_ENV_FLAGS=1 -DNLE_USE_ARENA_FREE=1 \
    -DNLE_USE_TILES -DNOCLIPPING -DNOCWD_ASSUMPTIONS -DNOMAIL -DNOTPARMDECL \
    ocean/nethack/nethack.c -o nethack \
    -L./vendor/fast-nle/build -lnethack -lm -lbz2 -lpthread \
    -Wl,-rpath=$(pwd)/vendor/fast-nle/build
```

## Performance

| Configuration | SPS | Notes |
|---|---|---|
| N=512 T=4 (4M model) | 40K | GPU-bound (train=68%) |
| N=4096 T=4 (4M model) | 128K | Balanced (env=66%, train=30%) |
| N=4096 T=16 (4M model) | ~400K | More threads = more env throughput |
| N=8192 T=64 B=2 (sweep) | ~1M | Full utilization with double-buffering |
