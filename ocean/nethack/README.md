# NetHack

PufferLib environment for NetHack 3.6.6 over
[fast-nle](https://github.com/FinlaySanders/fast-nle): 22-verb factored
action space (verb, item slot, direction), legality masking,
decomposed-score reward, custom CUDA encoder/decoder (`src/nethack.cu`).

## Setup

```bash
pip install -e .
./build.sh nethack    # clones + builds vendor/fast-nle, then the training backend
```

Run from the repo root — the engine finds its data at
`vendor/fast-nle/build/dat` (override with `NETHACKDIR`).

## Train

```bash
puffer train nethack
```

Reward coefficients and hypers live in `config/nethack.ini`.

## Watch a policy

```bash
./build.sh nethack --fast    # builds the ./nethack demo binary
./nethack                    # plays resources/nethack/nethack_weights.bin
NH_WEIGHTS=checkpoints/nethack/<run>/<step>.bin ./nethack
```

`./nethack [steps] [ms_per_frame]` — `0` ms runs headless. Set `NH_SEED`
to replay a run.

## Test

```bash
python tests/test_nethack_encoder.py    # encoder/decoder gradcheck vs torch
```
