#!/usr/bin/env bash
set -eu

# Reproduce the emergency-control curriculum used for the deployed
# 106-action checkpoint. The default legacy policy is read from the requested
# historical commit; neither repository files nor PufferLib core are changed.
# PPO and the offline AC recovery teacher both evaluate the complete action
# space. Deployment uses only the learned MinGRU: no masks, rollback, planner,
# lookahead, or runtime fallback control is present.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
LEGACY=${1:-}
OUTPUT=${2:-"$ROOT/checkpoints/power_grid/honest-surpass-run"}
PUFFER=${PUFFER_BIN:-"$ROOT/.venv/bin/puffer"}

if [ ! -x "$PUFFER" ]; then
    echo "Puffer executable not found: $PUFFER" >&2
    exit 1
fi

mkdir -p "$OUTPUT"
if [ -z "$LEGACY" ]; then
    LEGACY="$OUTPUT/legacy-91-actions.bin"
    git -C "$ROOT" show \
        fcef678b:resources/power_grid/policy.bin > "$LEGACY"
elif [ ! -f "$LEGACY" ]; then
    echo "Legacy checkpoint not found: $LEGACY" >&2
    exit 1
fi

START="$OUTPUT/expanded-106-actions.bin"
"$ROOT/.venv/bin/python" "$ROOT/ocean/power_grid/expand_checkpoint.py" \
    "$LEGACY" "$START"

latest_checkpoint() {
    find "$1" -type f -name '*.bin' -printf '%T@ %p\n' |
        sort -n | tail -1 | cut -d' ' -f2-
}

train_stage() {
    input=$1
    directory=$2
    seed=$3
    total_timesteps=$4
    learning_rate=$5
    shift 5

    "$PUFFER" train power_grid \
        --seed "$seed" \
        --load-model-path "$input" \
        --checkpoint-dir "$directory" \
        --train.total-timesteps "$total_timesteps" \
        --train.horizon 72 \
        --train.minibatch-size 18432 \
        --train.learning-rate "$learning_rate" \
        --env.offline-scenarios True \
        --env.offline-scenario-probability 1 \
        --env.offline-scenario-validation False \
        --env.random-events True \
        --env.random-event-probability 1 \
        --env.ac-power-flow True \
        --env.failure-reward -1 \
        --env.safe-step-reward 1 \
        --env.recovery-reward 1 \
        --env.switch-penalty 0.01 \
        --env.congestion-cost-weight 5 \
        --env.congestion-progress-weight 1 \
        "$@"
}

train_stage "$START" "$OUTPUT/01-emergency-reset" 4701 \
    10000000 0.0001 \
    --train.ent-coef 0.005 \
    --env.random-outage-count 2 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 1 \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery True --env.max-episode-steps 72 \
    --env.secure-switch-penalty 0.2 --env.unserved-load-cost-weight 0
STAGE1=$(latest_checkpoint "$OUTPUT/01-emergency-reset")

train_stage "$STAGE1" "$OUTPUT/02-one-step" 4703 \
    5000000 0.0001 \
    --train.ent-coef 0.001 \
    --env.random-outage-count 1 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 1 \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery True \
    --env.end-episode-on-recovery True --env.max-episode-steps 72 \
    --env.secure-switch-penalty 0.2 --env.unserved-load-cost-weight 10
STAGE2=$(latest_checkpoint "$OUTPUT/02-one-step")

train_stage "$STAGE2" "$OUTPUT/03-intact" 4705 \
    1000000 0.0001 \
    --train.ent-coef 0 \
    --env.random-outage-count 1 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0 \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery False --env.max-episode-steps 1 \
    --env.secure-switch-penalty 1 --env.unserved-load-cost-weight 10
STAGE3=$(latest_checkpoint "$OUTPUT/03-intact")

train_stage "$STAGE3" "$OUTPUT/04-intact-strong" 4706 \
    3000000 0.0003 \
    --train.ent-coef 0 \
    --env.random-outage-count 1 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0 \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery False --env.max-episode-steps 1 \
    --env.secure-switch-penalty 2 --env.unserved-load-cost-weight 100
STAGE4=$(latest_checkpoint "$OUTPUT/04-intact-strong")

train_stage "$STAGE4" "$OUTPUT/05-balanced-n1" 4707 \
    5000000 0.0001 \
    --train.ent-coef 0.001 \
    --env.random-outage-count 1 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0.5 \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery True \
    --env.end-episode-on-recovery True --env.max-episode-steps 4 \
    --env.secure-switch-penalty 1 --env.unserved-load-cost-weight 20
STAGE5=$(latest_checkpoint "$OUTPUT/05-balanced-n1")

train_stage "$STAGE5" "$OUTPUT/06-balanced-n2" 4709 \
    10000000 0.0001 \
    --train.ent-coef 0.005 \
    --env.random-outage-count 2 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0.5 \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery True --env.max-episode-steps 4 \
    --env.secure-switch-penalty 1 --env.unserved-load-cost-weight 20
STAGE6=$(latest_checkpoint "$OUTPUT/06-balanced-n2")

# Mix reset recovery with naturally delayed outages in the same full-day AC
# batch. This prevents alternating curricula from forgetting either immediate
# recovery or the recurrent transition observed after a user click.
train_stage "$STAGE6" "$OUTPUT/07-mixed-reset-timed" 4721 \
    10000000 0.00005 \
    --train.ent-coef 0.002 \
    --env.random-outage-count 2 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0.5 \
    --env.timed-outages-when-not-reset True \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery False --env.max-episode-steps 72 \
    --env.secure-switch-penalty 1 --env.unserved-load-cost-weight 20
STAGE7=$(latest_checkpoint "$OUTPUT/07-mixed-reset-timed")

train_stage "$STAGE7" "$OUTPUT/08-mixed-low-rate" 4723 \
    10000000 0.000025 \
    --train.ent-coef 0.001 \
    --env.random-outage-count 2 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0.5 \
    --env.timed-outages-when-not-reset True \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery False --env.max-episode-steps 72 \
    --env.secure-switch-penalty 1 --env.unserved-load-cost-weight 20
STAGE8=$(latest_checkpoint "$OUTPUT/08-mixed-low-rate")

# Reintroduce the deliberately stressed synthetic profiles alongside historical
# days to reduce year-specific overfitting before the held-out 2020 gate.
train_stage "$STAGE8" "$OUTPUT/09-mixed-stress" 4725 \
    10000000 0.000025 \
    --train.ent-coef 0.001 \
    --env.offline-scenario-probability 0.65 \
    --env.random-outage-count 2 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0.5 \
    --env.timed-outages-when-not-reset True \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery False --env.max-episode-steps 72 \
    --env.secure-switch-penalty 1 --env.unserved-load-cost-weight 20
STAGE9=$(latest_checkpoint "$OUTPUT/09-mixed-stress")

# Match the browser contract: the second delayed outage arrives after exactly
# one policy response to the first, with both click orders represented.
train_stage "$STAGE9" "$OUTPUT/10-sequential-clicks" 4727 \
    10000000 0.000025 \
    --train.ent-coef 0.001 \
    --env.offline-scenario-probability 0.65 \
    --env.random-outage-count 2 --env.random-outage-count-min 1 \
    --env.random-outages-at-reset True --env.reset-outage-probability 0.25 \
    --env.timed-outages-when-not-reset True \
    --env.sequential-outages True \
    --env.randomize-reset-operating-period True \
    --env.initial-outage-requires-one-step-recovery False \
    --env.end-episode-on-recovery False --env.max-episode-steps 72 \
    --env.secure-switch-penalty 1 --env.unserved-load-cost-weight 20
STAGE10=$(latest_checkpoint "$OUTPUT/10-sequential-clicks")

# Sparse PPO discovers too few of the secure actions that already exist. Build
# an offline 2019-only AC dataset, fit the same linear decoder, then aggregate
# one more dataset on the improved policy's state distribution. The 2020 gate
# is never read here, and the teacher is absent from the exported checkpoint.
GENERATOR="$OUTPUT/generate-recovery-dataset"
cc -O3 -std=c11 -Wno-unknown-pragmas -Wno-unused-function \
    -Wno-unused-parameter -Wno-unused-variable \
    -I"$ROOT/ocean/power_grid" -I"$ROOT/src" -I"$ROOT/vendor" \
    "$ROOT/ocean/power_grid/generate_recovery_dataset.c" -lm -o "$GENERATOR"

mkdir -p "$OUTPUT/11-offline-ac-teacher"
DATASET11="$OUTPUT/11-offline-ac-teacher/recovery-2019.bin"
STAGE11="$OUTPUT/11-offline-ac-teacher/policy.bin"
"$GENERATOR" "$STAGE10" "$DATASET11" 32
"$ROOT/.venv/bin/python" "$ROOT/ocean/power_grid/fit_recovery_decoder.py" \
    "$STAGE10" "$DATASET11" "$STAGE11" --epochs 100 --loss single

mkdir -p "$OUTPUT/12-aggregated-ac-teacher"
DATASET12="$OUTPUT/12-aggregated-ac-teacher/recovery-2019.bin"
STAGE12="$OUTPUT/12-aggregated-ac-teacher/policy.bin"
"$GENERATOR" "$STAGE11" "$DATASET12" 16
"$ROOT/.venv/bin/python" "$ROOT/ocean/power_grid/fit_recovery_decoder.py" \
    "$STAGE11" "$DATASET12" "$STAGE12" --epochs 100 --loss single \
    --no-op-weight-scale 0.0775

# A final aggregation pass avoids teaching one arbitrary topology action when
# several actions independently satisfy the same AC security and demand gate.
# The exported policy still selects one action directly; the acceptable-action
# set exists only in this offline 2019 training file.
mkdir -p "$OUTPUT/13-set-valued-ac-teacher"
DATASET13="$OUTPUT/13-set-valued-ac-teacher/recovery-2019.bin"
FINAL="$OUTPUT/13-set-valued-ac-teacher/policy.bin"
"$GENERATOR" "$STAGE12" "$DATASET13" 16
"$ROOT/.venv/bin/python" "$ROOT/ocean/power_grid/fit_recovery_decoder.py" \
    "$STAGE12" "$DATASET13" "$FINAL" --epochs 100 --loss set \
    --no-op-weight-scale 0.5

echo "Final checkpoint: $FINAL"
echo "Exhaustive held-out AC QA:"
echo "  $ROOT/build/qa_user_outages $FINAL 8"
