#!/usr/bin/env bash
set -eu

# Reproduce the emergency-control curriculum used for the deployed
# 106-action checkpoint. The default legacy policy is read from the requested
# historical commit; neither repository files nor PufferLib core are changed.
# PPO always selects directly from the complete action space without masks,
# rollback, greedy targets, lookahead rewards, or runtime fallback control.

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
FINAL=$(latest_checkpoint "$OUTPUT/08-mixed-low-rate")

echo "Final checkpoint: $FINAL"
echo "Exhaustive held-out AC QA:"
echo "  $ROOT/build/qa_user_outages $FINAL 8"
