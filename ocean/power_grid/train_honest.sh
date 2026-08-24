#!/usr/bin/env bash
set -eu

# Honest unrestricted PPO fine-tuning curriculum. This script changes neither
# its starting checkpoint nor PufferLib core files. Each stage writes to
# its own checkpoint directory and allows all 91 actions without masks,
# rollback, greedy targets, or lookahead rewards.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
START=${1:-"$ROOT/resources/power_grid/policy.bin"}
OUTPUT=${2:-"$ROOT/checkpoints/power_grid/honest-surpass-run"}
PUFFER=${PUFFER_BIN:-"$ROOT/.venv/bin/puffer"}

if [ ! -f "$START" ]; then
    echo "Starting checkpoint not found: $START" >&2
    exit 1
fi
if [ ! -x "$PUFFER" ]; then
    echo "Puffer executable not found: $PUFFER" >&2
    exit 1
fi

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
    outage_min=$6
    outage_max=$7

    "$PUFFER" train power_grid \
        --seed "$seed" \
        --load-model-path "$input" \
        --checkpoint-dir "$directory" \
        --train.total-timesteps "$total_timesteps" \
        --train.horizon 1024 \
        --train.minibatch-size 12288 \
        --train.learning-rate "$learning_rate" \
        --train.ent-coef 0 \
        --env.offline-scenarios True \
        --env.offline-scenario-probability 1 \
        --env.offline-scenario-validation False \
        --env.random-events True \
        --env.random-event-probability 1 \
        --env.random-outage-count "$outage_max" \
        --env.random-outage-count-min "$outage_min" \
        --env.random-outages-at-reset False \
        --env.end-episode-on-recovery False \
        --env.max-episode-steps 72 \
        --env.failure-reward -1 \
        --env.safe-step-reward 1 \
        --env.recovery-reward 1 \
        --env.switch-penalty 0.05 \
        --env.secure-switch-penalty 0.5 \
        --env.congestion-cost-weight 5 \
        --env.congestion-progress-weight 0
}

mkdir -p "$OUTPUT"

train_stage "$START" "$OUTPUT/01-n1" 3003 30000000 0.000001 1 1
STAGE1=$(latest_checkpoint "$OUTPUT/01-n1")

train_stage "$STAGE1" "$OUTPUT/02-n2" 3004 20000000 0.000001 2 2
STAGE2=$(latest_checkpoint "$OUTPUT/02-n2")

train_stage "$STAGE2" "$OUTPUT/03-n1-micro" 3005 5000000 0.0000005 1 1
FINAL=$(latest_checkpoint "$OUTPUT/03-n1-micro")

echo "Final checkpoint: $FINAL"
echo "Evaluate N-1:"
echo "  $ROOT/.venv/bin/python $ROOT/ocean/power_grid/benchmark.py --checkpoint $FINAL --episodes 1024 --physics dc --random-event-probability 1 --random-outages 1"
echo "Evaluate N-2:"
echo "  $ROOT/.venv/bin/python $ROOT/ocean/power_grid/benchmark.py --checkpoint $FINAL --episodes 1024 --physics dc --random-event-probability 1 --random-outages 2"
