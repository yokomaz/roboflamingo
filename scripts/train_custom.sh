#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-/inspire/dataset/calvin/task-abcd-d/task_ABCD_D}"
VISION_ENCODER="${VISION_ENCODER:-/inspire/hdd/global_user/ky26319/models/clip_vit_l14/modelscope_clip_vit_large_patch14}"
LANGUAGE_MODEL="${LANGUAGE_MODEL:-/inspire/hdd/global_user/ky26319/models/mpt_1b_dolly}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-/inspire/hdd/global_user/ky26319/models/openflamingo_3b_vitl_mpt1b_langinstruct/checkpoint.pt}"
OUTPUT="${OUTPUT:-./output_test_3}"

python scripts/train_roboflamigo.py \
    --dataset "$DATASET" \
    --vision-encoder "$VISION_ENCODER" \
    --language-model "$LANGUAGE_MODEL" \
    --backbone-checkpoint "$BACKBONE_CHECKPOINT" \
    --output "$OUTPUT" \
    --batch-size "${BATCH_SIZE:-8}" \
    --window-size "${WINDOW_SIZE:-32}" \
    --steps "${STEPS:-10000}" \
    --checkpoint-steps "${CHECKPOINT_STEPS:-1000}" \
    --warmup-steps "${WARMUP_STEPS:-100}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.01}" \
    --workers "${WORKERS:-4}" \
    --local-files-only \
    "$@"
