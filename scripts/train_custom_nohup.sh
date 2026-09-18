#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${OUTPUT:-./output_test_3}"
mkdir -p "$OUTPUT"

nohup bash "$ROOT_DIR/scripts/train_custom.sh" "$@" > "$OUTPUT/train.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$OUTPUT/train.pid"

echo "training started: PID $TRAIN_PID"
echo "log: $OUTPUT/train.log"
echo "follow: tail -f $OUTPUT/train.log"
