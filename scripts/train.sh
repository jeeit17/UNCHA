#!/bin/bash

set -e

# ------------------------------
# Defaults
# ------------------------------
CONFIG=""
NUM_GPUS=1
CHECKPOINT_PERIOD=5000
OUTPUT_DIR="./output"

usage() {
  echo "Usage: $0 \
    --config <config.py> \
    [--num-gpus N] \
    [--checkpoint-period N] \
    [--output-dir DIR]"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"; shift 2;;
    --num-gpus)
      NUM_GPUS="$2"; shift 2;;
    --checkpoint-period)
      CHECKPOINT_PERIOD="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Start Training"
echo "========================================"
echo "Config: $CONFIG"
echo "Num GPUs: $NUM_GPUS"
echo "Checkpoint period: $CHECKPOINT_PERIOD"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

python3 scripts/train.py \
  --config "$CONFIG" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-period "$CHECKPOINT_PERIOD"