#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG_FILE="$REPO_ROOT/configs/batch_inference.env"
CONFIG_FILE="${CONFIG_FILE:-$DEFAULT_CONFIG_FILE}"

if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    set +a
else
    echo "Missing config file: $CONFIG_FILE" >&2
    exit 1
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/final}"
INPUT_CSV="${INPUT_CSV:-data/batch_inputs/test.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/batch_results.csv}"
MODE="${MODE:-topk}"
TOP_K="${TOP_K:-3}"

echo "Starting batch inference"
echo "  config: $CONFIG_FILE"
echo "  input: $INPUT_CSV"
echo "  output: $OUTPUT_CSV"
echo "  checkpoint: $CHECKPOINT_PATH"
echo "  mode: $MODE"
echo "  top_k: $TOP_K"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python "$REPO_ROOT/scripts/batch_inference.py" "$REPO_ROOT/$INPUT_CSV" --output_csv "$REPO_ROOT/$OUTPUT_CSV" --model_path "$REPO_ROOT/$CHECKPOINT_PATH" --mode "$MODE" --top_k "$TOP_K" "$@"
