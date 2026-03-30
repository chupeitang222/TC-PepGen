#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG_FILE="$REPO_ROOT/configs/inference.env"
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
NUM_SEQUENCES="${NUM_SEQUENCES:-3}"
TOP_K="${TOP_K:-3}"
MODE="${MODE:-topk}"
PROTEIN_SEQUENCE="${PROTEIN_SEQUENCE:-}"
INPUT_CSV="${INPUT_CSV:-}"
INPUT_COL="${INPUT_COL:-}"
LIMIT="${LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

echo "Starting inference"
echo "  config: $CONFIG_FILE"
echo "  checkpoint: $CHECKPOINT_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  mode: $MODE"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CMD=(python "$REPO_ROOT/scripts/inference.py" --model_path "$REPO_ROOT/$CHECKPOINT_PATH" --num_sequences "$NUM_SEQUENCES" --top_k "$TOP_K" --mode "$MODE" --output_dir "$REPO_ROOT/$OUTPUT_DIR")

if [[ -n "$INPUT_CSV" ]]; then
    CMD+=(--batch_csv "$REPO_ROOT/$INPUT_CSV")
fi

if [[ -n "$INPUT_COL" ]]; then
    CMD+=(--input_col "$INPUT_COL")
fi

if [[ "$LIMIT" != "0" ]]; then
    CMD+=(--limit "$LIMIT")
fi

if [[ -n "$PROTEIN_SEQUENCE" ]]; then
    CMD+=("$PROTEIN_SEQUENCE")
fi

if [[ "$MODE" == "target_length_mode" && -z "$INPUT_CSV" ]]; then
    echo "target_length_mode requires INPUT_CSV" >&2
    exit 1
fi

"${CMD[@]}" "$@"
