"""Installed command-line entry points for TC-PepGen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tc_pepgen.inference.predictor import (
    ProteinInference,
    _resolve_binder_col,
    _resolve_input_col,
    _resolve_length_col,
    main as predictor_main,
)
from tc_pepgen.utils.config import Config


def main() -> None:
    """Forward wrapper arguments to the package inference CLI."""
    parser = argparse.ArgumentParser(description="TC-PepGen inference wrapper")
    parser.add_argument("protein_sequence", type=str, nargs="?", default=None, help="Input protein sequence")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument("--batch_csv", type=str, default=None, help="Batch inference CSV path")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k value")
    parser.add_argument("--num_sequences", type=int, default=None, help="Number of generated sequences")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for generated CSV outputs")
    parser.add_argument("--mode", choices=["topk", "target_length_mode"], default=None, help="Inference mode")
    parser.add_argument("--input_col", type=str, default=None, help="Input column for CSV inference")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process from CSV")
    args, remaining = parser.parse_known_args()

    forwarded_args = [sys.argv[0]]
    if args.model_path:
        forwarded_args.extend(["--model", args.model_path])
    if args.output_dir:
        forwarded_args.extend(["--output_dir", args.output_dir])
    if args.num_sequences is not None:
        forwarded_args.extend(["--num_sequences", str(args.num_sequences)])
    if args.top_k is not None:
        forwarded_args.extend(["--top_k", str(args.top_k)])
    if args.input_col:
        forwarded_args.extend(["--input_col", args.input_col])
    if args.limit is not None:
        forwarded_args.extend(["--limit", str(args.limit)])
    if args.batch_csv:
        forwarded_args.extend(["--batch_from_csv", args.batch_csv])
    if args.mode:
        forwarded_args.extend(["--mode", args.mode])
    if args.protein_sequence:
        forwarded_args.append(args.protein_sequence)
    forwarded_args.extend(remaining)

    sys.argv = forwarded_args
    predictor_main()


def _pick_final_model_dir(cfg: Config) -> Path:
    """Return the preferred model directory."""
    base_dir = Path(cfg.CHECKPOINT_DIR)
    candidates = [base_dir / "final", base_dir / "best", base_dir / "mix" / "final", base_dir / "mix" / "best", base_dir]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    return base_dir


def batch_main() -> None:
    """Run batch inference from a CSV file."""
    parser = argparse.ArgumentParser(description="TC-PepGen batch inference")
    parser.add_argument("input_csv", type=str, help="Input CSV path")
    parser.add_argument("--output_csv", type=str, default="batch_results.csv", help="Output CSV path")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument("--mode", choices=["topk", "target_length_mode"], default="topk", help="Inference mode")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k value")
    args = parser.parse_args()

    config = Config()
    model_path = args.model_path or _pick_final_model_dir(config)

    inference = ProteinInference(str(model_path))
    inference.load_model()

    dataframe = pd.read_csv(args.input_csv)
    input_col = _resolve_input_col(dataframe, config, None)
    dataframe = dataframe.dropna(subset=[input_col]).reset_index(drop=True)
    dataframe[input_col] = dataframe[input_col].astype(str).str.upper().str.strip()
    sequences = dataframe[input_col].tolist()
    length_col = None
    binder_col = None
    if args.mode == "target_length_mode":
        length_col = _resolve_length_col(dataframe)
        binder_col = _resolve_binder_col(dataframe)
        if length_col is None and binder_col is None:
            raise KeyError("target_length_mode requires a length column or binder column in the CSV")

    results = []
    for index, sequence in enumerate(sequences, start=1):
        print(f"Processing sequence {index}/{len(sequences)}: {sequence[:50]}...")
        try:
            if args.mode == "target_length_mode":
                row = dataframe.iloc[index - 1]
                target_length = None
                if length_col and pd.notna(row[length_col]):
                    target_length = int(row[length_col])
                elif binder_col and pd.notna(row[binder_col]):
                    binder_sequence = str(row[binder_col]).strip().upper()
                    clean_binder = "".join(char for char in binder_sequence if char.isalpha())
                    target_length = len(clean_binder) if clean_binder else None
                if target_length is None:
                    raise ValueError(f"Could not determine target length for row {index}")
                result = inference.generate_with_target_length(sequence, target_length, top_k=args.top_k)
            else:
                predictions = inference.predict(sequence, num_sequences=1, top_k=args.top_k)
                result = predictions[0] if predictions else "ERROR"
        except Exception as exc:
            print(f"Inference failed: {exc}")
            result = "ERROR"

        results.append(
            {
                "input_sequence": sequence,
                "predicted_sequence": result,
            }
        )

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Saved results to: {args.output_csv}")
