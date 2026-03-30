"""Compatibility wrapper for the slim TC-PepGen inference CLI."""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tc_pepgen.inference.predictor import main as predictor_main


def main() -> None:
    """Forward compatible wrapper arguments to the package CLI."""
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


if __name__ == "__main__":
    main()
