# TC-PepGen

TC-PepGen is an inference-focused repository for peptide generation from protein inputs.

It provides two inference modes:

- `topk`
- `target_length_mode`

## Project Structure

- `src/tc_pepgen/` - core package code
- `scripts/` - Python and Bash inference entry points
- `configs/` - runtime config files
- `examples/` - example CSV input data
- `checkpoints/` - local model checkpoints
- `outputs/` - generated CSV outputs

## Requirements

Recommended Python version: `3.12`.

`Python 3.14` is not recommended because the environment definition targets Python 3.12.

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate tc-pepgen
```

Available entry points:

- `python scripts/inference.py`
- `python scripts/batch_inference.py`

## Model Weights

This repository does not include model weights.

Place your exported Hugging Face-style model directory under one of these paths, or override `CHECKPOINT_PATH` / `--model_path`:

- `checkpoints/final`
- `checkpoints/best`
- `checkpoints/mix/final`
- `checkpoints/mix/best`

## Quick Run

Single-sequence `topk` generation:

```bash
python scripts/inference.py --model_path checkpoints/final --num_sequences 3 --top_k 5 "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVV"
```

Batch `target_length_mode` generation with the included example CSV:

```bash
python scripts/batch_inference.py examples/batch_inputs.csv --model_path checkpoints/final --mode target_length_mode --top_k 5 --output_csv outputs/batch_results.csv
```

If you prefer the existing shell wrappers:

```bash
bash scripts/run_inference.sh
bash scripts/run_batch_inference.sh
```

The default batch config already points to `examples/batch_inputs.csv`.

## Example Input

Included file:

- `examples/batch_inputs.csv`

Supported input columns are auto-detected.

For `target_length_mode`, target length is resolved in this order:

1. A length column such as `Sequence Length`, `sequence_length`, `binder_length`, `peptide_length`, `target_length`, or `length`
2. A binder column such as `binder`, `peptide`, `target_sequence`, `peptide_sequence`, or `binder_sequence`

## Runtime Config Files

- `configs/inference.env` - single-sequence or CSV inference settings
- `configs/batch_inference.env` - batch inference settings

Common settings:

- `CHECKPOINT_PATH`
- `MODE`
- `TOP_K`
- `OUTPUT_DIR` or `OUTPUT_CSV`

## Notes

- Only `topk` and `target_length_mode` are supported
- `target_length_mode` keeps `X` in generated sequences
- BOS is not counted toward the target length
- Generated CSV files are written to `outputs/`
- `scripts/` includes Bash wrappers for the main inference workflows
