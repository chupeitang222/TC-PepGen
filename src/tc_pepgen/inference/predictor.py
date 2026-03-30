"""Inference entry points for TC-PepGen."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, EncoderDecoderModel, LogitsProcessor

from ..models.hf_encoder_decoder_model import EsmWrapperConfig, EsmWrapperForEncoderDecoder
from ..utils.config import Config


class EOSBiasLogitsProcessor(LogitsProcessor):
    """Apply a bias to the EOS token score."""

    def __init__(self, eos_token_id: int, bias: float = 0.0):
        self.eos_token_id = eos_token_id
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bias != 0.0:
            scores[:, self.eos_token_id] += self.bias
        return scores


class ProteinInference:
    """Model wrapper for top-k and target-length generation."""

    def __init__(self, model_path: str):
        self.config = Config()
        self.model_path = Path(model_path)
        self.device = self.config.DEVICE

    def load_model(self) -> None:
        """Load the tokenizer and generation model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = EncoderDecoderModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id_before = getattr(self.tokenizer, "pad_token_id", None)
        bos_id_before = getattr(self.tokenizer, "bos_token_id", None)
        special_to_add = {}

        if getattr(self.config, "ENSURE_DISTINCT_SPECIAL_TOKENS", True):
            if (pad_id_before is None) or (pad_id_before == eos_id) or (pad_id_before == bos_id_before):
                special_to_add["pad_token"] = getattr(self.config, "PAD_TOKEN", "<pad>")
            if (bos_id_before is None) or (bos_id_before == eos_id) or (bos_id_before == pad_id_before):
                special_to_add["bos_token"] = getattr(self.config, "BOS_TOKEN", "<bos>")
            if eos_id is None:
                special_to_add["eos_token"] = getattr(self.config, "EOS_TOKEN", "<eos>")

        if special_to_add:
            print(f"Adding special tokens for inference: {special_to_add}")
            self.tokenizer.add_special_tokens(special_to_add)
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "resize_token_embeddings"):
                self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        final_eos_id = getattr(self.tokenizer, "eos_token_id", None)
        final_pad_id = getattr(self.tokenizer, "pad_token_id", None)
        final_bos_id = getattr(self.tokenizer, "bos_token_id", None)

        if final_bos_id is None:
            raise ValueError("Inference error: BOS token is required for seq2seq decoding")
        if final_eos_id is not None and final_bos_id == final_eos_id:
            raise ValueError(f"Inference error: BOS and EOS token IDs are identical ({final_bos_id})")

        self.model.config.decoder_start_token_id = final_bos_id
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.decoder_start_token_id = int(final_bos_id)
        if final_pad_id is not None:
            self.model.config.pad_token_id = int(final_pad_id)
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.pad_token_id = int(final_pad_id)
        if final_eos_id is not None:
            self.model.config.eos_token_id = int(final_eos_id)
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.eos_token_id = int(final_eos_id)

    def _build_encoder_outputs(self, protein_sequence: str):
        """Encode one protein sequence for generation."""
        sequence = protein_sequence.upper().strip()
        input_ids = torch.ones((1, len(sequence)), dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            raw_sequences=[sequence],
            tokenizer=self.tokenizer,
        )
        return encoder_outputs, attention_mask

    def _decoder_start_id(self):
        """Return the configured decoder start token."""
        return (
            getattr(self.model.generation_config, "decoder_start_token_id", None)
            or getattr(self.model.config, "decoder_start_token_id", None)
            or getattr(self.tokenizer, "bos_token_id", None)
            or getattr(self.tokenizer, "pad_token_id", None)
        )

    def _decode_outputs(self, outputs, target_length: int | None = None) -> list[str]:
        """Decode generated outputs and trim to the target length when needed."""
        sequences = getattr(outputs, "sequences", outputs)
        decoded_list = []
        for seq in sequences:
            decoded_text = self.tokenizer.decode(seq, skip_special_tokens=False)
            decoded = "".join(decoded_text.split())
            clean_decoded = decoded.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").strip()
            if target_length is not None and len(clean_decoded) > target_length:
                clean_decoded = clean_decoded[:target_length]
            decoded_list.append(clean_decoded)
        return decoded_list

    def predict(self, protein_sequence: str, num_sequences: int = 1, top_k: int = 3) -> list[str]:
        """Generate peptide binders for one protein sequence with top-k sampling."""
        encoder_outputs, attention_mask = self._build_encoder_outputs(protein_sequence)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or self.tokenizer.eos_token_id
        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_return_sequences=int(num_sequences),
            max_new_tokens=int(self.config.GEN_MAX_NEW_TOKENS),
            min_new_tokens=int(self.config.GEN_MIN_NEW_TOKENS),
            no_repeat_ngram_size=int(self.config.GEN_NO_REPEAT_NGRAM_SIZE),
            length_penalty=float(self.config.GEN_LENGTH_PENALTY),
            repetition_penalty=float(getattr(self.config, "GEN_REPETITION_PENALTY", 1.0)),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=True,
            top_k=int(top_k) if top_k is not None else int(self.config.GEN_TOP_K),
            top_p=float(self.config.GEN_TOP_P),
            temperature=float(self.config.GEN_TEMPERATURE),
            num_beams=1,
        )
        eos_bias = float(getattr(self.config, "GEN_EOS_BIAS", 0.0))
        if eos_bias != 0.0:
            gen_kwargs["logits_processor"] = [EOSBiasLogitsProcessor(self.tokenizer.eos_token_id, eos_bias)]

        start_id = self._decoder_start_id()
        if start_id is not None:
            gen_kwargs["decoder_input_ids"] = torch.full((1, 1), int(start_id), dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        return self._decode_outputs(outputs)

    def generate_with_target_length(self, protein_sequence: str, target_length: int, top_k: int = 3) -> str:
        """Generate one sequence for a target amino-acid length."""
        encoder_outputs, attention_mask = self._build_encoder_outputs(protein_sequence)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or self.tokenizer.eos_token_id
        max_tokens = max(int(target_length), 10)
        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_return_sequences=1,
            max_new_tokens=max_tokens,
            min_new_tokens=max(1, int(target_length)),
            no_repeat_ngram_size=int(self.config.GEN_NO_REPEAT_NGRAM_SIZE),
            length_penalty=float(self.config.GEN_LENGTH_PENALTY),
            repetition_penalty=float(getattr(self.config, "GEN_REPETITION_PENALTY", 1.0)),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=True,
            top_k=int(top_k) if top_k is not None else int(self.config.GEN_TOP_K),
            top_p=float(self.config.GEN_TOP_P),
            temperature=float(self.config.GEN_TEMPERATURE),
            num_beams=1,
        )
        eos_bias = float(getattr(self.config, "GEN_EOS_BIAS", 0.0))
        if eos_bias != 0.0:
            gen_kwargs["logits_processor"] = [EOSBiasLogitsProcessor(self.tokenizer.eos_token_id, eos_bias)]

        start_id = self._decoder_start_id()
        if start_id is not None:
            gen_kwargs["decoder_input_ids"] = torch.full((1, 1), int(start_id), dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        sequences = self._decode_outputs(outputs, target_length=int(target_length))
        return sequences[0] if sequences else ""


def _resolve_model_path(config: Config, model_arg: str | None) -> Path:
    base_dir = Path(model_arg) if model_arg else getattr(config, "CHECKPOINT_DIR", Path("checkpoints"))
    candidates = [base_dir / "final", base_dir / "best", base_dir / "mix" / "final", base_dir / "mix" / "best", base_dir]
    model_path = next((candidate for candidate in candidates if (candidate / "config.json").exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"No usable model directory found under {base_dir}")
    return model_path


def _resolve_input_col(dataframe: pd.DataFrame, config: Config, provided: str | None) -> str:
    cols = list(dataframe.columns)
    if provided and provided in cols:
        return provided
    lowered = {col.lower(): col for col in cols}
    for name in [getattr(config, "BATCH_INPUT_COLUMN", None), config.INPUT_COLUMN, "protein_sequence", "input_sequence", "receptor_sequence"]:
        if isinstance(name, str) and name in cols:
            return name
        if isinstance(name, str) and name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"Could not resolve the input column. CSV columns: {cols}")


def _resolve_label_col(dataframe: pd.DataFrame) -> str | None:
    cols = list(dataframe.columns)
    lowered = {col.lower(): col for col in cols}
    for name in ["sequence_id", "id", "protein_id", "name", "label", "uniprot", "pdb"]:
        if name in cols:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _resolve_length_col(dataframe: pd.DataFrame) -> str | None:
    cols = list(dataframe.columns)
    lowered = {col.lower(): col for col in cols}
    for name in ["Sequence Length", "sequence_length", "binder_length", "peptide_length", "target_length", "length"]:
        if name in cols:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _resolve_binder_col(dataframe: pd.DataFrame) -> str | None:
    cols = list(dataframe.columns)
    lowered = {col.lower(): col for col in cols}
    for name in ["binder", "peptide", "target_sequence", "peptide_sequence", "binder_sequence"]:
        if name in cols:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _export_results(results: list[dict], output_dir: Path, mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"{mode}_{timestamp}.csv"

    dataframe = pd.DataFrame(results)
    if "input_label" not in dataframe.columns:
        dataframe["input_label"] = ""
    dataframe["protein_sequence"] = dataframe["input_sequence"].astype(str)
    dataframe["output_clean"] = dataframe["generated_binder"].astype(str)
    dataframe["sample_id"] = range(1, len(dataframe) + 1)

    export_df = dataframe[["sample_id", "protein_sequence", "output_clean"]].rename(
        columns={"sample_id": "label", "protein_sequence": "protein_sequence", "output_clean": "output"}
    )
    export_df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="TC-PepGen inference")
    parser.add_argument("protein_sequence", type=str, nargs="?", default=None, help="Input protein sequence for single-item inference")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for generated CSV outputs")
    parser.add_argument("--num_sequences", type=int, default=3, help="Number of generated sequences")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k value used in sampling mode")
    parser.add_argument("--batch_from_csv", type=str, default=None, help="Generate from a specific CSV file")
    parser.add_argument("--input_col", type=str, default=None, help="Input column for --batch_from_csv; auto-detect when omitted")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of rows to process in batch mode; 0 means all")
    parser.add_argument("--mode", choices=["topk", "target_length_mode"], default="topk", help="Inference mode")
    args = parser.parse_args()

    config = Config()
    mode = args.mode
    model_path = _resolve_model_path(config, args.model)
    print(f"Model path: {model_path}")

    inference = ProteinInference(model_path)
    inference.load_model()
    results: list[dict] = []

    if args.batch_from_csv:
        csv_path = Path(args.batch_from_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        dataframe = pd.read_csv(csv_path)
        input_col = _resolve_input_col(dataframe, config, args.input_col)
        label_col = _resolve_label_col(dataframe)
        length_col = _resolve_length_col(dataframe) if mode == "target_length_mode" else None
        binder_col = _resolve_binder_col(dataframe) if mode == "target_length_mode" else None

        if int(args.limit) > 0:
            dataframe = dataframe.head(int(args.limit))
        dataframe = dataframe.dropna(subset=[input_col]).reset_index(drop=True)
        dataframe[input_col] = dataframe[input_col].astype(str).str.upper().str.strip()

        if mode == "target_length_mode" and length_col is None and binder_col is None:
            raise KeyError("target_length_mode requires a length column or binder column in the CSV")

        print(f"Batch inference from {csv_path} with {len(dataframe)} rows ({mode})")
        for index, row in dataframe.iterrows():
            sequence = str(row[input_col]).upper().strip()
            label = str(row[label_col]) if (label_col and pd.notna(row[label_col])) else ""
            print(f"\nSample {index + 1}/{len(dataframe)}")
            print(f"Protein sequence: {sequence[:60]}{'...' if len(sequence) > 60 else ''}")

            if mode == "target_length_mode":
                target_length = None
                if length_col and pd.notna(row[length_col]):
                    try:
                        target_length = int(row[length_col])
                    except Exception:
                        target_length = None
                if target_length is None and binder_col and pd.notna(row[binder_col]):
                    binder_sequence = str(row[binder_col]).strip().upper()
                    clean_binder = "".join(char for char in binder_sequence if char.isalpha())
                    target_length = len(clean_binder) if clean_binder else None
                if target_length is None:
                    raise ValueError(f"Could not determine target length for row {index + 1}")

                generated_sequence = inference.generate_with_target_length(sequence, target_length, args.top_k)
                print(f"Generated peptide: {generated_sequence}")
                results.append(
                    {
                        "input_label": label,
                        "input_sequence": sequence,
                        "generated_binder": generated_sequence,
                        "target_length": target_length,
                        "mode": mode,
                        "top_k": int(args.top_k),
                    }
                )
            else:
                outputs = inference.predict(sequence, int(args.num_sequences), args.top_k)
                for position, binder in enumerate(outputs, 1):
                    print(f"Generated peptide {position}: {binder}")
                    results.append(
                        {
                            "input_label": label,
                            "input_sequence": sequence,
                            "generated_binder": binder,
                            "mode": mode,
                            "top_k": int(args.top_k),
                        }
                    )
    else:
        if not args.protein_sequence:
            parser.error("Single-item mode requires protein_sequence or --batch_from_csv")
        if mode != "topk":
            parser.error("target_length_mode requires --batch_from_csv")

        print(f"Input sequence: {args.protein_sequence}")
        outputs = inference.predict(args.protein_sequence, int(args.num_sequences), args.top_k)
        print("\nGenerated sequences:")
        for position, sequence in enumerate(outputs, 1):
            print(f"Binder peptide {position}: {sequence}")
            results.append(
                {
                    "input_sequence": args.protein_sequence,
                    "generated_binder": sequence,
                    "mode": mode,
                    "top_k": int(args.top_k),
                }
            )

    output_dir = Path(args.output_dir) if args.output_dir else getattr(config, "OUTPUT_DIR", Path("outputs"))
    output_csv = _export_results(results, output_dir, mode)
    print(f"\nResults saved: {output_csv}")


if __name__ == "__main__":
    main()
