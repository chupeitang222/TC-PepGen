"""Model builder helpers for TC-PepGen."""

import torch
from torch import nn
from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModel, EncoderDecoderModel, AutoModelForCausalLM
import sys

from ..utils.config import Config
from .hf_encoder_decoder_model import EsmWrapperForEncoderDecoder, EsmWrapperConfig

class ModelBuilder:
    def __init__(self, config: Optional[Config] = None):
        self.config = config if config is not None else Config()
        self.config.setup_esm_environment()
    
    def build_model(self, local_encoder_path: str) -> Tuple[EncoderDecoderModel, AutoTokenizer, None]:
        """Build the encoder-decoder model stack."""
        
        print("Inspecting model hidden sizes...")
        decoder_source = self.config.DECODER_MODEL_ID
        try:
            local_dir = getattr(self.config, "DECODER_LOCAL_DIR", None)
            if local_dir is not None:
                local_dir = Path(local_dir)
                if local_dir.is_dir() and (local_dir / "config.json").exists():
                    decoder_source = str(local_dir)
                    print(f"Using local decoder directory: {decoder_source}")
        except Exception:
            pass

        try:
            temp_decoder_model = AutoModel.from_pretrained(
                decoder_source,
                trust_remote_code=True,
                local_files_only=(decoder_source != self.config.DECODER_MODEL_ID)
            )
        except Exception as e:
            if decoder_source == self.config.DECODER_MODEL_ID and local_dir is not None and (Path(local_dir) / "config.json").exists():
                temp_decoder_model = AutoModel.from_pretrained(
                    str(local_dir), trust_remote_code=True, local_files_only=True
                )
                decoder_source = str(local_dir)
                print(f"Remote load failed; falling back to the local decoder directory: {decoder_source}")
            else:
                raise e
        decoder_hidden_size = temp_decoder_model.config.hidden_size
        print(f"Decoder hidden size: {decoder_hidden_size}")
        
        encoder_hidden_size = 960
        print(f"Encoder hidden size: {encoder_hidden_size}")
        
        encoder_config = EsmWrapperConfig(
            esm_model_name=self.config.ENCODER_MODEL_NAME,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            dropout_rate=self.config.DROPOUT_RATE
        )
        encoder_model = EsmWrapperForEncoderDecoder(encoder_config)
        
        decoder_tokenizer = AutoTokenizer.from_pretrained(
            decoder_source,
            trust_remote_code=True,
            local_files_only=(decoder_source != self.config.DECODER_MODEL_ID)
        )
        try:
            eos_id = getattr(decoder_tokenizer, "eos_token_id", None)
            pad_id_before = getattr(decoder_tokenizer, "pad_token_id", None)
            bos_id_before = getattr(decoder_tokenizer, "bos_token_id", None)

            special_to_add = {}

            if getattr(self.config, "ENSURE_DISTINCT_SPECIAL_TOKENS", True):
                if (pad_id_before is None) or (pad_id_before == eos_id) or (pad_id_before == bos_id_before):
                    special_to_add["pad_token"] = getattr(self.config, "PAD_TOKEN", "<pad>")

                if (bos_id_before is None) or (bos_id_before == eos_id) or (bos_id_before == pad_id_before):
                    special_to_add["bos_token"] = getattr(self.config, "BOS_TOKEN", "<bos>")

                if eos_id is None:
                    special_to_add["eos_token"] = getattr(self.config, "EOS_TOKEN", "<eos>")
            else:
                if (pad_id_before is None) or (eos_id is not None and pad_id_before == eos_id):
                    special_to_add["pad_token"] = getattr(self.config, "PAD_TOKEN", "<pad>")
                if (bos_id_before is None) or (eos_id is not None and bos_id_before == eos_id):
                    special_to_add["bos_token"] = getattr(self.config, "BOS_TOKEN", "<bos>")

            if len(special_to_add) > 0:
                print(f"Adding special tokens: {special_to_add}")
                decoder_tokenizer.add_special_tokens(special_to_add)

            final_eos_id = getattr(decoder_tokenizer, "eos_token_id", None)
            final_pad_id = getattr(decoder_tokenizer, "pad_token_id", None)
            final_bos_id = getattr(decoder_tokenizer, "bos_token_id", None)

            print(f"Final special-token configuration:")
            print(f"   EOS: {final_eos_id} ('{decoder_tokenizer.eos_token if final_eos_id is not None else None}')")
            print(f"   PAD: {final_pad_id} ('{decoder_tokenizer.pad_token if final_pad_id is not None else None}')")
            print(f"   BOS: {final_bos_id} ('{decoder_tokenizer.bos_token if final_bos_id is not None else None}')")

            if final_eos_id is not None and final_pad_id is not None and final_eos_id == final_pad_id:
                raise ValueError(f"Critical error: EOS and PAD token IDs are identical ({final_eos_id})")
            if final_bos_id is not None and final_pad_id is not None and final_bos_id == final_pad_id:
                raise ValueError(f"Critical error: BOS and PAD token IDs are identical ({final_bos_id})")
            if final_bos_id is not None and final_eos_id is not None and final_bos_id == final_eos_id:
                raise ValueError(f"Critical error: BOS and EOS token IDs are identical ({final_bos_id})")

        except Exception as e:
            print(f"Special-token configuration warning: {e}")
            if not hasattr(decoder_tokenizer, "pad_token") or decoder_tokenizer.pad_token_id is None:
                decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        
        from transformers import GPT2Config
        decoder_config = GPT2Config.from_pretrained(
            decoder_source,
            local_files_only=(decoder_source != self.config.DECODER_MODEL_ID)
        )
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        try:
            attn_impl = getattr(self.config, "DECODER_ATTN_IMPLEMENTATION", None)
            if isinstance(attn_impl, str) and len(attn_impl) > 0:
                decoder_config.attn_implementation = attn_impl
                print(f"Decoder attention implementation: {decoder_config.attn_implementation}")
        except Exception:
            pass
        
        decoder_model = AutoModelForCausalLM.from_pretrained(
            decoder_source,
            config=decoder_config,
            trust_remote_code=True,
            local_files_only=(decoder_source != self.config.DECODER_MODEL_ID)
        )
        try:
            if hasattr(decoder_model, "resize_token_embeddings"):
                current_vocab_size = decoder_model.get_input_embeddings().weight.shape[0]
                if current_vocab_size != len(decoder_tokenizer):
                    decoder_model.resize_token_embeddings(len(decoder_tokenizer))
        except Exception:
            pass
        
        if decoder_tokenizer.pad_token is None:
            try:
                decoder_tokenizer.add_special_tokens({"pad_token": getattr(self.config, "PAD_TOKEN", "<pad>")})
                if hasattr(decoder_model, "resize_token_embeddings"):
                    decoder_model.resize_token_embeddings(len(decoder_tokenizer))
            except Exception:
                decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        
        model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        
        _bos_id = getattr(decoder_tokenizer, "bos_token_id", None)
        _eos_id = getattr(decoder_tokenizer, "eos_token_id", None)

        if _bos_id is None:
            raise ValueError("Critical error: BOS token is required for seq2seq decoding")
        if _eos_id is not None and _bos_id == _eos_id:
            raise ValueError(f"Critical error: BOS and EOS token IDs are identical ({_bos_id})")

        model.config.decoder_start_token_id = _bos_id
        print(f"decoder_start_token_id set to BOS token: {_bos_id} ('{decoder_tokenizer.bos_token}')")
        model.config.eos_token_id = decoder_tokenizer.eos_token_id
        model.config.pad_token_id = decoder_tokenizer.pad_token_id
        model.config.max_length = self.config.DECODER_MAX_LENGTH
        model.config.num_beams = self.config.NUM_BEAMS
        try:
            gen_cfg = getattr(model, "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.decoder_start_token_id = int(start_id) if start_id is not None else None
                gen_cfg.eos_token_id = int(decoder_tokenizer.eos_token_id) if decoder_tokenizer.eos_token_id is not None else None
                gen_cfg.pad_token_id = int(decoder_tokenizer.pad_token_id) if decoder_tokenizer.pad_token_id is not None else None
        except Exception:
            pass
        
        try:
            p = float(getattr(self.config, "ENC_TO_DEC_DROPOUT_RATE", 0.0))
            if p and p > 0.0 and hasattr(model, "enc_to_dec_proj") and isinstance(model.enc_to_dec_proj, nn.Module):
                model.enc_to_dec_dropout = nn.Dropout(p)
                def _proj_dropout_hook(mod, inputs, output):
                    if hasattr(model, "enc_to_dec_dropout") and model.training:
                        return model.enc_to_dec_dropout(output)
                    return output
                model.enc_to_dec_proj.register_forward_hook(_proj_dropout_hook)
                print(f"Enabled post-projection dropout: p={p}")
        except Exception:
            pass

        try:
            p_ca = float(getattr(self.config, "ENC_TO_DEC_DROPOUT_RATE", 0.0))
            if p_ca and p_ca > 0.0:
                dec = getattr(model, "decoder", None)
                tr = getattr(dec, "transformer", None)
                blocks = getattr(tr, "h", None)
                if blocks is not None:
                    if not hasattr(model, "cross_attn_dropouts"):
                        model.cross_attn_dropouts = nn.ModuleList()
                    for idx, blk in enumerate(blocks):
                        ca = getattr(blk, "crossattention", None)
                        if ca is None:
                            continue
                        dp = nn.Dropout(p_ca)
                        model.cross_attn_dropouts.append(dp)
                        def _make_hook(dropout_module):
                            def _hook(mod, inputs, outputs):
                                try:
                                    if isinstance(outputs, tuple):
                                        attn_out = outputs[0]
                                        attn_out = dropout_module(attn_out)
                                        return (attn_out,) + outputs[1:]
                                    else:
                                        return dropout_module(outputs)
                                except Exception:
                                    return outputs
                            return _hook
                        ca.register_forward_hook(_make_hook(dp))
                    print(f"Enabled cross-attention dropout: p={p_ca}")
        except Exception:
            pass
        
        model = model.to(self.config.DEVICE)
        
        return model, decoder_tokenizer, None

def build_encoder_decoder_model(local_encoder_path: str, config: Optional[Config] = None) -> Tuple[EncoderDecoderModel, AutoTokenizer, None]:
    """Build the encoder-decoder model with defaults."""
    builder = ModelBuilder(config)
    return builder.build_model(local_encoder_path)
