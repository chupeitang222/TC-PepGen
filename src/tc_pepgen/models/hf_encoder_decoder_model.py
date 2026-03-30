"""Encoder wrapper components for TC-PepGen."""

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


print(f"PyTorch version: {torch.__version__}")


class EsmWrapperConfig(PretrainedConfig):
    """Minimal config used by the encoder wrapper."""

    model_type = "esm_wrapper"

    def __init__(self, esm_model_name="esmc_300m", encoder_hidden_size=960, decoder_hidden_size=1280, dropout_rate=0.1, **kwargs):
        self.esm_model_name = esm_model_name
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = encoder_hidden_size
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)


class EsmWrapperForEncoderDecoder(PreTrainedModel):
    """Adapter that exposes ESMC as a Hugging Face encoder."""

    config_class = EsmWrapperConfig

    def __init__(self, config: EsmWrapperConfig):
        super().__init__(config)
        print(f"Initializing the ESM wrapper with {config.esm_model_name}...")

        import os

        os.environ["INFRA_PROVIDER"] = "True"

        self.esm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_client = ESMC.from_pretrained(config.esm_model_name).to(self.esm_device)
        self.tokenizer = self.esm_client.tokenizer

        for _, param in self.esm_client.named_parameters():
            param.requires_grad = True
        print("ESM encoder parameters are trainable.")

        self.dropout = nn.Dropout(config.dropout_rate)
        print("ESMC model loaded.")

    def _decode_sequence_from_ids(self, input_ids, tokenizer):
        """Decode a raw amino-acid sequence from token IDs."""
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        return text.replace(" ", "")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Encode protein sequences into hidden states."""
        batch_embeddings = []
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        for index in range(batch_size):
            raw_sequence = None

            if kwargs.get("raw_sequences") is not None and index < len(kwargs["raw_sequences"]):
                candidate = kwargs["raw_sequences"][index]
                if isinstance(candidate, str) and candidate.strip():
                    raw_sequence = candidate.strip().upper()

            if (raw_sequence is None or not raw_sequence) and "tokenizer" in kwargs:
                try:
                    decoded = self._decode_sequence_from_ids(input_ids[index], kwargs["tokenizer"])
                    if isinstance(decoded, str) and decoded.strip():
                        raw_sequence = decoded.strip().upper()
                except Exception as exc:
                    print(f"Failed to decode a sequence from input_ids: {exc}")
                    raw_sequence = None

            if raw_sequence is None or not raw_sequence:
                raw_sequence = "MKALIVLGAVILSVAAVAGILA"

            protein = ESMProtein(sequence=raw_sequence)
            protein_tensor = self.esm_client.encode(protein)
            logits_output = self.esm_client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            esm_embedding = logits_output.embeddings

            if esm_embedding.shape[1] != seq_length:
                if esm_embedding.shape[1] < seq_length:
                    padding = torch.zeros(1, seq_length - esm_embedding.shape[1], esm_embedding.shape[2], device=esm_embedding.device)
                    esm_embedding = torch.cat([esm_embedding, padding], dim=1)
                else:
                    esm_embedding = esm_embedding[:, :seq_length, :]

            batch_embeddings.append(esm_embedding.squeeze(0))

        final_embeddings = torch.stack(batch_embeddings, dim=0)
        final_embeddings = self.dropout(final_embeddings)
        return BaseModelOutput(last_hidden_state=final_embeddings)


from transformers import AutoConfig as _AutoConfig

_AutoConfig.register("esm_wrapper", EsmWrapperConfig)
AutoModel.register(EsmWrapperConfig, EsmWrapperForEncoderDecoder)
