"""Configuration helpers for TC-PepGen."""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Load and expose TC-PepGen configuration values."""

    def __init__(self, config_path: str = None):
        self.config_data = {}
        if config_path:
            self.load_from_yaml(config_path)
        else:
            self._set_defaults()

    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration values from YAML."""
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config_data = yaml.safe_load(handle)
        self._flatten_config()

    def _set_defaults(self) -> None:
        """Populate the built-in default configuration."""
        self.config_data = {
            "project": {
                "name": "TC-PepGen",
                "root": ".",
                "data_dir": "./data",
                "weights_dir": "./data/weights",
                "batch_data_dir": "./data/batch_inputs",
                "checkpoint_dir": "./checkpoints",
                "output_dir": "./outputs",
            },
            "model": {
                "encoder_model_name": "esmc_300m",
                "decoder_model_id": "nferruz/ProtGPT2",
                "decoder_local_dir": "./data/ProtGPT2",
                "esm_tokenizer_max_length": 1024,
                "encoder_max_length": 500,
                "decoder_max_length": 50,
                "dropout_rate": 0.1,
                "enc_to_dec_dropout_rate": 0.1,
                "decoder_attn_implementation": "flash_attention_2",
            },
            "data": {
                "input_column": "input_sequence",
                "target_column": "target_sequence",
                "batch_input_column": "protein_sequence",
                "batch_target_column": "binder_sequence",
                "train_val_split": 0.9,
            },
            "device": {
                "device": "cuda:0",
                "use_fp16": False,
            },
            "tokens": {
                "pad_token": "<pad>",
                "eos_token": "<eos>",
                "bos_token": "<bos>",
                "ensure_distinct_special_tokens": True,
            },
            "generation": {
                "num_beams": 4,
                "max_new_tokens": 20,
                "min_new_tokens": 2,
                "no_repeat_ngram_size": 2,
                "length_penalty": 0.5,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 20,
                "repetition_penalty": 1.2,
                "eos_bias": 40.0,
            },
        }
        self._flatten_config()

    def _flatten_config(self) -> None:
        """Flatten nested configuration values into attributes."""

        def _flatten_dict(data, parent_key="", sep="_"):
            items = []
            for key, value in data.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                if isinstance(value, dict):
                    items.extend(_flatten_dict(value, new_key, sep=sep).items())
                else:
                    items.append((new_key, value))
            return dict(items)

        flat_config = _flatten_dict(self.config_data)
        for key, value in flat_config.items():
            setattr(self, key.upper(), value)
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Convert configured path strings into Path objects."""
        root = Path(self.PROJECT_ROOT) if hasattr(self, "PROJECT_ROOT") else Path(".")
        path_attrs = [
            "PROJECT_DATA_DIR",
            "PROJECT_WEIGHTS_DIR",
            "PROJECT_BATCH_DATA_DIR",
            "PROJECT_CHECKPOINT_DIR",
            "PROJECT_OUTPUT_DIR",
        ]

        for attr in path_attrs:
            if hasattr(self, attr):
                path_str = getattr(self, attr)
                if isinstance(path_str, str):
                    setattr(self, attr.replace("PROJECT_", ""), root / path_str)

        search_paths = getattr(self, "PATHS_MODEL_SEARCH_PATHS", ["~", ".", "./data/weights"])
        self.MODEL_SEARCH_PATHS = [str(Path(path).expanduser()) for path in search_paths]

    def setup_esm_environment(self) -> None:
        """Configure environment variables required by ESM."""
        import os

        os.environ["INFRA_PROVIDER"] = "True"

    def get(self, key: str, default: Any = None):
        """Return a configuration value by key."""
        return getattr(self, key.upper(), default)

    def update(self, **kwargs) -> None:
        """Update configuration attributes at runtime."""
        for key, value in kwargs.items():
            setattr(self, key.upper(), value)
