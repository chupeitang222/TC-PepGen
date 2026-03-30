"""Dataset helpers for TC-PepGen."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from ..utils.config import Config


class ProteinDataLoader:
    """Load CSV data and convert it into Hugging Face datasets."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config if config is not None else Config()

    def load_csv_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load and normalize input CSV data."""
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.config.DATA_DIR / file_path

        data = pd.read_csv(file_path)
        data = data.dropna(subset=[self.config.INPUT_COLUMN, self.config.TARGET_COLUMN])
        data[self.config.INPUT_COLUMN] = data[self.config.INPUT_COLUMN].str.upper().str.strip()
        data[self.config.TARGET_COLUMN] = data[self.config.TARGET_COLUMN].str.upper().str.strip()
        return data

    def create_datasets(self, data: pd.DataFrame) -> DatasetDict:
        """Build train, validation, and test datasets."""
        shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
        total = len(shuffled)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)

        train_data = shuffled[:train_size]
        val_data = shuffled[train_size : train_size + val_size]
        test_data = shuffled[train_size + val_size :]

        datasets = DatasetDict(
            {
                "train": Dataset.from_pandas(train_data),
                "validation": Dataset.from_pandas(val_data),
                "test": Dataset.from_pandas(test_data),
            }
        )

        def tokenize_function(examples):
            input_sequences = examples[self.config.INPUT_COLUMN]
            target_sequences = examples[self.config.TARGET_COLUMN]
            return {
                "raw_input_sequences": input_sequences,
                "raw_target_sequences": target_sequences,
                "input_tokens": [list(seq) for seq in input_sequences],
                "target_tokens": [list(seq) for seq in target_sequences],
            }

        return datasets.map(tokenize_function, batched=True, remove_columns=datasets["train"].column_names)


def load_protein_data(csv_file: Union[str, Path], config: Optional[Config] = None) -> Tuple[DatasetDict, ProteinDataLoader]:
    """Load protein data and return datasets plus the loader."""
    config = config or Config()
    loader = ProteinDataLoader(config)
    data = loader.load_csv_data(csv_file)
    datasets = loader.create_datasets(data)
    return datasets, loader


def create_sample_data(output_file: Union[str, Path], num_samples: int = 100) -> None:
    """Create a small sample CSV for local testing."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    np.random.seed(42)

    data = []
    for _ in range(num_samples):
        input_seq = "".join(np.random.choice(list(amino_acids), np.random.randint(20, 100)))
        target_seq = "".join(np.random.choice(list(amino_acids), np.random.randint(5, 20)))
        data.append({"input_sequence": input_seq, "target_sequence": target_seq})

    pd.DataFrame(data).to_csv(output_file, index=False)
