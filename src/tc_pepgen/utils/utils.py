"""Utility helpers for TC-PepGen."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from .config import Config


def setup_logging() -> logging.Logger:
    """Create a default logger for inference workflows."""
    logger = logging.getLogger("protein_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def find_model_path(search_roots: List[Union[str, Path]], model_name: str) -> Optional[str]:
    """Search for a model artifact under the provided roots."""
    extensions = [".pth", ".bin", ".pt", ".ckpt"]
    patterns = [f"{model_name}*", f"*{model_name}*"]

    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in patterns:
            for extension in extensions:
                matches = list(root_path.glob(f"**/{pattern}{extension}"))
                if matches:
                    return str(matches[0])
    return None


def create_directories(config: Config) -> None:
    """Create the standard project directories when needed."""
    for directory in [config.DATA_DIR, config.WEIGHTS_DIR, config.CHECKPOINT_DIR, config.OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
