"""Build MTEB-compatible HuggingFace embedding models."""

import os
import mteb

from evaluation.config import ModelConfig
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def build_model(config: ModelConfig) -> Any:
    """Create a model through MTEB's official registry-first loader."""
    model_kwargs: dict[str, Any] = {}
    if config.use_safetensors is not None:
        model_kwargs["model_kwargs"] = {"use_safetensors": config.use_safetensors}

    return mteb.get_model(config.name, device="mps", **model_kwargs)
