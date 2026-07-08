"""Build MTEB-compatible HuggingFace embedding models."""

import os
import mteb

from evaluation.config import ModelConfig
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def build_model(config: ModelConfig, device: str) -> Any:
    """Create a model through MTEB's official registry-first loader."""
    kwargs: dict[str, Any] = {}
    if config.use_safetensors:
        kwargs["model_kwargs"] = {"use_safetensors": True}
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True

    return mteb.get_model(config.name, device=device, **kwargs)
