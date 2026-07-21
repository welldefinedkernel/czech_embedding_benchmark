"""Build MTEB-compatible HuggingFace embedding models."""

import os
import mteb

from evaluation.config import ModelConfig
from models.ollama_embedder import OllamaEmbedder
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def build_model(config: ModelConfig, device: str) -> Any:
    """Create a model through MTEB's official registry-first loader."""
    if config.name.startswith("ollama/"):
        model_prompts: dict[str, str] | None = None
        if config.query_prompt or config.document_prompt:
            model_prompts = {}
            if config.query_prompt:
                model_prompts["query"] = config.query_prompt
            if config.document_prompt:
                model_prompts["document"] = config.document_prompt

        return OllamaEmbedder(
            config.name.removeprefix("ollama/"),
            model_prompts=model_prompts,
        )

    kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if config.use_safetensors:
        model_kwargs["use_safetensors"] = True
    if config.device_map:
        model_kwargs["device_map"] = config.device_map
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True

    return mteb.get_model(config.name, device=device, **kwargs)
