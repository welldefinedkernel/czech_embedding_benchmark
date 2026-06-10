"""Build MTEB-compatible HuggingFace embedding models."""

import os
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from evaluation.config import ModelConfig
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper


def build_model(config: ModelConfig) -> Any:
    """Create MTEB-compatible SentenceTransformer model wrapper."""
    model_kwargs: dict[str, Any] = {}
    if config.use_safetensors is not None:
        model_kwargs["model_kwargs"] = {"use_safetensors": config.use_safetensors}

    return SentenceTransformerEncoderWrapper(
        model=config.name,
        device="mps",
        model_prompts=_model_prompts(config),
        **model_kwargs,
    )


def _model_prompts(config: ModelConfig) -> dict[str, str] | None:
    prompts = {}
    if config.query_instruction:
        prompts["query"] = config.query_instruction
    if config.document_instruction:
        prompts["document"] = config.document_instruction
    return prompts or None
