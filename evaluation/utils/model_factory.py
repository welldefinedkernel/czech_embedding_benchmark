"""Build MTEB-compatible HuggingFace embedding models."""

import os
import types
from typing import Any

import mteb

from backends.ollama_embedder import OllamaEmbedder
from evaluation.utils.config import ModelConfig

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_AUTOPROCESSOR_FALLBACK_PATCHED = False


def _patch_autoprocessor_tokenizer_fallback() -> None:
    """Fall back to ``AutoTokenizer`` when ``AutoProcessor`` fails to load.

    Some text embedding models (e.g. Gemma3-based KaLM) declare
    ``processor_class = "Gemma3Processor"`` in their ``tokenizer_config.json``.
    SentenceTransformers calls ``AutoProcessor.from_pretrained`` unconditionally,
    which then tries to build the multimodal processor and dies because the repo
    ships no image-processor config. These models are text-only, so falling back
    to ``AutoTokenizer`` yields the correct tokenizer.
    """
    global _AUTOPROCESSOR_FALLBACK_PATCHED
    if _AUTOPROCESSOR_FALLBACK_PATCHED:
        return

    from transformers import AutoProcessor, AutoTokenizer

    original_from_pretrained = AutoProcessor.from_pretrained.__func__

    def from_pretrained(cls, *args: Any, **kwargs: Any):
        try:
            return original_from_pretrained(cls, *args, **kwargs)
        except Exception:
            return AutoTokenizer.from_pretrained(*args, **kwargs)

    AutoProcessor.from_pretrained = classmethod(from_pretrained)
    _AUTOPROCESSOR_FALLBACK_PATCHED = True


def _preserve_sharding_during_encode(mteb_model: Any) -> None:
    """Stop ``encode`` from consolidating a multi-GPU sharded model onto one device.

    When a model is loaded with ``device_map="auto"`` its parameters are split
    across GPUs, and accelerate hooks move activations between devices during
    ``forward()``. SentenceTransformers' ``encode()`` calls ``self.to(device)``,
    which tries to pull every shard onto a single GPU and OOMs for models larger
    than one card. When the underlying model is sharded across multiple devices,
    replace ``.to`` on that SentenceTransformer instance with a no-op so the
    dispatch is preserved. Single-device models are left untouched.
    """
    st_model = getattr(mteb_model, "model", None)
    if st_model is None or not hasattr(st_model, "modules"):
        return

    device_map: dict[str, Any] | None = None
    for module in st_model.modules():
        candidate = getattr(module, "hf_device_map", None)
        if candidate:
            device_map = candidate
            break

    if not device_map:
        return

    distinct_devices = {str(dev) for dev in device_map.values()}
    if len(distinct_devices) <= 1:
        return

    def _noop_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self

    st_model.to = types.MethodType(_noop_to, st_model)


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
    if config.model_dtype:
        model_kwargs["torch_dtype"] = config.model_dtype
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True

    _patch_autoprocessor_tokenizer_fallback()

    model = mteb.get_model(config.name, device=device, **kwargs)
    _preserve_sharding_during_encode(model)
    _cap_max_seq_length(model, config.max_seq_length)
    _set_pooling_mode(model, config.pooling_mode)
    return model


def _cap_max_seq_length(mteb_model: Any, cap: int | None) -> None:
    """Lower a model's input truncation length to ``cap`` tokens.

    SentenceTransformer-backed models hold the limit on ``mteb_model.model``,
    while mteb's dedicated encoders (e.g. LlamaEmbedNemotron) tokenize
    themselves and hold it on the wrapper.
    """
    if cap is None:
        return

    for holder in (getattr(mteb_model, "model", None), mteb_model):
        current = getattr(holder, "max_seq_length", None)
        if current is not None:
            holder.max_seq_length = min(current, cap)
            print(f"max_seq_length: {current} -> {holder.max_seq_length}")
            return

    print(f"WARNING: cannot apply max_seq_length={cap}, model exposes none.")


def _set_pooling_mode(mteb_model: Any, pooling_mode: str | None) -> None:
    """Force the pooling strategy of a SentenceTransformer-backed model.

    Repos that ship no ``modules.json`` (e.g. Seznam's Czech models) get
    SentenceTransformer's default mean pooling, which silently produces the
    wrong embedding space for models trained on the CLS token.
    """
    if pooling_mode is None:
        return

    st_model = getattr(mteb_model, "model", None)
    pooling = next(
        (
            module
            for module in getattr(st_model, "children", list)()
            if type(module).__name__ == "Pooling"
        ),
        None,
    )
    if pooling is None:
        print(f"WARNING: cannot apply pooling_mode={pooling_mode}, no Pooling module.")
        return

    current = pooling.pooling_mode
    pooling.pooling_mode = pooling_mode
    print(f"pooling_mode: {current} -> {pooling.pooling_mode}")
