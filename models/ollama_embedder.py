"""MTEB-compatible embedding encoder run on a local Ollama server."""

import numpy as np
import os
import requests

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from typing import Any


class OllamaEmbedder(AbsEncoder):
    """Calls the Ollama /api/embed endpoint batch-wise."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        model_prompts: dict[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_prompts = model_prompts
        self.base_url = (
            base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=f"ollama/{model_name.replace(':', '__')}",
            revision="latest",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False,
            training_datasets=None,
        )

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt = ""
        if self.model_prompts is not None and task_metadata is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            if prompt_name:
                prompt = self.model_prompts.get(prompt_name, "")

        embeddings: list[list[float]] = []
        for batch in inputs:
            texts = [prompt + text if text.strip() else " " for text in batch["text"]]
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=300,
            )
            response.raise_for_status()
            embeddings.extend(response.json()["embeddings"])

        array = np.asarray(embeddings, dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            array /= np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)
        return array
