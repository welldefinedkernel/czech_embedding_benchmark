"""Create synthetic dataset for Czech Text Document Corpus, since it lacks qrels."""

from dataloaders.czech_text_document import CzechTextDocumentDatasetLoader
from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from pathlib import Path
from typing import Any, Literal

LLMModelArg = str | DeepEvalBaseLLM | type[DeepEvalBaseLLM] | None
EmbeddingModelArg = (
    str | DeepEvalBaseEmbeddingModel | type[DeepEvalBaseEmbeddingModel] | None
)
OutputFileType = Literal["json", "csv", "jsonl"]


def create_synthetic_dataset(
    dataset_name: str,
    llm_model: LLMModelArg,
    split: str = "train",
    dataset_loader: CzechTextDocumentDatasetLoader | None = None,
    dataset_dir: str | Path = Path("data/czech_text_document_corpus_v20"),
    output_dir: str | Path = Path("data/synthetic"),
    document_limit: int | None = None,
    max_goldens_per_context: int = 2,
    include_expected_output: bool = True,
    file_type: OutputFileType = "json",
    async_mode: bool = True,
    max_concurrent: int = 100,
    quiet: bool = False,
) -> None:
    """Generate and store a DeepEval synthetic dataset from full Czech text documents."""
    resolved_dataset_name = dataset_name.strip()
    if not resolved_dataset_name:
        raise ValueError("dataset_name cannot be empty")

    resolved_split = split.strip()
    if not resolved_split:
        raise ValueError("split cannot be empty")

    resolved_llm_model = _resolve_model(llm_model)

    loader = dataset_loader or CzechTextDocumentDatasetLoader(
        dataset_dir=dataset_dir,
        split=resolved_split,
    )
    dataset = loader.load(limit=document_limit)
    contexts, source_files = _build_full_document_contexts(dataset)
    if not contexts:
        raise ValueError("No Czech text documents were loaded")

    synthesizer = Synthesizer(
        model=resolved_llm_model,
        async_mode=async_mode,
        max_concurrent=max_concurrent,
    )
    synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=include_expected_output,
        max_goldens_per_context=max_goldens_per_context,
        source_files=source_files,
    )

    synthesizer.save_as(
        file_type=file_type,
        directory=str(output_dir),
        file_name=f"{resolved_dataset_name}_{resolved_split}",
        quiet=quiet,
    )


def _resolve_model(model: Any) -> Any:
    if isinstance(model, type):
        return model()
    return model


def _build_full_document_contexts(dataset: Any) -> tuple[list[list[str]], list[str]]:
    contexts: list[list[str]] = []
    source_files: list[str] = []

    for index, record in enumerate(dataset):
        text = str(record.get("text", "")).strip()
        if not text:
            continue

        contexts.append([text])
        source_files.append(str(record.get("document_id") or index))

    return contexts, source_files
